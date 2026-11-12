"""Runs jobs end to end.

For each job that holds leases the orchestrator
  1. starts a swarm runner process (DHT + trainer/serving client; needs BloomBee),
  2. tells the provider agents to start one worker per lease (layer range per GPU),
  3. turns runner/agent messages into events, and replaces a worker that dies (killed, crashed, GPU silent) by leasing
     another idle GPU and starting a fresh worker on it, never a preloaded spare,
  4. on success stores the adapter in the model registry, releases the leases (billing both sides) and starts the
     next queued job.
"""
from __future__ import annotations

import asyncio
import json
import uuid
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sqlmodel import Session, select

from app.agent_hub import AgentCommandError, AgentNotConnected
from app.db import utcnow
from app.events import EventType
from app.models import BaseModel, Deployment, FineTunedModel, Gpu, Job, Lease
from app.services import scheduler
from app.services.pool import effective_status

log = logging.getLogger("syntra.orchestrator")
RUNNER = Path(__file__).resolve().parent.parent / "runner" / "swarm_runner.py"


class RunnerProc:
    """The swarm runner subprocess: JSON lines in (stdin) and out (stdout)."""

    def __init__(self, proc: asyncio.subprocess.Process):
        self.proc = proc
        self.tail: list[str] = []

    async def send(self, obj: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(obj) + "\n").encode())
        await self.proc.stdin.drain()

    def terminate(self) -> None:
        import os, signal
        # Signal the whole group even when the runner itself is gone: its DHT helper processes may still be alive.
        try:
            os.killpg(self.proc.pid, signal.SIGKILL if self.proc.returncode is not None else signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            if self.proc.returncode is None:
                self.proc.terminate()


@dataclass
class JobRun:
    job_id: int
    owner_id: int
    kind: str
    prefix: str
    spec: dict
    runner: Optional[RunnerProc] = None
    initial_peers: list = field(default_factory=list)
    dht_ready: asyncio.Event = field(default_factory=asyncio.Event)
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    swarm_ready: bool = False
    workers: dict = field(default_factory=dict)  # worker_id -> lease_id
    killed: dict = field(default_factory=dict)  # worker_id -> reason chosen by the platform
    down_at: dict = field(default_factory=dict)  # layer_start -> monotonic time the stage lost its worker
    replacement_leases: set = field(default_factory=set)
    handled_down: set = field(default_factory=set)  # lease ids already reported down
    pending: list = field(default_factory=list)  # leases still waiting for a replacement GPU
    result: dict = field(default_factory=dict)  # train.finished payload
    results: dict = field(default_factory=dict)  # serve results by request id
    waiters: dict = field(default_factory=dict)
    cancelled: bool = False
    serving: bool = False
    error: str = ""
    task: Optional[asyncio.Task] = None
    last_step: dict = field(default_factory=dict)


class Orchestrator:
    def __init__(self, app):
        self.app = app
        self.settings = app.state.settings
        self.engine = app.state.engine
        self.events = app.state.events
        self.hub = app.state.agent_hub
        self.runs: dict[int, JobRun] = {}
        self.background: set = set()  # evaluation tasks etc., kept referenced until done
        self._loop_task: Optional[asyncio.Task] = None
        self.hub.add_listener(self._on_agent_message)

    # ---- db helper ---------------------------------------------------------------------------------
    def _sync(self, fn):
        with Session(self.engine, expire_on_commit=False) as session:
            return fn(session)

    async def _db(self, fn):
        return await asyncio.to_thread(self._sync, fn)

    @property
    def heartbeat_timeout(self) -> float:
        return self.settings.heartbeat_timeout_seconds

    # ---- lifecycle ---------------------------------------------------------------------------------
    async def start(self) -> None:
        self._loop_task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._loop_task:
            self._loop_task.cancel()
        for run in list(self.runs.values()):
            run.cancelled = True
            if run.runner:
                run.runner.terminate()
        await asyncio.gather(*(r.task for r in self.runs.values() if r.task), return_exceptions=True)

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception:
                log.exception("orchestrator tick failed")
            await asyncio.sleep(self.settings.orchestrator_interval)

    async def tick(self) -> None:
        """Start queued jobs that now fit, launch jobs that hold leases, watch for silent GPUs, retry replacements."""
        await self._db(lambda s: scheduler.dispatch_queue(s, self.events, self.heartbeat_timeout))
        starting = await self._db(lambda s: [j.id for j in s.exec(select(Job).where(Job.status == "starting")).all()])
        for job_id in starting:
            if job_id not in self.runs:
                await self.launch(job_id)
        for run in list(self.runs.values()):
            if run.finished.is_set():
                continue
            await self._check_silent_gpus(run)
            await self._retry_pending(run)

    async def submit(self, job_id: int) -> bool:
        """Called right after a job is created: lease immediately if the pool allows, otherwise it queues."""
        leases = await self._db(lambda s: scheduler.submit(s, self.events, s.get(Job, job_id), self.heartbeat_timeout))
        if leases is not None:
            await self.launch(job_id)
        return leases is not None

    async def launch(self, job_id: int) -> None:
        if job_id in self.runs:
            return
        job = await self._db(lambda s: s.get(Job, job_id))
        config = json.loads(job.config_json or "{}")
        run = JobRun(job_id=job.id, owner_id=job.owner_id, kind=job.kind, prefix=f"job{job.id}", spec=config)
        self.runs[job_id] = run
        run.task = asyncio.create_task(self._run(run))

    # ---- one job -----------------------------------------------------------------------------------
    def _model_path(self, base: BaseModel) -> str:
        if base.local_path and Path(base.local_path).exists():
            return base.local_path
        local = Path(self.settings.models_dir).expanduser() / base.slug
        return str(local) if local.exists() else base.hf_id

    async def _run(self, run: JobRun) -> None:
        try:
            await self._run_inner(run)
        except Exception as exc:
            log.exception("job %s crashed", run.job_id)
            run.error = f"{type(exc).__name__}: {exc}"
        finally:
            await self._finalize(run)

    async def _run_inner(self, run: JobRun) -> None:
        job, base = await self._db(lambda s: (s.get(Job, run.job_id), s.get(BaseModel, s.get(Job, run.job_id).base_model_id)))
        job_dir = Path(self.settings.jobs_dir).expanduser() / str(run.job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        model_path = run.spec.get("model_path") or self._model_path(base)
        if run.kind == "deploy" and run.spec.get("finetuned_model_id"):
            model_path = await self._ensure_merged(run, base)  # first deploy of a fine-tuned model merges its adapter once
        if Path(model_path).exists():
            run.spec["worker_model_path"] = model_path  # agents on this machine read the files in place
        spec = {**run.spec, "kind": run.kind, "job_id": run.job_id, "dht_prefix": run.prefix, "dht_port": self.settings.dht_port,
                "num_layers": base.num_layers, "model_path": model_path,
                "bloombee_dir": self.settings.bloombee_dir, "output_dir": str(job_dir), "ready_timeout": self.settings.ready_timeout,
                **self.settings.runner_spec_extra}
        (job_dir / "spec.json").write_text(json.dumps(spec, indent=1))
        argv = list(self.settings.runner_argv) or [self.settings.bloombee_python or sys.executable, str(RUNNER)]
        proc = await asyncio.create_subprocess_exec(
            *argv, str(job_dir / "spec.json"), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        run.runner = RunnerProc(proc)
        reader = asyncio.create_task(self._read_runner(run))
        try:
            await asyncio.wait_for(run.dht_ready.wait(), 120)
        except asyncio.TimeoutError:
            run.error = "swarm runner did not start (is SYNTRA_BLOOMBEE_PYTHON set to an environment with BloomBee?)"
            run.runner.terminate()
        else:
            def mark_running(s):
                j = s.get(Job, run.job_id)
                j.status, j.dht_prefix, j.updated_at = "running", run.prefix, utcnow()
                s.add(j); s.commit()
            await self._db(mark_running)
            leases = await self._db(lambda s: s.exec(select(Lease).where(Lease.job_id == run.job_id, Lease.ended_at == None)  # noqa: E711
                                                      .order_by(Lease.layer_start)).all())
            await asyncio.gather(*(self._start_worker(run, lease) for lease in leases))
        await run.finished.wait()
        await reader

    async def _read_runner(self, run: JobRun) -> None:
        proc = run.runner.proc
        assert proc.stdout is not None
        exited = asyncio.create_task(proc.wait())
        while True:
            # Do not wait for EOF: helper processes of the runner can hold the pipe open after it has exited.
            read = asyncio.create_task(proc.stdout.readline())
            await asyncio.wait({read, exited}, return_when=asyncio.FIRST_COMPLETED)
            if not read.done():
                try:  # the runner exited: take what is already buffered, then stop
                    line = await asyncio.wait_for(read, 1.0)
                except asyncio.TimeoutError:
                    break
            else:
                line = read.result()
            if not line:
                break
            text = line.decode(errors="replace").rstrip()
            run.runner.tail = (run.runner.tail + [text])[-15:]
            if not text.startswith("{"):
                continue
            try:
                await self._on_runner_message(run, json.loads(text))
            except Exception:
                log.exception("bad runner message: %s", text[:200])
        await proc.wait()
        run.finished.set()
        run.dht_ready.set()
        run.serving = False
        for future in run.waiters.values():
            if not future.done():
                future.set_result({"type": "error", "error": "the deployment stopped"})

    async def _on_runner_message(self, run: JobRun, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "dht_ready":
            run.initial_peers = msg["initial_peers"]
            run.dht_ready.set()
        elif kind == "swarm_ready":
            run.swarm_ready = True
        elif kind == "train.step":
            run.last_step = msg
            await self._db(lambda s: self.events.publish(
                s, EventType.TRAIN_STEP, job_id=run.job_id, user_id=run.owner_id,
                payload={k: msg[k] for k in ("step", "loss", "grad_norm", "step_seconds")}))
        elif kind == "train.finished":
            run.result = msg
        elif kind in ("swarm_timeout", "runner_error"):
            run.error = msg.get("error") or f"swarm timeout: {msg.get('layers_online')}/{msg.get('layers_total')} layers online"
        elif kind in ("serve_ready", "result", "error") and run.kind == "deploy":
            await self._on_serve_message(run, msg)

    async def _ensure_merged(self, run: JobRun, base: BaseModel) -> str:
        """Merge a fine-tuned model's LoRA adapter into FP16 base weights once; later deployments reuse the result."""
        model = await self._db(lambda s: s.get(FineTunedModel, run.spec["finetuned_model_id"]))
        if model.merged_path and Path(model.merged_path).exists():
            return model.merged_path
        await self._db(lambda s: self.events.publish(s, EventType.DEPLOY_MERGING, job_id=run.job_id, user_id=run.owner_id, payload={}))
        dest = Path(self.settings.registry_dir).expanduser() / str(model.id) / "merged"
        shutil.rmtree(dest, ignore_errors=True)  # merge_lora.py refuses to overwrite an existing directory
        argv = list(self.settings.merge_argv) or [self.settings.bloombee_python or sys.executable,
                                                  str(Path(self.settings.bloombee_dir).expanduser() / "examples" / "merge_lora.py")]
        proc = await asyncio.create_subprocess_exec(
            *argv, "--model", self._model_path(base), "--adapter", model.adapter_path, "--output", str(dest),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError("merging the adapter failed: " + out.decode(errors="replace")[-300:])

        def save(s):
            m = s.get(FineTunedModel, model.id)
            m.merged_path = str(dest)
            s.add(m); s.commit()
        await self._db(save)
        return str(dest)

    async def _on_serve_message(self, run: JobRun, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "serve_ready":
            run.serving = True

            def online(s):
                dep = s.exec(select(Deployment).where(Deployment.job_id == run.job_id)).first()
                leases = s.exec(select(Lease).where(Lease.job_id == run.job_id, Lease.ended_at == None)).all()  # noqa: E711
                if dep is not None:
                    dep.status, dep.num_gpus, dep.endpoint = "running", len(leases), f"/deployments/{dep.id}/generate"
                    s.add(dep); s.commit()
                    self.events.publish(s, EventType.DEPLOY_ONLINE, job_id=run.job_id, user_id=run.owner_id,
                                        payload={"endpoint": dep.endpoint, "num_gpus": dep.num_gpus})
            await self._db(online)
        elif kind in ("result", "error"):
            future = run.waiters.pop(msg.get("id"), None)
            if future is not None and not future.done():
                future.set_result(msg)

    async def generate(self, job_id: int, payload: dict, timeout: float = 120.0) -> dict:
        """Run one prompt on a live deployment (greedy unless do_sample)."""
        run = self.runs.get(job_id)
        if run is None or not run.serving or run.runner is None:
            raise LookupError("the deployment is not running")
        request_id = uuid.uuid4().hex
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        run.waiters[request_id] = future
        await run.runner.send({"op": "generate", "id": request_id, **payload})
        try:
            msg = await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            run.waiters.pop(request_id, None)
            raise TimeoutError("generation timed out")
        if msg.get("type") == "error":
            raise RuntimeError(msg.get("error", "generation failed"))
        return msg

    # ---- workers -----------------------------------------------------------------------------------
    def _command(self, run: JobRun, lease: Lease, gpu: Gpu, base: BaseModel) -> dict:
        return {"cmd": "start_worker", "worker_id": f"j{run.job_id}-l{lease.id}", "device": gpu.device,
                "device_index": gpu.device_index, "model": {"name": run.spec.get("worker_model_name") or base.slug,
                "hf_id": base.hf_id, "path": run.spec.get("worker_model_path", "")},
                "layer_start": lease.layer_start, "layer_end": lease.layer_end, "dht_prefix": run.prefix,
                "initial_peers": run.initial_peers, "mode": "train" if run.kind == "train" else "serve", "batch_size": 2}

    async def _start_worker(self, run: JobRun, lease: Lease, replacement: bool = False) -> None:
        gpu, base = await self._db(lambda s: (s.get(Gpu, lease.gpu_id), s.get(BaseModel, s.get(Job, run.job_id).base_model_id)))
        cmd = self._command(run, lease, gpu, base)
        run.workers[cmd["worker_id"]] = lease.id
        if not replacement:
            await self._db(lambda s: self.events.publish(
                s, EventType.WORKER_STARTING, job_id=run.job_id, gpu_id=gpu.id, user_id=run.owner_id,
                payload={"layer_start": lease.layer_start, "layer_end": lease.layer_end}))
        try:
            await self.hub.request(gpu.agent_token_id, cmd, timeout=60)
        except (AgentNotConnected, AgentCommandError, TypeError) as exc:
            log.warning("could not start worker for lease %s: %s", lease.id, exc)
            await self._handle_down(run, lease.id, "exited", detail=str(exc))

    async def _on_agent_message(self, token_id: int, msg: dict) -> None:
        worker_id = msg.get("worker_id", "")
        if not worker_id.startswith("j") or "-l" not in worker_id:
            return
        try:
            job_id, lease_id = (int(x) for x in worker_id[1:].split("-l"))
        except ValueError:
            return
        run = self.runs.get(job_id)
        if run is None or run.finished.is_set():
            return
        kind = msg.get("type")
        if kind == "worker.started":
            def store(s):
                lease = s.get(Lease, lease_id)
                if lease is not None:
                    lease.worker_peer_id = msg.get("peer_id", "")
                    s.add(lease); s.commit()
            await self._db(store)
        elif kind == "worker.online":
            await self._on_worker_online(run, lease_id, msg)
        elif kind == "worker.exited":
            reason = run.killed.pop(worker_id, None) or ("stopped" if msg.get("reason") == "stopped" else "exited")
            if reason != "stopped":
                await self._handle_down(run, lease_id, reason)

    async def _on_worker_online(self, run: JobRun, lease_id: int, msg: dict) -> None:
        lease, gpu = await self._db(lambda s: (s.get(Lease, lease_id), s.get(Gpu, s.get(Lease, lease_id).gpu_id)))
        payload = {"layer_start": lease.layer_start, "layer_end": lease.layer_end, "peer_id": msg.get("peer_id", "")}
        if lease_id in run.replacement_leases:
            since = round(time.monotonic() - run.down_at.get(lease.layer_start, time.monotonic()), 2)
            etype, payload = EventType.WORKER_REPLACEMENT_ONLINE, {**payload, "seconds_since_down": since}
        else:
            etype = EventType.WORKER_ONLINE
        await self._db(lambda s: self.events.publish(s, etype, job_id=run.job_id, gpu_id=gpu.id, user_id=run.owner_id, payload=payload))

    async def _handle_down(self, run: JobRun, lease_id: int, reason: str, detail: str = "") -> None:
        """A worker is gone: report it, free its GPU, and lease a replacement for the same layers."""
        if lease_id in run.handled_down or run.finished.is_set() or run.cancelled:
            return
        run.handled_down.add(lease_id)
        lease = await self._db(lambda s: s.get(Lease, lease_id))
        run.down_at[lease.layer_start] = time.monotonic()
        await self._db(lambda s: self.events.publish(
            s, EventType.WORKER_DOWN, job_id=run.job_id, gpu_id=lease.gpu_id, user_id=run.owner_id,
            payload={"reason": reason, "layer_start": lease.layer_start, "layer_end": lease.layer_end, "detail": detail}))
        await self._replace(run, lease, reason)

    async def _replace(self, run: JobRun, dead: Lease, reason: str = "worker_down") -> None:
        new = await self._db(lambda s: scheduler.lease_replacement(
            s, self.events, s.get(Job, run.job_id), s.get(Lease, dead.id), self.heartbeat_timeout, reason=reason))
        if new is None:
            if dead.id not in [d.id for d in run.pending]:
                run.pending.append(dead)
            return
        run.pending = [d for d in run.pending if d.id != dead.id]
        run.replacement_leases.add(new.id)
        await self._start_worker(run, new, replacement=True)

    async def _retry_pending(self, run: JobRun) -> None:
        for dead in list(run.pending):
            await self._replace(run, dead)

    async def _check_silent_gpus(self, run: JobRun) -> None:
        """A leased GPU whose heartbeat stopped is as good as dead (power loss, network, agent crash)."""
        def silent(s):
            now, out = utcnow(), []
            for lease in s.exec(select(Lease).where(Lease.job_id == run.job_id, Lease.ended_at == None)).all():  # noqa: E711
                gpu = s.get(Gpu, lease.gpu_id)
                if effective_status(gpu, now, self.heartbeat_timeout) == "offline" and lease.id not in run.handled_down:
                    out.append((lease.id, gpu.agent_token_id, f"j{run.job_id}-l{lease.id}"))
            return out
        for lease_id, token_id, worker_id in await self._db(silent):
            try:
                await self.hub.request(token_id, {"cmd": "kill_worker", "worker_id": worker_id}, timeout=3)
            except Exception:
                pass
            await self._handle_down(run, lease_id, "heartbeat_timeout")

    # ---- user actions --------------------------------------------------------------------------------
    async def kill_worker(self, job_id: int, layer_start: int) -> bool:
        """The demo 'kill this worker' button."""
        run = self.runs.get(job_id)
        if run is None or run.finished.is_set():
            return False
        lease, gpu = await self._db(lambda s: (lambda l: (l, s.get(Gpu, l.gpu_id)) if l else (None, None))(
            s.exec(select(Lease).where(Lease.job_id == job_id, Lease.layer_start == layer_start, Lease.ended_at == None)).first()))  # noqa: E711
        if lease is None:
            return False
        worker_id = f"j{job_id}-l{lease.id}"
        run.killed[worker_id] = "killed_by_user"
        try:
            await self.hub.request(gpu.agent_token_id, {"cmd": "kill_worker", "worker_id": worker_id}, timeout=10)
        except Exception:
            run.killed.pop(worker_id, None)
            return False
        return True

    async def cancel(self, job_id: int) -> bool:
        run = self.runs.get(job_id)
        if run is None:
            return False
        run.cancelled = True
        if run.runner:
            run.runner.terminate()
        run.finished.set()
        return True

    # ---- finishing -----------------------------------------------------------------------------------
    async def _stop_workers(self, run: JobRun) -> None:
        def active(s):
            return [(l.id, s.get(Gpu, l.gpu_id).agent_token_id) for l in
                    s.exec(select(Lease).where(Lease.job_id == run.job_id, Lease.ended_at == None)).all()]  # noqa: E711
        for lease_id, token_id in await self._db(active):
            try:
                await self.hub.request(token_id, {"cmd": "stop_worker", "worker_id": f"j{run.job_id}-l{lease_id}"}, timeout=15)
            except Exception:
                pass

    async def _finalize(self, run: JobRun) -> None:
        if run.runner:
            run.runner.terminate()
        await self._stop_workers(run)
        ok = run.kind == "train" and run.result.get("returncode") == 0 and not run.error and not run.cancelled
        model_id = None
        if ok:
            model_id = await self._db(lambda s: self._register_model(s, run))
        status = "stopped" if run.cancelled else "completed" if ok else "failed"
        if run.kind == "deploy":
            status = "failed" if run.error else "stopped"  # a deployment only ends by being stopped, or by failing
        reason = status

        def close(s):
            job = s.get(Job, run.job_id)
            scheduler.release_job(s, self.events, job, reason)
            job.status, job.updated_at = status, utcnow()
            job.result_json = json.dumps({"final": run.last_step, "model_id": model_id, "error": run.error,
                                          "tail": (run.result.get("tail") or (run.runner.tail if run.runner else []))[-8:] if status == "failed" else []})
            s.add(job); s.commit()
            if run.kind == "train" and status == "completed":
                self.events.publish(s, EventType.TRAIN_COMPLETED, job_id=job.id, user_id=job.owner_id, payload={"model_id": model_id})
            elif run.kind == "deploy":
                dep = s.exec(select(Deployment).where(Deployment.job_id == job.id)).first()
                if dep is not None:
                    dep.status = "failed" if status == "failed" else "stopped"
                    s.add(dep); s.commit()
                self.events.publish(s, EventType.DEPLOY_STOPPED, job_id=job.id, user_id=job.owner_id,
                                    payload={"reason": (run.error or "failed")[:300] if status == "failed" else "stopped"})
            elif run.kind == "train" and status == "failed":
                err = run.error or "; ".join((run.result.get("tail") or run.runner.tail if run.runner else [])[-3:]) or "training failed"
                self.events.publish(s, EventType.TRAIN_FAILED, job_id=job.id, user_id=job.owner_id, payload={"error": err[:500]})
        await self._db(close)
        self.runs.pop(run.job_id, None)

    def _register_model(self, s: Session, run: JobRun) -> int:
        job = s.get(Job, run.job_id)
        base = s.get(BaseModel, job.base_model_id)
        model = FineTunedModel(owner_id=job.owner_id, base_model_id=base.id, job_id=job.id,
                               name=run.spec.get("model_name") or f"{base.name} fine-tuned (run #{job.id})")
        s.add(model); s.commit(); s.refresh(model)
        dest = Path(self.settings.registry_dir).expanduser() / str(model.id) / "adapter"
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = Path(run.result.get("adapter_dir", ""))
        if src.exists():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        model.adapter_path = str(dest)
        s.add(model); s.commit()
        return model.id
