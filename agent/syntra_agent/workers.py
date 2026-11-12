"""Start, watch and stop BloomBee worker processes on this machine."""
from __future__ import annotations

import asyncio
import os
import platform
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

PEER_RE = re.compile(r"Running a server on \[[^\]]*?/p2p/([A-Za-z0-9]+)'")
STARTED_RE = re.compile(r"\]\s+Started\s*$")


@dataclass
class AgentConfig:
    bloombee_dir: Path = Path("~/BloomBee").expanduser()
    python: str = sys.executable  # interpreter that has BloomBee's dependencies installed
    models_dir: Path = Path("models")
    work_dir: Path = Path("agent-work")
    hf_home: Optional[Path] = None


@dataclass
class WorkerSpec:
    worker_id: str
    device: str  # mps | cuda
    device_index: int
    model: dict  # {"name": "tinyllama", "hf_id": "TinyLlama/..."}; resolved against models_dir first
    layer_start: int
    layer_end: int
    dht_prefix: str
    initial_peers: list[str]
    mode: str = "train"  # train (LoRA RPC enabled) | serve
    host_maddr: Optional[str] = None  # e.g. /ip4/127.0.0.1/tcp/31002
    announce_maddr: Optional[str] = None
    batch_size: int = 2
    extra_env: dict = field(default_factory=dict)

    @classmethod
    def from_command(cls, cmd: dict) -> "WorkerSpec":
        known = {k: cmd[k] for k in cls.__dataclass_fields__ if k in cmd}
        return cls(**known)


def resolve_model(model: dict, cfg: AgentConfig) -> str:
    """Prefer a local copy in models_dir; fall back to the Hugging Face id (downloaded by the worker)."""
    local = Path(cfg.models_dir).expanduser() / model.get("name", "")
    if model.get("name") and local.exists():
        return str(local)
    return model.get("hf_id") or str(local)


def build_bloombee_command(spec: WorkerSpec, cfg: AgentConfig) -> tuple[list[str], dict]:
    """The exact run_server flags verified in the fail-over and deploy experiments."""
    model = resolve_model(spec.model, cfg)
    argv = [
        cfg.python, "-m", "bloombee.cli.run_server", model,
        "--block_indices", f"{spec.layer_start}:{spec.layer_end}",
        "--dht_prefix", spec.dht_prefix,
        "--num_handlers", "1", "--batch_size", str(spec.batch_size), "--torch_dtype", "float16",
        "--throughput", "100", "--update_period", "2", "--expiration", "15",
        "--inference_max_length", "512", "--attn_cache_tokens", "1024", "--max_batch_size", "512",
        "--balance_quality", "0", "--no_auto_relay", "--skip_reachability_check",
        "--cache_gpu_percent", "100", "--w_gpu_percent", "100", "--device", spec.device,
        "--host_maddrs", spec.host_maddr or "/ip4/127.0.0.1/tcp/0",
    ]
    if spec.announce_maddr:
        argv += ["--announce_maddrs", spec.announce_maddr]
    argv += ["--initial_peers", *spec.initial_peers]
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(Path(cfg.bloombee_dir).expanduser() / "src") + os.pathsep + env.get("PYTHONPATH", ""),
        BLOOMBEE_ENABLE_SPEC_PRUNER="0",
        BLOOMBEE_ENABLE_LORA="1" if spec.mode == "train" else "0",
    )
    if spec.device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = str(spec.device_index)
    if cfg.hf_home:
        env["HF_HOME"] = str(cfg.hf_home)
    if Path(model).exists():
        env["HF_HUB_OFFLINE"] = "1"
    env.update({k: str(v) for k, v in spec.extra_env.items()})
    return argv, env


EventCallback = Callable[[dict], Awaitable[None]]


@dataclass
class Worker:
    spec: WorkerSpec
    proc: asyncio.subprocess.Process
    started_at: float = field(default_factory=time.monotonic)
    peer_id: str = ""
    online: bool = False
    stop_reason: str = ""  # set when we ended it on purpose: killed | stopped
    task: Optional[asyncio.Task] = None


class WorkerManager:
    def __init__(self, cfg: AgentConfig, on_event: EventCallback,
                 command_builder: Callable[[WorkerSpec, AgentConfig], tuple[list[str], dict]] = build_bloombee_command):
        self.cfg, self.on_event, self.command_builder = cfg, on_event, command_builder
        self.workers: dict[str, Worker] = {}
        self._caffeinate: Optional[asyncio.subprocess.Process] = None

    # ---- queries ----------------------------------------------------------------------------------
    def pids(self) -> list[int]:
        return [w.proc.pid for w in self.workers.values() if w.proc.returncode is None]

    def on_gpu(self, device_index: int) -> list[Worker]:
        return [w for w in self.workers.values() if w.spec.device_index == device_index and w.proc.returncode is None]

    def describe(self) -> list[dict]:
        return [{"worker_id": w.spec.worker_id, "pid": w.proc.pid, "peer_id": w.peer_id, "online": w.online,
                 "layers": [w.spec.layer_start, w.spec.layer_end], "device_index": w.spec.device_index}
                for w in self.workers.values() if w.proc.returncode is None]

    # ---- lifecycle --------------------------------------------------------------------------------
    async def start(self, spec: WorkerSpec) -> Worker:
        if spec.worker_id in self.workers and self.workers[spec.worker_id].proc.returncode is None:
            raise ValueError(f"worker {spec.worker_id} is already running")
        argv, env = self.command_builder(spec, self.cfg)
        log_dir = Path(self.cfg.work_dir).expanduser() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        proc = await asyncio.create_subprocess_exec(
            *argv, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True,
        )
        worker = Worker(spec=spec, proc=proc)
        self.workers[spec.worker_id] = worker
        worker.task = asyncio.create_task(self._watch(worker, log_dir / f"{spec.worker_id}.log"))
        await self._update_caffeinate()
        return worker

    async def _watch(self, worker: Worker, log_path: Path) -> None:
        spec = worker.spec
        with open(log_path, "ab") as log:
            assert worker.proc.stdout is not None
            while True:
                line = await worker.proc.stdout.readline()
                if not line:
                    break
                log.write(line)
                log.flush()
                text = line.decode(errors="replace").rstrip()
                if not worker.peer_id:
                    m = PEER_RE.search(text)
                    if m:
                        worker.peer_id = m.group(1)
                        await self.on_event({"type": "worker.started", "worker_id": spec.worker_id,
                                             "pid": worker.proc.pid, "peer_id": worker.peer_id})
                if not worker.online and STARTED_RE.search(text):
                    worker.online = True
                    await self.on_event({"type": "worker.online", "worker_id": spec.worker_id,
                                         "peer_id": worker.peer_id,
                                         "seconds_to_online": round(time.monotonic() - worker.started_at, 2)})
        code = await worker.proc.wait()
        reason = worker.stop_reason or "exited"
        await self.on_event({"type": "worker.exited", "worker_id": spec.worker_id, "returncode": code,
                             "reason": reason, "layers": [spec.layer_start, spec.layer_end]})
        await self._update_caffeinate()

    async def kill(self, worker_id: str) -> bool:
        """Immediate SIGKILL of the whole process group (the demo 'kill this worker' button)."""
        w = self.workers.get(worker_id)
        if w is None or w.proc.returncode is not None:
            return False
        w.stop_reason = "killed"
        try:
            os.killpg(w.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return True

    async def stop(self, worker_id: str, grace: float = 10.0) -> bool:
        w = self.workers.get(worker_id)
        if w is None or w.proc.returncode is not None:
            return False
        w.stop_reason = "stopped"
        try:
            os.killpg(w.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        try:
            await asyncio.wait_for(w.proc.wait(), grace)
        except asyncio.TimeoutError:
            try:
                os.killpg(w.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return True

    async def stop_all(self) -> None:
        await asyncio.gather(*(self.stop(wid, grace=5.0) for wid in list(self.workers)))
        for w in list(self.workers.values()):
            if w.task:
                try:
                    await asyncio.wait_for(w.task, 5)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
        await self._update_caffeinate()

    # ---- keep a Mac awake while it serves ---------------------------------------------------------
    async def _update_caffeinate(self) -> None:
        if platform.system() != "Darwin":
            return
        active = any(w.proc.returncode is None for w in self.workers.values())
        if active and self._caffeinate is None:
            try:
                self._caffeinate = await asyncio.create_subprocess_exec("caffeinate", "-i", "-w", str(os.getpid()))
            except OSError:
                self._caffeinate = None
        elif not active and self._caffeinate is not None:
            try:
                self._caffeinate.terminate()
            except ProcessLookupError:
                pass
            self._caffeinate = None
