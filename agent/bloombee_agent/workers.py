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
TRACEBACK_RE = re.compile(r"^Traceback \(most recent call last\):")
ERROR_LINE_RE = re.compile(r"^([A-Za-z_][\w.]*(Error|Exception|Interrupt)\b.*)$")


@dataclass
class AgentConfig:
    bloombee_dir: Path = Path("~/BloomBee").expanduser()
    python: str = sys.executable  # interpreter that has BloomBee's dependencies installed
    models_dir: Path = Path("models")
    work_dir: Path = Path("agent-work")
    hf_home: Optional[Path] = None
    port_range: Optional[tuple] = None  # (first, last): worker ports to use (forwarded when the machine sits behind a tunnel)
    update_period: float = 2.0  # how often a worker re-announces its blocks in the DHT (seconds)
    expiration: float = 15.0  # how long an announcement stays valid; must comfortably exceed how long one announcement takes
    # A worker that crashes while starting can leave its main process hanging (e.g. a helper process failed to create a
    # semaphore): the platform would wait for it forever. Such workers are killed so the platform replaces them.
    crash_grace: float = 15.0  # seconds after a traceback before a not-yet-online worker is killed
    startup_timeout: float = 1200.0  # a worker not online after this long is killed (weights download included)
    server: str = ""  # platform URL and this machine's token: DiLoCo replicas fetch weights and upload updates themselves
    token: str = ""


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
    weight_bits: int = 16  # 16 = float16 base weights; 4 = 4-bit compressed base (QLoRA training / light serving)
    kind: str = "bloombee"  # bloombee (pipeline stage) | diloco (full fine-tuning replica holding the whole model)
    job_id: int = 0
    lease_id: int = 0
    extra_env: dict = field(default_factory=dict)

    @classmethod
    def from_command(cls, cmd: dict) -> "WorkerSpec":
        known = {k: cmd[k] for k in cls.__dataclass_fields__ if k in cmd}
        return cls(**known)


def resolve_model(model: dict, cfg: AgentConfig) -> str:
    """An explicit path that exists here (e.g. a merged fine-tuned model), else a local copy in models_dir,
    else the Hugging Face id (downloaded by the worker)."""
    if model.get("path") and Path(model["path"]).expanduser().exists():
        return str(Path(model["path"]).expanduser())
    local = Path(cfg.models_dir).expanduser() / model.get("name", "")
    if model.get("name") and local.exists():
        return str(local)
    return model.get("hf_id") or str(local)


def build_diloco_command(spec: WorkerSpec, cfg: AgentConfig) -> tuple[list[str], dict]:
    """A DiLoCo replica: plain torch + transformers (no BloomBee), talks to the platform over HTTP with this machine's token."""
    if not cfg.server or not cfg.token:
        raise ValueError("DiLoCo replicas need the platform address and token (set by the agent)")
    model = resolve_model(spec.model, cfg)
    argv = [cfg.python, "-m", "bloombee_agent.diloco_worker", "--server", cfg.server, "--job", str(spec.job_id),
            "--lease", str(spec.lease_id), "--model", model, "--device", spec.device,
            "--work-dir", str(Path(cfg.work_dir).expanduser() / f"diloco-{spec.worker_id}")]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
    env["BLOOMBEE_AGENT_TOKEN"] = cfg.token  # environment, not argv: other users cannot read it with ps
    if spec.device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = str(spec.device_index)
    if cfg.hf_home:
        env["HF_HOME"] = str(cfg.hf_home)
    if Path(model).exists():
        env["HF_HUB_OFFLINE"] = "1"
    env.update({k: str(v) for k, v in spec.extra_env.items()})
    return argv, env


def build_bloombee_command(spec: WorkerSpec, cfg: AgentConfig) -> tuple[list[str], dict]:
    """The exact run_server flags verified in the fail-over and deploy experiments."""
    if spec.kind == "diloco":
        return build_diloco_command(spec, cfg)
    if spec.kind != "bloombee":
        raise ValueError(f"unknown worker kind {spec.kind!r}")
    model = resolve_model(spec.model, cfg)
    argv = [
        cfg.python, "-m", "bloombee.cli.run_server", model,
        "--block_indices", f"{spec.layer_start}:{spec.layer_end}",
        "--dht_prefix", spec.dht_prefix,
        "--num_handlers", "1", "--batch_size", str(spec.batch_size), "--torch_dtype", "float16",
        "--throughput", "100", "--update_period", str(cfg.update_period), "--expiration", str(cfg.expiration),
        "--inference_max_length", "512", "--attn_cache_tokens", "1024", "--max_batch_size", "512",
        "--balance_quality", "0", "--no_auto_relay", "--skip_reachability_check",
        "--cache_gpu_percent", "100", "--w_gpu_percent", "100", "--device", spec.device,
        "--host_maddrs", spec.host_maddr or "/ip4/127.0.0.1/tcp/0",
    ]
    if spec.weight_bits == 4:
        argv += ["--compress_weight"]
    elif spec.weight_bits != 16:
        raise ValueError(f"unsupported weight_bits {spec.weight_bits} (use 16 or 4)")
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


def group_members(pgid: int) -> list[tuple[int, str]]:
    """(pid, command) of every process in a process group."""
    import subprocess
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,pgid=,command="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    members = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[1] == str(pgid):
            members.append((int(parts[0]), parts[2] if len(parts) > 2 else ""))
    return members


def kill_process_group(pgid: int, sig: int = signal.SIGKILL) -> None:
    """Kill a worker's whole process group, except Python's multiprocessing resource_tracker.

    With the spawn start method (the macOS default) every multiprocessing lock keeps a *named* POSIX semaphore until the
    resource_tracker unlinks it. If the tracker is killed too, the names leak until the machine reboots, and after ~10,000
    of them (kern.posix.sem.max) no Python program on the Mac can create a lock any more. Left alive, the tracker notices
    its parent is gone, unlinks everything and exits by itself."""
    members = group_members(pgid)
    if not members:  # cannot list (or already gone): fall back to the whole group
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        return
    for pid, command in members:
        if "multiprocessing.resource_tracker" in command:
            continue
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def supports_4bit(cfg: AgentConfig) -> bool:
    """Whether this machine's BloomBee checkout has the --compress_weight flag (4-bit base weights)."""
    try:
        cli = Path(cfg.bloombee_dir).expanduser() / "src" / "bloombee" / "cli" / "run_server.py"
        return "--compress_weight" in cli.read_text()
    except OSError:
        return False


EventCallback = Callable[[dict], Awaitable[None]]


@dataclass
class Worker:
    spec: WorkerSpec
    proc: asyncio.subprocess.Process
    started_at: float = field(default_factory=time.monotonic)
    peer_id: str = ""
    online: bool = False
    stop_reason: str = ""  # set when we ended it on purpose: killed | stopped | failed_to_start
    error: str = ""  # last error line seen before the worker came online
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
    def _free_port(self) -> Optional[int]:
        """First port in the configured range that is neither used by a running worker nor bound by anyone else."""
        import socket
        if not self.cfg.port_range:
            return None
        used = {w.spec.host_maddr for w in self.workers.values() if w.proc.returncode is None}
        for port in range(self.cfg.port_range[0], self.cfg.port_range[1] + 1):
            if f"/ip4/127.0.0.1/tcp/{port}" in used:
                continue
            with socket.socket() as sock:
                try:
                    sock.bind(("127.0.0.1", port))
                except OSError:
                    continue
            return port
        raise RuntimeError(f"no free worker port in {self.cfg.port_range[0]}-{self.cfg.port_range[1]}")

    async def start(self, spec: WorkerSpec) -> Worker:
        if spec.worker_id in self.workers and self.workers[spec.worker_id].proc.returncode is None:
            raise ValueError(f"worker {spec.worker_id} is already running")
        if spec.host_maddr is None:
            port = self._free_port()
            if port is not None:
                spec.host_maddr = f"/ip4/127.0.0.1/tcp/{port}"
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

    async def _fail_unless_online(self, worker: Worker, delay: float, why: str) -> None:
        await asyncio.sleep(delay)
        if not worker.online and worker.proc.returncode is None:
            worker.stop_reason, worker.error = "failed_to_start", worker.error or why
            try:
                kill_process_group(worker.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    async def _watch(self, worker: Worker, log_path: Path) -> None:
        spec = worker.spec
        startup_guard = asyncio.create_task(self._fail_unless_online(worker, self.cfg.startup_timeout,
                                                                     f"not online after {self.cfg.startup_timeout:.0f} s"))
        crash_guard: Optional[asyncio.Task] = None
        in_traceback = False
        with open(log_path, "ab") as log:
            assert worker.proc.stdout is not None
            while True:
                line = await worker.proc.stdout.readline()
                if not line:
                    break
                log.write(line)
                log.flush()
                text = line.decode(errors="replace").rstrip()
                if not worker.online:
                    if TRACEBACK_RE.match(text):
                        in_traceback = True
                    elif in_traceback and ERROR_LINE_RE.match(text):
                        in_traceback, worker.error = False, text[:300]
                        if crash_guard is None:
                            crash_guard = asyncio.create_task(self._fail_unless_online(worker, self.cfg.crash_grace, text[:300]))
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
        for guard in (startup_guard, crash_guard):
            if guard is not None:
                guard.cancel()
        reason = worker.stop_reason or "exited"
        event = {"type": "worker.exited", "worker_id": spec.worker_id, "returncode": code,
                 "reason": reason, "layers": [spec.layer_start, spec.layer_end]}
        if worker.error and not worker.online:
            event["error"] = worker.error
        await self.on_event(event)
        await self._update_caffeinate()

    async def kill(self, worker_id: str) -> bool:
        """Immediate SIGKILL of the whole process group (the demo 'kill this worker' button)."""
        w = self.workers.get(worker_id)
        if w is None or w.proc.returncode is not None:
            return False
        w.stop_reason = "killed"
        try:
            kill_process_group(w.proc.pid, signal.SIGKILL)
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
                kill_process_group(w.proc.pid, signal.SIGKILL)
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
