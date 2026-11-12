"""The agent: registers GPUs, heartbeats, and executes platform commands over a WebSocket."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import httpx
from websockets.asyncio.client import connect as ws_connect

from syntra_agent import hardware
from syntra_agent.files import ensure_model_files
from syntra_agent.hardware import GpuInfo
from syntra_agent.workers import AgentConfig, WorkerManager, WorkerSpec

log = logging.getLogger("syntra_agent")


@dataclass
class AgentSettings:
    server: str  # http://host:8000
    token: str
    region: str = ""
    heartbeat_interval: float = 2.0
    lid_poll_interval: float = 0.3  # how often to look at the laptop lid (a closed lid means the machine is about to sleep)
    gpu_allowlist: Optional[list[int]] = None  # only these device indices are offered to the pool
    simulated: bool = False
    sim_count: int = 2
    sim_memory_gb: float = 6.0

    @property
    def ws_url(self) -> str:
        return self.server.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/ws/agent"


class Agent:
    def __init__(self, settings: AgentSettings, cfg: AgentConfig,
                 detector: Callable[[], list[GpuInfo]] | None = None, manager: WorkerManager | None = None,
                 lid_detector: Callable[[], bool] | None = None):
        self.s, self.cfg = settings, cfg
        self.manager = manager or WorkerManager(cfg, self._on_worker_event)
        self._detector = detector
        self.gpu_ids: dict[int, int] = {}  # device index -> platform gpu id
        self.last: dict[int, GpuInfo] = {}
        self._outbox: asyncio.Queue = asyncio.Queue()
        self._http: Optional[httpx.AsyncClient] = None
        self._stop = asyncio.Event()
        self.connected = asyncio.Event()
        self._detect_task: Optional[asyncio.Future] = None
        self._lid_detector = lid_detector or hardware.lid_closed
        self.lid_closed = False

    # ---- hardware ---------------------------------------------------------------------------------
    def detect(self) -> list[GpuInfo]:
        real = self._detector() if self._detector else hardware.detect(self.manager.pids())
        gpus = hardware.simulate(real, self.s.sim_count, self.s.sim_memory_gb) if self.s.simulated else real
        if self.s.gpu_allowlist is not None:
            gpus = [g for g in gpus if g.index in self.s.gpu_allowlist]
        return gpus

    async def detect_now(self, max_wait: float) -> Optional[list[GpuInfo]]:
        """The current hardware state, or None if the driver does not answer within max_wait.

        nvidia-smi can block for minutes in the kernel (uninterruptibly, so even its timeout does not fire) right after a CUDA
        process is SIGKILLed. One probe runs at a time; a slow probe is not abandoned, a later call picks up its result.
        Heartbeats must not wait for it: a silent agent makes the platform declare its healthy workers dead.
        """
        if self._detect_task is None or self._detect_task.done():
            self._detect_task = asyncio.ensure_future(asyncio.to_thread(self.detect))
        done, _ = await asyncio.wait({self._detect_task}, timeout=max_wait)
        if not done:
            return None
        try:
            return self._detect_task.result()
        except Exception as exc:  # noqa: BLE001
            log.warning("hardware probe failed: %s", exc)
            return None

    # ---- registration + heartbeat -----------------------------------------------------------------
    async def register(self) -> None:
        assert self._http is not None
        for gpu in await asyncio.to_thread(self.detect):
            r = await self._http.post("/agent/gpus", json={
                "name": gpu.name, "device": gpu.device, "device_index": gpu.index,
                "total_memory_mb": gpu.total_memory_mb, "free_memory_mb": gpu.free_memory_mb,
                "region": self.s.region, "simulated": gpu.simulated,
            })
            r.raise_for_status()
            self.gpu_ids[gpu.index] = r.json()["id"]
            self.last[gpu.index] = gpu
            log.info("registered %s (index %d) as gpu %d", gpu.name, gpu.index, self.gpu_ids[gpu.index])

    def _reported_status(self, gpu: GpuInfo) -> str:
        if self.lid_closed:
            return "offline"
        if gpu.busy_external:
            return "busy_external"
        return "leased" if self.manager.on_gpu(gpu.index) else "idle"

    def _free_memory(self, gpu: GpuInfo) -> int:
        if gpu.simulated:  # a simulated GPU is "full" by whatever its workers hold
            return max(0, gpu.total_memory_mb - 1200 * len(self.manager.on_gpu(gpu.index)))
        return gpu.free_memory_mb

    async def heartbeat_once(self) -> None:
        assert self._http is not None
        gpus = await self.detect_now(self.s.heartbeat_interval * 2)
        if gpus is None:  # driver not answering: keep heartbeating with the last known state
            log.warning("GPU driver not answering; heartbeating with the last known state")
            gpus = list(self.last.values())
        for gpu in gpus:
            self.last[gpu.index] = gpu
            gid = self.gpu_ids.get(gpu.index)
            if gid is None:
                continue
            r = await self._http.post(f"/agent/gpus/{gid}/heartbeat", json={
                "free_memory_mb": self._free_memory(gpu), "status": self._reported_status(gpu),
                "detail": gpu.busy_detail,
            })
            if r.status_code == 404:  # the platform lost this GPU (e.g. database reset): register again
                await self.register()
            elif r.status_code >= 400:
                log.warning("heartbeat for gpu %s failed: %s %s", gid, r.status_code, r.text[:120])

    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.heartbeat_once()
            except (httpx.HTTPError, OSError) as exc:
                log.warning("heartbeat failed: %s", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), self.s.heartbeat_interval)
            except asyncio.TimeoutError:
                pass

    # ---- laptop lid ---------------------------------------------------------------------------------
    async def _go_offline(self) -> None:
        """Lid closed: the machine sleeps within moments. Tell the platform first (so no replacement is placed on this machine's
        other GPUs), then kill the workers at once, so the platform replaces them now instead of after a heartbeat timeout."""
        assert self._http is not None

        async def say(gid: int) -> None:
            try:
                await self._http.post(f"/agent/gpus/{gid}/heartbeat", json={"free_memory_mb": 0, "status": "offline", "detail": "lid closed"}, timeout=2.0)
            except (httpx.HTTPError, OSError):
                pass
        await asyncio.gather(*(say(g) for g in self.gpu_ids.values()))
        await asyncio.gather(*(self.manager.kill(w["worker_id"]) for w in self.manager.describe()))
        log.warning("lid closed: GPUs reported offline, workers killed")

    async def _lid_loop(self) -> None:
        while not self._stop.is_set():
            closed = await asyncio.to_thread(self._lid_detector)
            if closed and not self.lid_closed:
                self.lid_closed = True
                await self._go_offline()
            elif not closed and self.lid_closed:
                self.lid_closed = False
                log.info("lid opened: GPUs available again")
            try:
                await asyncio.wait_for(self._stop.wait(), self.s.lid_poll_interval)
            except asyncio.TimeoutError:
                pass

    # ---- worker events -> platform ----------------------------------------------------------------
    async def _on_worker_event(self, event: dict) -> None:
        await self._outbox.put(event)

    # ---- command channel --------------------------------------------------------------------------
    async def _handle_command(self, cmd: dict) -> dict:
        name = cmd.get("cmd")
        try:
            if name == "start_worker":
                if self.lid_closed:
                    return {"ok": False, "error": "the lid of this machine is closed"}
                spec = WorkerSpec.from_command(cmd)
                gpu = self.last.get(spec.device_index)
                if gpu is None:
                    return {"ok": False, "error": f"unknown GPU index {spec.device_index}"}
                probed = await self.detect_now(20.0)
                if probed is None:  # never start on a GPU whose state cannot be checked
                    return {"ok": False, "error": f"GPU {spec.device_index}: the driver is not answering, try another GPU"}
                fresh = {g.index: g for g in probed}.get(spec.device_index, gpu)
                if fresh.busy_external:  # never start on a GPU somebody else is using
                    return {"ok": False, "error": f"GPU {spec.device_index} is busy: {fresh.busy_detail}"}
                fetch = cmd.get("fetch")
                if fetch and not (spec.model.get("path") and Path(spec.model["path"]).expanduser().exists()):
                    dest = Path(self.cfg.models_dir).expanduser() / f"finetuned-{fetch['model_id']}"
                    assert self._http is not None
                    spec.model = {**spec.model, "path": str(await ensure_model_files(self._http, fetch["model_id"], dest))}
                worker = await self.manager.start(spec)
                return {"ok": True, "pid": worker.proc.pid}
            if name == "kill_worker":
                return {"ok": await self.manager.kill(cmd["worker_id"])}
            if name == "stop_worker":
                return {"ok": await self.manager.stop(cmd["worker_id"])}
            if name == "list_workers":
                return {"ok": True, "workers": self.manager.describe()}
            return {"ok": False, "error": f"unknown command {name!r}"}
        except Exception as exc:  # report, never crash the agent
            log.exception("command %s failed", name)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    async def _channel_once(self) -> None:
        headers = {"Authorization": f"Bearer {self.s.token}"}
        async with ws_connect(self.s.ws_url, additional_headers=headers, ping_interval=20, max_size=8 * 1024 * 1024) as ws:
            hello = json.loads(await ws.recv())
            log.info("connected to platform: %s", hello)
            self.connected.set()

            async def pump_outbox() -> None:
                while True:
                    await ws.send(json.dumps(await self._outbox.get()))

            pump = asyncio.create_task(pump_outbox())
            try:
                async for raw in ws:
                    cmd = json.loads(raw)
                    if "cmd" not in cmd:
                        continue
                    reply = await self._handle_command(cmd)
                    await ws.send(json.dumps({"type": "ack", "id": cmd.get("id"), **reply}))
            finally:
                pump.cancel()
                self.connected.clear()

    async def _channel_loop(self) -> None:
        delay = 1.0
        while not self._stop.is_set():
            try:
                await self._channel_once()
                delay = 1.0
            except (OSError, httpx.HTTPError, Exception) as exc:  # reconnect with backoff; workers keep running
                if self._stop.is_set():
                    break
                log.warning("command channel lost (%s); retrying in %.0fs", exc, delay)
            try:
                await asyncio.wait_for(self._stop.wait(), delay)
            except asyncio.TimeoutError:
                pass
            delay = min(delay * 2, 10.0)

    # ---- run --------------------------------------------------------------------------------------
    async def run(self) -> None:
        self._http = httpx.AsyncClient(base_url=self.s.server, headers={"Authorization": f"Bearer {self.s.token}"},
                                       timeout=10.0)
        try:
            me = await self._http.get("/agent/me")
            me.raise_for_status()
            log.info("authenticated as provider %r (%s)", me.json()["provider_name"], me.json()["provider_type"])
            await self.register()
            tasks = [asyncio.create_task(self._heartbeat_loop()), asyncio.create_task(self._channel_loop()),
                     asyncio.create_task(self._lid_loop())]
            stopper = asyncio.create_task(self._stop.wait())
            await asyncio.wait([*tasks, stopper], return_when=asyncio.FIRST_COMPLETED)
            for t in (*tasks, stopper):
                t.cancel()
            results = await asyncio.gather(*tasks, stopper, return_exceptions=True)
            for r in results:  # surface a real failure, ignore cancellation caused by stop()
                if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
                    raise r
        finally:
            await self.manager.stop_all()
            await self._http.aclose()

    def stop(self) -> None:
        self._stop.set()
