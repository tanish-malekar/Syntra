"""The agent: registers GPUs, heartbeats, and executes platform commands over a WebSocket."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import httpx
from websockets.asyncio.client import connect as ws_connect

from syntra_agent import hardware
from syntra_agent.files import ensure_model_files
from syntra_agent.hardware import GpuInfo
from syntra_agent.control import ControlServer
from syntra_agent.relay import Relay
from syntra_agent.workers import AgentConfig, WorkerManager, WorkerSpec, supports_4bit

log = logging.getLogger("syntra_agent")


class AgentRevoked(RuntimeError):
    """The platform rejected this machine's token (revoked on the website, or the app signed out)."""


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
    relay: bool = False  # no inbound connections needed: GPU worker traffic is carried over the platform connection
    relay_dial: Optional[Callable[[int], tuple[str, int]]] = None  # tests only: where an incoming relayed port is dialled
    control: bool = False  # serve the local control API for the desktop app (127.0.0.1, secret in <state_dir>/agent.json)
    state_dir: Path = field(default_factory=lambda: Path("~/.syntra").expanduser())

    @property
    def ws_url(self) -> str:
        url = self.server.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/ws/agent"
        return url + "?relay=1" if self.relay else url


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
        self.relay: Optional[Relay] = None  # set while connected in relay mode
        self.supports_4bit = supports_4bit(cfg)  # 4-bit base weights (QLoRA); needs a BloomBee with --compress_weight
        # what the desktop app shows
        self.paused = False  # the owner stopped sharing; running work may still be draining
        self.state = "starting"  # starting | connected | reconnecting | paused | revoked | stopped
        self.last_error = ""
        self.last_heartbeat_ok: Optional[float] = None  # wall-clock seconds
        self.provider_name = ""
        self.control: Optional[ControlServer] = None

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
                "region": self.s.region, "simulated": gpu.simulated, "supports_4bit": self.supports_4bit,
            })
            r.raise_for_status()
            self.gpu_ids[gpu.index] = r.json()["id"]
            self.last[gpu.index] = gpu
            log.info("registered %s (index %d) as gpu %d", gpu.name, gpu.index, self.gpu_ids[gpu.index])

    def _reported_status(self, gpu: GpuInfo) -> str:
        if self.lid_closed:
            return "offline"
        working = bool(self.manager.on_gpu(gpu.index))
        if self.paused:  # draining: a GPU still running a job stays leased until that job ends, then it goes offline
            return "leased" if working else "offline"
        if gpu.busy_external:
            return "busy_external"
        return "leased" if working else "idle"

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
                "detail": "paused by the owner" if self.paused else gpu.busy_detail,
            })
            if r.status_code == 401:
                self._revoked()
                return
            if r.status_code == 404:  # the platform lost this GPU (e.g. database reset): register again
                await self.register()
            elif r.status_code >= 400:
                log.warning("heartbeat for gpu %s failed: %s %s", gid, r.status_code, r.text[:120])
                self.last_error = f"heartbeat: HTTP {r.status_code}"
            else:
                self.last_heartbeat_ok = time.time()
                if self.state in ("starting", "reconnecting"):
                    self.state = "paused" if self.paused else "connected"

    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.heartbeat_once()
            except (httpx.HTTPError, OSError) as exc:
                log.warning("heartbeat failed: %s", exc)
                self.last_error = f"cannot reach the platform: {exc}"
                if self.state in ("connected", "paused", "starting"):
                    self.state = "reconnecting"
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
                if self.paused:
                    return {"ok": False, "error": "the owner paused sharing on this machine"}
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
        async with ws_connect(self.s.ws_url, additional_headers=headers, ping_interval=20, ping_timeout=60, max_size=8 * 1024 * 1024) as ws:
            hello = json.loads(await ws.recv())
            log.info("connected to platform: %s", hello)
            relay = self.relay = Relay(ws.send, parity=1, dial=self.s.relay_dial) if self.s.relay else None
            self.connected.set()

            # What this agent knows about its running workers is sent again on every (re)connection: an event that was being sent
            # when the connection dropped would otherwise be lost for good, and the platform would wait for it forever.
            for w in self.manager.workers.values():
                if w.proc.returncode is None and w.peer_id:
                    self._outbox.put_nowait({"type": "worker.started", "worker_id": w.spec.worker_id, "pid": w.proc.pid, "peer_id": w.peer_id})
                    if w.online:
                        self._outbox.put_nowait({"type": "worker.online", "worker_id": w.spec.worker_id, "peer_id": w.peer_id, "seconds_to_online": 0})

            async def pump_outbox() -> None:
                while True:
                    event = await self._outbox.get()
                    try:
                        await ws.send(json.dumps(event))
                    except BaseException:
                        if not isinstance(event, dict) or event.get("type") not in ("worker.started", "worker.online"):
                            self._outbox.put_nowait(event)  # exits are not re-sent by the resync above, so keep them for the next connection
                        raise

            # Relayed frames are handled by their own task. Handling them in this loop let one slow local worker (writing to it
            # blocks) stop the loop from reading the socket, so keep-alive replies went unread and the platform connection was dropped.
            frames: asyncio.Queue = asyncio.Queue()

            async def carry_frames() -> None:
                while True:
                    await relay.handle(await frames.get())

            pump = asyncio.create_task(pump_outbox())
            carrier = asyncio.create_task(carry_frames()) if relay is not None else None
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):  # a relayed TCP frame
                        if relay is not None:
                            frames.put_nowait(raw)
                        continue
                    cmd = json.loads(raw)
                    if cmd.get("type") == "relay_config" and relay is not None:
                        self.cfg.port_range = (cmd["port_range"][0], cmd["port_range"][1])  # worker ports for this machine
                        log.info("relay: this machine's worker ports are %d-%d", *self.cfg.port_range)
                        continue
                    if cmd.get("type") == "relay_expose" and relay is not None:  # the other machines' ports, reached via the platform
                        await relay.listen(p for p in cmd["ports"] if not self.cfg.port_range or not self.cfg.port_range[0] <= p <= self.cfg.port_range[1])
                        continue
                    if "cmd" not in cmd:
                        continue
                    reply = await self._handle_command(cmd)
                    await ws.send(json.dumps({"type": "ack", "id": cmd.get("id"), **reply}))
            finally:
                pump.cancel()
                if carrier is not None:
                    carrier.cancel()
                if relay is not None:
                    await relay.close()
                self.relay = None
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

    # ---- desktop app controls ---------------------------------------------------------------------
    def _revoked(self) -> None:
        log.error("the platform rejected this machine's token (revoked or signed out); stopping")
        self.state, self.last_error = "revoked", "This machine was signed out or its token was revoked."
        self.stop()

    async def pause(self, mode: str = "drain") -> dict:
        """Stop sharing. drain: running jobs keep this GPU until they end; now: stop them (the platform moves them elsewhere)."""
        if mode not in ("drain", "now"):
            raise ValueError("mode must be drain or now")
        self.paused, self.state = True, "paused"
        if mode == "now":
            await asyncio.gather(*(self.manager.kill(w["worker_id"]) for w in self.manager.describe()))
        await self._heartbeat_quietly()
        return self.status()

    async def resume(self) -> dict:
        self.paused = False
        self.state = "connected" if self.connected.is_set() else "reconnecting"
        await self._heartbeat_quietly()
        return self.status()

    async def quit(self) -> dict:
        """Go offline at once (no waiting for a heartbeat timeout), stop the workers, then exit."""
        self.paused = True
        await asyncio.gather(*(self.manager.kill(w["worker_id"]) for w in self.manager.describe()))
        await self._heartbeat_quietly()
        self.state = "stopped"
        asyncio.get_running_loop().call_later(0.2, self.stop)  # let the reply go out first
        return self.status()

    async def _heartbeat_quietly(self) -> None:
        try:
            await self.heartbeat_once()
        except (httpx.HTTPError, OSError) as exc:
            log.warning("heartbeat failed: %s", exc)

    def status(self) -> dict:
        workers = self.manager.describe()
        gpus = []
        for index, gpu in sorted(self.last.items()):
            gpus.append({"index": index, "name": gpu.name, "device": gpu.device, "platform_id": self.gpu_ids.get(index),
                         "total_memory_mb": gpu.total_memory_mb, "status": self._reported_status(gpu),
                         "busy_detail": gpu.busy_detail, "workers": len(self.manager.on_gpu(index))})
        return {"state": self.state, "paused": self.paused, "connected": self.connected.is_set(), "lid_closed": self.lid_closed,
                "server": self.s.server, "provider_name": self.provider_name, "supports_4bit": self.supports_4bit,
                "last_heartbeat_ok": self.last_heartbeat_ok, "last_error": self.last_error,
                "draining": self.paused and bool(workers), "gpus": gpus, "workers": workers}

    async def _start_control(self) -> None:
        async def get_status(_):
            return self.status()

        async def post_pause(body):
            return await self.pause(body.get("mode", "drain"))

        async def post_resume(_):
            return await self.resume()

        async def post_quit(_):
            return await self.quit()

        self.control = ControlServer({("GET", "/status"): get_status, ("POST", "/pause"): post_pause,
                                      ("POST", "/resume"): post_resume, ("POST", "/quit"): post_quit}, self.s.state_dir)
        await self.control.start()

    # ---- run --------------------------------------------------------------------------------------
    async def run(self) -> None:
        self._http = httpx.AsyncClient(base_url=self.s.server, headers={"Authorization": f"Bearer {self.s.token}"},
                                       timeout=10.0)
        try:
            if self.s.control:
                await self._start_control()
            me = await self._authenticate()
            self.provider_name = me["provider_name"]
            log.info("authenticated as provider %r (%s)", me["provider_name"], me["provider_type"])
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
            if self.control is not None:
                await self.control.close()

    async def _authenticate(self) -> dict:
        """Wait for the platform (the app may start before the network is up); give up only on a rejected token."""
        delay = 1.0
        while True:
            try:
                me = await self._http.get("/agent/me")
                if me.status_code == 401:
                    self._revoked()
                    raise AgentRevoked(self.last_error)
                me.raise_for_status()
                return me.json()
            except (httpx.HTTPError, OSError) as exc:
                self.state, self.last_error = "reconnecting", f"cannot reach the platform: {exc}"
                log.warning("platform not reachable (%s); retrying in %.0fs", exc, delay)
                try:
                    await asyncio.wait_for(self._stop.wait(), delay)
                    raise asyncio.CancelledError  # stopped while waiting
                except asyncio.TimeoutError:
                    pass
                delay = min(delay * 2, 30.0)

    def stop(self) -> None:
        self._stop.set()
