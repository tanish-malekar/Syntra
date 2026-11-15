"""Agent <-> real backend (uvicorn in the test's event loop): registration, heartbeats, busy detection,
commands over /ws/agent, worker events, reconnects and token revocation. Workers are fake (tests/fake_worker.py)."""
from __future__ import annotations

import asyncio
import socket
import sys
from pathlib import Path

import httpx
import pytest
import uvicorn
from sqlmodel import Session, select

from app.agent_hub import AgentCommandError
from app.config import Settings
from app.main import create_app
from app.db import utcnow
from app.models import Gpu, Job, Lease, User
from app.seed import seed_reference_data
from syntra_agent.client import Agent, AgentRevoked, AgentSettings
from syntra_agent.hardware import GpuInfo
from syntra_agent.workers import AgentConfig, WorkerManager

FAKE_WORKER = Path(__file__).resolve().parent / "fake_worker.py"
PASSWORD = "correct horse battery"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Platform:
    """The backend running in this event loop, plus a signed-in user with a provider and agent token."""

    def __init__(self, **settings):
        self.app = create_app(Settings(database_url="sqlite://", run_background_tasks=False, **settings))
        with Session(self.app.state.engine) as s:
            seed_reference_data(s)
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.task = None
        self.user = httpx.AsyncClient(base_url=self.url)

    async def start(self):
        self.task = asyncio.create_task(self.server.serve())
        while not self.server.started:
            await asyncio.sleep(0.02)
        r = await self.user.post("/auth/signup", json={"email": "owner@example.com", "password": PASSWORD})
        assert r.status_code == 201
        self.provider_id = (await self.user.post("/providers", json={"name": "Lab", "type": "enterprise"})).json()["id"]
        t = (await self.user.post(f"/providers/{self.provider_id}/agent-tokens", json={"label": "m"})).json()
        self.token, self.token_id = t["token"], t["id"]

    async def stop(self):
        await self.user.aclose()
        self.server.should_exit = True
        await self.task

    def gpus(self):
        with Session(self.app.state.engine) as s:
            return s.exec(select(Gpu).order_by(Gpu.id)).all()


class FakeHardware:
    """A detector whose answers the test controls."""

    def __init__(self):
        self.busy: set[int] = set()

    def __call__(self):
        return [GpuInfo(0, "NVIDIA RTX A6000", "cuda", 48000, 47000, busy_external=0 in self.busy,
                        busy_detail="process 99 uses 46000 MiB" if 0 in self.busy else ""),
                GpuInfo(1, "NVIDIA RTX A6000", "cuda", 48000, 47000, busy_external=1 in self.busy,
                        busy_detail="process 99 uses 46000 MiB" if 1 in self.busy else "")]


def make_agent(platform: Platform, hardware, tmp_path, lid=lambda: False, **settings):
    cfg = AgentConfig(work_dir=tmp_path, python=sys.executable)

    async def noop(event): ...

    events: list[dict] = []
    manager = WorkerManager(cfg, noop, lambda s, c: ([sys.executable, str(FAKE_WORKER), "--peer", f"Peer{s.worker_id}"], {}))
    agent = Agent(AgentSettings(server=platform.url, token=platform.token, region="lab", heartbeat_interval=0.2, **settings),
                  cfg, detector=hardware, manager=manager, lid_detector=lid)
    manager.on_event = agent._on_worker_event
    return agent


async def wait_for(predicate, timeout=8.0, what="condition"):
    async def loop():
        while not predicate():
            await asyncio.sleep(0.05)
    try:
        await asyncio.wait_for(loop(), timeout)
    except asyncio.TimeoutError:
        raise AssertionError(f"timed out waiting for {what}")


def start_cmd(worker_id="w1", device_index=0):
    return {"cmd": "start_worker", "worker_id": worker_id, "device": "cuda", "device_index": device_index,
            "model": {"name": "m", "hf_id": "a/b"}, "layer_start": 0, "layer_end": 11, "dht_prefix": "job1",
            "initial_peers": ["/ip4/127.0.0.1/tcp/1/p2p/QmX"]}


def test_agent_registers_heartbeats_and_runs_worker_commands(tmp_path):
    async def scenario():
        platform = Platform()
        await platform.start()
        hardware = FakeHardware()
        agent = make_agent(platform, hardware, tmp_path)
        events: list[dict] = []
        platform.app.state.agent_hub.add_listener(lambda token_id, msg: events.append(msg))
        run = asyncio.create_task(agent.run())
        try:
            # 1. both GPUs join the pool, idle, with the right class and price
            await wait_for(lambda: len(platform.gpus()) == 2, what="GPU registration")
            pool = (await platform.user.get("/gpus")).json()
            assert [g["status"] for g in pool] == ["idle", "idle"] and pool[0]["gpu_class"].startswith("NVIDIA workstation")
            assert pool[0]["region"] == "lab" and pool[0]["provider_type"] == "enterprise"
            await asyncio.wait_for(agent.connected.wait(), 5)
            assert platform.app.state.agent_hub.is_connected(platform.token_id)

            # 2. a GPU someone else starts using flips to busy_external, and back
            hardware.busy.add(1)
            await wait_for(lambda: {g.status for g in platform.gpus() if g.device_index == 1} == {"busy_external"}, what="busy_external")
            hub = platform.app.state.agent_hub
            with pytest.raises(AgentCommandError, match="busy"):
                await hub.request(platform.token_id, start_cmd("wb", device_index=1))  # refuses to start there
            hardware.busy.discard(1)
            await wait_for(lambda: {g.status for g in platform.gpus() if g.device_index == 1} == {"idle"}, what="idle again")

            # 3. start a worker on GPU 0: ack, started, online, and the GPU reads as leased by the worker
            reply = await hub.request(platform.token_id, start_cmd("w1"))
            assert reply["ok"] and reply["pid"] > 0
            await wait_for(lambda: any(e["type"] == "worker.online" for e in events), what="worker.online")
            started = next(e for e in events if e["type"] == "worker.started")
            assert started["peer_id"] == "Peerw1" and started["worker_id"] == "w1"
            listing = await hub.request(platform.token_id, {"cmd": "list_workers"})
            assert listing["workers"][0]["online"] is True and listing["workers"][0]["layers"] == [0, 11]
            # the lease table (not the agent's own report) decides 'leased': no lease row -> still idle
            await asyncio.sleep(0.6)
            assert {g.status for g in platform.gpus() if g.device_index == 0} == {"idle"}
            with Session(platform.app.state.engine) as db:
                owner = db.exec(select(User)).one()
                job = Job(owner_id=owner.id, kind="train", status="running")
                db.add(job); db.commit(); db.refresh(job)
                gpu0 = next(g for g in db.exec(select(Gpu)).all() if g.device_index == 0)
                lease = Lease(job_id=job.id, gpu_id=gpu0.id, layer_start=0, layer_end=11)
                db.add(lease); db.commit(); db.refresh(lease); lease_id = lease.id
            await wait_for(lambda: {g.status for g in platform.gpus() if g.device_index == 0} == {"leased"}, what="leased")

            # 4. kill it: exit reported with reason 'killed', GPU returns to idle
            assert (await hub.request(platform.token_id, {"cmd": "kill_worker", "worker_id": "w1"}))["ok"]
            await wait_for(lambda: any(e["type"] == "worker.exited" for e in events), what="worker.exited")
            exited = next(e for e in events if e["type"] == "worker.exited")
            assert exited["reason"] == "killed" and exited["layers"] == [0, 11]
            await asyncio.sleep(0.6)
            assert {g.status for g in platform.gpus() if g.device_index == 0} == {"leased"}  # lease still open
            with Session(platform.app.state.engine) as db:
                lease = db.get(Lease, lease_id); lease.ended_at = utcnow(); db.add(lease); db.commit()
            await wait_for(lambda: {g.status for g in platform.gpus() if g.device_index == 0} == {"idle"}, what="idle after lease ends")

            # 5. unknown commands and bad GPU indices are refused, not fatal
            with pytest.raises(AgentCommandError, match="unknown command"):
                await hub.request(platform.token_id, {"cmd": "format_disk"})
            with pytest.raises(AgentCommandError, match="unknown GPU"):
                await hub.request(platform.token_id, start_cmd("wx", device_index=9))
        finally:
            agent.stop()
            await asyncio.wait_for(run, 15)
            await platform.stop()
    asyncio.run(scenario())


def test_agent_reconnects_after_the_platform_drops_the_command_channel(tmp_path):
    async def scenario():
        platform = Platform()
        await platform.start()
        agent = make_agent(platform, FakeHardware(), tmp_path)
        run = asyncio.create_task(agent.run())
        try:
            hub = platform.app.state.agent_hub
            await wait_for(lambda: hub.is_connected(platform.token_id), what="first connection")
            ws = hub._conns[platform.token_id]
            await ws.close()
            await wait_for(lambda: not hub.is_connected(platform.token_id), what="drop")
            await wait_for(lambda: hub.is_connected(platform.token_id), timeout=10, what="reconnect")
            assert (await hub.request(platform.token_id, {"cmd": "list_workers"}))["ok"]
        finally:
            agent.stop()
            await asyncio.wait_for(run, 15)
            await platform.stop()
    asyncio.run(scenario())


def test_agent_reregisters_when_the_platform_forgets_its_gpus(tmp_path):
    async def scenario():
        platform = Platform()
        await platform.start()
        agent = make_agent(platform, FakeHardware(), tmp_path)
        run = asyncio.create_task(agent.run())
        try:
            await wait_for(lambda: len(platform.gpus()) == 2, what="registration")
            with Session(platform.app.state.engine) as s:
                for g in s.exec(select(Gpu)).all():
                    s.delete(g)
                s.commit()
            assert platform.gpus() == []  # the platform really forgot them
            await wait_for(lambda: len(platform.gpus()) == 2, what="re-registration")  # (SQLite may reuse the ids)
            await asyncio.sleep(0.6)
            assert [g.status for g in platform.gpus()] == ["idle", "idle"]  # and heartbeats keep flowing
        finally:
            agent.stop()
            await asyncio.wait_for(run, 15)
            await platform.stop()
    asyncio.run(scenario())


def test_revoked_token_stops_the_agent_with_an_auth_error(tmp_path):
    async def scenario():
        platform = Platform()
        await platform.start()
        await platform.user.delete(f"/providers/{platform.provider_id}/agent-tokens/{platform.token_id}")
        agent = make_agent(platform, FakeHardware(), tmp_path)
        with pytest.raises(AgentRevoked):
            await asyncio.wait_for(agent.run(), 10)
        assert agent.state == "revoked" and "revoked" in agent.last_error
        assert platform.gpus() == []
        await platform.stop()
    asyncio.run(scenario())


def test_simulated_agent_offers_several_gpus_and_honours_the_allowlist(tmp_path):
    async def scenario():
        platform = Platform()
        await platform.start()
        agent = make_agent(platform, lambda: [GpuInfo(0, "Apple M4 Pro", "mps", 24000, 9000)], tmp_path,
                           simulated=True, sim_count=4, sim_memory_gb=5, gpu_allowlist=[0, 2, 3])
        run = asyncio.create_task(agent.run())
        try:
            await wait_for(lambda: len(platform.gpus()) == 3, what="3 allowed simulated GPUs")
            names = sorted(g["name"] for g in (await platform.user.get("/gpus")).json())
            assert names == ["Apple M4 Pro (sim 1)", "Apple M4 Pro (sim 3)", "Apple M4 Pro (sim 4)"]
            assert all(g["simulated"] and g["gpu_class"] == "Apple M-series" and g["total_memory_mb"] == 5120
                       for g in (await platform.user.get("/gpus")).json())
        finally:
            agent.stop()
            await asyncio.wait_for(run, 15)
            await platform.stop()
    asyncio.run(scenario())


def test_closing_the_lid_reports_offline_and_kills_workers_at_once(tmp_path):
    async def scenario():
        platform = Platform()
        await platform.start()
        closed = {"v": False}
        agent = make_agent(platform, FakeHardware(), tmp_path, lid=lambda: closed["v"], lid_poll_interval=0.05)
        events: list[dict] = []
        platform.app.state.agent_hub.add_listener(lambda token_id, msg: events.append(msg))
        run = asyncio.create_task(agent.run())
        try:
            await wait_for(lambda: len(platform.gpus()) == 2 and agent.connected.is_set(), what="registered and connected")
            hub = platform.app.state.agent_hub
            assert (await hub.request(platform.token_id, start_cmd("w1")))["ok"]
            await wait_for(lambda: any(e["type"] == "worker.online" for e in events), what="worker online")
            closed["v"] = True  # the lid closes
            await wait_for(lambda: {g.status for g in platform.gpus()} == {"offline"}, 2.0, "GPUs offline right away")
            await wait_for(lambda: any(e["type"] == "worker.exited" for e in events), 2.0, "worker killed right away")
            await asyncio.sleep(0.6)  # normal heartbeats must not bring the GPUs back while the lid is closed
            assert {g.status for g in platform.gpus()} == {"offline"}
            with pytest.raises(AgentCommandError, match="lid"):
                await hub.request(platform.token_id, start_cmd("w2"))
            closed["v"] = False  # the lid opens: back in the pool
            await wait_for(lambda: {g.status for g in platform.gpus()} == {"idle"}, 3.0, "idle again")
        finally:
            agent.stop()
            await asyncio.wait_for(run, 15)
            await platform.stop()
    asyncio.run(scenario())


def test_a_relayed_agent_gets_its_ports_and_traffic_flows_through_the_platform_connection(tmp_path):
    """No tunnel and no open ports: the platform host listens on the agent's block and carries connections through the agent's
    WebSocket; the agent listens on the job DHT port and carries those back."""
    async def scenario():
        dht = free_port()
        platform = Platform(dht_port=dht, local_ports=None)
        await platform.start()
        echo_port = free_port()

        async def echo(r, w):
            while data := await r.read(65536):
                w.write(data)
                await w.drain()
            w.close()
        echo_server = await asyncio.start_server(echo, "127.0.0.1", echo_port)
        agent = make_agent(platform, FakeHardware(), tmp_path, relay=True, relay_dial=lambda port: ("127.0.0.1", echo_port))
        run = asyncio.create_task(agent.run())
        try:
            await wait_for(lambda: agent.cfg.port_range == (31100, 31109), what="the relay port block")
            await wait_for(lambda: agent.relay is not None and agent.relay.listening == {dht}, what="agent listening on the DHT port")
            reader, writer = await asyncio.open_connection("127.0.0.1", 31100)  # the platform host's listener for this agent
            writer.write(b"through the platform connection")
            await writer.drain()
            assert await asyncio.wait_for(reader.readexactly(31), 5) == b"through the platform connection"
            writer.close()
            # a second relay agent joining: each is told about the other's ports; leaving frees the block
            hub = platform.app.state.relay_hub
            assert hub.expose_for(platform.token_id) == [dht]
        finally:
            agent.stop()
            await asyncio.wait_for(run, 10)
            await wait_for(lambda: not platform.app.state.relay_hub.agents, what="relay detached")
            echo_server.close()
            await platform.stop()
    asyncio.run(scenario())


async def control_call(agent: Agent, method: str, path: str, body: dict | None = None, secret: str | None = None):
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{agent.control.port}") as c:
        return await c.request(method, path, json=body, headers={"X-Syntra-Secret": secret or agent.control.secret})


def test_control_api_pause_drain_resume_and_quit(tmp_path):
    async def scenario():
        platform = Platform()
        await platform.start()
        agent = make_agent(platform, FakeHardware(), tmp_path, control=True, state_dir=tmp_path / "state")
        run = asyncio.create_task(agent.run())
        hub = platform.app.state.agent_hub
        try:
            await wait_for(lambda: agent.state == "connected" and agent.connected.is_set(), what="connected")
            info = __import__("json").loads((tmp_path / "state" / "agent.json").read_text())
            assert info["port"] == agent.control.port and info["secret"] == agent.control.secret
            assert (await control_call(agent, "GET", "/status", secret="wrong")).status_code == 401
            st = (await control_call(agent, "GET", "/status")).json()
            assert st["state"] == "connected" and [g["status"] for g in st["gpus"]] == ["idle", "idle"]

            # a job is running on GPU 0; pause with drain keeps it there and takes GPU 1 out of the pool
            assert (await hub.request(platform.token_id, start_cmd("w1")))["ok"]
            st = (await control_call(agent, "POST", "/pause", {"mode": "drain"})).json()
            assert st["paused"] and st["draining"] and [g["status"] for g in st["gpus"]] == ["leased", "offline"]
            # the platform's lease table (not the agent) decides "leased"; this test has no lease, so GPU 0 stays idle there
            await wait_for(lambda: [g.status for g in platform.gpus()] == ["idle", "offline"], what="platform sees drain")
            with pytest.raises(AgentCommandError, match="paused"):
                await hub.request(platform.token_id, start_cmd("w2", device_index=1))

            # pause now stops the running work; resume brings both GPUs back
            st = (await control_call(agent, "POST", "/pause", {"mode": "now"})).json()
            await wait_for(lambda: not agent.manager.describe(), what="workers stopped")
            await wait_for(lambda: [g.status for g in platform.gpus()] == ["offline", "offline"], what="offline")
            assert (await control_call(agent, "POST", "/pause", {"mode": "later"})).status_code == 400
            st = (await control_call(agent, "POST", "/resume")).json()
            assert not st["paused"] and st["state"] == "connected"
            await wait_for(lambda: [g.status for g in platform.gpus()] == ["idle", "idle"], what="idle again")

            # quit: offline at once, agent exits, control info removed
            st = (await control_call(agent, "POST", "/quit")).json()
            assert st["state"] == "stopped"
            await asyncio.wait_for(run, 5)
            assert [g.status for g in platform.gpus()] == ["offline", "offline"]
            assert not (tmp_path / "state" / "agent.json").exists()
        finally:
            agent.stop()
            await asyncio.gather(run, return_exceptions=True)
            await platform.stop()
    asyncio.run(scenario())


def test_token_revoked_while_running_stops_the_agent(tmp_path):
    async def scenario():
        platform = Platform()
        await platform.start()
        agent = make_agent(platform, FakeHardware(), tmp_path)
        run = asyncio.create_task(agent.run())
        try:
            await wait_for(lambda: agent.state == "connected", what="connected")
            await platform.user.delete(f"/providers/{platform.provider_id}/agent-tokens/{platform.token_id}")
            await asyncio.wait_for(run, 5)
            assert agent.state == "revoked"
        finally:
            agent.stop()
            await asyncio.gather(run, return_exceptions=True)
            await platform.stop()
    asyncio.run(scenario())


def test_instance_lock_allows_one_agent_and_takes_over_a_stale_lock(tmp_path):
    from syntra_agent.control import AlreadyRunning, InstanceLock
    lock = InstanceLock(tmp_path)
    lock.acquire()
    (tmp_path / "agent.lock").write_text("1")          # pid 1 (launchd) is alive: a live other agent
    with pytest.raises(AlreadyRunning):
        InstanceLock(tmp_path).acquire()
    (tmp_path / "agent.lock").write_text("999999")     # no such process: stale, taken over
    InstanceLock(tmp_path).acquire()
    assert (tmp_path / "agent.lock").read_text() == str(__import__("os").getpid())


def test_app_pairing_and_machine_usage(tmp_path):
    async def scenario():
        platform = Platform()
        await platform.start()
        anon = httpx.AsyncClient(base_url=platform.url)
        try:
            bad = await anon.post("/app/pair", json={"email": "owner@example.com", "password": "nope", "machine_id": "mac-uuid-1"})
            assert bad.status_code == 401
            first = (await anon.post("/app/pair", json={"email": "owner@example.com", "password": PASSWORD,
                                                       "machine_id": "mac-uuid-1", "machine_name": "Ryan's MacBook"})).json()
            assert first["provider_id"] == platform.provider_id      # reuses the existing provider profile
            again = (await anon.post("/app/pair", json={"email": "owner@example.com", "password": PASSWORD,
                                                       "machine_id": "mac-uuid-1", "machine_name": "Ryan's MacBook"})).json()
            old = httpx.AsyncClient(base_url=platform.url, headers={"Authorization": f"Bearer {first['token']}"})
            new = httpx.AsyncClient(base_url=platform.url, headers={"Authorization": f"Bearer {again['token']}"})
            assert (await old.get("/agent/me")).status_code == 401  # re-pairing the same machine revoked the old token
            assert (await new.get("/agent/me")).status_code == 200

            platform.token = again["token"]
            agent = make_agent(platform, FakeHardware(), tmp_path)
            run = asyncio.create_task(agent.run())
            await wait_for(lambda: len(platform.gpus()) == 2, what="GPUs registered")
            usage = (await new.get("/agent/usage")).json()
            assert usage["machine"] == "Ryan's MacBook" and len(usage["gpus"]) == 2
            assert usage["machine_total"]["gpu_seconds_total"] == 0 and not usage["machine_total"]["in_use"]

            # a lease runs on GPU 0 for a while: live estimate, then billed when it ends
            from datetime import timedelta
            from app.services import scheduler
            with Session(platform.app.state.engine) as s:
                user = s.exec(select(User)).first()
                gpu = s.exec(select(Gpu).order_by(Gpu.id)).first()
                job = Job(owner_id=user.id, kind="train", base_model_id=1)
                s.add(job); s.commit(); s.refresh(job)
                lease = Lease(job_id=job.id, gpu_id=gpu.id, layer_start=0, layer_end=22, started_at=utcnow() - timedelta(hours=1))
                s.add(lease); s.commit(); s.refresh(lease)
            usage = (await new.get("/agent/usage")).json()
            g0 = usage["gpus"][0]
            assert g0["in_use"] and g0["current"]["job_kind"] == "train" and 3590 < g0["gpu_seconds_running"] < 3700
            assert g0["earned_running_estimate"] == pytest.approx(g0["price_per_hour"], rel=0.05) and g0["earned_settled"] == 0
            with Session(platform.app.state.engine) as s:
                scheduler.release_lease(s, platform.app.state.events, s.get(Lease, lease.id), "completed")
            usage = (await new.get("/agent/usage")).json()
            g0 = usage["gpus"][0]
            assert not g0["in_use"] and g0["earned_running_estimate"] == 0
            assert g0["earned_settled"] == pytest.approx(g0["price_per_hour"], rel=0.05) and g0["jobs_served"] == 1
            assert usage["account_earned_settled"] == pytest.approx(g0["earned_settled"])
            assert len(g0["last_days"]) == 1 and g0["gpu_seconds_today"] > 0

            assert (await new.post("/agent/unpair")).status_code == 204
            await asyncio.wait_for(run, 5)                       # the agent notices and stops
            assert agent.state == "revoked"
            await old.aclose(); await new.aclose()
        finally:
            await anon.aclose()
            await platform.stop()
    asyncio.run(scenario())
