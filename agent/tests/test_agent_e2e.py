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
from syntra_agent.client import Agent, AgentSettings
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

    def __init__(self):
        self.app = create_app(Settings(database_url="sqlite://", run_background_tasks=False))
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


def make_agent(platform: Platform, hardware, tmp_path, **settings):
    cfg = AgentConfig(work_dir=tmp_path, python=sys.executable)

    async def noop(event): ...

    events: list[dict] = []
    manager = WorkerManager(cfg, noop, lambda s, c: ([sys.executable, str(FAKE_WORKER), "--peer", f"Peer{s.worker_id}"], {}))
    agent = Agent(AgentSettings(server=platform.url, token=platform.token, region="lab", heartbeat_interval=0.2, **settings),
                  cfg, detector=hardware, manager=manager)
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
        with pytest.raises(httpx.HTTPStatusError) as err:
            await asyncio.wait_for(agent.run(), 10)
        assert err.value.response.status_code == 401
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
