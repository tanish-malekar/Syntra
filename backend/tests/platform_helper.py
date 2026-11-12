"""A real backend (uvicorn in the test's event loop) plus users and provider agents with fake workers."""
from __future__ import annotations

import asyncio
import socket
import sys
from pathlib import Path

import httpx
import uvicorn
from sqlmodel import Session, select

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent"))
FAKE_WORKER = ROOT / "agent" / "tests" / "fake_worker.py"
FAKE_RUNNER = Path(__file__).resolve().parent / "fake_runner.py"
FAKE_MERGE = Path(__file__).resolve().parent / "fake_merge.py"

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.seed import seed_reference_data  # noqa: E402
from syntra_agent.client import Agent, AgentSettings  # noqa: E402
from syntra_agent.hardware import GpuInfo  # noqa: E402
from syntra_agent.workers import AgentConfig, WorkerManager  # noqa: E402

PASSWORD = "correct horse battery"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def wait_for(predicate, timeout=20.0, what="condition"):
    async def loop():
        while True:
            result = predicate()
            if asyncio.iscoroutine(result):
                result = await result
            if result:
                return result
            await asyncio.sleep(0.1)
    try:
        return await asyncio.wait_for(loop(), timeout)
    except asyncio.TimeoutError:
        raise AssertionError(f"timed out waiting for {what}")


class Platform:
    def __init__(self, tmp_path: Path, **settings):
        defaults = dict(database_url=f"sqlite:///{tmp_path}/syntra.db", run_background_tasks=True, sweep_interval_seconds=0.5,
                        heartbeat_timeout_seconds=3.0, orchestrator_interval=0.2, runner_argv=(sys.executable, str(FAKE_RUNNER)), merge_argv=(sys.executable, str(FAKE_MERGE)),
                        jobs_dir=str(tmp_path / "jobs"), registry_dir=str(tmp_path / "registry"),
                        datasets_dir=str(tmp_path / "datasets"))
        defaults.update(settings)
        self.tmp = tmp_path
        self.app = create_app(Settings(**defaults))
        with Session(self.app.state.engine) as s:
            seed_reference_data(s)
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.users: dict[str, httpx.AsyncClient] = {}
        self.agents: list[Agent] = []
        self.agent_tasks: list[asyncio.Task] = []

    async def __aenter__(self):
        self.task = asyncio.create_task(self.server.serve())
        while not self.server.started:
            await asyncio.sleep(0.02)
        return self

    async def __aexit__(self, *exc):
        for agent in self.agents:
            agent.stop()
        await asyncio.gather(*(asyncio.wait_for(t, 20) for t in self.agent_tasks), return_exceptions=True)
        for c in self.users.values():
            await c.aclose()
        self.server.should_exit = True
        await self.task

    async def user(self, name: str) -> httpx.AsyncClient:
        c = httpx.AsyncClient(base_url=self.url, timeout=30)
        r = await c.post("/auth/signup", json={"email": f"{name}@example.com", "password": PASSWORD, "display_name": name})
        assert r.status_code == 201, r.text
        self.users[name] = c
        return c

    async def provider(self, user: httpx.AsyncClient, name: str, gpus=3, memory_mb=6000, chip="Apple M4 Pro",
                       device="mps", simulated=True, type="individual") -> Agent:
        pid = (await user.post("/providers", json={"name": name, "type": type})).json()["id"]
        token = (await user.post(f"/providers/{pid}/agent-tokens", json={"label": name})).json()["token"]
        cfg = AgentConfig(work_dir=self.tmp / f"agent-{name}")
        manager = WorkerManager(cfg, None, lambda s, c: ([sys.executable, str(FAKE_WORKER), "--peer", f"Peer{s.worker_id}"], {}))
        hardware = lambda: [GpuInfo(0, chip, device, memory_mb * gpus, memory_mb * gpus)]
        agent = Agent(AgentSettings(server=self.url, token=token, region=name, heartbeat_interval=0.3, simulated=simulated,
                                    sim_count=gpus, sim_memory_gb=memory_mb / 1024), cfg, detector=hardware, manager=manager)
        manager.on_event = agent._on_worker_event
        self.agents.append(agent)
        self.agent_tasks.append(asyncio.create_task(agent.run()))
        await asyncio.wait_for(agent.connected.wait(), 10)
        return agent

    def gpus_sync(self):
        from app.models import Gpu
        with Session(self.app.state.engine) as s:
            return s.exec(select(Gpu).order_by(Gpu.id)).all()

    async def pool(self, user) -> list[dict]:
        return (await user.get("/gpus")).json()

    async def events(self, user, job_id) -> list[dict]:
        return (await user.get("/events", params={"job_id": job_id})).json()

    async def job(self, user, job_id) -> dict:
        return (await user.get(f"/jobs/{job_id}")).json()

    async def wait_status(self, user, job_id, *wanted, timeout=30):
        return await wait_for(lambda: self._status_is(user, job_id, wanted), timeout, f"job {job_id} -> {wanted}")

    async def _status_is(self, user, job_id, wanted):
        job = await self.job(user, job_id)
        return job if job["status"] in wanted else None

    async def wait_event(self, user, job_id, etype, count=1, timeout=30):
        async def check():
            evs = [e for e in await self.events(user, job_id) if e["type"] == etype]
            return evs if len(evs) >= count else None
        return await wait_for(check, timeout, f"{count}x {etype} for job {job_id}")

    async def train_model(self, user, **over) -> dict:
        job = await self.start_job(user, **over)
        done = await self.wait_status(user, job["id"], "completed", "failed", timeout=60)
        assert done["status"] == "completed", done["result"]
        models = (await user.get("/models")).json()
        return models[0]

    async def deploy(self, user, wait=True, **body) -> dict:
        r = await user.post("/deployments", json=body)
        assert r.status_code == 201, r.text
        dep = r.json()
        if wait:
            dep = await wait_for(lambda: self._dep_is(user, dep["id"], ("running", "failed")), 60, f"deployment {dep['id']} running")
            assert dep["status"] == "running", dep
        return dep

    async def _dep_is(self, user, dep_id, wanted):
        d = (await user.get(f"/deployments/{dep_id}")).json()
        return d if d["status"] in wanted else None

    async def start_job(self, user, **over):
        cfg = {"base_model_id": 1, "dataset_id": 1, "steps": 8, **over}
        r = await user.post("/jobs", json=cfg)
        assert r.status_code == 201, r.text
        return r.json()
