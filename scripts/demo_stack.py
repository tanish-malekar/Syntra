"""Demo / UI-development stack: the real backend and real provider agents, with a FAKE swarm (no GPU work).

Training runs finish in about a minute and deployments answer instantly, so every page can be exercised without waiting for models.

  backend/.venv/bin/python scripts/demo_stack.py            # API on :8000, writes the demo logins to demo-logins.txt (committed)
  cd frontend && npm run dev                                 # UI on :5173
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


async def serve() -> None:
    import uvicorn
    from tests.platform_helper import FAKE_MERGE, FAKE_RUNNER, FAKE_WORKER  # noqa: F401
    from app.config import Settings
    from app.main import create_app
    from app.seed import seed_reference_data
    from sqlmodel import Session

    work = Path(tempfile.mkdtemp(prefix="syntra-demo-"))
    settings = Settings(
        database_url=f"sqlite:///{work}/demo.db", sweep_interval_seconds=1.0, heartbeat_timeout_seconds=4.0, orchestrator_interval=0.3,
        runner_argv=(sys.executable, str(FAKE_RUNNER)), merge_argv=(sys.executable, str(FAKE_MERGE)),
        runner_spec_extra={"fake_step_delay": 0.5, "fake_ready_delay": 1.5}, jobs_dir=str(work / "jobs"),
        registry_dir=str(work / "registry"), datasets_dir=str(work / "datasets"),
    )
    app = create_app(settings)
    with Session(app.state.engine) as s:
        seed_reference_data(s)
    port = int(os.environ.get("PORT", 8000))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    import httpx
    from syntra_agent.client import Agent, AgentSettings
    from syntra_agent.hardware import GpuInfo
    from syntra_agent.workers import AgentConfig, WorkerManager

    url = f"http://127.0.0.1:{port}"
    logins = {}

    async def make_user(name: str) -> httpx.AsyncClient:
        pw = f"demo-{name}-pass"  # fixed, so the committed demo-logins.txt stays valid; these accounts exist only in this throwaway demo
        c = httpx.AsyncClient(base_url=url, timeout=30)
        r = await c.post("/auth/signup", json={"email": f"{name}@demo.test", "password": pw, "display_name": name.title()})
        assert r.status_code == 201, r.text
        logins[f"{name}@demo.test"] = pw
        return c

    async def provider(client, name, kind, chip, device, gpus, mem):
        pid = (await client.post("/providers", json={"name": name, "type": kind})).json()["id"]
        token = (await client.post(f"/providers/{pid}/agent-tokens", json={"label": name})).json()["token"]
        cfg = AgentConfig(work_dir=work / f"agent-{pid}")
        mgr = WorkerManager(cfg, None, lambda s, c: ([sys.executable, str(FAKE_WORKER), "--peer", f"Peer{s.worker_id}"], {}))
        agent = Agent(AgentSettings(server=url, token=token, region=name.split()[0].lower(), heartbeat_interval=1.0, simulated=True,
                                    sim_count=gpus, sim_memory_gb=mem / 1024), cfg, detector=lambda: [GpuInfo(0, chip, device, mem * gpus, mem * gpus)], manager=mgr)
        mgr.on_event = agent._on_worker_event
        agent.supports_4bit = True  # the fake swarm accepts 4-bit (QLoRA) jobs too
        asyncio.create_task(agent.run())
        return pid

    maya = await make_user("maya")      # an individual sharing three Mac GPUs
    acme = await make_user("acme")      # a data center sharing NVIDIA GPUs
    await make_user("alice")            # a user who trains and deploys
    await provider(maya, "Maya's home lab", "individual", "Apple M4 Pro", "mps", 3, 6144)
    pid = await provider(acme, "Acme Data Centers", "enterprise", "NVIDIA RTX A6000", "cuda", 3, 49140)
    # (an admin would normally verify the enterprise provider; do it directly so 'trusted only' has something to pick)
    from app.models import Provider
    with Session(app.state.engine) as s:
        prov = s.get(Provider, pid)
        prov.trust_tier = "verified"
        s.add(prov)
        s.commit()
    out = ROOT / "demo-logins.txt"
    out.write_text("\n".join(f"{k} {v}" for k, v in logins.items()) + "\n")
    print(f"demo API on {url}; logins in {out}; ctrl-c to stop", flush=True)
    await task


if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
