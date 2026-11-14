"""FastAPI application factory."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session

from app.agent_hub import AgentHub
from app.relay_hub import RelayHub
from app.api import (agent, agent_ws, api_keys, auth, deployments, diloco, events, files, install, jobs, ledger, pool, provider_app,
                     providers, sessions)
from app.config import Settings
from app.db import init_db, make_engine, utcnow
from app.deps import LoginLimiter
from app.events import EventService
from app.services.orchestrator import Orchestrator
from app.services.pool import sweep_offline

log = logging.getLogger("syntra")


def _sweep_once(app: FastAPI) -> None:
    with Session(app.state.engine) as session:
        sweep_offline(session, app.state.events, utcnow(), app.state.settings.heartbeat_timeout_seconds)


async def _sweeper(app: FastAPI) -> None:
    interval = app.state.settings.sweep_interval_seconds
    while True:
        await asyncio.sleep(interval)
        try:
            await run_in_threadpool(_sweep_once, app)
        except Exception:  # keep sweeping even if one pass fails
            log.exception("offline sweep failed")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(_sweeper(app)) if settings.run_background_tasks else None
        if settings.run_orchestrator:
            await app.state.orchestrator.start()
        try:
            yield
        finally:
            if task:
                task.cancel()
            await app.state.orchestrator.stop()

    app = FastAPI(title="Syntra", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = make_engine(settings.database_url)
    init_db(app.state.engine)
    app.state.events = EventService()
    app.state.agent_hub = AgentHub()
    app.state.relay_hub = RelayHub(app.state.agent_hub.send_bytes, app.state.agent_hub.send_json, settings.dht_port,
                                   settings.local_ports)
    app.state.orchestrator = Orchestrator(app)
    app.state.login_limiter = LoginLimiter(settings.login_max_failures, settings.login_window_seconds)

    app.add_middleware(
        CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )
    for module in (auth, providers, agent, agent_ws, install, pool, ledger, jobs, deployments, files, events, provider_app, sessions, api_keys, diloco):
        app.include_router(module.router)

    @app.get("/health", tags=["meta"])
    def health():
        return {"status": "ok"}

    return app


_default_app: FastAPI | None = None


def __getattr__(name: str):
    """`uvicorn app.main:app` gets a lazily created app, so importing this module (e.g. in tests)
    never opens the real database."""
    global _default_app
    if name == "app":
        if _default_app is None:
            _default_app = create_app()
        return _default_app
    raise AttributeError(name)
