"""Runtime settings, read from the environment with safe defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    database_url: str = f"sqlite:///{DATA_DIR / 'syntra.db'}"
    session_cookie: str = "syntra_session"
    session_ttl_seconds: int = 7 * 24 * 3600
    cookie_secure: bool = False  # set SYNTRA_COOKIE_SECURE=1 behind https
    cors_origins: tuple = ("http://localhost:5173", "http://127.0.0.1:5173")
    min_password_length: int = 10
    login_max_failures: int = 5
    login_window_seconds: int = 300
    heartbeat_timeout_seconds: float = 8.0
    sweep_interval_seconds: float = 2.0
    run_background_tasks: bool = True
    run_orchestrator: bool = True
    # --- running real BloomBee jobs -------------------------------------------------------------------
    bloombee_python: str = ""  # interpreter with torch + hivemind + BloomBee (runs the swarm runner); "" = this one
    bloombee_dir: str = "~/Desktop/Repos/BloomBee"
    models_dir: str = str(DATA_DIR / "models_cache")  # local base-model files: <models_dir>/<slug>
    jobs_dir: str = str(DATA_DIR / "jobs")
    registry_dir: str = str(DATA_DIR / "registry")  # fine-tuned adapters / merged models
    datasets_dir: str = str(DATA_DIR / "datasets")
    dht_port: int = 0  # 0 = random; set a fixed port (and forward it) when agents run on other machines
    web_dir: str = str(BACKEND_DIR.parent / "frontend" / "dist")  # built website served at / (empty = API only)
    public_url: str = ""  # the address providers use to reach this server (e.g. a temporary tunnel); shown in the join command
    local_ports: tuple = (31010, 31019)  # worker ports of agents running on this host; relayed agents are given access to them
    runner_argv: tuple = ()  # tests: replace the real runner with a fake command
    merge_argv: tuple = ()  # tests: replace examples/merge_lora.py
    runner_spec_extra: dict = field(default_factory=dict)  # tests: extra keys merged into the runner spec
    ready_timeout: float = 900.0
    session_idle_seconds: float = 600.0  # an RL session with no call for this long is saved and closed (GPUs released)
    session_call_timeout: float = 600.0  # one sampling / training call
    orchestrator_interval: float = 1.0

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.database_url = os.environ.get("SYNTRA_DB", s.database_url)
        s.cookie_secure = _env_bool("SYNTRA_COOKIE_SECURE", s.cookie_secure)
        s.bloombee_python = os.environ.get("SYNTRA_BLOOMBEE_PYTHON", s.bloombee_python)
        s.bloombee_dir = os.environ.get("SYNTRA_BLOOMBEE_DIR", s.bloombee_dir)
        s.models_dir = os.environ.get("SYNTRA_MODELS_DIR", s.models_dir)
        s.dht_port = int(os.environ.get("SYNTRA_DHT_PORT", s.dht_port))
        s.public_url = os.environ.get("SYNTRA_PUBLIC_URL", s.public_url).rstrip("/")
        s.web_dir = os.environ.get("SYNTRA_WEB_DIR", s.web_dir)
        local = os.environ.get("SYNTRA_LOCAL_PORTS")
        if local:
            lo, _, hi = local.partition("-")
            s.local_ports = (int(lo), int(hi or lo))
        data = os.environ.get("SYNTRA_DATA_DIR")  # one folder for jobs, registry and uploaded datasets
        if data:
            s.jobs_dir, s.registry_dir, s.datasets_dir = f"{data}/jobs", f"{data}/registry", f"{data}/datasets"
        origins = os.environ.get("SYNTRA_CORS_ORIGINS")
        if origins:
            s.cors_origins = tuple(o.strip() for o in origins.split(",") if o.strip())
        return s
