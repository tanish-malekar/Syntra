"""Runtime settings, read from the environment with safe defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
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

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.database_url = os.environ.get("SYNTRA_DB", s.database_url)
        s.cookie_secure = _env_bool("SYNTRA_COOKIE_SECURE", s.cookie_secure)
        origins = os.environ.get("SYNTRA_CORS_ORIGINS")
        if origins:
            s.cors_origins = tuple(o.strip() for o in origins.split(",") if o.strip())
        return s
