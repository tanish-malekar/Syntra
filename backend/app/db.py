"""Database engine, session dependency and a UTC clock helper."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import Request
from sqlalchemy import event
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine


def utcnow() -> datetime:
    """Timezone-aware UTC now (SQLModel's datetime column requires aware values)."""
    return datetime.now(timezone.utc)


def iso(dt: datetime | None) -> str | None:
    """ISO 8601 with a trailing Z; naive values are treated as UTC."""
    if dt is None:
        return None
    dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def make_engine(url: str):
    kwargs: dict = {"connect_args": {"check_same_thread": False, "timeout": 30}}
    in_memory = url in ("sqlite://", "sqlite:///:memory:")
    if in_memory:
        kwargs["poolclass"] = StaticPool  # one shared in-memory database: only for simple single-threaded tests
    elif url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite") and not in_memory:
        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_conn, _):  # concurrent readers + one writer without "database is locked"
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()
    return engine


def init_db(engine) -> None:
    from app import models  # noqa: F401  (register tables)

    SQLModel.metadata.create_all(engine)


def get_session(request: Request):
    with Session(request.app.state.engine) as session:
        yield session
