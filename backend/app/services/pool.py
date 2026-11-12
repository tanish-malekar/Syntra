"""GPU pool helpers: class mapping, heartbeat freshness, offline sweep."""
from __future__ import annotations

from datetime import datetime, timedelta

from sqlmodel import Session, select

from app.events import EventService, EventType
from app.models import Gpu, GpuClass, Lease, Provider

APPLE = "Apple M-series"
WORKSTATION = "NVIDIA workstation (RTX A6000)"
DATACENTER = "NVIDIA data center"

# Keywords, matched case-insensitively against the reported chip name.
_DATACENTER_WORDS = ("a100", "h100", "h200", "b200", "v100", "l40", "l4 ", "a10 ", "t4", "gh200")
_WORKSTATION_WORDS = ("rtx", "geforce", "quadro", "titan", "a6000", "a5000", "a4000")


def classify_gpu(name: str, device: str) -> str:
    """Map a reported GPU to one of the seeded GPU class names."""
    lowered = f" {name.lower()} "
    if device == "mps" or "apple" in lowered:
        return APPLE
    if any(w in lowered for w in _DATACENTER_WORDS):
        return DATACENTER
    if any(w in lowered for w in _WORKSTATION_WORDS):
        return WORKSTATION
    return WORKSTATION if device == "cuda" else APPLE


def get_class(session: Session, name: str) -> GpuClass:
    cls = session.exec(select(GpuClass).where(GpuClass.name == name)).first()
    if cls is None:
        raise LookupError(f"GPU class {name!r} is not seeded; run `python -m app.seed`")
    return cls


def effective_status(gpu: Gpu, now: datetime, timeout_seconds: float) -> str:
    """Stored status, except a stale heartbeat always reads as offline."""
    if gpu.last_heartbeat is None or now - gpu.last_heartbeat > timedelta(seconds=timeout_seconds):
        return "offline"
    return gpu.status


def sweep_offline(session: Session, events: EventService, now: datetime, timeout_seconds: float) -> list[int]:
    """Mark GPUs with a stale heartbeat offline and publish gpu.offline once per transition."""
    changed: list[int] = []
    for gpu in session.exec(select(Gpu).where(Gpu.status != "offline")).all():
        if effective_status(gpu, now, timeout_seconds) == "offline":
            gpu.status = "offline"
            session.add(gpu)
            session.commit()
            events.publish(session, EventType.GPU_OFFLINE, gpu_id=gpu.id, payload={"reason": "heartbeat_timeout"})
            changed.append(gpu.id)
    return changed


def provider_name(session: Session, provider_id: int) -> str:
    provider = session.get(Provider, provider_id)
    return provider.name if provider else ""


def has_active_lease(session: Session, gpu_id: int) -> bool:
    """A GPU is leased while any lease on it has not ended (the lease table is the source of truth)."""
    return session.exec(select(Lease).where(Lease.gpu_id == gpu_id, Lease.ended_at == None)).first() is not None  # noqa: E711
