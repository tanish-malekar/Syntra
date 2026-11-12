"""Simulated billing: one lease -> one provider credit and one consumer cost, priced by GPU class."""
from __future__ import annotations

from sqlmodel import Session

from app.events import EventService, EventType
from app.models import Gpu, GpuClass, Job, Lease, LedgerEntry, Provider


def lease_gpu_seconds(lease: Lease) -> float:
    if lease.ended_at is None:
        raise ValueError("lease has not ended")
    return max(0.0, (lease.ended_at - lease.started_at).total_seconds())


def record_lease_charge(
    session: Session, events: EventService, lease: Lease, gpu_seconds: float | None = None
) -> tuple[LedgerEntry, LedgerEntry]:
    """Create the credit (provider owner) and cost (job owner) entries for a finished lease."""
    gpu = session.get(Gpu, lease.gpu_id)
    gpu_class = session.get(GpuClass, gpu.gpu_class_id)
    provider = session.get(Provider, gpu.provider_id)
    job = session.get(Job, lease.job_id)
    seconds = lease_gpu_seconds(lease) if gpu_seconds is None else float(gpu_seconds)
    rate = gpu_class.price_per_hour
    amount = round(seconds / 3600.0 * rate, 6)
    common = dict(
        job_id=job.id, lease_id=lease.id, gpu_id=gpu.id, gpu_class_id=gpu_class.id,
        gpu_seconds=seconds, rate_per_hour=rate, amount=amount,
    )
    credit = LedgerEntry(kind="credit", user_id=provider.owner_id, provider_id=provider.id, **common)
    cost = LedgerEntry(kind="cost", user_id=job.owner_id, **common)
    session.add(credit)
    session.add(cost)
    session.commit()
    session.refresh(credit)
    session.refresh(cost)
    events.publish(
        session, EventType.LEDGER_CREDIT, job_id=job.id, gpu_id=gpu.id, user_id=provider.owner_id,
        payload={"provider_id": provider.id, "amount": amount, "gpu_seconds": seconds},
    )
    return credit, cost
