"""How much a provider's GPUs were used and what they earned: settled ledger credits plus a live estimate for running leases.

Settled numbers come from the ledger (one credit per finished lease). A lease that is still running has no ledger line yet,
so its time and earnings are estimated from its start time and the GPU class's current price.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlmodel import Session, select

from app.db import iso, utcnow
from app.models import Gpu, GpuClass, Job, Lease, LedgerEntry


def _aware(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def gpu_usage(session: Session, gpu: Gpu, now: datetime | None = None, days: int = 30) -> dict:
    now = now or utcnow()
    cls = session.get(GpuClass, gpu.gpu_class_id)
    credits = session.exec(select(LedgerEntry).where(LedgerEntry.gpu_id == gpu.id, LedgerEntry.kind == "credit")).all()
    settled_seconds = sum(c.gpu_seconds for c in credits)
    settled_amount = sum(c.amount for c in credits)

    running = session.exec(select(Lease).where(Lease.gpu_id == gpu.id, Lease.ended_at == None)).all()  # noqa: E711
    running_seconds = sum(max(0.0, (now - _aware(l.started_at)).total_seconds()) for l in running)
    running_amount = running_seconds / 3600.0 * cls.price_per_hour

    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_seconds = sum(c.gpu_seconds for c in credits if _aware(c.created_at) >= start_of_today)
    today_seconds += sum(max(0.0, (now - max(_aware(l.started_at), start_of_today)).total_seconds()) for l in running)
    today_amount = sum(c.amount for c in credits if _aware(c.created_at) >= start_of_today)
    today_amount += sum(max(0.0, (now - max(_aware(l.started_at), start_of_today)).total_seconds()) for l in running) / 3600.0 * cls.price_per_hour

    by_day: dict[str, dict] = {}
    first_day = (start_of_today - timedelta(days=days - 1)).date()
    for c in credits:  # a lease is attributed to the day it ended (when it was billed)
        day = _aware(c.created_at).date()
        if day >= first_day:
            d = by_day.setdefault(day.isoformat(), {"day": day.isoformat(), "gpu_seconds": 0.0, "amount": 0.0})
            d["gpu_seconds"] += c.gpu_seconds
            d["amount"] += c.amount

    current = None
    if running:
        lease = max(running, key=lambda l: _aware(l.started_at))
        job = session.get(Job, lease.job_id)
        # what the provider may see about the work on their GPU: its kind and how long it has run, never who runs it
        current = {"job_kind": job.kind if job else "", "since": iso(lease.started_at),
                   "layers": [lease.layer_start, lease.layer_end]}
    return {
        "gpu_id": gpu.id, "name": gpu.name, "device": gpu.device, "gpu_class": cls.name, "price_per_hour": cls.price_per_hour,
        "status": gpu.status, "in_use": bool(running), "current": current,
        "gpu_seconds_total": round(settled_seconds + running_seconds, 1), "gpu_seconds_running": round(running_seconds, 1),
        "earned_settled": round(settled_amount, 6), "earned_running_estimate": round(running_amount, 6),
        "earned_total": round(settled_amount + running_amount, 6),
        "gpu_seconds_today": round(today_seconds, 1), "earned_today": round(today_amount, 6),
        "jobs_served": len({c.job_id for c in credits} | {l.job_id for l in running}),
        "last_days": sorted(by_day.values(), key=lambda d: d["day"]),
    }


def summarize(per_gpu: list[dict]) -> dict:
    keys = ("gpu_seconds_total", "gpu_seconds_running", "earned_settled", "earned_running_estimate", "earned_total",
            "gpu_seconds_today", "earned_today", "jobs_served")
    total = {k: round(sum(g[k] for g in per_gpu), 6) for k in keys}
    total["in_use"] = any(g["in_use"] for g in per_gpu)
    return total
