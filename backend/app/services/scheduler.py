"""Scheduling against the live pool: eligibility, exclusive leases, FIFO queue with backfill, replacements.

One process owns the database, so a process-wide lock makes "check GPU is free, then lease it" atomic: a training run
and somebody else's deployment can never be given the same GPU.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass

from sqlmodel import Session, select

from app.db import utcnow
from app.events import EventService, EventType
from app.models import BaseModel, Gpu, GpuClass, Job, Lease, Provider
from app.services import planner
from app.services.ledger import record_lease_charge
from app.services.planner import Candidate, ModelSpec, Plan
from app.services.pool import effective_status, has_active_lease

_LOCK = threading.RLock()


@dataclass
class Eligible:
    gpu: Gpu
    provider: Provider
    gpu_class: GpuClass

    def candidate(self) -> Candidate:
        machine = self.gpu.agent_token_id if self.gpu.agent_token_id is not None else -self.gpu.provider_id
        return Candidate(gpu_id=self.gpu.id, free_memory_mb=self.gpu.free_memory_mb,
                         price_per_hour=self.gpu_class.price_per_hour, reliability=self.gpu.reliability, machine=machine)


def weight_bits(job: Job) -> int:
    """16 (float16 base weights) or 4 (4-bit base: QLoRA training, light serving)."""
    return int(json.loads(job.config_json or "{}").get("weight_bits", 16))


def model_spec(base: BaseModel, bits: int = 16) -> ModelSpec:
    return ModelSpec(num_layers=base.num_layers, per_layer_memory_mb=base.per_layer_memory_mb, weight_bits=bits)


def eligible_gpus(session: Session, heartbeat_timeout: float, trusted_only: bool = False,
                  exclude: set[int] | None = None, need_4bit: bool = False) -> list[Eligible]:
    """Idle GPUs: fresh heartbeat, no active lease, not used by someone else, trusted tier if requested,
    and (for 4-bit jobs) an agent whose BloomBee can compress weights."""
    now = utcnow()
    out: list[Eligible] = []
    for gpu in session.exec(select(Gpu)).all():
        if exclude and gpu.id in exclude:
            continue
        if need_4bit and not gpu.supports_4bit:
            continue
        if effective_status(gpu, now, heartbeat_timeout) != "idle" or has_active_lease(session, gpu.id):
            continue
        provider = session.get(Provider, gpu.provider_id)
        if trusted_only and provider.trust_tier != "verified":
            continue
        out.append(Eligible(gpu, provider, session.get(GpuClass, gpu.gpu_class_id)))
    return out


def plan_job(session: Session, job: Job, heartbeat_timeout: float) -> Plan | None:
    base = session.get(BaseModel, job.base_model_id)
    bits = weight_bits(job)
    cands = [e.candidate() for e in eligible_gpus(session, heartbeat_timeout, job.trusted_only, need_4bit=bits == 4)]
    return planner.make_plan(model_spec(base, bits), cands, job.requested_gpus)


def _lease(session: Session, events: EventService, job: Job, gpu: Gpu, start: int, end: int) -> Lease:
    lease = Lease(job_id=job.id, gpu_id=gpu.id, layer_start=start, layer_end=end)
    gpu.status = "leased"
    session.add(lease)
    session.add(gpu)
    session.commit()
    session.refresh(lease)
    events.publish(session, EventType.LEASE_CREATED, job_id=job.id, gpu_id=gpu.id, user_id=job.owner_id,
                   payload={"layer_start": start, "layer_end": end, "gpu_name": gpu.name})
    return lease


def try_start(session: Session, events: EventService, job: Job, heartbeat_timeout: float) -> list[Lease] | None:
    """Plan and lease atomically. Returns the leases, or None if the pool cannot run the job right now."""
    with _LOCK:
        plan = plan_job(session, job, heartbeat_timeout)
        if plan is None:
            return None
        leases = [_lease(session, events, job, session.get(Gpu, st.gpu_id), st.layer_start, st.layer_end)
                  for st in plan.stages]
        job.status = "starting"
        job.updated_at = utcnow()
        session.add(job)
        session.commit()
        return leases


def queue_position(session: Session, job: Job) -> int:
    """1 = next in line."""
    queued = session.exec(select(Job).where(Job.status == "queued").order_by(Job.created_at, Job.id)).all()
    return next((i + 1 for i, j in enumerate(queued) if j.id == job.id), 0)


def submit(session: Session, events: EventService, job: Job, heartbeat_timeout: float) -> list[Lease] | None:
    """Create-time entry point: start immediately if possible, otherwise queue and announce the position."""
    leases = try_start(session, events, job, heartbeat_timeout)
    if leases is None:
        job.status = "queued"
        session.add(job)
        session.commit()
        events.publish(session, EventType.JOB_QUEUED, job_id=job.id, user_id=job.owner_id,
                       payload={"position": queue_position(session, job)})
    return leases


def dispatch_queue(session: Session, events: EventService, heartbeat_timeout: float) -> list[tuple[Job, list[Lease]]]:
    """Start every queued job that fits now, oldest first (a small later job may backfill a gap)."""
    started: list[tuple[Job, list[Lease]]] = []
    for job in session.exec(select(Job).where(Job.status == "queued").order_by(Job.created_at, Job.id)).all():
        leases = try_start(session, events, job, heartbeat_timeout)
        if leases is not None:
            started.append((job, leases))
    for job in session.exec(select(Job).where(Job.status == "queued").order_by(Job.created_at, Job.id)).all():
        events.publish(session, EventType.JOB_QUEUED, job_id=job.id, user_id=job.owner_id,
                       payload={"position": queue_position(session, job)})
    return started


def release_lease(session: Session, events: EventService, lease: Lease, reason: str, charge: bool = True) -> None:
    with _LOCK:
        if lease.ended_at is not None:
            return
        lease.ended_at = utcnow()
        lease.end_reason = reason
        session.add(lease)
        gpu = session.get(Gpu, lease.gpu_id)
        if gpu.status == "leased" and not has_active_lease_excluding(session, gpu.id, lease.id):
            gpu.status = "idle"
            session.add(gpu)
        session.commit()
        session.refresh(lease)
    if charge:
        record_lease_charge(session, events, lease)


def has_active_lease_excluding(session: Session, gpu_id: int, lease_id: int) -> bool:
    return session.exec(
        select(Lease).where(Lease.gpu_id == gpu_id, Lease.ended_at == None, Lease.id != lease_id)  # noqa: E711
    ).first() is not None


def release_job(session: Session, events: EventService, job: Job, reason: str) -> int:
    leases = session.exec(select(Lease).where(Lease.job_id == job.id, Lease.ended_at == None)).all()  # noqa: E711
    for lease in leases:
        release_lease(session, events, lease, reason)
    return len(leases)


def lease_replacement(session: Session, events: EventService, job: Job, dead: Lease, heartbeat_timeout: float,
                      reason: str = "worker_down") -> Lease | None:
    """End the dead worker's lease and lease another idle GPU for the same layer range (never a preloaded spare).
    Returns None when no idle GPU fits; the caller retries (the dead lease stays ended)."""
    with _LOCK:
        if dead.ended_at is None:
            release_lease(session, events, dead, reason)
        base = session.get(BaseModel, job.base_model_id)
        layers = dead.layer_end - dead.layer_start
        # the GPU whose worker just failed is never reused for its own replacement
        bits = weight_bits(job)
        cands = [e.candidate() for e in eligible_gpus(session, heartbeat_timeout, job.trusted_only, exclude={dead.gpu_id},
                                                      need_4bit=bits == 4)]
        best = planner.replacement_candidate(model_spec(base, bits), cands, layers)
        if best is None:
            return None
        gpu = session.get(Gpu, best.gpu_id)
        lease = _lease(session, events, job, gpu, dead.layer_start, dead.layer_end)
    cls = session.get(GpuClass, gpu.gpu_class_id)
    events.publish(
        session, EventType.WORKER_REPLACEMENT_STARTING, job_id=job.id, gpu_id=gpu.id, user_id=job.owner_id,
        payload={"layer_start": dead.layer_start, "layer_end": dead.layer_end, "gpu_name": gpu.name,
                 "gpu_device": gpu.device, "gpu_class": cls.name, "gpu_memory_mb": gpu.total_memory_mb},
    )
    return lease
