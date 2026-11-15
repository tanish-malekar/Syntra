"""Endpoints used by the provider agent (authenticated with a bearer agent token, never a user password)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.api.pool import GpuOut, gpu_out
from app.db import get_session, utcnow
from app.deps import AgentContext, current_agent
from app.events import EventType
from app.models import DEVICES, Gpu
from app.services.pool import classify_gpu, get_class, has_active_lease

router = APIRouter(prefix="/agent", tags=["agent"])
REPORTABLE_STATUSES = ("idle", "leased", "busy_external", "offline")


class GpuRegisterIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    device: str
    device_index: int = Field(default=0, ge=0)
    total_memory_mb: int = Field(ge=0)
    free_memory_mb: int | None = Field(default=None, ge=0)
    region: str = Field(default="", max_length=60)
    simulated: bool = False
    supports_4bit: bool = False  # older agents do not send it: their GPUs only run float16 jobs


class HeartbeatIn(BaseModel):
    free_memory_mb: int = Field(ge=0)
    status: str | None = None  # idle | leased | busy_external | offline; omitted = keep (an offline GPU returns to idle)
    detail: str = Field(default="", max_length=200)


class AgentMe(BaseModel):
    provider_id: int
    provider_name: str
    provider_type: str
    trust_tier: str
    token_id: int
    label: str


@router.get("/me", response_model=AgentMe)
def agent_me(ctx: AgentContext = Depends(current_agent)):
    p = ctx.provider
    return AgentMe(
        provider_id=p.id, provider_name=p.name, provider_type=p.type, trust_tier=p.trust_tier,
        token_id=ctx.token.id, label=ctx.token.label,
    )


@router.post("/gpus", response_model=GpuOut)
def register_gpu(
    body: GpuRegisterIn, request: Request, ctx: AgentContext = Depends(current_agent),
    session: Session = Depends(get_session),
):
    """Register (or re-register after a restart) one GPU; it joins the pool as idle."""
    if body.device not in DEVICES:
        raise HTTPException(422, f"device must be one of {DEVICES}")
    gpu_class = get_class(session, classify_gpu(body.name, body.device))
    gpu = session.exec(
        select(Gpu).where(
            Gpu.provider_id == ctx.provider.id, Gpu.name == body.name, Gpu.device_index == body.device_index
        )
    ).first()
    created = gpu is None
    if created:
        gpu = Gpu(provider_id=ctx.provider.id, gpu_class_id=gpu_class.id, name=body.name, device=body.device,
                  device_index=body.device_index)
    gpu.gpu_class_id = gpu_class.id
    gpu.agent_token_id = ctx.token.id
    gpu.total_memory_mb = body.total_memory_mb
    gpu.free_memory_mb = body.total_memory_mb if body.free_memory_mb is None else body.free_memory_mb
    gpu.region = body.region
    gpu.simulated = body.simulated
    gpu.supports_4bit = body.supports_4bit
    gpu.status = "idle"
    gpu.last_heartbeat = utcnow()
    session.add(gpu)
    session.commit()
    session.refresh(gpu)
    request.app.state.events.publish(
        session, EventType.GPU_REGISTERED, gpu_id=gpu.id,
        payload={"name": gpu.name, "device": gpu.device, "gpu_class": gpu_class.name,
                 "total_memory_mb": gpu.total_memory_mb, "provider": ctx.provider.name, "new": created},
    )
    return gpu_out(session, gpu, request)


@router.post("/gpus/{gpu_id}/heartbeat", response_model=GpuOut)
def heartbeat(
    gpu_id: int, body: HeartbeatIn, request: Request, ctx: AgentContext = Depends(current_agent),
    session: Session = Depends(get_session),
):
    gpu = session.get(Gpu, gpu_id)
    if gpu is None or gpu.provider_id != ctx.provider.id:
        raise HTTPException(404, "GPU not found")
    if body.status is not None and body.status not in REPORTABLE_STATUSES:
        raise HTTPException(422, f"status must be one of {REPORTABLE_STATUSES}")
    previous = gpu.status
    if body.status == "offline":  # the agent says goodbye (laptop lid closed): offline now, not after a heartbeat timeout
        gpu.status, gpu.last_heartbeat = "offline", utcnow()
        session.add(gpu)
        session.commit()
        if previous != "offline":
            request.app.state.events.publish(session, EventType.GPU_OFFLINE, gpu_id=gpu.id, payload={"reason": body.detail or "reported_by_agent"})
        return gpu_out(session, gpu, request)
    if body.status is not None:
        gpu.status = body.status
    elif gpu.status == "offline":
        gpu.status = "idle"
    # The lease table, not the agent, says whether a GPU is leased: an 'idle' report never clears a live lease,
    # and a GPU with a lease stays leased unless the owner's own workload takes it over (busy_external).
    if gpu.status != "busy_external":
        gpu.status = "leased" if has_active_lease(session, gpu.id) else ("idle" if gpu.status == "leased" else gpu.status)
    gpu.free_memory_mb = body.free_memory_mb
    gpu.last_heartbeat = utcnow()
    session.add(gpu)
    session.commit()
    events = request.app.state.events
    if gpu.status == "busy_external" and previous != "busy_external":
        events.publish(session, EventType.GPU_BUSY_EXTERNAL, gpu_id=gpu.id, payload={"detail": body.detail})
    # Heartbeats arrive every ~2 s per GPU: stream them to listeners but do not store them.
    events.publish(
        session, EventType.GPU_HEARTBEAT, gpu_id=gpu.id, persist=False,
        payload={"status": gpu.status, "free_memory_mb": gpu.free_memory_mb},
    )
    return gpu_out(session, gpu, request)
