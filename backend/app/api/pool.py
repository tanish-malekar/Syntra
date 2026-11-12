"""The shared GPU pool, GPU classes (fixed price per class) and base models."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.db import get_session, iso, utcnow
from app.deps import current_user, require_admin
from app.models import BaseModel as BaseModelRow
from app.models import Gpu, GpuClass, Provider, User
from app.services.pool import effective_status

router = APIRouter(tags=["pool"])


class GpuOut(BaseModel):
    id: int
    provider_id: int
    provider_name: str
    provider_type: str
    trust_tier: str
    gpu_class: str
    price_per_hour: float
    name: str
    device: str
    device_index: int
    total_memory_mb: int
    free_memory_mb: int
    region: str
    status: str  # idle | leased | busy_external | offline (offline when the heartbeat is stale)
    last_heartbeat: str | None
    reliability: float
    simulated: bool


class GpuClassOut(BaseModel):
    id: int
    name: str
    price_per_hour: float
    device_hint: str


class GpuClassPatch(BaseModel):
    price_per_hour: float = Field(gt=0)


class BaseModelOut(BaseModel):
    id: int
    name: str
    hf_id: str
    family: str
    num_layers: int
    hidden_size: int
    per_layer_memory_mb: int


def gpu_out(session: Session, gpu: Gpu, request: Request) -> GpuOut:
    provider = session.get(Provider, gpu.provider_id)
    gpu_class = session.get(GpuClass, gpu.gpu_class_id)
    timeout = request.app.state.settings.heartbeat_timeout_seconds
    return GpuOut(
        id=gpu.id, provider_id=provider.id, provider_name=provider.name, provider_type=provider.type,
        trust_tier=provider.trust_tier, gpu_class=gpu_class.name, price_per_hour=gpu_class.price_per_hour,
        name=gpu.name, device=gpu.device, device_index=gpu.device_index, total_memory_mb=gpu.total_memory_mb,
        free_memory_mb=gpu.free_memory_mb, region=gpu.region, status=effective_status(gpu, utcnow(), timeout),
        last_heartbeat=iso(gpu.last_heartbeat), reliability=gpu.reliability, simulated=gpu.simulated,
    )


@router.get("/gpus", response_model=list[GpuOut])
def pool(request: Request, _: User = Depends(current_user), session: Session = Depends(get_session)):
    """The whole pool: every shared GPU, visible to any signed-in user."""
    return [gpu_out(session, g, request) for g in session.exec(select(Gpu).order_by(Gpu.id)).all()]


@router.get("/gpus/mine", response_model=list[GpuOut])
def my_gpus(request: Request, user: User = Depends(current_user), session: Session = Depends(get_session)):
    mine = select(Gpu).join(Provider, Provider.id == Gpu.provider_id).where(Provider.owner_id == user.id).order_by(Gpu.id)
    return [gpu_out(session, g, request) for g in session.exec(mine).all()]


@router.get("/gpu-classes", response_model=list[GpuClassOut])
def gpu_classes(_: User = Depends(current_user), session: Session = Depends(get_session)):
    rows = session.exec(select(GpuClass).order_by(GpuClass.id)).all()
    return [GpuClassOut(id=c.id, name=c.name, price_per_hour=c.price_per_hour, device_hint=c.device_hint) for c in rows]


@router.patch("/gpu-classes/{class_id}", response_model=GpuClassOut)
def set_class_price(
    class_id: int, body: GpuClassPatch, _: User = Depends(require_admin), session: Session = Depends(get_session)
):
    row = session.get(GpuClass, class_id)
    if row is None:
        raise HTTPException(404, "GPU class not found")
    row.price_per_hour = body.price_per_hour
    session.add(row)
    session.commit()
    return GpuClassOut(id=row.id, name=row.name, price_per_hour=row.price_per_hour, device_hint=row.device_hint)


@router.get("/base-models", response_model=list[BaseModelOut])
def base_models(_: User = Depends(current_user), session: Session = Depends(get_session)):
    rows = session.exec(select(BaseModelRow).order_by(BaseModelRow.id)).all()
    return [
        BaseModelOut(
            id=m.id, name=m.name, hf_id=m.hf_id, family=m.family, num_layers=m.num_layers,
            hidden_size=m.hidden_size, per_layer_memory_mb=m.per_layer_memory_mb,
        )
        for m in rows
    ]
