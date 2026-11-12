"""A user's simulated earnings (as a provider) and costs (as a consumer). Nobody sees anyone else's ledger."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session, select

from app.db import get_session, iso
from app.deps import current_user
from app.models import GpuClass, LedgerEntry, User

router = APIRouter(prefix="/ledger", tags=["ledger"])


class LedgerLine(BaseModel):
    id: int
    created_at: str
    gpu_class: str
    gpu_seconds: float
    rate_per_hour: float
    amount: float
    job_id: int | None
    provider_id: int | None


class LedgerOut(BaseModel):
    total: float
    gpu_seconds: float
    entries: list[LedgerLine]


def _ledger(session: Session, user: User, kind: str) -> LedgerOut:
    rows = session.exec(
        select(LedgerEntry).where(LedgerEntry.user_id == user.id, LedgerEntry.kind == kind).order_by(LedgerEntry.id.desc())
    ).all()
    lines = []
    for r in rows:
        cls = session.get(GpuClass, r.gpu_class_id) if r.gpu_class_id else None
        lines.append(
            LedgerLine(
                id=r.id, created_at=iso(r.created_at), gpu_class=cls.name if cls else "", gpu_seconds=r.gpu_seconds,
                rate_per_hour=r.rate_per_hour, amount=r.amount, job_id=r.job_id, provider_id=r.provider_id,
            )
        )
    return LedgerOut(total=round(sum(r.amount for r in rows), 6), gpu_seconds=sum(r.gpu_seconds for r in rows), entries=lines)


@router.get("/earnings", response_model=LedgerOut)
def earnings(user: User = Depends(current_user), session: Session = Depends(get_session)):
    return _ledger(session, user, "credit")


@router.get("/costs", response_model=LedgerOut)
def costs(user: User = Depends(current_user), session: Session = Depends(get_session)):
    return _ledger(session, user, "cost")
