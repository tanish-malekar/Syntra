"""Endpoints for the provider desktop app: one-time pairing with email + password, then usage and earnings by agent token.

The app never keeps a browser session: pairing returns an agent token for this machine (stored in the macOS Keychain), and
every later call authenticates with that token, exactly like the agent itself.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.db import get_session
from app.deps import AgentContext, current_agent
from app.models import AgentToken, Gpu, LedgerEntry, Provider, User
from app.security import new_token, token_hash, verify_password
from app.services.usage import gpu_usage, summarize

router = APIRouter(tags=["provider-app"])


class PairIn(BaseModel):
    email: str
    password: str
    machine_id: str = Field(min_length=8, max_length=128)  # stable per machine (macOS hardware UUID)
    machine_name: str = Field(default="", max_length=80)
    provider_type: str = "individual"


class PairOut(BaseModel):
    token: str
    token_id: int
    provider_id: int
    provider_name: str
    email: str


@router.post("/app/pair", response_model=PairOut, status_code=201)
def pair(body: PairIn, request: Request, session: Session = Depends(get_session)):
    """Sign in once from the app. Re-pairing the same machine revokes its earlier tokens (lost Keychain, reinstall) instead of
    piling up live tokens."""
    limiter = request.app.state.login_limiter
    email = body.email.strip().lower()
    key = (email, request.client.host if request.client else "unknown")
    if limiter.blocked(key):
        raise HTTPException(429, "Too many failed attempts. Try again in a few minutes.", headers={"Retry-After": "300"})
    user = session.exec(select(User).where(User.email == email)).first()
    if not verify_password(user.password_hash if user else None, body.password):
        limiter.record_failure(key)
        raise HTTPException(401, "Invalid email or password")
    limiter.reset(key)
    if body.provider_type not in ("individual", "enterprise"):
        raise HTTPException(422, "provider_type must be individual or enterprise")

    provider = session.exec(select(Provider).where(Provider.owner_id == user.id).order_by(Provider.id)).first()
    if provider is None:
        name = (user.display_name or email.split("@")[0]).strip()[:60]
        provider = Provider(owner_id=user.id, name=f"{name}'s GPUs", type=body.provider_type)
        session.add(provider)
        session.commit()
        session.refresh(provider)
    for old in session.exec(select(AgentToken).where(AgentToken.provider_id == provider.id,
                                                     AgentToken.machine_id == body.machine_id, AgentToken.revoked == False)).all():  # noqa: E712
        old.revoked = True
        session.add(old)
    plaintext = new_token("syn_agent_")
    label = body.machine_name.strip() or "Mac"
    row = AgentToken(provider_id=provider.id, token_hash=token_hash(plaintext), label=label, machine_id=body.machine_id)
    session.add(row)
    session.commit()
    session.refresh(row)
    return PairOut(token=plaintext, token_id=row.id, provider_id=provider.id, provider_name=provider.name, email=user.email)


@router.get("/agent/usage")
def machine_usage(ctx: AgentContext = Depends(current_agent), session: Session = Depends(get_session)):
    """Usage and earnings for this machine's GPUs, plus the provider owner's account-wide earnings."""
    gpus = session.exec(select(Gpu).where(Gpu.agent_token_id == ctx.token.id).order_by(Gpu.device_index, Gpu.id)).all()
    per_gpu = [gpu_usage(session, g) for g in gpus]
    account = session.exec(select(LedgerEntry).where(LedgerEntry.user_id == ctx.provider.owner_id,
                                                     LedgerEntry.kind == "credit")).all()
    return {"provider_name": ctx.provider.name, "machine": ctx.token.label, "gpus": per_gpu, "machine_total": summarize(per_gpu),
            "account_earned_settled": round(sum(c.amount for c in account), 6)}


@router.post("/agent/unpair", status_code=204)
def unpair(ctx: AgentContext = Depends(current_agent), session: Session = Depends(get_session)):
    """Sign out from the app: this machine's token stops working at once."""
    ctx.token.revoked = True
    session.add(ctx.token)
    session.commit()
