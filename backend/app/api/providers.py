"""Provider profiles (individuals and enterprise data centers) and their per-machine agent tokens."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.db import get_session, iso, utcnow
from app.deps import current_user, require_admin
from app.models import PROVIDER_TYPES, TRUST_TIERS, AgentToken, Provider, User
from app.security import new_token, token_hash

router = APIRouter(prefix="/providers", tags=["providers"])


class ProviderIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    type: str = "individual"


class ProviderOut(BaseModel):
    id: int
    name: str
    type: str
    trust_tier: str
    created_at: str


class TrustIn(BaseModel):
    trust_tier: str


class TokenIn(BaseModel):
    label: str = Field(default="", max_length=80)


class TokenOut(BaseModel):
    id: int
    label: str
    created_at: str
    last_used_at: str | None
    revoked: bool


class TokenCreated(BaseModel):
    id: int
    label: str
    token: str  # shown once; only its hash is stored
    created_at: str


def provider_out(p: Provider) -> ProviderOut:
    return ProviderOut(id=p.id, name=p.name, type=p.type, trust_tier=p.trust_tier, created_at=iso(p.created_at))


def token_out(t: AgentToken) -> TokenOut:
    return TokenOut(id=t.id, label=t.label, created_at=iso(t.created_at), last_used_at=iso(t.last_used_at), revoked=t.revoked)


def owned_provider(session: Session, provider_id: int, user: User) -> Provider:
    provider = session.get(Provider, provider_id)
    if provider is None or provider.owner_id != user.id:
        raise HTTPException(404, "Provider not found")  # 404, not 403: do not reveal other users' ids
    return provider


@router.post("", response_model=ProviderOut, status_code=201)
def create_provider(body: ProviderIn, user: User = Depends(current_user), session: Session = Depends(get_session)):
    if body.type not in PROVIDER_TYPES:
        raise HTTPException(422, f"type must be one of {PROVIDER_TYPES}")
    provider = Provider(owner_id=user.id, name=body.name.strip(), type=body.type)
    session.add(provider)
    session.commit()
    session.refresh(provider)
    return provider_out(provider)


@router.get("", response_model=list[ProviderOut])
def my_providers(user: User = Depends(current_user), session: Session = Depends(get_session)):
    rows = session.exec(select(Provider).where(Provider.owner_id == user.id).order_by(Provider.id)).all()
    return [provider_out(p) for p in rows]


@router.patch("/{provider_id}/trust", response_model=ProviderOut)
def set_trust_tier(
    provider_id: int, body: TrustIn, _: User = Depends(require_admin), session: Session = Depends(get_session)
):
    if body.trust_tier not in TRUST_TIERS:
        raise HTTPException(422, f"trust_tier must be one of {TRUST_TIERS}")
    provider = session.get(Provider, provider_id)
    if provider is None:
        raise HTTPException(404, "Provider not found")
    provider.trust_tier = body.trust_tier
    session.add(provider)
    session.commit()
    return provider_out(provider)


@router.post("/{provider_id}/agent-tokens", response_model=TokenCreated, status_code=201)
def create_agent_token(
    provider_id: int, body: TokenIn, user: User = Depends(current_user), session: Session = Depends(get_session)
):
    provider = owned_provider(session, provider_id, user)
    plaintext = new_token("syn_agent_")
    row = AgentToken(provider_id=provider.id, token_hash=token_hash(plaintext), label=body.label.strip())
    session.add(row)
    session.commit()
    session.refresh(row)
    return TokenCreated(id=row.id, label=row.label, token=plaintext, created_at=iso(row.created_at))


@router.get("/{provider_id}/agent-tokens", response_model=list[TokenOut])
def list_agent_tokens(provider_id: int, user: User = Depends(current_user), session: Session = Depends(get_session)):
    provider = owned_provider(session, provider_id, user)
    rows = session.exec(select(AgentToken).where(AgentToken.provider_id == provider.id).order_by(AgentToken.id)).all()
    return [token_out(t) for t in rows]


@router.delete("/{provider_id}/agent-tokens/{token_id}", status_code=204)
def revoke_agent_token(
    provider_id: int, token_id: int, user: User = Depends(current_user), session: Session = Depends(get_session)
):
    provider = owned_provider(session, provider_id, user)
    row = session.get(AgentToken, token_id)
    if row is None or row.provider_id != provider.id:
        raise HTTPException(404, "Token not found")
    row.revoked = True
    session.add(row)
    session.commit()
