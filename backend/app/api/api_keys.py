"""Personal API keys for the `bloombee_sdk` SDK. The plaintext is shown once; only its hash is stored."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.db import get_session, iso
from app.deps import API_KEY_PREFIX, current_user
from app.models import ApiKey, User
from app.security import new_token, token_hash

router = APIRouter(prefix="/api-keys", tags=["api-keys"])


class KeyIn(BaseModel):
    label: str = Field(default="", max_length=60)


def key_out(k: ApiKey) -> dict:
    return {"id": k.id, "label": k.label, "created_at": iso(k.created_at), "last_used_at": iso(k.last_used_at), "revoked": k.revoked}


@router.post("", status_code=201)
def create_key(body: KeyIn, user: User = Depends(current_user), session: Session = Depends(get_session)):
    plaintext = new_token(API_KEY_PREFIX)
    row = ApiKey(user_id=user.id, key_hash=token_hash(plaintext), label=body.label.strip())
    session.add(row)
    session.commit()
    session.refresh(row)
    return {**key_out(row), "key": plaintext}


@router.get("")
def list_keys(user: User = Depends(current_user), session: Session = Depends(get_session)):
    return [key_out(k) for k in session.exec(select(ApiKey).where(ApiKey.user_id == user.id).order_by(ApiKey.id)).all()]


@router.delete("/{key_id}", status_code=204)
def revoke_key(key_id: int, user: User = Depends(current_user), session: Session = Depends(get_session)):
    row = session.get(ApiKey, key_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(404, "API key not found")
    row.revoked = True
    session.add(row)
    session.commit()
