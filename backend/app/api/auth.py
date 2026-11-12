"""Email + password accounts with server-side sessions in an httpOnly cookie."""
from __future__ import annotations

import re
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlmodel import Session, select

from app.db import get_session, utcnow
from app.deps import current_user
from app.models import AuthSession, User
from app.security import hash_password, needs_rehash, new_token, token_hash, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class SignupIn(BaseModel):
    email: str
    password: str
    display_name: str = ""


class LoginIn(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: int
    email: str
    display_name: str
    is_admin: bool


def user_out(user: User) -> UserOut:
    return UserOut(id=user.id, email=user.email, display_name=user.display_name, is_admin=user.is_admin)


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _start_session(request: Request, response: Response, session: Session, user: User) -> None:
    settings = request.app.state.settings
    token = new_token()
    session.add(
        AuthSession(
            token_hash=token_hash(token),
            user_id=user.id,
            expires_at=utcnow() + timedelta(seconds=settings.session_ttl_seconds),
        )
    )
    session.commit()
    response.set_cookie(
        settings.session_cookie, token, httponly=True, samesite="lax", secure=settings.cookie_secure,
        max_age=settings.session_ttl_seconds, path="/",
    )


@router.post("/signup", response_model=UserOut, status_code=201)
def signup(body: SignupIn, request: Request, response: Response, session: Session = Depends(get_session)):
    settings = request.app.state.settings
    email = _normalize_email(body.email)
    if not EMAIL_RE.match(email) or len(email) > 254:
        raise HTTPException(422, "Enter a valid email address")
    if len(body.password) < settings.min_password_length:
        raise HTTPException(422, f"Password must be at least {settings.min_password_length} characters")
    if len(body.password) > 256:
        raise HTTPException(422, "Password is too long")
    if session.exec(select(User).where(User.email == email)).first() is not None:
        raise HTTPException(409, "An account with this email already exists")
    user = User(email=email, password_hash=hash_password(body.password), display_name=body.display_name.strip()[:80])
    session.add(user)
    session.commit()
    session.refresh(user)
    _start_session(request, response, session, user)
    return user_out(user)


@router.post("/login", response_model=UserOut)
def login(body: LoginIn, request: Request, response: Response, session: Session = Depends(get_session)):
    limiter = request.app.state.login_limiter
    email = _normalize_email(body.email)
    key = (email, request.client.host if request.client else "unknown")
    if limiter.blocked(key):
        raise HTTPException(429, "Too many failed attempts. Try again in a few minutes.", headers={"Retry-After": "300"})
    user = session.exec(select(User).where(User.email == email)).first()
    if not verify_password(user.password_hash if user else None, body.password):
        limiter.record_failure(key)
        raise HTTPException(401, "Invalid email or password")
    limiter.reset(key)
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(body.password)
        session.add(user)
        session.commit()
    _start_session(request, response, session, user)
    return user_out(user)


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response, session: Session = Depends(get_session)):
    settings = request.app.state.settings
    cookie = request.cookies.get(settings.session_cookie)
    if cookie:
        row = session.exec(select(AuthSession).where(AuthSession.token_hash == token_hash(cookie))).first()
        if row is not None:
            row.revoked = True
            session.add(row)
            session.commit()
    response.delete_cookie(settings.session_cookie, path="/")
    response.status_code = 204
    return response


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user_out(user)
