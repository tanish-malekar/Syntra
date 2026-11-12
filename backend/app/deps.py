"""Request dependencies: current user (cookie session), admin check, agent (bearer token), login rate limiter."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request
from sqlmodel import Session, select

from app.db import get_session, utcnow
from app.models import AgentToken, AuthSession, Provider, User
from app.security import token_hash


class LoginLimiter:
    """In-memory failed-login limiter keyed by (email, client ip)."""

    def __init__(self, max_failures: int, window_seconds: int, clock=time.monotonic):
        self.max_failures, self.window, self.clock = max_failures, window_seconds, clock
        self._failures: dict[tuple, deque] = defaultdict(deque)

    def _prune(self, key) -> deque:
        q = self._failures[key]
        while q and self.clock() - q[0] > self.window:
            q.popleft()
        return q

    def blocked(self, key) -> bool:
        return len(self._prune(key)) >= self.max_failures

    def record_failure(self, key) -> None:
        self._prune(key).append(self.clock())

    def reset(self, key) -> None:
        self._failures.pop(key, None)


def user_for_cookie(session: Session, cookie_value: str | None) -> User | None:
    """The user behind a session cookie, or None if missing, unknown, revoked or expired."""
    if not cookie_value:
        return None
    row = session.exec(select(AuthSession).where(AuthSession.token_hash == token_hash(cookie_value))).first()
    if row is None or row.revoked or row.expires_at <= utcnow():
        return None
    return session.get(User, row.user_id)


@dataclass
class AgentContext:
    provider: Provider
    token: AgentToken


def agent_for_bearer(session: Session, authorization: str | None) -> AgentContext | None:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization[7:].strip()
    row = session.exec(select(AgentToken).where(AgentToken.token_hash == token_hash(token))).first()
    if row is None or row.revoked:
        return None
    provider = session.get(Provider, row.provider_id)
    if provider is None:
        return None
    row.last_used_at = utcnow()
    session.add(row)
    session.commit()
    return AgentContext(provider=provider, token=row)


def current_user(request: Request, session: Session = Depends(get_session)) -> User:
    cookie = request.cookies.get(request.app.state.settings.session_cookie)
    user = user_for_cookie(session, cookie)
    if user is None:
        raise HTTPException(status_code=401, detail="Not signed in")
    return user


def require_admin(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")
    return user


def current_agent(request: Request, session: Session = Depends(get_session)) -> AgentContext:
    ctx = agent_for_bearer(session, request.headers.get("authorization"))
    if ctx is None:
        raise HTTPException(status_code=401, detail="Invalid or revoked agent token")
    return ctx
