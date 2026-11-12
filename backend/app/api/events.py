"""Event history (REST) and the live stream (WebSocket /ws/events)."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlmodel import Session, select

from app.db import get_session
from app.deps import agent_for_bearer, current_user, user_for_cookie
from app.events import event_to_dict
from app.models import Event, User

router = APIRouter(tags=["events"])


@router.get("/events")
def event_history(
    job_id: int | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
):
    """Stored events that belong to the caller (newest last). Pool-level gpu.* events are not user-owned."""
    query = select(Event).where(Event.user_id == user.id)
    if job_id is not None:
        query = query.where(Event.job_id == job_id)
    rows = session.exec(query.order_by(Event.id.desc()).limit(limit)).all()
    return [event_to_dict(e) for e in reversed(rows)]


@router.websocket("/ws/events")
async def events_ws(ws: WebSocket):
    """Users (session cookie) get their own job events plus pool-level gpu.* events;
    agents (Authorization: Bearer <agent token>) get pool-level events only."""
    app = ws.app
    settings = app.state.settings
    user_id, is_agent = None, False
    with Session(app.state.engine) as session:
        user = user_for_cookie(session, ws.cookies.get(settings.session_cookie))
        if user is not None:
            user_id = user.id
        elif agent_for_bearer(session, ws.headers.get("authorization")) is not None:
            is_agent = True
    if user_id is None and not is_agent:
        await ws.close(code=4401)  # rejected before the handshake completes
        return
    await ws.accept()
    sub = app.state.events.bus.subscribe(user_id, is_agent)
    send_lock = asyncio.Lock()

    async def pump() -> None:
        while True:
            event = await sub.queue.get()
            async with send_lock:
                await ws.send_json(event)

    async with send_lock:
        await ws.send_json({"type": "hello", "role": "agent" if is_agent else "user"})
    pump_task = asyncio.create_task(pump())
    try:
        while True:
            text = await ws.receive_text()
            if text == "ping":
                async with send_lock:
                    await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        pump_task.cancel()
        app.state.events.bus.unsubscribe(sub)
