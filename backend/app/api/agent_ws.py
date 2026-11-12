"""WebSocket command channel for provider agents: /ws/agent (Authorization: Bearer <agent token>)."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlmodel import Session

from app.deps import agent_for_bearer

router = APIRouter(tags=["agent"])


@router.websocket("/ws/agent")
async def agent_channel(ws: WebSocket):
    app = ws.app
    with Session(app.state.engine) as session:
        ctx = agent_for_bearer(session, ws.headers.get("authorization"))
        token_id = ctx.token.id if ctx else None
    if token_id is None:
        await ws.close(code=4401)
        return
    await ws.accept()
    hub = app.state.agent_hub
    hub.register(token_id, ws)
    await ws.send_json({"type": "hello", "token_id": token_id})
    # Status messages (worker.started / online / exited) are handled in order by a separate task: a listener may itself
    # send a command and wait for its ack, and that ack has to be read by this loop. Acks take the fast path.
    inbox: asyncio.Queue = asyncio.Queue()

    async def consume() -> None:
        while True:
            await hub.handle_message(token_id, await inbox.get())

    consumer = asyncio.create_task(consume())
    try:
        while True:
            message = await ws.receive_json()
            kind = message.get("type")
            if kind == "ping":
                await ws.send_json({"type": "pong"})
            elif kind == "ack":
                await hub.handle_message(token_id, message)
            else:
                inbox.put_nowait(message)
    except WebSocketDisconnect:
        pass
    finally:
        consumer.cancel()
        hub.unregister(token_id, ws)
