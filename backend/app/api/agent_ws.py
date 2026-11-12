"""WebSocket command channel for provider agents: /ws/agent (Authorization: Bearer <agent token>)."""
from __future__ import annotations

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
    try:
        while True:
            message = await ws.receive_json()
            if message.get("type") == "ping":
                await ws.send_json({"type": "pong"})
                continue
            await hub.handle_message(token_id, message)
    except WebSocketDisconnect:
        pass
    finally:
        hub.unregister(token_id, ws)
