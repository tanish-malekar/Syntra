"""WebSocket command channel for provider agents: /ws/agent (Authorization: Bearer <agent token>)."""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlmodel import Session

from app.deps import agent_for_bearer

log = logging.getLogger("bloombee_platform.agent_ws")
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
    log.warning("agent connected: token %s (relay=%s)", token_id, ws.query_params.get("relay") == "1")
    await ws.send_json({"type": "hello", "token_id": token_id})
    # Status messages (worker.started / online / exited) are handled in order by a separate task: a listener may itself
    # send a command and wait for its ack, and that ack has to be read by this loop. Acks take the fast path.
    inbox: asyncio.Queue = asyncio.Queue()

    async def consume() -> None:
        while True:
            await hub.handle_message(token_id, await inbox.get())

    # Relay frames (binary) also go through a queue: dialling a dead port must not delay command acks.
    relay_hub = app.state.relay_hub
    frames: asyncio.Queue = asyncio.Queue()

    async def carry() -> None:
        while True:
            await relay_hub.frame(token_id, await frames.get())

    consumer = asyncio.create_task(consume())
    carrier = asyncio.create_task(carry())
    try:
        if ws.query_params.get("relay") == "1":
            await relay_hub.attach(token_id)
        while True:
            incoming = await ws.receive()
            if incoming["type"] == "websocket.disconnect":
                break
            if incoming.get("bytes") is not None:
                frames.put_nowait(incoming["bytes"])
                continue
            message = json.loads(incoming["text"])
            kind = message.get("type")
            if kind == "ping":
                await hub.send_json(token_id, {"type": "pong"})
            elif kind == "ack":
                await hub.handle_message(token_id, message)
            else:
                inbox.put_nowait(message)
    except WebSocketDisconnect:
        pass
    finally:
        log.warning("agent disconnected: token %s", token_id)
        consumer.cancel()
        carrier.cancel()
        hub.unregister(token_id, ws)
        await relay_hub.detach(token_id)
