"""Live command channel to provider agents.

Agents connect to /ws/agent (bearer token). The platform sends commands (start_worker, stop_worker, kill_worker)
and awaits an acknowledgement; agents push worker status messages (worker.started / worker.online / worker.exited),
which are delivered to registered listeners (the orchestrator).
"""
from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Awaitable, Callable

from fastapi import WebSocket

log = logging.getLogger("syntra.agent_hub")

Listener = Callable[[int, dict], Awaitable[None] | None]


class AgentNotConnected(Exception):
    pass


class AgentCommandError(Exception):
    pass


class AgentHub:
    def __init__(self) -> None:
        self._conns: dict[int, WebSocket] = {}  # agent token id -> socket
        self._send_locks: dict[int, asyncio.Lock] = {}
        self._pending: dict[str, asyncio.Future] = {}
        self._listeners: list[Listener] = []
        self._ids = itertools.count(1)

    # ---- connection lifecycle -------------------------------------------------------------------
    def register(self, token_id: int, ws: WebSocket) -> None:
        self._conns[token_id] = ws
        self._send_locks[token_id] = asyncio.Lock()

    def unregister(self, token_id: int, ws: WebSocket) -> None:
        if self._conns.get(token_id) is ws:
            self._conns.pop(token_id, None)
            self._send_locks.pop(token_id, None)

    def is_connected(self, token_id: int) -> bool:
        return token_id in self._conns

    def connected_tokens(self) -> list[int]:
        return sorted(self._conns)

    async def wait_connected(self, token_id: int, timeout: float = 10.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while not self.is_connected(token_id):
            if asyncio.get_running_loop().time() > deadline:
                raise AgentNotConnected(f"agent for token {token_id} is not connected")
            await asyncio.sleep(0.05)

    async def send_json(self, token_id: int, message: dict) -> None:
        ws = self._conns.get(token_id)
        if ws is None:
            raise AgentNotConnected(f"agent for token {token_id} is not connected")
        async with self._send_locks[token_id]:
            await ws.send_json(message)

    async def send_bytes(self, token_id: int, data: bytes) -> None:
        ws = self._conns.get(token_id)
        if ws is None:
            raise AgentNotConnected(f"agent for token {token_id} is not connected")
        async with self._send_locks[token_id]:
            await ws.send_bytes(data)

    # ---- commands --------------------------------------------------------------------------------
    async def request(self, token_id: int, command: dict, timeout: float = 30.0) -> dict:
        """Send a command and wait for the agent's ack. Raises AgentCommandError if the agent refuses."""
        ws = self._conns.get(token_id)
        if ws is None:
            raise AgentNotConnected(f"agent for token {token_id} is not connected")
        request_id = f"r{next(self._ids)}"
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            async with self._send_locks[token_id]:
                await ws.send_json({**command, "id": request_id})
            reply = await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError as exc:
            raise AgentCommandError(f"agent did not acknowledge {command.get('cmd')} within {timeout}s") from exc
        finally:
            self._pending.pop(request_id, None)
        if not reply.get("ok", False):
            raise AgentCommandError(reply.get("error", "agent refused the command"))
        return reply

    # ---- messages from agents --------------------------------------------------------------------
    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Listener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    async def handle_message(self, token_id: int, message: dict) -> None:
        if message.get("type") == "ack":
            future = self._pending.get(message.get("id", ""))
            if future is not None and not future.done():
                future.set_result(message)
            return
        for listener in list(self._listeners):
            try:
                result = listener(token_id, message)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                log.exception("agent message listener failed")
