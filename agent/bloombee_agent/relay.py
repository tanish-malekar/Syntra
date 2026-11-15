"""TCP over one WebSocket, so machines behind home routers can join without any inbound connection or SSH.

Both ends run a Relay. It listens on some loopback ports; every connection accepted there becomes a *stream* that is carried
over the WebSocket and re-opened on the other end as a connection to 127.0.0.1:<same port>. Frames are binary:

    1 byte kind | 4 byte stream id | payload      kinds: OPEN (payload = port), DATA, CLOSE

This file is the single source: the backend imports it through the symlink backend/app/relay.py. Standard library only.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import struct
from typing import Awaitable, Callable, Iterable, Optional

log = logging.getLogger("bloombee_platform.relay")

OPEN, DATA, CLOSE = 1, 2, 3
HEADER = struct.Struct(">BI")
CHUNK = 64 * 1024
DIAL_TIMEOUT = 5.0

Send = Callable[[bytes], Awaitable[None]]
Dial = Callable[[int], tuple[str, int]]


class Relay:
    def __init__(self, send: Send, parity: int, dial: Optional[Dial] = None, bind_host: str = "127.0.0.1"):
        """parity: 0 on one end and 1 on the other, so the two ends never pick the same stream id."""
        self._send = send
        self._ids = itertools.count(parity, 2)
        self._dial = dial or (lambda port: ("127.0.0.1", port))
        self._bind_host = bind_host
        self._servers: dict[int, asyncio.AbstractServer] = {}
        self._streams: dict[int, asyncio.StreamWriter] = {}
        self._tasks: set[asyncio.Task] = set()

    @property
    def listening(self) -> set[int]:
        return set(self._servers)

    # ---- listening side -------------------------------------------------------------------------------
    async def listen(self, ports: Iterable[int]) -> None:
        """Listen on exactly these ports (closing listeners that are no longer wanted). A port already in use is skipped."""
        wanted = set(ports)
        for port in set(self._servers) - wanted:
            self._servers.pop(port).close()
        for port in wanted - set(self._servers):
            try:
                self._servers[port] = await asyncio.start_server(lambda r, w, p=port: self._accepted(p, r, w), self._bind_host, port)
            except OSError as exc:
                log.warning("relay: cannot listen on %s:%d (%s)", self._bind_host, port, exc)

    async def _accepted(self, port: int, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        sid = next(self._ids)
        self._streams[sid] = writer
        try:
            await self._frame(OPEN, sid, struct.pack(">H", port))
            await self._pump(sid, reader)
        except (OSError, ConnectionError, RuntimeError):
            pass
        finally:
            await self._end(sid, notify=True)

    # ---- frames from the other end ----------------------------------------------------------------------
    async def handle(self, data: bytes) -> None:
        if len(data) < HEADER.size:
            return
        kind, sid = HEADER.unpack_from(data)
        body = data[HEADER.size:]
        if kind == OPEN:
            host, port = self._dial(struct.unpack(">H", body[:2])[0])
            try:
                reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), DIAL_TIMEOUT)
            except (OSError, asyncio.TimeoutError):
                await self._frame(CLOSE, sid, b"")
                return
            self._streams[sid] = writer
            task = asyncio.create_task(self._pump_and_end(sid, reader))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        elif kind == DATA:
            writer = self._streams.get(sid)
            if writer is not None:
                try:
                    writer.write(body)
                    await writer.drain()
                except (OSError, ConnectionError):
                    await self._end(sid, notify=True)
        elif kind == CLOSE:
            await self._end(sid, notify=False)

    # ---- helpers ------------------------------------------------------------------------------------------
    async def _frame(self, kind: int, sid: int, payload: bytes) -> None:
        await self._send(HEADER.pack(kind, sid) + payload)

    async def _pump(self, sid: int, reader: asyncio.StreamReader) -> None:
        while True:
            chunk = await reader.read(CHUNK)
            if not chunk:
                return
            await self._frame(DATA, sid, chunk)

    async def _pump_and_end(self, sid: int, reader: asyncio.StreamReader) -> None:
        try:
            await self._pump(sid, reader)
        except (OSError, ConnectionError, RuntimeError):
            pass
        finally:
            await self._end(sid, notify=True)

    async def _end(self, sid: int, notify: bool) -> None:
        writer = self._streams.pop(sid, None)
        if writer is None:
            return
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass
        if notify:
            try:
                await self._frame(CLOSE, sid, b"")
            except (OSError, ConnectionError, RuntimeError):
                pass

    async def close(self) -> None:
        """Stop listening and drop every stream (the peers' connections then see an ordinary disconnect)."""
        for server in self._servers.values():
            server.close()
        self._servers.clear()
        for task in list(self._tasks):
            task.cancel()
        for writer in list(self._streams.values()):
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass
        self._streams.clear()
