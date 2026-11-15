"""The relay: TCP streams carried over a message pipe, in both directions."""
from __future__ import annotations

import asyncio
import os
import socket

import functools

from bloombee_agent.relay import Relay


def sync(fn):
    @functools.wraps(fn)
    def run():
        asyncio.run(fn())
    return run


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def echo_server(port: int):
    async def handle(r, w):
        while data := await r.read(65536):
            w.write(data)
            await w.drain()
        w.close()
    return await asyncio.start_server(handle, "127.0.0.1", port)


def pair(dial_a=None, dial_b=None):
    """Two relays wired back to back; frames are delivered in order by one task per direction."""
    queues = {"a": asyncio.Queue(), "b": asyncio.Queue()}
    a = Relay(lambda d: queues["b"].put(d), 0, dial=dial_a)
    b = Relay(lambda d: queues["a"].put(d), 1, dial=dial_b)

    async def deliver(q, relay):
        while True:
            await relay.handle(await q.get())

    tasks = [asyncio.create_task(deliver(queues["a"], a)), asyncio.create_task(deliver(queues["b"], b))]
    return a, b, tasks


@sync
async def test_bytes_flow_both_ways_through_the_relay():
    target, listen = free_port(), free_port()
    server = await echo_server(target)
    a, b, tasks = pair(dial_b=lambda port: ("127.0.0.1", target))  # a listens on `listen`; b dials the echo server
    await a.listen([listen])
    r, w = await asyncio.open_connection("127.0.0.1", listen)
    payload = os.urandom(500_000)  # larger than one chunk
    w.write(payload)
    await w.drain()
    assert await asyncio.wait_for(r.readexactly(len(payload)), 10) == payload
    w.close()
    for t in tasks:
        t.cancel()
    await a.close(); await b.close()
    server.close()


@sync
async def test_several_streams_at_once_and_close_propagates():
    target, listen = free_port(), free_port()
    server = await echo_server(target)
    a, b, tasks = pair(dial_b=lambda port: ("127.0.0.1", target))
    await a.listen([listen])
    conns = [await asyncio.open_connection("127.0.0.1", listen) for _ in range(5)]
    for i, (r, w) in enumerate(conns):
        w.write(f"hello {i}".encode())
        await w.drain()
    for i, (r, w) in enumerate(conns):
        assert await asyncio.wait_for(r.readexactly(7), 5) == f"hello {i}".encode()
    r, w = conns[0]
    w.close()  # the far end sees the close and drops its side too
    await asyncio.sleep(0.3)
    assert len(b._streams) == 4 and len(a._streams) == 4
    for t in tasks:
        t.cancel()
    await a.close(); await b.close()
    server.close()


@sync
async def test_a_dead_target_closes_the_stream_instead_of_hanging():
    listen, dead = free_port(), free_port()  # nothing listens on `dead`
    a, b, tasks = pair(dial_b=lambda port: ("127.0.0.1", dead))
    await a.listen([listen])
    r, w = await asyncio.open_connection("127.0.0.1", listen)
    w.write(b"anyone there?")
    assert await asyncio.wait_for(r.read(10), 5) == b""  # connection closed
    for t in tasks:
        t.cancel()
    await a.close(); await b.close()


@sync
async def test_listen_follows_the_wanted_set_and_skips_busy_ports():
    p1, p2 = free_port(), free_port()
    a, b, tasks = pair()
    await a.listen([p1, p2])
    assert a.listening == {p1, p2}
    await a.listen([p2])
    assert a.listening == {p2}
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        await a.listen([p2, busy.getsockname()[1]])  # in use by someone else: skipped, no crash
        assert a.listening == {p2}
    for t in tasks:
        t.cancel()
    await a.close(); await b.close()
