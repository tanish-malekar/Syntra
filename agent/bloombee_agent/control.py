"""Local control API for the provider desktop app (127.0.0.1 only, shared-secret header).

  GET  /status            state, GPUs, workers, last heartbeat, last error
  POST /pause  {"mode": "drain" | "now"}   stop taking work (drain: let running jobs finish; now: stop them, the platform moves them)
  POST /resume
  POST /quit              report the GPUs offline and exit

The port, the secret and the PID are written to `<state_dir>/agent.json` (mode 0600) so the app can find a running agent,
including one started from a terminal. Only the standard library is used (the agent's dependencies stay httpx + websockets).
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import secrets
from pathlib import Path
from typing import Awaitable, Callable

log = logging.getLogger("bloombee_platform_agent.control")

Handler = Callable[[dict], Awaitable[dict]]
MAX_BODY = 64 * 1024


class AlreadyRunning(RuntimeError):
    pass


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    return True


class InstanceLock:
    """One agent per machine and user. A stale lock (its process is gone) is taken over."""

    def __init__(self, state_dir: Path):
        self.path = Path(state_dir).expanduser() / "agent.lock"

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                try:
                    other = int(self.path.read_text().strip() or 0)
                except (OSError, ValueError):
                    other = 0
                if other != os.getpid() and pid_alive(other):
                    raise AlreadyRunning(f"another BloomBee agent is already running (pid {other})")
                self.path.unlink(missing_ok=True)  # stale: take it over
                continue
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
            return
        raise AlreadyRunning("could not take the agent lock")

    def release(self) -> None:
        try:
            if self.path.read_text().strip() == str(os.getpid()):
                self.path.unlink()
        except OSError:
            pass


class ControlServer:
    def __init__(self, routes: dict[tuple[str, str], Handler], state_dir: Path, port: int = 0):
        self.routes, self.state_dir, self.port = routes, Path(state_dir).expanduser(), port
        self.secret = secrets.token_urlsafe(24)
        self._server: asyncio.base_events.Server | None = None

    @property
    def info_path(self) -> Path:
        return self.state_dir / "agent.json"

    async def start(self) -> int:
        self._server = await asyncio.start_server(self._client, "127.0.0.1", self.port)
        self.port = self._server.sockets[0].getsockname()[1]
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.info_path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"port": self.port, "secret": self.secret, "pid": os.getpid()}, f)
        os.replace(tmp, self.info_path)
        log.info("control API on 127.0.0.1:%d", self.port)
        return self.port

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        try:
            if json.loads(self.info_path.read_text()).get("pid") == os.getpid():
                self.info_path.unlink()
        except (OSError, ValueError):
            pass

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        status, body = 500, {"error": "internal error"}
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
            lines = head.decode("latin-1").split("\r\n")
            method, path, _ = lines[0].split(" ", 2)
            headers = {k.strip().lower(): v.strip() for k, v in (l.split(":", 1) for l in lines[1:] if ":" in l)}
            length = int(headers.get("content-length", "0") or 0)
            if length > MAX_BODY:
                status, body = 413, {"error": "body too large"}
            elif not hmac.compare_digest(headers.get("x-bloombee-secret", ""), self.secret):
                status, body = 401, {"error": "bad secret"}
            else:
                raw = await asyncio.wait_for(reader.readexactly(length), 5) if length else b""
                payload = json.loads(raw) if raw else {}
                handler = self.routes.get((method.upper(), path.split("?", 1)[0]))
                if handler is None:
                    status, body = 404, {"error": f"no route {method} {path}"}
                else:
                    status, body = 200, await handler(payload if isinstance(payload, dict) else {})
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ValueError) as exc:
            status, body = 400, {"error": f"bad request: {exc}"}
        except Exception as exc:  # never let one request break the agent
            log.exception("control request failed")
            status, body = 500, {"error": f"{type(exc).__name__}: {exc}"}
        data = json.dumps(body).encode()
        reason = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 413: "Too Large"}.get(status, "Error")
        writer.write(f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n"
                     f"Connection: close\r\n\r\n".encode() + data)
        try:
            await writer.drain()
        finally:
            writer.close()
