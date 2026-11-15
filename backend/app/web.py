"""Serve the built website (frontend/dist) from the backend, so one public address gives both the site and the API.

- `/api/...` (HTTP and WebSocket) is the API with the prefix removed: the website calls the API there, because some page
  paths (/models, /deployments, ...) are also API paths.
- Files that exist in the build (/assets/..., /favicon.svg) are served as they are.
- A browser page load (GET accepting text/html) of any other path gets index.html; the website's router takes over.
- Everything else (agents, the SDK, the desktop app, curl | bash) reaches the API unchanged at its usual paths.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path

API_PREFIX = "/api"


class WebsiteMiddleware:
    def __init__(self, app, web_dir: Path):
        self.app, self.root = app, web_dir.resolve()
        self.index = (self.root / "index.html").read_bytes()

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        if path == API_PREFIX or path.startswith(API_PREFIX + "/"):
            inner = path[len(API_PREFIX):] or "/"
            scope = {**scope, "path": inner, "raw_path": inner.encode(), "root_path": scope.get("root_path", "")}
            return await self.app(scope, receive, send)
        if scope["type"] == "http" and scope.get("method") in ("GET", "HEAD"):
            target = (self.root / path.lstrip("/")).resolve()
            if path not in ("", "/") and target.is_file() and self.root in target.parents:
                ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                cache = b"public, max-age=31536000, immutable" if "/assets/" in path else b"no-cache"
                return await self._send(send, target.read_bytes(), ctype, cache)
            accept = dict(scope.get("headers", [])).get(b"accept", b"").decode()
            if "text/html" in accept:
                return await self._send(send, self.index, "text/html; charset=utf-8", b"no-cache")
        return await self.app(scope, receive, send)

    @staticmethod
    async def _send(send, body: bytes, ctype: str, cache: bytes):
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", ctype.encode()), (b"content-length", str(len(body)).encode()),
                                (b"cache-control", cache)]})
        await send({"type": "http.response.body", "body": body})
