"""Everything a provider needs to join with one command: the join script, the agent source and BloomBee's requirements.

    curl -fsSL https://<platform>/join.sh | bash -s -- --token <TOKEN>

Nothing here is secret (the token is passed on the command line by the provider).
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, Response

from app.config import BACKEND_DIR

router = APIRouter(tags=["install"])
REPO = BACKEND_DIR.parent
JOIN_SCRIPT = Path(__file__).resolve().parent.parent / "install" / "join.sh"
REQUIREMENTS = REPO / "experiments" / "bloombee-mac" / "requirements.txt"
AGENT_DIR = REPO / "agent" / "bloombee_agent"


def public_base(request: Request) -> str:
    """The address providers should use: BLOOMBEE_PUBLIC_URL, else the URL the caller used (honours a tunnel's forwarded headers)."""
    configured = request.app.state.settings.public_url
    if configured:
        return configured
    proto = request.headers.get("x-forwarded-proto", request.url.scheme).split(",")[0].strip()
    host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc)).split(",")[0].strip()
    return f"{proto}://{host}"


@router.get("/install/info")
def install_info(request: Request):
    base = public_base(request)
    return {"public_url": base, "join_command": f"curl -fsSL {base}/join.sh | bash -s -- --token <TOKEN>"}


@router.get("/join.sh", response_class=PlainTextResponse)
def join_script(request: Request):
    return PlainTextResponse(JOIN_SCRIPT.read_text().replace("__SERVER__", public_base(request)), media_type="text/x-shellscript")


@router.get("/install/requirements.txt", response_class=PlainTextResponse)
def requirements():
    if not REQUIREMENTS.exists():
        raise HTTPException(404, "requirements are not available on this server")
    return PlainTextResponse(REQUIREMENTS.read_text())


@router.get("/install/agent.tgz")
def agent_archive():
    if not AGENT_DIR.is_dir():
        raise HTTPException(404, "the agent source is not available on this server")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(AGENT_DIR.rglob("*.py")):
            tar.add(path, arcname=f"bloombee_agent/{path.relative_to(AGENT_DIR)}")
    return Response(buffer.getvalue(), media_type="application/gzip")
