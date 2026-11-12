"""Model files for provider agents on other machines (agent-token auth, least privilege).

An agent may fetch a fine-tuned model's merged weights only while one of its own GPUs holds an active lease for a
deployment of that model; nothing else is downloadable through here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from app.db import get_session
from app.deps import AgentContext, current_agent
from app.models import FineTunedModel, Gpu, Job, Lease

router = APIRouter(prefix="/files", tags=["files"])


def authorized_model_dir(session: Session, ctx: AgentContext, model_id: int) -> Path:
    leases = session.exec(
        select(Lease, Gpu, Job).where(Lease.ended_at == None, Gpu.id == Lease.gpu_id, Job.id == Lease.job_id,  # noqa: E711
                                      Gpu.agent_token_id == ctx.token.id, Job.kind == "deploy")
    ).all()
    allowed = any(json.loads(job.config_json or "{}").get("finetuned_model_id") == model_id for _, _, job in leases)
    model = session.get(FineTunedModel, model_id)
    if not allowed or model is None or not model.merged_path or not Path(model.merged_path).is_dir():
        raise HTTPException(404, "No such model for this agent")
    return Path(model.merged_path).resolve()


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@router.get("/models/{model_id}/manifest")
def manifest(model_id: int, ctx: AgentContext = Depends(current_agent), session: Session = Depends(get_session)):
    root = authorized_model_dir(session, ctx, model_id)
    files = [{"path": str(p.relative_to(root)), "size": p.stat().st_size, "sha256": sha256_of(p)}
             for p in sorted(root.rglob("*")) if p.is_file()]
    return {"model_id": model_id, "files": files}


@router.get("/models/{model_id}/file/{rel_path:path}")
def download(model_id: int, rel_path: str, ctx: AgentContext = Depends(current_agent), session: Session = Depends(get_session)):
    root = authorized_model_dir(session, ctx, model_id)
    target = (root / rel_path).resolve()
    if root not in target.parents or not target.is_file():  # blocks ../ traversal and symlink escapes
        raise HTTPException(404, "File not found")
    return FileResponse(target)
