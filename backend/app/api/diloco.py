"""File exchange between DiLoCo replicas (provider agents) and the job's coordinator (runner on the platform host).

The coordinator writes `<job dir>/diloco/state.json` and `rounds/<r>/global.safetensors`; replicas read them here and upload
their pseudo-gradient (theta_start - theta_end, bf16) for the current round. Every call names the replica's lease, which must
be active and belong to the calling agent: a provider can only reach jobs running on its own GPUs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from app.db import get_session
from app.deps import AgentContext, current_agent
from app.models import Gpu, Job, Lease

router = APIRouter(prefix="/agent/diloco", tags=["diloco"])
MAX_DELTA_BYTES = 8 * 1024 ** 3


def _job_dir(request: Request, job_id: int) -> Path:
    return Path(request.app.state.settings.jobs_dir).expanduser() / str(job_id) / "diloco"


def _replica(session: Session, ctx: AgentContext, job_id: int, lease_id: int) -> tuple[Job, Lease]:
    lease = session.get(Lease, lease_id)
    job = session.get(Job, job_id)
    gpu = session.get(Gpu, lease.gpu_id) if lease else None
    if (lease is None or job is None or lease.job_id != job_id or lease.ended_at is not None or gpu is None
            or gpu.agent_token_id != ctx.token.id or json.loads(job.config_json or "{}").get("method") != "diloco"):
        raise HTTPException(404, "No active replica lease of yours on this job")
    return job, lease


def _state(request: Request, job_id: int) -> dict:
    try:
        return json.loads((_job_dir(request, job_id) / "state.json").read_text())
    except (OSError, ValueError):
        raise HTTPException(409, "The coordinator has not published a round yet")


@router.get("/{job_id}/state")
def state(job_id: int, lease_id: int, request: Request, ctx: AgentContext = Depends(current_agent),
          session: Session = Depends(get_session)):
    job, lease = _replica(session, ctx, job_id, lease_id)
    cfg = json.loads(job.config_json or "{}")
    return {**_state(request, job_id), "slot": lease.layer_start, "replicas": job.requested_gpus or 1,
            "inner_steps": cfg.get("inner_steps", 20), "lr": cfg.get("lr", 1e-5), "batch_size": cfg.get("batch_size", 2),
            "seq_len": cfg.get("seq_len", 128)}


@router.get("/{job_id}/data")
def data(job_id: int, lease_id: int, ctx: AgentContext = Depends(current_agent), session: Session = Depends(get_session)):
    job, _ = _replica(session, ctx, job_id, lease_id)
    path = Path(job.dataset_path)
    if not path.is_file():
        raise HTTPException(404, "Dataset file is missing")
    return FileResponse(path, media_type="application/jsonl")


@router.get("/{job_id}/weights/{round_no}")
def weights(job_id: int, round_no: int, lease_id: int, request: Request, ctx: AgentContext = Depends(current_agent),
            session: Session = Depends(get_session)):
    _replica(session, ctx, job_id, lease_id)
    path = _job_dir(request, job_id) / "rounds" / str(round_no) / "global.safetensors"
    if not path.is_file():
        raise HTTPException(404, f"No global weights for round {round_no}")
    return FileResponse(path, media_type="application/octet-stream",
                        headers={"X-Sha256": (path.with_suffix(".sha256")).read_text().strip()
                                 if path.with_suffix(".sha256").exists() else ""})


@router.put("/{job_id}/delta/{round_no}")
async def upload_delta(job_id: int, round_no: int, lease_id: int, sha256: str, tokens: int, loss: float, request: Request,
                       ctx: AgentContext = Depends(current_agent), session: Session = Depends(get_session)):
    """Idempotent per (round, slot): a retry overwrites the same file. Uploads for an older round are refused (the round
    already closed without this replica)."""
    _, lease = _replica(session, ctx, job_id, lease_id)
    st = _state(request, job_id)
    if round_no != st.get("round") or st.get("status") != "running":
        raise HTTPException(409, f"Round {round_no} is closed (current round {st.get('round')}, {st.get('status')})")
    folder = _job_dir(request, job_id) / "rounds" / str(round_no)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f".delta-slot{lease.layer_start}-lease{lease.id}.part"
    digest, size = hashlib.sha256(), 0
    with open(tmp, "wb") as f:
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_DELTA_BYTES:
                f.close()
                tmp.unlink(missing_ok=True)
                raise HTTPException(413, "Upload too large")
            digest.update(chunk)
            f.write(chunk)
    if digest.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise HTTPException(422, "Checksum mismatch: upload again")
    final = folder / f"delta-slot{lease.layer_start}.safetensors"
    meta = {"slot": lease.layer_start, "lease_id": lease.id, "tokens": tokens, "loss": loss, "bytes": size, "sha256": sha256}
    (folder / f"delta-slot{lease.layer_start}.json.part").write_text(json.dumps(meta))
    os.replace(tmp, final)  # the coordinator only reads complete files: data first, then its metadata
    os.replace(folder / f"delta-slot{lease.layer_start}.json.part", folder / f"delta-slot{lease.layer_start}.json")
    return {"ok": True, "round": round_no, "slot": lease.layer_start, "bytes": size}
