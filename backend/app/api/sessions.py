"""RL / custom-training sessions (Tinker-style) for the `syntra` SDK and the website.

A session leases GPUs, keeps a LoRA (or QLoRA) adapter in a trainer process on the platform, and answers one primitive per
call: sample, logprobs, forward_backward (cross_entropy | dpo | ppo | importance_sampling), optim_step, advantages,
tokenize, detokenize. The user's own code runs the RL loop and computes rewards; only token ids and numbers travel.
Billing is the same as training (per leased GPU-second). An idle session is saved and closed automatically.
"""
from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.api.jobs import JobOut, check_weight_bits, init_adapter, job_out
from app.db import get_session
from app.deps import current_user
from app.models import BaseModel as BaseModelRow
from app.models import Job, User

router = APIRouter(tags=["sessions"])
CALL_OPS = ("sample", "logprobs", "forward_backward", "optim_step", "advantages", "tokenize", "detokenize", "state")
MAX_CALL_BYTES = 8 * 1024 * 1024


class SessionIn(BaseModel):
    base_model_id: int
    init_model_id: int | None = None  # continue from one of your fine-tuned models (it is also the reference policy)
    rank: int = Field(default=8, ge=1, le=64)
    lora_alpha: float = Field(default=16, gt=0, le=1024)
    weight_bits: Literal[16, 4] = 16
    requested_gpus: int | None = Field(default=None, ge=1)
    trusted_only: bool = False
    max_seq_len: int = Field(default=1024, ge=32, le=4096)
    name: str = Field(default="", max_length=80)


class SessionOut(BaseModel):
    id: int
    status: str  # queued | starting | running | stopped | failed
    ready: bool  # accepting calls
    base_model: str
    config: dict
    job: JobOut


class CallIn(BaseModel):
    op: str
    args: dict = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, max_length=128)


class SaveIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class CloseIn(BaseModel):
    save_as: str | None = Field(default=None, max_length=80)


def session_out(request: Request, session: Session, job: Job) -> SessionOut:
    run = request.app.state.orchestrator.runs.get(job.id)
    base = session.get(BaseModelRow, job.base_model_id)
    ready = bool(run and run.serving and not run.closing)
    status = "running" if ready else job.status
    cfg = {k: v for k, v in json.loads(job.config_json or "{}").items() if k not in ("init_adapter",)}
    return SessionOut(id=job.id, status=status, ready=ready, base_model=base.name if base else "", config=cfg,
                      job=job_out(session, job))


def owned_session(session: Session, session_id: int, user: User) -> Job:
    job = session.get(Job, session_id)
    if job is None or job.owner_id != user.id or job.kind != "session":
        raise HTTPException(404, "Session not found")
    return job


@router.post("/sessions", response_model=SessionOut, status_code=201)
async def create_session(body: SessionIn, request: Request, user: User = Depends(current_user),
                         session: Session = Depends(get_session)):
    base = session.get(BaseModelRow, body.base_model_id)
    if base is None:
        raise HTTPException(404, "Base model not found")
    if body.requested_gpus is not None and body.requested_gpus > base.num_layers:
        raise HTTPException(422, f"requested_gpus cannot exceed the model's {base.num_layers} layers")
    check_weight_bits(base, body.weight_bits)
    start = init_adapter(session, user, base, body.init_model_id)
    config = {"rank": body.rank, "lora_alpha": body.lora_alpha, "weight_bits": body.weight_bits, "max_seq_len": body.max_seq_len,
              "init_adapter": start, "init_model_id": body.init_model_id,
              "model_name": body.name.strip() or f"{base.name} RL session"}
    job = Job(owner_id=user.id, kind="session", status="queued", base_model_id=base.id, requested_gpus=body.requested_gpus,
              trusted_only=body.trusted_only, config_json=json.dumps(config))
    session.add(job)
    session.commit()
    session.refresh(job)
    await request.app.state.orchestrator.submit(job.id)
    session.refresh(job)
    return session_out(request, session, job)


@router.get("/sessions", response_model=list[SessionOut])
def list_sessions(request: Request, user: User = Depends(current_user), session: Session = Depends(get_session)):
    rows = session.exec(select(Job).where(Job.owner_id == user.id, Job.kind == "session").order_by(Job.id.desc())).all()
    return [session_out(request, session, j) for j in rows]


@router.get("/sessions/{session_id}", response_model=SessionOut)
def get_session_status(session_id: int, request: Request, user: User = Depends(current_user), session: Session = Depends(get_session)):
    return session_out(request, session, owned_session(session, session_id, user))


@router.post("/sessions/{session_id}/call")
async def call(session_id: int, body: CallIn, request: Request, user: User = Depends(current_user),
               session: Session = Depends(get_session)):
    owned_session(session, session_id, user)
    if body.op not in CALL_OPS:
        raise HTTPException(422, f"op must be one of {CALL_OPS} (save and close have their own endpoints)")
    if len(json.dumps(body.args)) > MAX_CALL_BYTES:
        raise HTTPException(413, "Call is larger than 8 MB; split the batch")
    orch = request.app.state.orchestrator
    try:
        data = await orch.call(session_id, body.op, body.args, body.idempotency_key)
    except LookupError:
        raise HTTPException(409, "The session is not running (still starting, or closed)")
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except TimeoutError as exc:
        raise HTTPException(504, str(exc))
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))
    return {"op": body.op, "data": data}


@router.post("/sessions/{session_id}/save")
async def save(session_id: int, body: SaveIn, request: Request, user: User = Depends(current_user),
               session: Session = Depends(get_session)):
    owned_session(session, session_id, user)
    try:
        model_id = await request.app.state.orchestrator.save_session(session_id, body.name.strip())
    except LookupError:
        raise HTTPException(409, "The session is not running")
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(502, str(exc))
    return {"model_id": model_id, "name": body.name.strip()}


@router.post("/sessions/{session_id}/close")
async def close(session_id: int, body: CloseIn, request: Request, user: User = Depends(current_user),
                session: Session = Depends(get_session)):
    job = owned_session(session, session_id, user)
    orch = request.app.state.orchestrator
    if job.status == "queued":
        job.status = "stopped"
        session.add(job)
        session.commit()
        return {"ok": True, "model_id": None}
    try:
        model_id = await orch.close_session(session_id, save_as=(body.save_as or "").strip() or None)
    except LookupError:
        raise HTTPException(409, "The session is not running")
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(502, f"saving before closing failed: {exc}")
    return {"ok": True, "model_id": model_id}
