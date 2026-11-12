"""Deploy a base model or one of your fine-tuned models on N pool GPUs, query it, compare, and evaluate."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.api.jobs import JobOut, job_out
from app.db import get_session, iso, utcnow
from app.deps import current_user
from app.models import BaseModel as BaseModelRow
from app.models import Deployment, FineTunedModel, Job, User
from app.services import evaluation

router = APIRouter(tags=["deployments"])


class DeploymentIn(BaseModel):
    model_kind: str = "finetuned"  # base | finetuned
    base_model_id: int | None = None
    finetuned_model_id: int | None = None
    requested_gpus: int | None = Field(default=None, ge=1)  # None = automatic
    trusted_only: bool = False


class DeploymentOut(BaseModel):
    id: int
    model_kind: str
    model_name: str
    base_model: str
    finetuned_model_id: int | None
    status: str  # queued | starting | running | stopped | failed
    num_gpus: int
    endpoint: str
    created_at: str
    job: JobOut


class GenerateIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=32, ge=1, le=256)
    do_sample: bool = False  # greedy by default: repeatable, and what the comparison uses
    temperature: float | None = Field(default=None, gt=0, le=5)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1, le=1000)
    seed: int | None = None


class CompareIn(GenerateIn):
    base_deployment_id: int
    tuned_deployment_id: int


class EvaluateIn(BaseModel):
    base_deployment_id: int
    tuned_deployment_id: int
    limit: int | None = Field(default=None, ge=1, le=500)


def deployment_out(session: Session, dep: Deployment) -> DeploymentOut:
    job = session.get(Job, dep.job_id)
    base = session.get(BaseModelRow, dep.base_model_id)
    model = session.get(FineTunedModel, dep.finetuned_model_id) if dep.finetuned_model_id else None
    status = "queued" if job.status == "queued" and dep.status == "starting" else dep.status
    return DeploymentOut(id=dep.id, model_kind=dep.model_kind, model_name=model.name if model else base.name, base_model=base.name,
                         finetuned_model_id=dep.finetuned_model_id, status=status, num_gpus=dep.num_gpus, endpoint=dep.endpoint,
                         created_at=iso(dep.created_at), job=job_out(session, job))


def owned_deployment(session: Session, dep_id: int, user: User) -> Deployment:
    dep = session.get(Deployment, dep_id)
    if dep is None or dep.owner_id != user.id:
        raise HTTPException(404, "Deployment not found")
    return dep


@router.post("/deployments", response_model=DeploymentOut, status_code=201)
async def create_deployment(body: DeploymentIn, request: Request, user: User = Depends(current_user),
                            session: Session = Depends(get_session)):
    if body.model_kind not in ("base", "finetuned"):
        raise HTTPException(422, "model_kind must be 'base' or 'finetuned'")
    if body.model_kind == "finetuned":
        model = session.get(FineTunedModel, body.finetuned_model_id or 0)
        if model is None or model.owner_id != user.id:
            raise HTTPException(404, "Model not found")
        base_id, tuned_id = model.base_model_id, model.id
    else:
        if session.get(BaseModelRow, body.base_model_id or 0) is None:
            raise HTTPException(404, "Base model not found")
        base_id, tuned_id = body.base_model_id, None
    base = session.get(BaseModelRow, base_id)
    if body.requested_gpus is not None and body.requested_gpus > base.num_layers:
        raise HTTPException(422, f"requested_gpus cannot exceed the model's {base.num_layers} layers")
    job = Job(owner_id=user.id, kind="deploy", status="queued", base_model_id=base_id, requested_gpus=body.requested_gpus,
              trusted_only=body.trusted_only, config_json=json.dumps({"finetuned_model_id": tuned_id}))
    session.add(job)
    session.commit()
    session.refresh(job)
    dep = Deployment(owner_id=user.id, model_kind=body.model_kind, base_model_id=base_id, finetuned_model_id=tuned_id,
                     job_id=job.id, num_gpus=body.requested_gpus or 0, status="starting")
    session.add(dep)
    session.commit()
    session.refresh(dep)
    await request.app.state.orchestrator.submit(job.id)
    session.refresh(dep)
    return deployment_out(session, dep)


@router.get("/deployments", response_model=list[DeploymentOut])
def list_deployments(user: User = Depends(current_user), session: Session = Depends(get_session)):
    rows = session.exec(select(Deployment).where(Deployment.owner_id == user.id).order_by(Deployment.id.desc())).all()
    return [deployment_out(session, d) for d in rows]


@router.get("/deployments/{dep_id}", response_model=DeploymentOut)
def get_deployment(dep_id: int, user: User = Depends(current_user), session: Session = Depends(get_session)):
    return deployment_out(session, owned_deployment(session, dep_id, user))


@router.post("/deployments/{dep_id}/stop")
async def stop_deployment(dep_id: int, request: Request, user: User = Depends(current_user), session: Session = Depends(get_session)):
    dep = owned_deployment(session, dep_id, user)
    job = session.get(Job, dep.job_id)
    if job.status == "queued":  # never started: nothing to release
        job.status, job.updated_at = "stopped", utcnow()
        dep.status = "stopped"
        session.add(job); session.add(dep)
        session.commit()
        return {"ok": True}
    if dep.status in ("stopped", "failed") or not await request.app.state.orchestrator.cancel(dep.job_id):
        raise HTTPException(409, "The deployment is not running")
    return {"ok": True}


def _generate_params(body: GenerateIn) -> dict:
    return {k: v for k, v in body.model_dump().items() if v is not None and k != "base_deployment_id"}


async def _generate(request: Request, dep: Deployment, params: dict) -> dict:
    orch = request.app.state.orchestrator
    try:
        out = await orch.generate(dep.job_id, params)
    except LookupError:
        raise HTTPException(409, "The deployment is not running")
    except TimeoutError:
        raise HTTPException(504, "The model did not answer in time")
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))
    return {"deployment_id": dep.id, "text": out["text"], "tokens": out.get("tokens", 0), "seconds": out.get("seconds", 0.0)}


@router.post("/deployments/{dep_id}/generate")
async def generate(dep_id: int, body: GenerateIn, request: Request, user: User = Depends(current_user),
                   session: Session = Depends(get_session)):
    dep = owned_deployment(session, dep_id, user)
    return await _generate(request, dep, _generate_params(body))


@router.post("/compare")
async def compare(body: CompareIn, request: Request, user: User = Depends(current_user), session: Session = Depends(get_session)):
    """One prompt, both models, same settings: the base model and the fine-tuned model side by side."""
    base = owned_deployment(session, body.base_deployment_id, user)
    tuned = owned_deployment(session, body.tuned_deployment_id, user)
    if base.model_kind != "base" or tuned.model_kind != "finetuned":
        raise HTTPException(422, "Pass a base deployment and a fine-tuned deployment")
    params = {k: v for k, v in _generate_params(body).items() if k not in ("base_deployment_id", "tuned_deployment_id")}
    a, b = await asyncio.gather(_generate(request, base, params), _generate(request, tuned, params))
    return {"prompt": body.prompt, "base": a, "finetuned": b, "sampled": body.do_sample}


@router.post("/models/{model_id}/evaluate", status_code=202)
async def evaluate(model_id: int, body: EvaluateIn, request: Request, user: User = Depends(current_user),
                   session: Session = Depends(get_session)):
    """Score the base and fine-tuned models on the dataset's held-out examples (runs in the background)."""
    model = session.get(FineTunedModel, model_id)
    if model is None or model.owner_id != user.id:
        raise HTTPException(404, "Model not found")
    base = owned_deployment(session, body.base_deployment_id, user)
    tuned = owned_deployment(session, body.tuned_deployment_id, user)
    if base.model_kind != "base" or tuned.finetuned_model_id != model.id:
        raise HTTPException(422, "Pass a base deployment and a deployment of this model")
    if base.status != "running" or tuned.status != "running":
        raise HTTPException(409, "Both deployments must be running")
    if json.loads(model.scores_json or "{}").get("status") == "running":
        raise HTTPException(409, "An evaluation is already running for this model")
    orch = request.app.state.orchestrator
    task = asyncio.create_task(evaluation.evaluate_model(orch, model.id, base.job_id, tuned.job_id, body.limit))
    orch.background.add(task)
    task.add_done_callback(orch.background.discard)
    return {"ok": True, "status": "running"}
