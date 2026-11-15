"""Datasets, plan preview, training jobs, and the stored fine-tuned models."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.db import get_session, iso
from app.deps import current_user
from app.models import BaseModel as BaseModelRow
from app.models import Dataset, Deployment, FineTunedModel, Gpu, GpuClass, Job, Lease, Provider, User
from app.services import planner, scheduler
from app.services.pool import effective_status
from app.db import utcnow

router = APIRouter(tags=["jobs"])
MAX_DATASET_BYTES = 10 * 1024 * 1024


# --------------------------------------------------------------------------------------------- datasets
class DatasetOut(BaseModel):
    id: int
    name: str
    num_rows: int
    builtin: bool
    has_heldout: bool
    format: str  # text (SFT) | preference (DPO)
    created_at: str


def dataset_out(d: Dataset) -> DatasetOut:
    return DatasetOut(id=d.id, name=d.name, num_rows=d.num_rows, builtin=d.builtin, has_heldout=bool(d.heldout_path),
                      format=d.format or "text", created_at=iso(d.created_at))


def visible_dataset(session: Session, dataset_id: int, user: User) -> Dataset:
    d = session.get(Dataset, dataset_id)
    if d is None or not (d.builtin or d.owner_id == user.id):
        raise HTTPException(404, "Dataset not found")
    return d


@router.get("/datasets", response_model=list[DatasetOut])
def list_datasets(user: User = Depends(current_user), session: Session = Depends(get_session)):
    rows = session.exec(select(Dataset).where((Dataset.builtin == True) | (Dataset.owner_id == user.id)).order_by(Dataset.id)).all()  # noqa: E712
    return [dataset_out(d) for d in rows]


PREFERENCE_FIELDS = ("prompt", "chosen", "rejected")


def validate_jsonl(raw: bytes) -> tuple[list[str], str]:
    """Every non-empty line is a JSON object of one format, detected from the first line:
    text ({"text": ...}, for supervised fine-tuning) or preference ({"prompt", "chosen", "rejected"}, for DPO)."""
    try:
        lines = [l for l in raw.decode("utf-8").splitlines() if l.strip()]
    except UnicodeDecodeError:
        raise HTTPException(422, "The file is not valid UTF-8 text")
    if len(lines) < 8:
        raise HTTPException(422, "A dataset needs at least 8 examples")
    fmt = None
    for i, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except ValueError:
            raise HTTPException(422, f"Line {i} is not valid JSON")
        if not isinstance(record, dict):
            raise HTTPException(422, f"Line {i} must be a JSON object")
        if fmt is None:
            fmt = "preference" if all(k in record for k in PREFERENCE_FIELDS) else "text"
        if fmt == "text":
            if not isinstance(record.get("text"), str) or not record["text"].strip():
                raise HTTPException(422, f'Line {i} must be an object with a non-empty string field "text" '
                                         f'(or, for preference data, "prompt", "chosen" and "rejected" on every line)')
        else:
            bad = [k for k in PREFERENCE_FIELDS if not isinstance(record.get(k), str) or not record[k].strip()]
            if bad:
                raise HTTPException(422, f"Line {i}: preference data needs non-empty strings {', '.join(PREFERENCE_FIELDS)} "
                                         f"(missing or empty: {', '.join(bad)})")
            if record["chosen"] == record["rejected"]:
                raise HTTPException(422, f'Line {i}: "chosen" and "rejected" are identical (no preference to learn)')
    return lines, fmt or "text"


@router.post("/datasets", response_model=DatasetOut, status_code=201)
async def upload_dataset(request: Request, name: str = Form(...), file: UploadFile = File(...),
                         user: User = Depends(current_user), session: Session = Depends(get_session)):
    raw = await file.read(MAX_DATASET_BYTES + 1)
    if len(raw) > MAX_DATASET_BYTES:
        raise HTTPException(413, "Dataset is larger than 10 MB")
    lines, fmt = validate_jsonl(raw)
    folder = Path(request.app.state.settings.datasets_dir).expanduser() / str(user.id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex}.jsonl"
    path.write_text("\n".join(lines) + "\n")
    d = Dataset(owner_id=user.id, name=name.strip()[:80] or "Untitled dataset", path=str(path), num_rows=len(lines), format=fmt)
    session.add(d)
    session.commit()
    session.refresh(d)
    return dataset_out(d)


# --------------------------------------------------------------------------------------------- plans
class StageOut(BaseModel):
    gpu_id: int
    gpu_name: str
    gpu_class: str
    device: str
    provider: str
    region: str
    layer_start: int
    layer_end: int
    memory_needed_mb: int
    free_memory_mb: int
    price_per_hour: float


class PlanOut(BaseModel):
    fits: bool
    reason: str = ""
    idle_gpus: int
    stages: list[StageOut] = []
    hourly_rate: float = 0.0
    machines: int = 0
    num_layers: int = 0


def build_plan(request: Request, session: Session, base_model_id: int, requested_gpus: int | None, trusted_only: bool,
               weight_bits: int = 16, method: str = "sft") -> PlanOut:
    base = session.get(BaseModelRow, base_model_id)
    if base is None:
        raise HTTPException(404, "Base model not found")
    timeout = request.app.state.settings.heartbeat_timeout_seconds
    eligible = scheduler.eligible_gpus(session, timeout, trusted_only, need_4bit=weight_bits == 4)
    spec = scheduler.model_spec(base, weight_bits)
    if method == "diloco":
        spec = scheduler.model_spec(base, full_ft=True)
    try:
        cands = [e.candidate() for e in eligible]
        plan = (planner.make_replica_plan(spec, cands, requested_gpus or planner.auto_replicas(spec, cands)) if method == "diloco"
                else planner.make_plan(spec, cands, requested_gpus))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    if plan is None:
        why = (f"Not enough idle GPUs for {requested_gpus} stages" if requested_gpus else "No set of idle GPUs can hold this model")
        if weight_bits == 4:
            why += " (4-bit needs GPUs whose agent supports it)"
        return PlanOut(fits=False, reason=why + " right now; the run would wait in the queue.", idle_gpus=len(eligible),
                       num_layers=base.num_layers)
    by_id = {e.gpu.id: e for e in eligible}
    stages = []
    for st in plan.stages:
        e = by_id[st.gpu_id]
        stages.append(StageOut(gpu_id=e.gpu.id, gpu_name=e.gpu.name, gpu_class=e.gpu_class.name, device=e.gpu.device,
                               provider=e.provider.name, region=e.gpu.region, layer_start=st.layer_start, layer_end=st.layer_end,
                               memory_needed_mb=planner.stage_memory_mb(spec, base.num_layers if method == "diloco" else st.layers),
                               free_memory_mb=e.gpu.free_memory_mb,
                               price_per_hour=e.gpu_class.price_per_hour))
    return PlanOut(fits=True, idle_gpus=len(eligible), stages=stages, hourly_rate=plan.hourly_rate, machines=plan.machines,
                   num_layers=base.num_layers)


FOUR_BIT_FAMILIES = ("llama",)  # BloomBee's 4-bit weight path is implemented for Llama-family blocks only


def check_weight_bits(base: BaseModelRow, weight_bits: int) -> None:
    if weight_bits == 4 and base.family not in FOUR_BIT_FAMILIES:
        raise HTTPException(422, f"4-bit base weights are not available for {base.family} models yet")


@router.get("/plans", response_model=PlanOut)
def preview_plan(request: Request, base_model_id: int, requested_gpus: int | None = None, trusted_only: bool = False,
                 weight_bits: int = 16, method: str = "sft",
                 _: User = Depends(current_user), session: Session = Depends(get_session)):
    """What a run would get right now (read-only; nothing is leased)."""
    if weight_bits not in (16, 4):  # a query string never matches Literal[16, 4], so check by hand
        raise HTTPException(422, "weight_bits must be 16 or 4")
    return build_plan(request, session, base_model_id, requested_gpus, trusted_only, weight_bits, method)


# --------------------------------------------------------------------------------------------- jobs
class JobIn(BaseModel):
    base_model_id: int
    dataset_id: int
    requested_gpus: int | None = Field(default=None, ge=1)  # None = automatic selection from the pool
    trusted_only: bool = False
    steps: int = Field(default=60, ge=1, le=500)
    batch_size: int = Field(default=2, ge=1, le=8)
    seq_len: int = Field(default=128, ge=32, le=512)
    lr: float = Field(default=2e-4, gt=0, le=0.01)
    rank: int = Field(default=8, ge=1, le=64)
    weight_bits: Literal[16, 4] = 16  # 4 = QLoRA: the frozen base model is kept in 4-bit on the workers
    method: Literal["sft", "dpo", "diloco"] = "sft"  # sft: LoRA on text; dpo: preference pairs; diloco: full fine-tuning
    inner_steps: int = Field(default=20, ge=1, le=1000)  # diloco: local steps per replica between synchronizations (H)
    outer_lr: float = Field(default=0.7, gt=0, le=2)  # diloco: Nesterov outer learning rate (DiLoCo paper: 0.7)
    outer_momentum: float = Field(default=0.9, ge=0, lt=1)
    beta: float = Field(default=0.1, gt=0, le=5)  # DPO strength (how far the policy may move from the reference)
    init_model_id: int | None = None  # continue from one of your fine-tuned models (e.g. DPO on top of an SFT model)
    name: str = Field(default="", max_length=80)


def init_adapter(session: Session, user: User, base: BaseModelRow, model_id: int | None) -> str:
    """The adapter directory of the user's fine-tuned model to continue from ("" = start fresh)."""
    if model_id is None:
        return ""
    m = session.get(FineTunedModel, model_id)
    if m is None or m.owner_id != user.id:
        raise HTTPException(404, "Model to continue from not found")
    if m.base_model_id != base.id:
        raise HTTPException(422, "That model was trained from a different base model")
    if not m.adapter_path or not (Path(m.adapter_path) / "adapter_config.json").exists():
        raise HTTPException(422, "That model has no stored adapter to continue from")
    return m.adapter_path


class LeaseOut(BaseModel):
    id: int
    gpu_id: int
    gpu_name: str
    gpu_class: str
    device: str
    layer_start: int
    layer_end: int
    active: bool
    end_reason: str
    peer_id: str


class JobOut(BaseModel):
    id: int
    kind: str
    status: str
    base_model: str
    requested_gpus: int | None
    trusted_only: bool
    config: dict
    created_at: str
    updated_at: str
    queue_position: int
    leases: list[LeaseOut]
    last_step: dict
    result: dict


def job_out(session: Session, job: Job) -> JobOut:
    base = session.get(BaseModelRow, job.base_model_id) if job.base_model_id else None
    leases = session.exec(select(Lease).where(Lease.job_id == job.id).order_by(Lease.id)).all()
    out = []
    for l in leases:
        gpu = session.get(Gpu, l.gpu_id)
        cls = session.get(GpuClass, gpu.gpu_class_id)
        out.append(LeaseOut(id=l.id, gpu_id=gpu.id, gpu_name=gpu.name, gpu_class=cls.name, device=gpu.device,
                            layer_start=l.layer_start, layer_end=l.layer_end, active=l.ended_at is None,
                            end_reason=l.end_reason, peer_id=l.worker_peer_id))
    result = json.loads(job.result_json or "{}")
    return JobOut(id=job.id, kind=job.kind, status=job.status, base_model=base.name if base else "",
                  requested_gpus=job.requested_gpus, trusted_only=job.trusted_only,
                  config={k: v for k, v in json.loads(job.config_json or "{}").items() if k not in ("dataset_path", "heldout_path", "model_path")},
                  created_at=iso(job.created_at), updated_at=iso(job.updated_at),
                  queue_position=scheduler.queue_position(session, job) if job.status == "queued" else 0,
                  leases=out, last_step=result.get("final", {}), result=result)


def owned_job(session: Session, job_id: int, user: User) -> Job:
    job = session.get(Job, job_id)
    if job is None or job.owner_id != user.id:
        raise HTTPException(404, "Job not found")
    return job


@router.post("/jobs", response_model=JobOut, status_code=201)
async def create_job(body: JobIn, request: Request, user: User = Depends(current_user), session: Session = Depends(get_session)):
    base = session.get(BaseModelRow, body.base_model_id)
    if base is None:
        raise HTTPException(404, "Base model not found")
    if body.requested_gpus is not None and body.requested_gpus > base.num_layers:
        raise HTTPException(422, f"requested_gpus cannot exceed the model's {base.num_layers} layers")
    check_weight_bits(base, body.weight_bits)
    dataset = visible_dataset(session, body.dataset_id, user)
    want = "preference" if body.method == "dpo" else "text"  # sft and diloco train on text
    if (dataset.format or "text") != want:
        raise HTTPException(422, "DPO needs a preference dataset (prompt, chosen, rejected per line)" if body.method == "dpo"
                            else "Supervised fine-tuning needs a text dataset ({\"text\": ...} per line); use DPO for preference pairs")
    if body.method == "diloco":
        if body.weight_bits != 16:
            raise HTTPException(422, "Full fine-tuning trains every weight: 4-bit base weights are only for LoRA (QLoRA)")
        if body.init_model_id is not None:
            raise HTTPException(422, "Full fine-tuning starts from a base model")
        if body.lr > 1e-3:
            raise HTTPException(422, "lr is too high for full fine-tuning (use about 1e-5 to 1e-4)")
    requested_gpus = body.requested_gpus
    if body.method == "diloco" and requested_gpus is None:
        # the replica count fixes how the data is sharded, so automatic selection is resolved once, now
        timeout = request.app.state.settings.heartbeat_timeout_seconds
        cands = [e.candidate() for e in scheduler.eligible_gpus(session, timeout, body.trusted_only)]
        requested_gpus = planner.auto_replicas(scheduler.model_spec(base, full_ft=True), cands)
    start = init_adapter(session, user, base, body.init_model_id)
    config = {"dataset_id": dataset.id, "dataset_path": dataset.path, "heldout_path": dataset.heldout_path, "steps": body.steps,
              "batch_size": body.batch_size, "seq_len": body.seq_len, "lr": body.lr, "rank": body.rank,
              "weight_bits": body.weight_bits, "method": body.method, "beta": body.beta, "init_adapter": start,
              "init_model_id": body.init_model_id, "inner_steps": body.inner_steps, "outer_lr": body.outer_lr,
              "outer_momentum": body.outer_momentum,
              "model_name": body.name.strip() or f"{base.name} {({'dpo': 'DPO', 'diloco': 'full fine-tune'}).get(body.method, 'fine-tuned')}"}
    job = Job(owner_id=user.id, kind="train", status="queued", base_model_id=base.id, requested_gpus=requested_gpus,
              trusted_only=body.trusted_only, dataset_path=dataset.path, config_json=json.dumps(config))
    session.add(job)
    session.commit()
    session.refresh(job)
    await request.app.state.orchestrator.submit(job.id)
    session.refresh(job)
    return job_out(session, job)


@router.get("/jobs", response_model=list[JobOut])
def list_jobs(user: User = Depends(current_user), session: Session = Depends(get_session)):
    rows = session.exec(select(Job).where(Job.owner_id == user.id, Job.kind == "train").order_by(Job.id.desc())).all()
    return [job_out(session, j) for j in rows]


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int, user: User = Depends(current_user), session: Session = Depends(get_session)):
    return job_out(session, owned_job(session, job_id, user))


class KillIn(BaseModel):
    layer_start: int


@router.post("/jobs/{job_id}/kill-worker")
async def kill_worker(job_id: int, body: KillIn, request: Request, user: User = Depends(current_user),
                      session: Session = Depends(get_session)):
    """Demo button: kill the worker that serves a layer range; the platform finds another GPU and recovers."""
    owned_job(session, job_id, user)
    if not await request.app.state.orchestrator.kill_worker(job_id, body.layer_start):
        raise HTTPException(409, "No running worker serves that layer range")
    return {"ok": True}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: int, request: Request, user: User = Depends(current_user), session: Session = Depends(get_session)):
    job = owned_job(session, job_id, user)
    if job.status == "queued":
        job.status = "stopped"
        job.updated_at = utcnow()
        session.add(job)
        session.commit()
        return {"ok": True}
    if not await request.app.state.orchestrator.cancel(job_id):
        raise HTTPException(409, "The job is not running")
    return {"ok": True}


# --------------------------------------------------------------------------------------------- models
class ModelOut(BaseModel):
    id: int
    name: str
    base_model: str
    base_model_id: int
    job_id: int | None
    created_at: str
    scores: dict
    deployed: bool
    weight_bits: int  # base-weight precision the model was trained with (4 = QLoRA)
    method: str  # sft | dpo | session (trained through the SDK)
    parent_model_id: int | None  # the model it continued from, if any


def model_out(session: Session, m: FineTunedModel) -> ModelOut:
    base = session.get(BaseModelRow, m.base_model_id)
    live = session.exec(select(Deployment).where(Deployment.finetuned_model_id == m.id,
                                                  Deployment.status.in_(["starting", "running"]))).first()
    job = session.get(Job, m.job_id) if m.job_id else None
    cfg = json.loads(job.config_json or "{}") if job else {}
    method = "session" if job is not None and job.kind == "session" else ("full" if cfg.get("method") == "diloco" else cfg.get("method", "sft"))
    return ModelOut(id=m.id, name=m.name, base_model=base.name, base_model_id=base.id, job_id=m.job_id,
                    created_at=iso(m.created_at), scores=json.loads(m.scores_json or "{}"), deployed=live is not None,
                    weight_bits=int(cfg.get("weight_bits", 16)), method=method, parent_model_id=cfg.get("init_model_id"))


@router.get("/models", response_model=list[ModelOut])
def my_models(user: User = Depends(current_user), session: Session = Depends(get_session)):
    rows = session.exec(select(FineTunedModel).where(FineTunedModel.owner_id == user.id).order_by(FineTunedModel.id.desc())).all()
    return [model_out(session, m) for m in rows]


@router.get("/models/{model_id}", response_model=ModelOut)
def get_model(model_id: int, user: User = Depends(current_user), session: Session = Depends(get_session)):
    m = session.get(FineTunedModel, model_id)
    if m is None or m.owner_id != user.id:
        raise HTTPException(404, "Model not found")
    return model_out(session, m)


@router.delete("/models/{model_id}", status_code=204)
def delete_model(model_id: int, user: User = Depends(current_user), session: Session = Depends(get_session)):
    m = session.get(FineTunedModel, model_id)
    if m is None or m.owner_id != user.id:
        raise HTTPException(404, "Model not found")
    if model_out(session, m).deployed:
        raise HTTPException(409, "Stop its deployment before deleting the model")
    import shutil
    for p in (m.adapter_path, m.merged_path):
        if p:
            shutil.rmtree(Path(p).parent if Path(p).name in ("adapter", "merged") else p, ignore_errors=True)
    session.delete(m)
    session.commit()
