"""Database tables (SQLModel). See the platform plan, "Data model"."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel

from app.db import utcnow

# ---- string "enums" kept as plain constants so SQLite stays simple -------------------------------
PROVIDER_TYPES = ("individual", "enterprise")
TRUST_TIERS = ("standard", "verified")
GPU_STATUSES = ("idle", "leased", "busy_external", "offline")
DEVICES = ("mps", "cuda")
JOB_KINDS = ("train", "deploy")
JOB_STATUSES = ("queued", "starting", "running", "completed", "failed", "stopped")
LEDGER_KINDS = ("credit", "cost")


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    password_hash: str
    display_name: str = ""
    is_admin: bool = False
    created_at: datetime = Field(default_factory=utcnow)


class AuthSession(SQLModel, table=True):
    """A login session; only the SHA-256 of the cookie token is stored."""

    id: Optional[int] = Field(default=None, primary_key=True)
    token_hash: str = Field(index=True, unique=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime
    revoked: bool = False


class Provider(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    owner_id: int = Field(foreign_key="user.id", index=True)
    name: str
    type: str = "individual"  # individual | enterprise
    trust_tier: str = "standard"  # standard | verified (admin-set)
    created_at: datetime = Field(default_factory=utcnow)


class AgentToken(SQLModel, table=True):
    """One per machine; plaintext is shown once at creation and never stored."""

    id: Optional[int] = Field(default=None, primary_key=True)
    provider_id: int = Field(foreign_key="provider.id", index=True)
    token_hash: str = Field(index=True, unique=True)
    label: str = ""
    machine_id: str = Field(default="", index=True)  # set when paired from the desktop app: one live token per machine
    created_at: datetime = Field(default_factory=utcnow)
    last_used_at: Optional[datetime] = None
    revoked: bool = False


class GpuClass(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    price_per_hour: float  # simulated currency per GPU-hour
    device_hint: str = ""  # mps | cuda


class ApiKey(SQLModel, table=True):
    """Personal API key for the SDK (`Authorization: Bearer syn_key_...`); plaintext shown once, only the hash is kept."""

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    key_hash: str = Field(index=True, unique=True)
    label: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    last_used_at: Optional[datetime] = None
    revoked: bool = False


class Gpu(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    provider_id: int = Field(foreign_key="provider.id", index=True)
    gpu_class_id: int = Field(foreign_key="gpuclass.id")
    agent_token_id: Optional[int] = Field(default=None, foreign_key="agenttoken.id")
    name: str  # chip / card name as reported by the agent
    device: str = "cuda"  # mps | cuda
    device_index: int = 0
    total_memory_mb: int = 0
    free_memory_mb: int = 0
    region: str = ""
    status: str = "idle"  # idle | leased | busy_external | offline
    last_heartbeat: Optional[datetime] = None
    reliability: float = 1.0
    simulated: bool = False
    supports_4bit: bool = False  # the agent's BloomBee can keep base weights in 4-bit (--compress_weight)
    created_at: datetime = Field(default_factory=utcnow)


class BaseModel(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    slug: str = ""  # short directory name used for local model files, e.g. "tinyllama"
    hf_id: str
    local_path: str = ""
    family: str = "llama"
    num_layers: int
    hidden_size: int
    per_layer_memory_mb: int  # weights + working memory per layer, from measurements


class Dataset(SQLModel, table=True):
    """Training data: the built-in demo set (owner None) or a user's uploaded JSONL ({"text": ...} per line)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    owner_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
    name: str
    path: str
    heldout_path: str = ""
    num_rows: int = 0
    builtin: bool = False
    format: str = "text"  # text: {"text"} per line (SFT) | preference: {"prompt", "chosen", "rejected"} per line (DPO)
    created_at: datetime = Field(default_factory=utcnow)


class FineTunedModel(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    owner_id: int = Field(foreign_key="user.id", index=True)
    base_model_id: int = Field(foreign_key="basemodel.id")
    job_id: Optional[int] = Field(default=None, foreign_key="job.id")
    name: str
    adapter_path: str = ""
    merged_path: str = ""  # filled on first deploy
    scores_json: str = "{}"  # held-out evaluation
    created_at: datetime = Field(default_factory=utcnow)


class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    owner_id: int = Field(foreign_key="user.id", index=True)
    kind: str = "train"  # train | deploy
    status: str = "queued"
    base_model_id: Optional[int] = Field(default=None, foreign_key="basemodel.id")
    requested_gpus: Optional[int] = None  # None = automatic
    trusted_only: bool = False
    dataset_path: str = ""
    dht_prefix: str = ""
    config_json: str = "{}"  # steps, batch size, dataset, ...
    result_json: str = "{}"
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Deployment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    owner_id: int = Field(foreign_key="user.id", index=True)
    model_kind: str = "finetuned"  # base | finetuned
    base_model_id: Optional[int] = Field(default=None, foreign_key="basemodel.id")
    finetuned_model_id: Optional[int] = Field(default=None, foreign_key="finetunedmodel.id")
    job_id: Optional[int] = Field(default=None, foreign_key="job.id")
    num_gpus: int = 1
    status: str = "starting"
    endpoint: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class Lease(SQLModel, table=True):
    """Exclusive claim of one GPU by one job for a layer range."""

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job.id", index=True)
    gpu_id: int = Field(foreign_key="gpu.id", index=True)
    layer_start: int = 0
    layer_end: int = 0
    worker_peer_id: str = ""
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: Optional[datetime] = None
    end_reason: str = ""


class Event(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=utcnow, index=True)
    type: str = Field(index=True)
    job_id: Optional[int] = Field(default=None, index=True)
    gpu_id: Optional[int] = Field(default=None, index=True)
    user_id: Optional[int] = Field(default=None, index=True)  # owner who may see it; None = pool-level
    payload_json: str = "{}"


class LedgerEntry(SQLModel, table=True):
    """Simulated money only. A lease produces one credit (provider owner) and one cost (consumer)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    kind: str  # credit | cost
    user_id: int = Field(foreign_key="user.id", index=True)  # whose ledger line this is
    provider_id: Optional[int] = Field(default=None, foreign_key="provider.id")
    job_id: Optional[int] = Field(default=None, foreign_key="job.id")
    lease_id: Optional[int] = Field(default=None, foreign_key="lease.id")
    gpu_id: Optional[int] = Field(default=None, foreign_key="gpu.id")
    gpu_class_id: Optional[int] = Field(default=None, foreign_key="gpuclass.id")
    gpu_seconds: float = 0.0
    rate_per_hour: float = 0.0
    amount: float = 0.0
    created_at: datetime = Field(default_factory=utcnow)
