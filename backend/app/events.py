"""Event types, persistence and the live fan-out used by /ws/events.

The same list of types is mirrored in `frontend/src/events.ts`; tests/test_events.py checks they agree.
Visibility: `gpu.*` events are pool-level (every logged-in user and every agent); everything else belongs to
one user (`Event.user_id`) and is delivered only to that user.
"""
from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from enum import Enum

from sqlmodel import Session

from app.db import iso
from app.models import Event


class EventType(str, Enum):
    GPU_REGISTERED = "gpu.registered"
    GPU_HEARTBEAT = "gpu.heartbeat"
    GPU_OFFLINE = "gpu.offline"
    GPU_BUSY_EXTERNAL = "gpu.busy_external"
    JOB_QUEUED = "job.queued"
    LEASE_CREATED = "lease.created"
    WORKER_STARTING = "worker.starting"
    WORKER_ONLINE = "worker.online"
    WORKER_DOWN = "worker.down"
    WORKER_REPLACEMENT_WAITING = "worker.replacement_waiting"
    WORKER_REPLACEMENT_STARTING = "worker.replacement_starting"
    WORKER_REPLACEMENT_ONLINE = "worker.replacement_online"
    TRAIN_STEP = "train.step"
    TRAIN_COMPLETED = "train.completed"
    TRAIN_FAILED = "train.failed"
    DEPLOY_MERGING = "deploy.merging"
    DEPLOY_ONLINE = "deploy.online"
    DEPLOY_STOPPED = "deploy.stopped"
    EVAL_RESULT = "eval.result"
    LEDGER_CREDIT = "ledger.credit"
    SESSION_READY = "session.ready"
    SESSION_SAVED = "session.saved"
    SESSION_CLOSED = "session.closed"


# Required payload keys per event type (extra keys are allowed).
PAYLOAD_KEYS: dict[EventType, tuple[str, ...]] = {
    EventType.GPU_REGISTERED: ("name", "device", "gpu_class", "total_memory_mb", "provider"),
    EventType.GPU_HEARTBEAT: ("status", "free_memory_mb"),
    EventType.GPU_OFFLINE: ("reason",),
    EventType.GPU_BUSY_EXTERNAL: ("detail",),
    EventType.JOB_QUEUED: ("position",),
    EventType.LEASE_CREATED: ("layer_start", "layer_end", "gpu_name"),
    EventType.WORKER_STARTING: ("layer_start", "layer_end"),
    EventType.WORKER_ONLINE: ("layer_start", "layer_end", "peer_id"),
    EventType.WORKER_DOWN: ("reason", "layer_start", "layer_end"),  # reason: exited | heartbeat_timeout | killed_by_user
    EventType.WORKER_REPLACEMENT_WAITING: ("layer_start", "layer_end"),  # no idle GPU can take over yet
    EventType.WORKER_REPLACEMENT_STARTING: (
        "layer_start", "layer_end", "gpu_name", "gpu_device", "gpu_class", "gpu_memory_mb",
    ),
    EventType.WORKER_REPLACEMENT_ONLINE: ("layer_start", "layer_end", "peer_id", "seconds_since_down"),
    EventType.TRAIN_STEP: ("step", "loss", "grad_norm", "step_seconds"),
    EventType.TRAIN_COMPLETED: ("model_id",),
    EventType.TRAIN_FAILED: ("error",),
    EventType.DEPLOY_MERGING: (),
    EventType.DEPLOY_ONLINE: ("endpoint", "num_gpus"),
    EventType.DEPLOY_STOPPED: ("reason",),
    EventType.EVAL_RESULT: ("model_label", "n", "valid_json", "dept_correct", "urgency_correct"),
    EventType.LEDGER_CREDIT: ("provider_id", "amount", "gpu_seconds"),
    EventType.SESSION_READY: ("trainable_parameters",),
    EventType.SESSION_SAVED: ("model_id", "name"),
    EventType.SESSION_CLOSED: ("reason",),
}
assert set(PAYLOAD_KEYS) == set(EventType)

WORKER_DOWN_REASONS = ("exited", "heartbeat_timeout", "killed_by_user", "failed_to_start")


def is_pool_level(event_type: str) -> bool:
    return event_type.startswith("gpu.")


def event_to_dict(ev: Event) -> dict:
    return {
        "id": ev.id,
        "type": ev.type,
        "ts": iso(ev.ts),
        "job_id": ev.job_id,
        "gpu_id": ev.gpu_id,
        "payload": json.loads(ev.payload_json or "{}"),
    }


@dataclass(eq=False)
class Subscriber:
    queue: asyncio.Queue
    loop: asyncio.AbstractEventLoop
    user_id: int | None = None  # None for agents
    is_agent: bool = False


@dataclass
class EventBus:
    _subs: set = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def subscribe(self, user_id: int | None, is_agent: bool = False) -> Subscriber:
        """Call from the event loop that will read the queue."""
        sub = Subscriber(asyncio.Queue(), asyncio.get_running_loop(), user_id, is_agent)
        with self._lock:
            self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        with self._lock:
            self._subs.discard(sub)

    def _visible(self, sub: Subscriber, event_type: str, owner_id: int | None) -> bool:
        if is_pool_level(event_type):
            return True
        return (not sub.is_agent) and owner_id is not None and sub.user_id == owner_id

    def broadcast(self, event: dict, owner_id: int | None) -> None:
        """Thread-safe: request handlers run in a thread pool, subscribers live on the event loop."""
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            if not self._visible(sub, event["type"], owner_id):
                continue
            try:
                sub.loop.call_soon_threadsafe(sub.queue.put_nowait, event)
            except RuntimeError:  # loop closed
                self.unsubscribe(sub)


class EventService:
    def __init__(self, bus: EventBus | None = None):
        self.bus = bus or EventBus()

    def publish(
        self,
        session: Session,
        type: EventType | str,
        *,
        job_id: int | None = None,
        gpu_id: int | None = None,
        user_id: int | None = None,
        payload: dict | None = None,
        persist: bool = True,
    ) -> Event:
        """Validate, persist (commits the caller's session) and broadcast one event.

        `persist=False` broadcasts only (used for high-frequency gpu.heartbeat); the returned Event has no id.
        """
        etype = EventType(type)
        payload = payload or {}
        missing = [k for k in PAYLOAD_KEYS[etype] if k not in payload]
        if missing:
            raise ValueError(f"event {etype.value} is missing payload keys: {missing}")
        if etype is EventType.WORKER_DOWN and payload["reason"] not in WORKER_DOWN_REASONS:
            raise ValueError(f"worker.down reason must be one of {WORKER_DOWN_REASONS}")
        if not is_pool_level(etype.value) and user_id is None:
            raise ValueError(f"event {etype.value} needs the owning user_id")
        ev = Event(type=etype.value, job_id=job_id, gpu_id=gpu_id, user_id=user_id, payload_json=json.dumps(payload))
        if persist:
            session.add(ev)
            session.commit()
            session.refresh(ev)
        self.bus.broadcast(event_to_dict(ev), user_id)
        return ev
