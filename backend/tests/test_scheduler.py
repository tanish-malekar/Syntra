from __future__ import annotations

import threading
from datetime import timedelta

import pytest
from sqlmodel import select

from app.db import utcnow
from app.models import AgentToken, BaseModel, Event, Gpu, Job, Lease, LedgerEntry, Provider, User
from app.security import hash_password, new_token, token_hash
from app.services import scheduler
from app.services.pool import classify_gpu, get_class

TIMEOUT = 8.0


class World:
    """Builds users, providers, machines (agent tokens) and GPUs straight in the database."""

    def __init__(self, app, db):
        self.app, self.db, self.ev = app, db, app.state.events
        self.base = db.exec(select(BaseModel)).one()
        self._n = 0

    def user(self, email=None) -> User:
        self._n += 1
        u = User(email=email or f"u{self._n}@example.com", password_hash=hash_password("x" * 12))
        self.db.add(u); self.db.commit(); self.db.refresh(u)
        return u

    def provider(self, owner: User, trust="standard", type="individual") -> Provider:
        p = Provider(owner_id=owner.id, name=f"prov{owner.id}", type=type, trust_tier=trust)
        self.db.add(p); self.db.commit(); self.db.refresh(p)
        return p

    def machine(self, provider: Provider) -> AgentToken:
        t = AgentToken(provider_id=provider.id, token_hash=token_hash(new_token()))
        self.db.add(t); self.db.commit(); self.db.refresh(t)
        return t

    def gpu(self, provider, token=None, name="Apple M4 Pro", device="mps", total=6000, free=None, index=0,
            status="idle", heartbeat=None, reliability=1.0) -> Gpu:
        cls = get_class(self.db, classify_gpu(name, device))
        g = Gpu(provider_id=provider.id, gpu_class_id=cls.id, agent_token_id=token.id if token else None, name=name,
                device=device, device_index=index, total_memory_mb=total, free_memory_mb=total if free is None else free,
                status=status, last_heartbeat=heartbeat or utcnow(), reliability=reliability)
        self.db.add(g); self.db.commit(); self.db.refresh(g)
        return g

    def job(self, owner: User, requested=None, trusted_only=False, kind="train") -> Job:
        j = Job(owner_id=owner.id, kind=kind, base_model_id=self.base.id, requested_gpus=requested, trusted_only=trusted_only)
        self.db.add(j); self.db.commit(); self.db.refresh(j)
        return j

    def leases(self, job: Job, active_only=True):
        q = select(Lease).where(Lease.job_id == job.id).order_by(Lease.layer_start)
        rows = self.db.exec(q).all()
        return [l for l in rows if l.ended_at is None] if active_only else rows


@pytest.fixture
def w(app, db):
    return World(app, db)


def start(w: World, job: Job):
    return scheduler.try_start(w.db, w.ev, job, TIMEOUT)


def test_auto_job_takes_one_big_gpu_and_marks_it_leased(w):
    owner, prov = w.user(), w.provider(w.user())
    g = w.gpu(prov, w.machine(prov), "NVIDIA RTX A6000", "cuda", total=48000)
    job = w.job(owner)
    leases = start(w, job)
    assert [(l.gpu_id, l.layer_start, l.layer_end) for l in leases] == [(g.id, 0, 22)]
    w.db.refresh(g); w.db.refresh(job)
    assert g.status == "leased" and job.status == "starting"
    assert [e.type for e in w.db.exec(select(Event)).all()] == ["lease.created"]


def test_requested_gpu_count_splits_a_small_model_across_gpus(w):
    owner, prov = w.user(), w.provider(w.user())
    tok = w.machine(prov)
    for i in range(3):
        w.gpu(prov, tok, "NVIDIA RTX A6000", "cuda", total=48000, index=i)
    job = w.job(owner, requested=2)
    leases = start(w, job)
    assert [(l.layer_start, l.layer_end) for l in leases] == [(0, 11), (11, 22)]
    assert len({l.gpu_id for l in leases}) == 2


def test_gpus_that_are_offline_busy_leased_or_untrusted_are_skipped(w):
    owner = w.user()
    prov = w.provider(w.user())
    verified = w.provider(w.user(), trust="verified", type="enterprise")
    tok = w.machine(prov)
    stale = w.gpu(prov, tok, index=0, heartbeat=utcnow() - timedelta(seconds=60))
    busy = w.gpu(prov, tok, index=1, status="busy_external")
    taken = w.gpu(prov, tok, index=2)
    other_job = w.job(w.user())
    w.db.add(Lease(job_id=other_job.id, gpu_id=taken.id, layer_start=0, layer_end=22)); w.db.commit()
    ok = w.gpu(prov, tok, index=3)
    elig = {e.gpu.id for e in scheduler.eligible_gpus(w.db, TIMEOUT)}
    assert elig == {ok.id}
    trusted = w.gpu(verified, w.machine(verified), "NVIDIA A100-SXM4-80GB", "cuda", total=80000)
    assert {e.gpu.id for e in scheduler.eligible_gpus(w.db, TIMEOUT, trusted_only=True)} == {trusted.id}
    job = w.job(owner, trusted_only=True)
    assert [l.gpu_id for l in start(w, job)] == [trusted.id]


def test_job_waits_in_a_fifo_queue_and_starts_when_gpus_free_up(w):
    a, b, c = w.user(), w.user(), w.user()
    prov = w.provider(w.user())
    tok = w.machine(prov)
    g1 = w.gpu(prov, tok, "NVIDIA RTX A6000", "cuda", total=48000, index=0)
    g2 = w.gpu(prov, tok, "NVIDIA RTX A6000", "cuda", total=48000, index=1)
    j1, j2, j3 = w.job(a, requested=2), w.job(b, requested=2), w.job(c, requested=1)
    assert scheduler.submit(w.db, w.ev, j1, TIMEOUT) is not None
    assert scheduler.submit(w.db, w.ev, j2, TIMEOUT) is None     # no idle GPUs left
    assert scheduler.submit(w.db, w.ev, j3, TIMEOUT) is None
    w.db.refresh(j2); w.db.refresh(j3)
    assert (j2.status, j3.status) == ("queued", "queued")
    assert (scheduler.queue_position(w.db, j2), scheduler.queue_position(w.db, j3)) == (1, 2)
    queued_events = [e for e in w.db.exec(select(Event)).all() if e.type == "job.queued"]
    assert [e.user_id for e in queued_events] == [b.id, c.id]    # each owner is told their own position

    scheduler.release_job(w.db, w.ev, j1, "completed")
    started = scheduler.dispatch_queue(w.db, w.ev, TIMEOUT)
    assert [j.id for j, _ in started] == [j2.id] and len(started[0][1]) == 2
    w.db.refresh(j3)
    assert j3.status == "queued" and scheduler.queue_position(w.db, j3) == 1


def test_a_small_later_job_backfills_while_a_big_one_waits(w):
    a, b = w.user(), w.user()
    prov = w.provider(w.user()); tok = w.machine(prov)
    w.gpu(prov, tok, "NVIDIA RTX A6000", "cuda", total=48000)
    big, small = w.job(a, requested=3), w.job(b, requested=1)
    assert scheduler.submit(w.db, w.ev, big, TIMEOUT) is None
    assert scheduler.submit(w.db, w.ev, small, TIMEOUT) is not None  # fits right now, does not wait behind the big one
    w.db.refresh(big)
    assert big.status == "queued"


def test_release_returns_the_gpu_and_bills_both_sides(w):
    consumer, owner = w.user(), w.user()
    prov = w.provider(owner)
    g = w.gpu(prov, w.machine(prov), "NVIDIA RTX A6000", "cuda", total=48000)
    job = w.job(consumer)
    (lease,) = start(w, job)
    lease.started_at = utcnow() - timedelta(minutes=30)
    w.db.add(lease); w.db.commit()
    assert scheduler.release_job(w.db, w.ev, job, "completed") == 1
    w.db.refresh(g)
    assert g.status == "idle"
    entries = w.db.exec(select(LedgerEntry)).all()
    assert sorted(e.kind for e in entries) == ["cost", "credit"] and entries[0].amount == pytest.approx(0.225, abs=0.01)
    assert {e.user_id for e in entries} == {owner.id, consumer.id}
    assert scheduler.release_job(w.db, w.ev, job, "again") == 0  # idempotent


def test_release_does_not_clear_busy_external(w):
    consumer = w.user(); prov = w.provider(w.user())
    g = w.gpu(prov, w.machine(prov), total=48000, device="cuda", name="NVIDIA RTX A6000")
    job = w.job(consumer)
    (lease,) = start(w, job)
    g.status = "busy_external"; w.db.add(g); w.db.commit()   # the owner started their own workload meanwhile
    scheduler.release_lease(w.db, w.ev, lease, "completed", charge=False)
    w.db.refresh(g)
    assert g.status == "busy_external"


def test_two_jobs_racing_for_the_same_gpus_never_share_one(app, w):
    prov = w.provider(w.user()); tok = w.machine(prov)
    for i in range(2):
        w.gpu(prov, tok, "NVIDIA RTX A6000", "cuda", total=48000, index=i)
    jobs = [w.job(w.user(), requested=2) for _ in range(4)]
    results = {}

    def attempt(job_id):
        from sqlmodel import Session
        with Session(app.state.engine) as s:
            job = s.get(Job, job_id)
            leases = scheduler.try_start(s, app.state.events, job, TIMEOUT)
            results[job_id] = None if leases is None else [l.gpu_id for l in leases]

    threads = [threading.Thread(target=attempt, args=(j.id,)) for j in jobs]
    [t.start() for t in threads]; [t.join() for t in threads]
    winners = [r for r in results.values() if r]
    assert len(winners) == 1                                    # exactly one job got the two GPUs
    active = w.db.exec(select(Lease).where(Lease.ended_at == None)).all()  # noqa: E711
    assert len(active) == 2 and len({l.gpu_id for l in active}) == 2


def test_replacement_takes_over_the_same_layers_on_another_idle_gpu(w):
    owner = w.user(); prov = w.provider(w.user()); tok = w.machine(prov)
    g1 = w.gpu(prov, tok, "NVIDIA RTX A6000", "cuda", total=48000, index=0)
    g2 = w.gpu(prov, tok, "NVIDIA RTX A6000", "cuda", total=48000, index=1)
    spare = w.gpu(prov, w.machine(prov), "Apple M4 Pro", "mps", total=24000, index=0)
    job = w.job(owner, requested=2)
    leases = start(w, job)
    dead = leases[1]
    new = scheduler.lease_replacement(w.db, w.ev, job, dead, TIMEOUT, reason="killed_by_user")
    w.db.refresh(dead)
    assert dead.ended_at is not None and dead.end_reason == "killed_by_user"
    assert new.gpu_id == spare.id and (new.layer_start, new.layer_end) == (dead.layer_start, dead.layer_end)
    assert {l.gpu_id for l in w.leases(job)} == {leases[0].gpu_id, spare.id}   # never reuses a working GPU
    ev = next(e for e in w.db.exec(select(Event)).all() if e.type == "worker.replacement_starting")
    assert ev.user_id == owner.id and '"gpu_device": "mps"' in ev.payload_json and "Apple M-series" in ev.payload_json


def test_replacement_returns_none_when_nothing_idle_fits_and_keeps_the_gap_closed(w):
    owner = w.user(); prov = w.provider(w.user()); tok = w.machine(prov)
    for i in range(2):
        w.gpu(prov, tok, "NVIDIA RTX A6000", "cuda", total=48000, index=i)
    job = w.job(owner, requested=2)
    leases = start(w, job)
    assert scheduler.lease_replacement(w.db, w.ev, job, leases[0], TIMEOUT) is None
    assert len(w.leases(job)) == 1                    # the dead lease is ended and billed, none invented
    w.gpu(prov, tok, "NVIDIA RTX A6000", "cuda", total=48000, index=5)  # a GPU joins the pool later
    retry = scheduler.lease_replacement(w.db, w.ev, job, leases[0], TIMEOUT)
    assert retry is not None and (retry.layer_start, retry.layer_end) == (0, 11)


def test_replacement_requires_enough_memory(w):
    owner = w.user(); prov = w.provider(w.user()); tok = w.machine(prov)
    big = [w.gpu(prov, tok, "NVIDIA RTX A6000", "cuda", total=48000, index=i) for i in range(2)]
    w.gpu(prov, w.machine(prov), "Apple M1", "mps", total=700, index=9)  # separate laptop, too small for 11 layers (1600 MB)
    job = w.job(owner, requested=2)
    leases = start(w, job)
    assert scheduler.lease_replacement(w.db, w.ev, job, leases[1], TIMEOUT) is None
