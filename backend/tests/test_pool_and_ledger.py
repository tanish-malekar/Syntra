from __future__ import annotations

from datetime import timedelta

import pytest
from sqlmodel import select

from app.db import utcnow
from app.events import EventService
from app.models import Event, Gpu, GpuClass, Job, Lease, LedgerEntry, User
from app.seed import seed_reference_data
from app.services.ledger import record_lease_charge
from app.services.pool import classify_gpu, sweep_offline
from tests.conftest import make_admin, provider_with_agent, register_gpu, signup


@pytest.mark.parametrize(
    "name,device,expected",
    [
        ("Apple M4 Pro", "mps", "Apple M-series"),
        ("Apple M1", "mps", "Apple M-series"),
        ("NVIDIA RTX A6000", "cuda", "NVIDIA workstation (RTX A6000)"),
        ("NVIDIA GeForce RTX 4090", "cuda", "NVIDIA workstation (RTX A6000)"),
        ("NVIDIA A100-SXM4-80GB", "cuda", "NVIDIA data center"),
        ("NVIDIA H100 80GB HBM3", "cuda", "NVIDIA data center"),
        ("Tesla V100-PCIE-32GB", "cuda", "NVIDIA data center"),
        ("Mystery card", "cuda", "NVIDIA workstation (RTX A6000)"),
    ],
)
def test_classify_gpu(name, device, expected):
    assert classify_gpu(name, device) == expected


def test_seed_reference_data_is_idempotent(app, db):
    seed_reference_data(db)
    seed_reference_data(db)
    assert sorted(c.name for c in db.exec(select(GpuClass)).all()) == [
        "Apple M-series", "NVIDIA data center", "NVIDIA workstation (RTX A6000)"]
    prices = {c.name: c.price_per_hour for c in db.exec(select(GpuClass)).all()}
    assert prices["Apple M-series"] < prices["NVIDIA workstation (RTX A6000)"] < prices["NVIDIA data center"]


def test_base_models_endpoint(app):
    c = signup(app, "bm@example.com")
    models = c.get("/base-models").json()
    assert models and models[0]["name"] == "TinyLlama-1.1B-Chat" and models[0]["num_layers"] == 22


def test_stale_heartbeat_reads_offline_and_sweep_publishes_once(app, db):
    owner = signup(app, "st@example.com")
    _, agent, _ = provider_with_agent(app, owner)
    gpu = register_gpu(agent, "NVIDIA RTX A6000", "cuda", 48000)
    assert owner.get("/gpus").json()[0]["status"] == "idle"

    row = db.get(Gpu, gpu["id"])
    row.last_heartbeat = utcnow() - timedelta(seconds=30)
    db.add(row)
    db.commit()
    assert owner.get("/gpus").json()[0]["status"] == "offline"  # derived immediately

    events: EventService = app.state.events
    assert sweep_offline(db, events, utcnow(), 8.0) == [gpu["id"]]
    assert sweep_offline(db, events, utcnow(), 8.0) == []  # no duplicate transition
    offline = [e for e in db.exec(select(Event)).all() if e.type == "gpu.offline"]
    assert len(offline) == 1 and offline[0].gpu_id == gpu["id"]

    r = agent.post(f"/agent/gpus/{gpu['id']}/heartbeat", json={"free_memory_mb": 40000})
    assert r.json()["status"] == "idle"  # a returning heartbeat brings it back


def test_admin_can_edit_class_price_and_others_cannot(app):
    admin = signup(app, "boss@example.com")
    make_admin(app, "boss@example.com")
    user = signup(app, "plain@example.com")
    cls = user.get("/gpu-classes").json()[0]
    assert user.patch(f"/gpu-classes/{cls['id']}", json={"price_per_hour": 9}).status_code == 403
    assert admin.patch(f"/gpu-classes/{cls['id']}", json={"price_per_hour": 0.25}).json()["price_per_hour"] == 0.25
    assert admin.patch(f"/gpu-classes/{cls['id']}", json={"price_per_hour": -1}).status_code == 422
    assert user.get("/gpu-classes").json()[0]["price_per_hour"] == 0.25


def test_only_admin_can_verify_a_provider(app):
    admin = signup(app, "adm@example.com")
    make_admin(app, "adm@example.com")
    owner = signup(app, "dc@example.com")
    pid = owner.post("/providers", json={"name": "Big DC", "type": "enterprise"}).json()["id"]
    assert owner.patch(f"/providers/{pid}/trust", json={"trust_tier": "verified"}).status_code == 403
    assert admin.patch(f"/providers/{pid}/trust", json={"trust_tier": "verified"}).json()["trust_tier"] == "verified"
    assert admin.patch(f"/providers/{pid}/trust", json={"trust_tier": "gold"}).status_code == 422
    assert owner.get("/providers").json()[0]["trust_tier"] == "verified"


def _lease(db, provider_owner_email: str, consumer_email: str, gpu_id: int, seconds: float) -> Lease:
    consumer = db.exec(select(User).where(User.email == consumer_email)).one()
    job = Job(owner_id=consumer.id, kind="train", status="running")
    db.add(job)
    db.commit()
    db.refresh(job)
    start = utcnow() - timedelta(seconds=seconds)
    lease = Lease(job_id=job.id, gpu_id=gpu_id, layer_start=0, layer_end=11, started_at=start, ended_at=utcnow())
    db.add(lease)
    db.commit()
    db.refresh(lease)
    return lease


@pytest.mark.parametrize(
    "name,device,rate",
    [("Apple M4 Pro", "mps", 0.10), ("NVIDIA RTX A6000", "cuda", 0.45), ("NVIDIA H100 80GB HBM3", "cuda", 1.80)],
)
def test_ledger_prices_a_lease_by_gpu_class(app, db, name, device, rate):
    provider = signup(app, "prov@example.com")
    consumer = signup(app, "cons@example.com")
    _, agent, _ = provider_with_agent(app, provider)
    gpu = register_gpu(agent, name, device, 48000)
    lease = _lease(db, "prov@example.com", "cons@example.com", gpu["id"], 1800)

    credit, cost = record_lease_charge(db, app.state.events, lease, gpu_seconds=1800)
    expected = round(1800 / 3600 * rate, 6)
    assert credit.amount == cost.amount == expected and credit.rate_per_hour == rate

    earnings = provider.get("/ledger/earnings").json()
    assert earnings["total"] == expected and earnings["entries"][0]["gpu_class"] == gpu["gpu_class"]
    assert provider.get("/ledger/costs").json()["total"] == 0  # the provider paid nothing
    assert consumer.get("/ledger/costs").json()["total"] == expected
    assert consumer.get("/ledger/earnings").json()["total"] == 0  # the consumer earned nothing


def test_ledger_uses_lease_duration_and_snapshots_the_rate(app, db):
    admin = signup(app, "adm2@example.com")
    make_admin(app, "adm2@example.com")
    provider = signup(app, "p2@example.com")
    signup(app, "c2@example.com")
    _, agent, _ = provider_with_agent(app, provider)
    gpu = register_gpu(agent, "NVIDIA RTX A6000", "cuda", 48000)

    lease = _lease(db, "p2@example.com", "c2@example.com", gpu["id"], 7200)
    first, _ = record_lease_charge(db, app.state.events, lease)  # duration taken from the lease
    assert round(first.gpu_seconds) == 7200 and first.amount == pytest.approx(0.9, abs=0.01)

    cls = admin.get("/gpu-classes").json()
    class_id = next(c["id"] for c in cls if c["name"].startswith("NVIDIA workstation"))
    admin.patch(f"/gpu-classes/{class_id}", json={"price_per_hour": 1.0})
    lease2 = _lease(db, "p2@example.com", "c2@example.com", gpu["id"], 3600)
    second, _ = record_lease_charge(db, app.state.events, lease2)
    assert second.amount == pytest.approx(1.0, abs=0.01)
    assert db.get(LedgerEntry, first.id).rate_per_hour == 0.45  # old entry keeps the rate it was billed at
    assert provider.get("/ledger/earnings").json()["total"] == pytest.approx(first.amount + second.amount, abs=1e-5)


def test_ledger_credit_event_goes_only_to_the_provider_owner(app, db):
    provider = signup(app, "pe@example.com")
    consumer = signup(app, "ce@example.com")
    _, agent, _ = provider_with_agent(app, provider)
    gpu = register_gpu(agent, "Apple M4 Pro", "mps", 24000)
    lease = _lease(db, "pe@example.com", "ce@example.com", gpu["id"], 600)
    record_lease_charge(db, app.state.events, lease)
    me = provider.get("/auth/me").json()["id"]
    assert [e["type"] for e in provider.get("/events").json()] == ["ledger.credit"]
    assert consumer.get("/events").json() == []
    stored = db.exec(select(Event).where(Event.type == "ledger.credit")).one()
    assert stored.user_id == me
