from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from app.models import AgentToken, Event, Gpu
from tests.conftest import PASSWORD, provider_with_agent, register_gpu, signup


def test_provider_types_and_validation(app):
    c = signup(app, "p@example.com")
    assert c.post("/providers", json={"name": "Home", "type": "individual"}).status_code == 201
    dc = c.post("/providers", json={"name": "Acme DC", "type": "enterprise"})
    assert dc.status_code == 201 and dc.json()["trust_tier"] == "standard"
    assert c.post("/providers", json={"name": "x", "type": "spaceship"}).status_code == 422
    assert c.post("/providers", json={"name": "", "type": "individual"}).status_code == 422
    assert [p["name"] for p in c.get("/providers").json()] == ["Home", "Acme DC"]


def test_agent_token_is_shown_once_and_stored_hashed(app, db):
    c = signup(app, "t@example.com")
    pid = c.post("/providers", json={"name": "P"}).json()["id"]
    created = c.post(f"/providers/{pid}/agent-tokens", json={"label": "mac"}).json()
    token = created["token"]
    assert token.startswith("syn_agent_")
    listing = c.get(f"/providers/{pid}/agent-tokens").json()
    assert listing and "token" not in listing[0]
    assert all(r.token_hash != token for r in db.exec(select(AgentToken)).all())


def test_agent_token_authentication(app):
    c = signup(app, "a@example.com")
    pid, agent, token = provider_with_agent(app, c, "Lab", "enterprise")
    me = agent.get("/agent/me")
    assert me.status_code == 200 and me.json()["provider_name"] == "Lab" and me.json()["provider_type"] == "enterprise"
    for headers in ({}, {"Authorization": "Bearer nope"}, {"Authorization": token}):  # missing, wrong, no scheme
        assert TestClient(app).get("/agent/me", headers=headers).status_code == 401
    # a user's password-session cookie is not an agent credential
    assert c.get("/agent/me").status_code == 401


def test_revoked_agent_token_is_rejected(app):
    c = signup(app, "r@example.com")
    pid, agent, _ = provider_with_agent(app, c)
    token_id = c.get(f"/providers/{pid}/agent-tokens").json()[0]["id"]
    assert agent.get("/agent/me").status_code == 200
    assert c.delete(f"/providers/{pid}/agent-tokens/{token_id}").status_code == 204
    assert agent.get("/agent/me").status_code == 401
    assert agent.post("/agent/gpus", json={"name": "x", "device": "cuda", "total_memory_mb": 1}).status_code == 401
    assert c.get(f"/providers/{pid}/agent-tokens").json()[0]["revoked"] is True


def test_agent_token_records_last_use(app):
    c = signup(app, "u@example.com")
    pid, agent, _ = provider_with_agent(app, c)
    assert c.get(f"/providers/{pid}/agent-tokens").json()[0]["last_used_at"] is None
    agent.get("/agent/me")
    assert c.get(f"/providers/{pid}/agent-tokens").json()[0]["last_used_at"] is not None


def test_register_gpu_classifies_and_joins_the_pool(app):
    owner = signup(app, "g@example.com")
    _, agent, _ = provider_with_agent(app, owner, "Mac Lab")
    mac = register_gpu(agent, "Apple M4 Pro", "mps", 24000)
    assert mac["gpu_class"] == "Apple M-series" and mac["price_per_hour"] == 0.10
    assert mac["status"] == "idle" and mac["free_memory_mb"] == 24000
    other = signup(app, "viewer@example.com")
    pool = other.get("/gpus").json()  # the pool is visible to every signed-in user
    assert [g["name"] for g in pool] == ["Apple M4 Pro"] and pool[0]["provider_name"] == "Mac Lab"


def test_register_is_idempotent_per_provider_name_and_index(app, db):
    owner = signup(app, "i@example.com")
    _, agent, _ = provider_with_agent(app, owner)
    a = register_gpu(agent, "NVIDIA RTX A6000", "cuda", 48000, index=0)
    b = register_gpu(agent, "NVIDIA RTX A6000", "cuda", 48000, index=0)
    c = register_gpu(agent, "NVIDIA RTX A6000", "cuda", 48000, index=2)
    assert a["id"] == b["id"] != c["id"]
    assert len(db.exec(select(Gpu)).all()) == 2


def test_register_validation(app):
    owner = signup(app, "v@example.com")
    _, agent, _ = provider_with_agent(app, owner)
    assert agent.post("/agent/gpus", json={"name": "x", "device": "tpu", "total_memory_mb": 1}).status_code == 422
    assert agent.post("/agent/gpus", json={"name": "x", "device": "cuda", "total_memory_mb": -5}).status_code == 422


def test_heartbeat_updates_memory_and_status(app):
    owner = signup(app, "h@example.com")
    _, agent, _ = provider_with_agent(app, owner)
    gpu = register_gpu(agent, "NVIDIA RTX A6000", "cuda", 48000)
    r = agent.post(f"/agent/gpus/{gpu['id']}/heartbeat", json={"free_memory_mb": 1000, "status": "busy_external",
                                                               "detail": "someone else's job"})
    assert r.status_code == 200 and r.json()["status"] == "busy_external" and r.json()["free_memory_mb"] == 1000
    bad = agent.post(f"/agent/gpus/{gpu['id']}/heartbeat", json={"free_memory_mb": 1, "status": "exploding"})
    assert bad.status_code == 422
    # an agent may say goodbye (laptop lid closed): offline at once, and back to idle on its next normal heartbeat
    bye = agent.post(f"/agent/gpus/{gpu['id']}/heartbeat", json={"free_memory_mb": 0, "status": "offline", "detail": "lid closed"})
    assert bye.status_code == 200 and bye.json()["status"] == "offline"
    back = agent.post(f"/agent/gpus/{gpu['id']}/heartbeat", json={"free_memory_mb": 1000, "status": "idle"})
    assert back.json()["status"] == "idle"


def test_heartbeat_for_another_providers_gpu_is_404(app):
    a, b = signup(app, "a2@example.com"), signup(app, "b2@example.com")
    _, agent_a, _ = provider_with_agent(app, a)
    _, agent_b, _ = provider_with_agent(app, b)
    gpu = register_gpu(agent_a, "NVIDIA RTX A6000", "cuda", 48000)
    assert agent_b.post(f"/agent/gpus/{gpu['id']}/heartbeat", json={"free_memory_mb": 1}).status_code == 404


def test_events_persisted_for_registration_and_busy_but_not_for_heartbeats(app, db):
    owner = signup(app, "e@example.com")
    _, agent, _ = provider_with_agent(app, owner)
    gpu = register_gpu(agent, "NVIDIA RTX A6000", "cuda", 48000)
    agent.post(f"/agent/gpus/{gpu['id']}/heartbeat", json={"free_memory_mb": 10, "status": "busy_external"})
    agent.post(f"/agent/gpus/{gpu['id']}/heartbeat", json={"free_memory_mb": 10, "status": "busy_external"})
    types = [e.type for e in db.exec(select(Event).order_by(Event.id)).all()]
    assert types == ["gpu.registered", "gpu.busy_external"]  # one transition event, no heartbeat rows


def test_cross_user_isolation_for_providers_and_tokens(app):
    a, b = signup(app, "iso-a@example.com"), signup(app, "iso-b@example.com")
    pid = a.post("/providers", json={"name": "A's rig"}).json()["id"]
    tid = a.post(f"/providers/{pid}/agent-tokens", json={}).json()["id"]
    assert b.get("/providers").json() == []
    assert b.post(f"/providers/{pid}/agent-tokens", json={}).status_code == 404
    assert b.get(f"/providers/{pid}/agent-tokens").status_code == 404
    assert b.delete(f"/providers/{pid}/agent-tokens/{tid}").status_code == 404
    assert a.get(f"/providers/{pid}/agent-tokens").json()[0]["revoked"] is False


def test_my_gpus_lists_only_my_providers_gpus(app):
    a, b = signup(app, "mg-a@example.com"), signup(app, "mg-b@example.com")
    _, agent_a, _ = provider_with_agent(app, a)
    _, agent_b, _ = provider_with_agent(app, b)
    register_gpu(agent_a, "NVIDIA RTX A6000", "cuda", 48000)
    register_gpu(agent_b, "Apple M2", "mps", 8000)
    assert [g["name"] for g in a.get("/gpus/mine").json()] == ["NVIDIA RTX A6000"]
    assert [g["name"] for g in b.get("/gpus/mine").json()] == ["Apple M2"]
    assert len(a.get("/gpus").json()) == 2
