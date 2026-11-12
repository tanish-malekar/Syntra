from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select
from starlette.websockets import WebSocketDisconnect

from app.events import PAYLOAD_KEYS, EventType, is_pool_level
from app.models import Event, User
from tests.conftest import provider_with_agent, signup

EVENTS_TS = Path(__file__).resolve().parents[2] / "frontend" / "src" / "events.ts"


def _uid(client: TestClient) -> int:
    return client.get("/auth/me").json()["id"]


def _train_step(app, db, user_id: int, step: int, job_id: int = 1):
    return app.state.events.publish(
        db, EventType.TRAIN_STEP, job_id=job_id, user_id=user_id,
        payload={"step": step, "loss": 1.0 / step, "grad_norm": 2.0, "step_seconds": 1.0},
    )


def _marker(app, db):
    return app.state.events.publish(db, EventType.GPU_OFFLINE, gpu_id=0, payload={"reason": "marker"})


def test_every_event_type_has_a_payload_spec():
    assert set(PAYLOAD_KEYS) == set(EventType)


def test_typescript_mirror_lists_exactly_the_python_event_types():
    text = EVENTS_TS.read_text()
    union = text.split("export type EventType =")[1].split(";")[0]
    ts_types = set(re.findall(r'"([a-z_.]+)"', union))
    assert ts_types == {t.value for t in EventType}
    payload_block = text.split("export interface EventPayloads {")[1].split("\n}\n")[0]
    for etype, keys in PAYLOAD_KEYS.items():
        line = next(l for l in payload_block.splitlines() + [""] if f'"{etype.value}"' in l or False)
        body = payload_block.split(f'"{etype.value}":')[1].split(";\n")[0]
        for key in keys:
            assert re.search(rf"\b{key}\b", body), f"{etype.value}: payload key {key} missing in events.ts"


def test_publish_validates_payload_reason_and_owner(app, db):
    events = app.state.events
    with pytest.raises(ValueError, match="missing payload keys"):
        events.publish(db, EventType.TRAIN_STEP, user_id=1, payload={"step": 1})
    with pytest.raises(ValueError, match="reason"):
        events.publish(db, EventType.WORKER_DOWN, user_id=1,
                       payload={"reason": "gremlins", "layer_start": 0, "layer_end": 1})
    with pytest.raises(ValueError, match="owning user_id"):
        events.publish(db, EventType.TRAIN_STEP, payload={"step": 1, "loss": 1, "grad_norm": 1, "step_seconds": 1})
    with pytest.raises(ValueError):
        events.publish(db, "not.a.type", payload={})
    assert db.exec(select(Event)).all() == []  # nothing half-written


def test_history_is_per_user_and_filterable_by_job(app, db):
    a, b = signup(app, "ha@example.com"), signup(app, "hb@example.com")
    for step in (1, 2, 3):
        _train_step(app, db, _uid(a), step, job_id=10)
    _train_step(app, db, _uid(a), 1, job_id=11)
    _train_step(app, db, _uid(b), 1, job_id=12)
    assert [e["payload"]["step"] for e in a.get("/events", params={"job_id": 10}).json()] == [1, 2, 3]
    assert len(a.get("/events").json()) == 4
    assert [e["job_id"] for e in b.get("/events").json()] == [12]


def test_websocket_rejects_unauthenticated_clients(app):
    with pytest.raises(WebSocketDisconnect):
        with TestClient(app).websocket_connect("/ws/events"):
            pass
    with pytest.raises(WebSocketDisconnect):
        with TestClient(app).websocket_connect("/ws/events", headers={"Authorization": "Bearer nope"}):
            pass


def test_websocket_hello_and_ping(app):
    with signup(app, "ws@example.com").websocket_connect("/ws/events") as ws:
        assert ws.receive_json() == {"type": "hello", "role": "user"}
        ws.send_text("ping")
        assert ws.receive_json() == {"type": "pong"}


def test_websocket_delivers_only_the_users_own_job_events(app, db):
    a, b = signup(app, "wa@example.com"), signup(app, "wb@example.com")
    with a.websocket_connect("/ws/events") as wa, b.websocket_connect("/ws/events") as wb:
        wa.receive_json(), wb.receive_json()  # hello
        _train_step(app, db, _uid(a), 1)       # A's job
        _train_step(app, db, _uid(b), 7)       # B's job
        _marker(app, db)                       # pool-level: both must get it, and in order
        first_a, second_a = wa.receive_json(), wa.receive_json()
        first_b, second_b = wb.receive_json(), wb.receive_json()
    assert (first_a["type"], first_a["payload"]["step"]) == ("train.step", 1)
    assert second_a["payload"]["reason"] == "marker"  # B's event never reached A
    assert (first_b["type"], first_b["payload"]["step"]) == ("train.step", 7)
    assert second_b["payload"]["reason"] == "marker"  # A's event never reached B


def test_websocket_agent_receives_pool_events_only(app, db):
    owner = signup(app, "wo@example.com")
    _, agent, token = provider_with_agent(app, owner)
    with agent.websocket_connect("/ws/events", headers={"Authorization": f"Bearer {token}"}) as ws:
        assert ws.receive_json() == {"type": "hello", "role": "agent"}
        _train_step(app, db, _uid(owner), 1)  # a user's job: not for agents
        _marker(app, db)
        assert ws.receive_json()["payload"]["reason"] == "marker"


def test_heartbeats_stream_live_but_are_not_stored(app, db):
    owner = signup(app, "hb-owner@example.com")
    _, agent, _ = provider_with_agent(app, owner)
    gpu = agent.post("/agent/gpus", json={"name": "NVIDIA RTX A6000", "device": "cuda", "total_memory_mb": 48000}).json()
    with owner.websocket_connect("/ws/events") as ws:
        ws.receive_json()
        agent.post(f"/agent/gpus/{gpu['id']}/heartbeat", json={"free_memory_mb": 123})
        event = ws.receive_json()
    assert event["type"] == "gpu.heartbeat" and event["id"] is None and event["payload"]["free_memory_mb"] == 123
    assert [e.type for e in db.exec(select(Event)).all()] == ["gpu.registered"]


def test_is_pool_level():
    assert is_pool_level("gpu.heartbeat") and not is_pool_level("train.step")
