from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.config import Settings
from app.main import create_app
from app.seed import seed_reference_data

PASSWORD = "correct horse battery"


@pytest.fixture
def app():
    settings = Settings(database_url="sqlite://", run_background_tasks=False)
    application = create_app(settings)
    with Session(application.state.engine) as session:
        seed_reference_data(session)
    return application


@pytest.fixture
def db(app):
    with Session(app.state.engine) as session:
        yield session


@pytest.fixture
def client(app):
    return TestClient(app)


def new_client(app) -> TestClient:
    return TestClient(app)


def signup(app, email: str, password: str = PASSWORD, display_name: str = "") -> TestClient:
    """A new browser-like client already signed in as a fresh user."""
    c = TestClient(app)
    r = c.post("/auth/signup", json={"email": email, "password": password, "display_name": display_name})
    assert r.status_code == 201, r.text
    return c


def make_admin(app, email: str) -> None:
    from sqlmodel import select
    from app.models import User

    with Session(app.state.engine) as s:
        user = s.exec(select(User).where(User.email == email)).one()
        user.is_admin = True
        s.add(user)
        s.commit()


def provider_with_agent(app, user_client: TestClient, name: str = "Test Provider", type: str = "individual"):
    """Create a provider for the signed-in user plus an agent token; returns (provider_id, agent_client)."""
    pid = user_client.post("/providers", json={"name": name, "type": type}).json()["id"]
    token = user_client.post(f"/providers/{pid}/agent-tokens", json={"label": "laptop"}).json()["token"]
    agent = TestClient(app)
    agent.headers.update({"Authorization": f"Bearer {token}"})
    return pid, agent, token


def register_gpu(agent: TestClient, name="Apple M4 Pro", device="mps", total=16000, index=0, **extra):
    r = agent.post("/agent/gpus", json={"name": name, "device": device, "device_index": index,
                                         "total_memory_mb": total, **extra})
    assert r.status_code == 200, r.text
    return r.json()
