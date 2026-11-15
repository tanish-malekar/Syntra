from __future__ import annotations

import stat

from fastapi.testclient import TestClient
from sqlmodel import select

from app.models import BaseModel, GpuClass, User
from app.seed import DEV_ACCOUNTS, seed_dev_accounts, seed_reference_data


def test_dev_accounts_get_generated_passwords_in_a_private_file(app, db, tmp_path):
    env = tmp_path / ".env.dev"
    created = seed_dev_accounts(db, env)
    assert set(created) == {email for email, _, _ in DEV_ACCOUNTS}
    assert len(set(created.values())) == len(created)  # every password is different
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    text = env.read_text()
    for email, password in created.items():
        assert f"{email}={password}" in text
    assert db.exec(select(User).where(User.email == "admin@bloombee.test")).one().is_admin is True

    email, password = next(iter(created.items()))
    c = TestClient(app)
    assert c.post("/auth/login", json={"email": email, "password": password}).status_code == 200


def test_reseeding_keeps_existing_accounts_and_passwords(app, db, tmp_path):
    env = tmp_path / ".env.dev"
    seed_dev_accounts(db, env)
    before = env.read_text()
    assert seed_dev_accounts(db, env) == {}
    assert env.read_text() == before
    seed_reference_data(db)
    assert len(db.exec(select(GpuClass)).all()) == 3 and len(db.exec(select(BaseModel)).all()) == 1
