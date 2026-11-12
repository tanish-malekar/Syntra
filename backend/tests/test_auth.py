from __future__ import annotations

from datetime import timedelta

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.db import utcnow
from app.models import AuthSession, User
from tests.conftest import PASSWORD, signup


def test_signup_logs_in_with_httponly_lax_cookie(app):
    c = TestClient(app)
    r = c.post("/auth/signup", json={"email": "New@Example.com", "password": PASSWORD, "display_name": "New"})
    assert r.status_code == 201
    assert r.json()["email"] == "new@example.com"  # normalized
    set_cookie = r.headers["set-cookie"].lower()
    assert "httponly" in set_cookie and "samesite=lax" in set_cookie and "syntra_session=" in set_cookie
    assert c.get("/auth/me").json()["email"] == "new@example.com"


def test_signup_rejects_duplicate_email_case_insensitively(app):
    signup(app, "dup@example.com")
    r = TestClient(app).post("/auth/signup", json={"email": "DUP@example.com", "password": PASSWORD})
    assert r.status_code == 409


def test_signup_validates_email_and_password(app):
    c = TestClient(app)
    assert c.post("/auth/signup", json={"email": "not-an-email", "password": PASSWORD}).status_code == 422
    assert c.post("/auth/signup", json={"email": "a@b.co", "password": "short"}).status_code == 422
    assert c.post("/auth/signup", json={"email": "a@b.co", "password": "x" * 300}).status_code == 422


def test_password_is_stored_as_argon2_hash(app, db):
    signup(app, "hash@example.com")
    user = db.exec(select(User).where(User.email == "hash@example.com")).one()
    assert user.password_hash.startswith("$argon2")
    assert PASSWORD not in user.password_hash


def test_session_token_is_stored_hashed(app, db):
    c = signup(app, "tok@example.com")
    cookie = c.cookies.get("syntra_session")
    rows = db.exec(select(AuthSession)).all()
    assert rows and all(r.token_hash != cookie for r in rows)


def test_login_wrong_password_and_unknown_email_give_the_same_error(app):
    signup(app, "real@example.com")
    c = TestClient(app)
    wrong = c.post("/auth/login", json={"email": "real@example.com", "password": "wrong password!"})
    unknown = c.post("/auth/login", json={"email": "ghost@example.com", "password": "wrong password!"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_login_success_sets_a_new_session(app):
    signup(app, "login@example.com")
    c = TestClient(app)
    assert c.get("/auth/me").status_code == 401
    assert c.post("/auth/login", json={"email": "LOGIN@example.com", "password": PASSWORD}).status_code == 200
    assert c.get("/auth/me").json()["email"] == "login@example.com"


def test_logout_revokes_the_session_server_side(app):
    c = signup(app, "out@example.com")
    stolen = c.cookies.get("syntra_session")
    assert c.post("/auth/logout").status_code == 204
    assert c.get("/auth/me").status_code == 401
    replay = TestClient(app)
    replay.cookies.set("syntra_session", stolen)
    assert replay.get("/auth/me").status_code == 401  # the old cookie value no longer works


def test_expired_session_is_rejected(app, db):
    c = signup(app, "exp@example.com")
    row = db.exec(select(AuthSession)).one()
    row.expires_at = utcnow() - timedelta(seconds=1)
    db.add(row)
    db.commit()
    assert c.get("/auth/me").status_code == 401


def test_login_is_rate_limited_per_email_and_ip(app):
    signup(app, "limit@example.com")
    c = TestClient(app)
    for _ in range(5):
        assert c.post("/auth/login", json={"email": "limit@example.com", "password": "nope nope nope"}).status_code == 401
    blocked = c.post("/auth/login", json={"email": "limit@example.com", "password": PASSWORD})
    assert blocked.status_code == 429  # even the right password is refused while blocked
    other = c.post("/auth/login", json={"email": "someone-else@example.com", "password": "nope nope nope"})
    assert other.status_code == 401  # a different email is not blocked


def test_protected_endpoints_require_a_session(client):
    for path in ("/auth/me", "/gpus", "/gpus/mine", "/providers", "/ledger/earnings", "/ledger/costs",
                 "/events", "/gpu-classes", "/base-models"):
        assert client.get(path).status_code == 401, path
