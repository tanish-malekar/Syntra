"""Seed reference data and dev accounts:  python -m app.seed

Dev account passwords are generated, written to backend/.env.dev (mode 600, gitignored) and never printed.
Re-running keeps existing accounts and passwords untouched.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from sqlmodel import Session, select

from app.config import BACKEND_DIR, Settings
from app.db import init_db, make_engine
from app.models import BaseModel, Dataset, GpuClass, User
from app.security import hash_password
from app.services.pool import APPLE, DATACENTER, WORKSTATION

GPU_CLASSES = [
    # name, simulated price per GPU-hour, device hint
    (APPLE, 0.10, "mps"),
    (WORKSTATION, 0.45, "cuda"),
    (DATACENTER, 1.80, "cuda"),
]

# per_layer_memory_mb: measured ~0.6-1.0 GB per half of TinyLlama (11 layers) incl. runtime overhead
BASE_MODELS = [
    dict(name="TinyLlama-1.1B-Chat", slug="tinyllama", hf_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", family="llama",
         num_layers=22, hidden_size=2048, per_layer_memory_mb=100),
]

SAMPLE_DIR = BACKEND_DIR / "sample_data"

DEV_ACCOUNTS = [
    ("admin@syntra.test", "Admin", True),
    ("alice@syntra.test", "Alice", False),
    ("bob@syntra.test", "Bob", False),
]


def seed_reference_data(session: Session) -> None:
    for name, price, hint in GPU_CLASSES:
        if session.exec(select(GpuClass).where(GpuClass.name == name)).first() is None:
            session.add(GpuClass(name=name, price_per_hour=price, device_hint=hint))
    for spec in BASE_MODELS:
        if session.exec(select(BaseModel).where(BaseModel.name == spec["name"])).first() is None:
            session.add(BaseModel(**spec))
    if session.exec(select(Dataset).where(Dataset.builtin == True)).first() is None:  # noqa: E712
        train = SAMPLE_DIR / "tickets_train.jsonl"
        rows = len(train.read_text().splitlines()) if train.exists() else 0
        session.add(Dataset(name="Support tickets (demo)", path=str(train), heldout_path=str(SAMPLE_DIR / "tickets_heldout.jsonl"),
                            num_rows=rows, builtin=True))
    session.commit()


def seed_dev_accounts(session: Session, env_path: Path) -> dict[str, str]:
    """Create missing dev accounts; returns {email: password} for the ones created now."""
    created: dict[str, str] = {}
    for email, display_name, is_admin in DEV_ACCOUNTS:
        if session.exec(select(User).where(User.email == email)).first() is None:
            password = secrets.token_urlsafe(16)
            session.add(User(email=email, display_name=display_name, is_admin=is_admin, password_hash=hash_password(password)))
            created[email] = password
    session.commit()
    if created:
        header = "# Dev accounts created by app.seed (do not commit). Passwords are only stored hashed in the database."
        lines = env_path.read_text().splitlines() if env_path.exists() else [header]
        lines += [f"{email}={password}" for email, password in created.items()]
        fd = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.chmod(env_path, 0o600)
    return created


def main() -> None:
    settings = Settings.from_env()
    engine = make_engine(settings.database_url)
    init_db(engine)
    with Session(engine) as session:
        seed_reference_data(session)
        created = seed_dev_accounts(session, BACKEND_DIR / ".env.dev")
    print(f"seeded GPU classes and base models; dev accounts created: {sorted(created) or 'none (already existed)'}")
    if created:
        print("passwords written to backend/.env.dev")


if __name__ == "__main__":
    main()
