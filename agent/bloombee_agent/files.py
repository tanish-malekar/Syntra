"""Fetch a fine-tuned model's merged weights from the platform (hash-checked, resumable by skipping verified files)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import httpx


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


async def ensure_model_files(http: httpx.AsyncClient, model_id: int, dest: Path) -> Path:
    """Download any file of the model that is missing or fails its checksum; returns the directory."""
    r = await http.get(f"/files/models/{model_id}/manifest", timeout=600)
    r.raise_for_status()
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    for entry in r.json()["files"]:
        target = (dest / entry["path"]).resolve()
        if root not in target.parents:
            raise ValueError(f"refusing to write outside the model directory: {entry['path']}")
        if target.exists() and target.stat().st_size == entry["size"] and _sha256(target) == entry["sha256"]:
            continue  # already here and verified
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        async with http.stream("GET", f"/files/models/{model_id}/file/{entry['path']}", timeout=None) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                async for chunk in resp.aiter_bytes(1 << 20):
                    f.write(chunk)
        if _sha256(tmp) != entry["sha256"]:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"checksum mismatch for {entry['path']}")
        tmp.replace(target)
    return dest
