"""Real end-to-end DiLoCo (full fine-tuning, data parallel) on one Mac: backend + provider agent (3 simulated GPUs) +
coordinator + replica processes training a small Llama (TinyLlama tokenizer, random weights) with real torch.

  BB_PYTHON=<python with torch/transformers> backend/.venv/bin/python scripts/e2e_diloco.py

Checks: replicas train and sync for several rounds, the loss falls, a killed replica is replaced and the run finishes,
the full model lands in the registry. Model size is set by HIDDEN / LAYERS (defaults keep it fast on a laptop).
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
BB_PYTHON = os.environ["BB_PYTHON"]
HIDDEN, LAYERS = int(os.environ.get("HIDDEN", 128)), int(os.environ.get("LAYERS", 2))
STEPS, INNER = int(os.environ.get("STEPS", 60)), int(os.environ.get("INNER", 10))
PASSWORD = "correct horse battery"
t0 = time.monotonic()
procs: list[subprocess.Popen] = []
results: dict = {}


def log(msg, **kw):
    print(json.dumps({"t": round(time.monotonic() - t0, 1), "msg": msg, **kw}), flush=True)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def spawn(argv, env=None, out=None, cwd=None):
    p = subprocess.Popen(argv, env={**os.environ, **(env or {})}, stdout=out or subprocess.DEVNULL, stderr=subprocess.STDOUT,
                         cwd=cwd, start_new_session=True)
    procs.append(p)
    return p


def wait(predicate, timeout, what):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            v = predicate()
        except httpx.HTTPError:
            v = None
        if v:
            return v
        time.sleep(0.5)
    raise TimeoutError(what)


def main():
    work = Path(tempfile.mkdtemp(prefix="syntra-diloco-"))
    model_dir = work / "models" / "tiny-diloco"
    subprocess.run([BB_PYTHON, "-c", f"""
import glob, torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
tok = AutoTokenizer.from_pretrained(glob.glob('{Path.home()}/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/*')[0])
torch.manual_seed(0)
m = LlamaForCausalLM(LlamaConfig(vocab_size=tok.vocab_size, hidden_size={HIDDEN}, intermediate_size={HIDDEN * 4},
    num_hidden_layers={LAYERS}, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256))
m.save_pretrained('{model_dir}'); tok.save_pretrained('{model_dir}')
"""], check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    log("tiny model created", path=str(model_dir))

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    py = str(ROOT / "backend/.venv/bin/python")
    env = {"SYNTRA_DB": f"sqlite:///{work}/syntra.db", "SYNTRA_BLOOMBEE_PYTHON": BB_PYTHON, "SYNTRA_MODELS_DIR": str(work / "models"),
           "SYNTRA_DATA_DIR": str(work / "data"), "PYTHONPATH": str(ROOT / "backend")}
    subprocess.run([py, "-m", "app.seed"], env={**os.environ, **env}, cwd=str(ROOT / "backend"), capture_output=True, check=True)
    subprocess.run([py, "-c", f"""
from sqlmodel import Session
from app.db import make_engine
from app.models import BaseModel
e = make_engine('sqlite:///{work}/syntra.db')
with Session(e) as s:
    s.add(BaseModel(name='Tiny Llama (DiLoCo test)', slug='tiny-diloco', hf_id='{model_dir}', local_path='{model_dir}',
                    family='llama', num_layers={LAYERS}, hidden_size={HIDDEN}, per_layer_memory_mb=20)); s.commit()
"""], env={**os.environ, **env}, cwd=str(ROOT / "backend"), check=True)
    spawn([py, "-m", "uvicorn", "app.main:app", "--port", str(port), "--log-level", "warning"], env=env,
          out=open(work / "backend.log", "w"), cwd=str(ROOT / "backend"))
    wait(lambda: httpx.get(f"{url}/health", timeout=2).status_code == 200, 40, "backend up")

    def new_user(name):
        c = httpx.Client(base_url=url, timeout=60)
        assert c.post("/auth/signup", json={"email": f"{name}@example.com", "password": PASSWORD}).status_code == 201
        return c

    alice, provider = new_user("alice"), new_user("provider")
    pid = provider.post("/providers", json={"name": "Mac lab", "type": "individual"}).json()["id"]
    token = provider.post(f"/providers/{pid}/agent-tokens", json={"label": "m4"}).json()["token"]
    spawn([py, "-m", "syntra_agent", "--server", url, "--token", token, "--simulated", "--sim-count", "3", "--sim-memory-gb", "6",
           "--python", BB_PYTHON, "--models-dir", str(work / "models"), "--work-dir", str(work / "agent")],
          env={"PYTHONPATH": str(ROOT / "agent")}, out=open(work / "agent.log", "w"))
    wait(lambda: len([g for g in alice.get("/gpus").json() if g["status"] == "idle"]) == 3, 40, "3 idle GPUs")
    base_id = next(m["id"] for m in alice.get("/base-models").json() if m["name"].startswith("Tiny Llama"))
    job = alice.post("/jobs", json={"base_model_id": base_id, "dataset_id": 1, "method": "diloco", "requested_gpus": 2,
                                    "steps": STEPS, "inner_steps": INNER, "lr": 1e-3, "batch_size": 4, "seq_len": 64,
                                    "name": "DiLoCo e2e"}).json()
    jid = job["id"]
    log("job created", id=jid, status=job["status"], leases=[(l["gpu_name"], l["layer_start"]) for l in job["leases"]])

    def events(kind):
        return [e for e in alice.get("/events", params={"job_id": jid}).json() if e["type"] == kind]

    wait(lambda: len(events("worker.online")) >= 2, 300, "both replicas online")
    wait(lambda: len(events("train.step")) >= 2, 600, "two rounds")
    log("rounds", steps=[e["payload"] for e in events("train.step")])
    t_kill = time.monotonic()
    assert alice.post(f"/jobs/{jid}/kill-worker", json={"layer_start": 1}).status_code == 200
    repl = wait(lambda: events("worker.replacement_online"), 300, "replacement online")[0]
    log("replacement online", seconds=round(time.monotonic() - t_kill, 1), payload=repl["payload"])
    done = wait(lambda: (lambda j: j if j["status"] in ("completed", "failed", "stopped") else None)(alice.get(f"/jobs/{jid}").json()),
                1200, "job finished")
    steps = [e["payload"] for e in events("train.step")]
    log("finished", status=done["status"], rounds=[(s["round"], s["loss"], s["replicas_reported"]) for s in steps], error=done["result"].get("error"))
    assert done["status"] == "completed", done["result"]
    assert steps[-1]["loss"] < steps[0]["loss"], "loss did not fall"
    assert any(s["replicas_reported"] == 2 for s in steps[2:]), "the replacement never contributed"
    model = next(m for m in alice.get("/models").json() if m["name"] == "DiLoCo e2e")
    assert model["method"] == "full"
    results.update(rounds=len(steps), first_loss=steps[0]["loss"], last_loss=steps[-1]["loss"],
                   replacement_seconds=round(time.monotonic() - t_kill, 1), model_id=model["id"], work=str(work))


try:
    main()
    log("DILOCO E2E PASSED", **results)
except BaseException as exc:
    import traceback
    traceback.print_exc()
    log("DILOCO E2E FAILED", error=repr(exc)[:400], **results)
    results["failed"] = True
finally:
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except Exception:
            pass
    time.sleep(2)
    sys.path.insert(0, str(ROOT / "agent"))
    from syntra_agent.workers import kill_process_group
    for p in procs:
        kill_process_group(p.pid)
    os._exit(1 if results.get("failed") else 0)
