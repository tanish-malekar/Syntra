"""Real end-to-end RL on one Mac: backend + provider agent (2 simulated GPUs) + BloomBee workers + TinyLlama.

  SCRATCH=<dir with models/tinyllama> BB_PYTHON=<BloomBee python> backend/.venv/bin/python scripts/e2e_rl.py

1. SDK session (LoRA, or QLoRA with WEIGHT_BITS=4): GRPO with a dense reward (short, finished answers score higher):
   the mean reward must rise. Then save the adapter as a model.
2. Managed DPO job on ticket preference pairs (JSON answer preferred over a rambling one): the reward margin must grow.
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
sys.path.insert(0, str(ROOT / "sdk"))
import syntra  # noqa: E402

SCRATCH = Path(os.environ["SCRATCH"])
BB_PYTHON = os.environ["BB_PYTHON"]
BB_DIR = os.environ.get("BLOOMBEE_DIR", str(Path.home() / ".syntra/BloomBee"))
WEIGHT_BITS = int(os.environ.get("WEIGHT_BITS", 16))
ITERS = int(os.environ.get("ITERS", 8))
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
        time.sleep(1)
    raise TimeoutError(what)


def main():
    work = Path(tempfile.mkdtemp(prefix="syntra-rl-", dir=SCRATCH))
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    py = str(ROOT / "backend/.venv/bin/python")
    env = {"SYNTRA_DB": f"sqlite:///{work}/syntra.db", "SYNTRA_BLOOMBEE_PYTHON": BB_PYTHON, "SYNTRA_BLOOMBEE_DIR": BB_DIR,
           "SYNTRA_MODELS_DIR": str(SCRATCH / "models"), "SYNTRA_DATA_DIR": str(work / "data"), "PYTHONPATH": str(ROOT / "backend")}
    subprocess.run([py, "-m", "app.seed"], env={**os.environ, **env}, cwd=str(ROOT / "backend"), capture_output=True, check=True)
    spawn([py, "-m", "uvicorn", "app.main:app", "--port", str(port), "--log-level", "warning"], env=env,
          out=open(work / "backend.log", "w"), cwd=str(ROOT / "backend"))
    wait(lambda: httpx.get(f"{url}/health", timeout=2).status_code == 200, 40, "backend up")

    def new_user(name):
        c = httpx.Client(base_url=url, timeout=120)
        assert c.post("/auth/signup", json={"email": f"{name}@example.com", "password": PASSWORD}).status_code == 201
        return c

    alice, provider = new_user("alice"), new_user("provider")
    pid = provider.post("/providers", json={"name": "Mac lab", "type": "individual"}).json()["id"]
    token = provider.post(f"/providers/{pid}/agent-tokens", json={"label": "m4"}).json()["token"]
    spawn([py, "-m", "syntra_agent", "--server", url, "--token", token, "--simulated", "--sim-count", "2", "--sim-memory-gb", "6",
           "--bloombee-dir", BB_DIR, "--python", BB_PYTHON, "--models-dir", str(SCRATCH / "models"), "--work-dir", str(work / "agent")],
          env={"PYTHONPATH": str(ROOT / "agent")}, out=open(work / "agent.log", "w"))
    wait(lambda: len([g for g in alice.get("/gpus").json() if g["status"] == "idle"]) == 2, 60, "2 idle GPUs")
    key = alice.post("/api-keys", json={"label": "e2e"}).json()["key"]

    # ---- 1. GRPO through the SDK ------------------------------------------------------------------------------------
    client = syntra.Client(api_key=key, base_url=url, timeout=1800)
    t = time.monotonic()
    tc = client.training_session("TinyLlama-1.1B-Chat", rank=8, weight_bits=WEIGHT_BITS, gpus=2, name="grpo e2e", timeout=900)
    log("session ready", seconds=round(time.monotonic() - t, 1), weight_bits=WEIGHT_BITS)
    max_new = 16
    prompts = ["Ticket: my package has not arrived yet.\nAnswer:", "Ticket: I was charged twice.\nAnswer:",
               "Ticket: I cannot log in.\nAnswer:", "Ticket: the app crashes on start.\nAnswer:"]

    def reward(prompt, completion, s):  # dense: finishing early scores higher (1.0 = EOS right away, 0 = never finished)
        return (1.0 - len(s.completion) / max_new) if s.finished else 0.0

    hist = syntra.rl.grpo(tc, prompts, reward, iterations=ITERS, group_size=8, prompts_per_iteration=2, max_new_tokens=max_new,
                          temperature=1.0, lr=3e-4, clip=0.2, seed=0,
                          on_iteration=lambda h: log("grpo iteration", **{k: (round(v, 4) if isinstance(v, float) else v)
                                                                         for k, v in h.items()}))
    first = sum(h["reward_mean"] for h in hist[:2]) / 2
    last = sum(h["reward_mean"] for h in hist[-2:]) / 2
    model_id = tc.save("grpo e2e")
    tc.close()
    log("grpo done", first_reward=round(first, 4), last_reward=round(last, 4), model_id=model_id)
    results.update(grpo_first=round(first, 4), grpo_last=round(last, 4))
    assert last > first, "GRPO did not raise the reward"

    # ---- 2. managed DPO job -------------------------------------------------------------------------------------------
    pairs = [{"prompt": f"Ticket: {t}\nAnswer:", "chosen": f' {{"department": "{d}", "urgency": "low"}}',
              "rejected": " Thank you for reaching out! I am sorry to hear that. Could you tell me more?"}
             for t, d in [("my package is late.", "shipping"), ("I was charged twice.", "billing"), ("I cannot log in.", "login"),
                          ("the app crashes.", "bug"), ("refund my order.", "billing"), ("wrong address on my parcel.", "shipping"),
                          ("password reset fails.", "login"), ("the page is blank.", "bug")]]
    content = "\n".join(json.dumps(p) for p in pairs).encode()
    ds = alice.post("/datasets", data={"name": "prefs"}, files={"file": ("p.jsonl", content, "application/jsonl")}).json()
    job = alice.post("/jobs", json={"base_model_id": 1, "dataset_id": ds["id"], "method": "dpo", "beta": 0.1, "steps": 12,
                                    "batch_size": 2, "lr": 5e-5, "requested_gpus": 2, "weight_bits": WEIGHT_BITS,
                                    "name": "dpo e2e"}).json()
    done = wait(lambda: (lambda j: j if j["status"] in ("completed", "failed", "stopped") else None)(alice.get(f"/jobs/{job['id']}").json()),
                1800, "DPO job finished")
    steps = [e["payload"] for e in alice.get("/events", params={"job_id": job["id"]}).json() if e["type"] == "train.step"]
    log("dpo done", status=done["status"], error=done["result"].get("error"),
        steps=[(s["step"], round(s["loss"], 4), s.get("reward_margin"), s.get("reward_accuracy")) for s in steps])
    assert done["status"] == "completed", done["result"]
    m0 = sum(s["reward_margin"] for s in steps[:3]) / 3
    m1 = sum(s["reward_margin"] for s in steps[-3:]) / 3
    results.update(dpo_margin_first=round(m0, 4), dpo_margin_last=round(m1, 4))
    assert m1 > m0, "DPO margin did not grow"


try:
    main()
    log("RL E2E PASSED", **results)
except BaseException as exc:
    import traceback
    traceback.print_exc()
    log("RL E2E FAILED", error=repr(exc)[:400], **results)
    results["failed"] = True
finally:
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except Exception:
            pass
    time.sleep(3)
    sys.path.insert(0, str(ROOT / "agent"))
    from syntra_agent.workers import kill_process_group
    for p in procs:
        kill_process_group(p.pid)
    os._exit(1 if results.get("failed") else 0)
