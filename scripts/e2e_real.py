"""Real end-to-end run on one Apple-silicon Mac: backend + provider agents + BloomBee workers + TinyLlama-1.1B.

  SCRATCH=<dir containing models/tinyllama>  BB_PYTHON=<python with BloomBee deps>  python scripts/e2e_real.py [stage]

Stages: train (default)  |  full (train, kill + recover, deploy, compare)
Everything runs as separate processes exactly as in production; only the 'providers' are simulated (3 virtual GPUs on this Mac).
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
SCRATCH = Path(os.environ["SCRATCH"])
BB_PYTHON = os.environ["BB_PYTHON"]
BB_DIR = os.environ.get("BLOOMBEE_DIR", str(Path.home() / "Desktop/Repos/BloomBee"))
STAGE = sys.argv[1] if len(sys.argv) > 1 else "train"
STEPS = int(os.environ.get("STEPS", 14))
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
                         start_new_session=True, cwd=cwd)
    procs.append(p)
    return p


def cleanup():
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except Exception:
            pass
    time.sleep(2)
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass


def wait(predicate, timeout, what):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = predicate()
        if v:
            return v
        time.sleep(0.5)
    raise TimeoutError(what)


def main():
    work = Path(tempfile.mkdtemp(prefix="syntra-e2e-", dir=SCRATCH))
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    backend_py = str(ROOT / "backend/.venv/bin/python")
    env = {"SYNTRA_DB": f"sqlite:///{work}/syntra.db", "SYNTRA_BLOOMBEE_PYTHON": BB_PYTHON, "SYNTRA_BLOOMBEE_DIR": BB_DIR,
           "SYNTRA_MODELS_DIR": str(SCRATCH / "models"), "SYNTRA_DATA_DIR": str(work / "data"), "PYTHONPATH": str(ROOT / "backend")}
    # data dirs live in the temp folder so repeated runs never collide
    (work / "backend.log").write_text("")
    spawn([backend_py, "-m", "uvicorn", "app.main:app", "--port", str(port), "--log-level", "warning"], env=env,
          out=open(work / "backend.log", "w"), cwd=str(ROOT / "backend"))
    # seed (same db) before first use
    subprocess.run([backend_py, "-m", "app.seed"], env={**os.environ, **env}, cwd=str(ROOT / "backend"), capture_output=True, check=True)
    wait(lambda: httpx.get(f"{url}/health", timeout=2).status_code == 200, 40, "backend up")
    log("backend up", url=url, work=str(work))

    def new_user(name):
        c = httpx.Client(base_url=url, timeout=60)
        assert c.post("/auth/signup", json={"email": f"{name}@example.com", "password": PASSWORD, "display_name": name}).status_code == 201
        return c

    alice, provider = new_user("alice"), new_user("provider")
    pid = provider.post("/providers", json={"name": "Mac lab", "type": "individual"}).json()["id"]
    token = provider.post(f"/providers/{pid}/agent-tokens", json={"label": "m4"}).json()["token"]
    spawn([backend_py, "-m", "syntra_agent", "--server", url, "--token", token, "--simulated", "--sim-count", "3", "--sim-memory-gb", "6",
           "--bloombee-dir", BB_DIR, "--python", BB_PYTHON, "--models-dir", str(SCRATCH / "models"), "--work-dir", str(work / "agent"),
           "--region", "home"], env={"PYTHONPATH": str(ROOT / "agent")}, out=open(work / "agent.log", "w"))
    wait(lambda: len([g for g in alice.get("/gpus").json() if g["status"] == "idle"]) == 3, 40, "3 idle GPUs")
    pool = alice.get("/gpus").json()
    log("pool ready", gpus=[(g["name"], g["gpu_class"], g["total_memory_mb"]) for g in pool])

    plan = alice.get("/plans", params={"base_model_id": 1, "requested_gpus": 2}).json()
    log("plan preview", fits=plan["fits"], stages=[(s["gpu_name"], s["layer_start"], s["layer_end"]) for s in plan["stages"]], hourly=plan["hourly_rate"])

    job = alice.post("/jobs", json={"base_model_id": 1, "dataset_id": 1, "requested_gpus": 2, "steps": STEPS, "batch_size": 2, "name": "Ticket classifier"}).json()
    jid = job["id"]
    log("job created", id=jid, status=job["status"], leases=[(l["gpu_name"], l["layer_start"], l["layer_end"]) for l in job["leases"]])

    def events(kind=None):
        evs = alice.get("/events", params={"job_id": jid}).json()
        return [e for e in evs if kind is None or e["type"] == kind]

    wait(lambda: len(events("worker.online")) >= 2, 420, "both workers online")
    log("workers online", peers=[e["payload"]["peer_id"][:12] for e in events("worker.online")])
    wait(lambda: len(events("train.step")) >= 3, 300, "3 training steps")
    log("training", first=events("train.step")[0]["payload"], last=events("train.step")[-1]["payload"])

    # --- kill a worker mid-training; the platform must find another GPU and the run must continue ------------------
    steps_before = len(events("train.step"))
    t_kill = time.monotonic()
    r = alice.post(f"/jobs/{jid}/kill-worker", json={"layer_start": 11})
    log("kill requested", status=r.status_code)
    down = wait(lambda: events("worker.down"), 30, "worker.down")[0]
    repl_start = wait(lambda: events("worker.replacement_starting"), 30, "replacement starting")[0]
    log("worker down -> replacement", down=down["payload"], new_gpu=repl_start["payload"])
    repl_online = wait(lambda: events("worker.replacement_online"), 300, "replacement online")[0]
    first_after = wait(lambda: events("train.step")[steps_before:] and events("train.step")[steps_before], 300, "step after recovery")
    recovery = time.monotonic() - t_kill
    log("recovered", replacement_online=repl_online["payload"], seconds_kill_to_next_step=round(recovery, 1), step=first_after["payload"])
    results["recovery_seconds"] = round(recovery, 1)

    done = wait(lambda: (lambda j: j if j["status"] in ("completed", "failed", "stopped") else None)(alice.get(f"/jobs/{jid}").json()), 900, "job finished")
    steps = [e["payload"] for e in events("train.step")]
    log("job finished", status=done["status"], steps=len(steps), first_loss=steps[0]["loss"], last_loss=steps[-1]["loss"], result=done["result"].get("error"))
    assert done["status"] == "completed", done["result"]
    killed = [l for l in done["leases"] if l["end_reason"] == "killed_by_user"]
    assert killed and all(l["gpu_id"] != killed[0]["gpu_id"] for l in done["leases"] if l["layer_start"] == 11 and l["id"] != killed[0]["id"])

    models = alice.get("/models").json()
    assert len(models) == 1, models
    adapter = Path(work).parent  # placeholder to keep linters quiet
    log("registry", models=[(m["id"], m["name"], m["job_id"]) for m in models])
    costs, earnings = alice.get("/ledger/costs").json(), provider.get("/ledger/earnings").json()
    log("billing", cost=costs["total"], earned=earnings["total"], entries=len(costs["entries"]))
    assert costs["total"] > 0 and abs(costs["total"] - earnings["total"]) < 1e-6
    results.update(job=jid, model=models[0]["id"], steps=len(steps), first_loss=steps[0]["loss"], last_loss=steps[-1]["loss"])
    results["work"] = str(work)

    if STAGE == "full":
        from e2e_deploy import run_deploy_stage  # noqa: E402  (added with the deploy milestone)
        run_deploy_stage(url, alice, provider, models[0], log, results, wait)


try:
    sys.path.insert(0, str(ROOT / "scripts"))
    main()
    log("E2E PASSED", **results)
except BaseException as exc:
    import traceback
    traceback.print_exc()
    log("E2E FAILED", error=repr(exc)[:400], **results)
    results["failed"] = True
finally:
    cleanup()
    sys.stdout.flush()
    os._exit(1 if results.get("failed") else 0)
