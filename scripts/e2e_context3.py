"""Platform end-to-end across two machines: this Mac (Apple GPU) and context3 (RTX A6000s, idle ones only).

  SCRATCH=<dir with models/tinyllama> BB_PYTHON=<python with BloomBee> backend/.venv/bin/python scripts/e2e_context3.py

Pool: one Mac GPU + context3 GPUs 0 and 2 (only if nvidia-smi shows them idle). alice trains on 2 GPUs (the two on context3),
a context3 worker is killed, and the replacement must land on the other machine (the Mac) through the SSH tunnel.
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
sys.path.insert(0, str(ROOT / "bridge"))
import ssh  # noqa: E402

SCRATCH = Path(os.environ["SCRATCH"])
BB_PYTHON = os.environ["BB_PYTHON"]
BB_DIR = os.environ.get("BLOOMBEE_DIR", str(Path.home() / "Desktop/Repos/BloomBee"))
C3 = "/data01/tmalekar/syntra"
STEPS = int(os.environ.get("STEPS", 12))
RELAY = os.environ.get("RELAY") == "1"  # context3's agent uses --relay: only the backend port is reachable, worker traffic rides the agent connection
START_DELAY = int(os.environ.get("START_DELAY", 0))  # seconds to wait before training, to open the UI first
HOLD = int(os.environ.get("HOLD", 0))  # seconds to keep the stack alive after the test, to look at the results in the UI
BACKEND_PORT, DHT_PORT = 8000, 31001
PASSWORD = "correct horse battery"
t0 = time.monotonic()
procs: list[subprocess.Popen] = []
results: dict = {}


def log(msg, **kw):
    print(json.dumps({"t": round(time.monotonic() - t0, 1), "msg": msg, **kw}), flush=True)


def wait(pred, timeout, what):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            v = pred()
        except (httpx.HTTPError, OSError):  # e.g. the server is not listening yet
            v = None
        if v:
            return v
        time.sleep(0.7)
    raise TimeoutError(what)


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
    # context3: only processes that belong to this run (agent + workers carry the run tag in their command lines)
    try:
        ssh.run_remote(f"pkill -u tmalekar -f 'bloombee_agent.*{TAG}' || true; pkill -u tmalekar -f 'dht_prefix job.*initial_peers /ip4/127.0.0.1/tcp/{DHT_PORT}' || true; sleep 2; "
                       f"pgrep -u tmalekar -f '{TAG}|initial_peers /ip4/127.0.0.1/tcp/{DHT_PORT}' | wc -l", timeout=60)
    except Exception:
        pass


TAG = f"e2e{int(time.time())}"


def main():
    work = Path(tempfile.mkdtemp(prefix="bloombee-c3-", dir=SCRATCH))
    py = str(ROOT / "backend/.venv/bin/python")

    # --- only idle GPUs on context3 ---------------------------------------------------------------------------------
    r = ssh.run_remote("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits")
    idle = [int(l.split(",")[0]) for l in r.stdout.strip().splitlines()
            if int(l.split(",")[1]) <= 5 and int(l.split(",")[2]) <= 1500 and int(l.split(",")[0]) in (0, 2, 3)]
    log("context3 idle GPUs (of 0,2,3)", idle=idle)
    if len(idle) < 2:
        raise SystemExit("fewer than two idle context3 GPUs right now: not running")
    offer = ",".join(str(i) for i in idle[:2])

    # --- ship the agent to context3 and make a small venv for it ------------------------------------------------------
    ssh.push_dir(str(ROOT / "agent"), "bloombee_agent", f"{C3}/agent")
    r = ssh.run_remote(f"[ -x {C3}/agent-venv/bin/python ] || /data01/tmalekar/conda-envs/bb/bin/python -m venv {C3}/agent-venv; "
                       f"export PIP_CACHE_DIR={C3}/tmp TMPDIR={C3}/tmp; {C3}/agent-venv/bin/pip install -q httpx 'websockets>=13' 2>&1 | tail -2; "
                       f"{C3}/agent-venv/bin/python -c 'import httpx, websockets; print(\"agent deps ok\")'", timeout=300)
    log("context3 agent env", out=r.stdout.strip()[-80:])

    # --- backend on this Mac with a fixed DHT port, tunnel, agents -------------------------------------------------------
    env = {"BLOOMBEE_DB": f"sqlite:///{work}/syntra.db", "BLOOMBEE_PYTHON": BB_PYTHON, "BLOOMBEE_DIR": BB_DIR,
           "BLOOMBEE_MODELS_DIR": str(SCRATCH / "models"), "BLOOMBEE_DATA_DIR": str(work / "data"), "BLOOMBEE_DHT_PORT": str(DHT_PORT),
           "PYTHONPATH": str(ROOT / "backend")}
    subprocess.run([py, "-m", "app.seed"], env={**os.environ, **env}, cwd=str(ROOT / "backend"), capture_output=True, check=True)
    spawn([py, "-m", "uvicorn", "app.main:app", "--port", str(BACKEND_PORT), "--log-level", "warning"], env=env, out=open(work / "backend.log", "w"), cwd=str(ROOT / "backend"))
    url = f"http://127.0.0.1:{BACKEND_PORT}"
    wait(lambda: httpx.get(f"{url}/health", timeout=2).status_code == 200, 40, "backend up")
    spawn([py, str(ROOT / "bridge/tunnel.py"), *(["--backend-only"] if RELAY else [])], out=open(work / "tunnel.log", "w"))
    time.sleep(6)
    log("backend + tunnel up")

    def new_user(name):
        c = httpx.Client(base_url=url, timeout=60)
        assert c.post("/auth/signup", json={"email": f"{name}@example.com", "password": PASSWORD, "display_name": name}).status_code == 201
        return c

    alice, mac_owner, lab_owner = new_user("alice"), new_user("macowner"), new_user("lab")

    def agent_token(owner, name, kind):
        pid = owner.post("/providers", json={"name": name, "type": kind}).json()["id"]
        return owner.post(f"/providers/{pid}/agent-tokens", json={"label": name}).json()["token"]

    mac_token, lab_token = agent_token(mac_owner, "Maya's MacBook", "individual"), agent_token(lab_owner, "Stony Brook lab (context3)", "enterprise")
    spawn([py, "-m", "bloombee_agent", "--server", url, "--token", mac_token, "--bloombee-dir", BB_DIR, "--python", BB_PYTHON,
           "--models-dir", str(SCRATCH / "models"), "--work-dir", str(work / "agent-mac"), "--port-range", "31010-31019", "--region", "home"],
          env={"PYTHONPATH": str(ROOT / "agent")}, out=open(work / "agent-mac.log", "w"))
    # the context3 agent: idle GPUs only (allowlist), its own env, models on /data01, workers on the forwarded ports
    remote = (f"cd {C3}/agent && export HF_HOME={C3}/hf TMPDIR={C3}/tmp PYTHONPATH={C3}/agent; "
              f"nohup {C3}/agent-venv/bin/python -m bloombee_agent --server http://127.0.0.1:{BACKEND_PORT} --token {lab_token} "
              f"--bloombee-dir {C3}/BloomBee --python /data01/tmalekar/conda-envs/bb/bin/python --models-dir {C3}/models "
              f"--work-dir {C3}/agent-work-{TAG} {'--relay' if RELAY else '--port-range 31030-31039'} --gpus {offer} --region lab-{TAG} > {C3}/out/agent_{TAG}.log 2>&1 < /dev/null & echo started")
    log("context3 agent", out=ssh.run_remote(remote).stdout.strip())
    wait(lambda: len([g for g in alice.get("/gpus").json() if g["status"] == "idle"]) == 3, 90, "3 idle GPUs (1 Mac + 2 context3)")
    pool = alice.get("/gpus").json()
    log("pool", gpus=[(g["provider_name"], g["name"], g["gpu_class"], g["device"]) for g in pool])
    assert {g["device"] for g in pool} == {"mps", "cuda"}

    # --- train across the lab's two RTX A6000s, kill one worker, the replacement must come from the other machine -----------
    log("UI ready: http://localhost:5173 , sign in as alice@example.com (password in PASSWORD above)", wait_seconds=START_DELAY)
    time.sleep(START_DELAY)  # time to open the UI before training starts
    plan = alice.get("/plans", params={"base_model_id": 1, "requested_gpus": 2}).json()
    log("plan", stages=[(s["provider"], s["gpu_name"], s["layer_start"], s["layer_end"]) for s in plan["stages"]])
    job = alice.post("/jobs", json={"base_model_id": 1, "dataset_id": 1, "requested_gpus": 2, "steps": STEPS, "name": "Cross-machine ticket model"}).json()
    jid = job["id"]

    def events(kind=None):
        return [e for e in alice.get("/events", params={"job_id": jid}).json() if kind is None or e["type"] == kind]
    wait(lambda: len(events("worker.online")) >= 2, 600, "context3 workers online")
    wait(lambda: len(events("train.step")) >= 3, 300, "training steps")
    log("training on context3", step=events("train.step")[-1]["payload"])
    t_kill = time.monotonic()
    assert alice.post(f"/jobs/{jid}/kill-worker", json={"layer_start": 11}).status_code == 200
    starting = wait(lambda: events("worker.replacement_starting"), 60, "replacement starting")[0]["payload"]
    log("replacement", **starting)
    assert starting["gpu_device"] == "mps", "the replacement should come from the other machine (the Mac)"
    wait(lambda: events("worker.replacement_online"), 600, "replacement online across the tunnel")
    n = len(events("train.step"))
    wait(lambda: len(events("train.step")) > n, 300, "a step after recovery")
    log("recovered across machines", seconds=round(time.monotonic() - t_kill, 1))
    done = wait(lambda: (lambda j: j if j["status"] in ("completed", "failed", "stopped") else None)(alice.get(f"/jobs/{jid}").json()), 900, "job finished")
    steps = [e["payload"] for e in events("train.step")]
    log("job finished", status=done["status"], steps=len(steps), first_loss=steps[0]["loss"], last_loss=steps[-1]["loss"], error=done["result"].get("error"))
    assert done["status"] == "completed", done["result"]
    costs = alice.get("/ledger/costs").json()
    earned = {n: c.get("/ledger/earnings").json()["total"] for n, c in (("mac", mac_owner), ("lab", lab_owner))}
    log("billing", cost=costs["total"], earned=earned)
    assert earned["lab"] > 0 and earned["mac"] > 0 and abs(costs["total"] - sum(earned.values())) < 1e-6
    results.update(steps=len(steps), first_loss=steps[0]["loss"], last_loss=steps[-1]["loss"])


try:
    main()
    log("CONTEXT3 E2E PASSED", **results)
except BaseException as exc:
    import traceback
    traceback.print_exc()
    log("CONTEXT3 E2E FAILED", error=repr(exc)[:400], **results)
    results["failed"] = True
finally:
    if HOLD:
        log("holding the stack up for the UI", seconds=HOLD)
        time.sleep(HOLD)
    cleanup()
    sys.stdout.flush()
    os._exit(1 if results.get("failed") else 0)
