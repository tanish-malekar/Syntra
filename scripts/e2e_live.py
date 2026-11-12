"""Live demo across three networks: this Mac, a second MacBook that joins with ONE command, and one idle context3 GPU.

  SCRATCH=<dir with models/tinyllama> BB_PYTHON=<python with BloomBee> PUBLIC_URL=https://<temporary public address of :8000> \
    C3_GPU=0 backend/.venv/bin/python scripts/e2e_live.py

1. The backend starts here; this Mac's GPU joins. The script prints the one-line command for the second MacBook and waits for it.
2. Training starts on the two MacBooks (watch it at http://localhost:5173, sign in as alice@demo.test, see demo-logins.txt).
3. An idle context3 GPU joins the pool (its agent runs with --relay behind one SSH connection to the backend port only).
4. Close the second MacBook's lid (or set AUTO_KILL=1 to kill its worker instead). The replacement must land on context3.
Optional: START_DELAY (seconds before training, to open the UI), HOLD (seconds to keep everything up afterwards), STEPS.
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
PUBLIC_URL = os.environ.get("PUBLIC_URL", "").rstrip("/")
C3_GPU = int(os.environ.get("C3_GPU", 0))
AUTO_KILL = os.environ.get("AUTO_KILL") == "1"
STEPS = int(os.environ.get("STEPS", 12))
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
        ssh.run_remote(f"pkill -u tmalekar -f 'syntra_agent.*{TAG}' || true; pkill -u tmalekar -f 'dht_prefix job.*initial_peers /ip4/127.0.0.1/tcp/{DHT_PORT}' || true; sleep 2; "
                       f"pgrep -u tmalekar -f '{TAG}|initial_peers /ip4/127.0.0.1/tcp/{DHT_PORT}' | wc -l", timeout=60)
    except Exception:
        pass


TAG = f"e2e{int(time.time())}"


def main():
    work = Path(tempfile.mkdtemp(prefix="syntra-live-", dir=SCRATCH))
    py = str(ROOT / "backend/.venv/bin/python")
    if not PUBLIC_URL:
        raise SystemExit("set PUBLIC_URL to the temporary public address of the backend (port 8000)")

    # --- the approved context3 GPU must be idle right now -----------------------------------------------------------------
    r = ssh.run_remote("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits")
    rows = {int(l.split(",")[0]): (int(l.split(",")[1]), int(l.split(",")[2])) for l in r.stdout.strip().splitlines()}
    if C3_GPU not in rows or rows[C3_GPU][0] > 5 or rows[C3_GPU][1] > 1500:
        raise SystemExit(f"context3 GPU {C3_GPU} is not idle right now ({rows.get(C3_GPU)}): not running")
    ssh.push_dir(str(ROOT / "agent"), "syntra_agent", f"{C3}/agent")
    r = ssh.run_remote(f"[ -x {C3}/agent-venv/bin/python ] || /data01/tmalekar/conda-envs/bb/bin/python -m venv {C3}/agent-venv; "
                       f"export PIP_CACHE_DIR={C3}/tmp TMPDIR={C3}/tmp; {C3}/agent-venv/bin/pip install -q httpx 'websockets>=13' 2>&1 | tail -2; "
                       f"{C3}/agent-venv/bin/python -c 'import httpx, websockets; print(\"agent deps ok\")'", timeout=300)
    log("context3 ready", gpu=C3_GPU, out=r.stdout.strip()[-40:])

    # --- backend here; this Mac's GPU joins ----------------------------------------------------------------------------
    env = {"SYNTRA_DB": f"sqlite:///{work}/syntra.db", "SYNTRA_BLOOMBEE_PYTHON": BB_PYTHON, "SYNTRA_BLOOMBEE_DIR": BB_DIR,
           "SYNTRA_MODELS_DIR": str(SCRATCH / "models"), "SYNTRA_DATA_DIR": str(work / "data"), "SYNTRA_DHT_PORT": str(DHT_PORT),
           "SYNTRA_PUBLIC_URL": PUBLIC_URL, "PYTHONPATH": str(ROOT / "backend")}
    subprocess.run([py, "-m", "app.seed"], env={**os.environ, **env}, cwd=str(ROOT / "backend"), capture_output=True, check=True)
    spawn([py, "-m", "uvicorn", "app.main:app", "--port", str(BACKEND_PORT), "--log-level", "warning"], env=env, out=open(work / "backend.log", "w"), cwd=str(ROOT / "backend"))
    url = f"http://127.0.0.1:{BACKEND_PORT}"
    wait(lambda: httpx.get(f"{url}/health", timeout=2).status_code == 200, 40, "backend up")
    if os.environ.get("SKIP_PUBLIC_CHECK") != "1":  # set it when this machine's own DNS cannot resolve the tunnel name (the other Mac still can)
        wait(lambda: httpx.get(f"{PUBLIC_URL}/health", timeout=10).status_code == 200, 60, "the public address reaches the backend")
    log("backend up and reachable at the public address", public=PUBLIC_URL)

    logins: dict[str, str] = {}

    def new_user(name):
        """Same convention as scripts/demo_stack.py, so demo-logins.txt always lists every login (email, then the password)."""
        c = httpx.Client(base_url=url, timeout=60)
        email, pw = f"{name}@demo.test", f"demo-{name}-pass"
        assert c.post("/auth/signup", json={"email": email, "password": pw, "display_name": name}).status_code == 201
        logins[email] = pw
        return c

    alice, a_owner, b_owner, lab_owner = new_user("alice"), new_user("maya"), new_user("macb"), new_user("lab")

    def agent_token(owner, name, kind):
        pid = owner.post("/providers", json={"name": name, "type": kind}).json()["id"]
        return owner.post(f"/providers/{pid}/agent-tokens", json={"label": name}).json()["token"]

    known = dict(l.split(" ", 1) for l in (ROOT / "demo-logins.txt").read_text().splitlines() if " " in l) if (ROOT / "demo-logins.txt").exists() else {}
    known.update(logins)
    (ROOT / "demo-logins.txt").write_text("".join(f"{e} {p}\n" for e, p in known.items()))
    tok_a, tok_b, tok_c = agent_token(a_owner, "MacBook A (this Mac)", "individual"), agent_token(b_owner, "MacBook B", "individual"), agent_token(lab_owner, "Stony Brook lab (context3)", "enterprise")
    spawn([py, "-m", "syntra_agent", "--server", url, "--token", tok_a, "--bloombee-dir", BB_DIR, "--python", BB_PYTHON,
           "--models-dir", str(SCRATCH / "models"), "--work-dir", str(work / "agent-a"), "--port-range", "31010-31019", "--region", "home-a"],
          env={"PYTHONPATH": str(ROOT / "agent")}, out=open(work / "agent-a.log", "w"))
    wait(lambda: len([g for g in alice.get("/gpus").json() if g["status"] == "idle"]) == 1, 60, "MacBook A in the pool")
    log("UI ready: http://localhost:5173 , sign in with a login from demo-logins.txt, e.g. alice@demo.test")

    # --- the second MacBook joins with one command ------------------------------------------------------------------------
    print("\n" + "=" * 78 + "\nOn the SECOND MacBook, paste this in a terminal (any network):\n\n  "
          f"curl -fsSL {PUBLIC_URL}/join.sh | bash -s -- --token {tok_b} --region home-b\n\n" + "=" * 78 + "\n", flush=True)
    wait(lambda: len([g for g in alice.get("/gpus").json() if g["status"] == "idle"]) == 2, 3600, "MacBook B joined the pool")
    log("pool", gpus=[(g["provider_name"], g["name"], g["device"]) for g in alice.get("/gpus").json()])
    time.sleep(START_DELAY)

    # --- training starts on the two MacBooks -----------------------------------------------------------------------------
    job = alice.post("/jobs", json={"base_model_id": 1, "dataset_id": 1, "requested_gpus": 2, "steps": STEPS, "name": "Three-network ticket model"}).json()
    jid = job["id"]

    def events(kind=None):
        return [e for e in alice.get("/events", params={"job_id": jid}).json() if kind is None or e["type"] == kind]
    wait(lambda: len(events("worker.online")) >= 2, 900, "both MacBook workers online")
    wait(lambda: len(events("train.step")) >= 2, 600, "training steps")
    log("training on the two MacBooks", step=events("train.step")[-1]["payload"],
        gpus=[(l["gpu_name"], l["layer_start"], l["layer_end"]) for l in alice.get(f"/jobs/{jid}").json()["leases"] if l["active"]])

    # --- context3 joins the pool (relay mode: only the backend port is reachable) ----------------------------------------
    spawn([py, str(ROOT / "bridge/tunnel.py"), "--backend-only"], out=open(work / "tunnel.log", "w"))
    time.sleep(5)
    remote = (f"cd {C3}/agent && export HF_HOME={C3}/hf TMPDIR={C3}/tmp PYTHONPATH={C3}/agent; "
              f"nohup {C3}/agent-venv/bin/python -m syntra_agent --server http://127.0.0.1:{BACKEND_PORT} --token {tok_c} --relay "
              f"--bloombee-dir {C3}/BloomBee --python /data01/tmalekar/conda-envs/bb/bin/python --models-dir {C3}/models "
              f"--work-dir {C3}/agent-work-{TAG} --gpus {C3_GPU} --region lab-{TAG} > {C3}/out/agent_{TAG}.log 2>&1 < /dev/null & echo started")
    log("context3 agent", out=ssh.run_remote(remote).stdout.strip())
    wait(lambda: [g for g in alice.get("/gpus").json() if g["device"] == "cuda" and g["status"] == "idle"], 120, "context3 GPU idle in the pool")
    log("context3 GPU joined the pool and waits as a spare")

    # --- one MacBook goes away --------------------------------------------------------------------------------------------
    n0 = len(events("train.step"))
    t_kill = time.monotonic()
    if AUTO_KILL:
        assert alice.post(f"/jobs/{jid}/kill-worker", json={"layer_start": 11}).status_code == 200
    else:
        print("\n>>> NOW close the lid of the SECOND MacBook (or just stop its agent). Waiting up to 10 minutes. <<<\n", flush=True)
        t_kill = None
    starting = wait(lambda: events("worker.replacement_starting"), 600, "a replacement starting")[0]["payload"]
    t_kill = t_kill or time.monotonic()
    log("replacement", **starting)
    assert starting["gpu_device"] == "cuda", "the replacement should be the context3 GPU"
    wait(lambda: events("worker.replacement_online"), 600, "replacement online across the relay")
    wait(lambda: len(events("train.step")) > n0 + 1, 300, "a step after recovery")
    log("recovered onto context3", seconds_since_failure_noticed=round(time.monotonic() - t_kill, 1))
    done = wait(lambda: (lambda j: j if j["status"] in ("completed", "failed", "stopped") else None)(alice.get(f"/jobs/{jid}").json()), 1200, "job finished")
    steps = [e["payload"] for e in events("train.step")]
    log("job finished", status=done["status"], steps=len(steps), first_loss=steps[0]["loss"], last_loss=steps[-1]["loss"], error=done["result"].get("error"))
    assert done["status"] == "completed", done["result"]
    results.update(steps=len(steps), first_loss=steps[0]["loss"], last_loss=steps[-1]["loss"])


try:
    main()
    log("LIVE DEMO PASSED", **results)
except BaseException as exc:
    import traceback
    traceback.print_exc()
    log("LIVE DEMO FAILED", error=repr(exc)[:400], **results)
    results["failed"] = True
finally:
    if HOLD:
        log("holding the stack up for the UI", seconds=HOLD)
        time.sleep(HOLD)
    cleanup()
    sys.stdout.flush()
    os._exit(1 if results.get("failed") else 0)
