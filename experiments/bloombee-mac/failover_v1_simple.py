"""Kill-and-replace test: real 1B model, 2 pipeline workers + LoRA trainer on one Mac (MPS).

Timeline: start swarm -> train -> SIGKILL worker 1 after step KILL_AFTER -> start a fresh
replacement worker for the same layers -> confirm training continues. Prints a stage-by-stage
timeline. Run with PYTHONPATH=<BloomBee>/src.
"""
import json, os, signal, subprocess, sys, threading, time
from pathlib import Path

S = Path(os.environ["SCRATCH"])
MODEL = S / "models" / "tinyllama"
OUT = Path(os.environ.get("OUT", S / "failover_out"))
BB = Path(os.environ.get("BLOOMBEE_DIR", "/Users/tanish/Desktop/Repos/BloomBee"))
KILL_AFTER = int(os.environ.get("KILL_AFTER", 3))
STEPS = int(os.environ.get("STEPS", 8))
TUNED = os.environ.get("TUNED", "1") == "1"
N_LAYERS = 22
SPLIT = 11
OUT.mkdir(parents=True, exist_ok=True)

from bloombee.data_structures import ServerState
from bloombee.utils.dht import get_remote_module_infos
from bloombee.utils.hivemind_compat import DHT

t0 = time.monotonic()
events = []
def ev(name, **kw):
    e = {"t": round(time.monotonic() - t0, 2), "event": name, **kw}
    events.append(e)
    print(json.dumps(e), flush=True)

# --- tiny synthetic support-ticket dataset
data = OUT / "train.jsonl"
cats = [("billing", "refund for a double charge"), ("login", "cannot reset my password"),
        ("shipping", "package has not arrived"), ("bug", "app crashes when I upload a photo")]
with data.open("w") as f:
    for i in range(120):
        c, d = cats[i % 4]
        urg = ["low", "medium", "high"][i % 3]
        f.write(json.dumps({"text": f"Ticket: Hi, I need help, {d} (order {1000+i}).\n"
                                    f"Answer: {{\"department\": \"{c}\", \"urgency\": \"{urg}\"}}"}) + "\n")

dht = DHT(start=True, host_maddrs=["/ip4/127.0.0.1/tcp/0"], use_relay=False)
peers = [str(a) for a in dht.get_visible_maddrs()]
prefix = "fo_" + str(int(time.time()))
procs = []

def start_worker(start, end, tag):
    log = (OUT / f"worker-{tag}.log").open("w")
    env = {**os.environ, "BLOOMBEE_ENABLE_LORA": "1", "BLOOMBEE_ENABLE_SPEC_PRUNER": "0",
           "BLOOMBEE_ENABLE_MICROBATCH_PIPELINE": "0", "BLOOMBEE_LOSSLESS_WRAPPER": "0", "HF_HUB_OFFLINE": "1"}
    cmd = [sys.executable, "-m", "bloombee.cli.run_server", str(MODEL), "--block_indices", f"{start}:{end}",
           "--dht_prefix", prefix, "--num_handlers", "1", "--batch_size", "2", "--torch_dtype", "float16",
           "--throughput", "100", "--update_period", "2" if TUNED else "60", "--expiration", "15" if TUNED else "30",
           "--inference_max_length", "256", "--attn_cache_tokens", "256", "--max_batch_size", "512",
           "--balance_quality", "0", "--no_auto_relay", "--host_maddrs", "/ip4/127.0.0.1/tcp/0",
           "--skip_reachability_check", "--cache_gpu_percent", "100", "--w_gpu_percent", "100",
           "--device", "mps", "--initial_peers", *peers]
    p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(p)
    return p

def blocks_online(start, end):
    uids = [f"{prefix}.{i}" for i in range(start, end)]
    infos = get_remote_module_infos(dht, uids, latest=True)
    return all(any(s.state == ServerState.ONLINE for s in it.servers.values()) for it in infos)

def wait_online(start, end, deadline, procs_alive=()):
    while time.monotonic() < deadline:
        if any(p.poll() is not None for p in procs_alive):
            raise RuntimeError("a worker exited early; see logs")
        if blocks_online(start, end):
            return True
        time.sleep(1)
    raise TimeoutError(f"blocks {start}:{end} not online")

def kill_all():
    for p in procs:
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: pass

try:
    w0 = start_worker(0, SPLIT, "0")
    w1 = start_worker(SPLIT, N_LAYERS, "1")
    ev("workers_started")
    wait_online(0, N_LAYERS, time.monotonic() + 420, (w0, w1))
    ev("swarm_online", layers=N_LAYERS)

    trainer_log = OUT / "trainer.log"
    tcmd = [sys.executable, "-u", str(BB / "examples/train_lora.py"), "--model", str(MODEL),
            "--initial_peers", *peers, "--dht_prefix", prefix, "--data", str(data), "--output", str(OUT / "adapter"),
            "--seq_len", "128", "--batch_size", "2", "--steps", str(STEPS), "--device", "mps",
            "--r", "8", "--lr", "2e-4", "--max_retries", "0", "--request_timeout", "60", "--connect_timeout", "10"]
    if TUNED:
        tcmd += ["--update_period", "2", "--max_backoff", "5"]
    trainer = subprocess.Popen(tcmd, stdout=trainer_log.open("w"), stderr=subprocess.STDOUT,
                               env={**os.environ, "HF_HUB_OFFLINE": "1"}, start_new_session=True)
    procs.append(trainer)
    ev("trainer_started")

    seen, killed, t_kill, replacement, t_online, resume_logged = 0, False, None, None, None, False
    steps_after_kill = []
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline and trainer.poll() is None:
        lines = [l for l in trainer_log.read_text().splitlines() if l.startswith("step=")]
        while seen < len(lines):
            ev("train_step", line=lines[seen]); seen += 1
            if killed: steps_after_kill.append(seen)
        if not killed and seen >= KILL_AFTER:
            os.killpg(w1.pid, signal.SIGKILL); t_kill = time.monotonic(); killed = True
            ev("worker_killed", layers=f"{SPLIT}:{N_LAYERS}")
            time.sleep(1)  # agent notices process exit
            replacement = start_worker(SPLIT, N_LAYERS, "1b"); t_spawn = time.monotonic()
            ev("replacement_spawned")
        if killed and t_online is None and replacement is not None:
            if replacement.poll() is not None:
                ev("replacement_died"); break
            rtxt = (OUT / 'worker-1b.log').read_text() if (OUT / 'worker-1b.log').exists() else ''
            if 'are joining' in rtxt and blocks_online(SPLIT, N_LAYERS) and 'Announced that blocks' in rtxt and time.monotonic() - t_kill > 3:
                t_online = time.monotonic(); ev("replacement_online", seconds_since_kill=round(t_online - t_kill, 1))
        time.sleep(0.5)
    rc = trainer.poll()
    lines = [l for l in trainer_log.read_text().splitlines() if l.startswith("step=")]
    ev("trainer_finished", returncode=rc, steps_logged=len(lines))
    summary = {"tuned": TUNED, "killed_after_step": KILL_AFTER, "steps_logged": len(lines),
               "trainer_returncode": rc, "recovered": bool(killed and len(lines) > KILL_AFTER)}
    ev("summary", **summary)
finally:
    kill_all()
    (OUT / "events.json").write_text(json.dumps(events, indent=1))
