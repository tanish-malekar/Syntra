"""Generalized fail-over test: N pipeline workers + LoRA trainer on one Mac (MPS), real TinyLlama-1.1B.

env: SCRATCH, OUT, N (stages, default 2), KILL_IDX (default last), MODE (kill|hang), KILL_AFTER, STEPS,
     REQ_TIMEOUT (trainer request timeout, default 60), HANG_DETECT (seconds the 'orchestrator' needs to
     notice a frozen worker, default 8). Records worker RSS memory and per-step timing.
"""
import json, os, signal, subprocess, sys, time
from pathlib import Path

S = Path(os.environ["SCRATCH"]); OUT = Path(os.environ["OUT"]); OUT.mkdir(parents=True, exist_ok=True)
MODEL = S / "models" / "tinyllama"; BB = Path(os.environ.get("BLOOMBEE_DIR", "/Users/tanish/Desktop/Repos/BloomBee"))
N = int(os.environ.get("N", 2)); N_LAYERS = 22
KILL_IDX = int(os.environ.get("KILL_IDX", N - 1)); MODE = os.environ.get("MODE", "kill")
KILL_AFTER = int(os.environ.get("KILL_AFTER", 3)); STEPS = int(os.environ.get("STEPS", 10))
REQ_TIMEOUT = os.environ.get("REQ_TIMEOUT", "60"); HANG_DETECT = float(os.environ.get("HANG_DETECT", 8))
DATA = S / "deploy_data" / "train.jsonl"

from bloombee.data_structures import ServerState
from bloombee.utils.dht import get_remote_module_infos
from bloombee.utils.hivemind_compat import DHT

t0 = time.monotonic(); events = []
def ev(name, **kw):
    e = {"t": round(time.monotonic() - t0, 2), "event": name, **kw}; events.append(e); print(json.dumps(e), flush=True)

def ranges(n):
    base, extra = divmod(N_LAYERS, n); out, a = [], 0
    for i in range(n):
        b = a + base + (1 if i < extra else 0); out.append((a, b)); a = b
    return out
RANGES = ranges(N)

dht = DHT(start=True, host_maddrs=["/ip4/127.0.0.1/tcp/0"], use_relay=False)
peers = [str(a) for a in dht.get_visible_maddrs()]
prefix = "fo2_" + str(int(time.time())); procs = []

HUB_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REPL_FROM_HUB = os.environ.get("REPL_FROM_HUB", "0") == "1"   # replacement must download its weights

def worker(a, b, tag, from_hub=False):
    env = {**os.environ, "BLOOMBEE_ENABLE_LORA": "1", "BLOOMBEE_ENABLE_SPEC_PRUNER": "0", "HF_HUB_OFFLINE": "1"}
    model_arg, extra = str(MODEL), []
    if from_hub:  # empty cache: the worker has to fetch the model from Hugging Face first
        env.pop("HF_HUB_OFFLINE"); env["HF_HOME"] = str(OUT / "hf_home")
        model_arg, extra = HUB_ID, ["--cache_dir", str(OUT / "hf_cache")]
    cmd = [sys.executable, "-m", "bloombee.cli.run_server", model_arg, *extra, "--block_indices", f"{a}:{b}", "--dht_prefix", prefix,
           "--num_handlers", "1", "--batch_size", "2", "--torch_dtype", "float16", "--throughput", "100",
           "--update_period", "2", "--expiration", "15", "--inference_max_length", "256", "--attn_cache_tokens", "256",
           "--max_batch_size", "512", "--balance_quality", "0", "--no_auto_relay", "--host_maddrs", "/ip4/127.0.0.1/tcp/0",
           "--skip_reachability_check", "--cache_gpu_percent", "100", "--w_gpu_percent", "100", "--device", "mps",
           "--initial_peers", *peers]
    p = subprocess.Popen(cmd, env=env, stdout=(OUT / f"worker-{tag}.log").open("w"), stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(p); return p

def blocks_online(a, b):
    infos = get_remote_module_infos(dht, [f"{prefix}.{i}" for i in range(a, b)], latest=True)
    return all(any(s.state == ServerState.ONLINE for s in it.servers.values()) for it in infos)

def rss_mb(p):
    try: return round(int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(p.pid)]).split()[0]) / 1024)
    except Exception: return None

try:
    ws = [worker(a, b, str(i)) for i, (a, b) in enumerate(RANGES)]
    ev("workers_started", ranges=RANGES)
    dl = time.monotonic() + 420
    while time.monotonic() < dl and not blocks_online(0, N_LAYERS):
        if any(p.poll() is not None for p in ws): raise RuntimeError("worker exited early")
        time.sleep(1)
    ev("swarm_online", worker_rss_mb=[rss_mb(p) for p in ws])
    tlog = OUT / "trainer.log"
    tcmd = [sys.executable, "-u", str(BB / "examples/train_lora.py"), "--model", str(MODEL), "--initial_peers", *peers,
            "--dht_prefix", prefix, "--data", str(DATA), "--output", str(OUT / "adapter"), "--seq_len", "128", "--batch_size", "2",
            "--steps", str(STEPS), "--device", "mps", "--r", "8", "--lr", "2e-4", "--max_retries", "0",
            "--request_timeout", REQ_TIMEOUT, "--connect_timeout", "10", "--update_period", "2", "--max_backoff", "5"]
    trainer = subprocess.Popen(tcmd, stdout=tlog.open("w"), stderr=subprocess.STDOUT, env={**os.environ, "HF_HUB_OFFLINE": "1"}, start_new_session=True)
    procs.append(trainer); ev("trainer_started")
    seen, hit, t_hit, repl, t_online, step_times = 0, False, None, None, None, []
    dl = time.monotonic() + 900; last = time.monotonic()
    while time.monotonic() < dl and trainer.poll() is None:
        lines = [l for l in tlog.read_text().splitlines() if l.startswith("step=")]
        while seen < len(lines):
            now = time.monotonic(); step_times.append(round(now - last, 2)); last = now
            ev("train_step", line=lines[seen], since_prev=step_times[-1]); seen += 1
        if not hit and seen >= KILL_AFTER:
            ev("before_fault", worker_rss_mb=[rss_mb(p) for p in ws])
            victim = ws[KILL_IDX]; a, b = RANGES[KILL_IDX]
            os.killpg(victim.pid, signal.SIGKILL if MODE == "kill" else signal.SIGSTOP)
            t_hit = time.monotonic(); hit = True; ev("fault_injected", mode=MODE, worker=KILL_IDX, layers=f"{a}:{b}")
            time.sleep(1 if MODE == "kill" else HANG_DETECT)
            repl = worker(a, b, f"{KILL_IDX}r", from_hub=REPL_FROM_HUB); ev("replacement_spawned", from_hub=REPL_FROM_HUB, after_seconds=round(time.monotonic() - t_hit, 1))
        if hit and t_online is None and repl is not None:
            if repl.poll() is not None: ev("replacement_died"); break
            txt = (OUT / f"worker-{KILL_IDX}r.log").read_text()
            if "are joining" in txt and "Announced that blocks" in txt and blocks_online(*RANGES[KILL_IDX]) and time.monotonic() - t_hit > 3:
                t_online = time.monotonic(); ev("replacement_online", seconds_since_fault=round(t_online - t_hit, 1))
        time.sleep(0.5)
    rc = trainer.poll(); lines = [l for l in tlog.read_text().splitlines() if l.startswith("step=")]
    gap = None
    if hit and len(lines) > KILL_AFTER and len(step_times) > KILL_AFTER: gap = step_times[KILL_AFTER]
    ev("summary", N=N, mode=MODE, kill_idx=KILL_IDX, req_timeout=REQ_TIMEOUT, trainer_returncode=rc, steps_logged=len(lines),
       recovered=bool(hit and rc == 0 and len(lines) >= STEPS), first_step_after_fault_seconds=gap,
       normal_step_seconds=round(sum(step_times[1:KILL_AFTER]) / max(1, KILL_AFTER - 1), 2) if len(step_times) > 1 else None)
finally:
    for p in procs:
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: pass
    (OUT / "events.json").write_text(json.dumps(events, indent=1))
# DHT helper processes keep the script alive after the test: shut them down, then exit.
import multiprocessing as _mp
try: dht.shutdown()
except Exception: pass
for _c in _mp.active_children():
    try: _c.terminate()
    except Exception: pass
sys.stdout.flush(); os._exit(0)
