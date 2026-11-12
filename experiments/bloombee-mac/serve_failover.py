"""Serve-path fail-over: kill a serving worker mid-generation and start a replacement on demand.

TinyLlama base on 2 workers (MPS). Run 1: uninterrupted greedy generation (baseline). Run 2: same prompt, but worker 1
is SIGKILLed a few seconds in and a fresh replacement is started 1 s later. Checks that generation completes and the
text equals the baseline. env: SCRATCH, OUT, KILL_AT (seconds into generation, default 3), NEW_TOKENS (default 200),
REPL_FROM_HUB=1 (replacement downloads weights instead of reading local files).
"""
import json, os, signal, subprocess, sys, threading, time
from pathlib import Path

S = Path(os.environ["SCRATCH"]); OUT = Path(os.environ["OUT"]); OUT.mkdir(parents=True, exist_ok=True)
MODEL = S / "models" / "tinyllama"; N_LAYERS = 22; SPLIT = 11
KILL_AT = float(os.environ.get("KILL_AT", 3)); NEW_TOKENS = int(os.environ.get("NEW_TOKENS", 200))
FROM_HUB = os.environ.get("REPL_FROM_HUB", "0") == "1"
HUB_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

import torch
from bloombee import AutoDistributedModelForCausalLM
from bloombee.data_structures import ServerState
from bloombee.utils.dht import get_remote_module_infos
from bloombee.utils.hivemind_compat import DHT
from transformers import AutoTokenizer

t0 = time.monotonic(); events = []
def ev(name, **kw):
    e = {"t": round(time.monotonic() - t0, 2), "event": name, **kw}; events.append(e); print(json.dumps(e), flush=True)

dht = DHT(start=True, host_maddrs=["/ip4/127.0.0.1/tcp/0"], use_relay=False)
peers = [str(a) for a in dht.get_visible_maddrs()]
prefix = "sf_" + str(int(time.time())); procs = []

def worker(a, b, tag, from_hub=False):
    env = {**os.environ, "BLOOMBEE_ENABLE_SPEC_PRUNER": "0", "HF_HUB_OFFLINE": "1"}
    model_arg, extra = str(MODEL), []
    if from_hub:
        env.pop("HF_HUB_OFFLINE"); env["HF_HOME"] = str(OUT / "hf_home"); model_arg, extra = HUB_ID, ["--cache_dir", str(OUT / "hf_cache")]
    cmd = [sys.executable, "-m", "bloombee.cli.run_server", model_arg, *extra, "--block_indices", f"{a}:{b}", "--dht_prefix", prefix,
           "--num_handlers", "1", "--batch_size", "1", "--torch_dtype", "float16", "--throughput", "100", "--update_period", "2",
           "--expiration", "15", "--inference_max_length", "512", "--attn_cache_tokens", "1024", "--max_batch_size", "512",
           "--balance_quality", "0", "--no_auto_relay", "--host_maddrs", "/ip4/127.0.0.1/tcp/0", "--skip_reachability_check",
           "--cache_gpu_percent", "100", "--w_gpu_percent", "100", "--device", "mps", "--initial_peers", *peers]
    p = subprocess.Popen(cmd, env=env, stdout=(OUT / f"worker-{tag}.log").open("w"), stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(p); return p

def online(a, b):
    infos = get_remote_module_infos(dht, [f"{prefix}.{i}" for i in range(a, b)], latest=True)
    return all(any(s.state == ServerState.ONLINE for s in it.servers.values()) for it in infos)

results = {}
try:
    w0 = worker(0, SPLIT, "0"); w1 = worker(SPLIT, N_LAYERS, "1")
    dl = time.monotonic() + 300
    while time.monotonic() < dl and not online(0, N_LAYERS):
        if w0.poll() is not None or w1.poll() is not None: raise RuntimeError("worker exited early")
        time.sleep(1)
    ev("swarm_online")
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    model = AutoDistributedModelForCausalLM.from_pretrained(str(MODEL), initial_peers=peers, dht_prefix=prefix, torch_dtype=torch.float16,
        request_timeout=float(os.environ.get("REQ_TIMEOUT", 60)), connect_timeout=10, update_period=2, max_backoff=5).to("mps")
    prompt = "Write a long, detailed explanation of how a bicycle works, covering gears, brakes and steering.\n"
    ids = tok(prompt, return_tensors="pt").input_ids.to("mps")

    def run(box):
        try:
            out = model.generate(ids, max_new_tokens=NEW_TOKENS, do_sample=False)
            box["text"] = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True); box["tokens"] = out.shape[1] - ids.shape[1]
        except Exception as e:
            box["error"] = repr(e)[:300]

    base = {}; t = time.monotonic(); run(base); base["seconds"] = round(time.monotonic() - t, 1)
    ev("baseline_done", seconds=base["seconds"], tokens=base.get("tokens"), error=base.get("error"))

    box = {}; th = threading.Thread(target=run, args=(box,), daemon=True); t_start = time.monotonic(); th.start()
    time.sleep(KILL_AT)
    if not th.is_alive(): raise RuntimeError("generation finished before the fault; raise NEW_TOKENS")
    os.killpg(w1.pid, signal.SIGKILL); t_kill = time.monotonic(); ev("worker_killed", after_generation_seconds=KILL_AT)
    time.sleep(1); repl = worker(SPLIT, N_LAYERS, "1r", from_hub=FROM_HUB); ev("replacement_spawned", from_hub=FROM_HUB)
    online_at = None
    while th.is_alive() and time.monotonic() - t_start < 420:
        if online_at is None and "are joining" in (OUT / "worker-1r.log").read_text() and online(SPLIT, N_LAYERS) and time.monotonic() - t_kill > 3:
            online_at = time.monotonic(); ev("replacement_online", seconds_since_kill=round(online_at - t_kill, 1))
        time.sleep(0.5)
    hung = th.is_alive()
    total = round(time.monotonic() - t_start, 1)
    ev("generation_finished", hung=hung, total_seconds=total, tokens=box.get("tokens"), error=box.get("error"))
    results = {"completed": (not hung) and "text" in box, "hung": hung, "error": box.get("error"), "baseline_seconds": base["seconds"],
               "fault_run_seconds": total, "extra_seconds_vs_baseline": round(total - base["seconds"], 1),
               "text_equals_baseline": box.get("text") == base.get("text"), "tokens": box.get("tokens"),
               "replacement_online_seconds_after_kill": round(online_at - t_kill, 1) if online_at else None, "repl_from_hub": FROM_HUB}
    if box.get("text") and base.get("text") and box["text"] != base["text"]:
        n = next((i for i, (x, y) in enumerate(zip(box["text"], base["text"])) if x != y), None); results["first_diff_char"] = n
    ev("summary", **results)
except Exception as e:
    results = {"status": "failed", "error": repr(e)[:400]}; print("FAILED", results, flush=True)
finally:
    for p in procs:
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: pass
    (OUT / "results.json").write_text(json.dumps({"events": events, "results": results}, indent=1))
# DHT helper processes keep the script alive after the test: shut them down, then exit.
import multiprocessing as _mp
try: dht.shutdown()
except Exception: pass
for _c in _mp.active_children():
    try: _c.terminate()
    except Exception: pass
sys.stdout.flush(); os._exit(0)
