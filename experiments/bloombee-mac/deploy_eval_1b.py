"""Serve base + merged TinyLlama on N workers each (MPS), evaluate on held-out tickets, test sampling.

Usage: N=2 python deploy_eval_1b.py   (env SCRATCH, OUT, N)
"""
import json, os, re, signal, subprocess, sys, time
from pathlib import Path

S = Path(os.environ["SCRATCH"]); OUT = Path(os.environ["OUT"]); OUT.mkdir(parents=True, exist_ok=True)
N = int(os.environ.get("N", 1)); N_LAYERS = 22
BASE = S / "models" / "tinyllama"; MERGED = S / "models" / "tinyllama_merged"
HELD = [json.loads(l) for l in (S / "deploy_data" / "heldout.jsonl").read_text().splitlines()]
LIMIT = int(os.environ.get("LIMIT", len(HELD))); HELD = HELD[:LIMIT]

import torch
from bloombee import AutoDistributedModelForCausalLM
from bloombee.data_structures import ServerState
from bloombee.utils.dht import get_remote_module_infos
from bloombee.utils.hivemind_compat import DHT
from transformers import AutoTokenizer

dht = DHT(start=True, host_maddrs=["/ip4/127.0.0.1/tcp/0"], use_relay=False)
peers = [str(a) for a in dht.get_visible_maddrs()]
procs = []
t0 = time.monotonic()
def log(msg, **kw): print(json.dumps({"t": round(time.monotonic() - t0, 1), "msg": msg, **kw}), flush=True)

def ranges(n):
    base, extra = divmod(N_LAYERS, n); out, a = [], 0
    for i in range(n):
        b = a + base + (1 if i < extra else 0); out.append((a, b)); a = b
    return out

def serve(model_dir, prefix, tag):
    for i, (a, b) in enumerate(ranges(N)):
        env = {**os.environ, "BLOOMBEE_ENABLE_SPEC_PRUNER": "0", "HF_HUB_OFFLINE": "1"}
        cmd = [sys.executable, "-m", "bloombee.cli.run_server", str(model_dir), "--block_indices", f"{a}:{b}",
               "--dht_prefix", prefix, "--num_handlers", "1", "--batch_size", "1", "--torch_dtype", "float16",
               "--throughput", "100", "--update_period", "2", "--expiration", "15", "--inference_max_length", "256",
               "--attn_cache_tokens", "512", "--max_batch_size", "512", "--balance_quality", "0", "--no_auto_relay",
               "--host_maddrs", "/ip4/127.0.0.1/tcp/0", "--skip_reachability_check", "--cache_gpu_percent", "100",
               "--w_gpu_percent", "100", "--device", "mps", "--initial_peers", *peers]
        procs.append(subprocess.Popen(cmd, env=env, stdout=(OUT / f"{tag}-worker-{i}.log").open("w"),
                                      stderr=subprocess.STDOUT, start_new_session=True))

def wait_online(prefix):
    uids = [f"{prefix}.{i}" for i in range(N_LAYERS)]
    dl = time.monotonic() + 420
    while time.monotonic() < dl:
        if any(p.poll() is not None for p in procs): raise RuntimeError("worker exited; see logs")
        infos = get_remote_module_infos(dht, uids, latest=True)
        if all(any(s.state == ServerState.ONLINE for s in it.servers.values()) for it in infos): return
        time.sleep(1)
    raise TimeoutError(prefix)

def parse(text):
    m = re.search(r"\{.*?\}", text, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

def evaluate(model_dir, prefix, label, **gen):
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoDistributedModelForCausalLM.from_pretrained(str(model_dir), initial_peers=peers, dht_prefix=prefix,
                                                            torch_dtype=torch.float16, request_timeout=120).to("mps")
    res, ntok, secs, outs = {"valid_json": 0, "dept_correct": 0, "urgency_correct": 0, "n": 0}, 0, 0.0, []
    for ex in HELD:
        ids = tok(ex["prompt"], return_tensors="pt").input_ids.to("mps")
        t = time.monotonic()
        out = model.generate(ids, max_new_tokens=28, **gen)
        secs += time.monotonic() - t
        new = out[0, ids.shape[1]:]; ntok += len(new)
        text = tok.decode(new, skip_special_tokens=True); outs.append(text)
        js = parse(text); res["n"] += 1
        if js is not None:
            res["valid_json"] += 1
            res["dept_correct"] += int(js.get("department") == ex["department"])
            res["urgency_correct"] += int(js.get("urgency") == ex["urgency"])
    res.update(label=label, N=N, tokens_per_second=round(ntok / secs, 2), seconds=round(secs, 1), samples=outs[:4])
    return res, outs

results = {}
try:
    pb, pm = f"base_{int(time.time())}", f"merged_{int(time.time())}"
    serve(BASE, pb, "base"); serve(MERGED, pm, "merged"); log("workers_started", per_model=N)
    wait_online(pb); wait_online(pm); log("both_models_online")
    results["base_greedy"], base_out = evaluate(BASE, pb, "base greedy", do_sample=False); log("base_done", **{k: v for k, v in results["base_greedy"].items() if k != "samples"})
    results["merged_greedy"], merged_out = evaluate(MERGED, pm, "merged greedy", do_sample=False); log("merged_done", **{k: v for k, v in results["merged_greedy"].items() if k != "samples"})
    # repeatability: second greedy pass on first 8 prompts must match
    again = []
    tok = AutoTokenizer.from_pretrained(str(MERGED))
    model = AutoDistributedModelForCausalLM.from_pretrained(str(MERGED), initial_peers=peers, dht_prefix=pm, torch_dtype=torch.float16, request_timeout=120).to("mps")
    for ex in HELD[:8]:
        ids = tok(ex["prompt"], return_tensors="pt").input_ids.to("mps")
        o = model.generate(ids, max_new_tokens=28, do_sample=False)
        again.append(tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True))
    results["greedy_repeatable"] = again == merged_out[:8]
    # sampling works and varies
    samp = []
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        ids = tok(HELD[0]["prompt"], return_tensors="pt").input_ids.to("mps")
        o = model.generate(ids, max_new_tokens=28, do_sample=True, temperature=0.9, top_p=0.95, top_k=50)
        samp.append(tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True))
    results["sampling_samples"] = samp
    results["sampling_varies"] = len(set(samp)) > 1
    # the tuned model is near-deterministic, so also check sampling on the base model (high entropy)
    btok = AutoTokenizer.from_pretrained(str(BASE))
    bmodel = AutoDistributedModelForCausalLM.from_pretrained(str(BASE), initial_peers=peers, dht_prefix=pb, torch_dtype=torch.float16, request_timeout=120).to("mps")
    bsamp = []
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        ids = btok(HELD[0]["prompt"], return_tensors="pt").input_ids.to("mps")
        o = bmodel.generate(ids, max_new_tokens=28, do_sample=True, temperature=1.0, top_p=0.95, top_k=50)
        bsamp.append(btok.decode(o[0, ids.shape[1]:], skip_special_tokens=True))
    results["base_sampling_samples"] = bsamp
    results["base_sampling_varies"] = len(set(bsamp)) > 1
    # reference check: HF merged model locally vs served merged output
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(str(MERGED), dtype=torch.float16).to("mps").eval()
    match = 0
    for ex, served in list(zip(HELD, merged_out))[:8]:
        ids = tok(ex["prompt"], return_tensors="pt").input_ids.to("mps")
        with torch.no_grad():
            o = ref.generate(ids, max_new_tokens=28, do_sample=False)
        match += int(tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True) == served)
    results["served_matches_local_hf_out_of_8"] = match
    results["status"] = "done"
except Exception as e:
    results["status"] = "failed"; results["error"] = repr(e)[:400]
finally:
    for p in procs:
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: pass
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print("RESULTS", json.dumps(results), flush=True)
# DHT helper processes keep the script alive after the test: shut them down, then exit.
import multiprocessing as _mp
try: dht.shutdown()
except Exception: pass
for _c in _mp.active_children():
    try: _c.terminate()
    except Exception: pass
sys.stdout.flush(); os._exit(0)
