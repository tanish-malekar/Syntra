"""Train a LoRA adapter on TinyLlama over 2 pipeline workers (MPS). Saves adapter to OUT/adapter."""
import json, os, signal, subprocess, sys, time
from pathlib import Path

S = Path(os.environ["SCRATCH"]); OUT = Path(os.environ["OUT"]); OUT.mkdir(parents=True, exist_ok=True)
MODEL = S / "models" / "tinyllama"; BB = Path(os.environ.get("BLOOMBEE_DIR", "/Users/tanish/Desktop/Repos/BloomBee"))
DATA = Path(os.environ["DATA"]); STEPS = int(os.environ.get("STEPS", 80))

from bloombee.data_structures import ServerState
from bloombee.utils.dht import get_remote_module_infos
from bloombee.utils.hivemind_compat import DHT

dht = DHT(start=True, host_maddrs=["/ip4/127.0.0.1/tcp/0"], use_relay=False)
peers = [str(a) for a in dht.get_visible_maddrs()]
prefix = "tr_" + str(int(time.time())); procs = []
def worker(a, b, tag):
    env = {**os.environ, "BLOOMBEE_ENABLE_LORA": "1", "BLOOMBEE_ENABLE_SPEC_PRUNER": "0", "HF_HUB_OFFLINE": "1"}
    cmd = [sys.executable, "-m", "bloombee.cli.run_server", str(MODEL), "--block_indices", f"{a}:{b}", "--dht_prefix", prefix,
           "--num_handlers", "1", "--batch_size", "4", "--torch_dtype", "float16", "--throughput", "100",
           "--update_period", "2", "--expiration", "15", "--inference_max_length", "256", "--attn_cache_tokens", "256",
           "--max_batch_size", "512", "--balance_quality", "0", "--no_auto_relay", "--host_maddrs", "/ip4/127.0.0.1/tcp/0",
           "--skip_reachability_check", "--cache_gpu_percent", "100", "--w_gpu_percent", "100", "--device", "mps",
           "--initial_peers", *peers]
    p = subprocess.Popen(cmd, env=env, stdout=(OUT / f"worker-{tag}.log").open("w"), stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(p); return p
try:
    ws = [worker(0, 11, "0"), worker(11, 22, "1")]
    uids = [f"{prefix}.{i}" for i in range(22)]
    dl = time.monotonic() + 420
    while time.monotonic() < dl:
        if any(p.poll() is not None for p in ws): raise RuntimeError("worker exited")
        infos = get_remote_module_infos(dht, uids, latest=True)
        if all(any(s.state == ServerState.ONLINE for s in it.servers.values()) for it in infos): break
        time.sleep(1)
    cmd = [sys.executable, "-u", str(BB / "examples/train_lora.py"), "--model", str(MODEL), "--initial_peers", *peers,
           "--dht_prefix", prefix, "--data", str(DATA), "--output", str(OUT / "adapter"), "--seq_len", "128",
           "--batch_size", "4", "--steps", str(STEPS), "--device", "mps", "--r", "16", "--lora_alpha", "32", "--lr", "3e-4",
           "--target_modules", "q_proj", "k_proj", "v_proj", "o_proj", "--max_retries", "0", "--request_timeout", "120",
           "--connect_timeout", "15", "--update_period", "2", "--max_backoff", "5"]
    t = time.monotonic()
    r = subprocess.run(cmd, stdout=(OUT / "trainer.log").open("w"), stderr=subprocess.STDOUT,
                       env={**os.environ, "HF_HUB_OFFLINE": "1"})
    print(json.dumps({"trainer_returncode": r.returncode, "train_seconds": round(time.monotonic() - t, 1)}))
finally:
    for p in procs:
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: pass
# DHT helper processes keep the script alive after the test: shut them down, then exit.
import multiprocessing as _mp
try: dht.shutdown()
except Exception: pass
for _c in _mp.active_children():
    try: _c.terminate()
    except Exception: pass
sys.stdout.flush(); os._exit(0)
