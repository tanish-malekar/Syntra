"""Connectivity spike: one LoRA training pipeline across this Mac (MPS) and context3 (CUDA).

  Mac (DHT, trainer, worker A: layers 0:11, MPS)  <== ssh tunnel ==>  context3 (worker B: layers 11:22, CUDA GPU)

Ports (same number on both machines through the tunnel): DHT 31001, Mac worker 31002, context3 worker 31003.
Usage:  SCRATCH=<dir with models/tinyllama and deploy_data/train.jsonl> CUDA_GPU=2 python bridge/mixed_spike.py
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ssh  # noqa: E402

SCRATCH = Path(os.environ["SCRATCH"])
OUT = Path(os.environ.get("OUT", SCRATCH / "mixed_out")); OUT.mkdir(parents=True, exist_ok=True)
BB_MAC = Path(os.environ.get("BLOOMBEE_DIR", "/Users/tanish/Desktop/Repos/BloomBee"))
C3 = "/data01/tmalekar/syntra"
GPU = os.environ.get("CUDA_GPU", "2")
STEPS = int(os.environ.get("STEPS", 6))
DHT_PORT, MAC_PORT, C3_PORT = 31001, 31002, 31003
PREFIX = f"mixed_{int(time.time())}"
MODEL_MAC = SCRATCH / "models" / "tinyllama"
MODEL_C3 = f"{C3}/models/tinyllama"

from bloombee.data_structures import ServerState  # noqa: E402
from bloombee.utils.dht import get_remote_module_infos  # noqa: E402
from bloombee.utils.hivemind_compat import DHT  # noqa: E402

t0 = time.monotonic()
def log(msg, **kw): print(json.dumps({"t": round(time.monotonic() - t0, 1), "msg": msg, **kw}), flush=True)

procs: list[subprocess.Popen] = []
def cleanup():
    for p in procs:
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            try: p.kill()
            except Exception: pass
    # context3 worker: only processes carrying this run's unique prefix
    ssh.run_remote(f"pkill -u tmalekar -f 'dht_prefix {PREFIX}' || true; sleep 1; pgrep -u tmalekar -f 'dht_prefix {PREFIX}' | wc -l", timeout=60)

result: dict = {"prefix": PREFIX}
try:
    # --- safety: only an idle context3 GPU ------------------------------------------------------------
    chk = ssh.run_remote(f"nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits -i {GPU}")
    util, mem = [int(x) for x in chk.stdout.strip().split(",")[1:3]]
    log("context3 GPU state", gpu=GPU, util=util, mem_mib=mem)
    if util > 10 or mem > 1500:
        raise SystemExit(f"context3 GPU {GPU} is not idle (util {util}%, {mem} MiB): refusing to use it")

    dht = DHT(start=True, host_maddrs=[f"/ip4/127.0.0.1/tcp/{DHT_PORT}"], use_relay=False)
    peers = [str(a) for a in dht.get_visible_maddrs()]
    log("DHT up", peers=peers)

    tun = ssh.tunnel([C3_PORT], [DHT_PORT, MAC_PORT]); procs.append(tun)
    time.sleep(4)
    if tun.poll() is not None:
        raise RuntimeError("ssh tunnel died: " + tun.stderr.read().decode()[-400:])
    log("tunnel up")

    common = ["--block_indices", "", "--dht_prefix", PREFIX, "--num_handlers", "1", "--batch_size", "2", "--torch_dtype", "float16",
              "--throughput", "100", "--update_period", "2", "--expiration", "15", "--inference_max_length", "256",
              "--attn_cache_tokens", "512", "--max_batch_size", "512", "--balance_quality", "0", "--no_auto_relay",
              "--skip_reachability_check", "--cache_gpu_percent", "100", "--w_gpu_percent", "100"]
    def flags(layers, port, device):
        f = list(common); f[1] = layers
        return f + ["--device", device, "--host_maddrs", f"/ip4/127.0.0.1/tcp/{port}", "--initial_peers", *peers]

    # --- worker B on context3 (CUDA) --------------------------------------------------------------------
    argv_c3 = " ".join(["/data01/tmalekar/conda-envs/bb/bin/python", "-m", "bloombee.cli.run_server", MODEL_C3, *flags("11:22", C3_PORT, "cuda")])
    r = ssh.run_remote(f"""set -e
cd {C3}/BloomBee
export PYTHONPATH={C3}/BloomBee/src CUDA_VISIBLE_DEVICES={GPU} BLOOMBEE_ENABLE_LORA=1 BLOOMBEE_ENABLE_SPEC_PRUNER=0 HF_HUB_OFFLINE=1 HF_HOME={C3}/hf TMPDIR={C3}/tmp
nohup setsid {argv_c3} > {C3}/out/worker_{PREFIX}.log 2>&1 < /dev/null &
echo started $!""")
    log("context3 worker launch", out=r.stdout.strip(), err=r.stderr.strip()[-200:])

    # --- worker A on the Mac (MPS) ----------------------------------------------------------------------
    env = {**os.environ, "PYTHONPATH": f"{BB_MAC}/src", "BLOOMBEE_ENABLE_LORA": "1", "BLOOMBEE_ENABLE_SPEC_PRUNER": "0", "HF_HUB_OFFLINE": "1"}
    wa = subprocess.Popen([sys.executable, "-m", "bloombee.cli.run_server", str(MODEL_MAC), *flags("0:11", MAC_PORT, "mps")],
                          env=env, stdout=(OUT / "worker_mac.log").open("w"), stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(wa)

    uids = [f"{PREFIX}.{i}" for i in range(22)]
    deadline = time.monotonic() + 300
    online = False
    while time.monotonic() < deadline:
        if wa.poll() is not None: raise RuntimeError("Mac worker exited early")
        infos = get_remote_module_infos(dht, uids, latest=True)
        have = [i for i, it in enumerate(infos) if any(s.state == ServerState.ONLINE for s in it.servers.values())]
        if len(have) == 22: online = True; break
        time.sleep(2)
    log("layers online", all_22=online, mac_layers_online=len([h for h in have if h < 11]), context3_layers_online=len([h for h in have if h >= 11]))
    result["all_layers_online"] = online
    if not online:
        raise RuntimeError("not all layers came online (see worker logs)")

    # --- train across both machines --------------------------------------------------------------------
    tcmd = [sys.executable, "-u", str(BB_MAC / "examples/train_lora.py"), "--model", str(MODEL_MAC), "--initial_peers", *peers,
            "--dht_prefix", PREFIX, "--data", str(SCRATCH / "deploy_data" / "train.jsonl"), "--output", str(OUT / "adapter"),
            "--seq_len", "128", "--batch_size", "2", "--steps", str(STEPS), "--device", "mps", "--r", "8", "--lr", "2e-4",
            "--max_retries", "0", "--request_timeout", "120", "--connect_timeout", "20", "--update_period", "2", "--max_backoff", "5"]
    tlog = OUT / "trainer.log"
    tp = subprocess.Popen(tcmd, env={**env, "PYTHONPATH": f"{BB_MAC}/src"}, stdout=tlog.open("w"), stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(tp)
    stamps = []
    seen = 0
    while tp.poll() is None and time.monotonic() < deadline + 600:
        lines = [l for l in tlog.read_text().splitlines() if l.startswith("step=")]
        while seen < len(lines):
            stamps.append(round(time.monotonic() - t0, 1)); log("train_step", line=lines[seen]); seen += 1
        time.sleep(0.5)
    lines = [l for l in tlog.read_text().splitlines() if l.startswith("step=")]
    result.update(trainer_returncode=tp.poll(), steps=len(lines), first_loss=lines[0] if lines else None, last_loss=lines[-1] if lines else None,
                  step_seconds=[round(b - a, 1) for a, b in zip(stamps, stamps[1:])])
    r = ssh.run_remote(f"grep -c 'rpc_forward\\|LoRA' {C3}/out/worker_{PREFIX}.log || true; tail -n 3 {C3}/out/worker_{PREFIX}.log | cut -c1-160")
    result["context3_worker_log_tail"] = r.stdout.strip()[-400:]
    log("RESULT", **result)
finally:
    cleanup()
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    sys.stdout.flush()
    os._exit(0)
