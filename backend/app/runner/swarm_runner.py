"""Swarm runner: the per-job process that needs BloomBee (torch, hivemind). Runs under SYNTRA_BLOOMBEE_PYTHON.

It is deliberately standalone (stdlib + BloomBee only; it does not import the backend) and talks to the backend through
JSON lines on stdout (events) and, in serve mode, stdin (requests).

  python swarm_runner.py <spec.json>

train mode:  start a DHT -> {"type":"dht_ready"} -> wait until every layer is online -> {"type":"swarm_ready"} ->
             run examples/train_lora.py -> {"type":"train.step",...} per step -> {"type":"train.finished"}
serve mode:  start a DHT -> wait for the layers -> load the client -> answer {"op":"generate",...} requests.
Workers are started by the backend through provider agents; this process only hosts the DHT and the client.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

spec = json.loads(Path(sys.argv[1]).read_text())
bb_dir = Path(os.path.expanduser(spec["bloombee_dir"]))
sys.path.insert(0, str(bb_dir / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_out_lock = threading.Lock()


def emit(obj: dict) -> None:
    with _out_lock:
        print(json.dumps(obj), flush=True)


children: list[subprocess.Popen] = []


def kill_group(pgid: int, sig: int = signal.SIGKILL) -> None:
    """Kill a process group but spare Python's multiprocessing resource_tracker: on macOS it must outlive its parent to
    unlink the named semaphores, or they leak until reboot (see agent/syntra_agent/workers.py kill_process_group)."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,pgid=,command="], capture_output=True, text=True, timeout=5).stdout
        members = [l.split(None, 2) for l in out.splitlines()]
        members = [(int(m[0]), m[2] if len(m) > 2 else "") for m in members if len(m) >= 2 and m[1] == str(pgid)]
    except Exception:
        members = []
    if not members:
        try:
            os.killpg(pgid, sig)
        except Exception:
            pass
        return
    for pid, command in members:
        if "multiprocessing.resource_tracker" not in command:
            try:
                os.kill(pid, sig)
            except Exception:
                pass


def _shutdown(*_):
    for c in children:
        try:
            kill_group(c.pid)
        except Exception:
            pass
    sys.stdout.flush()
    os._exit(143)


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)

from bloombee.data_structures import ServerState  # noqa: E402
from bloombee.utils.dht import get_remote_module_infos  # noqa: E402
from bloombee.utils.hivemind_compat import DHT  # noqa: E402

port = int(spec.get("dht_port") or 0)
dht = DHT(start=True, host_maddrs=[f"/ip4/127.0.0.1/tcp/{port}"], use_relay=False)
peers = [str(a) for a in dht.get_visible_maddrs()]
emit({"type": "dht_ready", "initial_peers": peers})

prefix, num_layers = spec["dht_prefix"], int(spec["num_layers"])
uids = [f"{prefix}.{i}" for i in range(num_layers)]


def layers_online() -> list[int]:
    infos = get_remote_module_infos(dht, uids, latest=True)
    return [i for i, it in enumerate(infos) if any(s.state == ServerState.ONLINE for s in it.servers.values())]


deadline = time.monotonic() + float(spec.get("ready_timeout", 900))
last_report, have = 0.0, []
while time.monotonic() < deadline:
    have = layers_online()
    if len(have) == num_layers:
        break
    if time.monotonic() - last_report > 5:
        emit({"type": "swarm_waiting", "layers_online": len(have), "layers_total": num_layers})
        last_report = time.monotonic()
    time.sleep(1.0)
else:
    emit({"type": "swarm_timeout", "layers_online": len(have), "layers_total": num_layers})
    os._exit(3)
emit({"type": "swarm_ready", "layers_total": num_layers})


def pick_device() -> str:
    want = spec.get("client_device", "auto")
    if want != "auto":
        return want
    import torch
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def run_train() -> int:
    out = Path(spec["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(bb_dir / "examples" / "train_lora.py"), "--model", spec["model_path"],
           "--initial_peers", *peers, "--dht_prefix", prefix, "--data", spec["dataset_path"], "--output", str(out / "adapter"),
           "--seq_len", str(spec.get("seq_len", 128)), "--batch_size", str(spec.get("batch_size", 2)),
           "--steps", str(spec["steps"]), "--device", pick_device(), "--r", str(spec.get("rank", 8)),
           "--lr", str(spec.get("lr", 2e-4)), "--checkpoint_every", str(spec.get("checkpoint_every", 0)),
           "--max_retries", "0", "--request_timeout", str(spec.get("request_timeout", 60)), "--connect_timeout", "10",
           "--update_period", "2", "--max_backoff", "5"]
    if spec.get("lora_alpha"):
        cmd += ["--lora_alpha", str(spec["lora_alpha"])]
    if spec.get("init_adapter"):  # continue from a stored model (its rank and alpha come from the adapter)
        cmd += ["--resume_adapter", spec["init_adapter"]]
    log_path = out / "trainer.log"
    logf = open(log_path, "w")
    # Output goes to a file that we tail: BloomBee's helper processes inherit pipes and keep them open after the trainer
    # has exited, so waiting for EOF on a pipe would hang forever.
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True, start_new_session=True,
                            env={**os.environ, "PYTHONPATH": f"{bb_dir / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"})
    children.append(proc)
    last = time.monotonic()
    tail: list[str] = []
    reader = open(log_path, "r")

    def drain() -> None:
        nonlocal last, tail
        for line in reader.readlines():
            line = line.rstrip()
            tail = (tail + [line])[-12:]
            if line.startswith("step="):
                try:
                    kv = dict(p.split("=") for p in line.split())
                    now = time.monotonic()
                    emit({"type": "train.step", "step": int(kv["step"]), "loss": float(kv["loss"]),
                          "grad_norm": float(kv["grad_norm"]), "step_seconds": round(now - last, 3)})
                    last = now
                except (ValueError, KeyError):
                    pass

    while proc.poll() is None:
        drain()
        time.sleep(0.2)
    drain()
    try:  # the trainer's own helper processes (DHT, p2pd) must not outlive it
        kill_group(proc.pid)
    except Exception:
        pass
    rc = proc.returncode
    emit({"type": "train.finished", "returncode": rc, "adapter_dir": str(out / "adapter"), "tail": tail if rc else []})
    return rc


def run_serve() -> int:
    import torch
    from transformers import AutoTokenizer
    from bloombee import AutoDistributedModelForCausalLM

    device = pick_device()
    tok = AutoTokenizer.from_pretrained(spec["model_path"])
    model = AutoDistributedModelForCausalLM.from_pretrained(
        spec["model_path"], initial_peers=peers, dht_prefix=prefix, torch_dtype=torch.float16,
        request_timeout=float(spec.get("request_timeout", 15)), connect_timeout=10, update_period=2, max_backoff=5,
    ).to(device)
    emit({"type": "serve_ready", "device": device})
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        req = json.loads(raw)
        if req.get("op") == "shutdown":
            break
        rid = req.get("id")
        try:
            t = time.monotonic()
            ids = tok(req["prompt"], return_tensors="pt").input_ids.to(device)
            kwargs = {"max_new_tokens": int(req.get("max_new_tokens", 32)), "do_sample": bool(req.get("do_sample", False))}
            if kwargs["do_sample"]:
                if req.get("seed") is not None:
                    torch.manual_seed(int(req["seed"]))
                for k in ("temperature", "top_p", "top_k"):
                    if req.get(k) is not None:
                        kwargs[k] = req[k]
            out = model.generate(ids, **kwargs)
            new = out[0, ids.shape[1]:]
            emit({"type": "result", "id": rid, "text": tok.decode(new, skip_special_tokens=True), "tokens": int(len(new)),
                  "seconds": round(time.monotonic() - t, 3)})
        except Exception as exc:  # report to the caller, keep serving
            emit({"type": "error", "id": rid, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
    return 0


def build_engine():
    """A DistributedLoRA client on the swarm (fresh adapter, or continuing from a stored one) wrapped in the RL engine."""
    import torch
    from transformers import AutoTokenizer
    from bloombee import AutoDistributedModelForCausalLM, DistributedLoRA, LoRAConfig
    from rl_engine import Engine

    device = pick_device()
    tok = AutoTokenizer.from_pretrained(spec["model_path"])
    base = AutoDistributedModelForCausalLM.from_pretrained(
        spec["model_path"], initial_peers=peers, dht_prefix=prefix, torch_dtype=torch.float16,
        request_timeout=float(spec.get("request_timeout", 60)), connect_timeout=10, update_period=2, max_backoff=5,
        max_retries=None,  # retry forever: a dead worker is replaced by the platform and the call continues
    ).to(device)
    if spec.get("init_adapter"):
        lora = DistributedLoRA.from_pretrained(base, spec["init_adapter"])
    else:
        lora = DistributedLoRA(base, LoRAConfig(int(spec.get("rank", 8)), float(spec.get("lora_alpha", 16)), 0.0,
                                                tuple(spec.get("target_modules", ("q_proj", "v_proj")))))
    lora.to(device).train()
    params = dict(lora._adapter_state())
    engine = Engine(lambda ids: lora(input_ids=ids, use_cache=False).logits, params, tok, device,
                    save_fn=lambda path: lora.save_pretrained(path), max_seq_len=int(spec.get("max_seq_len", 1024)))
    return engine


def run_session() -> int:
    """Tinker-style training session: one request per stdin line {"op", "id", "args"}, one reply per request."""
    from rl_engine import EngineError

    engine = build_engine()
    emit({"type": "serve_ready", "device": pick_device(), **engine.state()})
    saves = Path(spec["output_dir"]) / "saves"
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        req = json.loads(raw)
        if req.get("op") == "shutdown":
            break
        rid, op, args = req.get("id"), req.get("op"), dict(req.get("args") or {})
        t = time.monotonic()
        try:
            if op == "save":  # where adapters go is decided here, not by the caller
                name = "".join(c for c in str(args.get("name", "")) if c.isalnum() or c in "-_")[:60] or f"step{engine.steps}"
                args = {"out_dir": str(saves / f"{name}-{int(time.time())}")}
            data = engine.call(op, args)
            emit({"type": "result", "id": rid, "data": data, "seconds": round(time.monotonic() - t, 3)})
        except EngineError as exc:
            emit({"type": "error", "id": rid, "error": str(exc), "bad_request": True})
        except Exception as exc:  # a swarm failure the client could not recover from: report, keep the session alive
            emit({"type": "error", "id": rid, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
    return 0


def run_dpo() -> int:
    """Managed DPO: pairs {"prompt", "chosen", "rejected"} from the dataset file; reference = the adapter at the start."""
    import random

    engine = build_engine()
    rows = [json.loads(l) for l in Path(spec["dataset_path"]).read_text().splitlines() if l.strip()]
    pairs = []
    for r in rows:
        prompt = engine.tok(r["prompt"])["input_ids"]
        pairs.append({"chosen": {"tokens": prompt + engine.tok(r["chosen"], add_special_tokens=False)["input_ids"], "prompt_len": len(prompt)},
                      "rejected": {"tokens": prompt + engine.tok(r["rejected"], add_special_tokens=False)["input_ids"], "prompt_len": len(prompt)}})
    batch_size, beta = int(spec.get("batch_size", 2)), float(spec.get("beta", 0.1))
    refs = []
    for i in range(0, len(pairs), batch_size):  # reference log-probs once, before any update
        refs.extend(engine.reference_sums(pairs[i: i + batch_size]))
    for p, (c, r) in zip(pairs, refs):
        p["ref_chosen"], p["ref_rejected"] = c, r
    rng = random.Random(int(spec.get("seed", 42)))
    last = time.monotonic()
    for step in range(1, int(spec["steps"]) + 1):
        batch = rng.sample(pairs, min(batch_size, len(pairs)))
        out = engine.forward_backward("dpo", batch, beta=beta)
        opt = engine.optim_step(lr=float(spec.get("lr", 5e-5)))
        now = time.monotonic()
        emit({"type": "train.step", "step": step, "loss": round(out["loss"], 6), "grad_norm": round(opt["grad_norm"], 6),
              "step_seconds": round(now - last, 3), **{k: round(v, 4) for k, v in out["metrics"].items()}})
        last = now
    out_dir = Path(spec["output_dir"]) / "adapter"
    engine.save(str(out_dir))
    emit({"type": "train.finished", "returncode": 0, "adapter_dir": str(out_dir), "tail": []})
    return 0


try:
    if spec["kind"] == "train":
        rc = run_dpo() if spec.get("method") == "dpo" else run_train()
    elif spec["kind"] == "session":
        rc = run_session()
    else:
        rc = run_serve()
except Exception as exc:
    emit({"type": "runner_error", "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
    rc = 1
sys.stdout.flush()
for c in children:
    try:
        kill_group(c.pid)
    except Exception:
        pass
try:
    dht.shutdown()
except Exception:
    pass
import multiprocessing
for c in multiprocessing.active_children():
    try:
        c.kill()
    except Exception:
        pass
os._exit(rc)
