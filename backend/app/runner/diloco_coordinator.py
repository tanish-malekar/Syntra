"""DiLoCo coordinator: the per-job process on the platform host for full fine-tuning (method "diloco").

  python diloco_coordinator.py <spec.json>          (JSON-lines events on stdout, like swarm_runner.py)

Round r: publish global weights -> replicas (provider agents) train `inner_steps` locally on their shard and upload
theta_start - theta_end -> when every live replica has reported (the orchestrator sends {"op": "replicas", "live": [...]} on
stdin), or at the round deadline with whoever reported, apply the outer Nesterov step -> round r+1.
Late uploads for a closed round are refused by the backend. Everything is on disk, so a restarted coordinator resumes.
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from pathlib import Path

spec = json.loads(Path(sys.argv[1]).read_text())
sys.path.insert(0, str(Path(__file__).resolve().parent))


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


live: set[int] = set(range(int(spec.get("replicas", 1))))
lock = threading.Lock()


def read_stdin() -> None:
    global live
    for raw in sys.stdin:
        try:
            msg = json.loads(raw)
        except ValueError:
            continue
        if msg.get("op") == "replicas":
            with lock:
                live = set(msg.get("live", []))
        elif msg.get("op") == "shutdown":
            os._exit(0)


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def main() -> int:
    import hashlib

    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from diloco_core import OuterOptimizer

    root = Path(spec["output_dir"]) / "diloco"
    rounds_dir = root / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    inner = int(spec.get("inner_steps", 20))
    total_rounds = max(1, math.ceil(int(spec["steps"]) / inner))
    round_timeout = float(spec.get("round_timeout", 900))
    max_wait = float(spec.get("no_replica_timeout", 1800))

    model = AutoModelForCausalLM.from_pretrained(spec["model_path"], torch_dtype=torch.float32)
    names = [n for n, _ in model.named_parameters()]
    params = {n: p.detach() for n, p in model.named_parameters()}
    opt = OuterOptimizer(params, lr=float(spec.get("outer_lr", 0.7)), momentum=float(spec.get("outer_momentum", 0.9)))
    start = 0
    state_file = root / "state.json"
    if state_file.exists():  # resume after a coordinator restart
        old = json.loads(state_file.read_text())
        g = rounds_dir / str(old["round"]) / "global.safetensors"
        if old.get("status") == "running" and g.exists():
            opt.params = {k: v.float() for k, v in load_file(str(g)).items()}
            if (root / "outer_state.pt").exists():
                opt.load_state_dict(torch.load(root / "outer_state.pt", weights_only=True))
            start = int(old["round"])
    threading.Thread(target=read_stdin, daemon=True).start()
    emit({"type": "dht_ready", "initial_peers": []})  # the orchestrator starts the replicas now
    emit({"type": "swarm_ready", "layers_total": int(spec.get("num_layers", 0))})

    bad_rounds, last = 0, time.monotonic()
    for r in range(start, total_rounds):
        folder = rounds_dir / str(r)
        folder.mkdir(exist_ok=True)
        g = folder / "global.safetensors"
        if not g.exists():
            save_file({k: v.contiguous() for k, v in opt.params.items()}, str(g) + ".part")
            os.replace(str(g) + ".part", g)
            g.with_suffix(".sha256").write_text(hashlib.sha256(g.read_bytes()).hexdigest())
        torch.save(opt.state_dict(), root / "outer_state.pt")
        opened = time.time()
        write_json(state_file, {"round": r, "rounds": total_rounds, "status": "running", "deadline": opened + round_timeout})
        while True:  # wait for the replicas
            reported = {int(p.stem.split("slot")[1]) for p in folder.glob("delta-slot*.json")}
            with lock:
                alive = set(live)
            now = time.time()
            if reported and alive and alive <= reported:
                break
            if reported and now > opened + round_timeout:
                break  # close with whoever reported; the others' work for this round is dropped
            if not reported and now > opened + round_timeout + max_wait:
                emit({"type": "runner_error", "error": f"no replica reported for round {r} within {round_timeout + max_wait:.0f} s"})
                return 1
            time.sleep(0.5)
        write_json(state_file, {"round": r, "rounds": total_rounds, "status": "closing", "deadline": 0})
        deltas, losses, dropped = [], [], []
        for meta_path in sorted(folder.glob("delta-slot*.json")):
            meta = json.loads(meta_path.read_text())
            try:
                delta = {k: v.float() for k, v in load_file(str(meta_path.with_suffix(".safetensors"))).items()}
            except Exception as exc:
                dropped.append(f"slot {meta['slot']}: unreadable ({exc})")
                continue
            why = opt.check(delta) or (None if math.isfinite(meta.get("loss", 0.0)) else "loss is not finite")
            if why:
                dropped.append(f"slot {meta['slot']}: {why}")
                continue
            deltas.append((delta, max(1.0, float(meta.get("tokens", 1)))))
            losses.append((meta["loss"], float(meta.get("tokens", 1))))
        if not deltas:
            bad_rounds += 1
            emit({"type": "diloco.round_skipped", "round": r + 1, "dropped": dropped})
            if bad_rounds >= 2:
                emit({"type": "runner_error", "error": f"two rounds in a row without a usable update: {'; '.join(dropped)[:300]}"})
                return 1
            continue
        bad_rounds = 0
        norm = opt.step(deltas)
        now = time.monotonic()
        loss = sum(l * w for l, w in losses) / sum(w for _, w in losses)
        emit({"type": "train.step", "step": (r + 1) * inner, "loss": round(loss, 6), "grad_norm": round(norm, 6),
              "step_seconds": round(now - last, 3), "round": r + 1, "rounds": total_rounds, "replicas_reported": len(deltas),
              "dropped": len(dropped)})
        last = now
        if r >= 1:  # keep disk use bounded: rounds older than the previous one are not needed any more
            old = rounds_dir / str(r - 1)
            for p in old.glob("*"):
                p.unlink(missing_ok=True)
    # final model: the global weights after the last outer step
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(opt.params[n])
    out = Path(spec["output_dir"]) / "model"
    model.save_pretrained(out)
    try:
        AutoTokenizer.from_pretrained(spec["model_path"]).save_pretrained(out)
    except Exception:
        pass
    write_json(state_file, {"round": total_rounds, "rounds": total_rounds, "status": "done", "deadline": 0})
    emit({"type": "train.finished", "returncode": 0, "model_dir": str(out), "tail": []})
    return 0


try:
    rc = main()
except Exception as exc:
    emit({"type": "runner_error", "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
    rc = 1
sys.stdout.flush()
os._exit(rc)
