"""Stand-in for app/runner/swarm_runner.py: same JSON-lines protocol, no BloomBee needed.
Spec keys read from the job config: steps, fake_step_delay, fake_ready_delay, fake_fail."""
import json
import os
import signal
import sys
import time
from pathlib import Path

spec = json.loads(Path(sys.argv[1]).read_text())
signal.signal(signal.SIGTERM, lambda *_: os._exit(143))


def emit(o):
    print(json.dumps(o), flush=True)


emit({"type": "dht_ready", "initial_peers": ["/ip4/127.0.0.1/tcp/31001/p2p/QmFakeDht"]})
time.sleep(float(spec.get("fake_ready_delay", 0.5)))
emit({"type": "swarm_ready", "layers_total": spec["num_layers"]})
if spec["kind"] == "deploy":
    emit({"type": "serve_ready", "device": "fake"})
    tuned = "merged" in spec["model_path"]
    DEPT = {"billing": ("charge", "refund", "payment", "invoice", "money", "receipt", "billed", "fee", "promo"),
            "login": ("password", "sign in", "login", "credentials", "authenticator", "verification", "account"),
            "shipping": ("package", "order", "delivery", "courier", "box", "parcel", "arrived", "tracking", "address", "shipped")}
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("op") == "shutdown":
            break
        t = time.monotonic()
        prompt = req["prompt"].lower()
        if tuned:
            dept = next((d for d, words in DEPT.items() if any(w in prompt for w in words)), "bug")
            urgency = "high" if any(w in prompt for w in ("urgent", "asap", "immediately")) else "low"
            text = json.dumps({"department": dept, "urgency": urgency})
        else:
            text = "Sure! I can help with that. Please tell me more about the problem you are having."
        if req.get("do_sample"):
            text += f" [sampled t={req.get('temperature')} p={req.get('top_p')} k={req.get('top_k')} seed={req.get('seed')}]"
        time.sleep(float(spec.get("fake_gen_delay", 0.02)))
        emit({"type": "result", "id": req["id"], "text": text, "tokens": len(text.split()), "seconds": round(time.monotonic() - t, 3)})
    os._exit(0)
if spec["kind"] == "session":
    import random
    LOSSES = ("cross_entropy", "dpo", "ppo", "importance_sampling")
    state = {"step": 0, "pending": 0}
    emit({"type": "serve_ready", "device": "fake", "step": 0, "pending_backward": 0, "tokens_seen": 0, "trainable_parameters": 1234})

    def op_call(op, a):
        if op == "tokenize":
            return {"tokens": [[2 + (sum(map(ord, w)) % 500) for w in t.split()] for t in a["texts"]]}
        if op == "detokenize":
            return {"texts": [" ".join(f"w{i}" for i in t) for t in a["tokens"]]}
        if op == "sample":
            n = int(a.get("n", 1))
            if not 1 <= n <= 64:
                raise ValueError("n must be between 1 and 64")
            rng = random.Random(a.get("seed"))
            out = []
            for pi, p in enumerate(a["prompts"]):
                ids = [2 + (sum(map(ord, w)) % 500) for w in p.split()] if isinstance(p, str) else list(p)
                for _ in range(n):
                    comp = [rng.randrange(2, 500) for _ in range(int(a.get("max_new_tokens", 8)))]
                    out.append({"prompt_index": pi, "tokens": ids + comp, "prompt_len": len(ids), "completion": comp,
                                "logprobs": [-1.0] * len(comp), "finished": False, "text": " ".join(f"w{i}" for i in comp)})
            return {"samples": out}
        if op == "logprobs":
            return {"logprobs": [[-1.0] * (len(x["tokens"]) - max(x["prompt_len"], 1)) for x in a["sequences"]]}
        if op == "forward_backward":
            if a.get("loss") not in LOSSES:
                raise ValueError(f"unknown loss {a.get('loss')!r}")
            if not a.get("batch"):
                raise ValueError("empty batch")
            state["pending"] += 1
            return {"loss": round(1.0 / (1 + state["step"]), 4), "metrics": {}, "pending_backward": state["pending"]}
        if op == "optim_step":
            if not state["pending"]:
                raise ValueError("optim_step without a forward_backward since the last step")
            state["step"] += 1
            state["pending"] = 0
            return {"step": state["step"], "grad_norm": 0.5}
        if op == "advantages":
            r, g = a["rewards"], int(a["group_size"])
            if len(r) % g:
                raise ValueError("rewards do not split into groups")
            adv = []
            for i in range(0, len(r), g):
                grp = r[i:i + g]
                m = sum(grp) / g
                sd = (sum((x - m) ** 2 for x in grp) / g) ** 0.5
                adv += [(x - m) / (sd + 1e-6) for x in grp]
            return {"advantages": adv, "informative": [True] * len(r)}
        if op == "state":
            return {"step": state["step"], "pending_backward": state["pending"], "tokens_seen": 0, "trainable_parameters": 1234}
        if op == "save":
            name = "".join(c for c in str(a.get("name", "")) if c.isalnum() or c in "-_")[:60] or f"step{state['step']}"
            out = Path(spec["output_dir"]) / "saves" / f"{name}-{time.time_ns()}"
            out.mkdir(parents=True, exist_ok=True)
            (out / "adapter_config.json").write_text(json.dumps({"r": spec.get("rank", 8)}))
            (out / "adapter_model.safetensors").write_bytes(b"fake-session-adapter")
            return {"adapter_dir": str(out), "step": state["step"]}
        raise ValueError(f"unknown op {op!r}")

    for line in sys.stdin:
        req = json.loads(line)
        if req.get("op") == "shutdown":
            break
        time.sleep(float(spec.get("fake_call_delay", 0.0)))
        try:
            emit({"type": "result", "id": req["id"], "data": op_call(req["op"], req.get("args") or {})})
        except (ValueError, KeyError, TypeError) as exc:
            emit({"type": "error", "id": req["id"], "error": str(exc), "bad_request": True})
    os._exit(0)
if spec.get("fake_fail"):
    emit({"type": "runner_error", "error": "boom: simulated trainer failure"})
    os._exit(1)
last = time.monotonic()
for step in range(1, int(spec["steps"]) + 1):
    time.sleep(float(spec.get("fake_step_delay", 0.1)))
    now = time.monotonic()
    extra = {"reward_margin": round(0.1 * step, 4), "reward_accuracy": min(1.0, 0.5 + 0.1 * step)} if spec.get("method") == "dpo" else {}
    emit({"type": "train.step", "step": step, "loss": round(2.0 / (1 + 0.1 * step), 4), "grad_norm": 1.5, "step_seconds": round(now - last, 3),
          **extra})
    last = now
out = Path(spec["output_dir"]) / "adapter"
out.mkdir(parents=True, exist_ok=True)
(out / "adapter_config.json").write_text(json.dumps({"r": spec.get("rank", 8)}))
(out / "adapter_model.safetensors").write_bytes(b"fake-adapter-weights")
emit({"type": "train.finished", "returncode": 0, "adapter_dir": str(out), "tail": []})
os._exit(0)
