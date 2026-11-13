"""PPO on N parallel games. Trains the LoRA adapters and the value head; the base model stays frozen.

    python -m mario.train.ppo --run-dir runs/rl --init runs/warm/ckpt/warmup --iters 200
    python -m mario.train.ppo --run-dir runs/scratch --iters 200            (no warm-up: "from scratch" curve)

Writes runs/<run>/metrics.jsonl (one line per iteration), ckpt/<name>/ (adapters + value head) and
replays/<name>/ (greedy Level 1 evaluation games, re-playable on any machine without a GPU).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from mario.game.vec import VecEnv
from mario.train.evaluate import eval_seeds, run_episodes, save_replays, summarize
from mario.train.policy import DEFAULT_MODEL, Policy


def gae(rewards, values, dones, last_value, gamma, lam):
    """rewards/values/dones: [T][N] lists; returns advantages and returns as [T][N] tensors."""
    T = len(rewards)
    adv = torch.zeros(T, len(rewards[0]))
    nxt_v, nxt_a = last_value, torch.zeros(len(rewards[0]))
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * nxt_v * nonterm - values[t]
        nxt_a = delta + gamma * lam * nonterm * nxt_a
        adv[t] = nxt_a
        nxt_v = values[t]
    return adv, adv + torch.stack(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="default: the --init checkpoint's model, else Qwen2.5-0.5B")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--init", default="", help="adapter checkpoint to start from (e.g. the warm-up)")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=64, help="decisions per env per iteration")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--mb", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--vf-lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--target-kl", type=float, default=0.02)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-episodes", type=int, default=64)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--stop-at", type=float, default=0.97, help="stop early once eval clear rate reaches this twice")
    ap.add_argument("--hours", type=float, default=0, help="stop cleanly after this long (0 = no limit)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text-version", type=int, default=0, help="text state version (0 = newest)")
    ap.add_argument("--death", type=float, default=-15.0, help="death penalty (before the 0.1 reward scale)")
    ap.add_argument("--flag", type=float, default=30.0, help="flag bonus (before the 0.1 reward scale)")
    ap.add_argument("--mid-start", type=float, default=0.0, help="share of Level 1 training games started mid-level")
    ap.add_argument("--level1-frac", type=float, default=0.5, help="share of training games on (randomised) Level 1")
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.run_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    with open(os.path.join(a.run_dir, "run.json"), "w") as f:
        json.dump({"label": a.label or os.path.basename(a.run_dir.rstrip("/")), **vars(a)}, f, indent=1)
    metrics = open(os.path.join(a.run_dir, "metrics.jsonl"), "a")

    if a.model is None:
        mp = os.path.join(a.init, "meta.json") if a.init else ""
        a.model = json.load(open(mp))["model"] if mp and os.path.exists(mp) else DEFAULT_MODEL
    pol = Policy(a.model, adapter=a.init or None, tiny=a.tiny)
    lora, vh = pol.trainable_parameters()
    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr}, {"params": vh, "lr": a.vf_lr}], weight_decay=0.0)
    from mario.game.textstate import TEXT_VERSION
    pol.text_version = a.text_version or TEXT_VERSION
    dev = pol.device_name
    venv = VecEnv(a.envs, seed=a.seed + 17, mode="train", level1_frac=a.level1_frac, mid_start=a.mid_start,
                  death=a.death, flag=a.flag)
    venv.set_text_version(pol.text_version)
    texts = venv.texts()
    t_start = time.time()
    best, hits = -1.0, 0
    recent: list[dict] = []

    def evaluate(name: str, it: int) -> dict:
        pol.eval()
        reps = run_episodes(pol, [("level1", s) for s in eval_seeds(a.eval_episodes)])
        pol.train()
        save_replays(reps, os.path.join(a.run_dir, "replays", name), name)
        return summarize(reps)

    # iteration 0: where we start from
    s0 = evaluate("start", 0)
    metrics.write(json.dumps({"iter": 0, "eval_clear": s0["clear"], "eval_frac": s0["frac"], "seconds": 0}) + "\n")
    metrics.flush()
    print(json.dumps({"iter": 0, **s0}), flush=True)

    for it in range(1, a.iters + 1):
        t0 = time.time()
        pol.eval()
        b_tok, b_act, b_logp = [], [], []
        rewards, values, dones = [], [], []
        finished = []
        with torch.no_grad():
            for _ in range(a.steps):
                toks = pol.tokenize(texts)
                logits, v = pol(*pol.pad(toks))
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
                r, d, fin = venv.step(act.tolist())
                finished += fin
                b_tok.append(toks)
                b_act.append(act.cpu())
                b_logp.append(dist.log_prob(act).cpu())
                values.append(v.cpu())
                rewards.append(torch.tensor(r))
                dones.append(torch.tensor(d, dtype=torch.float32))
                texts = venv.texts()
            _, last_v = pol(*pol.pad(pol.tokenize(texts)))
        t_roll = time.time() - t0
        adv, ret = gae(rewards, values, dones, last_v.cpu(), a.gamma, a.lam)

        flat_tok = [tk for step in b_tok for tk in step]
        flat_act = torch.cat(b_act)
        flat_logp = torch.cat(b_logp)
        flat_adv = adv.reshape(-1)
        flat_ret = ret.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        n = len(flat_tok)

        pol.train()
        stats = {"pg": 0.0, "vl": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0, "n": 0}
        stop = False
        for _ in range(a.epochs):
            order = torch.randperm(n)
            ep_kl, ep_n = 0.0, 0
            for k in range(0, n - a.mb + 1, a.mb):
                idx = order[k:k + a.mb]
                logits, v = pol(*pol.pad([flat_tok[i] for i in idx]))
                logp_all = F.log_softmax(logits, -1)
                act = flat_act[idx].to(dev)
                logp = logp_all.gather(1, act[:, None]).squeeze(1)
                old = flat_logp[idx].to(dev)
                ratio = torch.exp(logp - old)
                ad = flat_adv[idx].to(dev)
                pg = -torch.min(ratio * ad, ratio.clamp(1 - a.clip, 1 + a.clip) * ad).mean()
                vl = F.mse_loss(v, flat_ret[idx].to(dev))
                ent = -(logp_all.exp() * logp_all).sum(-1).mean()
                loss = pg + a.vf * vl - a.ent * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora + vh, 1.0)
                opt.step()
                with torch.no_grad():
                    lr_ = logp - old
                    kl = (torch.exp(lr_) - 1 - lr_).mean().item()   # non-negative KL estimate (k3)
                    ep_kl += kl
                    ep_n += 1
                    stats["pg"] += pg.item(); stats["vl"] += vl.item(); stats["ent"] += ent.item()
                    stats["kl"] += kl; stats["clipfrac"] += ((ratio - 1).abs() > a.clip).float().mean().item()
                    stats["n"] += 1
                if kl > 4 * a.target_kl:      # hard guard against a single runaway step
                    stop = True
                    break
            if stop or ep_kl / max(1, ep_n) > a.target_kl:   # otherwise judge KL over the whole epoch
                stop = True
                break
        m = max(1, stats.pop("n"))
        row = {"iter": it, **{k: round(v / m, 4) for k, v in stats.items()}, "kl_stop": int(stop),
               "rollout_s": round(t_roll, 1), "iter_s": round(time.time() - t0, 1),
               "seconds": round(time.time() - t_start, 1), "samples": n}
        recent = (recent + finished)[-256:]
        if finished:
            l1 = [f for f in finished if f["level"] == "level1" and "start_col" not in f]
            row.update({"episodes": len(finished), "clear": round(sum(f["won"] for f in finished) / len(finished), 3),
                        "frac": round(sum(f["frac"] for f in finished) / len(finished), 3),
                        "return": round(sum(f["return"] for f in finished) / len(finished), 3)})
            if l1:
                row["clear_level1"] = round(sum(f["won"] for f in l1) / len(l1), 3)
        if it % a.eval_every == 0 or it == a.iters:
            s = evaluate(f"iter{it:04d}", it)
            row.update({"eval_clear": s["clear"], "eval_frac": round(s["frac"], 3)})
            if s["clear"] > best:
                best = s["clear"]
                pol.save(os.path.join(a.run_dir, "ckpt", "best"), {"iter": it, "eval_clear": best})
            hits = hits + 1 if s["clear"] >= a.stop_at else 0
        if it % a.save_every == 0:
            pol.save(os.path.join(a.run_dir, "ckpt", f"iter{it:04d}"), {"iter": it})
        metrics.write(json.dumps(row) + "\n")
        metrics.flush()
        print(json.dumps(row), flush=True)
        if hits >= 2:
            print(json.dumps({"stopping": "eval clear rate reached --stop-at twice", "best": best}), flush=True)
            break
        if a.hours and time.time() - t_start > a.hours * 3600:
            print(json.dumps({"stopping": "time limit", "best": best}), flush=True)
            break
    pol.save(os.path.join(a.run_dir, "ckpt", "final"), {"iter": it})
    s = evaluate("final", it)
    print(json.dumps({"final": s, "best": best}), flush=True)


if __name__ == "__main__":
    main()
