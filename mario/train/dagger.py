"""Correction rounds (DAgger-style fallback): the model plays, a look-ahead search labels the right move in the states the
model actually visits (including the ones where it gets stuck or dies), and the model is fine-tuned on those labels.

The expert is a short-horizon beam search over the same deterministic sim (too slow to play live, fine offline,
spread over many CPU cores). Afterwards, PPO can continue from the corrected checkpoint.

    python -m mario.train.dagger --run-dir runs/main --init runs/main/ckpt/best --rounds 2
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from multiprocessing import Pool

import torch
import torch.nn.functional as F

from mario.game.sim import TILE, Sim
from mario.game.textstate import render_text
from mario.game.vec import GameEnv
from mario.train.evaluate import eval_seeds, run_episodes, save_replays, summarize
from mario.train.policy import Policy


PREFERENCE = [0, 1, 2, 5, 4, 3]   # run, long jump, hop, wait, jump in place, left: simplest move that is as good


def _score(s: Sim) -> float:
    if s.won:
        return 1e6 - s.frame
    if s.dead:
        return -1e9
    return s.x + 0.25 * s.max_x


def _search(start: Sim, horizon: int, beam: int) -> float:
    """Best score reachable from `start` within `horizon` moves (beam search, deaths pruned)."""
    if start.done:
        return _score(start)
    frontier, best = [start], -1e9
    for _ in range(horizon):
        cand, seen = [], set()
        for s in frontier:
            for a in range(6):
                t = s.clone()
                t.step(a)
                t.events = []
                if t.dead:
                    continue
                if t.won:
                    best = max(best, _score(t))
                    continue
                key = (round(t.x / 3), round(t.y / 3), round(t.vy, 1))
                if key in seen:
                    continue
                seen.add(key)
                cand.append(t)
        if not cand:
            break
        cand.sort(key=lambda t: -_score(t))
        frontier = cand[:beam]
        best = max(best, _score(frontier[0]))
    return best


def move_scores(sim: Sim, horizon: int = 30, beam: int = 12) -> list[float]:
    out = []
    for a in range(6):
        t = sim.clone()
        t.step(a)
        t.events = []
        out.append(_score(t) if t.done else _search(t, horizon - 1, beam))
    return out


def expert_move(sim: Sim, horizon: int = 30, beam: int = 12, margin: float = 6.0) -> int:
    """Consistent label: the simplest move (see PREFERENCE) whose look-ahead outcome is within `margin` px of the best."""
    sc = move_scores(sim, horizon, beam)
    best = max(sc)
    for a in PREFERENCE:
        if sc[a] >= best - margin:
            return a
    return max(range(6), key=lambda a: sc[a])


def ok_set(sim: Sim, horizon: int = 20, beam: int = 8, margin: float = 6.0) -> int:
    """Bitmask of every move whose look-ahead outcome is within `margin` px of the best one (all are fine to play)."""
    sc = move_scores(sim, horizon, beam)
    best = max(sc)
    return sum(1 << a for a in range(6) if sc[a] >= best - margin)


def _label(sim: Sim) -> int:
    return ok_set(sim)


def collect(pol: Policy, episodes: int, seed: int, workers: int, level1_frac: float = 0.5, mid_start: float = 0.0):
    """Play `episodes` training games with the current model (sampled moves); return visited states + expert labels."""
    rng = random.Random(seed)
    sims, texts = [], []
    envs = []
    for i in range(episodes):
        e = GameEnv(seed * 1000 + i, level1_frac=level1_frac, mid_start=mid_start)
        e.reset()   # training seeds; Level 1 share and mid-level starts as configured
        e.text_version = pol.text_version
        envs.append(e)
    live = list(range(len(envs)))
    with torch.no_grad():
        while live:
            batch_texts = [envs[i].text() for i in live]
            for i, t in zip(live, batch_texts):
                sims.append(envs[i].sim.clone())
                texts.append(t)
            logits, _ = pol.act_texts(batch_texts)
            acts = torch.distributions.Categorical(logits=logits).sample().tolist()
            nxt = []
            for i, a in zip(live, acts):
                _, d = envs[i].step(a)
                if not d:
                    nxt.append(i)
            live = nxt
    with Pool(workers) as p:
        labels = p.map(_label, sims, chunksize=16)
    return texts, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--init", default="", help="adapter to start from; empty = fresh LoRA on --model")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--lora-r", type=int, default=16, help="adapter rank for a fresh start (capacity)")
    ap.add_argument("--max-train-states", type=int, default=0, help="per round: new states + sampled old ones (0 = all)")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast. Do NOT use with Qwen2.5: activations overflow in fp16 (a rank-256 run fell from 75%% to 23%%)")
    ap.add_argument("--pretrain-epochs", type=int, default=0, help="first train this many passes on --data alone")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--mb", type=int, default=64)
    ap.add_argument("--eval-episodes", type=int, default=64)
    ap.add_argument("--text-version", type=int, default=0, help="0 = newest")
    ap.add_argument("--data", default="", help="earlier rounds' labelled states (jsonl) to start from")
    ap.add_argument("--epochs", type=int, default=1, help="passes over all labelled states per round")
    ap.add_argument("--first-round", type=int, default=1)
    ap.add_argument("--level1-frac", type=float, default=0.5)
    ap.add_argument("--mid-start", type=float, default=0.0)
    ap.add_argument("--then-ppo", default="", help="after the rounds, run PPO from the last round with these args")
    a = ap.parse_args()
    if a.init:
        meta = json.load(open(os.path.join(a.init, "meta.json")))
        pol = Policy(meta["model"], adapter=a.init)
    else:
        pol = Policy(a.model, lora_r=a.lora_r)
    from mario.game.textstate import TEXT_VERSION
    pol.text_version = a.text_version or TEXT_VERSION
    lora, _ = pol.trainable_parameters()
    opt = torch.optim.AdamW(lora, lr=a.lr, weight_decay=0.0)
    log = open(os.path.join(a.run_dir, "dagger_log.jsonl"), "a")
    data_t, data_y = [], []
    if a.data:
        for line in open(a.data):
            d = json.loads(line)
            data_t.append(d["t"])
            data_y.append(d["y"])
    data_file = open(os.path.join(a.run_dir, "dagger_data.jsonl"), "a")
    t0 = time.time()
    pol.amp = a.amp
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp)

    def train_pass(epochs: int, seed: int, new: int = 0) -> float:
        """`epochs` passes over the labelled states; with --max-train-states, over the newest `new` states plus a
        random sample of older ones (keeps rounds short while still revisiting old mistakes)."""
        pol.train()
        rng_ = random.Random(seed)
        pool = list(range(len(data_t)))
        if a.max_train_states and len(pool) > a.max_train_states:
            newest = pool[len(pool) - new:] if new else []
            older = pool[:len(pool) - new] if new else pool
            pool = newest + rng_.sample(older, max(0, a.max_train_states - len(newest)))
        toks = {i: t for i, t in zip(pool, pol.tokenize([data_t[i] for i in pool]))}
        order = [i for e in range(epochs) for i in random.Random(seed * 100 + e).sample(pool, len(pool))]
        loss = torch.zeros(())
        for k in range(0, len(order) - a.mb + 1, a.mb):
            idx = order[k:k + a.mb]
            logits, _ = pol(*pol.pad([toks[i] for i in idx]))
            ok = torch.tensor([[(data_y[i] >> j) & 1 for j in range(6)] for i in idx], device=pol.device_name).bool()
            logp = F.log_softmax(logits, -1)
            loss = -(torch.logsumexp(logp.masked_fill(~ok, -1e9), -1)).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(lora, 1.0)
            scaler.step(opt)
            scaler.update()
        return loss.item()

    if a.pretrain_epochs and data_t:
        l = train_pass(a.pretrain_epochs, 0)
        pol.save(os.path.join(a.run_dir, "ckpt", "pretrain"), {"phase": "pretrain", "samples": len(data_t)})
        pol.eval()
        reps = run_episodes(pol, [("level1", s_) for s_ in eval_seeds(a.eval_episodes)])
        save_replays(reps, os.path.join(a.run_dir, "replays", "pretrain"), "pretrain")
        row = {"phase": "pretrain", "new_states": 0, "total_states": len(data_t), "loss": round(l, 4), **summarize(reps),
               "seconds": round(time.time() - t0, 1)}
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()
    last = a.first_round + a.rounds - 1
    for rnd in range(a.first_round, last + 1):
        pol.eval()
        texts, labels = collect(pol, a.episodes, rnd, a.workers, a.level1_frac, a.mid_start)
        data_t += texts
        data_y += labels
        for t_, y_ in zip(texts, labels):
            data_file.write(json.dumps({"t": t_, "y": y_}) + "\n")
        data_file.flush()
        # set-valued target: maximise the total probability of the acceptable moves. Where the model's own
        # choice is already acceptable this barely changes anything; it corrects only real mistakes.
        loss_v = train_pass(a.epochs, rnd, new=len(texts))
        name = f"dagger{rnd}"
        pol.save(os.path.join(a.run_dir, "ckpt", name), {"phase": name, "samples": len(data_t)})
        pol.eval()
        reps = run_episodes(pol, [("level1", s) for s in eval_seeds(a.eval_episodes)])
        save_replays(reps, os.path.join(a.run_dir, "replays", name), name)
        wrong = sum(1 for y in labels if y == 0)
        row = {"phase": name, "new_states": len(texts), "total_states": len(data_t), "loss": round(loss_v, 4),
               "no_ok_move": wrong,
               **summarize(reps), "seconds": round(time.time() - t0, 1)}
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()
    if a.then_ppo:
        import shlex
        import subprocess
        import sys
        del pol, opt, scaler
        torch.cuda.empty_cache()
        subprocess.run([sys.executable, "-u", "-m", "mario.train.ppo", "--run-dir", a.run_dir,
                        "--init", os.path.join(a.run_dir, "ckpt", f"dagger{last}"), *shlex.split(a.then_ppo)], check=True)


if __name__ == "__main__":
    main()
