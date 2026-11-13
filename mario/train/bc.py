"""Warm-up: the LLM imitates the scripted bot for a few minutes (cures the cold start; RL does the real work).

The bot plays with some random moves (so the data covers states off the bot's own path), but every label is the
bot's choice in that state. The value head is fitted to the discounted returns of those games at the same time.

    python -m mario.train.bc --run-dir runs/warm --samples 40000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from mario.game.bot import NoisyBot, bot_action
from mario.game.vec import GameEnv, VecEnv
from mario.train.evaluate import eval_seeds, run_episodes, save_replays, summarize
from mario.train.policy import DEFAULT_MODEL, Policy


def bot_clear_rate(episodes: int) -> float:
    won = 0
    for s in eval_seeds(episodes):
        e = GameEnv(0)
        e.reset(seed=s, level="level1")
        done = False
        while not done:
            _, done = e.step(bot_action(e.sim))
        won += e.sim.won
    return won / episodes


def collect(samples: int, eps: float, seed: int, gamma: float):
    venv = VecEnv(64, seed=seed, mode="train")
    noisy = [NoisyBot(eps, seed * 1000 + i) for i in range(64)]
    traj = [[] for _ in range(64)]          # per env: (text, label, reward)
    data = []
    while len(data) < samples:
        texts = venv.texts()
        labels = [bot_action(e.sim) for e in venv.envs]
        acts = [n(e.sim) for n, e in zip(noisy, venv.envs)]
        rewards, dones, _ = venv.step(acts)
        for i in range(64):
            traj[i].append((texts[i], labels[i], rewards[i]))
            if dones[i]:
                g = 0.0
                for t, l, r in reversed(traj[i]):
                    g = r + gamma * g
                    data.append((t, l, g))
                traj[i] = []
    return data[:samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--samples", type=int, default=40000)
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--mb", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-episodes", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.run_dir, exist_ok=True)
    log = open(os.path.join(a.run_dir, "bc_log.jsonl"), "a")

    def emit(d):
        print(json.dumps(d), flush=True)
        log.write(json.dumps(d) + "\n")
        log.flush()

    t0 = time.time()
    emit({"phase": "bot", "bot_clear_level1": bot_clear_rate(100), "episodes": 100})
    data = collect(a.samples, a.eps, a.seed, 0.99)
    counts = [sum(1 for d in data if d[1] == k) for k in range(6)]
    emit({"phase": "collect", "samples": len(data), "label_counts": counts, "seconds": round(time.time() - t0, 1)})

    pol = Policy(a.model, tiny=a.tiny)
    lora, vh = pol.trainable_parameters()
    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr}, {"params": vh, "lr": 1e-3}], weight_decay=0.0)
    rng = random.Random(a.seed)
    toks = pol.tokenize([d[0] for d in data])
    step = 0
    for ep in range(a.epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for k in range(0, len(order) - a.mb + 1, a.mb):
            idx = order[k:k + a.mb]
            ids, mask = pol.pad([toks[i] for i in idx])
            y = torch.tensor([data[i][1] for i in idx], device=pol.device_name)
            g = torch.tensor([data[i][2] for i in idx], device=pol.device_name, dtype=torch.float32)
            logits, v = pol(ids, mask)
            ce = F.cross_entropy(logits, y)
            vl = F.mse_loss(v, g)
            loss = ce + 0.5 * vl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora + vh, 1.0)
            opt.step()
            step += 1
            if step % 25 == 0:
                acc = (logits.argmax(-1) == y).float().mean().item()
                emit({"phase": "train", "step": step, "ce": round(ce.item(), 4), "value_loss": round(vl.item(), 4),
                      "acc": round(acc, 3), "seconds": round(time.time() - t0, 1)})
    ck = os.path.join(a.run_dir, "ckpt", "warmup")
    pol.save(ck, {"phase": "warmup", "samples": len(data), "steps": step})
    pol.eval()
    reps = run_episodes(pol, [("level1", s) for s in eval_seeds(a.eval_episodes)])
    s = summarize(reps)
    save_replays(reps, os.path.join(a.run_dir, "replays", "warmup"), "warmup")
    emit({"phase": "eval", "checkpoint": "warmup", **s, "seconds": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
