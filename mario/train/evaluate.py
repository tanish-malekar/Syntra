"""Clear rate on randomised Level 1 episodes with seeds never used in training, plus replays.

    python -m mario.train.evaluate --checkpoint runs/<run>/ckpt/final --episodes 100 --out runs/<run>/replays/final
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from mario.game.vec import EVAL_SEED_BASE, GameEnv


def eval_seeds(n: int, offset: int = 0) -> list[int]:
    return [EVAL_SEED_BASE + offset + i for i in range(n)]


@torch.no_grad()
def run_episodes(policy, episodes: list[tuple[str, int]], greedy: bool = True, batch: int = 64,
                 difficulty: int = 1, record_probs: bool = True) -> list[dict]:
    """Play the given (level, seed) episodes to the end, `batch` at a time. Returns one replay dict per episode."""
    out = []
    for k in range(0, len(episodes), batch):
        chunk = episodes[k:k + batch]
        envs = []
        for lv, seed in chunk:
            e = GameEnv(0, difficulty=difficulty)
            e.reset(seed=seed, level=lv)
            e.probs = []
            e.text_version = getattr(policy, "text_version", e.text_version)
            envs.append(e)
        live = list(range(len(envs)))
        while live:
            logits, _ = policy.act_texts([envs[i].text() for i in live])
            probs = torch.softmax(logits, -1)
            acts = probs.argmax(-1) if greedy else torch.multinomial(probs, 1).squeeze(-1)
            nxt = []
            for j, i in enumerate(live):
                if record_probs:
                    envs[i].probs.append([round(float(p), 3) for p in probs[j]])
                _, done = envs[i].step(int(acts[j]))
                if not done:
                    nxt.append(i)
            live = nxt
        for e in envs:
            rep = e.replay()
            if record_probs:
                rep["probs"] = e.probs
            out.append(rep)
    return out


def summarize(reps: list[dict]) -> dict:
    n = max(1, len(reps))
    return {"episodes": len(reps), "clear": sum(r["won"] for r in reps) / n,
            "frac": sum(r["frac"] for r in reps) / n, "return": sum(r["return"] for r in reps) / n}


def save_replays(reps: list[dict], out_dir: str, checkpoint: str, keep: int = 6):
    """Writes a few replays (wins first, then the longest failures) plus a summary."""
    os.makedirs(out_dir, exist_ok=True)
    wins = [r for r in reps if r["won"]]
    fails = sorted([r for r in reps if not r["won"]], key=lambda r: -r["dist"])
    chosen = wins[: keep // 2] + fails[: keep - min(len(wins), keep // 2)]
    chosen += wins[keep // 2: keep // 2 + keep - len(chosen)]
    for i, r in enumerate(chosen[:keep]):
        r = {**r, "checkpoint": checkpoint,
             "label": f"{checkpoint} · {r['level']} seed {r['seed']} · {'cleared' if r['won'] else r['reason'] + ' at ' + str(r['dist'])}"}
        with open(os.path.join(out_dir, f"{i:02d}.json"), "w") as f:
            json.dump(r, f)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({**summarize(reps), "checkpoint": checkpoint,
                   "results": [{k: r[k] for k in ("level", "seed", "won", "reason", "dist")} for r in reps]}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="", help="adapter dir; empty = untrained base model")
    ap.add_argument("--model", default=None)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--level", default="level1")
    ap.add_argument("--sample", action="store_true", help="sample moves instead of taking the most likely one")
    ap.add_argument("--out", default="")
    ap.add_argument("--run-dir", default="", help="when launched via mario.remote: results go to <run-dir>/replays/<label>")
    ap.add_argument("--label", default="")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="use eval seeds from EVAL_SEED_BASE+offset (5000 = the independent test set no run selects on)")
    a = ap.parse_args()
    from mario.train.policy import DEFAULT_MODEL, Policy
    model = a.model
    if model is None and a.checkpoint and os.path.exists(os.path.join(a.checkpoint, "meta.json")):
        model = json.load(open(os.path.join(a.checkpoint, "meta.json")))["model"]
    pol = Policy(model or DEFAULT_MODEL, adapter=a.checkpoint or None, tiny=a.tiny, trainable=False).eval()
    t = time.time()
    reps = run_episodes(pol, [(a.level, s) for s in eval_seeds(a.episodes, a.seed_offset)], greedy=not a.sample)
    s = summarize(reps)
    print(json.dumps({**s, "seconds": round(time.time() - t, 1)}))
    if not a.out and a.run_dir:
        a.out = os.path.join(a.run_dir, "replays", a.label or "eval")
    if a.out:
        save_replays(reps, a.out, a.label or os.path.basename(a.checkpoint.rstrip("/")) or "untrained")


if __name__ == "__main__":
    main()
