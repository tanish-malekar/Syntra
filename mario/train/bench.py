"""Speed + numerics check on one GPU: time each part of a rollout step and of a PPO minibatch, and measure how much
a state's move scores change with batch composition / padding (bf16 vs fp32)."""
from __future__ import annotations

import argparse
import json
import time

import torch

from mario.game.vec import VecEnv
from mario.train.policy import DEFAULT_MODEL, Policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    out = {}
    v = VecEnv(64, seed=5)
    for _ in range(20):
        v.step([i % 3 for i in range(64)])
    texts = v.texts()
    t = time.time(); [pol_txt for pol_txt in v.texts()]; out["render_text_64_ms"] = round((time.time() - t) * 1000, 1)
    t = time.time(); v.step([0] * 64); out["sim_step_64_ms"] = round((time.time() - t) * 1000, 1)
    for name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        torch.backends.cuda.matmul.allow_tf32 = True
        pol = Policy(a.model, dtype=dt)
        t = time.time(); toks = pol.tokenize(texts); out["tokenize_64_ms"] = round((time.time() - t) * 1000, 1)
        out["prompt_tokens"] = sum(map(len, toks)) / len(toks)
        with torch.no_grad():
            for _ in range(2):
                pol(*pol.pad(toks))
            torch.cuda.synchronize(); t = time.time()
            for _ in range(5):
                full, _ = pol(*pol.pad(toks))
            torch.cuda.synchronize(); out[f"{name}_forward_64_ms"] = round((time.time() - t) * 200, 1)
            lp_full = torch.log_softmax(full, -1)
            perm = torch.randperm(64)
            part, _ = pol(*pol.pad([toks[i] for i in perm[:16]]))
            lp_part = torch.log_softmax(part, -1)
            out[f"{name}_logp_maxdiff_batching"] = round((lp_full[perm[:16]] - lp_part).abs().max().item(), 4)
            p = lp_full.exp()
            out[f"{name}_kl_batching"] = round((p[perm[:16]] * (lp_full[perm[:16]] - lp_part)).sum(-1).mean().item(), 5)
        pol.train()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(3):
            logits, val = pol(*pol.pad(toks))
            (logits.logsumexp(-1).mean() + val.mean()).backward()
        torch.cuda.synchronize(); out[f"{name}_fwd_bwd_64_ms"] = round((time.time() - t) * 1000 / 3, 1)
        out[f"{name}_max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        del pol
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
