"""Full recipe for one model size, on one GPU: bot imitation -> expert-correction rounds -> PPO.

    python -m mario.train.pipeline --run-dir runs/big --model Qwen/Qwen2.5-1.5B-Instruct --mb 32 --rounds 3 \
        --ppo "--death -30 --flag 45 --mid-start 0.5 --level1-frac 0.9 --iters 400 --hours 4"
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mb", type=int, default=32)
    ap.add_argument("--bc-samples", type=int, default=20000)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--ppo", default="")
    a = ap.parse_args()
    py = [sys.executable, "-u", "-m"]
    subprocess.run(py + ["mario.train.bc", "--run-dir", a.run_dir, "--model", a.model, "--samples", str(a.bc_samples),
                         "--mb", str(a.mb)], check=True)
    subprocess.run(py + ["mario.train.dagger", "--run-dir", a.run_dir, "--init", f"{a.run_dir}/ckpt/warmup",
                         "--rounds", str(a.rounds), "--episodes", str(a.episodes), "--workers", str(a.workers),
                         "--mb", str(a.mb), "--then-ppo", f"--mb {a.mb} {a.ppo}"], check=True)


if __name__ == "__main__":
    main()
