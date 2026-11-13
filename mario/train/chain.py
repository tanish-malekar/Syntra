"""Warm-up then RL in one launch (same GPU): python -m mario.train.chain --run-dir runs/main [ppo args...]"""
from __future__ import annotations

import subprocess
import sys


def main():
    args = sys.argv[1:]
    run_dir = args[args.index("--run-dir") + 1]
    rest = [a for i, a in enumerate(args) if a != "--run-dir" and (i == 0 or args[i - 1] != "--run-dir")]
    py = sys.executable
    subprocess.run([py, "-u", "-m", "mario.train.bc", "--run-dir", run_dir], check=True)
    subprocess.run([py, "-u", "-m", "mario.train.ppo", "--run-dir", run_dir, "--init", f"{run_dir}/ckpt/warmup", *rest],
                   check=True)


if __name__ == "__main__":
    main()
