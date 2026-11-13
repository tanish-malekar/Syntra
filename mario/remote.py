"""Run training on context3: push code, set up the venv, launch / watch / stop runs, fetch results.

Everything lives under /data01/tmalekar/mario (code, venv, HF cache, pip cache, tmp, runs); nothing in /home.
Every run is pinned with CUDA_VISIBLE_DEVICES and wrapped in `timeout`.

    backend/.venv/bin/python -m mario.remote gpus
    backend/.venv/bin/python -m mario.remote push
    backend/.venv/bin/python -m mario.remote setup
    backend/.venv/bin/python -m mario.remote launch --gpu 0 --name rl --hours 8 -- mario.train.ppo --init runs/warm/ckpt/warmup
    backend/.venv/bin/python -m mario.remote status
    backend/.venv/bin/python -m mario.remote fetch --name rl
    backend/.venv/bin/python -m mario.remote stop --name rl
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bridge.ssh import context3_args, run_remote  # noqa: E402

ROOT = "/data01/tmalekar/mario"
BB_PY = "/data01/tmalekar/conda-envs/bb/bin/python"
HERE = os.path.dirname(os.path.abspath(__file__))
ENV = (f"export HF_HOME={ROOT}/hf PIP_CACHE_DIR={ROOT}/pip-cache TMPDIR={ROOT}/tmp XDG_CACHE_HOME={ROOT}/cache "
       f"PYTHONPATH={ROOT}/code HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false; mkdir -p {ROOT}/tmp {ROOT}/runs; ")


def sh(script: str, timeout: float = 300) -> str:
    r = run_remote(script, timeout=timeout)
    out = (r.stdout or "") + (("\n[stderr] " + r.stderr[-2000:]) if r.returncode and r.stderr else "")
    if r.returncode:
        print(out)
        raise SystemExit(f"remote command failed ({r.returncode})")
    return out


def cmd_gpus(_):
    print(sh("nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader; "
             "echo; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; "
             f"echo; ps -u tmalekar -o pid,etime,args | grep -E 'mario|python' | grep -v grep || true"))


def cmd_push(_):
    parent = os.path.dirname(HERE)
    tar = subprocess.Popen(["tar", "czf", "-", "--exclude", ".venv", "--exclude", "runs", "--exclude", "__pycache__",
                            "--exclude", "replays", "-C", parent, "mario"], stdout=subprocess.PIPE)
    done = subprocess.run(context3_args() + [f"mkdir -p {ROOT}/code && rm -rf {ROOT}/code/mario && tar xzf - -C {ROOT}/code"],
                          stdin=tar.stdout, capture_output=True, text=True, timeout=180)
    tar.wait()
    if done.returncode or tar.returncode:
        raise SystemExit("push failed: " + done.stderr[-500:])
    print(sh(f"find {ROOT}/code/mario -name '*.py' | wc -l") .strip(), "python files on context3")


def cmd_setup(_):
    print(sh(ENV + f"""
set -e
if [ ! -x {ROOT}/venv/bin/python ]; then {BB_PY} -m venv --system-site-packages {ROOT}/venv; fi
{ROOT}/venv/bin/python -m pip install -q --upgrade "peft>=0.15" 2>&1 | tail -3
{ROOT}/venv/bin/python -c "import torch, transformers, peft, sys; print(sys.version.split()[0], torch.__version__, transformers.__version__, peft.__version__)"
{ROOT}/venv/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-0.5B-Instruct'))"
du -sh {ROOT} 2>/dev/null
""", timeout=1200))


def cmd_launch(a):
    run = f"{ROOT}/runs/{a.name}"
    mod = a.cmd[0] if a.cmd and a.cmd[0] != "--" else a.cmd[1]
    rest = a.cmd[a.cmd.index(mod) + 1:]
    args = " ".join(shlex.quote(x) for x in rest)
    run_dir_arg = "" if "--run-dir" in rest else f"--run-dir {run}"
    secs = int(a.hours * 3600)
    # Safety: shared machine. Refuse a GPU that has memory in use, load, or any process on it (checked right now,
    # immediately before launching; never launch after a wait loop that merely timed out).
    for g in str(a.gpu).split(","):
        chk = sh(f"""m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {g})
u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i {g})
id=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i {g})
n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$id" || true)
echo "$m $u $n" """).split()
        if int(chk[0]) > 100 or int(chk[1]) > 5 or int(chk[2]) > 0:
            raise SystemExit(f"GPU {g} is busy (memory {chk[0]} MiB, util {chk[1]}%, {chk[2]} processes): not launching")
    script = ENV + f"""
set -e
mkdir -p {run}
if [ -f {run}/pid ] && kill -0 $(cat {run}/pid) 2>/dev/null; then echo "run {a.name} is already running"; exit 3; fi
cd {ROOT}
CUDA_VISIBLE_DEVICES={a.gpu} setsid nohup timeout --signal=TERM --kill-after=120 {secs} \
  {ROOT}/venv/bin/python -u -m {mod} {run_dir_arg} {args} > {run}/log.txt 2>&1 < /dev/null &
echo $! > {run}/pid
echo "gpu={a.gpu} hours={a.hours} cmd={mod} {args}" > {run}/launch.txt
sleep 2; echo "launched {a.name} pid $(cat {run}/pid) on GPU {a.gpu}"
"""
    print(sh(script))


def cmd_status(a):
    names = a.name or "*"
    print(sh(f"""
for d in {ROOT}/runs/{names}; do
  [ -d "$d" ] || continue
  n=$(basename $d); alive=no
  [ -f $d/pid ] && kill -0 $(cat $d/pid) 2>/dev/null && alive=yes
  echo "== $n running=$alive $(cat $d/launch.txt 2>/dev/null)"
  tail -n {a.lines} $d/log.txt 2>/dev/null | cut -c1-400
done
echo; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
"""))


def cmd_stop(a):
    print(sh(f"""
d={ROOT}/runs/{a.name}
if [ -f $d/pid ]; then p=$(cat $d/pid); kill -TERM -- -$(ps -o pgid= $p | tr -d ' ') 2>/dev/null || kill $p 2>/dev/null || true; sleep 5; fi
ps -u tmalekar -o pid,args | grep "runs/{a.name}" | grep -v grep || echo "no process left for {a.name}"
"""))


def cmd_fetch(a):
    dest = os.path.join(HERE, "runs")
    os.makedirs(dest, exist_ok=True)
    excl = "" if a.with_ckpt else "--exclude=*/ckpt/iter*"
    p = subprocess.Popen(context3_args() + [f"cd {ROOT}/runs && tar czf - {excl} {a.name}"], stdout=subprocess.PIPE)
    r = subprocess.run(["tar", "xzf", "-", "-C", dest], stdin=p.stdout, capture_output=True, text=True, timeout=600)
    p.wait()
    if r.returncode or p.returncode:
        raise SystemExit("fetch failed " + r.stderr[-300:])
    print("fetched into", os.path.join(dest, a.name))


def cmd_cleanup(_):
    print(sh(f"""ps -u tmalekar -o pid,etime,args | grep {ROOT} | grep -v grep || echo 'no mario processes'
du -sh {ROOT}/* 2>/dev/null"""))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("gpus").set_defaults(f=cmd_gpus)
    sp.add_parser("push").set_defaults(f=cmd_push)
    sp.add_parser("setup").set_defaults(f=cmd_setup)
    p = sp.add_parser("launch")
    p.add_argument("--gpu", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--hours", type=float, default=8)
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    p.set_defaults(f=cmd_launch)
    p = sp.add_parser("status")
    p.add_argument("--name", default="")
    p.add_argument("--lines", type=int, default=3)
    p.set_defaults(f=cmd_status)
    p = sp.add_parser("stop")
    p.add_argument("--name", required=True)
    p.set_defaults(f=cmd_stop)
    p = sp.add_parser("fetch")
    p.add_argument("--name", required=True)
    p.add_argument("--with-ckpt", action="store_true")
    p.set_defaults(f=cmd_fetch)
    sp.add_parser("cleanup").set_defaults(f=cmd_cleanup)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
