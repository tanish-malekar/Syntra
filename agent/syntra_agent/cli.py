"""Command line entry point:  python -m syntra_agent --server URL --token TOKEN [options]"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from pathlib import Path

from syntra_agent.client import Agent, AgentSettings
from syntra_agent.workers import AgentConfig


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="syntra_agent", description="Share this machine's idle GPUs with Syntra.")
    p.add_argument("--server", required=True, help="Syntra backend URL, e.g. http://localhost:8000")
    p.add_argument("--token", default=os.environ.get("SYNTRA_AGENT_TOKEN"), help="agent token (or SYNTRA_AGENT_TOKEN)")
    p.add_argument("--bloombee-dir", type=Path, default=Path(os.environ.get("BLOOMBEE_DIR", "~/BloomBee")),
                   help="BloomBee checkout (branch mac-mps-ft)")
    p.add_argument("--python", default=os.environ.get("BLOOMBEE_PYTHON", ""), help="Python with BloomBee's dependencies")
    p.add_argument("--models-dir", type=Path, default=Path("~/.syntra/models"), help="local model files")
    p.add_argument("--work-dir", type=Path, default=Path("~/.syntra/agent"), help="logs and scratch space")
    p.add_argument("--hf-home", type=Path, default=None, help="Hugging Face cache directory")
    p.add_argument("--port-range", default=None, help="worker ports, e.g. 31030-31049 (use with an SSH tunnel)")
    p.add_argument("--region", default="", help="label shown in the pool, e.g. 'home' or 'lab'")
    p.add_argument("--gpus", default=None, help="comma-separated GPU indices to offer (default: all)")
    p.add_argument("--simulated", action="store_true", help="present one real GPU as several virtual providers' GPUs")
    p.add_argument("--sim-count", type=int, default=2)
    p.add_argument("--sim-memory-gb", type=float, default=6.0)
    p.add_argument("--heartbeat-interval", type=float, default=2.0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    if not args.token:
        p.error("--token or SYNTRA_AGENT_TOKEN is required")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = AgentConfig(bloombee_dir=args.bloombee_dir.expanduser(), models_dir=args.models_dir.expanduser(),
                      work_dir=args.work_dir.expanduser(), hf_home=args.hf_home.expanduser() if args.hf_home else None)
    if args.python:
        cfg.python = args.python
    if args.port_range:
        lo, hi = args.port_range.split("-")
        cfg.port_range = (int(lo), int(hi))
    allow = [int(x) for x in args.gpus.split(",")] if args.gpus else None
    settings = AgentSettings(server=args.server, token=args.token, region=args.region, gpu_allowlist=allow,
                             heartbeat_interval=args.heartbeat_interval, simulated=args.simulated,
                             sim_count=args.sim_count, sim_memory_gb=args.sim_memory_gb)
    agent = Agent(settings, cfg)

    async def runner() -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, agent.stop)
        await agent.run()

    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
