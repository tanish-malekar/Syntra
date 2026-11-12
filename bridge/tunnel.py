"""Keep an SSH tunnel to context3 open with the port ranges the platform needs.

  python bridge/tunnel.py [--backend-port 8000] [--dht-port 31001] [--mac-ports 31010-31019] [--c3-ports 31030-31039]

-R (context3 reaches this Mac): the backend API, the job DHT, and the Mac agent's worker ports
-L (this Mac reaches context3):  the context3 agent's worker ports
Workers announce 127.0.0.1:<port>, which is valid on both machines through these forwards.
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ssh  # noqa: E402


def port_range(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def open_tunnel(backend_port: int, dht_port: int, mac_ports: list[int], c3_ports: list[int]) -> subprocess.Popen:
    return ssh.tunnel(forwards_local=c3_ports, forwards_remote=[backend_port, dht_port, *mac_ports])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend-port", type=int, default=8000)
    p.add_argument("--dht-port", type=int, default=31001)
    p.add_argument("--mac-ports", default="31010-31019")
    p.add_argument("--c3-ports", default="31030-31039")
    a = p.parse_args()
    stop = False
    signal.signal(signal.SIGTERM, lambda *_: globals().update(stop=True))
    while not stop:
        proc = open_tunnel(a.backend_port, a.dht_port, port_range(a.mac_ports), port_range(a.c3_ports))
        print("tunnel up", flush=True)
        while proc.poll() is None and not stop:
            time.sleep(1)
        if proc.poll() is None:
            proc.terminate()
        if not stop:
            print("tunnel dropped:", (proc.stderr.read().decode()[-200:] if proc.stderr else ""), "reconnecting in 3 s", flush=True)
            time.sleep(3)


if __name__ == "__main__":
    main()
