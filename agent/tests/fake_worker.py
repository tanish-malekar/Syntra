"""Stand-in for `bloombee.cli.run_server`: prints the same log lines the agent watches for, then idles."""
import argparse
import signal
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--peer", default="12D3KooWFakeFakeFakeFake")
p.add_argument("--start-delay", type=float, default=0.1)
p.add_argument("--exit-after", type=float, default=0)  # 0 = run until stopped
p.add_argument("--exit-code", type=int, default=0)
a = p.parse_args()

signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
print(f"Nov 12 03:00:00.000 [INFO] Running a server on ['/ip4/127.0.0.1/tcp/31999/p2p/{a.peer}']", flush=True)
time.sleep(a.start_delay)
print("Nov 12 03:00:00.500 [INFO] Started", flush=True)
t0 = time.time()
while True:
    time.sleep(0.05)
    if a.exit_after and time.time() - t0 >= a.exit_after:
        sys.exit(a.exit_code)
