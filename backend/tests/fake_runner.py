"""Stand-in for app/runner/swarm_runner.py: same JSON-lines protocol, no BloomBee needed.
Spec keys read from the job config: steps, fake_step_delay, fake_ready_delay, fake_fail."""
import json
import os
import signal
import sys
import time
from pathlib import Path

spec = json.loads(Path(sys.argv[1]).read_text())
signal.signal(signal.SIGTERM, lambda *_: os._exit(143))


def emit(o):
    print(json.dumps(o), flush=True)


emit({"type": "dht_ready", "initial_peers": ["/ip4/127.0.0.1/tcp/31001/p2p/QmFakeDht"]})
time.sleep(float(spec.get("fake_ready_delay", 0.5)))
emit({"type": "swarm_ready", "layers_total": spec["num_layers"]})
if spec.get("fake_fail"):
    emit({"type": "runner_error", "error": "boom: simulated trainer failure"})
    os._exit(1)
last = time.monotonic()
for step in range(1, int(spec["steps"]) + 1):
    time.sleep(float(spec.get("fake_step_delay", 0.1)))
    now = time.monotonic()
    emit({"type": "train.step", "step": step, "loss": round(2.0 / (1 + 0.1 * step), 4), "grad_norm": 1.5, "step_seconds": round(now - last, 3)})
    last = now
out = Path(spec["output_dir"]) / "adapter"
out.mkdir(parents=True, exist_ok=True)
(out / "adapter_config.json").write_text(json.dumps({"r": spec.get("rank", 8)}))
(out / "adapter_model.safetensors").write_bytes(b"fake-adapter-weights")
emit({"type": "train.finished", "returncode": 0, "adapter_dir": str(out), "tail": []})
os._exit(0)
