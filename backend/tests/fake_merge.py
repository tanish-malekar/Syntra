"""Stand-in for BloomBee's examples/merge_lora.py: writes a 'merged model' directory."""
import argparse
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--model"); p.add_argument("--adapter"); p.add_argument("--output")
a = p.parse_args()
out = Path(a.output)
if out.exists():
    sys.exit("Merged output must be a new directory")
assert Path(a.adapter, "adapter_config.json").exists(), "adapter missing"
out.mkdir(parents=True)
(out / "config.json").write_text("{}")
(out / "model.safetensors").write_bytes(b"fake-merged-weights")
print(f"Merged FP16 model saved to {out}")
