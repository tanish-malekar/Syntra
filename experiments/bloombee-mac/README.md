# BloomBee on Apple GPU: experiments

Scripts used to verify distributed LoRA training, fail-over and serving of a real 1B model (TinyLlama-1.1B) on an
Apple-silicon Mac (MPS). They drive BloomBee processes directly; nothing here is part of the platform yet.

## Setup
1. BloomBee branch: `mac-mps-ft` of the fork (Apple-GPU patches + fault-tolerant LoRA client + retry options in
   `examples/train_lora.py`). Point `BLOOMBEE_DIR` at the checkout.
2. Python 3.12 venv (3.14 untested): `python3.12 -m venv venv && venv/bin/pip install -r requirements.txt`
3. Work folder (`SCRATCH`) containing `models/tinyllama` (TinyLlama-1.1B-Chat safetensors, ~2.1 GB):
   `huggingface_hub.snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0', local_dir='$SCRATCH/models/tinyllama',
   allow_patterns=['*.safetensors','*.json','tokenizer.model'])`
4. Ticket data: `python make_data.py $SCRATCH/deploy_data`

Run every script with `PYTHONPATH=$BLOOMBEE_DIR/src SCRATCH=... OUT=... <venv>/bin/python script.py`.

## Scripts
| Script | What it does |
|---|---|
| `smoke_lora_mac.py` | Mac-adapted copy of BloomBee's `tests/run_lora_rpc.py` (tiny random model; `SMOKE_DEVICE=mps`, `--check-resume --check-merged`) |
| `train_1b.py` | Train a LoRA adapter on TinyLlama over 2 pipeline workers |
| `failover_v2.py` | Train, then kill or freeze (`MODE=kill|hang`) one of `N` workers and start a replacement; `REPL_FROM_HUB=1` makes the replacement download its weights |
| `failover_v1_simple.py` | First, simpler fail-over test (2 workers) |
| `deploy_eval_1b.py` | Serve base and merged models on `N` workers each; held-out evaluation, sampling and reference checks |
| `make_data.py` | Synthetic support-ticket train / held-out data |

`examples/merge_lora.py` (in BloomBee) merges an adapter into the base model for serving.
