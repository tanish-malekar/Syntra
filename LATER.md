# Later / deferred items

Things we decided to do later, parked ideas, and known open issues. Newest changes go in the
"Changelog" at the bottom. Status: `[ ]` open, `[x]` done, `[~]` partly done.

## Deferred by you
- [ ] Sampling for LoRA adapter generation (`DistributedLoRA.generate` is greedy-only: `argmax`).
      Add `do_sample`, temperature, top-p, top-k and a seed. Client-side change only.
- [ ] Sampling controls on the Deploy page and playground (temperature, top-p, top-k, max tokens, seed).
      Base-vs-fine-tuned comparison stays greedy or uses a fixed seed. Held-out evaluation stays greedy.
- [x] Verify sampling really varies on the served path. Tuned model: identical samples (very confident);
      base model: three different outputs for three seeds, so `do_sample=True` works on the served path.

- [ ] Full fine-tuning (all weights) as a later training mode, next to LoRA. Much bigger change than QLoRA:
  - Workers are stateless and frozen today; full fine-tuning needs trainable weights, gradients and optimizer state
    on the workers (roughly 16 bytes per parameter with Adam, versus 2 for inference weights), so far less model
    fits per GPU.
  - The "stateless worker, trainer owns the adapters" fail-over trick stops working: a dead worker takes weights
    and optimizer state with it, so we need periodic checkpoints or replicas.
  - Several pipelines would need weight synchronization (DiLoCo-style low-communication methods become relevant).
  - Deploy gets simpler (no merge step) but the saved weights are multi-GB, not a small adapter.
  - Leads to check first: upstream branches `train_lm_head` and `train_mlp` in `origin` (unread), and the
    `ptune` code, for how BloomBee already trains parts of a model.

## Parked optimizations (not needed for the hackathon core)
- [ ] Log bytes per step split into activations / adapter weights / gradients (cheap; decides everything below).
- [ ] Adapters only on the last K layers (backward stops early; K slider in the Planner).
- [ ] Cache frozen-layer outputs after epoch 1 (trainer is already the hub).
- [ ] Send adapter weights to workers once per optimizer step (version-tagged cache; trainer stays source of truth).
- [ ] Worker-side optimizer updates (needs checkpoints/replication for fail-over).
- [ ] Direct worker-to-worker passing for the LoRA path (today everything goes through the trainer).
- [ ] 8-bit / 4-bit activation compression between stages (AQ-SGD style).
- [ ] Try `BLOOMBEE_LOSSLESS_WRAPPER=1` and `--micro_batch_size`; compare bytes and step time.
- [ ] Workers have a `--compression` option for network tensors (hivemind compression types). The LoRA path sends
      activations with `CompressionType.NONE` explicitly; check whether it can be switched on there.
- [ ] Side-tuning experiment (LST / MobiLLM) to remove backward traffic through the base model.
- [ ] Exact adapter aggregation for several pipelines (Fed-SB / FedEx-LoRA); data-parallel trainers
      (code is on branch `lora-dp-wip`).
- [ ] Optional: beam search on the served path.

## Open risks / unverified
- [ ] Qwen3 merged serving on MPS: client `TimeoutError` after 4 verified steps (Llama passes). Cause not found;
      the Qwen3 client model (`models/qwen3/model.py`) differs from Llama's (past_key_values handling etc.). Training
      and resume pass on Qwen3. Low priority: use a Llama-family model for the demo.
- [x] 3 pipeline stages: TinyLlama split 8/7/7, middle worker killed, replacement online in 5.6 s, training finished.
- [ ] Intermittent stall at the start of a served generation: once the first worker's direct push of its result to the
      second worker failed (`_push_outputs` / `rpc_push`), the second worker waited until the client's request timeout
      (60 s), then the client's recovery path finished the job. Seen once in 3 runs; not in 96 earlier generations.
      Keep client `request_timeout` small (about 15 s) until the cause is found.
- [ ] 4 or more stages, and stages on different machines.
- [ ] Two real machines (second Mac): networking, NAT, firewall prompts, `--announce_maddrs`.
- [ ] Slow-link emulation on macOS: `scripts/mac_throttle.sh` is untested (needs sudo; pf/dummynet rules unverified).
- [ ] Laptop behavior: sleep (`caffeinate`), battery, thermals, 8-16 GB Macs.
- [~] Fresh-machine install: works on Python 3.12 (wheel built without Go, no cache). Python 3.14 inconclusive
      (hivemind dry-run fine; torch/numpy<2/peft dry-run printed nothing). Needs an install script and a version
      check in the agent: pin 3.12.
- [~] Fail-over details:
  - [x] Hung (frozen, SIGSTOP) worker: recovered. With trainer `request_timeout` 15 s and 8 s simulated detection,
        replacement online 19.6 s after the freeze, first step 23.6 s after. Recovery is bounded by timeouts.
  - [ ] Trainer crash needs checkpoints (checkpointing exists in `train_lora.py`; not tested with the platform).
  - [x] Served (inference) path: killed a serving worker mid-generation (TinyLlama, 2 workers, 300 tokens); the client
        recovered by itself (session history replayed to a replacement) and the text was IDENTICAL to the no-fault run.
        Extra time = client `request_timeout` (15 s here) + replacement start: +15 s with local files.
  - [x] Replacement that must download its weights (empty cache, fetched from Hugging Face, whole 2.2 GB file):
        generation finished identically, +83 s total vs +15 s with local files. So expect about a minute or more per
        replacement on this network unless model files are cached near the GPU.
  - [ ] The scripts' "replacement online" time is unreliable (a stale DHT record of the dead worker looks ONLINE);
        trust the "first step / generation resumed" times instead.
- [ ] QLoRA / 4-bit (planned for later): NOT available today. The README says `--quant_type nf4`, but the flag accepts
      only NONE/INT8 and INT8 is implemented only for DeepSeek-V3 experts. Treat the README claim as stale.
  - FlexGen already has 4-bit group-wise weight compression (policy `compress_weight`, `comp_weight_config`
    num_bits=4, group_size=64; compressed-weight decompress branches in `pytorch_backend.py`), but `server.py:329`
    hardcodes `compress_weight=False` and there is no CLI flag.
  - [ ] Add a flag to turn it on; check the LoRA path (`lora_linear(... w_q.data ...)`) works with compressed weights.
  - [ ] Verify gradients against a reference with a quantized base, and the accuracy loss on the ticket task.
  - [ ] Check it on MPS (device-agnostic ops?) and CUDA; bitsandbytes is CUDA-only, so don't depend on it.
  - [ ] Deploy path: you cannot merge an adapter into a 4-bit base; either merge into FP16 then re-quantize, or
        serve the quantized base with the adapter separately.
- [ ] Bigger models (7B+): split-across-GPUs proof run (not done; needs more memory than this Mac comfortably has).
- [~] Memory and step time on the Mac (TinyLlama-1.1B, batch 2, seq 128): worker process RSS about 0.6 GB (first
      stage), 1.0 GB (last stage) with 2-3 stages; step time about 1 s with 2 stages and 1-2.5 s with 3 stages.
      RSS likely understates GPU memory; not measured with a GPU profiler.
- [ ] Remaining hard-coded CUDA spots (about 40, mostly cache offload, speculative decoding, Falcon); none on the
      tested path.
- [ ] Trust and privacy for shared GPUs: owners see the data. Use public/synthetic data, "Trusted GPUs only"
      allowlist, no real payments.

## Platform status
- [x] Backend, agent, scheduler, orchestrator, registry, deployments, compare/evaluate, frontend (see README). Tested: 92 backend,
      17 agent, 6 frontend tests; real runs on this Mac (train, kill and recover, deploy, compare, evaluate).
- [x] Marketplace for individuals and data centers: provider profiles, per-machine agent tokens, pool, fixed per-class prices, simulated ledger.
- [x] Model files reach agents on other machines (token-authenticated, lease-scoped, sha256-checked).
- [ ] Verify on real GPUs: serving fail-over on the real stack (the fake-swarm test passes; the real run was blocked by a test setup
      with no spare GPU, since fixed), and the two-machine run (`scripts/e2e_context3.py`, see changelog for its last result).
- [ ] Admin screen for verifying providers / editing class prices (API exists, no UI).
- [ ] Provider payouts, real money, email verification and password reset (none built; amounts are simulated).
- [ ] Production hardening: HTTPS/cookie secure flag, rate limits beyond login, quotas per user, audit log, Postgres instead of SQLite.
- [ ] Tunnel setup is manual (`bridge/tunnel.py`); no automatic NAT traversal for providers behind home routers.
- [ ] Later: cloud spot instances in the pool (AWS/GCP/Azure/GPU clouds).
  - [ ] Platform launches and terminates instances through provider APIs (needs cloud credentials held by the
        platform: budget caps, per-job cost limits; you handle credentials and billing, not me).
  - [ ] Machine image with BloomBee + Python 3.12 preinstalled; model files cached in a bucket per region.
  - [ ] Preemption handling: listen for the interruption notice (about 2 minutes on AWS, about 30 s on GCP),
        tell the orchestrator to drain and start a replacement before the instance disappears.
  - [ ] Recovery time on real cloud: boot plus model download is minutes, not the 8 s measured locally.
  - [ ] Price-aware scheduling (spot prices vary) and cost metering per run.
  - [ ] These are NVIDIA/CUDA machines: exercises the CUDA path and mixed Mac + CUDA swarms.

- [ ] Email verification and password reset (no email sending built; accounts work with email + password only).

## Housekeeping
- [ ] Decide what `hackathon` should contain (it currently equals upstream `main`; fault-tolerance and Mac
      support live on `mac-mps-ft`).
- [ ] Rebase `mac-mps-ft` onto upstream `main` when it moves; consider upstream PRs for MPS support and
      LoRA fault tolerance.
- [x] Test scripts moved to `Syntra/experiments/bloombee-mac` (with README). Syntra is not a git repository yet.

## Done so far
- [x] LoRA training on Apple GPU (MPS), 2-stage pipeline, tiny + TinyLlama-1.1B; resume exact.
- [x] Merged-model serving on MPS (Llama), exact greedy match with local Hugging Face.
- [x] Fail-over: killed worker, replacement took over, training finished (about 8 s).
- [x] Deploy test on TinyLlama-1.1B, 1 and 2 workers per model: tuned model 48/48 valid JSON vs base 0/48;
      served output matches local Hugging Face (8/8); greedy repeatable; sampling works.
- [x] Clean install on a fresh venv (no pip cache, no Go).

## Changelog
- 2026-11-12: file created from the planning conversation.
- 2026-11-12: 2-worker deploy run finished; sampling verified on the served path.
- 2026-11-12: marketplace is in scope now (moved to SCOPE.md); only cloud spot instances stay deferred.
- 2026-11-12: backend M1-2 built (auth, data model, pool/provider API, events, ledger): 54 tests + 15 live checks pass.
- 2026-11-12: platform built end to end (README); agent file delivery; replacement-waiting event.
- 2026-11-12: serve-path fail-over and download-replacement tested; scripts moved into Syntra/experiments.
- 2026-11-12: full fine-tuning added as a later item.
- 2026-11-12: QLoRA kept as a later item with the steps needed.
- 2026-11-12: risk pass: 3 stages and frozen-worker fail-over verified; QLoRA/4-bit found unavailable; Qwen3
  cause not found; memory and step times recorded.
