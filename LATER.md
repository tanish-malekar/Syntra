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

- [ ] Sponsor integrations (Hack Princeton Fall 2026): work out how the sponsors' products could be used inside the platform.
      Only the sponsors whose products genuinely fit, not all of them. Example from you: Photon for notification alerts
      (e.g. "training stopped", "worker went down", "replacement online", "training finished").
  - First step: get the sponsor list and their products/APIs (not looked up yet; nothing here is verified).
  - Natural hook points already exist: the event stream (`worker.down`, `worker.replacement_online`, `train.completed`,
    `train.failed`, `gpu.offline`, `deploy.online`). A notifier would subscribe to those and push alerts to the job owner
    and to providers (e.g. "your GPU was leased", earnings).
  - Other places a sponsor product might fit (ideas only): model/data storage, auth/payments (we simulate payments),
    monitoring/logging, hosting the backend at a stable public address, email.
  - Decide per sponsor: does it earn its place in the product, or is it a gimmick? Keep the demo working without them.

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
- [~] QLoRA / 4-bit: BUILT (BloomBee branch `qlora-4bit`: `--compress_weight` = FlexGen 4-bit group quantization, groups of 64).
  - [x] Flag on the worker; the Llama path decompresses each layer before `lora_linear`; LoRA backward keeps only the compressed
        handle (`_FrozenCompressedLinear`), otherwise autograd kept every layer's float16 copy and 4-bit used MORE memory than FP16.
  - [x] Loss parity on this Mac (TinyLlama, 2 workers, 15 steps, same seed): 4-bit 2.10 -> 1.66, FP16 2.14 -> 1.69.
  - [x] Memory per worker (11 layers): idle 0.6-0.9 GB vs 1.4-1.7 GB FP16; GPU memory during training ~360 MB vs ~1150 MB
        (the allocator cache is released after each backward once it exceeds `BLOOMBEE_QLORA_CACHE_SLACK_MB`, default 256).
  - [x] Step time: ~2x slower than FP16 on MPS (decompression + cache release; 4.3 s without release vs 6.5 s with, FP16 2.8 s).
  - [x] Platform: `weight_bits` on jobs and deployments, planner memory model, only 4-bit-capable agents, UI toggles.
  - [x] Deploy: the adapter is merged into float16 (as before); the user picks float16 or 4-bit serving.
  - [ ] Real end-to-end run (`WEIGHT_BITS=4 DEPLOY_BITS=4 scripts/e2e_real.py full`): blocked on this Mac until a reboot
        (leaked POSIX semaphores, see below). 4-bit SERVING has not run on real hardware yet.
  - [ ] Accuracy on the held-out ticket task vs FP16 LoRA (48/48).
  - [ ] CUDA check on an idle context3 GPU (with approval); push `qlora-4bit` and point join.sh at it (with approval).
- [ ] macOS leaked semaphores (found 2026-11-14): with the spawn start method every multiprocessing lock is a NAMED POSIX semaphore
      until Python's resource_tracker unlinks it. SIGKILLing a worker's whole process group killed the tracker too, so each
      killed BloomBee worker leaked names until reboot; after ~10,000 (kern.posix.sem.max) no Python program on the Mac can
      create a lock ("OSError: [Errno 28] No space left on device"). Fixed in the agent, swarm runner and e2e_real
      (kill the group but spare `multiprocessing.resource_tracker`); this Mac still needs one reboot to clear the old leaks.
- [x] A worker that crashed while starting could hang forever (the platform waited for it): the agent now kills a worker that
      printed a traceback and is not online 15 s later, or is not online after 20 minutes, and reports why (`failed_to_start`).
- [ ] Pre-existing flaky test: tests/test_deployments_e2e.py::test_train_then_come_back_later_deploy_compare_and_evaluate
      (2/8 failures on main: the 1-GPU base deployment sometimes reports 2 GPUs).
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
- [x] Verified on real GPUs: full real run on one Mac (train, kill+recover 8.2 s, deploy, compare, evaluate 48/48 vs 0/48, serving fail-over 8.3 s
      with an identical answer) and the two-machine run (Mac + context3 RTX A6000s, replacement across machines, both providers paid).
- [x] Cross-machine recovery was slow (250 s): root cause was `nvidia-smi` on context3 blocking ~110 s in the kernel right after a CUDA worker is SIGKILLed; the agent's heartbeat waited on it, so the platform declared the healthy worker dead. Fixed: heartbeats no longer wait for the probe (`Agent.detect_now`), start_worker refuses a GPU it cannot check. Two-machine recovery now 13-14 s, also when the stall occurs.
- [ ] Admin screen for verifying providers / editing class prices (API exists, no UI).
- [ ] Provider payouts, real money, email verification and password reset (none built; amounts are simulated).
- [ ] Production hardening: HTTPS/cookie secure flag, rate limits beyond login, quotas per user, audit log, Postgres instead of SQLite.
- [ ] NAT traversal for providers behind home routers: BUILT, NOT YET PROVEN ON TWO REAL NETWORKS. `agent --relay` carries worker/DHT
  traffic over the agent's WebSocket through the platform host (`agent/syntra_agent/relay.py`, `backend/app/relay_hub.py`); the
  one-command `join.sh` installs and starts it. Passing: 4 relay unit tests, an agent<->backend relay test. Still to prove: a real
  second MacBook + context3 through a public address (`scripts/e2e_live.py`). Limits: all worker traffic is relayed through the
  platform host (fine for the demo, a bandwidth bottleneck at scale); relay ports are 31100+ (10 per agent, 100 agents).
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

## Pip's Run (mario/) leftovers
- [ ] Syntra integration: run the Mario RL training as a Syntra job on leased GPUs (the demo is standalone today).
- [ ] The real NES game (momentum, frame-tight timing, RAM-decoded state) once our own platformer works.
- [x] Level 1 target reached: 97.5% on 400 unseen seeds (0.5B, rank-128 LoRA, expert corrections + PPO). You stopped
      further runs at this point (2026-11-13).
- [ ] Remaining 2.5%: all enemy collisions after the tower (tile ~170), between the first pipes (45-47) and under the
      coin blocks (141-153). Untried: more correction + PPO cycles from the final model (a cluster job for this was
      set up and stopped before its first round), longer PPO at lr 3e-6.
- [ ] Rank-256 adapter: skipped at your request (one context3 attempt died after round 1 at 75%).
- [ ] Qwen2.5-1.5B: one attempt reached 57% after 2 correction rounds with little training; not pursued.
- [x] Speed: the shared system-prompt KV is computed once per batch (rollouts 26 s -> 16 s per iteration).
- [ ] AI cluster: venvs ready in /home/tmalekar/mario-rl (venv = CUDA 12.8 for RTX 8000/H100; venv-v100 = CUDA 12.6
      for V100). Use fp32 there: fp16 breaks Qwen2.5 and V100s have no bf16.
- [ ] Old processes of yours on context3 from 11-26 days ago (BloomBee `grad_parity.py`, `train_smoke.py`,
      `e2e_check.py`, `auto_apply.worker`): not using GPUs, left alone; decide whether to kill them.

## Changelog
- 2026-11-12: file created from the planning conversation.
- 2026-11-12: 2-worker deploy run finished; sampling verified on the served path.
- 2026-11-12: marketplace is in scope now (moved to SCOPE.md); only cloud spot instances stay deferred.
- 2026-11-12: backend M1-2 built (auth, data model, pool/provider API, events, ledger): 54 tests + 15 live checks pass.
- 2026-11-12: real full e2e PASSED (scripts/e2e_real.py full); two-machine e2e PASSED (scripts/e2e_context3.py).
- 2026-11-12: platform built end to end (README); agent file delivery; replacement-waiting event.
- 2026-11-12: serve-path fail-over and download-replacement tested; scripts moved into Syntra/experiments.
- 2026-11-12: full fine-tuning added as a later item.
- 2026-11-12: QLoRA kept as a later item with the steps needed.
- 2026-11-12: risk pass: 3 stages and frozen-worker fail-over verified; QLoRA/4-bit found unavailable; Qwen3
  cause not found; memory and step times recorded.
- 2026-11-12: Pip's Run (mario/): own platformer + Qwen2.5-0.5B trained with PPO on context3 GPUs 2 and 3.
- 2026-11-13: Pip's Run final model clears Level 1 on 97.5% of 400 unseen seeds; all runs stopped and cleaned up.
- 2026-11-14: QLoRA (4-bit base) built and measured; provider desktop app (Stage 2 of ROADMAP.md); leaked-semaphore and
  hung-worker fixes.

## Laptop lid
- [x] Agent polls the lid (macOS `AppleClamshellState`, Linux ACPI); on close it reports its GPUs offline and SIGKILLs its workers at once, so the platform replaces them without waiting for a heartbeat timeout. Tested with a simulated lid (agent + real backend).
- [ ] Not tested with a physical lid: macOS may suspend the machine before the agent's next 0.3 s poll on battery; the 8 s heartbeat timeout remains the fallback.
