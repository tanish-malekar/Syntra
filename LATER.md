# Later / deferred items

Things we decided to do later, parked ideas, and known open issues. Finished items are removed; what was done is in the
"Changelog" at the bottom. Status: `[ ]` open, `[~]` partly done.

## Deferred by you
- [ ] Sampling for LoRA adapter generation (`DistributedLoRA.generate` is greedy-only: `argmax`).
      Add `do_sample`, temperature, top-p, top-k and a seed. Client-side change only.
- [ ] Sampling controls on the Deploy page and playground (temperature, top-p, top-k, max tokens, seed).
      Base-vs-fine-tuned comparison stays greedy or uses a fixed seed. Held-out evaluation stays greedy.

- [~] Full fine-tuning (all weights): built as DiLoCo data-parallel replicas (ROADMAP.md stage 4). Still open:
  - [ ] A real-size run (Qwen2.5-0.5B on context3, needs GPU approval) and bf16 replicas on CUDA.
  - [ ] Models that do not fit on one GPU (sharding / pipeline + replicas, e.g. replicated stages with peer state transfer).

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
- [ ] Intermittent stall at the start of a served generation: once the first worker's direct push of its result to the
      second worker failed (`_push_outputs` / `rpc_push`), the second worker waited until the client's request timeout
      (60 s), then the client's recovery path finished the job. Seen once in 3 runs; not in 96 earlier generations.
      Keep client `request_timeout` small (about 15 s) until the cause is found.
- [ ] 4 or more pipeline stages.
- [ ] Slow-link emulation on macOS: `scripts/mac_throttle.sh` is untested (needs sudo; pf/dummynet rules unverified).
- [ ] Laptop behavior: sleep (`caffeinate`), battery, thermals, 8-16 GB Macs.
- [ ] Fail-over details still open:
  - [ ] Trainer crash needs checkpoints (checkpointing exists in `train_lora.py`; not tested with the platform).
  - [ ] The scripts' "replacement online" time is unreliable (a stale DHT record of the dead worker looks ONLINE);
        trust the "first step / generation resumed" times instead.
  - [ ] Lid-close takeover across networks: MacBook B closes its lid and an idle context3 GPU (joined through the relay)
        takes over. The pieces ran separately in the live run; the takeover itself was never completed.
- [~] QLoRA / 4-bit: built (BloomBee `--compress_weight`, now on `mac-mps-ft`). Still open: a real end-to-end run
      (`WEIGHT_BITS=4 DEPLOY_BITS=4 scripts/e2e_real.py full`), 4-bit serving on real hardware, accuracy vs FP16 LoRA
      (48/48), and a CUDA check on an idle context3 GPU.
- [ ] RL service real run: `scripts/e2e_rl.py` (GRPO through the SDK + managed DPO on TinyLlama) has not run on real GPUs.
- [ ] Provider app: manual click-through of the macOS app (built and unit-tested).
- [ ] Bigger models (7B+): split-across-GPUs proof run (not done; needs more memory than this Mac comfortably has).
- [~] Memory and step time on the Mac (TinyLlama-1.1B, batch 2, seq 128): worker process RSS about 0.6 GB (first
      stage), 1.0 GB (last stage) with 2-3 stages; step time about 1 s with 2 stages and 1-2.5 s with 3 stages.
      RSS likely understates GPU memory; not measured with a GPU profiler.
- [ ] macOS leaked semaphores (found 2026-11-14): with the spawn start method every multiprocessing lock is a NAMED POSIX semaphore
      until Python's resource_tracker unlinks it. SIGKILLing a worker's whole process group killed the tracker too, so each
      killed BloomBee worker leaked names until reboot; after ~10,000 (kern.posix.sem.max) no Python program on the Mac can
      create a lock ("OSError: [Errno 28] No space left on device"). Fixed in the agent, swarm runner and e2e_real
      (kill the group but spare `multiprocessing.resource_tracker`); a Mac that hit it still needs one reboot to clear the old leaks.
- [ ] Pre-existing flaky test: tests/test_deployments_e2e.py::test_train_then_come_back_later_deploy_compare_and_evaluate
      (2/8 failures on main: the 1-GPU base deployment sometimes reports 2 GPUs).
- [ ] Remaining hard-coded CUDA spots (about 40, mostly cache offload, speculative decoding, Falcon); none on the
      tested path.
- [ ] Trust and privacy for shared GPUs: owners see the data. Use public/synthetic data, "Trusted GPUs only"
      allowlist, no real payments.

## Platform
- [ ] Admin screen for verifying providers / editing class prices (API exists, no UI).
- [ ] Provider payouts, real money, email verification and password reset (none built; amounts are simulated).
- [ ] Production hardening: HTTPS/cookie secure flag, rate limits beyond login, quotas per user, audit log, Postgres instead of SQLite.
- [ ] Relay limits: all relayed worker traffic goes through the platform host (fine for the demo, a bandwidth bottleneck at
      scale); relay ports are 31100+ (10 per agent, 100 agents). The temporary public address (localhost.run / Cloudflare
      quick tunnel) changes on every restart; campus DNS cannot resolve `*.trycloudflare.com`.
- [ ] Cloud spot instances in the pool (AWS/GCP/Azure/GPU clouds).
  - [ ] Our AWS account is blocked: it sits in an AWS Organization whose policies deny all
        spot launches and allow EC2 only in `us-east-2`; it is on the Free plan ($100 credit) and its GPU quotas are 0.
        Likely fix: upgrade to the paid plan, then request "All G and VT Spot Instance Requests" = 8 vCPUs.
  - [ ] Platform launches and terminates instances through provider APIs (needs cloud credentials held by the
        platform: budget caps, per-job cost limits; you handle credentials and billing, not me).
  - [ ] Machine image with BloomBee + Python 3.12 preinstalled; model files cached in a bucket per region.
  - [ ] Preemption handling: listen for the interruption notice (about 2 minutes on AWS, about 30 s on GCP),
        tell the orchestrator to drain and start a replacement before the instance disappears.
  - [ ] Recovery time on real cloud: boot plus model download is minutes, not the 8 s measured locally.
  - [ ] Price-aware scheduling (spot prices vary) and cost metering per run.
  - [ ] These are NVIDIA/CUDA machines: exercises the CUDA path and mixed Mac + CUDA swarms.
- [ ] Benchmark against Together AI: their published recipe is Llama-3-8B + LoRA on MathInstruct (47.2% -> 65.2% on 1,000
      held-out problems, under $100, 2-14 h). The full recipe on two GPUs across networks is estimated at 2-3 weeks; a
      scaled-down run (2-5k examples, 200 eval problems) at 6-15 h. Not run.

## Housekeeping
- [ ] Decide what `hackathon` should contain (it currently equals upstream `main`; fault-tolerance and Mac
      support live on `mac-mps-ft`).
- [ ] Rebase `mac-mps-ft` onto upstream `main` when it moves; consider upstream PRs for MPS support and
      LoRA fault tolerance.

## Pip's Run (mario/) leftovers
- [ ] Syntra integration: run the Mario RL training as a Syntra job on leased GPUs (the demo is standalone today; you chose
      to leave it standalone on 2026-11-14).
- [ ] The real NES game (momentum, frame-tight timing, RAM-decoded state) once our own platformer works.
- [ ] Remaining 2.5%: all enemy collisions after the tower (tile ~170), between the first pipes (45-47) and under the
      coin blocks (141-153). Untried: more correction + PPO cycles from the final model (a cluster job for this was
      set up and stopped before its first round), longer PPO at lr 3e-6.
- [ ] Rank-256 adapter: skipped at your request (one context3 attempt died after round 1 at 75%).
- [ ] Qwen2.5-1.5B: one attempt reached 57% after 2 correction rounds with little training; not pursued.
- [ ] AI cluster: venvs ready in /home/tmalekar/mario-rl (venv = CUDA 12.8 for RTX 8000/H100; venv-v100 = CUDA 12.6
      for V100). Use fp32 there: fp16 breaks Qwen2.5 and V100s have no bf16.
- [ ] Old processes of yours on context3 from 11-26 days ago (BloomBee `grad_parity.py`, `train_smoke.py`,
      `e2e_check.py`, `auto_apply.worker`): not using GPUs, left alone; decide whether to kill them.

## Laptop lid
- [ ] Not tested with a physical lid: macOS may suspend the machine before the agent's next 0.3 s poll on battery; the 8 s
      heartbeat timeout remains the fallback. (Lid polling and instant offline report work with a simulated lid.)

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
- 2026-11-14: relay (NAT traversal) proven on real networks: MacBook B on another network joined with the one-command
  `join.sh`, trained 12 steps with this Mac through the relay (layers 0-15 M4 Pro, 15-22 M4, about 2.6 s/step), and a
  context3 GPU joined as a spare through the relay. Fixed on the way: lost worker events on reconnect, WebSocket keep-alive
  deadlock, DHT expiry too short over the relay (relay agents use 10 s / 120 s). The install script pins Python 3.12.
- 2026-11-14: QLoRA (4-bit base) built and measured (TinyLlama, 2 workers: same loss curve as FP16, ~360 MB vs ~1150 MB GPU
  memory per worker during training, steps ~2x slower); provider desktop app (ROADMAP.md stage 2); leaked-semaphore fix; a worker
  that crashes while starting is now killed and reported (`failed_to_start`) instead of hanging its job.
- 2026-11-14: Stage 3 (RL: managed DPO, sessions API, `syntra` SDK with GRPO/PPO/DPO) and Stage 4 (DiLoCo full fine-tuning,
  real local run passed: loss 9.04 -> 0.90, killed replica replaced in 4.6 s) built. AWS CLI set up; spot blocked on the
  account (see Platform).
- 2026-11-14: Rohan's branch `feature/qlora-rl-diloco-app` merged into `main`; BloomBee `qlora-4bit` merged into `mac-mps-ft`.
- 2026-11-14: finished items removed from this file; their record is this changelog.
