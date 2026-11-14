# Roadmap: QLoRA, provider app, RL, full fine-tuning (DiLoCo)

Branch `feature/qlora-rl-diloco-app`. Each stage is committed only after its tests pass. Nothing is pushed without approval.
Status: `[ ]` open, `[x]` done, `[~]` partly done.

| Stage | What | Depends on |
|---|---|---|
| 1 | Finish QLoRA (4-bit base) | - |
| 2 | Provider app: one-click sharing, status, GPU time used, earnings | - |
| 3 | RL: managed DPO jobs + `syntra` Python SDK (Tinker-style primitives, GRPO/PPO loops) | 1 (LoRA/QLoRA trainer) |
| 4 | Full fine-tuning, DiLoCo-style data parallelism | 2 (agent worker kinds) |

---

## Stage 1: finish QLoRA

Already done (see `LATER.md` changelog):
- BloomBee `--compress_weight` flag (FlexGen 4-bit group quantization) on branch `qlora-4bit` in `~/.syntra/BloomBee`.
- Memory-light backward: decompressed weights are not kept for backward.
- `weight_bits` (16 | 4) on training jobs and deployments, end to end. The scheduler only picks 4-bit-capable agents.
- Planner memory model, UI toggles, tests.

Remaining:
- [ ] Real end-to-end run on this Mac (blocked until a reboot clears ~10,000 leaked semaphores; see `LATER.md`) with `WEIGHT_BITS=4 DEPLOY_BITS=4 scripts/e2e_real.py full`: train, kill and recover a worker, merge, serve in 4-bit, compare, and evaluate on the held-out tickets.
- [ ] Fix whatever that run finds (4-bit serving has not run on real hardware yet).
- [x] Commit the BloomBee change on its branch.
- [x] Docs: `README.md`, and the QLoRA section in `LATER.md` with measured numbers (loss parity, memory, step time).
- [ ] Follow-ups that need approval: push the BloomBee branch and point `join.sh` at it; run a CUDA check on an idle context3 GPU.

Edge cases:
- **Old agents and old BloomBee checkouts** report `supports_4bit=false`. A 4-bit job then waits in the queue with a reason, and it is never sent to such an agent.
- **Replacement GPUs** must also support 4-bit.
- **Non-Llama models** get a 422 response, because the 4-bit path exists only for Llama blocks.
- **Unknown widths**: a `weight_bits` value other than 16 or 4 gets a 422 from the API, and the agent refuses it.
- **Deploying a QLoRA model**: the adapter is merged into float16 weights, as usual. The user chooses float16 or 4-bit serving.

## Stage 2: provider app (macOS menu bar, SwiftUI) [x] built and tested; manual click-through pending

Goal: no terminal. The provider signs in once and clicks **Share my GPU**. The app shows:
- Shared / Paused / Disconnected;
- whether a job is using the GPU right now;
- GPU time used (today and in total);
- earnings.

Linux machines keep the one-line command, and their agent gets the same local status endpoint.

### Pieces
1. **Agent control API** (Python, `agent/syntra_agent/control.py`), served on `127.0.0.1` on a free port. The port and a random secret go into `~/.syntra/agent.json` with mode 0600.
   - `GET /status`: connection state, the GPUs and their status, workers running, last heartbeat, last error.
   - `POST /pause`: stops accepting work. Running workers finish or are drained, and the GPUs are reported `offline`.
   - `POST /resume`.
   - `POST /quit`.
2. **Agent service mode** (`python -m syntra_agent service`): reads its settings (server, token, models directory) from `~/.syntra/config.json`. It reconnects with backoff. It never exits on transient errors; it exits only when the token is revoked, and reports that state first.
3. **Backend: provider usage API.** `GET /providers/me/usage` returns, per GPU and in total:
   - GPU seconds leased, now and in total;
   - earnings: settled, plus a running estimate for open leases;
   - the current job, if any;
   - the last 30 days by day.

   Settled numbers come from the existing ledger. A running lease is estimated from `lease.started_at` (not yet billed).
4. **Backend: app pairing.** `POST /providers/app-pair`, signed in with email and password, returns a device token.
   - Creates (or reuses) the user's provider profile, plus an agent token labelled with the machine name.
   - Calling it again from the same machine (hardware UUID) reuses the token rather than creating more.
5. **Mac app** (`provider-app/`, Swift Package, `SyntraProvider.app`):
   - **First run**: server URL and sign-in. The app pairs the device and stores the token in the macOS Keychain.
   - **Setup check**: Python 3.12 and git present? The app runs the same install steps as `join.sh` (venv in `~/.syntra`, BloomBee, agent) with progress shown, and sends errors to a log viewer.
   - **Share / Pause button**: Share starts the agent as a child process. The app restarts it if it crashes, but stops after 5 crashes in 2 minutes and shows the error. Pause calls `/pause`.
   - **Menu bar icon**: green = shared and idle, blue = in use, grey = paused, red = disconnected or error.
   - **Panel**: the status text, plus per-GPU usage and earnings from `/providers/me/usage`, refreshed every 10 s and on wake.
   - **Start at login** (optional, via `SMAppService`).
   - **Quit**: drain the agent (`/quit`), so the GPUs go offline at once rather than after the heartbeat timeout.

### Edge cases
| Case | Behaviour |
|---|---|
| Lid closed / sleep | The agent already reports offline on lid close. On wake the app restarts the heartbeat at once, and the status shows "Reconnecting". |
| Network lost | The agent keeps retrying with backoff, and the app shows Disconnected with the time it was last seen. Leases on this GPU are replaced by the platform after the heartbeat timeout (existing behaviour). |
| Token revoked on the website | The agent receives a 401, stops, and reports `revoked`. The app asks the user to sign in again. |
| Two copies of the app, or an agent from the terminal already running | A lock file (`~/.syntra/agent.lock`, PID check) blocks the second one. The app attaches to the running agent's control API instead of starting another. |
| GPU busy with the owner's own work | Already reported as `busy_external` by the agent. The app shows "Busy with your apps (not shared)". |
| Pause while a job is using the GPU | Ask: "Finish the current job first" (drain) or "Stop now" (kill; the platform moves the work to another GPU). |
| Battery | Pause automatically on battery if the setting is on (off by default). |
| Install fails (no Python 3.12, no git, no disk space) | Show the exact failing step and a "Copy log" button. Nothing is half-registered: the token is only used once the install succeeds. |
| Server URL wrong or unreachable | A test connection runs before pairing, with a clear error. |
| Earnings while a job is running | Shown as "running: ≈ $x" until the lease ends and the ledger settles. |

## Stage 3: RL service

Two layers, modelled on Together AI (managed DPO jobs) and Tinker / OpenPipe ART (an SDK with low-level primitives; the user's code computes rewards).

### 3a. Training primitives in the swarm runner (one trainer process per job; LoRA or QLoRA on pooled GPUs)
- A new runner mode, `session`: the runner hosts a `DistributedLoRA` model and serves requests over stdin/stdout (already used in serve mode).
  - `forward_backward(batch, loss)`: loss is `cross_entropy`, `dpo`, `ppo` or `importance_sampling`. Gradients accumulate.
  - `optim_step(lr, ...)`.
  - `sample(prompts, n, temperature, top_p, top_k, max_new_tokens, seed)`: returns tokens and their log-probs under the current policy.
  - `logprobs(sequences, reference=True|False)`: `reference=True` runs with the adapter switched off, which is exactly the base model and needs no second copy.
  - `save(name)`: puts the adapter into the model registry.
- **Sampling with temperature / top-p / top-k / seed** in the LoRA generate path (today it is greedy only; this item was deferred in `LATER.md`).

### 3b. Managed DPO job (UI and REST)
- Dataset format: `{"prompt": ..., "chosen": ..., "rejected": ...}` per line, validated on upload.
- `POST /jobs` with `method: "dpo"`, `beta`, and `base: <base model or one of your fine-tuned models>`. The recommended flow is SFT first, then DPO on top of that model.
- Reference log-probs are computed once at the start, with the adapter off.
- Events: `train.step` with `loss`, `reward_margin` and `reward_accuracy`.
- Fail-over is unchanged: workers are stateless.

### 3c. `syntra` Python SDK (`sdk/`, pip-installable)
```python
import syntra
client = syntra.Client(api_key=...)          # personal API key from the website
tc = client.training_session(base_model="TinyLlama-1.1B-Chat", rank=16, weight_bits=4, gpus=2)
out = tc.sample(prompts, n=8, temperature=1.0)            # returns tokens + logprobs
tc.forward_backward(batch, loss="ppo", clip=0.2)          # or "importance_sampling", "dpo", "cross_entropy"
tc.optim_step(lr=1e-5)
tc.save("my-grpo-model")                                  # lands in My models, deployable like any other
```
Ready-made loops:
- `syntra.rl.grpo(tc, prompts, reward_fn, group_size=8, ...)`: group-relative advantages, PPO-clip loss, optional KL to the reference.
- `syntra.rl.ppo(...)`: same loss, advantages from the user, or reward minus a moving baseline.
- `syntra.rl.dpo(tc, pairs, beta)`.

The reward function runs on the user's machine. Only token ids and floats cross the network.

**Backend**: API keys (hashed, revocable). `/sessions` endpoints proxy each call to the session's runner with an idempotency key, and the call is not repeated after a retry.

### Edge cases
- **A worker dies mid-call**: the existing LoRA client fail-over retries that call. Session state lives in the trainer (adapters and optimizer), so nothing is lost.
- **The trainer process dies**: the session is marked failed, and the last `save()` / automatic checkpoint (every N optimizer steps) can be resumed into a new session.
- **SDK client disappears**: the session times out after an idle limit (default 10 min). It auto-saves, then releases the GPUs, and billing stops.
- **Duplicate calls on network retry**: idempotency keys, so a step is never applied twice.
- **Mismatched shapes / empty groups / all-equal rewards**: an all-equal group gets zero advantage and is skipped, with a warning in the response.
- **Prompts longer than `seq_len`**: rejected with a clear message.
- **Rewards that are NaN or inf**: rejected.
- **Billing**: same lease billing as training, and the session shows its running cost.

## Stage 4: full fine-tuning, DiLoCo style

Every replica holds the whole model, with trainable weights and an AdamW inner optimizer, on one GPU. It trains on its own shard for H inner steps. The outer step averages the replicas' pseudo-gradients (Δθ = θ_start − θ_end) and applies Nesterov momentum.

- **New job kind `full_ft`, `method: "diloco"`**: `replicas`, `inner_steps` (H), `outer_lr`, `outer_momentum`, `inner_lr`.
- **Planner**: a replica needs about 16 bytes per parameter plus activations (Qwen2.5-0.5B ≈ 8-9 GB). It picks N GPUs that each fit the full model. There is no layer split.
- **Agent worker kind `diloco_replica`**: a small trainer script (`agent/syntra_agent/diloco_worker.py`, torch + transformers only, no BloomBee). Per round it:
  1. downloads the global weights,
  2. trains H steps on its shard,
  3. uploads Δθ in bf16 (half the bytes),
  4. waits for the next round.
- **Outer coordinator** in the job runner on the platform host:
  - collects Δθ;
  - applies the outer step when all live replicas have reported, or after a deadline with the replicas that did report (weighted by their tokens);
  - publishes the new global weights;
  - saves a checkpoint every round.
- **Transfer**: the existing token-authenticated, lease-scoped file endpoints, extended with upload and sha256 checks. Relay-mode agents use the same HTTP path.
- **Result**: a full model in the registry (no merge step). It deploys like a merged model.
- **Test target**: Qwen2.5-0.5B on idle context3 GPUs, with your approval each time. Local tests use a tiny random Llama so CI runs on this Mac.

### Edge cases
| Case | Behaviour |
|---|---|
| A replica dies mid-round | The round closes at its deadline with the replicas that reported. A replacement GPU is leased and joins at the next round from the global weights. The dead replica's data shard is reassigned. |
| All replicas die | The job waits for GPUs. The last outer checkpoint is kept, and it resumes from there. |
| Coordinator / backend restart | Rounds are numbered and stored on disk. Replicas re-upload if the coordinator lost their upload (idempotent by round + replica id). |
| A slow replica | The deadline-based round close above, plus late uploads for an old round are dropped. |
| Corrupt or partial upload | sha256 is checked, and the replica re-uploads. |
| Memory too small | The planner excludes the GPU, so the job queues with the reason. |
| Mac (MPS) vs CUDA replicas | Both supported. bf16 on CUDA; fp32 on MPS and V100 (fp16 breaks Qwen2.5, see `mario/README.md`). |
| Diverging loss (NaN) | The replica reports it. The coordinator drops that Δθ, and the job fails after 2 consecutive NaN rounds. |

---

## Verification per stage
1. QLoRA: `scripts/e2e_real.py full` with 4-bit; backend, agent and frontend test suites.
2. Provider app:
   - agent control-API tests;
   - backend usage/pairing tests;
   - `swift build`, plus a manual run with the demo stack (share, pause, quit, revoke token, kill agent);
   - a screenshot of the panel.
3. RL:
   - unit tests of the DPO, PPO and GRPO losses against reference formulas;
   - a session API e2e with the fake runner;
   - a real GRPO run on TinyLlama (reward = valid JSON) on this Mac whose reward rises;
   - a real DPO run.
4. DiLoCo:
   - local e2e with a tiny Llama and 3 simulated replicas, killing one mid-round;
   - with approval, Qwen2.5-0.5B on context3 GPUs, comparing the loss with a single-GPU baseline.
