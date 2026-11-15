# Pip's Run: a small LLM learns a platformer with RL

Our own Mario-style platformer (original art, drawn in code) plus **Qwen2.5-0.5B-Instruct** that plays it by reading
the screen as text and answering with one digit. It is trained with **PPO** (LoRA adapters + a value head) on
context3 GPUs. Standalone: nothing here touches the BloomBee app.

## Run the demo page

```bash
backend/.venv/bin/python -m uvicorn mario.server:app --port 8010
```

Open http://localhost:8010. Modes:

- **You play**: arrow keys or A/D, Space or ↑ to jump (hold it to jump higher), R to restart.
- **Scripted bot**: the simple rule-based player used for the warm-up. It clears about 9% of Level 1.
- **Model plays live**: the LLM picks every move. Start the policy worker first; it uses the Mac GPU (MPS) at
  about 50 ms per decision:
  ```bash
  mario/.venv/bin/python -m mario.train.serve --port 8011
  ```
  The checkpoint picker lists `untrained` and every adapter under `mario/runs/*/ckpt/`.
- **Replay**: re-runs a saved game (level, seed and the list of moves) with the same simulation. This needs no GPU and
  no model, so it is the backup demo. Replays saved from the model also include its move probabilities, so the bars
  still work.

The side panel shows the exact text the model sees, its six move probabilities, and the training curves from
`mario/runs/*/metrics.jsonl`.

## How it works

| Piece | File |
|---|---|
| Deterministic sim, 60 frames/s: physics, enemies, coins, coin blocks, flag | `game/sim.py` |
| Level 1 (201 tiles) and seeded random variants | `game/level.py` |
| Text state (≈50 tokens: 12x7 grid, floor/pit line, nearby enemies) | `game/textstate.py` |
| Scripted bot, beam-search solver (proves levels are clearable) | `game/bot.py`, `game/solve.py` |
| Episodes, rewards, N games at once | `game/vec.py` |
| LLM + LoRA + value head; scores for the tokens `1`-`6` at the last position | `train/policy.py` |
| Warm-up (imitate the bot), PPO, evaluation + replays | `train/bc.py`, `train/ppo.py`, `train/evaluate.py` |
| context3: push, set up, launch, status, fetch, stop | `remote.py` |
| Page server (WebSocket, 60 frames/s) and canvas renderer | `server.py`, `web/` |

**Moves** (one decision every 6 frames, 0.1 s): 1 run right · 2 run right + long jump · 3 run right + short hop ·
4 left · 5 jump in place · 6 wait. The model's answer is one digit (the plan said letters A-F; digits avoid clashing
with `E` = enemy and `F` = flag on the grid).

**Reward** (×0.1): +1 per tile of new rightmost progress, −0.05 per decision, +0.5 coin, +1 stomp, −15 death,
+30 flag. An episode ends on death, at the flag, or after 3 s without progress.

**Training data** is half Level 1 with randomised start offset and enemy positions and directions, and half generated
variants. Evaluation always uses Level 1 with seeds ≥ 1,000,000, which training never uses, and takes the most likely
move.

## Training on context3

```bash
backend/.venv/bin/python -m mario.remote gpus      # look first; use only idle GPUs you approved
backend/.venv/bin/python -m mario.remote push
backend/.venv/bin/python -m mario.remote setup     # venv in /data01/tmalekar/mario (inherits torch from `bb`)
backend/.venv/bin/python -m mario.remote launch --gpu 2 --name main --hours 8 -- mario.train.chain --iters 200
backend/.venv/bin/python -m mario.remote status
backend/.venv/bin/python -m mario.remote fetch --name main      # metrics + replays (+ final/best adapters)
```

Everything remote lives under `/data01/tmalekar/mario`. Each run is pinned with `CUDA_VISIBLE_DEVICES` and wrapped in
`timeout`.

Measured on one A6000 (0.5B, fp32 with TF32): a forward pass for 64 games takes about 0.4 s, and forward + backward
about 1 s. One PPO iteration (4,096 decisions, 2 epochs) takes about 1-2 minutes.

Lessons from the first runs:
- **bf16 made PPO ratios noisy.** The same state's log-probabilities moved by up to 0.18 depending on batch and
  padding, so updates stopped on the KL limit at once. fp32 brings that down to 0.005.
- **The value loss must not train the shared LoRA weights.** The value head is now a small MLP on a detached hidden
  state.

## Results

The final model is Qwen2.5-0.5B-Instruct with a rank-128 LoRA adapter, trained with expert-correction rounds and
then 50 PPO iterations. On **400 Level 1 seeds that no run used for training or for picking checkpoints, it clears
97.5%** (390/400) and reaches 98.7% of the level on average. All 10 failures are enemy collisions (after the tower,
between the first pipes, under the coin blocks). Checkpoint: `mario/runs/r128/ckpt/r128_rl_final`; demo replays:
`mario/replays/final_win*.json`.

How we got there, in Level 1 clear rate. "Selection seeds" are the 64 or 128 seeds the runs use to pick
checkpoints; "fresh seeds" are never used for training or selection.

| Step | Clear rate |
|---|---|
| Scripted bot (warm-up teacher) | 9% |
| Warm-up imitation of the bot | 0% (gets 40% of the way) |
| PPO only, LoRA rank 16, several reward and curriculum variants | 50-78% (selection seeds); 62% on 100 fresh seeds |
| Same, after text state v2 (see below) plus expert corrections | 72-78% (selection seeds) |
| Rank-128 adapter trained on 167k expert-labelled states, then 3 correction rounds | 87% (selection); 83% (200 fresh) |
| + PPO (gentle: lr 5e-6, KL 0.01, 128 games, mid-level starts, death −30 / flag +45), iteration 25 | 97% (selection); 93.75% (400 fresh) |
| **same run, iteration 50 (final)** | **98% (selection); 97.5% (400 fresh)** |
| From scratch, PPO only (comparison) | ≤ 50% |

What mattered, in order:
1. **Text state v2.** Analysing deaths showed that while the hero was high in the air the grid hid the ground and the
   enemies below. v2 always shows the grid down to the ground row and adds `y`, `vx` and `drop` (height above the
   landing spot). Old checkpoints keep the old format (`text_version` in `meta.json`).
2. **Expert corrections with acceptable-move sets** (`train/dagger.py`). The model plays; a 2-second look-ahead
   search over the same sim marks every move that is as good as the best one; the model is trained to put its
   probability on that set. Labelling a single "best" move instead made things worse (0%): it taught last-moment jumps.
   Every round keeps all earlier labelled states.
3. **Adapter capacity.** At rank 16, every method stopped at about 75-78%, even though the model fitted 98.9% of its
   training states. Level 1 needs precise play: the search expert itself clears 96% with 2% random slips but only 67%
   with 5%. Rank 128 brought the error rate down far enough.
4. **PPO on top** fixed what imitation cannot. The expert assumes perfect follow-up; RL trains under the model's own
   play, so it favours moves that leave room for error. It took the model from 83% to 97.5% (fresh seeds) in 50
   iterations.
5. Smaller fixes: fp32 instead of bf16 (PPO ratios were noise); a value head on a detached hidden state; KL measured
   per epoch, not per minibatch (that had been cutting most updates to one or two steps); the shared system prompt
   computed once per batch (rollouts 26 s → 16 s per iteration, results unchanged).

Also learned the hard way: fp16 autocast breaks Qwen2.5 (activations overflow; a rank-256 model fell from 75% to
23%), so V100s (no bf16) train in fp32. And a 128-seed selection test is noisy: the checkpoint it ranked best scored
96.25% on fresh seeds, the final one 97.5%.

## Tests

```bash
backend/.venv/bin/python -m pytest mario/tests     # game: determinism, collisions, clearable levels, text state
mario/.venv/bin/python -m pytest mario/tests       # + policy smoke test with a tiny random model (2 PPO iterations)
```

`mario/.venv` is a local venv (Python 3.12) with torch, transformers, peft, fastapi and pytest.
