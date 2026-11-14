# Syntra

Fine-tune models on GPUs that people and data centers share, and earn from the GPUs you share.

- **Providers** (individuals or data centers) run a small agent on a machine; its GPUs join a shared **pool**.
- **Consumers** start a training run. The scheduler picks idle GPUs from the pool, splits the model across them
  (LoRA via BloomBee), and keeps training if a GPU dies: it finds another idle GPU, loads that slice there, and continues.
  Closing a laptop lid takes that GPU out at once and triggers the same replacement.
- **QLoRA**: choose "4-bit" base weights on the Train page and the frozen model takes about a third of the GPU memory, so it
  fits on fewer or smaller GPUs (slower steps). Deployments can also be served in 4-bit.
- Every fine-tuned model is **stored per user**. Come back later, deploy it on N GPUs, try it (greedy or sampling),
  compare it with the base model, and score it on held-out examples.
- Several users work at once on different GPUs; extra jobs wait in a FIFO queue. Payments are simulated (priced per GPU class).

## Layout

| Folder | What it is |
|---|---|
| `backend/` | FastAPI + SQLite: accounts, pool, scheduler, orchestrator, model registry, deployments, events, ledger |
| `agent/` | Provider agent (`python -m syntra_agent`): Mac and NVIDIA discovery, worker processes, command channel, model downloads, lid watcher |
| `frontend/` | React + Vite + TypeScript UI (light theme; Train is the landing page) |
| `provider-app/` | macOS menu bar app for providers: sign in once, click **Share my GPU**; shows status, GPU time used and earnings |
| `bridge/` | SSH tunnels to a machine that cannot reach the platform directly (context3) |
| `scripts/` | `dev.sh`, `demo_stack.py` (fake swarm for the UI), `e2e_real.py`, `e2e_context3.py` |
| `experiments/` | The early BloomBee-on-Apple-GPU experiments the platform is built on |
| `SCOPE.md`, `LATER.md` | What is built, and what is deliberately deferred or still open |

BloomBee (fork branch `mac-mps-ft` of `tanish-malekar/BloomBee`: Apple-GPU support + fault-tolerant LoRA client) does the
distributed work. The backend never imports it: each job runs a **swarm runner** (`backend/app/runner/swarm_runner.py`)
under the Python interpreter named by `SYNTRA_BLOOMBEE_PYTHON`.

## Prerequisites

| Needed for | What |
|---|---|
| everything | Python 3.12 (3.14 untested), Node 20+ |
| backend and demo | `backend/.venv` with the backend requirements (created by `scripts/dev.sh`, or by hand below) |
| real GPU runs | the BloomBee checkout on branch `mac-mps-ft`, a Python 3.12 venv with BloomBee's dependencies, and TinyLlama-1.1B-Chat weights |
| context3 runs | SSH access to context3 (key and jump host in `~/.ssh/config`), a Python env and BloomBee checkout there |

Create the backend venv (once):

```bash
python3.12 -m venv backend/.venv && backend/.venv/bin/pip install -r backend/requirements.txt
```

The agent needs only `httpx` and `websockets` (`agent/requirements.txt`); the backend venv already has them.
The frontend: `cd frontend && npm install`.

Real GPU runs also need (one-time, see `experiments/bloombee-mac/README.md` for details):

```bash
git clone -b mac-mps-ft https://github.com/tanish-malekar/BloomBee ~/BloomBee
python3.12 -m venv ~/bb-venv && ~/bb-venv/bin/pip install -r experiments/bloombee-mac/requirements.txt
# TinyLlama weights into <models dir>/tinyllama:
~/bb-venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('TinyLlama/TinyLlama-1.1B-Chat-v1.0', local_dir='models/tinyllama', allow_patterns=['*.safetensors','*.json','tokenizer.model'])"
```

## 1. See the UI (no GPUs, about a minute)

Two terminals in the repo root:

```bash
PYTHONPATH=agent backend/.venv/bin/python scripts/demo_stack.py     # API on :8000, fake model swarm, 6 simulated GPUs
```
```bash
cd frontend && npm run dev                                          # UI on http://localhost:5173
```

Sign in with a demo account from `demo-logins.txt` (committed; the demo uses fixed throwaway passwords on a temporary database):
`alice@demo.test` trains, deploys and compares; `maya@demo.test` and `acme@demo.test` are providers (Provide GPUs, earnings).
Training runs finish in about a minute and deployments answer instantly. Stop with ctrl-c.

## 2. Run it for real on one Mac (simulated providers)

```bash
# backend: seeds data, starts on :8000 (dev logins are written to backend/.env.dev)
SYNTRA_BLOOMBEE_PYTHON=~/bb-venv/bin/python SYNTRA_BLOOMBEE_DIR=~/BloomBee SYNTRA_MODELS_DIR=/path/to/models scripts/dev.sh
# frontend
cd frontend && npm install && npm run dev
# a provider: sign up in the UI, "Provide GPUs" -> create profile -> create token, then run the command it shows, e.g.
cd agent && python -m syntra_agent --server http://localhost:8000 --token <TOKEN> --bloombee-dir ~/BloomBee \
  --python ~/bb-venv/bin/python --models-dir /path/to/models --simulated --sim-count 3 --sim-memory-gb 6
```

`--simulated` splits the one real GPU into several labelled GPUs so a single Mac can show multi-GPU training and fail-over.
Without it the agent offers the Mac's real GPU (one) or every NVIDIA GPU that `nvidia-smi` shows idle (`--gpus 0,2` limits it).
Real training takes minutes per run (TinyLlama-1.1B, a few seconds per step).

Backend settings (environment): `SYNTRA_DB`, `SYNTRA_DATA_DIR`, `SYNTRA_BLOOMBEE_PYTHON`, `SYNTRA_BLOOMBEE_DIR`,
`SYNTRA_MODELS_DIR`, `SYNTRA_DHT_PORT`, `SYNTRA_COOKIE_SECURE`, `SYNTRA_CORS_ORIGINS`.

## 3. Two machines (this Mac and context3)

context3 cannot reach your Mac directly, so `bridge/tunnel.py` keeps one SSH connection with fixed port ranges forwarded
(backend 8000, DHT 31001, Mac workers 31010-19, context3 workers 31030-39). The easiest way to run it is the test below, which
starts the backend, the tunnel, a Mac agent and a context3 agent, trains across the machines, kills a worker and checks recovery.
It uses only context3 GPUs that `nvidia-smi` shows idle and never touches busy ones.

## 4. Join from any network with one command (relay)

A provider on another network (home Wi-Fi, a lab, anywhere) needs no SSH, no open ports and no shared network. Their agent
connects out to the backend and carries the GPU traffic over that same connection (`--relay`). On the Provide GPUs page, create a
token and copy the command it shows:

```bash
curl -fsSL https://<your-server>/join.sh | bash -s -- --token <TOKEN> --region home
```

The first run installs everything under `~/.syntra` (Python 3.12 venv, BloomBee, the agent; needs Python 3.12 and git), then the
GPU joins the pool. The server must be reachable from the internet: set `SYNTRA_PUBLIC_URL` to its address (for a test, a temporary
tunnel to port 8000) and `SYNTRA_DHT_PORT=31001`. Status: built and unit-tested, not yet proven on two real networks (see `LATER.md`).

Live demo (this Mac + a second MacBook that runs the command above + one idle context3 GPU that takes over when the second
MacBook's lid is closed): `scripts/e2e_live.py` (usage in its docstring).

## 5. Share a Mac with one click (provider app)

```bash
provider-app/build.sh --open      # builds provider-app/dist/SyntraProvider.app and starts it (menu bar icon)
```

Sign in with the server address and your Syntra account, then click **Share my GPU**. The first time, the app installs the
agent under `~/.syntra` (the same steps as `join.sh --install-only`). It keeps the agent running in the background and shows:
- shared / in use / paused / disconnected;
- GPU time used and earnings, today and in total, including an estimate for the job running now.

Pause offers "finish the current job first" or "stop now". Quitting takes the GPUs out of the pool at once. Optional settings:
pause on battery, start at login. Details and edge cases are in `provider-app/README.md` and `ROADMAP.md`.

## Tests

```bash
cd backend && .venv/bin/python -m pytest                      # 92: auth, pool, scheduler, orchestrator, deployments, files (fake swarm, real agents)
cd agent && PYTHONPATH=. ../backend/.venv/bin/python -m pytest tests   # 20: hardware parsing, workers, hanging driver, lid close, agent <-> real backend
cd frontend && npx tsc -b && npx vitest run                    # type check + 6 unit tests
cd provider-app && swift test                                  # provider app: state and usage parsing
```

End-to-end with real models (slow; set `SCRATCH` to a folder containing `models/tinyllama`, `BB_PYTHON` to the BloomBee venv's python):

```bash
# one Mac: backend + agents + BloomBee + TinyLlama. Trains, kills a worker and recovers, deploys the tuned model on 2 GPUs and the
# base on 1, compares, evaluates on held-out tickets, kills a serving worker, stops, checks billing (about 3-5 minutes)
SCRATCH=... BB_PYTHON=... backend/.venv/bin/python scripts/e2e_real.py full
# training only
SCRATCH=... BB_PYTHON=... backend/.venv/bin/python scripts/e2e_real.py
# two machines (needs SSH access to context3; 1.5-5 minutes)
SCRATCH=... BB_PYTHON=... backend/.venv/bin/python scripts/e2e_context3.py
```

What was measured is recorded in `LATER.md` (status and changelog).

## How the pieces talk

`user -> backend (REST + /ws/events)`; `agent -> backend (REST + /ws/agent)` (outbound only); `backend -> agent`: start/stop/kill worker.
Per job: the runner hosts a DHT and the trainer/serving client; workers (one per leased GPU) join that DHT. Behind a tunnel, fixed port
ranges are forwarded (`bridge/tunnel.py`, `--port-range` on the agent, `SYNTRA_DHT_PORT` on the backend).

## Known limits

See `LATER.md`: Qwen3 serving on Apple GPUs, QLoRA and full fine-tuning (deferred), cloud spot instances, real payouts, email
verification, HTTPS and a production database, automatic NAT traversal for providers behind home routers.
