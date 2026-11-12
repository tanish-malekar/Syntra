# Syntra

Fine-tune models on GPUs that people and data centers share, and earn from the GPUs you share.

- **Providers** (individuals or data centers) run a small agent on a machine; its GPUs join a shared **pool**.
- **Consumers** start a training run. The scheduler picks idle GPUs from the pool, splits the model across them
  (LoRA via BloomBee), and keeps training if a GPU dies: it finds another idle GPU, loads that slice there, and continues.
- Every fine-tuned model is **stored per user**. Come back later, deploy it on N GPUs, try it (greedy or sampling),
  compare it with the base model, and score it on held-out examples.
- Several users work at once on different GPUs; extra jobs wait in a FIFO queue. Payments are simulated (priced per GPU class).

## Layout

| Folder | What it is |
|---|---|
| `backend/` | FastAPI + SQLite: accounts, pool, scheduler, orchestrator, model registry, deployments, events, ledger |
| `agent/` | Provider agent (`python -m syntra_agent`): Mac and NVIDIA discovery, worker processes, command channel, model downloads |
| `frontend/` | React + Vite + TypeScript UI |
| `bridge/` | SSH tunnels to a machine that cannot reach the platform directly (context3) |
| `scripts/` | `dev.sh`, `demo_stack.py` (fake swarm for UI work), `e2e_real.py`, `e2e_context3.py` |
| `experiments/` | The early BloomBee-on-Apple-GPU experiments the platform is built on |
| `SCOPE.md`, `LATER.md` | What is built, and what is deliberately deferred or still open |

BloomBee (fork branch `mac-mps-ft`: Apple-GPU support + fault-tolerant LoRA client) does the distributed work. The backend never imports
it: each job runs a **swarm runner** (`backend/app/runner/swarm_runner.py`) under `SYNTRA_BLOOMBEE_PYTHON`.

## Run it (one Mac, simulated providers)

```bash
# backend (creates .venv, seeds data, starts on :8000)
SYNTRA_BLOOMBEE_PYTHON=/path/to/python-with-bloombee SYNTRA_BLOOMBEE_DIR=~/BloomBee SYNTRA_MODELS_DIR=/path/to/models scripts/dev.sh
# frontend
cd frontend && npm install && npm run dev          # http://localhost:5173
# a provider: sign up in the UI, "Provide GPUs" -> create profile -> create token, then run the command it shows, e.g.
cd agent && python -m syntra_agent --server http://localhost:8000 --token <TOKEN> --bloombee-dir ~/BloomBee --python /path/to/python-with-bloombee \
  --models-dir /path/to/models --simulated --sim-count 3 --sim-memory-gb 6
```
Models: put TinyLlama-1.1B-Chat in `<models dir>/tinyllama` (see `experiments/bloombee-mac/README.md`). Dev logins are generated into `backend/.env.dev`.
Python 3.12 is the tested version.

UI without GPUs: `PYTHONPATH=agent backend/.venv/bin/python scripts/demo_stack.py` starts the API on :8000 with a fake swarm (logins in `demo-logins.txt`).

## Tests

```bash
cd backend && .venv/bin/python -m pytest                      # 92: auth, pool, scheduler, orchestrator, deployments, files (fake swarm, real agents)
cd agent && PYTHONPATH=. ../backend/.venv/bin/python -m pytest tests   # 17: hardware parsing, workers, agent <-> real backend
cd frontend && npx tsc -b && npx vitest run                    # type check + unit tests
# real thing on one Mac: backend + agent + BloomBee + TinyLlama (train, kill+recover, deploy, compare, evaluate)
SCRATCH=<dir with models/tinyllama> BB_PYTHON=<python with BloomBee> backend/.venv/bin/python scripts/e2e_real.py full
# two machines (needs SSH access to context3; uses only GPUs nvidia-smi shows idle)
SCRATCH=... BB_PYTHON=... backend/.venv/bin/python scripts/e2e_context3.py
```

## How the pieces talk

`user -> backend (REST + /ws/events)`; `agent -> backend (REST + /ws/agent)` (outbound only); `backend -> agent`: start/stop/kill worker.
Per job: the runner hosts a DHT and the trainer/serving client; workers (one per leased GPU) join that DHT. Behind a tunnel, fixed port
ranges are forwarded (`bridge/tunnel.py`, `--port-range` on the agent, `SYNTRA_DHT_PORT` on the backend).
