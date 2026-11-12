# Platform scope (building now)

Deferred ideas live in `LATER.md`. This file is what we are building for the hackathon.

## Two sides of the marketplace
- **Providers**: individuals and enterprise data centers who share idle GPUs.
- **Consumers**: people who start a training run, deploy the result, and compare models.

## Provider flow
1. Sign up on the marketplace page, choose type (individual / enterprise data center), get a one-time agent token.
2. Run the agent on the GPU machine (Mac or Linux/NVIDIA). It registers, reports hardware, sends heartbeats over
   an outbound connection, and starts or stops BloomBee workers when the platform asks.
3. Each GPU lands in the **pool**. Providers see their GPUs, status, jobs served and earnings (simulated payments).

## Pool and scheduler
- Pool registry: GPU id, owner, type, VRAM and free VRAM, region, measured bandwidth/latency, ask price,
  status (idle / leased / offline), reliability, trust tier.
- When a training run starts, the scheduler picks idle GPUs from the pool automatically
  (VRAM, trust tier, latency, reliability, price), computes the layer split, leases them, and starts workers.
- Demo override: the user can choose the number of GPUs (training and deploy, 1..N).
- When a worker dies, the scheduler leases a replacement from the pool and the run continues (measured ~8 s locally).

## Consumer flow
Studio (task + dataset) -> Planner (GPUs, split, cost estimate) -> Live Dashboard (loss, GPU health, GPU down/up
events with new-GPU details, kill-worker button) -> Results -> Deploy Model (N GPUs from the pool) ->
base-vs-fine-tuned comparison.

## Platform services
- Orchestrator: launches/monitors workers, runs training and deploy jobs, replaces dead workers.
- Event stream to the frontend: worker down, replacement up (with GPU info), step, loss, deploy status.
- Simulated payments only: an earnings ledger, no real money.

## Stack
FastAPI + SQLite backend, React frontend, Python agent. BloomBee branch: `mac-mps-ft` (fork).

## Explicitly later (see LATER.md)
Cloud spot instances in the pool, parked LoRA-traffic optimizations, sampling for LoRA generate.
