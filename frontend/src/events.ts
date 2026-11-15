// Mirror of backend/app/events.py (EventType + required payload keys).
// tests/test_events.py checks that this list and the Python enum agree.

export type EventType =
  | "gpu.registered"
  | "gpu.heartbeat"
  | "gpu.offline"
  | "gpu.busy_external"
  | "job.queued"
  | "lease.created"
  | "worker.starting"
  | "worker.online"
  | "worker.down"
  | "worker.replacement_waiting"
  | "worker.replacement_starting"
  | "worker.replacement_online"
  | "train.step"
  | "train.completed"
  | "train.failed"
  | "deploy.merging"
  | "deploy.online"
  | "deploy.stopped"
  | "eval.result"
  | "ledger.credit"
  | "session.ready"
  | "session.saved"
  | "session.closed";

export type GpuStatus = "idle" | "leased" | "busy_external" | "offline";
export type WorkerDownReason = "exited" | "heartbeat_timeout" | "killed_by_user" | "failed_to_start";

export interface EventPayloads {
  "gpu.registered": { name: string; device: "mps" | "cuda"; gpu_class: string; total_memory_mb: number; provider: string; new?: boolean };
  "gpu.heartbeat": { status: GpuStatus; free_memory_mb: number };
  "gpu.offline": { reason: string };
  "gpu.busy_external": { detail: string };
  "job.queued": { position: number };
  "lease.created": { layer_start: number; layer_end: number; gpu_name: string };
  "worker.starting": { layer_start: number; layer_end: number };
  "worker.online": { layer_start: number; layer_end: number; peer_id: string };
  "worker.down": { reason: WorkerDownReason; layer_start: number; layer_end: number; detail?: string };
  "worker.replacement_waiting": { layer_start: number; layer_end: number };
  "worker.replacement_starting": {
    layer_start: number; layer_end: number; gpu_name: string; gpu_device: "mps" | "cuda"; gpu_class: string; gpu_memory_mb: number;
  };
  "worker.replacement_online": { layer_start: number; layer_end: number; peer_id: string; seconds_since_down: number };
  "train.step": { step: number; loss: number; grad_norm: number; step_seconds: number; reward_margin?: number; reward_accuracy?: number };
  "train.completed": { model_id: number };
  "train.failed": { error: string };
  "deploy.merging": Record<string, never>;
  "deploy.online": { endpoint: string; num_gpus: number };
  "deploy.stopped": { reason: string };
  "eval.result": { model_label: string; n: number; valid_json: number; dept_correct: number; urgency_correct: number };
  "ledger.credit": { provider_id: number; amount: number; gpu_seconds: number };
  "session.ready": { trainable_parameters: number };
  "session.saved": { model_id: number; name: string; step?: number };
  "session.closed": { reason: string };
}

export interface SyntraEvent<T extends EventType = EventType> {
  id: number | null; // null for broadcast-only events (gpu.heartbeat)
  type: T;
  ts: string; // ISO 8601, UTC
  job_id: number | null;
  gpu_id: number | null;
  payload: EventPayloads[T];
}

/** First message on /ws/events, then events as they happen; "pong" answers a "ping" text frame. */
export type SocketMessage = { type: "hello"; role: "user" | "agent" } | { type: "pong" } | SyntraEvent;

export const POOL_LEVEL_PREFIX = "gpu."; // gpu.* events go to every signed-in user; others only to the owner
