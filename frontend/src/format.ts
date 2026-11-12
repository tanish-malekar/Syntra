// Small pure helpers (unit-tested in format.test.ts).
import type { StoredEvent } from "./api";

export function mb(value: number): string {
  return value >= 1024 ? `${(value / 1024).toFixed(value % 1024 === 0 ? 0 : 1)} GB` : `${value} MB`;
}

export function money(value: number): string {
  return `$${value.toFixed(value < 1 ? 4 : 2)}`;
}

export function duration(seconds: number): string {
  if (seconds < 90) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  const m = Math.floor(seconds / 60);
  return `${m} min ${Math.round(seconds - m * 60)} s`;
}

export function ago(iso: string | null, now: number = Date.now()): string {
  if (!iso) return "never";
  const s = Math.max(0, (now - new Date(iso).getTime()) / 1000);
  if (s < 5) return "just now";
  if (s < 90) return `${Math.round(s)} s ago`;
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  return new Date(iso).toLocaleString();
}

export function pct(part: number, whole: number): string {
  return whole ? `${Math.round((100 * part) / whole)}%` : "–";
}

export function layersLabel(start: number, end: number): string {
  return end - start === 1 ? `layer ${start}` : `layers ${start}–${end - 1}`;
}

export interface TimelineItem { key: string; ts: string; tone: "info" | "good" | "bad" | "warn"; title: string; detail?: string }

/** Turn raw events into human sentences for the run timeline (step events are shown in the chart instead). */
export function describeEvent(e: StoredEvent): TimelineItem | null {
  const p = e.payload;
  const key = `${e.id ?? e.ts}-${e.type}`;
  switch (e.type) {
    case "job.queued": return { key, ts: e.ts, tone: "warn", title: `Waiting for GPUs (position ${p.position} in the queue)` };
    case "lease.created": return { key, ts: e.ts, tone: "info", title: `Reserved ${p.gpu_name}`, detail: layersLabel(p.layer_start, p.layer_end) };
    case "worker.starting": return { key, ts: e.ts, tone: "info", title: "Starting a worker", detail: layersLabel(p.layer_start, p.layer_end) };
    case "worker.online": return { key, ts: e.ts, tone: "good", title: "Worker online", detail: `${layersLabel(p.layer_start, p.layer_end)} · ${String(p.peer_id).slice(0, 12)}…` };
    case "worker.down": {
      const why = p.reason === "killed_by_user" ? "you stopped it" : p.reason === "heartbeat_timeout" ? "the GPU stopped answering" : "it exited";
      return { key, ts: e.ts, tone: "bad", title: `A worker went down (${why})`, detail: layersLabel(p.layer_start, p.layer_end) };
    }
    case "worker.replacement_waiting":
      return { key, ts: e.ts, tone: "warn", title: "Waiting for a free GPU to take over", detail: `${layersLabel(p.layer_start, p.layer_end)} · it starts as soon as a GPU in the pool is idle` };
    case "worker.replacement_starting":
      return { key, ts: e.ts, tone: "warn", title: `New GPU found: ${p.gpu_name}`,
        detail: `${p.gpu_class} · ${String(p.gpu_device).toUpperCase()} · ${mb(p.gpu_memory_mb)} · taking over ${layersLabel(p.layer_start, p.layer_end)}` };
    case "worker.replacement_online":
      return { key, ts: e.ts, tone: "good", title: "Replacement online", detail: `${duration(p.seconds_since_down)} after the failure · ${layersLabel(p.layer_start, p.layer_end)}` };
    case "train.completed": return { key, ts: e.ts, tone: "good", title: "Training finished, model saved to My models" };
    case "train.failed": return { key, ts: e.ts, tone: "bad", title: "Training failed", detail: String(p.error) };
    case "deploy.merging": return { key, ts: e.ts, tone: "info", title: "Preparing the model for serving (merging the adapter)" };
    case "deploy.online": return { key, ts: e.ts, tone: "good", title: `Deployment online on ${p.num_gpus} GPU${p.num_gpus === 1 ? "" : "s"}` };
    case "deploy.stopped": return { key, ts: e.ts, tone: "info", title: "Deployment stopped", detail: String(p.reason) };
    case "eval.result": return { key, ts: e.ts, tone: "info", title: `Evaluation: ${p.model_label}`, detail: `${p.valid_json}/${p.n} valid answers · ${p.dept_correct} correct departments` };
    case "ledger.credit": return { key, ts: e.ts, tone: "info", title: `Earned ${money(p.amount)}`, detail: `${duration(p.gpu_seconds)} of GPU time` };
    default: return null; // train.step and gpu.* are not timeline entries
  }
}

export interface LossPoint { step: number; loss: number }

export function lossSeries(events: StoredEvent[]): LossPoint[] {
  const byStep = new Map<number, number>();
  for (const e of events) if (e.type === "train.step") byStep.set(e.payload.step, e.payload.loss);
  return [...byStep.entries()].sort((a, b) => a[0] - b[0]).map(([step, loss]) => ({ step, loss }));
}

/** State of the worker that serves a layer range, from the event history (latest event wins). */
export type StageState = "starting" | "online" | "down" | "replacing";
export function stageStates(events: StoredEvent[]): Map<number, StageState> {
  const states = new Map<number, StageState>();
  for (const e of events) {
    const p = e.payload;
    if (p.layer_start === undefined) continue;
    if (e.type === "worker.starting" || e.type === "lease.created") states.set(p.layer_start, states.get(p.layer_start) ?? "starting");
    else if (e.type === "worker.online" || e.type === "worker.replacement_online") states.set(p.layer_start, "online");
    else if (e.type === "worker.down") states.set(p.layer_start, "down");
    else if (e.type === "worker.replacement_starting" || e.type === "worker.replacement_waiting") states.set(p.layer_start, "replacing");
  }
  return states;
}
