// Typed client for the Syntra backend (cookie session, JSON).
export const API: string = (import.meta.env.VITE_API as string | undefined) ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { ...(init.headers as Record<string, string> | undefined) };
  if (init.body && !(init.body instanceof FormData)) headers["Content-Type"] = "application/json";
  const res = await fetch(API + path, { credentials: "include", ...init, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const get = <T,>(path: string) => request<T>(path);
const post = <T,>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });
const patch = <T,>(path: string, body: unknown) => request<T>(path, { method: "PATCH", body: JSON.stringify(body) });
const del = (path: string) => request<void>(path, { method: "DELETE" });

// ---------------------------------------------------------------------------------------------- types
export interface User { id: number; email: string; display_name: string; is_admin: boolean }
export interface Provider { id: number; name: string; type: "individual" | "enterprise"; trust_tier: string; created_at: string }
export interface AgentTokenInfo { id: number; label: string; created_at: string; last_used_at: string | null; revoked: boolean }
export interface TokenCreated { id: number; label: string; token: string; created_at: string }
export type GpuStatus = "idle" | "leased" | "busy_external" | "offline";
export interface Gpu {
  id: number; provider_id: number; provider_name: string; provider_type: string; trust_tier: string;
  gpu_class: string; price_per_hour: number; name: string; device: "mps" | "cuda"; device_index: number;
  total_memory_mb: number; free_memory_mb: number; region: string; status: GpuStatus;
  last_heartbeat: string | null; reliability: number; simulated: boolean;
}
export interface GpuClass { id: number; name: string; price_per_hour: number; device_hint: string }
export interface BaseModel { id: number; name: string; hf_id: string; family: string; num_layers: number; hidden_size: number; per_layer_memory_mb: number }
export interface Dataset { id: number; name: string; num_rows: number; builtin: boolean; has_heldout: boolean; created_at: string }
export interface Stage {
  gpu_id: number; gpu_name: string; gpu_class: string; device: string; provider: string; region: string;
  layer_start: number; layer_end: number; memory_needed_mb: number; free_memory_mb: number; price_per_hour: number;
}
export interface Plan { fits: boolean; reason: string; idle_gpus: number; stages: Stage[]; hourly_rate: number; machines: number; num_layers: number }
export interface Lease {
  id: number; gpu_id: number; gpu_name: string; gpu_class: string; device: string; layer_start: number; layer_end: number;
  active: boolean; end_reason: string; peer_id: string;
}
export interface StepInfo { step?: number; loss?: number; grad_norm?: number; step_seconds?: number }
export interface Job {
  id: number; kind: string; status: "queued" | "starting" | "running" | "completed" | "failed" | "stopped";
  base_model: string; requested_gpus: number | null; trusted_only: boolean; config: Record<string, unknown>;
  created_at: string; updated_at: string; queue_position: number; leases: Lease[]; last_step: StepInfo;
  result: { final?: StepInfo; model_id?: number | null; error?: string; tail?: string[] };
}
export interface EvalScores {
  status?: "running" | "done" | "failed"; error?: string; n?: number;
  base?: { n: number; valid_json: number; dept_correct: number; urgency_correct: number; mean_seconds: number; examples: string[] };
  finetuned?: { n: number; valid_json: number; dept_correct: number; urgency_correct: number; mean_seconds: number; examples: string[] };
}
export interface Model {
  id: number; name: string; base_model: string; base_model_id: number; job_id: number | null; created_at: string;
  scores: EvalScores; deployed: boolean;
}
export interface Deployment {
  id: number; model_kind: "base" | "finetuned"; model_name: string; base_model: string; finetuned_model_id: number | null;
  status: "queued" | "starting" | "running" | "stopped" | "failed"; num_gpus: number; endpoint: string; created_at: string; job: Job;
}
export interface GenerateParams {
  prompt: string; max_new_tokens?: number; do_sample?: boolean; temperature?: number; top_p?: number; top_k?: number; seed?: number;
}
export interface Generation { deployment_id: number; text: string; tokens: number; seconds: number }
export interface LedgerLine { id: number; created_at: string; gpu_class: string; gpu_seconds: number; rate_per_hour: number; amount: number; job_id: number | null; provider_id: number | null }
export interface Ledger { total: number; gpu_seconds: number; entries: LedgerLine[] }
export interface StoredEvent { id: number | null; type: string; ts: string; job_id: number | null; gpu_id: number | null; payload: Record<string, any> }

export interface JobInput {
  base_model_id: number; dataset_id: number; requested_gpus: number | null; trusted_only: boolean; steps: number;
  batch_size?: number; seq_len?: number; lr?: number; rank?: number; name?: string;
}

// ---------------------------------------------------------------------------------------------- calls
export const api = {
  signup: (email: string, password: string, display_name: string) => post<User>("/auth/signup", { email, password, display_name }),
  login: (email: string, password: string) => post<User>("/auth/login", { email, password }),
  logout: () => post<void>("/auth/logout"),
  me: () => get<User>("/auth/me"),

  gpus: () => get<Gpu[]>("/gpus"),
  myGpus: () => get<Gpu[]>("/gpus/mine"),
  gpuClasses: () => get<GpuClass[]>("/gpu-classes"),
  setClassPrice: (id: number, price_per_hour: number) => patch<GpuClass>(`/gpu-classes/${id}`, { price_per_hour }),
  baseModels: () => get<BaseModel[]>("/base-models"),

  providers: () => get<Provider[]>("/providers"),
  createProvider: (name: string, type: string) => post<Provider>("/providers", { name, type }),
  tokens: (providerId: number) => get<AgentTokenInfo[]>(`/providers/${providerId}/agent-tokens`),
  installInfo: () => get<{ public_url: string; join_command: string }>(`/install/info`),
  createToken: (providerId: number, label: string) => post<TokenCreated>(`/providers/${providerId}/agent-tokens`, { label }),
  revokeToken: (providerId: number, tokenId: number) => del(`/providers/${providerId}/agent-tokens/${tokenId}`),

  datasets: () => get<Dataset[]>("/datasets"),
  uploadDataset: (name: string, file: File) => {
    const form = new FormData();
    form.append("name", name);
    form.append("file", file);
    return request<Dataset>("/datasets", { method: "POST", body: form });
  },
  plan: (baseModelId: number, requestedGpus: number | null, trustedOnly: boolean) => {
    const q = new URLSearchParams({ base_model_id: String(baseModelId), trusted_only: String(trustedOnly) });
    if (requestedGpus) q.set("requested_gpus", String(requestedGpus));
    return get<Plan>(`/plans?${q}`);
  },

  jobs: () => get<Job[]>("/jobs"),
  job: (id: number) => get<Job>(`/jobs/${id}`),
  createJob: (input: JobInput) => post<Job>("/jobs", input),
  cancelJob: (id: number) => post<{ ok: boolean }>(`/jobs/${id}/cancel`),
  killWorker: (id: number, layerStart: number) => post<{ ok: boolean }>(`/jobs/${id}/kill-worker`, { layer_start: layerStart }),
  jobEvents: (id: number) => get<StoredEvent[]>(`/events?job_id=${id}&limit=1000`),

  models: () => get<Model[]>("/models"),
  model: (id: number) => get<Model>(`/models/${id}`),
  deleteModel: (id: number) => del(`/models/${id}`),
  evaluate: (modelId: number, baseDeploymentId: number, tunedDeploymentId: number) =>
    post<{ ok: boolean }>(`/models/${modelId}/evaluate`, { base_deployment_id: baseDeploymentId, tuned_deployment_id: tunedDeploymentId }),

  deployments: () => get<Deployment[]>("/deployments"),
  deployment: (id: number) => get<Deployment>(`/deployments/${id}`),
  deploy: (body: { model_kind: "base" | "finetuned"; base_model_id?: number; finetuned_model_id?: number; requested_gpus: number | null; trusted_only?: boolean }) =>
    post<Deployment>("/deployments", body),
  stopDeployment: (id: number) => post<{ ok: boolean }>(`/deployments/${id}/stop`),
  generate: (id: number, params: GenerateParams) => post<Generation>(`/deployments/${id}/generate`, params),
  compare: (baseId: number, tunedId: number, params: GenerateParams) =>
    post<{ prompt: string; base: Generation; finetuned: Generation; sampled: boolean }>("/compare", {
      ...params, base_deployment_id: baseId, tuned_deployment_id: tunedId,
    }),

  earnings: () => get<Ledger>("/ledger/earnings"),
  costs: () => get<Ledger>("/ledger/costs"),
};
