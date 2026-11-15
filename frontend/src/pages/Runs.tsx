import { useCallback, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Icon } from "../components/icons";
import { api, type Job, type StoredEvent } from "../api";
import { Chip, EmptyState, ErrorBox, LossChart, PageHeader, SearchBox, StatusChip, Timeline, useToast } from "../components/ui";
import { duration, layersLabel, lossSeries, stageStates, type StageState } from "../format";
import { useLive, useOnEvent } from "../state";

const METHOD_NAME: Record<string, string> = { sft: "Supervised", dpo: "DPO", diloco: "Full (DiLoCo)" };

export function RunsPage() {
  const nav = useNavigate();
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [q, setQ] = useState("");
  const [status, setStatus] = useState("all");
  const [method, setMethod] = useState("all");
  useLive(() => { api.jobs().then(setJobs).catch(() => undefined); }, 4000, (e) => e.type === "train.completed" || e.type === "train.failed" || e.type === "job.queued" || e.type === "lease.created");
  const shown = (jobs ?? []).filter((j) => {
    const text = `${j.id} ${String(j.config.model_name ?? "")} ${j.base_model}`.toLowerCase();
    return (!q || text.includes(q.toLowerCase())) && (status === "all" || j.status === status)
      && (method === "all" || String(j.config.method ?? "sft") === method);
  });
  return (
    <div className="stack">
      <PageHeader icon={Icon.finetune} title="Fine-tuning" subtitle="View and manage your fine-tuning jobs: supervised, preference (DPO) and full fine-tuning."
        actions={<button className="primary" onClick={() => nav("/fine-tuning/new")}>{Icon.plus}New fine-tuning job</button>} />
      <div className="filters">
        <SearchBox value={q} onChange={setQ} placeholder="Search jobs" />
        <select value={status} onChange={(e) => setStatus(e.target.value)} aria-label="Status">
          <option value="all">All statuses</option>
          {["queued", "starting", "running", "completed", "failed", "stopped"].map((s) => <option key={s} value={s}>{s[0].toUpperCase() + s.slice(1)}</option>)}
        </select>
        <select value={method} onChange={(e) => setMethod(e.target.value)} aria-label="Method">
          <option value="all">All methods</option>
          {Object.entries(METHOD_NAME).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
      </div>
      <div className="card table-card">
        {jobs === null ? <div className="empty">Loading…</div> : jobs.length === 0 ? (
          <EmptyState title="No fine-tuning jobs" action={<button className="honey" onClick={() => nav("/fine-tuning/new")}>Get started</button>}>
            You have not run any fine-tuning jobs yet. Pick a model and a dataset, and the pool does the rest.
          </EmptyState>
        ) : shown.length === 0 ? <div className="empty">No job matches these filters.</div> : (
          <div className="table-wrap">
            <table>
              <thead><tr><th>Job</th><th>Base model</th><th>Method</th><th>GPUs</th><th>Progress</th><th>Status</th><th>Started</th></tr></thead>
              <tbody>
                {shown.map((j) => {
                  const active = j.leases.filter((l) => l.active).length || j.leases.length;
                  const steps = Number(j.config.steps ?? 0);
                  const step = j.last_step.step ?? 0;
                  return (
                    <tr key={j.id} className="link" onClick={() => nav(`/fine-tuning/${j.id}`)}>
                      <td><Link to={`/fine-tuning/${j.id}`} onClick={(e) => e.stopPropagation()}><strong>#{j.id} {String(j.config.model_name ?? "")}</strong></Link></td>
                      <td>{j.base_model}</td>
                      <td><span className="row tight">{METHOD_NAME[String(j.config.method ?? "sft")] ?? "Supervised"}{Number(j.config.weight_bits) === 4 && <Chip tone="plain">4-bit</Chip>}</span></td>
                      <td>{active || "–"}</td>
                      <td className="mono">{j.status === "queued" ? `queue #${j.queue_position}` : steps ? `${step} / ${steps}${j.last_step.loss !== undefined ? ` · loss ${j.last_step.loss.toFixed(3)}` : ""}` : "–"}</td>
                      <td><StatusChip status={j.status} /></td>
                      <td className="small muted">{new Date(j.created_at).toLocaleString()}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

const STATE_LABEL: Record<StageState, string> = { starting: "Starting", online: "Online", down: "Down", replacing: "Finding a new GPU" };
const STATE_TONE: Record<StageState, "info" | "good" | "bad" | "warn"> = { starting: "info", online: "good", down: "bad", replacing: "warn" };

export default function RunPage() {
  const id = Number(useParams().id);
  const nav = useNavigate();
  const toast = useToast();
  const [job, setJob] = useState<Job | null>(null);
  const [events, setEvents] = useState<StoredEvent[]>([]);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    Promise.all([api.job(id), api.jobEvents(id)]).then(([j, ev]) => { setJob(j); setEvents(ev); setError(null); }).catch((e) => setError(e.message));
  }, [id]);
  useLive(load, 5000, (e) => e.job_id === id && e.type !== "train.step");
  useOnEvent((e) => {
    if (e.job_id !== id || e.type !== "train.step") return;
    setEvents((cur) => (cur.some((x) => x.type === "train.step" && x.payload.step === e.payload.step) ? cur : [...cur, e]));
  });

  if (error && !job) return <ErrorBox error={error} />;
  if (!job) return <div className="empty">Loading…</div>;

  const points = lossSeries(events);
  const last = points[points.length - 1];
  const states = stageStates(events);
  const active = job.leases.filter((l) => l.active);
  const past = job.leases.filter((l) => !l.active);
  const totalSteps = Number(job.config.steps ?? 0);
  const done = job.status === "completed" || job.status === "failed" || job.status === "stopped";
  const stepTimes = events.filter((e) => e.type === "train.step").map((e) => e.payload.step_seconds as number);
  const avgStep = stepTimes.length ? stepTimes.reduce((a, b) => a + b, 0) / stepTimes.length : 0;
  const modelId = job.result.model_id ?? null;
  const totalLayers = Math.max(1, ...job.leases.map((l) => l.layer_end));

  const kill = async (layerStart: number) => {
    try { await api.killWorker(id, layerStart); toast("Worker stopped. Watch the timeline: the platform will find another GPU."); } catch (e) { toast((e as Error).message); }
  };
  const cancel = async () => {
    try { await api.cancelJob(id); load(); } catch (e) { toast((e as Error).message); }
  };

  return (
    <div className="stack">
      <PageHeader back={() => nav("/fine-tuning")}
        title={<span className="row" style={{ gap: 10 }}>Job #{job.id} · {String(job.config.model_name ?? job.base_model)}<StatusChip status={job.status} />{Number(job.config.weight_bits) === 4 && <Chip tone="info">QLoRA 4-bit</Chip>}</span>}
        subtitle={`${job.base_model} · ${job.requested_gpus ? `${job.requested_gpus} GPUs requested` : "GPUs chosen automatically"}${job.trusted_only ? " · trusted GPUs only" : ""}`}
        actions={<>
          {modelId && <button className="primary" onClick={() => nav("/models")}>Open the saved model</button>}
          {!done && <button className="danger" onClick={cancel}>Cancel job</button>}
        </>} />
      <ErrorBox error={error} />
      {job.status === "queued" && <div className="alert warn">Waiting for GPUs: position {job.queue_position} in the queue. The run starts by itself when enough GPUs are idle.</div>}
      {job.status === "failed" && <div className="alert error"><strong>Training failed.</strong> {job.result.error || (job.result.tail ?? []).slice(-2).join(" ")}</div>}
      {job.status === "completed" && <div className="alert good">Training finished and the model is saved. You can deploy it any time from My models.</div>}

      <div className="stats">
        <div className="stat"><div className="v">{last ? last.step : 0}{totalSteps ? <span className="muted small"> / {totalSteps}</span> : null}</div><div className="k">Steps</div></div>
        <div className="stat"><div className="v">{last ? last.loss.toFixed(3) : "–"}</div><div className="k">Loss {points.length > 1 ? `(from ${points[0].loss.toFixed(2)})` : ""}</div></div>
        <div className="stat"><div className="v">{avgStep ? duration(avgStep) : "–"}</div><div className="k">Average step time</div></div>
        <div className="stat"><div className="v">{active.length}</div><div className="k">GPUs working</div></div>
      </div>

      <div className="grid cols-2">
        <div className="card"><header><h2 style={{ margin: 0 }}>Training loss</h2></header><LossChart points={points} /></div>
        <div className="card"><header><h2 style={{ margin: 0 }}>What is happening</h2></header><Timeline events={events} /></div>
      </div>

      <div className="card">
        <header><h2 style={{ margin: 0 }}>Workers</h2><span className="small muted">Each GPU holds a slice of the model</span></header>
        {active.length === 0 ? <div className="empty">No GPU is assigned right now.</div> : (
          <div className="grid cols-3">
            {active.map((l) => {
              const state = states.get(l.layer_start) ?? "starting";
              return (
                <div key={l.id} className={`stage ${state}`}>
                  <div className="row spread"><strong>{l.gpu_name}</strong><Chip tone={STATE_TONE[state]} pulse={state !== "online"}>{STATE_LABEL[state]}</Chip></div>
                  <div className="small muted">{l.gpu_class} · {l.device.toUpperCase()}</div>
                  <div className="layers-bar" aria-hidden><i style={{ width: `${((l.layer_end - l.layer_start) / totalLayers) * 100}%`, marginLeft: `${(l.layer_start / totalLayers) * 100}%`, background: "var(--accent)" }} /></div>
                  <div className="small">{layersLabel(l.layer_start, l.layer_end)}</div>
                  {l.peer_id && <div className="small mono muted">{l.peer_id.slice(0, 14)}…</div>}
                  {job.status === "running" && state === "online" && <button className="small danger" style={{ marginTop: 8 }} onClick={() => kill(l.layer_start)}>Kill this worker</button>}
                </div>
              );
            })}
          </div>
        )}
        {past.length > 0 && (
          <details style={{ marginTop: 12 }}>
            <summary className="small">Earlier workers ({past.length})</summary>
            <div className="table-wrap"><table><tbody>
              {past.map((l) => <tr key={l.id}><td>{l.gpu_name}</td><td>{layersLabel(l.layer_start, l.layer_end)}</td><td className="muted">{l.end_reason.replace(/_/g, " ")}</td></tr>)}
            </tbody></table></div>
          </details>
        )}
        {job.status === "running" && <p className="hint">The kill button simulates a GPU failing. The platform finds another idle GPU, loads that slice of the model there, and training continues.</p>}
      </div>
    </div>
  );
}
