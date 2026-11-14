import { useCallback, useState } from "react";
import { Link } from "react-router-dom";
import { api, type Deployment, type GenerateParams, type Generation } from "../api";
import { Chip, ErrorBox, StatusChip, useToast } from "../components/ui";
import { layersLabel } from "../format";
import { useLive } from "../state";

export const SAMPLE_PROMPTS = [
  "Ticket: my package has not arrived yet. Please fix this asap.\nAnswer:",
  "Ticket: the app crashes when I upload a photo.\nAnswer:",
  "Ticket: I was charged twice for my subscription.\nAnswer:",
];

export function SamplingControls({ value, onChange }: { value: GenerateParams; onChange: (v: GenerateParams) => void }) {
  const set = (patch: Partial<GenerateParams>) => onChange({ ...value, ...patch });
  return (
    <div className="stack">
      <div className="row">
        <label style={{ margin: 0 }}><input type="checkbox" checked={!!value.do_sample} onChange={(e) => set({ do_sample: e.target.checked })} /> Sample instead of picking the most likely word</label>
        <label style={{ margin: 0 }}>Max new tokens <input type="number" min={1} max={256} value={value.max_new_tokens ?? 32} onChange={(e) => set({ max_new_tokens: Number(e.target.value) })} style={{ width: 80, display: "inline-block" }} /></label>
      </div>
      {value.do_sample && (
        <div className="grid cols-3">
          <div><label htmlFor="temp">Temperature</label><input id="temp" type="number" step="0.1" min={0.1} max={5} value={value.temperature ?? 0.8} onChange={(e) => set({ temperature: Number(e.target.value) })} /></div>
          <div><label htmlFor="topp">Top-p</label><input id="topp" type="number" step="0.05" min={0.05} max={1} value={value.top_p ?? 0.95} onChange={(e) => set({ top_p: Number(e.target.value) })} /></div>
          <div><label htmlFor="topk">Top-k</label><input id="topk" type="number" min={1} max={1000} value={value.top_k ?? 50} onChange={(e) => set({ top_k: Number(e.target.value) })} /></div>
          <div><label htmlFor="seed">Seed (repeatable)</label><input id="seed" type="number" value={value.seed ?? ""} onChange={(e) => set({ seed: e.target.value === "" ? undefined : Number(e.target.value) })} placeholder="random" /></div>
        </div>
      )}
      <div className="hint">{value.do_sample ? "Sampling gives different answers on each try; set a seed to repeat one." : "Greedy decoding: the same prompt always gives the same answer."}</div>
    </div>
  );
}

function Playground({ dep }: { dep: Deployment }) {
  const [prompt, setPrompt] = useState(SAMPLE_PROMPTS[0]);
  const [params, setParams] = useState<GenerateParams>({ prompt: "", max_new_tokens: 28, do_sample: false });
  const [out, setOut] = useState<Generation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true); setError(null);
    try { setOut(await api.generate(dep.id, { ...params, prompt })); } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };
  return (
    <div className="stack">
      <div className="field">
        <label htmlFor={`p${dep.id}`}>Prompt</label>
        <textarea id={`p${dep.id}`} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
        <div className="row" style={{ marginTop: 6 }}>{SAMPLE_PROMPTS.map((s, i) => <button key={i} className="small" onClick={() => setPrompt(s)}>Example {i + 1}</button>)}</div>
      </div>
      <SamplingControls value={params} onChange={setParams} />
      <ErrorBox error={error} />
      <div className="row"><button className="primary" disabled={busy || !prompt.trim()} onClick={run}>{busy ? "Generating…" : "Generate"}</button></div>
      {out && (
        <div><div className="answer" aria-live="polite">{out.text}</div><div className="small muted">{out.tokens} tokens in {out.seconds.toFixed(2)} s</div></div>
      )}
    </div>
  );
}

export default function DeploymentsPage() {
  const [deps, setDeps] = useState<Deployment[] | null>(null);
  const [open, setOpen] = useState<number | null>(null);
  const toast = useToast();
  const load = useCallback(() => { api.deployments().then(setDeps).catch(() => undefined); }, []);
  useLive(load, 4000, (e) => e.type.startsWith("deploy.") || e.type === "worker.online" || e.type === "worker.down" || e.type === "worker.replacement_online");

  const stop = async (d: Deployment) => {
    try { await api.stopDeployment(d.id); load(); } catch (e) { toast((e as Error).message); }
  };
  const live = deps?.filter((d) => d.status !== "stopped" && d.status !== "failed") ?? [];
  const ended = deps?.filter((d) => d.status === "stopped" || d.status === "failed") ?? [];
  return (
    <div className="stack">
      <div className="row spread">
        <div><h1>Deployments</h1><p className="muted">Models served on GPUs from the pool. Stop a deployment to give its GPUs back.</p></div>
        <Link to="/models"><button className="primary">Deploy a model</button></Link>
      </div>
      {deps === null ? <div className="empty">Loading…</div> : live.length === 0 ? (
        <div className="card empty"><p>Nothing is deployed right now.</p><Link to="/models">Pick a model to deploy</Link></div>
      ) : (
        <div className="stack">
          {live.map((d) => (
            <div className="card stack" key={d.id}>
              <header>
                <div>
                  <h2 style={{ margin: 0 }}>{d.model_name} <span className="small muted">#{d.id}</span></h2>
                  <span className="small muted">{d.model_kind === "base" ? "Base model" : "Fine-tuned"} · {d.num_gpus || d.job.requested_gpus || "auto"} GPU{(d.num_gpus || d.job.requested_gpus) === 1 ? "" : "s"}{d.weight_bits === 4 ? " · 4-bit weights" : ""}</span>
                </div>
                <div className="row"><StatusChip status={d.status} />{d.status === "running" && <button className="small" onClick={() => setOpen(open === d.id ? null : d.id)}>{open === d.id ? "Hide" : "Try it"}</button>}<button className="small danger" onClick={() => stop(d)}>Stop</button></div>
              </header>
              {d.status === "queued" && <div className="alert warn">Waiting for GPUs: position {d.job.queue_position} in the queue.</div>}
              {d.job.leases.filter((l) => l.active).length > 0 && (
                <div className="row">
                  {d.job.leases.filter((l) => l.active).map((l) => <Chip key={l.id} tone="info" pulse={false}>{l.gpu_name} · {layersLabel(l.layer_start, l.layer_end)}</Chip>)}
                </div>
              )}
              {d.status === "running" && <div className="small muted">Endpoint <span className="mono">POST {d.endpoint}</span></div>}
              {open === d.id && d.status === "running" && <Playground dep={d} />}
            </div>
          ))}
        </div>
      )}
      {ended.length > 0 && (
        <details>
          <summary className="small">Stopped deployments ({ended.length})</summary>
          <div className="card flat"><div className="table-wrap"><table><tbody>
            {ended.map((d) => <tr key={d.id}><td>{d.model_name}</td><td className="muted small">{new Date(d.created_at).toLocaleString()}</td><td><StatusChip status={d.status} /></td></tr>)}
          </tbody></table></div></div>
        </details>
      )}
    </div>
  );
}
