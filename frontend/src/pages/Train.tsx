import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type BaseModel, type Dataset, type Plan } from "../api";
import { ErrorBox } from "../components/ui";
import { layersLabel, mb, money } from "../format";
import { useOnEvent } from "../state";

const PALETTE = ["#2459d6", "#14804a", "#b4690e", "#8b3fb8", "#c1272d", "#0b7a85"];

export function PlanView({ plan, requested }: { plan: Plan; requested: number | null }) {
  if (!plan.fits) {
    return (
      <div className="alert warn">
        <strong>{requested ? `${requested} GPUs are not free right now.` : "No idle GPU set can hold this model right now."}</strong>{" "}
        {plan.reason} {plan.idle_gpus} idle GPU{plan.idle_gpus === 1 ? "" : "s"} in the pool.
      </div>
    );
  }
  return (
    <div className="stack">
      <div className="layers-bar" aria-hidden>
        {plan.stages.map((s, i) => <i key={s.gpu_id} style={{ width: `${((s.layer_end - s.layer_start) / plan.num_layers) * 100}%`, background: PALETTE[i % PALETTE.length] }} />)}
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Stage</th><th>GPU</th><th>Layers</th><th>Memory needed / free</th><th>Price</th></tr></thead>
          <tbody>
            {plan.stages.map((s, i) => (
              <tr key={s.gpu_id}>
                <td><span className="chip plain" style={{ background: PALETTE[i % PALETTE.length], color: "#fff" }}>{i + 1}</span></td>
                <td><strong>{s.gpu_name}</strong><div className="small muted">{s.provider} · {s.gpu_class} · {s.device.toUpperCase()}</div></td>
                <td>{layersLabel(s.layer_start, s.layer_end)}<div className="small muted">{s.layer_end - s.layer_start} of {plan.num_layers}</div></td>
                <td className="mono">{mb(s.memory_needed_mb)} / {mb(s.free_memory_mb)}</td>
                <td className="mono">{money(s.price_per_hour)}/h</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        Estimated cost about <strong>{money(plan.hourly_rate)}</strong> per hour while these GPUs are in use, across {plan.machines} machine{plan.machines === 1 ? "" : "s"}.
        Layers are split in proportion to each GPU's free memory.
      </p>
    </div>
  );
}

export default function TrainPage() {
  const nav = useNavigate();
  const [models, setModels] = useState<BaseModel[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [baseId, setBaseId] = useState<number | null>(null);
  const [datasetId, setDatasetId] = useState<number | null>(null);
  const [auto, setAuto] = useState(true);
  const [gpuCount, setGpuCount] = useState(2);
  const [trustedOnly, setTrustedOnly] = useState(false);
  const [steps, setSteps] = useState(60);
  const [name, setName] = useState("");
  const [plan, setPlan] = useState<Plan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [upName, setUpName] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.baseModels().then((m) => { setModels(m); setBaseId(m[0]?.id ?? null); }).catch((e) => setError(e.message));
    api.datasets().then((d) => { setDatasets(d); setDatasetId(d[0]?.id ?? null); }).catch((e) => setError(e.message));
  }, []);

  const base = models.find((m) => m.id === baseId) ?? null;
  const requested = auto ? null : gpuCount;
  const refresh = () => {
    if (baseId === null) return;
    api.plan(baseId, requested, trustedOnly).then((p) => { setPlan(p); setError(null); }).catch((e) => { setPlan(null); setError(e.message); });
  };
  useEffect(() => { const t = setTimeout(refresh, 250); return () => clearTimeout(t); }, [baseId, requested, trustedOnly]); // eslint-disable-line react-hooks/exhaustive-deps
  useOnEvent((e) => { if (e.type === "gpu.registered" || e.type === "gpu.offline" || e.type === "gpu.busy_external" || e.type === "lease.created") refresh(); });

  const upload = async () => {
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    setError(null);
    try {
      const d = await api.uploadDataset(upName.trim() || file.name, file);
      setDatasets((cur) => [...cur, d]);
      setDatasetId(d.id);
      setUpName("");
      if (fileRef.current) fileRef.current.value = "";
    } catch (e) { setError((e as Error).message); }
  };

  const start = async () => {
    if (baseId === null || datasetId === null) return;
    setBusy(true); setError(null);
    try {
      const job = await api.createJob({ base_model_id: baseId, dataset_id: datasetId, requested_gpus: requested, trusted_only: trustedOnly, steps, name: name.trim() || undefined });
      nav(`/runs/${job.id}`);
    } catch (e) { setError((e as Error).message); setBusy(false); }
  };

  return (
    <div className="stack">
      <div>
        <h1>Train a model</h1>
        <p className="muted">Fine-tune with LoRA on GPUs from the pool. The model is split across them, and training continues automatically if one fails.</p>
      </div>
      <ErrorBox error={error} />
      <div className="grid cols-2">
        <div className="card stack">
          <h2>1. Task</h2>
          <div className="field">
            <label htmlFor="base">Base model</label>
            <select id="base" value={baseId ?? ""} onChange={(e) => setBaseId(Number(e.target.value))}>
              {models.map((m) => <option key={m.id} value={m.id}>{m.name} ({m.num_layers} layers)</option>)}
            </select>
          </div>
          <div className="field">
            <label htmlFor="data">Training data</label>
            <select id="data" value={datasetId ?? ""} onChange={(e) => setDatasetId(Number(e.target.value))}>
              {datasets.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.num_rows} examples{d.has_heldout ? ", has held-out set" : ""})</option>)}
            </select>
            <div className="hint">A .jsonl file with one {"{"}"text": "…"{"}"} per line. The demo set teaches support-ticket routing as JSON.</div>
          </div>
          <details>
            <summary className="small">Upload your own dataset</summary>
            <div className="row" style={{ marginTop: 8 }}>
              <input value={upName} onChange={(e) => setUpName(e.target.value)} placeholder="Dataset name" style={{ flex: 1 }} aria-label="Dataset name" />
              <input type="file" accept=".jsonl,.json,.txt" ref={fileRef} aria-label="Dataset file" style={{ flex: 2 }} />
              <button onClick={upload}>Upload</button>
            </div>
          </details>
          <div className="grid cols-2">
            <div className="field"><label htmlFor="steps">Training steps</label><input id="steps" type="number" min={1} max={500} value={steps} onChange={(e) => setSteps(Number(e.target.value))} /></div>
            <div className="field"><label htmlFor="name">Name your model</label><input id="name" value={name} onChange={(e) => setName(e.target.value)} placeholder="Support ticket router" /></div>
          </div>
        </div>
        <div className="card stack">
          <h2>2. GPUs</h2>
          <div className="field">
            <label><input type="radio" checked={auto} onChange={() => setAuto(true)} /> Choose for me (fewest GPUs that fit)</label>
            <label><input type="radio" checked={!auto} onChange={() => setAuto(false)} /> Use exactly <input type="number" min={1} max={base?.num_layers ?? 22} value={gpuCount} disabled={auto} onChange={(e) => setGpuCount(Math.max(1, Number(e.target.value)))} style={{ width: 70, display: "inline-block" }} aria-label="Number of GPUs" /> GPUs</label>
            <div className="hint">A small model fits on one GPU, but splitting it across several shows how larger models are trained and lets you watch fail-over.</div>
          </div>
          <label><input type="checkbox" checked={trustedOnly} onChange={(e) => setTrustedOnly(e.target.checked)} /> Trusted GPUs only (verified providers)</label>
          <div className="hint">Providers can see the data their GPUs process. Use this for anything sensitive.</div>
        </div>
      </div>
      <div className="card stack">
        <h2>3. Plan <span className="small muted">(what you would get right now)</span></h2>
        {plan ? <PlanView plan={plan} requested={requested} /> : <div className="empty">Planning…</div>}
        <div className="row">
          <button className="primary" disabled={busy || baseId === null || datasetId === null} onClick={start}>{plan && !plan.fits ? "Start (wait in queue)" : "Start training"}</button>
        </div>
      </div>
    </div>
  );
}
