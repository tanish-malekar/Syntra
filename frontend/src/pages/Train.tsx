import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type BaseModel, type Dataset, type Model, type Plan, type WeightBits } from "../api";
import { ErrorBox } from "../components/ui";
import { layersLabel, mb, money } from "../format";
import { useOnEvent } from "../state";

const PALETTE = ["#2459d6", "#14804a", "#b4690e", "#8b3fb8", "#c1272d", "#0b7a85"];

/** Float16 or 4-bit base weights on the workers. Used for training (QLoRA) and for deployments. */
export function WeightBitsField({ value, onChange, name, purpose }: { value: WeightBits; onChange: (v: WeightBits) => void; name: string; purpose: "train" | "deploy" }) {
  return (
    <div className="field">
      <span className="label">Base model weights</span>
      <label><input type="radio" name={name} checked={value === 16} onChange={() => onChange(16)} /> Full precision (float16)</label>
      <label><input type="radio" name={name} checked={value === 4} onChange={() => onChange(4)} /> 4-bit{purpose === "train" ? " (QLoRA)" : ""}</label>
      <div className="hint">
        {purpose === "train"
          ? "4-bit keeps the frozen model in about a third of the GPU memory, so it fits on fewer or smaller GPUs. Steps are slower and results can be slightly less accurate. Only GPUs whose agent supports 4-bit are used."
          : "4-bit serves the model in about a third of the GPU memory; answers can differ slightly from float16."}
      </div>
    </div>
  );
}

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
  const [weightBits, setWeightBits] = useState<WeightBits>(16);
  const [method, setMethod] = useState<"sft" | "dpo" | "diloco">("sft");
  const [innerSteps, setInnerSteps] = useState(20);
  const [beta, setBeta] = useState(0.1);
  const [myModels, setMyModels] = useState<Model[]>([]);
  const [initModel, setInitModel] = useState<number | null>(null);
  const [name, setName] = useState("");
  const [plan, setPlan] = useState<Plan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [upName, setUpName] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.baseModels().then((m) => { setModels(m); setBaseId(m[0]?.id ?? null); }).catch((e) => setError(e.message));
    api.datasets().then((d) => { setDatasets(d); setDatasetId(d.find((x) => x.format === "text")?.id ?? null); }).catch((e) => setError(e.message));
    api.models().then(setMyModels).catch(() => undefined);
  }, []);
  const wantFormat = method === "dpo" ? "preference" : "text";
  const usable = datasets.filter((d) => (d.format ?? "text") === wantFormat);
  const startFrom = myModels.filter((m) => m.base_model_id === baseId);
  const chooseMethod = (m: "sft" | "dpo" | "diloco") => {
    setMethod(m);
    if (m === "diloco") { setWeightBits(16); setInitModel(null); }
    setDatasetId(datasets.find((d) => (d.format ?? "text") === (m === "dpo" ? "preference" : "text"))?.id ?? null);
  };

  const base = models.find((m) => m.id === baseId) ?? null;
  const requested = auto ? null : gpuCount;
  const refresh = () => {
    if (baseId === null) return;
    api.plan(baseId, method === "diloco" ? (requested ?? 1) : requested, trustedOnly, weightBits, method).then((p) => { setPlan(p); setError(null); }).catch((e) => { setPlan(null); setError(e.message); });
  };
  useEffect(() => { const t = setTimeout(refresh, 250); return () => clearTimeout(t); }, [baseId, requested, trustedOnly, weightBits, method]); // eslint-disable-line react-hooks/exhaustive-deps
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
      const job = await api.createJob({ base_model_id: baseId, dataset_id: datasetId, requested_gpus: requested, trusted_only: trustedOnly, steps, weight_bits: weightBits, name: name.trim() || undefined,
        method, beta, init_model_id: method === "diloco" ? null : initModel,
        ...(method === "diloco" ? { inner_steps: innerSteps, lr: 1e-5, requested_gpus: requested ?? 1 } : {}) });
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
              {usable.length === 0 && <option value="">No {wantFormat} dataset yet: upload one below</option>}
              {usable.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.num_rows} examples{d.has_heldout ? ", has held-out set" : ""})</option>)}
            </select>
            <div className="hint">{method === "dpo"
              ? <>A .jsonl file with {"{"}"prompt", "chosen", "rejected"{"}"} per line: the model learns to prefer the chosen answer.</>
              : <>A .jsonl file with one {"{"}"text": "…"{"}"} per line. The demo set teaches support-ticket routing as JSON.</>}</div>
          </div>
          <details>
            <summary className="small">Upload your own dataset</summary>
            <div className="row" style={{ marginTop: 8 }}>
              <input value={upName} onChange={(e) => setUpName(e.target.value)} placeholder="Dataset name" style={{ flex: 1 }} aria-label="Dataset name" />
              <input type="file" accept=".jsonl,.json,.txt" ref={fileRef} aria-label="Dataset file" style={{ flex: 2 }} />
              <button onClick={upload}>Upload</button>
            </div>
          </details>
          <div className="field">
            <span className="label">Training method</span>
            <label><input type="radio" name="method" checked={method === "sft"} onChange={() => chooseMethod("sft")} /> Supervised (learn from examples)</label>
            <label><input type="radio" name="method" checked={method === "dpo"} onChange={() => chooseMethod("dpo")} /> Preference tuning, DPO (learn which answer is better)</label>
            <label><input type="radio" name="method" checked={method === "diloco"} onChange={() => chooseMethod("diloco")} /> Full fine-tuning, DiLoCo (every weight; each GPU holds the whole model)</label>
            {method === "diloco" && (
              <div className="row" style={{ marginTop: 6 }}>
                <label htmlFor="inner" style={{ margin: 0 }}>Sync every</label>
                <input id="inner" type="number" min={1} max={1000} value={innerSteps} onChange={(e) => setInnerSteps(Number(e.target.value))} style={{ width: 80 }} />
                <span className="hint" style={{ margin: 0 }}>local steps. "Use exactly N GPUs" = N replicas, each training on its share of the data.</span>
              </div>
            )}
            {method === "dpo" && (
              <div className="row" style={{ marginTop: 6 }}>
                <label htmlFor="beta" style={{ margin: 0 }}>Beta</label>
                <input id="beta" type="number" step="0.05" min={0.01} max={5} value={beta} onChange={(e) => setBeta(Number(e.target.value))} style={{ width: 90 }} />
                <span className="hint" style={{ margin: 0 }}>higher = stay closer to the starting model</span>
              </div>
            )}
            <div className="hint">For RL with your own reward (GRPO, PPO), use the Python SDK: see Developers.</div>
          </div>
          {method !== "diloco" && <div className="field">
            <label htmlFor="init">Start from</label>
            <select id="init" value={initModel ?? ""} onChange={(e) => setInitModel(e.target.value ? Number(e.target.value) : null)}>
              <option value="">The base model</option>
              {startFrom.map((m) => <option key={m.id} value={m.id}>My model: {m.name}</option>)}
            </select>
            {method === "dpo" && <div className="hint">Recommended: start DPO from a model you fine-tuned with supervised training first.</div>}
          </div>}
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
          {method !== "diloco" && <WeightBitsField value={weightBits} onChange={setWeightBits} name="train-bits" purpose="train" />}
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
