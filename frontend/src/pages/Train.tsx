import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, type BaseModel, type Dataset, type Model, type Plan, type WeightBits } from "../api";
import { Icon } from "../components/icons";
import { Choices, ErrorBox, PageHeader } from "../components/ui";
import { layersLabel, mb, money } from "../format";
import { useOnEvent } from "../state";

const PALETTE = ["#f2a516", "#e2576a", "#7c6cf0", "#128a5c", "#2f7fd6", "#b8640a"];

/** Float16 or 4-bit base weights on the workers. Used for training (QLoRA) and for deployments. */
export function WeightBitsField({ value, onChange, name, purpose }: { value: WeightBits; onChange: (v: WeightBits) => void; name: string; purpose: "train" | "deploy" }) {
  return (
    <div className="field">
      <span className="label">Base model weights</span>
      <Choices<"16" | "4"> name={name} value={String(value) as "16" | "4"} onChange={(v) => onChange(Number(v) as WeightBits)} options={[
        { value: "16", title: "Float16", detail: "Full precision. Fastest steps." },
        { value: "4", title: purpose === "train" ? "4-bit (QLoRA)" : "4-bit", badge: "⅓ memory",
          detail: purpose === "train" ? "Fits on fewer or smaller GPUs; steps are slower." : "Serves in about a third of the memory; answers can differ slightly." },
      ]} />
    </div>
  );
}

export function PlanView({ plan, requested }: { plan: Plan; requested: number | null }) {
  if (!plan.fits) {
    return (
      <div className="alert warn">
        <strong>{requested ? `${requested} GPUs are not free right now.` : "No idle GPU set can hold this model right now."}</strong>{" "}
        {plan.reason} {plan.idle_gpus} idle GPU{plan.idle_gpus === 1 ? "" : "s"} in the pool. You can still create the job: it waits in the queue.
      </div>
    );
  }
  return (
    <div className="stack sm">
      <div className="layers-bar" aria-hidden style={{ height: 12 }}>
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
      <p className="small muted" style={{ margin: 0 }}>Layers are split in proportion to each GPU's free memory, across {plan.machines} machine{plan.machines === 1 ? "" : "s"}.</p>
    </div>
  );
}

type Method = "sft" | "dpo" | "diloco";
const METHOD_LABEL: Record<Method, string> = { sft: "Supervised (LoRA)", dpo: "Preference (DPO)", diloco: "Full fine-tuning (DiLoCo)" };

export default function TrainPage() {
  const nav = useNavigate();
  const [models, setModels] = useState<BaseModel[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [baseId, setBaseId] = useState<number | null>(null);
  const [datasetId, setDatasetId] = useState<number | null>(null);
  const [trustedOnly, setTrustedOnly] = useState(false);
  const [steps, setSteps] = useState(60);
  const [weightBits, setWeightBits] = useState<WeightBits>(16);
  const [method, setMethod] = useState<Method>("sft");
  const [innerSteps, setInnerSteps] = useState(20);
  const [beta, setBeta] = useState(0.1);
  const [myModels, setMyModels] = useState<Model[]>([]);
  const [source, setSource] = useState<"base" | "mine">("base");
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
  const chooseMethod = (m: Method) => {
    setMethod(m);
    if (m === "diloco") { setWeightBits(16); setSource("base"); setInitModel(null); }
    setDatasetId(datasets.find((d) => (d.format ?? "text") === (m === "dpo" ? "preference" : "text"))?.id ?? null);
  };
  const chooseSource = (s: "base" | "mine") => {
    setSource(s);
    if (s === "base") setInitModel(null);
    else if (myModels[0]) { setInitModel(myModels[0].id); setBaseId(myModels[0].base_model_id); }
  };

  const base = models.find((m) => m.id === baseId) ?? null;
  const from = myModels.find((m) => m.id === initModel) ?? null;
  const dataset = datasets.find((d) => d.id === datasetId) ?? null;
  const refresh = () => {
    if (baseId === null) return;
    api.plan(baseId, null, trustedOnly, weightBits, method).then((p) => { setPlan(p); setError(null); }).catch((e) => { setPlan(null); setError(e.message); });
  };
  useEffect(() => { const t = setTimeout(refresh, 250); return () => clearTimeout(t); }, [baseId, trustedOnly, weightBits, method]); // eslint-disable-line react-hooks/exhaustive-deps
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
      const job = await api.createJob({ base_model_id: baseId, dataset_id: datasetId, requested_gpus: null, trusted_only: trustedOnly, steps, weight_bits: weightBits, name: name.trim() || undefined,
        method, beta, init_model_id: method === "diloco" ? null : initModel,
        ...(method === "diloco" ? { inner_steps: innerSteps, lr: 1e-5 } : {}) });
      nav(`/fine-tuning/${job.id}`);
    } catch (e) { setError((e as Error).message); setBusy(false); }
  };

  return (
    <div className="stack">
      <PageHeader back={() => nav("/fine-tuning")} title="Create fine-tuning job"
        subtitle="Pick a model, a dataset and a method. BloomBee splits the model across GPUs from the pool and keeps training if one drops." />
      <ErrorBox error={error} />
      <div className="create">
        <div className="stack">
          <div className="card">
            <h2><span className="step">1</span>Model</h2>
            <div className="field">
              <span className="label">Choose a source</span>
              <Choices name="source" value={source} onChange={chooseSource} options={[
                { value: "base", title: "From base model", detail: "Fine-tune a supported base model." },
                { value: "mine", title: "From my model", detail: myModels.length ? "Continue from a model you trained." : "You have no trained models yet.",
                  disabled: myModels.length === 0 || method === "diloco" },
              ]} />
            </div>
            {source === "base" ? (
              <div className="field">
                <label htmlFor="base">Base model</label>
                <select id="base" value={baseId ?? ""} onChange={(e) => setBaseId(Number(e.target.value))}>
                  {models.map((m) => <option key={m.id} value={m.id}>{m.name} · {m.num_layers} layers</option>)}
                </select>
              </div>
            ) : (
              <div className="field">
                <label htmlFor="init">Your model</label>
                <select id="init" value={initModel ?? ""} onChange={(e) => { const m = myModels.find((x) => x.id === Number(e.target.value)); if (m) { setInitModel(m.id); setBaseId(m.base_model_id); } }}>
                  {myModels.map((m) => <option key={m.id} value={m.id}>{m.name} ({m.base_model})</option>)}
                </select>
                {method === "dpo" && <div className="hint">Good choice: DPO works best on top of a model you fine-tuned with supervised training first.</div>}
              </div>
            )}
          </div>

          <div className="card">
            <h2><span className="step">2</span>Method</h2>
            <Choices name="method" value={method} onChange={chooseMethod} options={[
              { value: "sft", title: "Supervised", badge: "LoRA", detail: "Learn from example answers. The usual choice." },
              { value: "dpo", title: "Preference (DPO)", detail: "Learn which of two answers is better." },
              { value: "diloco", title: "Full fine-tuning", badge: "DiLoCo", detail: "Train every weight. Replicas each train on their share of the data and sync every N steps." },
            ]} />
            {method === "dpo" && (
              <div className="field" style={{ marginTop: 14 }}>
                <label htmlFor="beta">Beta</label>
                <input id="beta" type="number" step="0.05" min={0.01} max={5} value={beta} onChange={(e) => setBeta(Number(e.target.value))} style={{ maxWidth: 140 }} />
                <div className="hint">Higher keeps the model closer to where it started.</div>
              </div>
            )}
            {method === "diloco" && (
              <div className="field" style={{ marginTop: 14 }}>
                <label htmlFor="inner">Sync every (local steps)</label>
                <input id="inner" type="number" min={1} max={1000} value={innerSteps} onChange={(e) => setInnerSteps(Number(e.target.value))} style={{ maxWidth: 140 }} />
                <div className="hint">Each replica trains on its share of the data. A lost replica is replaced and rejoins at the next sync.</div>
              </div>
            )}
            <p className="hint" style={{ marginTop: 14, marginBottom: 0 }}>Want to train with your own reward (GRPO, PPO)? Use <Link to="/rl">Reinforcement learning</Link>.</p>
          </div>

          <div className="card">
            <h2><span className="step">3</span>Dataset</h2>
            <div className="field">
              <label htmlFor="data">Training file</label>
              <select id="data" value={datasetId ?? ""} onChange={(e) => setDatasetId(Number(e.target.value))}>
                {usable.length === 0 && <option value="">No {wantFormat} dataset yet: upload one below</option>}
                {usable.map((d) => <option key={d.id} value={d.id}>{d.name} · {d.num_rows} examples{d.has_heldout ? " · has held-out set" : ""}</option>)}
              </select>
              <div className="hint">{method === "dpo"
                ? <>A .jsonl file with {"{"}"prompt", "chosen", "rejected"{"}"} on each line.</>
                : <>A .jsonl file with one {"{"}"text": "…"{"}"} per line. The demo set teaches support-ticket routing as JSON.</>}</div>
            </div>
            <div className="field">
              <span className="label">Upload a file</span>
              <div className="row">
                <input value={upName} onChange={(e) => setUpName(e.target.value)} placeholder="Dataset name" style={{ flex: 1, minWidth: 160 }} aria-label="Dataset name" />
                <input type="file" accept=".jsonl,.json,.txt" ref={fileRef} aria-label="Dataset file" style={{ flex: 2, minWidth: 200 }} />
                <button onClick={upload}>{Icon.upload}Upload</button>
              </div>
            </div>
          </div>

          <div className="card">
            <h2><span className="step">4</span>Training</h2>
            <div className="grid cols-2" style={{ gap: 14 }}>
              <div className="field"><label htmlFor="steps">Training steps</label><input id="steps" type="number" min={1} max={500} value={steps} onChange={(e) => setSteps(Number(e.target.value))} /></div>
              <div className="field"><label htmlFor="name">Model name</label><input id="name" value={name} onChange={(e) => setName(e.target.value)} placeholder="Support ticket router" /></div>
            </div>
          </div>

          <div className="card">
            <h2><span className="step">5</span>GPUs</h2>
            <p className="small muted" style={{ marginTop: 0 }}>{method === "diloco"
              ? "BloomBee picks the GPUs for you: up to 4 replicas from the idle GPUs in the pool."
              : "BloomBee picks the GPUs for you: the fewest idle GPUs that fit the model. If one drops, another takes over."}</p>
            <div className="field">
              <label><input type="checkbox" checked={trustedOnly} onChange={(e) => setTrustedOnly(e.target.checked)} /> Trusted GPUs only (verified providers)</label>
              <div className="hint">Providers can see the data their GPUs process. Use this for anything sensitive.</div>
            </div>
            {method !== "diloco" && <WeightBitsField value={weightBits} onChange={setWeightBits} name="train-bits" purpose="train" />}
          </div>

          <div className="card">
            <h2><span className="step">6</span>Plan <span className="small muted" style={{ fontWeight: 500 }}>what you would get right now</span></h2>
            {plan ? <PlanView plan={plan} requested={null} /> : <div className="empty">Planning…</div>}
          </div>
        </div>

        <aside className="summary card">
          <h2>Summary</h2>
          <dl>
            <dt>Model</dt><dd>{from ? from.name : base?.name ?? "–"}</dd>
            {from && <><dt>Base</dt><dd>{from.base_model}</dd></>}
            <dt>Method</dt><dd>{METHOD_LABEL[method]}</dd>
            <dt>Dataset</dt><dd>{dataset ? `${dataset.name}` : "–"}</dd>
            <dt>Steps</dt><dd>{steps}{method === "diloco" ? `, sync every ${innerSteps}` : ""}</dd>
            <dt>Weights</dt><dd>{weightBits === 4 ? "4-bit (QLoRA)" : "Float16"}</dd>
            <dt>GPUs</dt><dd>{plan?.fits ? `${plan.stages.length} on ${plan.machines} machine${plan.machines === 1 ? "" : "s"}` : "Automatic"}</dd>
          </dl>
          <hr className="divider" />
          <div className="small muted">Estimated cost while running</div>
          <div className="price">{plan?.fits ? money(plan.hourly_rate) : "–"}<span className="small muted" style={{ fontWeight: 500 }}> / hour</span></div>
          <button className="primary block" style={{ marginTop: 16 }} disabled={busy || baseId === null || datasetId === null} onClick={start}>
            {busy ? "Creating…" : plan && !plan.fits ? "Create job (wait in queue)" : "Create job"}
          </button>
          <p className="hint" style={{ marginBottom: 0 }}>Billed per GPU-hour while it runs (simulated). The result is saved to My models.</p>
        </aside>
      </div>
    </div>
  );
}
