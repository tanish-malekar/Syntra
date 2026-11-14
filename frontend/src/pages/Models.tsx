import { useCallback, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, type BaseModel, type Model, type WeightBits } from "../api";
import { Chip, ErrorBox, useToast } from "../components/ui";
import { pct } from "../format";
import { useLive } from "../state";
import { WeightBitsField } from "./Train";

export function ScoreBars({ scores }: { scores: Model["scores"] }) {
  if (scores.status === "running") return <div className="alert info">Evaluating on the held-out examples…</div>;
  if (scores.status === "failed") return <div className="alert error">Evaluation failed: {scores.error}</div>;
  if (scores.status !== "done" || !scores.base || !scores.finetuned) return <p className="small muted">Not evaluated yet. Open Compare to score it against the base model.</p>;
  const rows: [string, number, number][] = [
    ["Answers in the right format", scores.base.valid_json, scores.finetuned.valid_json],
    ["Correct department", scores.base.dept_correct, scores.finetuned.dept_correct],
    ["Correct urgency", scores.base.urgency_correct, scores.finetuned.urgency_correct],
  ];
  const n = scores.n ?? 0;
  return (
    <div className="bars">
      <div className="small muted">Held-out examples: {n}</div>
      {rows.map(([label, b, t]) => (
        <div key={label}>
          <div className="row spread small"><span>{label}</span><span><span className="muted">base {pct(b, n)}</span> · <strong>fine-tuned {pct(t, n)}</strong></span></div>
          <div className="bar"><i style={{ width: `${n ? (100 * b) / n : 0}%` }} /></div>
          <div className="bar tuned"><i style={{ width: `${n ? (100 * t) / n : 0}%` }} /></div>
        </div>
      ))}
    </div>
  );
}

function DeployForm({ model, onDone }: { model: Model; onDone: () => void }) {
  const nav = useNavigate();
  const [auto, setAuto] = useState(true);
  const [count, setCount] = useState(2);
  const [weightBits, setWeightBits] = useState<WeightBits>(model.weight_bits);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const go = async () => {
    setBusy(true); setError(null);
    try {
      await api.deploy({ model_kind: "finetuned", finetuned_model_id: model.id, requested_gpus: auto ? null : count, weight_bits: weightBits });
      onDone();
      nav("/deployments");
    } catch (e) { setError((e as Error).message); setBusy(false); }
  };
  return (
    <div className="stack" style={{ marginTop: 12 }}>
      <ErrorBox error={error} />
      <div className="row">
        <label style={{ margin: 0 }}><input type="radio" checked={auto} onChange={() => setAuto(true)} /> Automatic</label>
        <label style={{ margin: 0 }}><input type="radio" checked={!auto} onChange={() => setAuto(false)} /> Exactly <input type="number" min={1} max={22} value={count} disabled={auto} onChange={(e) => setCount(Math.max(1, Number(e.target.value)))} style={{ width: 64, display: "inline-block" }} aria-label="Number of GPUs" /> GPUs</label>
        <button className="primary" disabled={busy} onClick={go}>Deploy</button>
      </div>
      <WeightBitsField value={weightBits} onChange={setWeightBits} name={`deploy-bits-${model.id}`} purpose="deploy" />
    </div>
  );
}

export default function ModelsPage() {
  const [models, setModels] = useState<Model[] | null>(null);
  const [bases, setBases] = useState<BaseModel[]>([]);
  const [open, setOpen] = useState<number | null>(null);
  const toast = useToast();
  const load = useCallback(() => { api.models().then(setModels).catch(() => undefined); api.baseModels().then(setBases).catch(() => undefined); }, []);
  useLive(load, 5000, (e) => e.type === "train.completed" || e.type === "eval.result" || e.type === "deploy.online" || e.type === "deploy.stopped");

  const remove = async (m: Model) => {
    if (!window.confirm(`Delete "${m.name}"? Its trained weights will be removed.`)) return;
    try { await api.deleteModel(m.id); load(); } catch (e) { toast((e as Error).message); }
  };
  return (
    <div className="stack">
      <div><h1>My models</h1><p className="muted">Every model you have fine-tuned is stored here. Come back any time to deploy it, try it, and compare it with the base model.</p></div>
      {models === null ? <div className="empty">Loading…</div> : models.length === 0 ? (
        <div className="card empty"><p>No fine-tuned models yet.</p><Link to="/train">Train your first model</Link></div>
      ) : (
        <div className="grid cols-2">
          {models.map((m) => (
            <div className="card stack" key={m.id}>
              <header>
                <div><h2 style={{ margin: 0 }}>{m.name}</h2><span className="small muted">{m.base_model} · {new Date(m.created_at).toLocaleString()}</span></div>
                <div className="row">
                  {m.weight_bits === 4 && <Chip tone="info">QLoRA 4-bit</Chip>}
                  {m.deployed ? <Chip tone="good" pulse>Deployed</Chip> : <Chip tone="plain">Not deployed</Chip>}
                </div>
              </header>
              <ScoreBars scores={m.scores} />
              <div className="row">
                <button className="primary" onClick={() => setOpen(open === m.id ? null : m.id)}>Deploy…</button>
                <Link to={`/compare?model=${m.id}`}><button>Compare with base</button></Link>
                {m.job_id && <Link to={`/runs/${m.job_id}`}><button>Training run</button></Link>}
                <button className="danger" onClick={() => remove(m)} disabled={m.deployed}>Delete</button>
              </div>
              {open === m.id && <DeployForm model={m} onDone={() => setOpen(null)} />}
            </div>
          ))}
        </div>
      )}
      {bases.length > 0 && <p className="small muted">Base models on offer: {bases.map((b) => b.name).join(", ")}.</p>}
    </div>
  );
}
