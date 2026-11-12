import { useCallback, useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, type Deployment, type GenerateParams, type Model } from "../api";
import { ErrorBox, StatusChip, useToast } from "../components/ui";
import { duration } from "../format";
import { useLive } from "../state";
import { SAMPLE_PROMPTS, SamplingControls } from "./Deployments";
import { ScoreBars } from "./Models";

type Answers = { base: string; tuned: string; baseSeconds: number; tunedSeconds: number; sampled: boolean } | null;

export default function ComparePage() {
  const [params, setParams] = useSearchParams();
  const toast = useToast();
  const [models, setModels] = useState<Model[]>([]);
  const [deps, setDeps] = useState<Deployment[]>([]);
  const [tunedGpus, setTunedGpus] = useState(2);
  const [prompt, setPrompt] = useState(SAMPLE_PROMPTS[0]);
  const [gen, setGen] = useState<GenerateParams>({ prompt: "", max_new_tokens: 28, do_sample: false });
  const [answers, setAnswers] = useState<Answers>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const modelId = Number(params.get("model")) || models[0]?.id || 0;
  const model = models.find((m) => m.id === modelId) ?? null;
  const load = useCallback(() => {
    api.models().then(setModels).catch(() => undefined);
    api.deployments().then(setDeps).catch(() => undefined);
  }, []);
  useLive(load, 3000, (e) => e.type.startsWith("deploy.") || e.type === "eval.result" || e.type === "worker.online");
  useEffect(() => { if (!params.get("model") && models[0]) setParams({ model: String(models[0].id) }, { replace: true }); }, [models, params, setParams]);

  const alive = (d: Deployment) => d.status !== "stopped" && d.status !== "failed";
  const tuned = deps.find((d) => d.finetuned_model_id === modelId && alive(d)) ?? null;
  const base = deps.find((d) => d.model_kind === "base" && model && d.base_model === model.base_model && alive(d)) ?? null;
  const ready = tuned?.status === "running" && base?.status === "running";

  const startBoth = async () => {
    if (!model) return;
    setError(null);
    try {
      if (!tuned) await api.deploy({ model_kind: "finetuned", finetuned_model_id: model.id, requested_gpus: tunedGpus });
      if (!base) await api.deploy({ model_kind: "base", base_model_id: model.base_model_id, requested_gpus: 1 });
      load();
    } catch (e) { setError((e as Error).message); }
  };
  const stopBoth = async () => {
    for (const d of [tuned, base]) if (d) await api.stopDeployment(d.id).catch((e) => toast(e.message));
    load();
  };
  const run = async () => {
    if (!base || !tuned) return;
    setBusy(true); setError(null);
    try {
      const r = await api.compare(base.id, tuned.id, { ...gen, prompt });
      setAnswers({ base: r.base.text, tuned: r.finetuned.text, baseSeconds: r.base.seconds, tunedSeconds: r.finetuned.seconds, sampled: r.sampled });
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };
  const evaluate = async () => {
    if (!base || !tuned || !model) return;
    setError(null);
    try { await api.evaluate(model.id, base.id, tuned.id); load(); } catch (e) { setError((e as Error).message); }
  };

  if (models.length === 0) return <div className="card empty"><p>You need a fine-tuned model to compare.</p><Link to="/train">Train one</Link></div>;
  return (
    <div className="stack">
      <div><h1>Compare with the base model</h1><p className="muted">The same prompt goes to the original model and to your fine-tuned one, both running on pool GPUs.</p></div>
      <ErrorBox error={error} />
      <div className="card stack">
        <div className="grid cols-2">
          <div className="field">
            <label htmlFor="m">Fine-tuned model</label>
            <select id="m" value={modelId} onChange={(e) => { setParams({ model: e.target.value }); setAnswers(null); }}>
              {models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </div>
          <div className="field"><label htmlFor="g">GPUs for the fine-tuned model</label><input id="g" type="number" min={1} max={22} value={tunedGpus} onChange={(e) => setTunedGpus(Math.max(1, Number(e.target.value)))} disabled={!!tuned} /><div className="hint">The base model uses 1 GPU.</div></div>
        </div>
        <div className="row">
          <span>Fine-tuned: {tuned ? <StatusChip status={tuned.status} /> : <span className="muted">not deployed</span>}</span>
          <span>Base: {base ? <StatusChip status={base.status} /> : <span className="muted">not deployed</span>}</span>
          {!ready && <button className="primary" onClick={startBoth} disabled={!!tuned && !!base}>Deploy both</button>}
          {(tuned || base) && <button className="danger" onClick={stopBoth}>Stop both</button>}
        </div>
        {(tuned?.status === "starting" || base?.status === "starting") && <div className="alert info">Starting up. The first deployment of a model prepares it once, which takes a few seconds.</div>}
        {(tuned?.status === "queued" || base?.status === "queued") && <div className="alert warn">Waiting for idle GPUs in the pool.</div>}
      </div>

      <div className="card stack">
        <h2>Try a prompt</h2>
        <div className="field">
          <label htmlFor="pr">Prompt</label>
          <textarea id="pr" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          <div className="row" style={{ marginTop: 6 }}>{SAMPLE_PROMPTS.map((s, i) => <button key={i} className="small" onClick={() => setPrompt(s)}>Example {i + 1}</button>)}</div>
        </div>
        <SamplingControls value={gen} onChange={setGen} />
        <div className="row"><button className="primary" disabled={!ready || busy || !prompt.trim()} onClick={run}>{busy ? "Asking both models…" : "Run on both"}</button>{!ready && <span className="small muted">Both models need to be running.</span>}</div>
        {answers && (
          <div className="grid cols-2">
            <div><h3>Base model</h3><div className="answer" aria-live="polite">{answers.base}</div><div className="small muted">{duration(answers.baseSeconds)}</div></div>
            <div><h3>Your fine-tuned model</h3><div className="answer" aria-live="polite">{answers.tuned}</div><div className="small muted">{duration(answers.tunedSeconds)}</div></div>
          </div>
        )}
        {answers?.sampled && <p className="hint">Sampling was on, so answers can differ run to run. Turn it off for a like-for-like comparison.</p>}
      </div>

      <div className="card stack">
        <header><h2 style={{ margin: 0 }}>Score on held-out examples</h2><button disabled={!ready || model?.scores.status === "running"} onClick={evaluate}>{model?.scores.status === "done" ? "Evaluate again" : "Evaluate"}</button></header>
        <p className="small muted">Runs both models on examples that were not used for training and checks whether each answer is the expected JSON with the right department and urgency. Uses greedy decoding so the result is repeatable.</p>
        {model && <ScoreBars scores={model.scores} />}
      </div>
    </div>
  );
}
