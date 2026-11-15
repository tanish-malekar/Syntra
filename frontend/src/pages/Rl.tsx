// Reinforcement learning: a policy samples on pool GPUs, the user's reward function scores answers on their own machine,
// and GRPO / PPO updates the adapter. The website plans the run and writes the script; the SDK runs the loop.
import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api, type BaseModel, type Model, type Plan, type RlSession, type WeightBits } from "../api";
import { Icon } from "../components/icons";
import { Choices, CopyBox, EmptyState, ErrorBox, IconTile, PageHeader, StatusChip, Tag, useToast } from "../components/ui";
import { money } from "../format";
import { useLive } from "../state";
import { WeightBitsField } from "./Train";

// ---------------------------------------------------------------------------------------------- reward templates
type Template = "json" | "concise" | "custom";
const TEMPLATES: Record<Template, { title: string; detail: string; prompts: string[]; reward: string }> = {
  json: {
    title: "Valid JSON (guided example)",
    detail: "Route support tickets: reward answers that are JSON with a department and an urgency.",
    prompts: [
      "Ticket: my package has not arrived yet. Please fix this asap.\nAnswer:",
      "Ticket: the app crashes when I upload a photo.\nAnswer:",
      "Ticket: I was charged twice for my subscription.\nAnswer:",
      "Ticket: how do I change the email on my account?\nAnswer:",
    ],
    reward: `def reward(prompt, completion, sample):
    """1.0 for JSON with a department and an urgency, 0.5 for other JSON, 0.0 otherwise."""
    try:
        data = json.loads(completion.strip())
    except ValueError:
        return 0.0
    if isinstance(data, dict) and {"department", "urgency"} <= data.keys():
        return 1.0
    return 0.5`,
  },
  concise: {
    title: "Short, complete answers",
    detail: "Reward answers that end with a full stop and stay under about 30 words.",
    prompts: ["Explain what a GPU does in one sentence.", "What is fine-tuning?", "Why do models need more memory for training than for inference?"],
    reward: `def reward(prompt, completion, sample):
    """Prefer short answers that end with a full stop."""
    words = len(completion.split())
    ends_well = 1.0 if completion.strip().endswith(".") else 0.0
    return ends_well - 0.02 * max(0, words - 30)`,
  },
  custom: {
    title: "Write your own",
    detail: "Any Python: run tests, call a checker or a judge model. It runs on your machine.",
    prompts: ["Your first prompt", "Your second prompt"],
    reward: `def reward(prompt, completion, sample):
    """Return a number for each answer: higher is better."""
    return 0.0`,
  },
};

type Algo = "grpo" | "ppo";

function pyString(s: string): string {
  return JSON.stringify(s); // JSON string escapes are valid Python string literals
}

function buildScript(o: {
  baseName: string; initModel: Model | null; rank: number; weightBits: WeightBits; trusted: boolean; name: string;
  algo: Algo; iterations: number; group: number; lr: number; kl: number; maxNew: number; prompts: string[]; reward: string; sessionId: number | null;
}): string {
  const open = o.sessionId !== null
    ? `tc = client.session(${o.sessionId})   # the GPUs you reserved on the website\ntc.wait_ready()\nwith tc:`
    : `with client.training_session(\n    ${pyString(o.baseName)}, rank=${o.rank}, weight_bits=${o.weightBits},${o.initModel ? ` init_model=${o.initModel.id},` : ""}${o.trusted ? " trusted_only=True," : ""}\n    name=${pyString(o.name)},\n) as tc:`;
  const loop = o.algo === "grpo"
    ? `bloombee_sdk.rl.grpo(\n        tc, PROMPTS, reward, iterations=${o.iterations}, group_size=${o.group}, lr=${o.lr}, kl_coef=${o.kl},\n        max_new_tokens=${o.maxNew}, on_iteration=report,\n    )`
    : `bloombee_sdk.rl.ppo(\n        tc, PROMPTS, reward, iterations=${o.iterations}, samples_per_prompt=${o.group}, lr=${o.lr}, kl_coef=${o.kl},\n        max_new_tokens=${o.maxNew}, on_iteration=report,\n    )`;
  return `import json
import bloombee_sdk

client = bloombee_sdk.Client()   # reads BLOOMBEE_API_KEY and BLOOMBEE_BASE_URL

PROMPTS = [
${o.prompts.map((p) => `    ${pyString(p)},`).join("\n")}
]

${o.reward.trim()}


def report(h):
    print(f"iteration {h['iteration']:>3}  reward {h['reward_mean']:.3f}  best {h['reward_max']:.3f}")


${open}
    ${loop}
    model_id = tc.save(${pyString(o.name)})   # appears in My models
    print("saved as model", model_id)
`;
}

// ---------------------------------------------------------------------------------------------- RL home
export default function RlPage() {
  const nav = useNavigate();
  const toast = useToast();
  const [sessions, setSessions] = useState<RlSession[] | null>(null);
  const [idle, setIdle] = useState<number | null>(null);
  const load = useCallback(() => {
    api.sessions().then(setSessions).catch(() => setSessions([]));
    api.gpus().then((g) => setIdle(g.filter((x) => x.status === "idle").length)).catch(() => undefined);
  }, []);
  useLive(load, 5000, (e) => e.type.startsWith("session.") || e.type === "lease.created" || e.type === "worker.online");
  const close = async (s: RlSession) => {
    const name = window.prompt("Save the adapter before closing? Enter a name, or leave empty to discard.", String(s.config.model_name ?? ""));
    if (name === null) return;
    try { await api.closeSession(s.id, name.trim() || null); load(); } catch (e) { toast((e as Error).message); }
  };
  const live = (sessions ?? []).filter((s) => !["stopped", "failed"].includes(s.status));
  const ordered = [...live, ...(sessions ?? []).filter((s) => !live.includes(s))];

  return (
    <div className="stack">
      <PageHeader icon={Icon.rl} tone="rose" title="Reinforcement learning"
        subtitle="Train with a reward instead of example answers. The model writes answers on pool GPUs, your reward function scores them on your machine, and GRPO or PPO updates the model."
        actions={<button className="primary" onClick={() => nav("/rl/new")}>{Icon.plus}New RL run</button>} />

      <div className="grid cols-3">
        <div className="card feature">
          <div className="top"><IconTile icon={Icon.sparkle} /><Tag>On the pool</Tag></div>
          <h3>Sample</h3>
          <p>The model writes several answers to each prompt, on GPUs from the pool.</p>
          <div className="foot">{Icon.check}<span>LoRA or 4-bit QLoRA, split across GPUs, with fail-over.</span></div>
        </div>
        <div className="card feature">
          <div className="top"><IconTile icon={Icon.code} tone="rose" /><Tag tone="rose">On your machine</Tag></div>
          <h3>Score</h3>
          <p>Your reward function grades every answer: tests, a checker, rules or a judge model.</p>
          <div className="foot">{Icon.check}<span>Your reward code and its secrets never leave your machine.</span></div>
        </div>
        <div className="card feature">
          <div className="top"><IconTile icon={Icon.rl} tone="violet" /><Tag tone="violet">GRPO · PPO</Tag></div>
          <h3>Update</h3>
          <p>Answers that beat the rest of their group are reinforced; a KL penalty keeps the model close to where it started.</p>
          <div className="foot">{Icon.check}<span>Saved to My models: deploy and compare it like any model.</span></div>
        </div>
      </div>

      <div className="strip">
        <span><b>RL sessions:</b> {live.length} active · {(sessions ?? []).length} in total</span>
        <span>GPU pool <span className="num">{idle ?? "–"} idle</span></span>
      </div>

      <div className="card table-card">
        {sessions === null ? <div className="empty">Loading…</div> : sessions.length === 0 ? (
          <EmptyState title="Create your first RL run" wink action={
            <div className="row" style={{ justifyContent: "center" }}>
              <button className="honey" onClick={() => nav("/rl/new")}>{Icon.plus}New RL run</button>
              <button onClick={() => nav("/rl/new?template=json")}>{Icon.play}Guided example: JSON routing</button>
            </div>
          }>
            Pick a model and a reward, copy the script, and watch the reward go up. Runs show here while they hold GPUs.
          </EmptyState>
        ) : (
          <div className="table-wrap"><table>
            <thead><tr><th>Session</th><th>Base model</th><th>Weights</th><th>GPUs</th><th>Status</th><th>Started</th><th></th></tr></thead>
            <tbody>
              {ordered.slice(0, 30).map((s) => (
                <tr key={s.id}>
                  <td><strong>#{s.id} {String(s.config.model_name ?? "")}</strong></td>
                  <td>{s.base_model}</td>
                  <td>{Number(s.config.weight_bits) === 4 ? "4-bit" : "Float16"}</td>
                  <td>{s.job.leases.filter((l) => l.active).length || "–"}</td>
                  <td><StatusChip status={s.ready ? "running" : (s.status as never)} /></td>
                  <td className="small muted">{new Date(s.job.created_at).toLocaleString()}</td>
                  <td>{live.includes(s) && <button className="small" onClick={() => close(s)}>Close…</button>}</td>
                </tr>
              ))}
            </tbody>
          </table></div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------- new RL run
export function NewRlRunPage() {
  const nav = useNavigate();
  const [params] = useSearchParams();
  const [bases, setBases] = useState<BaseModel[]>([]);
  const [mine, setMine] = useState<Model[]>([]);
  const [baseId, setBaseId] = useState<number | null>(null);
  const [initId, setInitId] = useState<number | null>(null);
  const [rank, setRank] = useState(16);
  const [weightBits, setWeightBits] = useState<WeightBits>(16);
  const [trusted, setTrusted] = useState(false);
  const [algo, setAlgo] = useState<Algo>("grpo");
  const [iterations, setIterations] = useState(20);
  const [group, setGroup] = useState(8);
  const [lr, setLr] = useState(1e-5);
  const [kl, setKl] = useState(0.02);
  const [maxNew, setMaxNew] = useState(48);
  const initial = (params.get("template") as Template) in TEMPLATES ? (params.get("template") as Template) : "json";
  const [template, setTemplate] = useState<Template>(initial);
  const [prompts, setPrompts] = useState(TEMPLATES[initial].prompts.join("\n---\n"));
  const [reward, setReward] = useState(TEMPLATES[initial].reward);
  const [name, setName] = useState("");
  const [plan, setPlan] = useState<Plan | null>(null);
  const [reserved, setReserved] = useState<RlSession | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.baseModels().then((m) => { setBases(m); setBaseId(m[0]?.id ?? null); }).catch((e) => setError(e.message));
    api.models().then(setMine).catch(() => undefined);
  }, []);
  useEffect(() => {
    if (baseId === null) return;
    const t = setTimeout(() => { api.plan(baseId, null, trusted, weightBits, "sft").then(setPlan).catch(() => setPlan(null)); }, 250);
    return () => clearTimeout(t);
  }, [baseId, trusted, weightBits]);
  useLive(() => { if (reserved) api.sessions().then((all) => setReserved(all.find((s) => s.id === reserved.id) ?? reserved)).catch(() => undefined); },
    3000, (e) => e.type.startsWith("session.") || e.type === "worker.online");

  const base = bases.find((b) => b.id === baseId) ?? null;
  const init = mine.find((m) => m.id === initId) ?? null;
  const chooseTemplate = (t: Template) => { setTemplate(t); setPrompts(TEMPLATES[t].prompts.join("\n---\n")); setReward(TEMPLATES[t].reward); };
  const promptList = prompts.split(/\n---\n/).map((p) => p.trim()).filter(Boolean);
  const runName = name.trim() || `${base?.name ?? "model"} ${algo.toUpperCase()}`;
  const script = useMemo(() => buildScript({
    baseName: base?.name ?? "", initModel: init, rank, weightBits, trusted, name: runName, algo, iterations, group, lr, kl, maxNew,
    prompts: promptList, reward, sessionId: reserved && !["stopped", "failed"].includes(reserved.status) ? reserved.id : null,
  }), [base, init, rank, weightBits, trusted, runName, algo, iterations, group, lr, kl, maxNew, promptList.join("\u0000"), reward, reserved]); // eslint-disable-line react-hooks/exhaustive-deps

  const reserve = async () => {
    if (baseId === null) return;
    setBusy(true); setError(null);
    try {
      setReserved(await api.createSession({ base_model_id: baseId, init_model_id: initId, rank, weight_bits: weightBits, requested_gpus: null, trusted_only: trusted, name: runName }));
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };
  const answers = iterations * group * Math.min(algo === "grpo" ? 4 : 8, promptList.length || 1);

  return (
    <div className="stack">
      <PageHeader back={() => nav("/rl")} title="New RL run"
        subtitle="Choose the model, the algorithm and the reward. BloomBee writes the script; run it anywhere with Python and your API key." />
      <ErrorBox error={error} />
      <div className="create">
        <div className="stack">
          <div className="card">
            <h2><span className="step">1</span>Policy</h2>
            <div className="grid cols-2" style={{ gap: 14 }}>
              <div className="field">
                <label htmlFor="rl-base">Base model</label>
                <select id="rl-base" value={baseId ?? ""} onChange={(e) => { setBaseId(Number(e.target.value)); setInitId(null); }}>
                  {bases.map((b) => <option key={b.id} value={b.id}>{b.name} · {b.num_layers} layers</option>)}
                </select>
              </div>
              <div className="field">
                <label htmlFor="rl-init">Start from</label>
                <select id="rl-init" value={initId ?? ""} onChange={(e) => setInitId(e.target.value ? Number(e.target.value) : null)}>
                  <option value="">The base model</option>
                  {mine.filter((m) => m.base_model_id === baseId).map((m) => <option key={m.id} value={m.id}>My model: {m.name}</option>)}
                </select>
                <div className="hint">Starting from a fine-tuned model usually learns much faster.</div>
              </div>
            </div>
            <div className="grid cols-2" style={{ gap: 14 }}>
              <div className="field">
                <label htmlFor="rl-rank">LoRA rank</label>
                <select id="rl-rank" value={rank} onChange={(e) => setRank(Number(e.target.value))}>
                  {[8, 16, 32, 64].map((r) => <option key={r} value={r}>{r}</option>)}
                </select>
              </div>
              <div className="field">
                <label htmlFor="rl-name">Model name</label>
                <input id="rl-name" value={name} onChange={(e) => setName(e.target.value)} placeholder={runName} />
              </div>
            </div>
            <WeightBitsField value={weightBits} onChange={setWeightBits} name="rl-bits" purpose="train" />
          </div>

          <div className="card">
            <h2><span className="step">2</span>Algorithm</h2>
            <Choices name="algo" value={algo} onChange={(a) => { setAlgo(a); setGroup(a === "grpo" ? 8 : 4); }} options={[
              { value: "grpo", title: "GRPO", badge: "Recommended", detail: "Several answers per prompt; each is scored against its group. No value model." },
              { value: "ppo", title: "PPO", detail: "A moving reward baseline and several clipped epochs per batch." },
            ]} />
            <div className="grid cols-3" style={{ gap: 14, marginTop: 14 }}>
              <div className="field"><label htmlFor="rl-it">Iterations</label><input id="rl-it" type="number" min={1} max={1000} value={iterations} onChange={(e) => setIterations(Number(e.target.value))} /></div>
              <div className="field"><label htmlFor="rl-g">{algo === "grpo" ? "Group size" : "Samples per prompt"}</label><input id="rl-g" type="number" min={2} max={64} value={group} onChange={(e) => setGroup(Number(e.target.value))} /></div>
              <div className="field"><label htmlFor="rl-max">Max answer tokens</label><input id="rl-max" type="number" min={4} max={512} value={maxNew} onChange={(e) => setMaxNew(Number(e.target.value))} /></div>
              <div className="field"><label htmlFor="rl-lr">Learning rate</label><input id="rl-lr" type="number" step="0.000001" min={0} value={lr} onChange={(e) => setLr(Number(e.target.value))} /></div>
              <div className="field"><label htmlFor="rl-kl">KL penalty</label><input id="rl-kl" type="number" step="0.01" min={0} max={1} value={kl} onChange={(e) => setKl(Number(e.target.value))} /><div className="hint">0 lets it drift freely.</div></div>
            </div>
          </div>

          <div className="card">
            <h2><span className="step">3</span>Reward</h2>
            <Choices name="tpl" value={template} onChange={chooseTemplate} options={(Object.keys(TEMPLATES) as Template[]).map((t) => ({ value: t, title: TEMPLATES[t].title, detail: TEMPLATES[t].detail }))} />
            <div className="field" style={{ marginTop: 14 }}>
              <label htmlFor="rl-prompts">Prompts <span className="muted" style={{ fontWeight: 500 }}>(separate them with a line containing only ---)</span></label>
              <textarea id="rl-prompts" className="mono" style={{ minHeight: 130 }} value={prompts} onChange={(e) => setPrompts(e.target.value)} />
              <div className="hint">{promptList.length} prompt{promptList.length === 1 ? "" : "s"}.</div>
            </div>
            <div className="field">
              <label htmlFor="rl-reward">Reward function (Python)</label>
              <textarea id="rl-reward" className="mono" style={{ minHeight: 190 }} value={reward} onChange={(e) => setReward(e.target.value)} spellCheck={false} />
              <div className="hint">Called for every answer with the prompt, the answer text and the sample. Return a number; higher is better.</div>
            </div>
          </div>

          <div className="card">
            <h2><span className="step">4</span>GPUs</h2>
            <p className="small muted" style={{ marginTop: 0 }}>BloomBee picks the GPUs for you: the fewest idle GPUs that fit the model. If one drops, another takes over.</p>
            <label><input type="checkbox" checked={trusted} onChange={(e) => setTrusted(e.target.checked)} /> Trusted GPUs only (verified providers)</label>
          </div>

          <div className="card">
            <h2><span className="step">5</span>Run it</h2>
            <ol className="small" style={{ paddingLeft: 18, margin: "0 0 14px", display: "grid", gap: 6 }}>
              <li>Create an API key on <Link to="/api-keys">API keys</Link>.</li>
              <li>In a terminal: <span className="mono">pip install -e sdk/</span>, then set <span className="mono">BLOOMBEE_API_KEY</span> and <span className="mono">BLOOMBEE_BASE_URL={window.location.origin.replace(":5173", ":8000")}</span>.</li>
              <li>Save this as <span className="mono">rl_run.py</span> and run it. The reward is printed every iteration.</li>
            </ol>
            <CopyBox text={script} tall />
          </div>
        </div>

        <aside className="summary card">
          <h2>Summary</h2>
          <dl>
            <dt>Policy</dt><dd>{init ? init.name : base?.name ?? "–"}</dd>
            <dt>Adapter</dt><dd>LoRA rank {rank}{weightBits === 4 ? ", 4-bit" : ""}</dd>
            <dt>Algorithm</dt><dd>{algo.toUpperCase()}</dd>
            <dt>Reward</dt><dd>{TEMPLATES[template].title.replace(" (guided example)", "")}</dd>
            <dt>Prompts</dt><dd>{promptList.length}</dd>
            <dt>Answers</dt><dd>about {answers.toLocaleString()}</dd>
            <dt>GPUs</dt><dd>{plan?.fits ? `${plan.stages.length} on ${plan.machines} machine${plan.machines === 1 ? "" : "s"}` : "Automatic"}</dd>
          </dl>
          <hr className="divider" />
          <div className="small muted">Estimated cost while the GPUs are held</div>
          <div className="price">{plan?.fits ? money(plan.hourly_rate) : "–"}<span className="small muted" style={{ fontWeight: 500 }}> / hour</span></div>
          {reserved ? (
            <div className="alert info" style={{ marginTop: 14 }}>
              <div className="row spread"><strong>Session #{reserved.id}</strong><StatusChip status={reserved.ready ? "running" : (reserved.status as never)} /></div>
              <div className="small" style={{ marginTop: 4 }}>{reserved.ready ? "GPUs are loaded. The script below now attaches to this session." : "Loading the model onto the GPUs. The script waits for it."}</div>
            </div>
          ) : (
            <button className="honey block" style={{ marginTop: 16 }} disabled={busy || baseId === null} onClick={reserve}>{busy ? "Reserving…" : "Reserve GPUs now"}</button>
          )}
          <p className="hint" style={{ marginBottom: 0 }}>Optional: reserve now so the model is loaded when your script starts. Otherwise the script leases GPUs itself. Idle sessions are saved and closed after 10 minutes.</p>
        </aside>
      </div>
    </div>
  );
}
