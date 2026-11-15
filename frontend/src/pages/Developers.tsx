import { useCallback, useState } from "react";
import { api, type ApiKeyInfo, type RlSession } from "../api";
import { ErrorBox, StatusChip, useToast } from "../components/ui";
import { useLive } from "../state";

const EXAMPLE = `import json, bloombee_sdk

client = bloombee_sdk.Client()   # BLOOMBEE_API_KEY, BLOOMBEE_BASE_URL
with client.training_session("TinyLlama-1.1B-Chat", rank=16, weight_bits=4, gpus=2) as tc:
    def reward(prompt, completion, sample):
        try:
            json.loads(completion); return 1.0
        except ValueError:
            return 0.0
    bloombee_sdk.rl.grpo(tc, prompts, reward, iterations=20, group_size=8, lr=1e-5)
    tc.save("my-grpo-model")   # appears in My models`;

export default function DevelopersPage() {
  const [keys, setKeys] = useState<ApiKeyInfo[]>([]);
  const [sessions, setSessions] = useState<RlSession[]>([]);
  const [label, setLabel] = useState("");
  const [fresh, setFresh] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const toast = useToast();
  const load = useCallback(() => {
    api.apiKeys().then(setKeys).catch((e) => setError(e.message));
    api.sessions().then(setSessions).catch(() => undefined);
  }, []);
  useLive(load, 5000, (e) => e.type.startsWith("session."));

  const create = async () => {
    try { const k = await api.createApiKey(label.trim()); setFresh(k.key ?? null); setLabel(""); load(); }
    catch (e) { setError((e as Error).message); }
  };
  const revoke = async (k: ApiKeyInfo) => {
    if (!window.confirm(`Revoke "${k.label || "key " + k.id}"? Scripts using it stop working at once.`)) return;
    try { await api.revokeApiKey(k.id); load(); } catch (e) { toast((e as Error).message); }
  };
  const close = async (s: RlSession) => {
    const name = window.prompt("Save the adapter before closing? Enter a name, or leave empty to discard.", String(s.config.model_name ?? ""));
    if (name === null) return;
    try { await api.closeSession(s.id, name.trim() || null); load(); } catch (e) { toast((e as Error).message); }
  };
  const live = sessions.filter((s) => !["stopped", "failed"].includes(s.status));
  return (
    <div className="stack">
      <div>
        <h1>Developers: SDK and RL</h1>
        <p className="muted">Run reinforcement learning (GRPO, PPO, DPO) or your own training loop from Python. Your reward function runs on your machine;
          the adapter trains on GPUs from the pool, with LoRA or 4-bit QLoRA, and lands in My models.</p>
      </div>
      <ErrorBox error={error} />
      <div className="grid cols-2">
        <div className="card stack">
          <h2>API keys</h2>
          {fresh && <div className="alert info"><strong>Copy it now, it is shown once:</strong> <span className="mono">{fresh}</span></div>}
          <div className="row">
            <input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Label, e.g. laptop" aria-label="Key label" style={{ flex: 1 }} />
            <button className="primary" onClick={create}>Create key</button>
          </div>
          {keys.filter((k) => !k.revoked).length === 0 ? <p className="small muted">No keys yet.</p> : (
            <div className="table-wrap"><table><tbody>
              {keys.filter((k) => !k.revoked).map((k) => (
                <tr key={k.id}><td>{k.label || `key ${k.id}`}</td>
                  <td className="small muted">{k.last_used_at ? `used ${new Date(k.last_used_at).toLocaleString()}` : "never used"}</td>
                  <td><button className="small danger" onClick={() => revoke(k)}>Revoke</button></td></tr>
              ))}
            </tbody></table></div>
          )}
        </div>
        <div className="card stack">
          <h2>Quick start</h2>
          <pre className="mono small" style={{ whiteSpace: "pre-wrap", margin: 0 }}>{`pip install -e sdk/\nexport BLOOMBEE_API_KEY=syn_key_...\nexport BLOOMBEE_BASE_URL=${window.location.origin.replace(":5173", ":8000")}`}</pre>
          <pre className="mono small" style={{ whiteSpace: "pre-wrap", margin: 0 }}>{EXAMPLE}</pre>
          <p className="small muted">Losses: cross_entropy, dpo, ppo, importance_sampling. Sessions with no call for 10 minutes are saved and closed automatically.</p>
        </div>
      </div>
      <div className="card stack">
        <h2>Training sessions</h2>
        {sessions.length === 0 ? <p className="small muted">No sessions yet. Start one from the SDK.</p> : (
          <div className="table-wrap"><table>
            <thead><tr><th>Session</th><th>Base model</th><th>Status</th><th>GPUs</th><th></th></tr></thead>
            <tbody>
              {[...live, ...sessions.filter((s) => !live.includes(s))].slice(0, 20).map((s) => (
                <tr key={s.id}>
                  <td><strong>#{s.id} {String(s.config.model_name ?? "")}</strong>{Number(s.config.weight_bits) === 4 && <span className="small muted"> · 4-bit</span>}</td>
                  <td>{s.base_model}</td>
                  <td><StatusChip status={s.ready ? "running" : (s.status as never)} /></td>
                  <td>{s.job.leases.filter((l) => l.active).length}</td>
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
