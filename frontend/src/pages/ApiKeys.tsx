import { useCallback, useState } from "react";
import { Link } from "react-router-dom";
import { api, type ApiKeyInfo } from "../api";
import { Icon } from "../components/icons";
import { CopyBox, EmptyState, ErrorBox, PageHeader, useToast } from "../components/ui";
import { ago } from "../format";
import { useLive } from "../state";

export default function ApiKeysPage() {
  const [keys, setKeys] = useState<ApiKeyInfo[] | null>(null);
  const [label, setLabel] = useState("");
  const [fresh, setFresh] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const toast = useToast();
  const load = useCallback(() => { api.apiKeys().then(setKeys).catch((e) => setError(e.message)); }, []);
  useLive(load, 15000);

  const create = async () => {
    try { const k = await api.createApiKey(label.trim()); setFresh(k.key ?? null); setLabel(""); load(); }
    catch (e) { setError((e as Error).message); }
  };
  const revoke = async (k: ApiKeyInfo) => {
    if (!window.confirm(`Revoke "${k.label || "key " + k.id}"? Scripts using it stop working at once.`)) return;
    try { await api.revokeApiKey(k.id); load(); } catch (e) { toast((e as Error).message); }
  };
  const active = (keys ?? []).filter((k) => !k.revoked);
  const base = window.location.origin.replace(":5173", ":8000");
  return (
    <div className="stack">
      <PageHeader icon={Icon.key} tone="violet" title="API keys"
        subtitle={<>Keys let the Python SDK act for you: RL runs, custom training loops and sessions. See <Link to="/rl">Reinforcement learning</Link> for a ready-made script.</>} />
      <ErrorBox error={error} />
      <div className="card">
        <h2>Create a key</h2>
        <div className="row">
          <input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Label, e.g. my laptop" aria-label="Key label" style={{ flex: 1, maxWidth: 420 }} />
          <button className="primary" onClick={create}>{Icon.plus}Create key</button>
        </div>
        {fresh && (
          <div className="alert info" style={{ marginTop: 14 }}>
            <strong>Copy it now: it is shown only once.</strong>
            <div style={{ marginTop: 8 }}><CopyBox text={`export BLOOMBEE_API_KEY=${fresh}\nexport BLOOMBEE_BASE_URL=${base}`} /></div>
          </div>
        )}
      </div>
      <div className="card table-card">
        {keys === null ? <div className="empty">Loading…</div> : active.length === 0 ? (
          <EmptyState title="No API keys yet">Create one above, then install the SDK with <span className="mono">pip install -e sdk/</span>.</EmptyState>
        ) : (
          <div className="table-wrap"><table>
            <thead><tr><th>Label</th><th>Created</th><th>Last used</th><th></th></tr></thead>
            <tbody>
              {active.map((k) => (
                <tr key={k.id}>
                  <td><strong>{k.label || `key ${k.id}`}</strong></td>
                  <td className="small muted">{ago(k.created_at)}</td>
                  <td className="small muted">{ago(k.last_used_at)}</td>
                  <td style={{ textAlign: "right" }}><button className="small danger" onClick={() => revoke(k)}>Revoke</button></td>
                </tr>
              ))}
            </tbody>
          </table></div>
        )}
      </div>
    </div>
  );
}
