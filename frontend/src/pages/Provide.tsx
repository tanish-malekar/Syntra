import { useCallback, useEffect, useState } from "react";
import { API, api, type AgentTokenInfo, type Gpu, type Ledger, type Provider } from "../api";
import { CopyBox, ErrorBox, GpuChip, Chip } from "../components/ui";
import { ago, mb, money } from "../format";
import { useLive } from "../state";

function agentCommand(token: string, flavor: "mac" | "nvidia"): string {
  const python = flavor === "mac" ? "~/bloombee-venv/bin/python" : "/path/to/bloombee-env/bin/python";
  const extra = flavor === "nvidia" ? " \\\n  --gpus 0,1" : "";
  return `cd syntra/agent && python -m syntra_agent \\\n  --server ${API} \\\n  --token ${token} \\\n  --bloombee-dir ~/BloomBee --python ${python} \\\n  --models-dir ~/syntra-models --region home${extra}`;
}

function ProviderCard({ provider }: { provider: Provider }) {
  const [tokens, setTokens] = useState<AgentTokenInfo[]>([]);
  const [fresh, setFresh] = useState<string | null>(null);
  const [label, setLabel] = useState("");
  const [flavor, setFlavor] = useState<"mac" | "nvidia">("mac");
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(() => { api.tokens(provider.id).then(setTokens).catch((e) => setError(e.message)); }, [provider.id]);
  useEffect(load, [load]);

  const add = async () => {
    setError(null);
    try {
      const t = await api.createToken(provider.id, label.trim() || "my machine");
      setFresh(t.token);
      setLabel("");
      load();
    } catch (e) { setError((e as Error).message); }
  };
  return (
    <div className="card stack">
      <header>
        <div>
          <h2 style={{ margin: 0 }}>{provider.name}</h2>
          <span className="small muted">{provider.type === "enterprise" ? "Data center" : "Individual"}</span>
        </div>
        <Chip tone={provider.trust_tier === "verified" ? "good" : "plain"}>{provider.trust_tier === "verified" ? "Verified" : "Standard"}</Chip>
      </header>
      <ErrorBox error={error} />
      <div>
        <h3>Connect a machine</h3>
        <p className="small muted">Each machine gets its own token. Run the agent on it; it registers the machine's GPUs and they join the pool within seconds. It only makes outgoing connections, so no ports need opening.</p>
        <div className="row">
          <div style={{ flex: 1, minWidth: 180 }}><input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Machine name, e.g. MacBook Pro" aria-label="Machine name" /></div>
          <button className="primary" onClick={add}>Create token</button>
        </div>
      </div>
      {fresh && (
        <div className="alert info">
          <strong>Copy this now.</strong> The token is shown once.
          <div className="row" style={{ margin: "8px 0" }}>
            <button className={flavor === "mac" ? "small primary" : "small"} onClick={() => setFlavor("mac")}>Mac (Apple GPU)</button>
            <button className={flavor === "nvidia" ? "small primary" : "small"} onClick={() => setFlavor("nvidia")}>Linux (NVIDIA)</button>
          </div>
          <CopyBox text={agentCommand(fresh, flavor)} />
          <p className="small" style={{ marginTop: 8 }}>You need Python 3.12 with BloomBee installed (branch <span className="mono">mac-mps-ft</span>) and the agent's requirements.</p>
        </div>
      )}
      {tokens.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead><tr><th>Machine</th><th>Created</th><th>Last seen</th><th></th></tr></thead>
            <tbody>
              {tokens.map((t) => (
                <tr key={t.id}>
                  <td>{t.label || "unnamed"} {t.revoked && <Chip tone="bad">revoked</Chip>}</td>
                  <td className="small muted">{ago(t.created_at)}</td>
                  <td className="small muted">{ago(t.last_used_at)}</td>
                  <td>{!t.revoked && <button className="small danger" onClick={() => { void api.revokeToken(provider.id, t.id).then(load); }}>Revoke</button>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default function ProvidePage() {
  const [providers, setProviders] = useState<Provider[] | null>(null);
  const [gpus, setGpus] = useState<Gpu[]>([]);
  const [earnings, setEarnings] = useState<Ledger | null>(null);
  const [name, setName] = useState("");
  const [type, setType] = useState<"individual" | "enterprise">("individual");
  const [error, setError] = useState<string | null>(null);
  const loadProviders = useCallback(() => { api.providers().then(setProviders).catch((e) => setError(e.message)); }, []);
  useEffect(loadProviders, [loadProviders]);
  useLive(() => { api.myGpus().then(setGpus).catch(() => undefined); api.earnings().then(setEarnings).catch(() => undefined); }, 4000, (e) => e.type !== "gpu.heartbeat");

  const create = async () => {
    setError(null);
    try { await api.createProvider(name.trim(), type); setName(""); loadProviders(); } catch (e) { setError((e as Error).message); }
  };

  return (
    <div className="stack">
      <div>
        <h1>Share your GPUs</h1>
        <p className="muted">Lend idle GPUs to the pool, as an individual or as a data center. You earn for every GPU-hour they are used (simulated payments).</p>
      </div>
      <ErrorBox error={error} />
      {earnings && (
        <div className="stats">
          <div className="stat"><div className="v">{money(earnings.total)}</div><div className="k">Earned so far</div></div>
          <div className="stat"><div className="v">{(earnings.gpu_seconds / 3600).toFixed(2)} h</div><div className="k">GPU time served</div></div>
          <div className="stat"><div className="v">{gpus.length}</div><div className="k">Your GPUs in the pool</div></div>
        </div>
      )}
      <div className="card">
        <h2>{providers && providers.length > 0 ? "Add another provider profile" : "Create your provider profile"}</h2>
        <div className="grid cols-2">
          <div className="field"><label htmlFor="pname">Name</label><input id="pname" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Maya's home lab, or Acme Data Centers" /></div>
          <div className="field">
            <label htmlFor="ptype">Who are you?</label>
            <select id="ptype" value={type} onChange={(e) => setType(e.target.value as "individual" | "enterprise")}>
              <option value="individual">An individual with a few GPUs</option>
              <option value="enterprise">A data center or company</option>
            </select>
          </div>
        </div>
        <button className="primary" disabled={!name.trim()} onClick={create}>Create profile</button>
      </div>
      {providers?.map((p) => <ProviderCard key={p.id} provider={p} />)}
      {gpus.length > 0 && (
        <div className="card">
          <h2>Your GPUs</h2>
          <div className="table-wrap">
            <table>
              <thead><tr><th>GPU</th><th>Class · price</th><th>Memory</th><th>Status</th></tr></thead>
              <tbody>
                {gpus.map((g) => (
                  <tr key={g.id}>
                    <td><strong>{g.name}</strong>{g.simulated && <> <Chip tone="plain">simulated</Chip></>}<div className="small muted">{g.provider_name}</div></td>
                    <td>{g.gpu_class}<div className="small muted">{money(g.price_per_hour)} / hour</div></td>
                    <td className="mono">{mb(g.free_memory_mb)} / {mb(g.total_memory_mb)}</td>
                    <td><GpuChip status={g.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
