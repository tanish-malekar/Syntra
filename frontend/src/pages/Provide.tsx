import { useCallback, useEffect, useState } from "react";
import { API, api, type AgentTokenInfo, type Gpu, type Ledger, type Provider } from "../api";
import { Icon } from "../components/icons";
import { CopyBox, ErrorBox, GpuChip, Chip, PageHeader } from "../components/ui";
import { ago, mb, money } from "../format";
import { useLive } from "../state";

function joinCommand(base: string, token: string): string {
  return `curl -fsSL ${base}/join.sh | bash -s -- --token ${token}`;
}

function ProviderCard({ provider }: { provider: Provider }) {
  const [tokens, setTokens] = useState<AgentTokenInfo[]>([]);
  const [fresh, setFresh] = useState<string | null>(null);
  const [label, setLabel] = useState("");
  const [base, setBase] = useState<string>(API);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(() => { api.tokens(provider.id).then(setTokens).catch((e) => setError(e.message)); }, [provider.id]);
  useEffect(load, [load]);
  useEffect(() => { api.installInfo().then((i) => setBase(i.public_url)).catch(() => undefined); }, []);

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
          <p className="small" style={{ margin: "8px 0" }}>Paste this in a terminal on the machine (Mac or Linux with an NVIDIA GPU). It installs what is needed the first time and
            adds the machine's GPUs to the pool. Leave the window open; Ctrl-C to leave.</p>
          <CopyBox prompt text={joinCommand(base, fresh)} />
          <p className="small" style={{ marginTop: 8 }}>Needs Python 3.12 (on a Mac: <span className="mono">brew install python@3.12</span>). No ports to open, any network.</p>
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
      <PageHeader icon={Icon.provide} tone="green" title="Become a provider" subtitle="Lend idle GPUs to the pool, as an individual or as a data center. You earn for every GPU-hour they are used (simulated payments)." />
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
