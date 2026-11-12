import { useState } from "react";
import { api, type Ledger } from "../api";
import { duration, money } from "../format";
import { useLive } from "../state";

function Table({ title, note, data }: { title: string; note: string; data: Ledger | null }) {
  return (
    <div className="card stack">
      <header><div><h2 style={{ margin: 0 }}>{title}</h2><span className="small muted">{note}</span></div><div className="stat"><div className="v">{money(data?.total ?? 0)}</div><div className="k">{duration(data?.gpu_seconds ?? 0)} of GPU time</div></div></header>
      {!data || data.entries.length === 0 ? <div className="empty">Nothing yet.</div> : (
        <div className="table-wrap">
          <table>
            <thead><tr><th>When</th><th>GPU class</th><th>Time</th><th>Rate</th><th>Amount</th></tr></thead>
            <tbody>
              {data.entries.map((e) => (
                <tr key={e.id}><td className="small muted">{new Date(e.created_at).toLocaleString()}</td><td>{e.gpu_class}</td><td>{duration(e.gpu_seconds)}</td><td className="mono">{money(e.rate_per_hour)}/h</td><td className="mono">{money(e.amount)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default function LedgerPage() {
  const [earnings, setEarnings] = useState<Ledger | null>(null);
  const [costs, setCosts] = useState<Ledger | null>(null);
  useLive(() => { api.earnings().then(setEarnings).catch(() => undefined); api.costs().then(setCosts).catch(() => undefined); }, 5000, (e) => e.type === "ledger.credit" || e.type === "train.completed" || e.type === "deploy.stopped");
  return (
    <div className="stack">
      <div><h1>Earnings and costs</h1><p className="muted">Priced per GPU-hour by GPU class. All amounts are simulated, no real money moves.</p></div>
      <div className="grid cols-2">
        <Table title="Earned by your GPUs" note="When others use GPUs you share" data={earnings} />
        <Table title="Spent on your jobs" note="Training runs and deployments you started" data={costs} />
      </div>
    </div>
  );
}
