import { useState } from "react";
import { api, type Ledger } from "../api";
import { Icon } from "../components/icons";
import { PageHeader } from "../components/ui";
import { duration, money } from "../format";
import { useLive } from "../state";

function Table({ title, note, data }: { title: string; note: string; data: Ledger | null }) {
  return (
    <div className="card stack">
      <header><div><h2 style={{ margin: 0 }}>{title}</h2><span className="small muted">{note}</span></div><div className="stat"><div className="v">{money(data?.total ?? 0)}</div><div className="k">{duration(data?.gpu_seconds ?? 0)} of GPU time</div></div></header>
      {!data || data.entries.length === 0 ? <div className="empty">Nothing yet.</div> : (
        <div className="table-wrap nowrap">
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
      <PageHeader icon={Icon.coin} title="Earnings and costs" subtitle="Priced per GPU-hour by GPU class. All amounts are simulated: no real money moves." />
      <div className="stack">
        <Table title="Earned by your GPUs" note="When others use GPUs you share" data={earnings} />
        <Table title="Spent on your jobs" note="Training runs and deployments you started" data={costs} />
      </div>
    </div>
  );
}
