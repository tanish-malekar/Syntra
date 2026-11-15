import { useState } from "react";
import { api, type Gpu } from "../api";
import { Link } from "react-router-dom";
import { Icon } from "../components/icons";
import { Chip, EmptyState, ErrorBox, GpuChip, PageHeader } from "../components/ui";
import { ago, mb, money } from "../format";
import { useLive } from "../state";

export default function PoolPage() {
  const [gpus, setGpus] = useState<Gpu[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useLive(() => { api.gpus().then((g) => { setGpus(g); setError(null); }).catch((e) => setError(e.message)); }, 3000, (e) => e.type !== "gpu.heartbeat");

  const count = (s: Gpu["status"]) => gpus?.filter((g) => g.status === s).length ?? 0;
  const idleMemory = gpus?.filter((g) => g.status === "idle").reduce((a, g) => a + g.free_memory_mb, 0) ?? 0;
  return (
    <div className="stack">
      <PageHeader icon={Icon.pool} title="GPU pool" subtitle="Every GPU shared by individuals and data centers. Training, RL and deployments take idle GPUs from here automatically."
        actions={<Link to="/provide"><button className="primary">{Icon.plus}Become a provider</button></Link>} />
      <ErrorBox error={error} />
      <div className="stats">
        <div className="stat"><div className="v">{gpus?.length ?? "–"}</div><div className="k">GPUs in the pool</div></div>
        <div className="stat"><div className="v">{count("idle")}</div><div className="k">Idle, ready for work</div></div>
        <div className="stat"><div className="v">{count("leased")}</div><div className="k">In use</div></div>
        <div className="stat"><div className="v">{count("offline") + count("busy_external")}</div><div className="k">Offline or busy elsewhere</div></div>
        <div className="stat"><div className="v">{mb(idleMemory)}</div><div className="k">Idle memory</div></div>
      </div>
      <div className="card table-card">
        {gpus === null ? <div className="empty">Loading…</div> : gpus.length === 0 ? (
          <EmptyState title="The pool is empty" action={<Link to="/provide"><button className="honey">Become a provider</button></Link>}>Any Mac or NVIDIA machine can join with one command.</EmptyState>
        ) : (
          <div className="table-wrap">
            <table>
              <thead><tr><th>GPU</th><th>Provider</th><th>Class · price</th><th>Memory (free / total)</th><th>Region</th><th>Status</th><th>Seen</th></tr></thead>
              <tbody>
                {gpus.map((g) => (
                  <tr key={g.id}>
                    <td>
                      <strong>{g.name}</strong> {g.simulated && <Chip tone="plain">simulated</Chip>}
                      <div className="small muted">{g.device.toUpperCase()} · #{g.device_index}</div>
                    </td>
                    <td>{g.provider_name}<div className="small muted">{g.provider_type === "enterprise" ? "Data center" : "Individual"}{g.trust_tier === "verified" ? " · verified" : ""}</div></td>
                    <td>{g.gpu_class}<div className="small muted">{money(g.price_per_hour)} / hour</div></td>
                    <td className="mono">{mb(g.free_memory_mb)} / {mb(g.total_memory_mb)}</td>
                    <td>{g.region || "–"}</td>
                    <td><GpuChip status={g.status} /></td>
                    <td className="small muted">{ago(g.last_heartbeat)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
