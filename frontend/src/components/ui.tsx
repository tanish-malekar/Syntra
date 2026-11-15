// Shared building blocks: chips, copy box, loss chart, timeline, toasts.
import { createContext, useCallback, useContext, useState, type ReactNode } from "react";
import type { GpuStatus, StoredEvent } from "../api";
import { describeEvent, type LossPoint } from "../format";
import { Bee } from "./Bee";
import { Icon } from "./icons";

export function Chip({ tone, children, pulse }: { tone?: "good" | "bad" | "warn" | "info" | "plain" | "violet"; children: ReactNode; pulse?: boolean }) {
  return <span className={`chip ${tone ?? ""} ${pulse ? "pulse" : ""}`}>{children}</span>;
}

const GPU_TONE: Record<GpuStatus, "good" | "info" | "warn" | "bad"> = { idle: "good", leased: "info", busy_external: "warn", offline: "bad" };
const GPU_LABEL: Record<GpuStatus, string> = { idle: "Idle", leased: "In use", busy_external: "Busy elsewhere", offline: "Offline" };
export function GpuChip({ status }: { status: GpuStatus }) {
  return <Chip tone={GPU_TONE[status]} pulse={status === "leased"}>{GPU_LABEL[status]}</Chip>;
}

const JOB_TONE: Record<string, "good" | "info" | "warn" | "bad" | "plain"> = {
  queued: "warn", starting: "info", running: "good", completed: "good", failed: "bad", stopped: "plain",
};
export function StatusChip({ status }: { status: string }) {
  return <Chip tone={JOB_TONE[status] ?? "plain"} pulse={status === "running" || status === "starting"}>{status[0].toUpperCase() + status.slice(1)}</Chip>;
}

export function CopyBox({ text, prompt, tall }: { text: string; prompt?: boolean; tall?: boolean }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className={`codebox${tall ? " tall" : ""}`}>
      {prompt && <span className="prompt">$</span>}{text}
      <button
        className="small"
        onClick={() => { void navigator.clipboard?.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1500); }}
        aria-label="Copy to clipboard"
      >
        {copied ? "Copied" : "Copy"}
      </button>
    </div>
  );
}

export function ErrorBox({ error }: { error: string | null }) {
  return error ? <div className="alert error" role="alert">{error}</div> : null;
}

type Tone = "honey" | "rose" | "violet" | "green";
const TILE_CLASS: Record<Tone, string> = { honey: "", rose: "rose", violet: "violet", green: "green" };

export function IconTile({ icon, tone = "honey", small }: { icon: ReactNode; tone?: Tone; small?: boolean }) {
  return <span className={`tile ${TILE_CLASS[tone]} ${small ? "sm" : ""}`}>{icon}</span>;
}

export function Tag({ children, tone = "honey" }: { children: ReactNode; tone?: Tone }) {
  return <span className={`tag ${TILE_CLASS[tone]}`}>{children}</span>;
}

/** Page title row: icon tile, title, one-line description, actions on the right. */
export function PageHeader({ icon, tone, title, subtitle, actions, back }: {
  icon?: ReactNode; tone?: Tone; title: ReactNode; subtitle?: ReactNode; actions?: ReactNode; back?: () => void;
}) {
  return (
    <div className="page-head">
      {back && <button className="back" onClick={back} aria-label="Back">{Icon.back}</button>}
      {icon && <IconTile icon={icon} tone={tone} />}
      <div className="text"><h1>{title}</h1>{subtitle && <p>{subtitle}</p>}</div>
      {actions && <div className="actions">{actions}</div>}
    </div>
  );
}

/** Friendly empty state with the bee. */
export function EmptyState({ title, children, action, wink }: { title: string; children?: ReactNode; action?: ReactNode; wink?: boolean }) {
  return (
    <div className="empty-state">
      <Bee size={92} float wink={wink} />
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

/** Radio buttons drawn as cards. */
export function Choices<T extends string>({ name, value, onChange, options, columns }: {
  name: string; value: T; onChange: (v: T) => void; columns?: number;
  options: { value: T; title: ReactNode; detail?: ReactNode; badge?: string; disabled?: boolean }[];
}) {
  return (
    <div className="choices" role="radiogroup" style={columns ? { gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` } : undefined}>
      {options.map((o) => (
        <label key={o.value} className={`choice ${value === o.value ? "on" : ""} ${o.disabled ? "disabled" : ""}`}>
          <input type="radio" name={name} checked={value === o.value} disabled={o.disabled} onChange={() => onChange(o.value)} />
          <span className="t">{o.title}{o.badge && <span className="badge">{o.badge}</span>}</span>
          {o.detail && <span className="d">{o.detail}</span>}
        </label>
      ))}
    </div>
  );
}

export function SearchBox({ value, onChange, placeholder }: { value: string; onChange: (v: string) => void; placeholder: string }) {
  return (
    <div className="search">
      {Icon.search}
      <input value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} aria-label={placeholder} />
    </div>
  );
}

// ---------------------------------------------------------------------------------------------- toasts
const ToastContext = createContext<(message: string) => void>(() => undefined);
export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<{ id: number; text: string }[]>([]);
  const push = useCallback((text: string) => {
    const id = Date.now() + Math.random();
    setItems((cur) => [...cur, { id, text }]);
    setTimeout(() => setItems((cur) => cur.filter((t) => t.id !== id)), 5000);
  }, []);
  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="toasts" aria-live="polite">{items.map((t) => <div className="toast" key={t.id}>{t.text}</div>)}</div>
    </ToastContext.Provider>
  );
}
export const useToast = () => useContext(ToastContext);

// ---------------------------------------------------------------------------------------------- loss chart
export function LossChart({ points }: { points: LossPoint[] }) {
  const W = 640, H = 220, L = 44, R = 12, T = 12, B = 28;
  if (points.length === 0) {
    return <div className="empty">The loss curve appears here as soon as training starts.</div>;
  }
  const xs = points.map((p) => p.step), ys = points.map((p) => p.loss);
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = (y1 - y0) * 0.1 || 0.1;
  const lo = Math.max(0, y0 - pad), hi = y1 + pad;
  const sx = (x: number) => L + ((x - x0) / (x1 - x0)) * (W - L - R);
  const sy = (y: number) => T + (1 - (y - lo) / (hi - lo)) * (H - T - B);
  const path = points.map((p, i) => `${i ? "L" : "M"}${sx(p.step).toFixed(1)},${sy(p.loss).toFixed(1)}`).join(" ");
  const area = `${path} L${sx(points[points.length - 1].step).toFixed(1)},${H - B} L${sx(points[0].step).toFixed(1)},${H - B} Z`;
  const yTicks = [lo, (lo + hi) / 2, hi];
  const xTicks = [x0, Math.round((x0 + x1) / 2), x1];
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Training loss from ${ys[0].toFixed(2)} to ${ys[ys.length - 1].toFixed(2)} over ${points.length} steps`}>
      {yTicks.map((t) => (
        <g key={t}>
          <line className="axis" x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} />
          <text className="tick" x={L - 6} y={sy(t) + 4} textAnchor="end">{t.toFixed(2)}</text>
        </g>
      ))}
      {xTicks.map((t, i) => <text key={i} className="tick" x={sx(t)} y={H - 8} textAnchor="middle">{t}</text>)}
      <path className="area" d={area} />
      <path className="line" d={path} />
      {points.length < 40 && points.map((p) => <circle key={p.step} className="dotpt" cx={sx(p.step)} cy={sy(p.loss)} r="3" />)}
      <text className="tick" x={L} y={T + 2} dy="-2">loss</text>
    </svg>
  );
}

// ---------------------------------------------------------------------------------------------- timeline
export function Timeline({ events }: { events: StoredEvent[] }) {
  const items = events.map(describeEvent).filter((x): x is NonNullable<ReturnType<typeof describeEvent>> => x !== null);
  if (items.length === 0) return <div className="empty">Nothing has happened yet.</div>;
  return (
    <ol className="timeline">
      {items.map((i) => (
        <li key={i.key}>
          <span className={`dot ${i.tone}`} />
          <div>
            <div>{i.title}</div>
            {i.detail && <div className="small muted">{i.detail}</div>}
          </div>
          <span className="t">{new Date(i.ts).toLocaleTimeString()}</span>
        </li>
      ))}
    </ol>
  );
}
