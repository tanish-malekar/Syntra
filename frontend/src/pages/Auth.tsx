import { useState, type FormEvent, type ReactNode } from "react";
import { Bee, BeeMark } from "../components/Bee";
import { Icon } from "../components/icons";
import { CopyBox, ErrorBox, IconTile } from "../components/ui";
import { useAuth } from "../state";

// Nodes around the bee: what the pool is made of and what it does. Positions are % of the hub box.
const NODES: { x: number; y: number; icon: ReactNode; tone?: "honey" | "rose" | "violet" | "green"; title: string; sub: string; hot?: boolean }[] = [
  { x: 50, y: 7, icon: Icon.laptop, title: "MacBook Pro", sub: "Apple M4 Pro · 48 GB", hot: true },
  { x: 18, y: 20, icon: Icon.server, tone: "violet", title: "RTX A6000", sub: "University lab · CUDA" },
  { x: 82, y: 20, icon: Icon.server, tone: "green", title: "Data center", sub: "Verified provider" },
  { x: 12, y: 50, icon: Icon.finetune, title: "Fine-tuning", sub: "LoRA · QLoRA · DiLoCo" },
  { x: 88, y: 50, icon: Icon.rl, tone: "rose", title: "Reinforcement learning", sub: "GRPO · PPO · DPO" },
  { x: 20, y: 81, icon: Icon.shield, tone: "green", title: "Fail-over", sub: "Lost GPU replaced mid-run" },
  { x: 80, y: 81, icon: Icon.coin, title: "Earn", sub: "Paid per GPU-hour" },
  { x: 50, y: 94, icon: Icon.deploy, tone: "violet", title: "Deploy and compare", sub: "Base vs fine-tuned", hot: true },
];

function Hub() {
  return (
    <div className="hub" aria-hidden="true">
      <svg className="lines" viewBox="0 0 100 100" preserveAspectRatio="none">
        {NODES.map((n, i) => <path key={i} d={`M50 50 Q ${(50 + n.x) / 2 + (n.y < 50 ? -6 : 6)} ${(50 + n.y) / 2} ${n.x} ${n.y}`} vectorEffect="non-scaling-stroke" />)}
      </svg>
      <div className="center"><Bee size={170} float wink /></div>
      {NODES.map((n) => (
        <div key={n.title} className={`node${n.hot ? " hot" : ""}`} style={{ left: `${n.x}%`, top: `${n.y}%` }}>
          <IconTile icon={n.icon} tone={n.tone} />
          <div><b>{n.title}</b><span>{n.sub}</span></div>
        </div>
      ))}
    </div>
  );
}

export default function AuthPage() {
  const { login, signup } = useAuth();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true); setError(null);
    try {
      if (mode === "login") await login(email, password);
      else await signup(email, password, name);
    } catch (err) { setError((err as Error).message); } finally { setBusy(false); }
  };
  const swap = (m: "login" | "signup") => { setError(null); setMode(m); };
  return (
    <div className="auth-page">
      <nav className="auth-nav">
        <div className="brand"><span className="logo"><BeeMark /></span><span>Bloom<b>Bee</b></span></div>
        <div className="links"><span>Fine-tuning</span><span>Reinforcement learning</span><span>Become a provider</span></div>
        <div className="row tight">
          <button className="ghost" onClick={() => swap("login")}>Log in</button>
          <button className="primary" onClick={() => swap("signup")}>Get started ›</button>
        </div>
      </nav>
      <div className="auth-body">
        <section className="auth-hero">
          <h1>Turn the world's idle compute into <em>your AI cloud.</em></h1>
          <p className="lede">Fine-tune, reinforce and serve models on a pool of laptops, lab machines and data centers, and keep going when one drops.</p>
          <div className="auth-form">
            <form className="card" onSubmit={submit}>
              <h2>{mode === "login" ? "Welcome back" : "Create your account"}</h2>
              <p className="small muted">{mode === "login" ? "Sign in to your BloomBee workspace." : "It takes a minute. No card needed: payments are simulated."}</p>
              <ErrorBox error={error} />
              {mode === "signup" && <div className="field"><label htmlFor="name">Name</label><input id="name" value={name} onChange={(e) => setName(e.target.value)} autoComplete="name" /></div>}
              <div className="field"><label htmlFor="email">Email</label><input id="email" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" /></div>
              <div className="field">
                <label htmlFor="pw">Password</label>
                <input id="pw" type="password" required minLength={mode === "signup" ? 10 : undefined} value={password} onChange={(e) => setPassword(e.target.value)} autoComplete={mode === "login" ? "current-password" : "new-password"} />
                {mode === "signup" && <div className="hint">At least 10 characters.</div>}
              </div>
              <button className="primary block" disabled={busy}>{busy ? "One moment…" : mode === "login" ? "Sign in" : "Create account"}</button>
              <p className="small muted" style={{ marginTop: 14, marginBottom: 0 }}>
                {mode === "login" ? "New here? " : "Already have an account? "}
                <a href="#" onClick={(e) => { e.preventDefault(); swap(mode === "login" ? "signup" : "login"); }}>{mode === "login" ? "Create an account" : "Sign in"}</a>
              </p>
            </form>
          </div>
          <div className="compat">
            Become a provider with one line, from any network:
            <div style={{ marginTop: 8, maxWidth: 560 }}><CopyBox prompt text="curl -fsSL <server>/join.sh | bash -s -- --token <token>" /></div>
          </div>
        </section>
        <Hub />
      </div>
    </div>
  );
}
