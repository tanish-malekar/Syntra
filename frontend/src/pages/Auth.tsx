import { useState, type FormEvent } from "react";
import { ErrorBox } from "../components/ui";
import { useAuth } from "../state";

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
  return (
    <div className="auth">
      <div className="brand" style={{ padding: 0, fontSize: 26 }}>Syn<span>tra</span></div>
      <p className="muted">Fine-tune models on GPUs shared by people and data centers, and earn from the GPUs you share.</p>
      <form className="card" onSubmit={submit}>
        <h2>{mode === "login" ? "Sign in" : "Create your account"}</h2>
        <ErrorBox error={error} />
        {mode === "signup" && <div className="field"><label htmlFor="name">Name</label><input id="name" value={name} onChange={(e) => setName(e.target.value)} autoComplete="name" /></div>}
        <div className="field"><label htmlFor="email">Email</label><input id="email" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" /></div>
        <div className="field">
          <label htmlFor="pw">Password</label>
          <input id="pw" type="password" required minLength={mode === "signup" ? 10 : undefined} value={password} onChange={(e) => setPassword(e.target.value)} autoComplete={mode === "login" ? "current-password" : "new-password"} />
          {mode === "signup" && <div className="hint">At least 10 characters.</div>}
        </div>
        <button className="primary" style={{ width: "100%" }} disabled={busy}>{mode === "login" ? "Sign in" : "Create account"}</button>
        <p className="small muted" style={{ marginTop: 14, marginBottom: 0 }}>
          {mode === "login" ? "New here? " : "Already have an account? "}
          <a href="#" onClick={(e) => { e.preventDefault(); setError(null); setMode(mode === "login" ? "signup" : "login"); }}>{mode === "login" ? "Create an account" : "Sign in"}</a>
        </p>
      </form>
    </div>
  );
}
