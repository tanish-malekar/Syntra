import { useState } from "react";
import { BrowserRouter, Navigate, NavLink, Route, Routes, useParams } from "react-router-dom";
import { api } from "./api";
import { BeeMark } from "./components/Bee";
import { Icon } from "./components/icons";
import { Chip, ToastProvider } from "./components/ui";
import { money } from "./format";
import ApiKeysPage from "./pages/ApiKeys";
import AuthPage from "./pages/Auth";
import ComparePage from "./pages/Compare";
import DeploymentsPage from "./pages/Deployments";
import LedgerPage from "./pages/Ledger";
import ModelsPage from "./pages/Models";
import PoolPage from "./pages/Pool";
import ProvidePage from "./pages/Provide";
import RlPage, { NewRlRunPage } from "./pages/Rl";
import RunPage, { RunsPage } from "./pages/Runs";
import TrainPage from "./pages/Train";
import { AuthProvider, EventsProvider, useAuth, useEventStream, useLive } from "./state";

function Balance() {
  const [spent, setSpent] = useState<number | null>(null);
  const [earned, setEarned] = useState<number | null>(null);
  useLive(() => {
    api.costs().then((l) => setSpent(l.total)).catch(() => undefined);
    api.earnings().then((l) => setEarned(l.total)).catch(() => undefined);
  }, 15000, (e) => e.type === "ledger.credit" || e.type === "train.completed" || e.type === "deploy.stopped");
  return (
    <NavLink to="/ledger" className="balance" style={{ textDecoration: "none", fontWeight: 400 }}>
      <div className="line"><span>Spent</span><b>{spent === null ? "–" : money(spent)}</b></div>
      <div className="line"><span>Earned</span><b>{earned === null ? "–" : money(earned)}</b></div>
      <div className="line" style={{ fontSize: 11.5 }}><span>Simulated payments</span></div>
    </NavLink>
  );
}

function Shell() {
  const { user, logout } = useAuth();
  const { connected } = useEventStream();
  const link = ({ isActive }: { isActive: boolean }) => (isActive ? "active" : "");
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand"><span className="logo"><BeeMark /></span><span>Bloom<b>Bee</b></span></div>
        <nav className="nav" aria-label="Main">
          <div className="group">Model shaping</div>
          <NavLink to="/fine-tuning" className={link}>{Icon.finetune}Fine-tuning</NavLink>
          <NavLink to="/rl" className={link}>{Icon.rl}Reinforcement learning</NavLink>
          <NavLink to="/models" className={link}>{Icon.models}My models</NavLink>
          <div className="group">Inference</div>
          <NavLink to="/deployments" className={link}>{Icon.deploy}Deployments</NavLink>
          <NavLink to="/compare" className={link}>{Icon.compare}Compare</NavLink>
          <div className="group">GPU network</div>
          <NavLink to="/pool" className={link}>{Icon.pool}GPU pool</NavLink>
          <NavLink to="/provide" className={link}>{Icon.provide}Become a provider</NavLink>
          <NavLink to="/ledger" className={link}>{Icon.ledger}Earnings and costs</NavLink>
          <div className="group">Developers</div>
          <NavLink to="/api-keys" className={link}>{Icon.key}API keys</NavLink>
        </nav>
        <div style={{ marginTop: "auto", paddingTop: 16 }}>
          <Balance />
          <div className="me">
            <div className="avatar">{(user?.display_name || user?.email || "?")[0].toUpperCase()}</div>
            <div className="who">
              <div className="small"><strong>{user?.display_name || user?.email}</strong></div>
              <div className="small"><Chip tone={connected ? "good" : "warn"} pulse={!connected}>{connected ? "Live" : "Reconnecting"}</Chip></div>
            </div>
            <button className="small ghost" onClick={() => void logout()}>Sign out</button>
          </div>
        </div>
      </aside>
      <main className="main" id="main">
        <Routes>
          <Route path="/" element={<Navigate to="/fine-tuning/new" replace />} />
          <Route path="/fine-tuning" element={<RunsPage />} />
          <Route path="/fine-tuning/new" element={<TrainPage />} />
          <Route path="/fine-tuning/:id" element={<RunPage />} />
          <Route path="/rl" element={<RlPage />} />
          <Route path="/rl/new" element={<NewRlRunPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/deployments" element={<DeploymentsPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/pool" element={<PoolPage />} />
          <Route path="/provide" element={<ProvidePage />} />
          <Route path="/ledger" element={<LedgerPage />} />
          <Route path="/api-keys" element={<ApiKeysPage />} />
          {/* old addresses */}
          <Route path="/train" element={<Navigate to="/fine-tuning/new" replace />} />
          <Route path="/runs" element={<Navigate to="/fine-tuning" replace />} />
          <Route path="/runs/:id" element={<OldRun />} />
          <Route path="/developers" element={<Navigate to="/rl" replace />} />
          <Route path="*" element={<Navigate to="/fine-tuning/new" replace />} />
        </Routes>
      </main>
    </div>
  );
}

function OldRun() {
  return <Navigate to={`/fine-tuning/${useParams().id}`} replace />;
}

function Gate() {
  const { user, loading } = useAuth();
  if (loading) return <div className="empty">Loading…</div>;
  return user ? <Shell /> : <AuthPage />;
}

export default function App() {
  return (
    <BrowserRouter>
      <ToastProvider>
        <AuthProvider>
          <EventsProvider>
            <Gate />
          </EventsProvider>
        </AuthProvider>
      </ToastProvider>
    </BrowserRouter>
  );
}
