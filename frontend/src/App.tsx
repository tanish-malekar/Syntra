import { BrowserRouter, Navigate, NavLink, Route, Routes } from "react-router-dom";
import { Icon } from "./components/icons";
import { Chip, ToastProvider } from "./components/ui";
import AuthPage from "./pages/Auth";
import ComparePage from "./pages/Compare";
import DeploymentsPage from "./pages/Deployments";
import LedgerPage from "./pages/Ledger";
import ModelsPage from "./pages/Models";
import PoolPage from "./pages/Pool";
import ProvidePage from "./pages/Provide";
import RunPage, { RunsPage } from "./pages/Runs";
import TrainPage from "./pages/Train";
import { AuthProvider, EventsProvider, useAuth, useEventStream } from "./state";

function Shell() {
  const { user, logout } = useAuth();
  const { connected } = useEventStream();
  const link = ({ isActive }: { isActive: boolean }) => (isActive ? "active" : "");
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand"><span className="logo">S</span><span>Syntra</span></div>
        <nav className="nav" aria-label="Main">
          <div className="group">Use GPUs</div>
          <NavLink to="/train" className={link}>{Icon.train}Train a model</NavLink>
          <NavLink to="/runs" className={link}>{Icon.runs}Training runs</NavLink>
          <NavLink to="/models" className={link}>{Icon.models}My models</NavLink>
          <NavLink to="/deployments" className={link}>{Icon.deploy}Deployments</NavLink>
          <NavLink to="/compare" className={link}>{Icon.compare}Compare</NavLink>
          <div className="group">Share GPUs</div>
          <NavLink to="/provide" className={link}>{Icon.provide}Provide GPUs</NavLink>
          <NavLink to="/pool" className={link}>{Icon.pool}GPU pool</NavLink>
          <NavLink to="/ledger" className={link}>{Icon.ledger}Earnings and costs</NavLink>
        </nav>
        <div className="me">
          <div className="avatar">{(user?.display_name || user?.email || "?")[0].toUpperCase()}</div>
          <div className="small"><strong>{user?.display_name || user?.email}</strong></div>
          <div className="small muted" style={{ marginBottom: 6 }}>{user?.email}</div>
          <div className="row">
            <Chip tone={connected ? "good" : "warn"} pulse={!connected}>{connected ? "Live" : "Reconnecting"}</Chip>
            <button className="small" onClick={() => void logout()}>Sign out</button>
          </div>
        </div>
      </aside>
      <main className="main" id="main">
        <Routes>
          <Route path="/" element={<Navigate to="/train" replace />} />
          <Route path="/pool" element={<PoolPage />} />
          <Route path="/provide" element={<ProvidePage />} />
          <Route path="/train" element={<TrainPage />} />
          <Route path="/runs" element={<RunsPage />} />
          <Route path="/runs/:id" element={<RunPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/deployments" element={<DeploymentsPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/ledger" element={<LedgerPage />} />
          <Route path="*" element={<Navigate to="/train" replace />} />
        </Routes>
      </main>
    </div>
  );
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
