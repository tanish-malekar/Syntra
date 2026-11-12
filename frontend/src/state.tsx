// Session + live event stream (one WebSocket per signed-in user).
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { API, ApiError, api, type StoredEvent, type User } from "./api";
import type { SocketMessage } from "./events";

// ---------------------------------------------------------------------------------------------- auth
interface AuthValue {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string, name: string) => Promise<void>;
  logout: () => Promise<void>;
}
const AuthContext = createContext<AuthValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    api.me().then(setUser).catch((e) => { if (!(e instanceof ApiError) || e.status !== 401) console.error(e); }).finally(() => setLoading(false));
  }, []);
  const value = useMemo<AuthValue>(() => ({
    user, loading,
    login: async (email, password) => setUser(await api.login(email, password)),
    signup: async (email, password, name) => setUser(await api.signup(email, password, name)),
    logout: async () => { await api.logout(); setUser(null); },
  }), [user, loading]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthValue {
  const v = useContext(AuthContext);
  if (!v) throw new Error("AuthProvider missing");
  return v;
}

// ---------------------------------------------------------------------------------------------- live events
type Handler = (event: StoredEvent) => void;
interface EventsValue {
  connected: boolean;
  subscribe: (handler: Handler) => () => void;
}
const EventsContext = createContext<EventsValue>({ connected: false, subscribe: () => () => undefined });

export function EventsProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const [connected, setConnected] = useState(false);
  const handlers = useRef(new Set<Handler>());

  useEffect(() => {
    if (!user) return;
    let ws: WebSocket | null = null;
    let closed = false;
    let retry = 1000;
    const connect = () => {
      ws = new WebSocket(API.replace(/^http/, "ws") + "/ws/events");
      ws.onopen = () => { setConnected(true); retry = 1000; };
      ws.onmessage = (m) => {
        const msg = JSON.parse(m.data) as SocketMessage;
        if (msg.type === "hello" || msg.type === "pong") return;
        handlers.current.forEach((h) => h(msg as unknown as StoredEvent));
      };
      ws.onclose = () => {
        setConnected(false);
        if (!closed) setTimeout(connect, retry = Math.min(retry * 2, 10000));
      };
    };
    connect();
    return () => { closed = true; ws?.close(); };
  }, [user]);

  const subscribe = useCallback((h: Handler) => {
    handlers.current.add(h);
    return () => { handlers.current.delete(h); };
  }, []);
  const value = useMemo(() => ({ connected, subscribe }), [connected, subscribe]);
  return <EventsContext.Provider value={value}>{children}</EventsContext.Provider>;
}

export function useEventStream(): EventsValue {
  return useContext(EventsContext);
}

/** Call `handler` for every live event while mounted. */
export function useOnEvent(handler: Handler): void {
  const { subscribe } = useEventStream();
  const ref = useRef(handler);
  ref.current = handler;
  useEffect(() => subscribe((e) => ref.current(e)), [subscribe]);
}

/** Re-run `load` on mount, every `ms`, and whenever a matching live event arrives (so pages stay fresh without polling hard). */
export function useLive(load: () => void, ms: number, match?: (e: StoredEvent) => boolean): void {
  const ref = useRef(load);
  ref.current = load;
  useEffect(() => {
    ref.current();
    const t = setInterval(() => ref.current(), ms);
    return () => clearInterval(t);
  }, [ms]);
  useOnEvent((e) => { if (!match || match(e)) ref.current(); });
}
