// Small stroke icons for the sidebar (24x24, currentColor).
import type { ReactNode } from "react";

const svg = (path: ReactNode) => (
  <svg className="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{path}</svg>
);

export const Icon = {
  train: svg(<><path d="M13 3 5 13h6l-1 8 8-10h-6l1-8Z" /></>),
  runs: svg(<><path d="M3 12h4l3-8 4 16 3-8h4" /></>),
  models: svg(<><path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z" /><path d="m4 7.5 8 4.5 8-4.5M12 12v9" /></>),
  deploy: svg(<><rect x="3" y="4" width="18" height="6" rx="1.5" /><rect x="3" y="14" width="18" height="6" rx="1.5" /><path d="M7 7h.01M7 17h.01" /></>),
  compare: svg(<><path d="M8 4v16M16 4v16M4 8h8M12 16h8" /></>),
  provide: svg(<><path d="M12 15V4M8 8l4-4 4 4" /><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" /></>),
  pool: svg(<><circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3c3 3.2 3 14.8 0 18M12 3c-3 3.2-3 14.8 0 18" /></>),
  ledger: svg(<><rect x="3" y="6" width="18" height="13" rx="2" /><path d="M3 10h18M7 15h4" /></>),
};
