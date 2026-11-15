// Small stroke icons (24x24, currentColor).
import type { ReactNode } from "react";

const svg = (path: ReactNode) => (
  <svg className="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{path}</svg>
);

export const Icon = {
  finetune: svg(<><path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0" /><circle cx="16" cy="6" r="2" /><circle cx="10" cy="12" r="2" /><circle cx="18" cy="18" r="2" /></>),
  rl: svg(<><circle cx="12" cy="12" r="8.5" /><circle cx="12" cy="12" r="4.5" /><circle cx="12" cy="12" r="0.8" fill="currentColor" /><path d="m15 9 5.5-5.5M17 3.5h3.5V7" /></>),
  train: svg(<><path d="M13 3 5 13h6l-1 8 8-10h-6l1-8Z" /></>),
  runs: svg(<><path d="M3 12h4l3-8 4 16 3-8h4" /></>),
  models: svg(<><path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z" /><path d="m4 7.5 8 4.5 8-4.5M12 12v9" /></>),
  deploy: svg(<><rect x="3" y="4" width="18" height="6" rx="1.5" /><rect x="3" y="14" width="18" height="6" rx="1.5" /><path d="M7 7h.01M7 17h.01" /></>),
  compare: svg(<><path d="M8 4v16M16 4v16M4 8h8M12 16h8" /></>),
  provide: svg(<><path d="M12 15V4M8 8l4-4 4 4" /><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" /></>),
  pool: svg(<><path d="M12 2.8 19.8 7.3v9L12 20.8l-7.8-4.5v-9L12 2.8Z" /><path d="M12 8.2 15.3 10v3.9L12 15.8l-3.3-1.9V10L12 8.2Z" /></>),
  ledger: svg(<><rect x="3" y="6" width="18" height="13" rx="2" /><path d="M3 10h18M7 15h4" /></>),
  key: svg(<><circle cx="8" cy="15" r="4" /><path d="m11 12 9-9M17 6l3 3M15 8l2 2" /></>),
  plus: svg(<><path d="M12 5v14M5 12h14" /></>),
  search: svg(<><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></>),
  back: svg(<><path d="m15 18-6-6 6-6" /></>),
  check: svg(<><circle cx="12" cy="12" r="9" /><path d="m8 12 3 3 5-6" /></>),
  upload: svg(<><path d="M12 16V4M7 9l5-5 5 5" /><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" /></>),
  play: svg(<><path d="M7 4.5v15l12-7.5-12-7.5Z" /></>),
  laptop: svg(<><rect x="4" y="5" width="16" height="11" rx="1.5" /><path d="M2 19h20" /></>),
  server: svg(<><rect x="3" y="3" width="18" height="7" rx="1.5" /><rect x="3" y="14" width="18" height="7" rx="1.5" /><path d="M7 6.5h.01M7 17.5h.01M11 6.5h6M11 17.5h6" /></>),
  shield: svg(<><path d="M12 3 4.5 6v6c0 4.6 3.2 7.9 7.5 9 4.3-1.1 7.5-4.4 7.5-9V6L12 3Z" /><path d="m9 12 2 2 4-4" /></>),
  bolt: svg(<><path d="M13 3 5 13h6l-1 8 8-10h-6l1-8Z" /></>),
  layers: svg(<><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 13 9 5 9-5" /></>),
  code: svg(<><path d="m8 7-5 5 5 5M16 7l5 5-5 5M14 4l-4 16" /></>),
  sparkle: svg(<><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6" /></>),
  coin: svg(<><circle cx="12" cy="12" r="8.5" /><path d="M14.5 9.2c-.6-.8-1.5-1.2-2.5-1.2-1.5 0-2.6.8-2.6 2 0 2.8 5.3 1.4 5.3 4 0 1.2-1.1 2-2.7 2-1.1 0-2.1-.5-2.7-1.3M12 6.5V8M12 16v1.5" /></>),
};
