// BloomBee's mascot: a round, friendly bee drawn in SVG (no image assets).
import { useId } from "react";

export function Bee({ size = 120, float = false, wink = false, title }: { size?: number; float?: boolean; wink?: boolean; title?: string }) {
  const id = useId().replace(/:/g, "");
  return (
    <svg className={`bee${float ? " float" : ""}`} width={size} height={size} viewBox="0 0 120 120" role={title ? "img" : undefined}
      aria-hidden={title ? undefined : true} aria-label={title}>
      {/* wings */}
      <g className="wing">
        <ellipse cx="44" cy="34" rx="17" ry="22" transform="rotate(-24 44 34)" fill="#eaf6ff" stroke="#b9d9ef" strokeWidth="2" />
        <ellipse cx="78" cy="34" rx="17" ry="22" transform="rotate(24 78 34)" fill="#eaf6ff" stroke="#b9d9ef" strokeWidth="2" />
      </g>
      {/* antennae */}
      <path d="M50 30 C46 18 40 14 34 13" stroke="#2a2017" strokeWidth="3" fill="none" strokeLinecap="round" />
      <path d="M70 30 C74 18 80 14 86 13" stroke="#2a2017" strokeWidth="3" fill="none" strokeLinecap="round" />
      <circle cx="33" cy="13" r="4.5" fill="#2a2017" />
      <circle cx="87" cy="13" r="4.5" fill="#2a2017" />
      {/* stinger */}
      <path d="M60 106 l-5 -6 h10 Z" fill="#2a2017" />
      {/* body */}
      <defs>
        <radialGradient id={`body${id}`} cx="40%" cy="35%" r="70%">
          <stop offset="0%" stopColor="#ffe48a" />
          <stop offset="70%" stopColor="#ffc52e" />
          <stop offset="100%" stopColor="#f2a516" />
        </radialGradient>
        <clipPath id={`clip${id}`}><ellipse cx="60" cy="68" rx="36" ry="34" /></clipPath>
      </defs>
      <ellipse cx="60" cy="68" rx="36" ry="34" fill={`url(#body${id})`} stroke="#2a2017" strokeWidth="3" />
      <g clipPath={`url(#clip${id})`} fill="#2a2017">
        <rect x="20" y="78" width="80" height="8" rx="3" />
        <rect x="20" y="93" width="80" height="8" rx="3" />
      </g>
      {/* face */}
      {wink ? (
        <path d="M42 61 q6 -6 12 0" stroke="#2a2017" strokeWidth="3.2" fill="none" strokeLinecap="round" />
      ) : (
        <g>
          <ellipse cx="48" cy="61" rx="6" ry="7" fill="#2a2017" />
          <circle cx="50" cy="58.5" r="2.2" fill="#fff" />
        </g>
      )}
      <ellipse cx="72" cy="61" rx="6" ry="7" fill="#2a2017" />
      <circle cx="74" cy="58.5" r="2.2" fill="#fff" />
      <ellipse cx="40" cy="71" rx="5.5" ry="3.5" fill="#ff8f8f" opacity="0.6" />
      <ellipse cx="80" cy="71" rx="5.5" ry="3.5" fill="#ff8f8f" opacity="0.6" />
      <path d="M53 70 q7 7 14 0" stroke="#2a2017" strokeWidth="3" fill="none" strokeLinecap="round" />
    </svg>
  );
}

/** The small logo mark (no animation). */
export function BeeMark() {
  return <Bee size={26} />;
}
