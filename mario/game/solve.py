"""Beam search over moves: proves a level is clearable and produces a winning move list (tests, demo fixtures)."""
from __future__ import annotations


from .vec import make_sim


def solve(level: str, seed: int, difficulty: int = 1, beam: int = 48, max_steps: int = 2000) -> str | None:
    root = make_sim(level, seed, difficulty)
    frontier = [(root, "")]
    for _ in range(max_steps):
        cand, seen = [], set()
        for s, moves in frontier:
            for a in range(6):
                t = s.clone()
                t.step(a)
                t.events.clear()
                if t.won:
                    return moves + str(a)
                if t.done:
                    continue
                key = (round(t.x / 4), round(t.y / 4), round(t.vy), tuple(e.alive for e in t.enemies))
                if key in seen:
                    continue
                seen.add(key)
                cand.append((t, moves + str(a)))
        if not cand:
            return None
        cand.sort(key=lambda p: -(p[0].x + 0.2 * p[0].max_x))
        frontier = cand[:beam]
    return None
