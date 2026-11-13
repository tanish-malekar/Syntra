"""Episodes, rewards and N games at once (for training)."""
from __future__ import annotations

import os
import random

from .level import get_level
from .sim import TILE, Sim
from .textstate import TEXT_VERSION, render_text

REWARD_SCALE = 0.1
R_PROGRESS = 1.0      # per tile of new rightmost progress
R_TIME = -0.05        # per decision
R_COIN = 0.5
R_STOMP = 1.0
R_DEATH = -15.0
R_FLAG = 30.0

EVAL_SEED_BASE = 1_000_000   # evaluation seeds are never used in training

# Opt-in "hard spot" starts for training (off unless set): MARIO_HARD_STARTS="42,78,168" lists tiles where the model
# still fails; MARIO_HARD_FRAC is the share of mid-level starts that begin 2-10 tiles before one of them.
HARD_STARTS = [int(x) for x in os.environ.get("MARIO_HARD_STARTS", "").split(",") if x.strip()]
HARD_FRAC = float(os.environ.get("MARIO_HARD_FRAC", "0.5"))


def make_sim(level: str, seed: int, difficulty: int = 1, randomize: bool = True, start_col: int | None = None) -> Sim:
    return Sim(get_level(level), seed=seed, randomize=randomize, difficulty=difficulty, start_col=start_col)


def safe_starts(level: str) -> list[int]:
    """Columns where a mid-level start is fair: solid footing 2 tiles wide, headroom, no enemy spawn within 3 tiles."""
    lv = get_level(level)
    solid = lambda c, r: lv.tiles[r][c] in "#B?USP"
    tops = []
    for c in range(lv.w):
        t = next((r for r in range(lv.h) if solid(c, r)), None)
        tops.append(t)
    out = []
    for c in range(8, lv.flag_col - 6):
        if tops[c] is None or tops[c] < 3 or tops[c + 1] != tops[c]:
            continue
        if any(lv.tiles[r][c] in "#B?USP" for r in range(max(0, tops[c] - 3), tops[c])):
            continue
        if any(abs(ec - c) <= 3 for ec, _ in lv.enemies):
            continue
        out.append(c)
    return out


class GameEnv:
    """One game. mode 'level1': always Level 1 (randomised); 'train': half Level 1, half generated variants."""

    def __init__(self, seed: int, mode: str = "train", difficulty: int = 1, variants: int = 200, level1_frac: float = 0.5,
                 mid_start: float = 0.0, death: float = R_DEATH, flag: float = R_FLAG):
        self.rng = random.Random(seed)
        self.r_death, self.r_flag = death, flag
        self.mode, self.difficulty, self.variants, self.level1_frac = mode, difficulty, variants, level1_frac
        self.mid_start = mid_start   # share of Level 1 training games that start at a random safe spot mid-level
        self.reset()

    def reset(self, seed: int | None = None, level: str | None = None):
        self.ep_seed = self.rng.randrange(EVAL_SEED_BASE) if seed is None else seed
        if level is None:
            if self.mode == "level1" or self.rng.random() < self.level1_frac:
                level = "level1"
            else:
                level = f"gen:{self.rng.randrange(self.variants)}:{self.difficulty}"
        self.level_name = level
        self.start_col = None
        if seed is None and level == "level1" and self.mid_start and self.rng.random() < self.mid_start:
            starts = safe_starts("level1")
            if HARD_STARTS and self.rng.random() < HARD_FRAC:
                near = [c for c in starts if any(2 <= h - c <= 10 for h in HARD_STARTS)]
                starts = near or starts
            self.start_col = self.rng.choice(starts)
        self.sim = make_sim(level, self.ep_seed, self.difficulty, start_col=self.start_col)
        self.actions: list[int] = []
        self.ep_return = 0.0
        return self.text()

    text_version = TEXT_VERSION

    def text(self) -> str:
        return render_text(self.sim, self.text_version)

    def step(self, action: int) -> tuple[float, bool]:
        s = self.sim
        max0, coins0, stomps0 = s.max_x, s.coins, s.stomps
        s.step(action)
        s.events.clear()
        self.actions.append(action)
        r = R_PROGRESS * (s.max_x - max0) / TILE + R_TIME + R_COIN * (s.coins - coins0) + R_STOMP * (s.stomps - stomps0)
        if s.done:
            if s.won:
                r += self.r_flag
            elif s.dead:
                r += self.r_death
        r *= REWARD_SCALE
        self.ep_return += r
        return r, s.done

    def summary(self) -> dict:
        s = self.sim
        return {"level": self.level_name, "seed": self.ep_seed, "difficulty": self.difficulty, "won": int(s.won),
                "reason": s.reason, "dist": round(s.max_x / TILE, 1), "frac": round(s.max_x / TILE / s.flag_col, 3),
                "coins": s.coins, "stomps": s.stomps, "decisions": len(self.actions),
                "return": round(self.ep_return, 3)}

    def replay(self) -> dict:
        rep = {**self.summary(), "actions": "".join(str(a) for a in self.actions)}
        if self.start_col is not None:
            rep["start_col"] = self.start_col
        return rep


class VecEnv:
    def __init__(self, n: int, seed: int = 0, mode: str = "train", difficulty: int = 1, level1_frac: float = 0.5,
                 mid_start: float = 0.0, death: float = R_DEATH, flag: float = R_FLAG):
        self.envs = [GameEnv(seed * 10_007 + i, mode, difficulty, level1_frac=level1_frac, mid_start=mid_start,
                             death=death, flag=flag) for i in range(n)]

    def set_text_version(self, v: int):
        for e in self.envs:
            e.text_version = v

    def texts(self) -> list[str]:
        return [e.text() for e in self.envs]

    def step(self, actions) -> tuple[list[float], list[bool], list[dict]]:
        rewards, dones, finished = [], [], []
        for e, a in zip(self.envs, actions):
            r, d = e.step(int(a))
            rewards.append(r)
            dones.append(d)
            if d:
                finished.append(e.replay())
                e.reset()
        return rewards, dones, finished


def replay_sim(rep: dict) -> Sim:
    """Re-run a replay (seed + moves) with the same simulation; used by tests and the demo page."""
    s = make_sim(rep["level"], rep["seed"], rep.get("difficulty", 1), start_col=rep.get("start_col"))
    for ch in rep["actions"]:
        if s.done:
            break
        s.step(int(ch))
        s.events.clear()
    return s
