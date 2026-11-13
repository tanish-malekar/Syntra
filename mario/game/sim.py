"""Deterministic platformer simulation, pure Python, 60 logic frames per second.

The same code runs headless for training (on context3) and behind the demo page (the browser only draws what this
sends), so training and demo can never disagree. Units are pixels, y grows downwards, one tile is 16 px.
"""
from __future__ import annotations

import math
import random

TILE = 16
GRAVITY = 0.35
MAX_FALL = 6.0
RUN = 2.0                 # fixed run speed (px/frame): forgiving, no momentum
JUMP_LONG = 6.6           # ~3.9 tiles high, ~4.7 tiles far at run speed
JUMP_SHORT = 4.4          # ~1.7 tiles high
JUMP_CUT = 2.5            # releasing the jump key while rising caps upward speed (keyboard play only)
STOMP_BOUNCE = 4.5
HERO_W, HERO_H = 12, 14
ENEMY_W, ENEMY_H = 14, 14
ENEMY_SPEED = {0: 0.4, 1: 0.6, 2: 0.8}   # by difficulty
ENEMY_WAKE = 20 * TILE    # enemies start walking once the hero is this close
FRAMES_PER_DECISION = 6
STUCK_FRAMES = 180        # 3 s without new progress ends the episode
MAX_FRAMES = 60 * 120

SOLID = set("#B?USP")

# (name, horizontal input, jump kind): jump 0 none, 1 short hop, 2 long jump
MOVES = [
    ("run right", 1, 0),
    ("run right + long jump", 1, 2),
    ("run right + short hop", 1, 1),
    ("left", -1, 0),
    ("jump in place", 0, 2),
    ("wait", 0, 0),
]
N_MOVES = len(MOVES)


def floor(v: float) -> int:
    return int(math.floor(v))


class Enemy:
    __slots__ = ("x", "y", "vx", "vy", "alive", "active", "squash", "gone")

    def __init__(self, x: float, y: float, vx: float):
        self.x, self.y, self.vx, self.vy = x, y, vx, 0.0
        self.alive = True      # can hurt / be stomped
        self.active = False    # walking (woken up by the hero coming close)
        self.squash = 0        # frames since stomped (for the renderer)
        self.gone = False      # fell out of the level or squash animation finished


class Sim:
    def __init__(self, level, seed: int = 0, randomize: bool = True, difficulty: int = 1,
                 stuck_frames: int | None = STUCK_FRAMES, start_col: int | None = None):
        self.level = level
        self.seed = seed
        self.difficulty = difficulty
        self.stuck_frames = stuck_frames   # None: never end for lack of progress (keyboard play)
        self.rng = random.Random(seed)
        self.w, self.h = level.w, level.h
        self.tiles = [list(row) for row in level.tiles]
        self.coin_tiles = set(level.coins)
        self.flag_col = level.flag_col
        offset = self.rng.uniform(0, 2 * TILE) if randomize else 0.0
        self.start_col = level.start_col if start_col is None else start_col
        self.x = self.start_col * TILE + 2 + offset
        self.y = (level.start_row + 1) * TILE - HERO_H
        if start_col is not None:   # mid-level start (training curriculum): stand on top of that column
            top = next((r for r in range(self.h) if self.solid_at(start_col, r)), self.h)
            self.y = top * TILE - HERO_H
        self.vx = self.vy = 0.0
        self.on_ground = True
        self.face = 1
        speed = ENEMY_SPEED[difficulty]
        self.enemies: list[Enemy] = []
        for c, r in level.enemies:
            ex = c * TILE + 1.0
            if randomize:
                cand = ex + self.rng.uniform(-1.5, 1.5) * TILE
                if self._box_free(cand, (r + 1) * TILE - ENEMY_H, ENEMY_W, ENEMY_H) and \
                        self.solid_at(floor((cand + ENEMY_W / 2) / TILE), r + 1):
                    ex = cand
            vx = -speed if (not randomize or self.rng.random() < 0.7) else speed
            self.enemies.append(Enemy(ex, (r + 1) * TILE - ENEMY_H, vx))
        self.frame = 0
        self.max_x = self.x
        self.last_progress = 0
        self.coins = 0
        self.stomps = 0
        self.dead = False
        self.won = False
        self.done = False
        self.reason = ""
        self.post = 0          # frames since death/win (animation only, after `done`)
        self.events: list[list] = []

    def clone(self) -> "Sim":
        """Fast copy (shares the immutable level; copies everything that changes)."""
        s = Sim.__new__(Sim)
        s.__dict__.update(self.__dict__)
        s.rng = random.Random()
        s.rng.setstate(self.rng.getstate())
        s.tiles = [row[:] for row in self.tiles]
        s.coin_tiles = set(self.coin_tiles)
        s.enemies = []
        for e in self.enemies:
            f = Enemy.__new__(Enemy)
            for k in Enemy.__slots__:
                setattr(f, k, getattr(e, k))
            s.enemies.append(f)
        s.events = []
        return s

    # ---------- tiles ----------
    def solid_at(self, c: int, r: int) -> bool:
        if c < 0 or c >= self.w:
            return True
        if r < 0 or r >= self.h:
            return False
        return self.tiles[r][c] in SOLID

    def _box_free(self, x, y, w, h) -> bool:
        for r in range(floor(y / TILE), floor((y + h - 0.01) / TILE) + 1):
            for c in range(floor(x / TILE), floor((x + w - 0.01) / TILE) + 1):
                if self.solid_at(c, r):
                    return False
        return True

    def _move_x(self, x, y, w, h, vx):
        x += vx
        r0, r1 = floor(y / TILE), floor((y + h - 0.01) / TILE)
        if vx > 0:
            c = floor((x + w - 0.01) / TILE)
            if any(self.solid_at(c, r) for r in range(r0, r1 + 1)):
                return c * TILE - w, True
        elif vx < 0:
            c = floor(x / TILE)
            if any(self.solid_at(c, r) for r in range(r0, r1 + 1)):
                return (c + 1) * TILE, True
        return x, False

    def _move_y(self, x, y, w, h, vy):
        """Returns (new y, 'floor' | 'ceil' | None, bumped tile or None)."""
        y += vy
        c0, c1 = floor(x / TILE), floor((x + w - 0.01) / TILE)
        if vy > 0:
            r = floor((y + h - 0.01) / TILE)
            if any(self.solid_at(c, r) for c in range(c0, c1 + 1)):
                return r * TILE - h, "floor", None
        elif vy < 0:
            r = floor(y / TILE)
            hits = [c for c in range(c0, c1 + 1) if self.solid_at(c, r)]
            if hits:
                cx = x + w / 2
                c = min(hits, key=lambda k: abs(k * TILE + TILE / 2 - cx))
                return (r + 1) * TILE, "ceil", (c, r)
        return y, None, None

    # ---------- stepping ----------
    def step_frame(self, mx: int, jump: int) -> bool:
        """One logic frame. jump: 0 none, 1 short, 2 long, 3 cut (key released). Returns True if a jump started."""
        self.frame += 1
        if self.done:
            self._animate_post()
            return False
        jumped = False
        self.vx = mx * RUN
        if mx:
            self.face = mx
        if jump in (1, 2) and self.on_ground:
            self.vy = -(JUMP_LONG if jump == 2 else JUMP_SHORT)
            self.on_ground = False
            jumped = True
            self.events.append(["jump"])
        elif jump == 3 and self.vy < -JUMP_CUT:
            self.vy = -JUMP_CUT

        prev_bottom = self.y + HERO_H
        self.x, _ = self._move_x(self.x, self.y, HERO_W, HERO_H, self.vx)
        self.vy = min(self.vy + GRAVITY, MAX_FALL)
        self.y, hit, bumped = self._move_y(self.x, self.y, HERO_W, HERO_H, self.vy)
        self.on_ground = hit == "floor"
        if hit:
            self.vy = 0.0
        if bumped:
            self._bump(*bumped)

        self._collect_coins()
        self._step_enemies(prev_bottom)

        if not self.done and self.y > self.h * TILE + 8:
            self._die("fell")
        if not self.done and self.x + HERO_W >= self.flag_col * TILE - 2:
            self.won = True
            self._finish("flag")
            self.events.append(["win"])
        if self.x > self.max_x + 0.01:
            self.max_x = self.x
            self.last_progress = self.frame
        if not self.done and self.stuck_frames and self.frame - self.last_progress >= self.stuck_frames:
            self._finish("stuck")
        if not self.done and self.frame >= MAX_FRAMES:
            self._finish("time")
        return jumped

    def step(self, action: int) -> None:
        """One agent decision: hold a move for FRAMES_PER_DECISION frames (a jump fires on the first grounded frame)."""
        _, mx, j = MOVES[action]
        jumped = False
        for _ in range(FRAMES_PER_DECISION):
            if self.done:
                break
            jumped |= self.step_frame(mx, 0 if jumped else j)

    def _bump(self, c, r):
        t = self.tiles[r][c]
        if t == "?":
            self.tiles[r][c] = "U"
            self.coins += 1
            self.events.append(["bump", c, r, 1])
        elif t == "B":
            self.events.append(["bump", c, r, 0])

    def _collect_coins(self):
        if not self.coin_tiles:
            return
        for r in range(floor(self.y / TILE), floor((self.y + HERO_H - 0.01) / TILE) + 1):
            for c in range(floor(self.x / TILE), floor((self.x + HERO_W - 0.01) / TILE) + 1):
                if (c, r) in self.coin_tiles:
                    self.coin_tiles.discard((c, r))
                    self.coins += 1
                    self.events.append(["coin", c, r])

    def _step_enemies(self, prev_bottom):
        for e in self.enemies:
            if e.gone:
                continue
            if not e.alive:
                e.squash += 1
                if e.squash > 30:
                    e.gone = True
                continue
            if not e.active:
                if abs(e.x - self.x) < ENEMY_WAKE:
                    e.active = True
                else:
                    continue
            e.x, hit = self._move_x(e.x, e.y, ENEMY_W, ENEMY_H, e.vx)
            if hit:
                e.vx = -e.vx
            e.vy = min(e.vy + GRAVITY, MAX_FALL)
            e.y, h, _ = self._move_y(e.x, e.y, ENEMY_W, ENEMY_H, e.vy)
            if h:
                e.vy = 0.0
            if e.y > self.h * TILE + 8:
                e.alive, e.gone = False, True
                continue
            if self.done:
                continue
            # generous hitboxes: enemy shrunk by 2 px on each side
            ex0, ey0, ex1, ey1 = e.x + 2, e.y + 2, e.x + ENEMY_W - 2, e.y + ENEMY_H
            if self.x < ex1 and self.x + HERO_W > ex0 and self.y < ey1 and self.y + HERO_H > ey0:
                if self.vy > 0 and prev_bottom <= ey0 + 6:
                    e.alive = False
                    self.stomps += 1
                    self.vy = -STOMP_BOUNCE
                    self.on_ground = False
                    self.events.append(["stomp", round(e.x, 1), round(e.y, 1)])
                else:
                    self._die("enemy")

    def _die(self, reason):
        self.dead = True
        self._finish(reason)
        self.vy = -5.0
        self.events.append(["die", reason])

    def _finish(self, reason):
        self.done = True
        self.reason = reason

    def _animate_post(self):
        """Death / win animation for the demo page; does not affect training (episode is already done)."""
        self.post += 1
        for e in self.enemies:
            if not e.alive and not e.gone:
                e.squash += 1
                if e.squash > 30:
                    e.gone = True
        if self.dead:
            if self.post > 20:
                self.vy = min(self.vy + GRAVITY, MAX_FALL)
                self.y += self.vy
        elif self.won:
            ground = self._ground_below(self.flag_col)
            if self.y + HERO_H < ground:
                self.y = min(self.y + 2.5, ground - HERO_H)
            elif self.post > 40:
                self.x += 1.0
                self.face = 1

    def _ground_below(self, c):
        for r in range(self.h):
            if self.solid_at(c, r):
                return r * TILE
        return self.h * TILE

    # ---------- views ----------
    @property
    def progress_tiles(self) -> float:
        return self.max_x / TILE

    def snapshot(self) -> dict:
        """Everything the page needs to draw one frame (small JSON)."""
        return {
            "f": self.frame,
            "m": [round(self.x, 2), round(self.y, 2), round(self.vx, 2), round(self.vy, 2), self.face,
                  int(self.on_ground), int(self.dead), int(self.won)],
            "e": [[round(e.x, 1), round(e.y, 1), (1 if e.vx > 0 else -1), (0 if e.alive else e.squash)]
                  for e in self.enemies if not e.gone and (e.active or abs(e.x - self.x) < 30 * TILE)],
            "ev": self.events,
            "hud": {"coins": self.coins, "stomps": self.stomps, "dist": round(self.max_x / TILE, 1),
                    "time": round(self.frame / 60, 1), "done": int(self.done), "reason": self.reason},
        }

    def pop_events(self) -> list:
        ev, self.events = self.events, []
        return ev
