"""A deliberately simple scripted player: run right, long-jump when a wall, a pit or an enemy is close.

It is only used for the imitation warm-up (so the LLM does not start from "dies at the first enemy forever").
It is not good: it never waits for enemies and misjudges tall walls; its clear rate is measured and reported.
"""
from __future__ import annotations

import random

from .sim import ENEMY_W, HERO_H, HERO_W, TILE, Sim, floor

RUN, LONG, HOP, LEFT, UP, WAIT = range(6)


def bot_action(sim: Sim) -> int:
    if not sim.on_ground:
        return RUN
    front = floor((sim.x + HERO_W + 1) / TILE)
    feet = floor((sim.y + HERO_H - 0.01) / TILE)
    head = floor(sim.y / TILE)
    for cc in (front, front + 1):
        if any(sim.solid_at(cc, rr) for rr in range(head, feet + 1)):
            return LONG
    for cc in (front, front + 1):
        if not any(sim.solid_at(cc, rr) for rr in range(feet + 1, sim.h)):
            return LONG
    for e in sim.enemies:
        if not e.alive or e.gone:
            continue
        dx = (e.x - (sim.x + HERO_W)) / TILE
        if -0.5 < dx < 2.6 and abs(e.y - sim.y) < TILE:
            return LONG
    return RUN


class NoisyBot:
    """The bot with epsilon-random moves (used to make warm-up data more varied)."""

    def __init__(self, eps: float = 0.1, seed: int = 0):
        self.eps = eps
        self.rng = random.Random(seed)

    def __call__(self, sim: Sim) -> int:
        if self.rng.random() < self.eps:
            return self.rng.randrange(6)
        return bot_action(sim)
