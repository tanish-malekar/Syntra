"""Levels: the hand-made Level 1 plus seeded random variants used during training.

Tile characters: '#' ground, 'B' brick, '?' coin block, 'U' used block, 'S' stone/stairs, 'P' pipe (all solid);
coins, enemy spawns and the flag are stored next to the grid.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

H = 15
GROUND = 13          # ground occupies rows 13-14; the hero stands in row 12
STAND = GROUND - 1


@dataclass
class Level:
    name: str
    tiles: list[str]
    coins: list[tuple[int, int]] = field(default_factory=list)
    enemies: list[tuple[int, int]] = field(default_factory=list)
    flag_col: int = 0
    start_col: int = 2
    start_row: int = STAND

    @property
    def w(self) -> int:
        return len(self.tiles[0])

    @property
    def h(self) -> int:
        return len(self.tiles)

    def to_json(self) -> dict:
        return {"name": self.name, "w": self.w, "h": self.h, "tiles": self.tiles, "coins": self.coins,
                "enemies": self.enemies, "flag": self.flag_col}


class Builder:
    def __init__(self, width: int):
        self.g = [["."] * width for _ in range(H)]
        self.coins: set[tuple[int, int]] = set()
        self.enemies: list[tuple[int, int]] = []
        for c in range(width):
            self.g[GROUND][c] = self.g[GROUND + 1][c] = "#"

    def gap(self, c0, c1):
        for c in range(c0, c1 + 1):
            for r in range(H):
                self.g[r][c] = "."

    def pipe(self, c, height):
        for r in range(STAND - height + 1, STAND + 1):
            self.g[r][c] = self.g[r][c + 1] = "P"

    def col(self, c, height, ch="S"):
        for r in range(STAND - height + 1, STAND + 1):
            self.g[r][c] = ch

    def stairs(self, c, heights):
        for i, hh in enumerate(heights):
            self.col(c + i, hh)

    def blocks(self, c0, row, pattern):
        for i, ch in enumerate(pattern):
            if ch != " ":
                self.g[row][c0 + i] = ch

    def coin_row(self, c0, c1, row):
        for c in range(c0, c1 + 1):
            self.coins.add((c, row))

    def enemy(self, c, row=STAND):
        self.enemies.append((c, row))

    def build(self, name, flag_col) -> Level:
        # flag base block
        self.g[STAND][flag_col] = "S"
        return Level(name=name, tiles=["".join(r) for r in self.g], coins=sorted(self.coins),
                     enemies=list(self.enemies), flag_col=flag_col)


def level1() -> Level:
    b = Builder(212)
    # opening: coins and the first blocks
    b.coin_row(9, 11, 9)
    b.blocks(16, 9, "?   B?B?B")
    b.blocks(22, 5, "?")
    b.enemy(23)
    # pipes with enemies between them
    b.pipe(30, 2)
    b.enemy(35)
    b.pipe(40, 3)
    b.enemy(45)
    b.enemy(47)
    b.pipe(50, 3)
    b.coin_row(50, 51, 8)
    # first gap
    b.gap(58, 59)
    b.coin_row(57, 60, 9)
    # floating bricks over a pair of walkers
    b.blocks(64, 9, "BB?BB")
    b.coin_row(64, 68, 8)
    b.enemy(68)
    b.enemy(71)
    b.gap(77, 78)
    # stairs up, a gap between two towers, stairs down
    b.stairs(84, [1, 2, 3, 4])
    b.gap(88, 89)
    b.stairs(90, [4, 3, 2, 1])
    b.coin_row(87, 90, 7)
    # low brick ceiling, then walkers
    b.blocks(99, 8, "B?B?B")
    b.enemy(106)
    b.enemy(108)
    b.pipe(114, 2)
    b.enemy(118)
    b.pipe(122, 3)
    # wide gap
    b.gap(128, 130)
    b.coin_row(127, 131, 9)
    # coin blocks and a group of three walkers
    b.blocks(137, 9, "? ? ?")
    b.coin_row(137, 141, 5)
    b.enemy(144)
    b.enemy(146)
    b.enemy(149)
    # second tower section
    b.stairs(154, [1, 2, 3, 4, 4, 4])
    b.gap(160, 161)
    b.stairs(162, [4, 4, 3, 2, 1])
    b.enemy(170)
    b.enemy(174)
    b.blocks(171, 9, "BBB")
    b.coin_row(171, 173, 8)
    # final staircase and the flag
    b.stairs(183, [1, 2, 3, 4, 5, 6, 7, 8, 8])
    b.coin_row(194, 198, 11)
    return b.build("level1", flag_col=201)


def generate(seed: int, difficulty: int = 1) -> Level:
    """A random level built from the same pieces as Level 1 (always clearable by construction)."""
    rng = random.Random(seed * 7919 + difficulty)
    width = 180
    b = Builder(width + 12)
    c = 14
    max_gap = 2 if difficulty == 0 else 3
    while c < width - 16:
        kind = rng.choices(["flat", "gap", "pipe", "enemies", "stairs", "blocks", "tower"],
                           weights=[2, 3, 3, 4, 1, 2, 1])[0]
        if kind == "flat":
            if rng.random() < 0.5:
                b.coin_row(c + 1, c + 3, rng.choice([9, 10, 11]))
            c += rng.randint(3, 6)
        elif kind == "gap":
            g = rng.randint(1, max_gap)
            b.gap(c, c + g - 1)
            if rng.random() < 0.4:
                b.coin_row(c - 1, c + g, 9)
            c += g + rng.randint(3, 5)
        elif kind == "pipe":
            b.pipe(c, rng.randint(1, 3))
            c += 2 + rng.randint(3, 6)
        elif kind == "enemies":
            n = rng.randint(1, 3 if difficulty else 2)
            for i in range(n):
                b.enemy(c + 1 + 2 * i)
            c += 2 * n + rng.randint(4, 7)
        elif kind == "stairs":
            n = rng.randint(2, 4)
            b.stairs(c, list(range(1, n + 1)))
            if rng.random() < 0.5:
                b.stairs(c + n, list(range(n, 0, -1)))
                c += 2 * n
            else:
                c += n
            c += rng.randint(3, 5)
        elif kind == "blocks":
            pat = "".join(rng.choice("BB?") for _ in range(rng.randint(3, 6)))
            b.blocks(c, rng.choice([8, 9]), pat)
            if rng.random() < 0.5:
                b.enemy(c + len(pat) // 2)
            c += len(pat) + rng.randint(3, 5)
        elif kind == "tower":
            n = rng.randint(2, 4)
            g = rng.randint(1, 2)
            b.stairs(c, list(range(1, n + 1)))
            b.gap(c + n, c + n + g - 1)
            b.stairs(c + n + g, list(range(n, 0, -1)))
            c += 2 * n + g + rng.randint(3, 5)
    return b.build(f"gen:{seed}:{difficulty}", flag_col=width - 4)


_LEVEL1 = None


def get_level(name: str) -> Level:
    """'level1' or 'gen:<seed>:<difficulty>'."""
    global _LEVEL1
    if name == "level1":
        if _LEVEL1 is None:
            _LEVEL1 = level1()
        return _LEVEL1
    if name.startswith("gen:"):
        _, seed, diff = name.split(":")
        return generate(int(seed), int(diff))
    raise KeyError(name)
