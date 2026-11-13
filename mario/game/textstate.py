"""What the LLM sees: a short text description of the screen around the hero (about 100-150 tokens).

Example:
    x 37.4/201 vy +0.0 ground yes coins 3
    ............
    ......o.....
    ............
    ..........##
    ..M....E..##
    ############
    ############
    floor ######__####
    enemy +5.2,+0.0 left

Grid: 12 columns (2 behind the hero, 9 ahead) x 7 rows (4 above, 2 below).
'#' solid, '.' air, 'o' coin, 'E' enemy, 'F' flag, 'M' the hero.
'floor' marks, for each grid column, whether there is any ground below the hero's feet ('#') or a pit ('_').

Version 2 (TEXT_VERSION, used by every checkpoint trained after the death analysis) adds what the analysis showed
was missing: the grid always reaches down to the ground row (while high in the air the old 2 rows below the hero
hid the landing spot and the enemies on it), enemies are listed regardless of height, and the header gives the
hero's feet height `y` (in tiles), horizontal speed `vx` and `drop` (tiles to the ground right below, or `pit`):
    x 159.9/201 y 7.5 vx +2.0 vy -4.6 ground no drop 5.5
"""
from __future__ import annotations

from .sim import HERO_H, HERO_W, ENEMY_W, ENEMY_H, TILE, Sim, floor

COLS_BEHIND, COLS_AHEAD = 2, 9
ROWS_ABOVE, ROWS_BELOW = 4, 2
GROUND_ROW = 13
TEXT_VERSION = 2


def render_text(sim: Sim, version: int = TEXT_VERSION) -> str:
    cx, cy = sim.x + HERO_W / 2, sim.y + HERO_H / 2
    c, r = floor(cx / TILE), floor(cy / TILE)
    cols = range(c - COLS_BEHIND, c + COLS_AHEAD + 1)
    bottom = r + ROWS_BELOW if version < 2 else max(r + ROWS_BELOW, GROUND_ROW)
    rows = range(r - ROWS_ABOVE, bottom + 1)
    grid = {}
    for rr in rows:
        for cc in cols:
            if sim.solid_at(cc, rr):
                ch = "#"
            elif (cc, rr) in sim.coin_tiles:
                ch = "o"
            elif cc == sim.flag_col and rr < 13:
                ch = "F"
            else:
                ch = "."
            grid[cc, rr] = ch
    near = []
    for e in sim.enemies:
        if not e.alive or e.gone:
            continue
        ec, er = floor((e.x + ENEMY_W / 2) / TILE), floor((e.y + ENEMY_H / 2) / TILE)
        if (ec, er) in grid:
            grid[ec, er] = "E"
        dx = (e.x + ENEMY_W / 2 - cx) / TILE
        dy = (e.y + ENEMY_H / 2 - cy) / TILE
        if -COLS_BEHIND - 1 <= dx <= COLS_AHEAD + 1 and (abs(dy) <= 5 or version >= 2):
            near.append((abs(dx), dx, dy, e.vx))
    grid[c, r] = "M"

    feet = floor((sim.y + HERO_H - 0.01) / TILE)
    floor_line = "".join("#" if any(sim.solid_at(cc, rr) for rr in range(feet + 1, sim.h)) else "_" for cc in cols)

    if version >= 2:
        feet_px = sim.y + HERO_H
        land = next((rr for rr in range(max(0, floor(feet_px / TILE)), sim.h)
                     if any(sim.solid_at(k, rr) for k in (floor(sim.x / TILE), floor((sim.x + HERO_W - 0.01) / TILE)))),
                    None)
        drop = "pit" if land is None else f"{(land * TILE - feet_px) / TILE + 0.0:.1f}"
        lines = [f"x {sim.x / TILE:.1f}/{sim.flag_col} y {feet_px / TILE:.1f} vx {sim.vx + 0.0:+.1f} "
                 f"vy {-sim.vy + 0.0:+.1f} ground {'yes' if sim.on_ground else 'no'} drop {drop}"]
    else:
        lines = [f"x {sim.x / TILE:.1f}/{sim.flag_col} vy {-sim.vy + 0.0:+.1f} ground {'yes' if sim.on_ground else 'no'}"]
    lines += ["".join(grid[cc, rr] for cc in cols) for rr in rows]
    lines.append("floor " + floor_line)
    for _, dx, dy, vx in sorted(near)[:3]:
        lines.append(f"enemy {dx + 0.0:+.1f},{-dy + 0.0:+.1f} {'left' if vx < 0 else 'right'}")
    if not near:
        lines.append("enemy none")
    return "\n".join(lines)
