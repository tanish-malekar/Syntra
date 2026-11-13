from mario.game.level import get_level
from mario.game.sim import TILE, Sim
from mario.game.textstate import render_text


def grid(text):
    lines = text.split("\n")
    return lines[0], lines[1:8], lines[8], lines[9:]


def test_shape_and_hero():
    s = Sim(get_level("level1"), randomize=False)
    head, rows, floor, enemies = grid(render_text(s))
    assert head.startswith("x ") and "ground yes" in head
    assert len(rows) == 7 and all(len(r) == 12 for r in rows)
    assert rows[4][2] == "M"
    assert rows[5] == "#" * 12  # ground right under the hero
    assert floor == "floor " + "#" * 12


def test_matches_sim_enemy_coin_pit():
    s = Sim(get_level("level1"), randomize=False)
    e = s.enemies[0]
    s.x = e.x - 5 * TILE
    head, rows, floor, enemies = grid(render_text(s))
    assert "E" in rows[4] and rows[4].index("E") - 2 in (4, 5, 6)
    assert enemies[0].startswith("enemy +") and enemies[0].endswith("left")
    s = Sim(get_level("level1"), randomize=False)
    s.enemies.clear()
    s.x = 55 * TILE + 2  # gap at columns 58-59
    head, rows, floor, enemies = grid(render_text(s))
    assert floor == "floor " + "#####__#####"
    assert enemies == ["enemy none"]
    s.x = 8 * TILE + 2  # coins at row 9, columns 9-11
    head, rows, floor, enemies = grid(render_text(s))
    assert "o" in rows[1]


def test_flag_visible_near_end():
    s = Sim(get_level("level1"), randomize=False)
    s.enemies.clear()
    s.x = 195 * TILE
    _, rows, _, _ = grid(render_text(s))
    assert any("F" in r for r in rows)
