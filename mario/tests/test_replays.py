"""Every replay saved anywhere (bundled or fetched from context3) must reproduce exactly with this machine's sim."""
import glob
import json
import os

import pytest

from mario.game.sim import TILE
from mario.game.vec import replay_sim

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = [p for p in glob.glob(os.path.join(HERE, "replays", "*.json")) + glob.glob(os.path.join(HERE, "runs", "*", "replays", "*", "*.json"))
         if not p.endswith("summary.json")]


@pytest.mark.parametrize("path", FILES, ids=[os.path.relpath(p, HERE) for p in FILES])
def test_replay_reproduces(path):
    rep = json.load(open(path))
    s = replay_sim(rep)
    assert (int(s.won), s.reason, round(s.max_x / TILE, 1), s.coins, s.stomps) == \
        (rep["won"], rep["reason"], rep["dist"], rep["coins"], rep["stomps"])
