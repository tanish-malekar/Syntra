import random

from mario.game.bot import bot_action
from mario.game.level import get_level
from mario.game.sim import HERO_H, TILE, Sim
from mario.game.solve import solve
from mario.game.vec import GameEnv, VecEnv, make_sim, replay_sim


def play(level, seed, moves):
    s = make_sim(level, seed)
    for m in moves:
        if s.done:
            break
        s.step(m)
        s.events.clear()
    return s


def state(s):
    return (s.x, s.y, s.vy, s.frame, s.coins, s.stomps, s.reason, [(e.x, e.y, e.alive) for e in s.enemies])


def test_same_seed_same_moves_same_result():
    rng = random.Random(1)
    moves = [rng.choice([0, 0, 0, 1, 2, 5]) for _ in range(300)]
    assert state(play("level1", 7, moves)) == state(play("level1", 7, moves))
    assert state(play("level1", 7, moves)) != state(play("level1", 8, moves))  # seeds do randomise


def test_level1_is_clearable_and_replay_reproduces():
    moves = solve("level1", 0)
    assert moves is not None
    s = play("level1", 0, [int(c) for c in moves])
    assert s.won and s.reason == "flag"
    rep = {"level": "level1", "seed": 0, "difficulty": 1, "actions": moves}
    assert replay_sim(rep).won


def test_generated_levels_clearable():
    for seed in range(3):
        assert solve(f"gen:{seed}:1", seed) is not None


def test_cannot_walk_through_a_pipe_and_ground_holds():
    s = Sim(get_level("level1"), randomize=False)
    y0 = s.y
    for _ in range(60):
        s.step(5)  # wait
    assert s.y == y0 and s.on_ground
    s.enemies.clear()
    for _ in range(80):
        s.step(0)  # run right into the first pipe (col 30)
    assert s.x + 12 <= 30 * TILE + 0.001
    assert s.reason == "stuck" or not s.done


def test_pit_kills_and_stuck_ends():
    s = Sim(get_level("level1"), randomize=False)
    s.enemies.clear()
    s.x = 56 * TILE
    for _ in range(40):
        s.step(0)
    assert s.dead and s.reason == "fell"
    s = Sim(get_level("level1"), randomize=False)
    for _ in range(40):
        s.step(5)
    assert s.done and s.reason == "stuck"


def test_stomp_and_enemy_contact():
    s = Sim(get_level("level1"), randomize=False)
    e = s.enemies[0]
    e.active, e.vx = True, 0.0
    s.x, s.y, s.vy = e.x, e.y - HERO_H - 20, 1.0
    s.on_ground = False
    for _ in range(30):
        s.step_frame(0, 0)
        if not e.alive:
            break
    assert not e.alive and s.stomps == 1 and not s.dead
    s = Sim(get_level("level1"), randomize=False)
    e = s.enemies[0]
    e.active = True
    s.x = e.x - 20
    for _ in range(30):
        s.step_frame(1, 0)
    assert s.dead and s.reason == "enemy"


def test_coin_block_bump():
    s = Sim(get_level("level1"), randomize=False)
    s.enemies.clear()
    s.x = 16 * TILE + 2  # under the first '?' block (row 9)
    s.step(4)            # jump in place
    for _ in range(10):
        s.step(5)
    assert s.tiles[9][16] == "U" and s.coins >= 1


def test_env_rewards_and_vec_autoreset():
    env = GameEnv(0, mode="level1")
    r, d = env.step(0)
    assert r > 0 and not d
    v = VecEnv(4, seed=1)
    finished = []
    for _ in range(400):
        _, _, f = v.step([bot_action(e.sim) for e in v.envs])
        finished += f
    assert finished and all("actions" in f for f in finished)
    for f in finished[:5]:  # every finished episode replays to the same result
        s = replay_sim(f)
        assert s.reason == f["reason"] and round(s.max_x / TILE, 1) == f["dist"]
