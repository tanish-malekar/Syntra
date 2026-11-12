from __future__ import annotations

import pytest

from app.services.planner import (
    RUNTIME_OVERHEAD_MB, Candidate, ModelSpec, capacity_layers, make_plan, order_candidates, replacement_candidate,
    split_layers, stage_memory_mb,
)

TINY = ModelSpec(num_layers=22, per_layer_memory_mb=100)


def cand(gid, free=24000, price=0.1, machine=0, reliability=1.0):
    return Candidate(gid, free, price, reliability, machine)


def test_memory_math():
    assert stage_memory_mb(TINY, 11) == 11 * 100 + RUNTIME_OVERHEAD_MB
    assert capacity_layers(TINY, 1500) == 10 and capacity_layers(TINY, 499) == 0 and capacity_layers(TINY, 500) == 0


@pytest.mark.parametrize(
    "caps,expected",
    [([22], [22]), ([30, 30], [11, 11]), ([30, 30, 30], [8, 7, 7]), ([150, 50], [17, 5]),
     ([5, 40], [2, 20]), ([10, 10, 10], [8, 7, 7])],
)
def test_split_layers_is_proportional_and_respects_capacity(caps, expected):
    got = split_layers(22, caps)
    assert got == expected and sum(got) == 22 and all(1 <= g <= c for g, c in zip(got, caps))


def test_split_layers_rejects_impossible_cases():
    assert split_layers(22, [10, 10]) is None      # total capacity too small
    assert split_layers(22, [30, 0]) is None       # a GPU that holds no layer
    assert split_layers(2, [5, 5, 5]) is None      # more stages than layers
    assert split_layers(22, []) is None


def test_split_always_covers_all_layers_contiguously():
    import random
    rng = random.Random(1)
    for _ in range(300):
        n = rng.randint(1, 6)
        caps = [rng.randint(1, 40) for _ in range(n)]
        got = split_layers(22, caps)
        if sum(caps) >= 22 and n <= 22:
            assert got is not None and sum(got) == 22 and all(1 <= g <= c for g, c in zip(got, caps)), (caps, got)
        else:
            assert got is None


def test_auto_plan_uses_the_fewest_gpus_that_fit():
    one = make_plan(TINY, [cand(1, 24000), cand(2, 24000)])
    assert [(s.gpu_id, s.layer_start, s.layer_end) for s in one.stages] == [(1, 0, 22)]
    small = [cand(i, 1500, machine=i) for i in (1, 2, 3, 4)]  # 10 layers each
    three = make_plan(TINY, small)
    assert len(three.stages) == 3 and sum(s.layers for s in three.stages) == 22
    assert make_plan(TINY, small[:2]) is None            # 20 layers of capacity: does not fit, so it must queue
    assert make_plan(TINY, []) is None


def test_requested_gpu_count_forces_that_many_stages():
    plan = make_plan(TINY, [cand(1), cand(2), cand(3)], requested_gpus=2)
    assert len(plan.stages) == 2 and [(s.layer_start, s.layer_end) for s in plan.stages] == [(0, 11), (11, 22)]
    assert make_plan(TINY, [cand(1)], requested_gpus=2) is None  # only one GPU idle: wait
    with pytest.raises(ValueError):
        make_plan(TINY, [cand(1)], requested_gpus=0)
    with pytest.raises(ValueError):
        make_plan(TINY, [cand(1)], requested_gpus=23)


def test_stages_are_contiguous_and_ordered():
    plan = make_plan(TINY, [cand(i, 3000, machine=i) for i in range(1, 6)], requested_gpus=5)
    edges = [(s.layer_start, s.layer_end) for s in plan.stages]
    assert edges[0][0] == 0 and edges[-1][1] == 22 and all(a[1] == b[0] for a, b in zip(edges, edges[1:]))


def test_selection_prefers_one_machine_then_price_then_reliability():
    a1, a2 = cand(1, price=0.45, machine=10), cand(2, price=0.45, machine=10)   # one workstation, 2 GPUs
    b1, c1 = cand(3, price=0.10, machine=20), cand(4, price=0.10, machine=30)   # two cheap laptops
    plan = make_plan(TINY, [b1, c1, a1, a2], requested_gpus=2)
    assert {s.gpu_id for s in plan.stages} == {1, 2} and plan.machines == 1      # fewer network hops beat price
    cheap = make_plan(TINY, [cand(5, price=0.45, machine=1), cand(6, price=0.10, machine=2)], requested_gpus=1)
    assert cheap.stages[0].gpu_id == 6                                           # otherwise the cheaper GPU
    flaky = make_plan(TINY, [cand(7, price=0.10, reliability=0.5), cand(8, price=0.45, reliability=1.0)], requested_gpus=1)
    assert flaky.stages[0].gpu_id == 8                                           # reliability outranks price
    assert [c.gpu_id for c in order_candidates([b1, a1, a2], 2)][:2] == [1, 2]


def test_hourly_rate_sums_the_class_prices():
    plan = make_plan(TINY, [cand(1, price=0.10, machine=1), cand(2, price=0.45, machine=2)], requested_gpus=2)
    assert plan.hourly_rate == pytest.approx(0.55)


def test_replacement_candidate_needs_enough_memory_and_picks_the_best():
    cands = [cand(1, free=600), cand(2, free=3000, price=0.45), cand(3, free=3000, price=0.10)]
    assert replacement_candidate(TINY, cands, layers=11).gpu_id == 3
    assert replacement_candidate(TINY, [cand(1, free=600)], layers=11) is None
    assert replacement_candidate(TINY, [], layers=11) is None
