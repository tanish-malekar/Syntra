import pytest

torch = pytest.importorskip("torch")
from diloco_core import OuterOptimizer  # noqa: E402


def test_one_replica_without_momentum_lands_on_its_weights():
    start = {"w": torch.tensor([1.0, 2.0])}
    end = torch.tensor([0.5, 2.5])
    opt = OuterOptimizer(start, lr=1.0, momentum=0.0)
    opt.step([({"w": start["w"] - end}, 1.0)])
    assert torch.allclose(opt.params["w"], end)


def test_weighted_average_and_nesterov_momentum():
    opt = OuterOptimizer({"w": torch.zeros(1)}, lr=0.5, momentum=0.9)
    norm = opt.step([({"w": torch.tensor([1.0])}, 3.0), ({"w": torch.tensor([3.0])}, 1.0)])  # avg = 1.5
    assert norm == pytest.approx(1.5)
    # buf = 1.5; update = 1.5 + 0.9 * 1.5 = 2.85; w = -0.5 * 2.85
    assert float(opt.params["w"]) == pytest.approx(-1.425)
    opt.step([({"w": torch.tensor([1.5])}, 1.0)])  # buf = 0.9*1.5+1.5 = 2.85; update = 1.5 + 0.9*2.85 = 4.065
    assert float(opt.params["w"]) == pytest.approx(-1.425 - 0.5 * 4.065)


def test_check_rejects_bad_updates():
    opt = OuterOptimizer({"w": torch.zeros(2)})
    assert opt.check({"w": torch.zeros(2)}) is None
    assert "shape" in opt.check({"w": torch.zeros(3)})
    assert "names" in opt.check({"v": torch.zeros(2)})
    assert "non-finite" in opt.check({"w": torch.tensor([0.0, float("nan")])})
    with pytest.raises(ValueError):
        opt.step([])
