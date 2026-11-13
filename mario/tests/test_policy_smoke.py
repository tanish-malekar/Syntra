"""Policy + PPO smoke test with a tiny random Qwen2 (needs torch/transformers/peft: run with mario/.venv)."""
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from mario.game.vec import VecEnv  # noqa: E402
from mario.train.evaluate import run_episodes  # noqa: E402
from mario.train.policy import DEFAULT_MODEL, Policy  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def pol():
    return Policy(DEFAULT_MODEL, tiny=True, device="cpu")


def test_policy_picks_only_valid_moves(pol):
    v = VecEnv(3, seed=0)
    logits, value = pol.act_texts(v.texts())
    assert logits.shape == (3, 6) and value.shape == (3,)
    acts = torch.distributions.Categorical(logits=logits).sample().tolist()
    assert all(0 <= x < 6 for x in acts)
    v.step(acts)


def test_left_padding_does_not_change_scores(pol):
    v = VecEnv(2, seed=3)
    for _ in range(30):
        v.step([0, 1])
    t = v.texts()
    both, _ = pol.act_texts(t)
    one, _ = pol.act_texts([t[0]])
    assert torch.allclose(both[0], one[0], atol=1e-4)


def test_eval_episode_finishes(pol):
    reps = run_episodes(pol, [("level1", 1_000_000)], batch=1)
    assert reps[0]["decisions"] > 0 and len(reps[0]["probs"]) == reps[0]["decisions"]


def test_two_ppo_iterations_run(tmp_path):
    cmd = [sys.executable, "-m", "mario.train.ppo", "--tiny", "--run-dir", str(tmp_path / "run"), "--iters", "2",
           "--envs", "4", "--steps", "8", "--mb", "8", "--eval-every", "1", "--eval-episodes", "2", "--save-every", "1"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900, env={**os.environ, "PYTHONPATH": ROOT})
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "run" / "metrics.jsonl").read_text().count("\n") == 3
    assert (tmp_path / "run" / "ckpt" / "final" / "adapter_config.json").exists()
    # the saved adapter loads again and continues
    p = Policy(DEFAULT_MODEL, adapter=str(tmp_path / "run" / "ckpt" / "final"), tiny=True, device="cpu")
    assert p.act_texts(VecEnv(1).texts())[0].shape == (1, 6)
