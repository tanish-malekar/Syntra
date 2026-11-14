from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from rl_engine import Engine, EngineError  # noqa: E402
from rl_losses import (dpo_loss, group_advantages, importance_sampling_loss, ppo_clip_loss, sample_next,  # noqa: E402
                       token_logprobs)


# ----------------------------------------------------------------------------------------------- losses
def test_dpo_loss_matches_the_formula():
    pc, pr, rc, rr = torch.tensor([-1.0, -2.0]), torch.tensor([-3.0, -1.0]), torch.tensor([-2.0, -2.0]), torch.tensor([-2.0, -2.0])
    loss, m = dpo_loss(pc, pr, rc, rr, beta=0.5)
    margins = 0.5 * ((pc - rc) - (pr - rr))
    assert float(loss) == pytest.approx(float(-torch.nn.functional.logsigmoid(margins).mean()))
    assert m["reward_accuracy"] == 0.5 and m["reward_margin"] == pytest.approx(float(margins.mean()))
    with pytest.raises(ValueError):
        dpo_loss(pc, pr, rc, rr, beta=0)


def test_ppo_clip_and_importance_sampling():
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    same = torch.zeros(1, 3)
    loss, m = ppo_clip_loss(same, same, torch.tensor([2.0]), mask)
    assert float(loss) == pytest.approx(-2.0) and m["clip_fraction"] == 0 and m["ratio_mean"] == pytest.approx(1.0)
    # ratio e^1 > 1.2 with positive advantage: clipped at 1.2
    loss, m = ppo_clip_loss(torch.ones(1, 3), same, torch.tensor([1.0]), mask, clip=0.2)
    assert float(loss) == pytest.approx(-1.2) and m["clip_fraction"] == 1.0
    # negative advantage: min picks the unclipped (more pessimistic) term
    loss, _ = ppo_clip_loss(torch.ones(1, 3), same, torch.tensor([-1.0]), mask, clip=0.2)
    assert float(loss) == pytest.approx(math.e)
    # KL penalty is >= 0 and zero when the policy equals the reference
    _, m = ppo_clip_loss(same, same, torch.tensor([1.0]), mask, ref_logp=same, kl_coef=0.1)
    assert m["kl"] == pytest.approx(0.0)
    loss, _ = importance_sampling_loss(same, same, torch.tensor([3.0]), mask)
    assert float(loss) == pytest.approx(-3.0)


def test_group_advantages():
    adv, info = group_advantages([1, 0, 1, 0, 2, 2, 2, 2], group_size=4)
    assert adv[:4] == pytest.approx([1, -1, 1, -1], abs=1e-4) and adv[4:] == [0, 0, 0, 0]
    assert info == [True] * 4 + [False] * 4
    with pytest.raises(ValueError):
        group_advantages([1, 2, 3], 2)
    with pytest.raises(ValueError):
        group_advantages([1.0, float("nan")], 2)


def test_sample_next_greedy_topk_topp_and_logprob():
    logits = torch.tensor([[0.0, 3.0, 1.0, 2.0]])
    full = torch.log_softmax(logits, -1)
    for kwargs in ({"temperature": 0}, {"top_k": 1}, {"top_p": 0.01}):
        tok, lp = sample_next(logits, **kwargs)
        assert int(tok) == 1 and float(lp) == pytest.approx(float(full[0, 1]))
    g = torch.Generator().manual_seed(0)
    seen = {int(sample_next(logits, top_k=2, generator=g)[0]) for _ in range(200)}
    assert seen == {1, 3}


# ----------------------------------------------------------------------------------------------- engine on a tiny model
class Tok:
    """Word-level tokenizer over w0..w31; 1 = eos, 0 = pad."""
    pad_token_id, eos_token_id, vocab_size = 0, 1, 32

    def __len__(self):
        return 32

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [int(w[1:]) for w in text.split()]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{i}" for i in ids if not (skip_special_tokens and i in (0, 1)))


def make_engine(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128)).eval()
    base.requires_grad_(False)
    bias = torch.nn.Parameter(torch.zeros(32))  # stands in for the LoRA adapter: the only trainable weights
    model = lambda ids: base(input_ids=ids).logits + bias
    saved = {}
    return Engine(model, {"bias": bias}, Tok(), "cpu", save_fn=lambda p: saved.setdefault("p", p), max_seq_len=64, max_lr=1.0), bias


def test_sampled_logprobs_equal_a_fresh_logprobs_pass_and_seed_repeats():
    eng, _ = make_engine()
    a = eng.sample(["w5 w6"], n=4, max_new_tokens=6, temperature=1.0, seed=7)["samples"]
    b = eng.sample(["w5 w6"], n=4, max_new_tokens=6, temperature=1.0, seed=7)["samples"]
    assert [s["tokens"] for s in a] == [s["tokens"] for s in b]
    again = eng.logprobs([{"tokens": s["tokens"], "prompt_len": s["prompt_len"]} for s in a])["logprobs"]
    for s, lp in zip(a, again):
        assert len(s["logprobs"]) == len(s["completion"]) == len(lp)
        assert s["logprobs"] == pytest.approx(lp, abs=1e-4)
        if s["finished"]:
            assert s["completion"][-1] == 1


def test_grpo_loop_raises_the_reward_and_the_reference_stays_fixed():
    eng, bias = make_engine()
    target = 7  # reward: fraction of completion tokens equal to w7
    reward = lambda s: sum(t == target for t in s["completion"]) / max(1, len(s["completion"]))
    start = eng.sample(["w5"], n=16, max_new_tokens=5, seed=1, stop_at_eos=False)["samples"]
    first = sum(map(reward, start)) / len(start)
    ref_before = eng.logprobs([{"tokens": start[0]["tokens"], "prompt_len": 1}], reference=True)["logprobs"]
    for it in range(15):
        samples = eng.sample(["w5", "w9"], n=8, max_new_tokens=5, seed=100 + it, stop_at_eos=False)["samples"]
        adv = eng.advantages([reward(s) for s in samples], group_size=8)["advantages"]
        batch = [{"tokens": s["tokens"], "prompt_len": s["prompt_len"], "advantage": a, "old_logprobs": s["logprobs"]}
                 for s, a in zip(samples, adv)]
        out = eng.forward_backward("ppo", batch, clip=0.2, kl_coef=0.01)
        assert math.isfinite(out["loss"]) and "kl" in out["metrics"]
        eng.optim_step(lr=0.5)
    end = eng.sample(["w5"], n=16, max_new_tokens=5, seed=1, stop_at_eos=False)["samples"]
    assert sum(map(reward, end)) / len(end) > first + 0.3
    assert eng.logprobs([{"tokens": start[0]["tokens"], "prompt_len": 1}], reference=True)["logprobs"][0] == pytest.approx(ref_before[0])
    assert float(bias[target]) > 0 and eng.state()["step"] == 15


def test_dpo_prefers_chosen_over_rejected():
    eng, _ = make_engine()
    pairs = [{"chosen": {"tokens": [5, 7, 7], "prompt_len": 1}, "rejected": {"tokens": [5, 9, 9], "prompt_len": 1}}] * 4
    refs = eng.reference_sums(pairs)
    batch = [{**p, "ref_chosen": c, "ref_rejected": r} for p, (c, r) in zip(pairs, refs)]
    first = eng.forward_backward("dpo", batch, beta=0.5)
    eng.optim_step(lr=0.05)
    for _ in range(10):
        last = eng.forward_backward("dpo", batch, beta=0.5)
        eng.optim_step(lr=0.05)
    assert first["metrics"]["reward_margin"] == pytest.approx(0.0, abs=1e-6)  # policy == reference at the start
    assert last["metrics"]["reward_margin"] > 0.5 and last["metrics"]["reward_accuracy"] == 1.0 and last["loss"] < first["loss"]
    # without precomputed reference sums the engine computes them itself, with the same result
    auto = eng.forward_backward("dpo", pairs, beta=0.5)
    assert auto["metrics"]["reward_margin"] == pytest.approx(eng.forward_backward("dpo", batch, beta=0.5)["metrics"]["reward_margin"])


def test_cross_entropy_and_bad_requests():
    eng, _ = make_engine()
    out = eng.forward_backward("cross_entropy", [{"tokens": [5, 6, 7], "prompt_len": 1}])
    assert out["loss"] > 0 and out["pending_backward"] == 1
    eng.optim_step(lr=1e-3)
    bad = [
        ("forward_backward", {"loss": "nope", "batch": [{"tokens": [1, 2], "prompt_len": 1}]}),
        ("forward_backward", {"loss": "ppo", "batch": [{"tokens": [1, 2], "prompt_len": 1}]}),  # no advantage
        ("forward_backward", {"loss": "ppo", "batch": [{"tokens": [1, 2], "prompt_len": 1, "advantage": float("inf")}]}),
        ("forward_backward", {"loss": "cross_entropy", "batch": [{"tokens": [1, 2], "prompt_len": 2}]}),  # no completion
        ("forward_backward", {"loss": "cross_entropy", "batch": [{"tokens": list(range(2, 30)) * 3, "prompt_len": 1}]}),  # too long
        ("forward_backward", {"loss": "ppo", "batch": [{"tokens": [5, 6, 7], "prompt_len": 1, "advantage": 1.0,
                                                         "old_logprobs": [0.0]}]}),  # wrong length
        ("forward_backward", {"loss": "cross_entropy", "batch": [{"tokens": [5, 99], "prompt_len": 1}]}),  # vocab
        ("optim_step", {"lr": 1e-3}),  # nothing accumulated
        ("optim_step", {"lr": 5.0}),
        ("sample", {"prompts": ["w5"], "n": 0}),
        ("advantages", {"rewards": [1, 2, 3], "group_size": 2}),
        ("bogus", {}),
        ("sample", {"prompts": ["w5"], "unknown_arg": 1}),
    ]
    for op, args in bad:
        with pytest.raises(EngineError):
            eng.call(op, args)
    assert eng.state()["step"] == 1  # nothing above changed the model
