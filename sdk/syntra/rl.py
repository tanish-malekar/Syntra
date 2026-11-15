"""Ready-made training loops on a TrainingSession. Your reward function runs here, on your machine.

    history = syntra.rl.grpo(tc, prompts, reward_fn, iterations=20, group_size=8)
    history = syntra.rl.dpo(tc, pairs, beta=0.1, epochs=2)

reward_fn(prompt: str, completion: str, sample: Sample) -> float. Must return a finite number.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Sequence

from syntra.client import BadRequest, Sample, TrainingSession

RewardFn = Callable[[str, str, Sample], float]


def group_advantages(rewards: Sequence[float], group_size: int, normalize_std: bool = True, eps: float = 1e-6):
    """GRPO advantages: (r - group mean) / group std, per consecutive group. Returns (advantages, informative flags)."""
    if group_size < 1 or len(rewards) % group_size:
        raise BadRequest(f"{len(rewards)} rewards do not split into groups of {group_size}")
    adv, informative = [], []
    for i in range(0, len(rewards), group_size):
        g = [float(r) for r in rewards[i:i + group_size]]
        mean = sum(g) / len(g)
        std = math.sqrt(sum((x - mean) ** 2 for x in g) / len(g))
        adv += [(x - mean) / (std + eps) if normalize_std else x - mean for x in g]
        informative += [std > eps] * len(g)
    return adv, informative


def _rewards(reward_fn: RewardFn, prompts: Sequence[str], samples: list[Sample]) -> list[float]:
    out = []
    for s in samples:
        r = reward_fn(prompts[s.prompt_index], s.text, s)
        try:
            r = float(r)
        except (TypeError, ValueError):
            raise BadRequest(f"reward_fn must return a number, got {r!r}")
        if not math.isfinite(r):
            raise BadRequest(f"reward_fn returned {r} for {s.text[:60]!r}")
        out.append(r)
    return out


def _policy_step(tc: TrainingSession, samples: list[Sample], advantages: list[float], keep: list[bool], *, lr: float,
                 clip: float, kl_coef: float, epochs: int, minibatch: int, loss: str) -> dict:
    rows = [{"tokens": s.tokens, "prompt_len": s.prompt_len, "advantage": a, "old_logprobs": s.logprobs}
            for s, a, k in zip(samples, advantages, keep) if k and len(s.completion) > 0]
    if not rows:
        return {"skipped": True}
    last = {}
    for _ in range(epochs):
        for i in range(0, len(rows), minibatch):
            chunk = rows[i:i + minibatch]
            args = {"clip": clip, "kl_coef": kl_coef} if loss == "ppo" else {}
            last = tc.forward_backward(chunk, loss=loss, weight=len(chunk) / len(rows), **args)
        last.update(tc.optim_step(lr=lr))
    return last


def grpo(tc: TrainingSession, prompts: Sequence[str], reward_fn: RewardFn, *, iterations: int = 10, group_size: int = 8,
         prompts_per_iteration: int = 4, max_new_tokens: int = 64, temperature: float = 1.0, top_p: float = 1.0, lr: float = 1e-5,
         clip: float = 0.2, kl_coef: float = 0.0, epochs: int = 1, minibatch: int = 32, seed: int | None = None,
         on_iteration: Callable[[dict], None] | None = None) -> list[dict]:
    """Group Relative Policy Optimization: sample `group_size` completions per prompt, score them with reward_fn, use the
    reward relative to the group as the advantage, and take PPO-clip steps (optionally with a KL penalty to the reference).
    Groups where every completion got the same reward carry no signal and are skipped."""
    if not prompts:
        raise BadRequest("no prompts")
    rng = random.Random(seed)
    history = []
    for it in range(1, iterations + 1):
        batch = rng.sample(list(prompts), min(prompts_per_iteration, len(prompts)))
        samples = tc.sample(batch, n=group_size, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                            seed=None if seed is None else seed + it)
        rewards = _rewards(reward_fn, batch, samples)
        adv, informative = group_advantages(rewards, group_size)
        step = _policy_step(tc, samples, adv, informative, lr=lr, clip=clip, kl_coef=kl_coef, epochs=epochs,
                            minibatch=minibatch, loss="ppo")
        entry = {"iteration": it, "reward_mean": sum(rewards) / len(rewards), "reward_max": max(rewards),
                 "informative_fraction": sum(informative) / len(informative), "skipped": bool(step.get("skipped")),
                 **{k: step[k] for k in ("loss", "grad_norm", "step") if k in step}, **step.get("metrics", {})}
        history.append(entry)
        if on_iteration:
            on_iteration(entry)
    return history


def ppo(tc: TrainingSession, prompts: Sequence[str], reward_fn: RewardFn, *, iterations: int = 10, samples_per_prompt: int = 4,
        prompts_per_iteration: int = 8, max_new_tokens: int = 64, temperature: float = 1.0, lr: float = 1e-5, clip: float = 0.2,
        kl_coef: float = 0.02, baseline_decay: float = 0.9, epochs: int = 2, minibatch: int = 32, seed: int | None = None,
        on_iteration: Callable[[dict], None] | None = None) -> list[dict]:
    """PPO with a moving-average reward baseline (no value network): advantage = (reward - baseline), normalized over the
    batch; several clipped epochs over each batch; KL penalty to the reference policy."""
    rng = random.Random(seed)
    baseline = None
    history = []
    for it in range(1, iterations + 1):
        batch = rng.sample(list(prompts), min(prompts_per_iteration, len(prompts)))
        samples = tc.sample(batch, n=samples_per_prompt, max_new_tokens=max_new_tokens, temperature=temperature,
                            seed=None if seed is None else seed + it)
        rewards = _rewards(reward_fn, batch, samples)
        mean = sum(rewards) / len(rewards)
        baseline = mean if baseline is None else baseline_decay * baseline + (1 - baseline_decay) * mean
        centered = [r - baseline for r in rewards]
        std = math.sqrt(sum(c * c for c in centered) / len(centered)) or 1.0
        adv = [c / std for c in centered]
        step = _policy_step(tc, samples, adv, [True] * len(adv), lr=lr, clip=clip, kl_coef=kl_coef, epochs=epochs,
                            minibatch=minibatch, loss="ppo")
        entry = {"iteration": it, "reward_mean": mean, "baseline": baseline,
                 **{k: step[k] for k in ("loss", "grad_norm", "step") if k in step}, **step.get("metrics", {})}
        history.append(entry)
        if on_iteration:
            on_iteration(entry)
    return history


def dpo(tc: TrainingSession, pairs: Sequence[dict], *, beta: float = 0.1, epochs: int = 1, batch_size: int = 8, lr: float = 5e-6,
        seed: int | None = None, on_step: Callable[[dict], None] | None = None) -> list[dict]:
    """Direct Preference Optimization on {"prompt", "chosen", "rejected"} text pairs. Reference log-probs come from the
    session's starting adapter and are computed once, before any update."""
    if not pairs:
        raise BadRequest("no preference pairs")
    prompts = tc.tokenize([p["prompt"] for p in pairs])
    chosen = tc.tokenize([p["chosen"] for p in pairs], add_special_tokens=False)
    rejected = tc.tokenize([p["rejected"] for p in pairs], add_special_tokens=False)
    rows = []
    for p, c, r in zip(prompts, chosen, rejected):
        if not c or not r:
            raise BadRequest("chosen and rejected must not be empty")
        rows.append({"chosen": {"tokens": p + c, "prompt_len": len(p)}, "rejected": {"tokens": p + r, "prompt_len": len(p)}})
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        seqs = [x["chosen"] for x in chunk] + [x["rejected"] for x in chunk]
        lps = [sum(l) for l in tc.logprobs(seqs, reference=True)]
        for j, x in enumerate(chunk):
            x["ref_chosen"], x["ref_rejected"] = lps[j], lps[len(chunk) + j]
    rng = random.Random(seed)
    history = []
    for epoch in range(1, epochs + 1):
        order = rows[:]
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            out = tc.forward_backward(order[i:i + batch_size], loss="dpo", beta=beta)
            out.update(tc.optim_step(lr=lr))
            entry = {"epoch": epoch, "loss": out["loss"], **out.get("metrics", {}), "step": out["step"]}
            history.append(entry)
            if on_step:
                on_step(entry)
    return history
