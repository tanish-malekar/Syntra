"""Losses for RL and preference tuning on LoRA adapters (pure PyTorch, no BloomBee: unit-tested on their own).

Conventions: a batch row is a full token sequence (prompt + completion), right-padded. `mask` marks the completion tokens that
count (1) and everything else (prompt, padding: 0). Log-probabilities are per token: `logp[b, t]` = log p(token[t+1] | ...<=t),
so they line up with `tokens[:, 1:]` and with `mask[:, 1:]`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def token_logprobs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """[B, T, V] logits, [B, T] tokens -> [B, T-1] log p(tokens[:, 1:]). Computed in float32."""
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    return logp.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)


def completion_mask(lengths: list[int], prompt_lengths: list[int], width: int, device=None) -> torch.Tensor:
    """[B, width] mask over token positions: 1 on completion tokens (prompt_len <= t < length)."""
    pos = torch.arange(width, device=device)[None, :]
    lo = torch.tensor(prompt_lengths, device=device)[:, None]
    hi = torch.tensor(lengths, device=device)[:, None]
    return ((pos >= lo) & (pos < hi)).float()


def masked_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum(-1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1)


# --------------------------------------------------------------------------------------------------- DPO
def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor, ref_chosen: torch.Tensor, ref_rejected: torch.Tensor,
             beta: float = 0.1, label_smoothing: float = 0.0) -> tuple[torch.Tensor, dict]:
    """Direct Preference Optimization (Rafailov et al. 2023). Inputs are summed completion log-probs per pair, shape [B].

    loss = -log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))   (with optional conservative label smoothing)
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    chosen_reward = beta * (policy_chosen - ref_chosen)
    rejected_reward = beta * (policy_rejected - ref_rejected)
    margin = chosen_reward - rejected_reward
    loss = -(1 - label_smoothing) * F.logsigmoid(margin) - label_smoothing * F.logsigmoid(-margin)
    with torch.no_grad():
        metrics = {"reward_chosen": float(chosen_reward.mean()), "reward_rejected": float(rejected_reward.mean()),
                   "reward_margin": float(margin.mean()), "reward_accuracy": float((margin > 0).float().mean())}
    return loss.mean(), metrics


# --------------------------------------------------------------------------------------------------- policy gradient
def ppo_clip_loss(logp: torch.Tensor, old_logp: torch.Tensor, advantages: torch.Tensor, mask: torch.Tensor,
                  clip: float = 0.2, ref_logp: torch.Tensor | None = None, kl_coef: float = 0.0) -> tuple[torch.Tensor, dict]:
    """PPO clipped surrogate per token (Schulman et al. 2017), averaged over completion tokens.

    advantages: [B] (one per sequence, broadcast to its tokens, as in GRPO) or [B, T-1] per token.
    ref_logp + kl_coef: optional KL penalty to the reference policy (k3 estimator, always >= 0).
    """
    if advantages.dim() == 1:
        advantages = advantages[:, None].expand_as(logp)
    ratio = torch.exp(logp - old_logp)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
    per_token = -torch.minimum(unclipped, clipped)
    metrics = {}
    if ref_logp is not None and kl_coef:
        log_r = ref_logp - logp
        kl = torch.exp(log_r) - log_r - 1  # k3: unbiased, non-negative estimate of KL(pi || ref)
        per_token = per_token + kl_coef * kl
        metrics["kl"] = float(masked_mean(kl.detach(), mask))
    loss = masked_mean(per_token, mask)
    with torch.no_grad():
        metrics["clip_fraction"] = float(masked_mean(((ratio - 1).abs() > clip).float(), mask))
        metrics["ratio_mean"] = float(masked_mean(ratio, mask))
        metrics["approx_kl_old"] = float(masked_mean(old_logp - logp, mask))
    return loss, metrics


def importance_sampling_loss(logp: torch.Tensor, old_logp: torch.Tensor, advantages: torch.Tensor,
                             mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """REINFORCE with an importance weight for off-policy samples: -(pi/pi_old) * A, averaged over completion tokens.
    With freshly sampled data (pi == pi_old) the gradient equals plain policy gradient."""
    if advantages.dim() == 1:
        advantages = advantages[:, None].expand_as(logp)
    ratio = torch.exp(logp - old_logp)
    loss = masked_mean(-ratio * advantages, mask)
    return loss, {"ratio_mean": float(masked_mean(ratio.detach(), mask))}


def cross_entropy_loss(logp: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Supervised fine-tuning on the masked tokens (e.g. only the answer, not the prompt)."""
    loss = -masked_mean(logp, mask)
    return loss, {"nll": float(loss.detach())}


# --------------------------------------------------------------------------------------------------- advantages
def group_advantages(rewards: list[float], group_size: int, normalize_std: bool = True, eps: float = 1e-6) -> tuple[list[float], list[bool]]:
    """GRPO (Shao et al. 2024): advantage = (r - mean(group)) / std(group). Groups are consecutive runs of `group_size`.

    Returns (advantages, informative) where informative[i] is False for rows whose group had identical rewards (zero
    advantage: no learning signal; callers can skip them to save compute)."""
    if group_size < 1 or len(rewards) % group_size:
        raise ValueError(f"{len(rewards)} rewards do not split into groups of {group_size}")
    r = torch.tensor(rewards, dtype=torch.float64)
    if not torch.isfinite(r).all():
        raise ValueError("rewards must be finite numbers")
    g = r.view(-1, group_size)
    centered = g - g.mean(dim=1, keepdim=True)
    std = g.std(dim=1, unbiased=False, keepdim=True)
    adv = centered / (std + eps) if normalize_std else centered
    informative = (std > eps).expand_as(g).reshape(-1).tolist()
    return adv.reshape(-1).tolist(), informative


# --------------------------------------------------------------------------------------------------- sampling
def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0,
                generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """One sampling step. logits: [B, V]. Returns (token [B], log-prob of that token under the *unmodified* policy [B]).

    The returned log-prob is from the full softmax (temperature 1, no truncation): that is the behaviour policy the losses
    compare against. temperature == 0 means greedy."""
    logits = logits.float()
    full_logp = F.log_softmax(logits, dim=-1)
    if temperature == 0:
        token = logits.argmax(-1)
    else:
        scaled = logits / temperature
        if top_k and top_k < scaled.shape[-1]:
            kth = torch.topk(scaled, top_k, dim=-1).values[:, -1:]
            scaled = scaled.masked_fill(scaled < kth, float("-inf"))
        if top_p < 1.0:
            sorted_logits, order = torch.sort(scaled, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(-1)
            drop = cum - probs > top_p  # keep the smallest prefix whose mass reaches top_p (always keeps the top token)
            sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
            scaled = torch.full_like(scaled, float("-inf")).scatter(-1, order, sorted_logits)
        probs = F.softmax(scaled, dim=-1)
        token = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return token, full_logp.gather(-1, token[:, None]).squeeze(-1)
