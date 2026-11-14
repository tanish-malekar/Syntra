"""The training primitives behind RL sessions and managed DPO (Tinker-style): sample, logprobs, forward_backward, optim_step, save.

The engine works on any causal LM whose trainable weights are a dict of named parameters (the LoRA A/B matrices of a
BloomBee `DistributedLoRA` in production, a tiny Hugging Face model in tests). Everything model-specific is passed in, so this
file needs only torch.

Sequences are lists of token ids: `{"tokens": [...prompt..., ...completion...], "prompt_len": n}`. Batches are right-padded
(BloomBee's LoRA path uses consecutive positions from 0, so left padding is not possible; right padding never changes the
log-probs of earlier tokens in a causal model).

The reference policy is the adapter as it was when the session started (the base model for a fresh adapter, the SFT model
when continuing from one): its weights are swapped in for reference passes, so no second model copy is needed.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import torch

from rl_losses import (completion_mask, cross_entropy_loss, dpo_loss, group_advantages, importance_sampling_loss,
                       masked_sum, ppo_clip_loss, sample_next, token_logprobs)

LOSSES = ("cross_entropy", "dpo", "ppo", "importance_sampling")
MAX_BATCH_TOKENS = 64 * 1024


class EngineError(ValueError):
    """A bad request (shapes, lengths, values): reported to the caller, the session keeps running."""


class Engine:
    def __init__(self, model: Callable, params: dict[str, torch.nn.Parameter], tokenizer, device: str,
                 save_fn: Callable[[Path], None], max_seq_len: int = 1024, eos_token_ids: list[int] | None = None,
                 max_lr: float = 1e-2):
        self.model, self.params, self.tok, self.device = model, params, tokenizer, device
        self.save_fn, self.max_seq_len, self.max_lr = save_fn, max_seq_len, max_lr
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
        eos = eos_token_ids if eos_token_ids is not None else tokenizer.eos_token_id
        self.eos = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
        self.reference = {k: p.detach().clone() for k, p in params.items()}
        self.optimizer = torch.optim.AdamW(list(params.values()), lr=1e-5)
        self.steps = 0
        self.pending_backward = 0  # forward_backward calls since the last optim_step
        self.tokens_seen = 0

    # ------------------------------------------------------------------------------------------- helpers
    def _check_seq(self, seq: dict) -> tuple[list[int], int]:
        tokens, prompt_len = seq.get("tokens"), seq.get("prompt_len", 0)
        if not isinstance(tokens, list) or not tokens or not all(isinstance(t, int) for t in tokens):
            raise EngineError("each sequence needs a non-empty 'tokens' list of token ids")
        if not isinstance(prompt_len, int) or not 0 <= prompt_len < len(tokens):
            raise EngineError(f"prompt_len must be in [0, {len(tokens) - 1}] (the completion must have at least one token)")
        if len(tokens) > self.max_seq_len:
            raise EngineError(f"sequence of {len(tokens)} tokens is longer than this session's limit of {self.max_seq_len}")
        vocab = getattr(self.tok, "vocab_size", None)
        if vocab and max(tokens) >= len(self.tok):
            raise EngineError("token id out of range for this model's vocabulary")
        return tokens, prompt_len

    def _pad(self, seqs: list[dict]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        checked = [self._check_seq(s) for s in seqs]
        width = max(len(t) for t, _ in checked)
        if width * len(checked) > MAX_BATCH_TOKENS:
            raise EngineError(f"batch too large ({len(checked)} x {width} tokens); split it into several calls")
        ids = torch.full((len(checked), width), self.pad_id, dtype=torch.long)
        for i, (t, _) in enumerate(checked):
            ids[i, :len(t)] = torch.tensor(t)
        mask = completion_mask([len(t) for t, _ in checked], [p for _, p in checked], width)
        return ids.to(self.device), mask[:, 1:].to(self.device), [len(t) for t, _ in checked]

    def _logp(self, ids: torch.Tensor) -> torch.Tensor:
        return token_logprobs(self.model(ids), ids)

    @contextmanager
    def _reference_weights(self):
        saved = {k: p.detach().clone() for k, p in self.params.items()}
        with torch.no_grad():
            for k, p in self.params.items():
                p.copy_(self.reference[k])
        try:
            yield
        finally:
            with torch.no_grad():
                for k, p in self.params.items():
                    p.copy_(saved[k])

    def _completion_logprobs(self, seqs: list[dict], reference: bool) -> list[list[float]]:
        ids, mask, lengths = self._pad(seqs)
        with torch.no_grad(), (self._reference_weights() if reference else _null()):
            logp = self._logp(ids)
        out = []
        for i, s in enumerate(seqs):
            out.append(logp[i, s["prompt_len"] - 1 if s["prompt_len"] else 0: lengths[i] - 1].tolist()
                       if s["prompt_len"] else logp[i, : lengths[i] - 1].tolist())
        return out

    # ------------------------------------------------------------------------------------------- ops
    def tokenize(self, texts: list[str], add_special_tokens: bool = True) -> dict:
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise EngineError("texts must be a list of strings")
        return {"tokens": [self.tok(t, add_special_tokens=add_special_tokens)["input_ids"] for t in texts]}

    def detokenize(self, tokens: list[list[int]], skip_special_tokens: bool = True) -> dict:
        return {"texts": [self.tok.decode(t, skip_special_tokens=skip_special_tokens) for t in tokens]}

    def sample(self, prompts: list, n: int = 1, max_new_tokens: int = 32, temperature: float = 1.0, top_p: float = 1.0,
               top_k: int = 0, seed: int | None = None, stop_at_eos: bool = True) -> dict:
        """n completions per prompt (prompt = text or token ids). Returns full sequences with the per-token log-probs of the
        sampled completion under the current policy (the 'old' log-probs for PPO)."""
        if not 1 <= n <= 64:
            raise EngineError("n must be between 1 and 64")
        if not 1 <= max_new_tokens <= 1024:
            raise EngineError("max_new_tokens must be between 1 and 1024")
        if temperature < 0 or not 0 < top_p <= 1 or top_k < 0:
            raise EngineError("need temperature >= 0, 0 < top_p <= 1, top_k >= 0")
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(int(seed))
        else:
            gen.seed()
        samples = []
        for pi, prompt in enumerate(prompts):
            ids = self.tok(prompt)["input_ids"] if isinstance(prompt, str) else list(prompt)
            if not ids:
                raise EngineError(f"prompt {pi} is empty")
            if len(ids) + max_new_tokens > self.max_seq_len:
                raise EngineError(f"prompt {pi} ({len(ids)} tokens) + max_new_tokens exceeds the limit of {self.max_seq_len}")
            tokens = torch.tensor([ids] * n, dtype=torch.long, device=self.device)
            logps = torch.zeros(n, 0)
            done = torch.zeros(n, dtype=torch.bool)
            lengths = torch.full((n,), len(ids) + max_new_tokens)
            with torch.no_grad():
                for step in range(max_new_tokens):
                    logits = self.model(tokens)[:, -1].cpu()
                    nxt, lp = sample_next(logits, temperature, top_p, top_k, generator=gen)
                    nxt = torch.where(done, torch.full_like(nxt, self.pad_id), nxt)
                    tokens = torch.cat((tokens, nxt.to(self.device)[:, None]), dim=1)
                    logps = torch.cat((logps, torch.where(done, torch.zeros_like(lp), lp)[:, None]), dim=1)
                    if stop_at_eos and self.eos:
                        hit = torch.isin(nxt, torch.tensor(self.eos)) & ~done
                        lengths = torch.where(hit, torch.tensor(len(ids) + step + 1), lengths)
                        done |= hit
                        if bool(done.all()):
                            break
            for j in range(n):
                full = tokens[j, : int(lengths[j])].tolist()
                comp = full[len(ids):]
                samples.append({"prompt_index": pi, "tokens": full, "prompt_len": len(ids), "completion": comp,
                                "logprobs": logps[j, : len(comp)].tolist(), "finished": bool(done[j]),
                                "text": self.tok.decode(comp, skip_special_tokens=True)})
        return {"samples": samples}

    def logprobs(self, sequences: list[dict], reference: bool = False) -> dict:
        return {"logprobs": self._completion_logprobs(sequences, reference)}

    def forward_backward(self, loss: str, batch: list[dict], **args) -> dict:
        """Accumulate gradients for one batch. The loss is averaged within the call; several calls before an optim_step add up
        (scale with `weight` if they should count differently)."""
        if loss not in LOSSES:
            raise EngineError(f"unknown loss {loss!r}; choose one of {LOSSES}")
        if not batch:
            raise EngineError("empty batch")
        weight = float(args.pop("weight", 1.0))
        if loss == "dpo":
            value, metrics = self._dpo(batch, **args)
        else:
            ids, mask, _ = self._pad(batch)
            logp = self._logp(ids)
            if loss == "cross_entropy":
                value, metrics = cross_entropy_loss(logp, mask)
            else:
                adv = self._advantages(batch, mask)
                old = self._aligned(batch, "old_logprobs", mask, default=logp.detach())
                if loss == "ppo":
                    ref = None
                    if args.get("kl_coef"):
                        ref = self._aligned(batch, "ref_logprobs", mask, default=None)
                        if ref is None:
                            with torch.no_grad(), self._reference_weights():
                                ref = self._logp(ids)
                    value, metrics = ppo_clip_loss(logp, old, adv, mask, clip=float(args.get("clip", 0.2)),
                                                   ref_logp=ref, kl_coef=float(args.get("kl_coef", 0.0)))
                else:
                    value, metrics = importance_sampling_loss(logp, old, adv, mask)
        if not torch.isfinite(value):
            raise EngineError("the loss is not finite (check rewards/advantages and learning rate)")
        (value * weight).backward()
        self.pending_backward += 1
        n_tokens = sum(len(b.get("tokens", [])) for b in batch) if loss != "dpo" else 0
        self.tokens_seen += n_tokens
        return {"loss": float(value.detach()), "metrics": metrics, "pending_backward": self.pending_backward}

    def _advantages(self, batch: list[dict], mask: torch.Tensor) -> torch.Tensor:
        if all("advantage" in b for b in batch):
            vals = [float(b["advantage"]) for b in batch]
            if not all(math.isfinite(v) for v in vals):
                raise EngineError("advantages must be finite")
            return torch.tensor(vals, device=mask.device)
        return self._aligned(batch, "advantages", mask, default=None, required=True)

    def _aligned(self, batch, key, mask, default, required=False):
        """Per-token values given for the completion only -> [B, T-1] aligned with log-probs (0 elsewhere)."""
        if not all(key in b for b in batch):
            if required:
                raise EngineError(f"every row needs '{key}' (or a scalar 'advantage')")
            return default
        out = torch.zeros_like(mask)
        for i, b in enumerate(batch):
            vals = b[key]
            start = max(b["prompt_len"] - 1, 0)
            want = len(b["tokens"]) - b["prompt_len"] if b["prompt_len"] else len(b["tokens"]) - 1
            if len(vals) != want:
                raise EngineError(f"row {i}: '{key}' has {len(vals)} values for {want} completion tokens")
            out[i, start: start + want] = torch.tensor([float(v) for v in vals], device=mask.device)
        if not torch.isfinite(out).all():
            raise EngineError(f"'{key}' must be finite")
        return out

    def _dpo(self, batch: list[dict], beta: float = 0.1, label_smoothing: float = 0.0) -> tuple[torch.Tensor, dict]:
        for i, b in enumerate(batch):
            if "chosen" not in b or "rejected" not in b:
                raise EngineError(f"DPO row {i} needs 'chosen' and 'rejected' sequences")
        seqs = [b["chosen"] for b in batch] + [b["rejected"] for b in batch]
        if all("ref_chosen" in b and "ref_rejected" in b for b in batch):
            ref = torch.tensor([b["ref_chosen"] for b in batch] + [b["ref_rejected"] for b in batch], device=self.device)
        else:
            ids, mask, _ = self._pad(seqs)
            with torch.no_grad(), self._reference_weights():
                ref = masked_sum(self._logp(ids), mask)
        ids, mask, _ = self._pad(seqs)
        pol = masked_sum(self._logp(ids), mask)
        k = len(batch)
        return dpo_loss(pol[:k], pol[k:], ref[:k], ref[k:], beta=beta, label_smoothing=label_smoothing)

    def reference_sums(self, pairs: list[dict]) -> list[tuple[float, float]]:
        """Summed reference log-probs of (chosen, rejected) per pair: computed once at the start of a DPO run."""
        seqs = [p["chosen"] for p in pairs] + [p["rejected"] for p in pairs]
        ids, mask, _ = self._pad(seqs)
        with torch.no_grad(), self._reference_weights():
            s = masked_sum(self._logp(ids), mask).tolist()
        return list(zip(s[: len(pairs)], s[len(pairs):]))

    def optim_step(self, lr: float = 1e-5, weight_decay: float = 0.0, max_grad_norm: float = 1.0,
                   betas: tuple[float, float] = (0.9, 0.999)) -> dict:
        if self.pending_backward == 0:
            raise EngineError("optim_step without a forward_backward since the last step")
        if not 0 < lr <= self.max_lr:
            raise EngineError(f"lr must be in (0, {self.max_lr}]")
        for group in self.optimizer.param_groups:
            group.update(lr=lr, weight_decay=weight_decay, betas=tuple(betas))
        params = list(self.params.values())
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm) if max_grad_norm else \
            torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None]))
        if not torch.isfinite(norm):
            self.optimizer.zero_grad(set_to_none=True)
            self.pending_backward = 0
            raise EngineError("gradient is not finite; the step was skipped and the gradients cleared")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.steps += 1
        self.pending_backward = 0
        return {"step": self.steps, "grad_norm": float(norm)}

    def save(self, out_dir: str) -> dict:
        path = Path(out_dir)
        self.save_fn(path)
        return {"adapter_dir": str(path), "step": self.steps}

    def state(self) -> dict:
        return {"step": self.steps, "pending_backward": self.pending_backward, "tokens_seen": self.tokens_seen,
                "trainable_parameters": sum(p.numel() for p in self.params.values())}

    def advantages(self, rewards: list[float], group_size: int, normalize_std: bool = True) -> dict:
        try:
            adv, informative = group_advantages(rewards, group_size, normalize_std)
        except ValueError as exc:
            raise EngineError(str(exc))
        return {"advantages": adv, "informative": informative}

    OPS = ("tokenize", "detokenize", "sample", "logprobs", "forward_backward", "optim_step", "save", "state", "advantages")

    def call(self, op: str, args: dict) -> dict:
        if op not in self.OPS:
            raise EngineError(f"unknown op {op!r}")
        try:
            return getattr(self, op)(**args)
        except TypeError as exc:  # wrong/missing arguments
            raise EngineError(f"{op}: {exc}")


@contextmanager
def _null():
    yield
