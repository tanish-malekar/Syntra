"""DiLoCo outer optimization (Douillard et al. 2023), pure torch: average the replicas' pseudo-gradients and take a Nesterov
momentum step on the global weights. Unit-tested on its own (backend/runner_tests/test_diloco.py)."""
from __future__ import annotations

import torch


class OuterOptimizer:
    def __init__(self, params: dict[str, torch.Tensor], lr: float = 0.7, momentum: float = 0.9, nesterov: bool = True):
        self.params = {k: v.detach().float().clone() for k, v in params.items()}
        self.lr, self.mu, self.nesterov = lr, momentum, nesterov
        self.buf = {k: torch.zeros_like(v) for k, v in self.params.items()}

    def check(self, delta: dict[str, torch.Tensor]) -> str | None:
        """Why this pseudo-gradient cannot be used (None if it can): wrong names or shapes, or non-finite values."""
        if delta.keys() != self.params.keys():
            missing, extra = self.params.keys() - delta.keys(), delta.keys() - self.params.keys()
            return f"parameter names differ (missing {sorted(missing)[:3]}, unexpected {sorted(extra)[:3]})"
        for k, v in delta.items():
            if v.shape != self.params[k].shape:
                return f"{k}: shape {tuple(v.shape)} != {tuple(self.params[k].shape)}"
            if not torch.isfinite(v).all():
                return f"{k}: non-finite values"
        return None

    def step(self, deltas: list[tuple[dict[str, torch.Tensor], float]]) -> float:
        """deltas: (theta_start - theta_end, weight) per replica; weight = tokens trained. Returns the norm of the average."""
        if not deltas:
            raise ValueError("no pseudo-gradients to apply")
        total = sum(w for _, w in deltas)
        if total <= 0:
            raise ValueError("weights must be positive")
        sq = 0.0
        for k, p in self.params.items():
            avg = sum(d[k].float() * (w / total) for d, w in deltas)
            sq += float(avg.pow(2).sum())
            self.buf[k].mul_(self.mu).add_(avg)
            update = avg + self.mu * self.buf[k] if self.nesterov else self.buf[k]
            p.sub_(self.lr * update)
        return sq ** 0.5

    def state_dict(self) -> dict:
        return {"buf": self.buf, "lr": self.lr, "mu": self.mu}

    def load_state_dict(self, state: dict) -> None:
        self.buf = {k: v.float() for k, v in state["buf"].items()}
