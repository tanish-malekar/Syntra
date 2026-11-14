"""Pure planning logic: choose GPUs for a model and split its layers over them.

No database access here (the scheduler builds `Candidate`s from the pool), so everything is easy to test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

RUNTIME_OVERHEAD_MB = 500  # per worker: runtime, KV cache and activation buffers (measured ~0.6-1.0 GB per TinyLlama half)
# Memory per layer relative to float16 when the frozen base weights are stored in 4-bit (FlexGen group quantization:
# 4 bits + an fp16 scale/min pair per 64 values = 4.5 of 16 bits, rounded up for the working buffers).
WEIGHT_BITS_FRACTION = {16: 1.0, 4: 0.3}


@dataclass(frozen=True)
class ModelSpec:
    num_layers: int
    per_layer_memory_mb: int
    weight_bits: int = 16  # 16 = float16 base weights, 4 = 4-bit base (QLoRA training / light serving)


@dataclass(frozen=True)
class Candidate:
    gpu_id: int
    free_memory_mb: int
    price_per_hour: float
    reliability: float = 1.0
    machine: int = 0  # GPUs sharing a machine (agent token) are preferred together: fewer network hops


@dataclass(frozen=True)
class Stage:
    gpu_id: int
    layer_start: int
    layer_end: int  # exclusive

    @property
    def layers(self) -> int:
        return self.layer_end - self.layer_start


@dataclass
class Plan:
    stages: list[Stage]
    hourly_rate: float
    machines: int = 1
    notes: list[str] = field(default_factory=list)


def layer_memory_mb(model: ModelSpec) -> int:
    return max(1, math.ceil(model.per_layer_memory_mb * WEIGHT_BITS_FRACTION[model.weight_bits]))


def overhead_mb(model: ModelSpec) -> int:
    """Per-worker fixed memory. A 4-bit worker also holds one layer decompressed to float16 while computing it."""
    return RUNTIME_OVERHEAD_MB + (model.per_layer_memory_mb if model.weight_bits != 16 else 0)


def stage_memory_mb(model: ModelSpec, layers: int) -> int:
    return layers * layer_memory_mb(model) + overhead_mb(model)


def capacity_layers(model: ModelSpec, free_memory_mb: int) -> int:
    """How many layers fit on a GPU with this much free memory (0 if not even one)."""
    return max(0, (free_memory_mb - overhead_mb(model)) // layer_memory_mb(model))


def split_layers(total: int, capacities: list[int]) -> list[int] | None:
    """Layers per stage, proportional to capacity, each stage >= 1 and <= its capacity; None if they cannot fit."""
    n = len(capacities)
    if n == 0 or n > total or any(c < 1 for c in capacities) or sum(capacities) < total:
        return None
    cap_sum = sum(capacities)
    ideal = [total * c / cap_sum for c in capacities]
    layers = [min(c, max(1, int(x))) for x, c in zip(ideal, capacities)]
    while sum(layers) > total:  # the max(1, ...) floor can overshoot: trim the stage that is furthest over its ideal
        i = max(range(n), key=lambda k: (layers[k] - ideal[k], -k) if layers[k] > 1 else (-1e9, 0))
        layers[i] -= 1
    while sum(layers) < total:  # hand out the rest by largest fractional remainder, within capacity
        room = [k for k in range(n) if layers[k] < capacities[k]]
        i = max(room, key=lambda k: (ideal[k] - layers[k], -k))
        layers[i] += 1
    return layers


def _member_key(c: Candidate):
    return (-c.reliability, c.price_per_hour, -c.free_memory_mb, c.gpu_id)


def order_candidates(cands: list[Candidate], n: int) -> list[Candidate]:
    """Machines that can supply all n GPUs alone first (fewest hops), then cheaper machines; fill machine by machine."""
    groups: dict[int, list[Candidate]] = {}
    for c in cands:
        groups.setdefault(c.machine, []).append(c)
    ordered_groups = sorted(
        groups.items(),
        key=lambda kv: (0 if len(kv[1]) >= n else 1, min(m.price_per_hour for m in kv[1]), -len(kv[1]), kv[0]),
    )
    out: list[Candidate] = []
    for _, members in ordered_groups:
        out.extend(sorted(members, key=_member_key))
    return out


def _try(model: ModelSpec, cands: list[Candidate], n: int) -> Plan | None:
    chosen = order_candidates(cands, n)[:n]
    if len(chosen) < n:
        return None
    caps = [capacity_layers(model, c.free_memory_mb) for c in chosen]
    layers = split_layers(model.num_layers, caps)
    if layers is None:
        return None
    stages, start = [], 0
    for c, count in zip(chosen, layers):
        stages.append(Stage(c.gpu_id, start, start + count))
        start += count
    return Plan(stages=stages, hourly_rate=round(sum(c.price_per_hour for c in chosen), 6),
                machines=len({c.machine for c in chosen}))


def make_plan(model: ModelSpec, cands: list[Candidate], requested_gpus: int | None = None) -> Plan | None:
    """A plan on exactly `requested_gpus` GPUs (demo override), or on the fewest GPUs that fit. None = cannot run now."""
    if requested_gpus is not None:
        if requested_gpus < 1 or requested_gpus > model.num_layers:
            raise ValueError(f"requested_gpus must be between 1 and {model.num_layers} for this model")
        return _try(model, cands, requested_gpus)
    for n in range(1, min(len(cands), model.num_layers) + 1):
        plan = _try(model, cands, n)
        if plan is not None:
            return plan
    return None


def replacement_candidate(model: ModelSpec, cands: list[Candidate], layers: int) -> Candidate | None:
    """Best idle GPU able to take over a stage with `layers` layers (never one that is already working)."""
    need = stage_memory_mb(model, layers)
    fitting = [c for c in cands if c.free_memory_mb >= need]
    return sorted(fitting, key=_member_key)[0] if fitting else None
