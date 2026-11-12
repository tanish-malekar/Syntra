"""Held-out evaluation of a fine-tuned model against its base model, both served on the platform."""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from app.events import EventType
from app.models import FineTunedModel, Job

_JSON_RE = re.compile(r"\{.*?\}", re.S)


def parse_prediction(text: str) -> dict | None:
    """The first JSON object in a model's answer, or None."""
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        value = json.loads(m.group(0))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def score(rows: list[dict], outputs: list[str]) -> dict:
    """Share of answers that are valid JSON, and that name the right department / urgency."""
    n = len(rows)
    valid = dept = urgency = 0
    for row, text in zip(rows, outputs):
        pred = parse_prediction(text)
        if pred is None:
            continue
        valid += 1
        dept += int(pred.get("department") == row.get("department"))
        urgency += int(pred.get("urgency") == row.get("urgency"))
    return {"n": n, "valid_json": valid, "dept_correct": dept, "urgency_correct": urgency}


def load_heldout(path: str, limit: int | None = None) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return rows[:limit] if limit else rows


async def evaluate_model(orchestrator, model_id: int, base_job_id: int, tuned_job_id: int, limit: int | None = None) -> dict:
    """Same prompts, greedy decoding, on the base and fine-tuned deployments; stores and announces the scores."""
    def prepare(s):
        model = s.get(FineTunedModel, model_id)
        job = s.get(Job, model.job_id)
        config = json.loads(job.config_json or "{}")
        model.scores_json = json.dumps({"status": "running"})
        s.add(model); s.commit()
        return model.owner_id, config.get("heldout_path", "")

    owner_id, heldout = await orchestrator._db(prepare)
    try:
        if not heldout or not Path(heldout).exists():
            raise FileNotFoundError("this model's dataset has no held-out examples to evaluate on")
        rows = load_heldout(heldout, limit)
        outputs: dict[str, list[str]] = {"base": [], "finetuned": []}
        seconds: dict[str, float] = {"base": 0.0, "finetuned": 0.0}
        for row in rows:
            request = {"prompt": row["prompt"], "max_new_tokens": 28, "do_sample": False}
            base, tuned = await asyncio.gather(orchestrator.generate(base_job_id, request, timeout=180),
                                               orchestrator.generate(tuned_job_id, request, timeout=180))
            outputs["base"].append(base["text"]); outputs["finetuned"].append(tuned["text"])
            seconds["base"] += base["seconds"]; seconds["finetuned"] += tuned["seconds"]
        result = {"status": "done", "n": len(rows)}
        for label in ("base", "finetuned"):
            result[label] = {**score(rows, outputs[label]), "mean_seconds": round(seconds[label] / max(1, len(rows)), 3),
                             "examples": outputs[label][:3]}
        error = None
    except Exception as exc:
        result, error = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300]}, str(exc)

    def save(s):
        model = s.get(FineTunedModel, model_id)
        model.scores_json = json.dumps(result)
        s.add(model); s.commit()
        if error is None:
            for label in ("base", "finetuned"):
                orchestrator.events.publish(
                    s, EventType.EVAL_RESULT, job_id=tuned_job_id, user_id=owner_id,
                    payload={"model_label": label, **{k: result[label][k] for k in ("n", "valid_json", "dept_correct", "urgency_correct")}})
    await orchestrator._db(save)
    return result
