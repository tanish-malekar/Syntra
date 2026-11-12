"""Deploy stage of scripts/e2e_real.py: runs against the live backend with real BloomBee workers and TinyLlama."""
from __future__ import annotations

import time

PROMPT = "Ticket: my package has not arrived yet. Please fix this asap.\nAnswer:"


def run_deploy_stage(url, alice, provider, model, log, results, wait):
    """alice trained `model` earlier; she comes back, deploys it (2 GPUs) next to the base model (1 GPU) and compares."""
    t = time.monotonic()
    tuned = alice.post("/deployments", json={"model_kind": "finetuned", "finetuned_model_id": model["id"], "requested_gpus": 2}).json()
    base = alice.post("/deployments", json={"model_kind": "base", "base_model_id": 1, "requested_gpus": 1}).json()
    log("deployments requested", tuned=tuned["id"], base=base["id"], tuned_status=tuned["status"], base_status=base["status"])

    def dep_running(dep_id):
        d = alice.get(f"/deployments/{dep_id}").json()
        assert d["status"] != "failed", d
        return d if d["status"] == "running" else None

    tuned = wait(lambda: dep_running(tuned["id"]), 900, "tuned deployment running")
    base = wait(lambda: dep_running(base["id"]), 900, "base deployment running")
    log("deployments running", seconds=round(time.monotonic() - t, 1), tuned_gpus=tuned["num_gpus"], base_gpus=base["num_gpus"],
        layers=[(l["gpu_name"], l["layer_start"], l["layer_end"]) for l in tuned["job"]["leases"] if l["active"]])
    assert tuned["num_gpus"] == 2 and base["num_gpus"] == 1
    kinds = [e["type"] for e in alice.get("/events", params={"job_id": tuned["job"]["id"]}).json()]
    assert kinds.count("deploy.merging") == 1 and kinds[-1] == "deploy.online", kinds

    a = alice.post(f"/deployments/{tuned['id']}/generate", json={"prompt": PROMPT, "max_new_tokens": 28}).json()
    b = alice.post(f"/deployments/{base['id']}/generate", json={"prompt": PROMPT, "max_new_tokens": 28}).json()
    log("greedy answers", tuned=a["text"], base=b["text"][:100], tuned_seconds=a["seconds"], base_seconds=b["seconds"])
    assert '"department"' in a["text"] and "{" not in b["text"]

    s1 = alice.post(f"/deployments/{base['id']}/generate", json={"prompt": PROMPT, "max_new_tokens": 24, "do_sample": True, "temperature": 1.0, "top_p": 0.95, "seed": 1}).json()
    s2 = alice.post(f"/deployments/{base['id']}/generate", json={"prompt": PROMPT, "max_new_tokens": 24, "do_sample": True, "temperature": 1.0, "top_p": 0.95, "seed": 2}).json()
    log("sampling", differs=s1["text"] != s2["text"], one=s1["text"][:60], two=s2["text"][:60])
    assert s1["text"] != s2["text"]
    again = alice.post(f"/deployments/{tuned['id']}/generate", json={"prompt": PROMPT, "max_new_tokens": 28}).json()
    assert again["text"] == a["text"]  # greedy is repeatable

    cmp = alice.post("/compare", json={"prompt": PROMPT, "max_new_tokens": 28, "base_deployment_id": base["id"], "tuned_deployment_id": tuned["id"]}).json()
    log("compare", base=cmp["base"]["text"][:70], finetuned=cmp["finetuned"]["text"])

    # --- the held-out evaluation, stored with the model ------------------------------------------------------------
    r = alice.post(f"/models/{model['id']}/evaluate", json={"base_deployment_id": base["id"], "tuned_deployment_id": tuned["id"]})
    assert r.status_code == 202, r.text
    scores = wait(lambda: (lambda s: s if s.get("status") in ("done", "failed") else None)(alice.get(f"/models/{model['id']}").json()["scores"]), 900, "evaluation")
    log("evaluation", status=scores.get("status"), n=scores.get("n"), base={k: scores["base"][k] for k in ("valid_json", "dept_correct", "urgency_correct")} if scores.get("base") else None,
        finetuned={k: scores["finetuned"][k] for k in ("valid_json", "dept_correct", "urgency_correct")} if scores.get("finetuned") else None, error=scores.get("error"))
    assert scores["status"] == "done" and scores["finetuned"]["valid_json"] >= 40 and scores["base"]["valid_json"] <= 2, scores
    results["eval"] = {"base": scores["base"]["valid_json"], "tuned": scores["finetuned"]["valid_json"], "dept": scores["finetuned"]["dept_correct"], "n": scores["n"]}

    # --- a serving worker dies: the platform finds another GPU, the endpoint keeps answering -------------------------------
    assert alice.post(f"/deployments/{base['id']}/stop").status_code == 200  # the pool has no spare GPU until the base model is stopped
    wait(lambda: len([g for g in alice.get("/gpus").json() if g["status"] == "idle"]) >= 1, 60, "a spare GPU")
    jid = tuned["job"]["id"]
    before = {l["gpu_id"] for l in alice.get(f"/jobs/{jid}").json()["leases"]}
    t_kill = time.monotonic()
    assert alice.post(f"/jobs/{jid}/kill-worker", json={"layer_start": 11}).status_code == 200
    wait(lambda: [e for e in alice.get("/events", params={"job_id": jid}).json() if e["type"] == "worker.replacement_online"], 300, "serving replacement online")
    after = alice.post(f"/deployments/{tuned['id']}/generate", json={"prompt": PROMPT, "max_new_tokens": 28}, timeout=300).json()
    log("serving survived a killed worker", seconds=round(time.monotonic() - t_kill, 1), same_answer=after["text"] == a["text"])
    assert after["text"] == a["text"]
    new_leases = [l for l in alice.get(f"/jobs/{jid}").json()["leases"] if l["active"]]
    assert {l["gpu_id"] for l in new_leases} != before

    # --- stop: GPUs return to the pool, the model remains stored ---------------------------------------------------------
    assert alice.post(f"/deployments/{tuned['id']}/stop").status_code == 200
    wait(lambda: {g["status"] for g in alice.get("/gpus").json()} == {"idle"}, 120, "all GPUs idle after stopping")
    assert alice.get(f"/models/{model['id']}").json()["deployed"] is False
    costs, earnings = alice.get("/ledger/costs").json(), provider.get("/ledger/earnings").json()
    log("final billing", cost=costs["total"], earned=earnings["total"], entries=len(costs["entries"]))
    assert abs(costs["total"] - earnings["total"]) < 1e-6
