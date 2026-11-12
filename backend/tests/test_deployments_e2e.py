"""Deploy from the registry: merge once, serve on N GPUs, generate, compare, evaluate, stop; several users at once."""
from __future__ import annotations

import asyncio

import pytest

from tests.platform_helper import Platform, wait_for


def run(coro):
    return asyncio.run(coro)


def test_train_then_come_back_later_deploy_compare_and_evaluate(tmp_path):
    async def scenario():
        async with Platform(tmp_path, runner_spec_extra={"fake_step_delay": 0.05}) as p:
            alice, lab = await p.user("alice"), await p.user("lab")
            await p.provider(lab, "lab", gpus=4)
            await wait_for(lambda: asyncio.sleep(0, result=len(p.gpus_sync()) == 4), what="GPUs")
            model = await p.train_model(alice, requested_gpus=2, steps=4, name="Tickets v1")
            assert model["deployed"] is False and model["scores"] == {}
            assert {g["status"] for g in await p.pool(alice)} == {"idle"}   # training is over, GPUs are back in the pool

            # "later": deploy the stored model on 2 GPUs and the base model on 1 for comparison
            tuned = await p.deploy(alice, model_kind="finetuned", finetuned_model_id=model["id"], requested_gpus=2)
            base = await p.deploy(alice, model_kind="base", base_model_id=1, requested_gpus=1)
            assert tuned["num_gpus"] == 2 and base["num_gpus"] == 1 and tuned["endpoint"] == f"/deployments/{tuned['id']}/generate"
            assert tuned["model_name"] == "Tickets v1" and base["model_name"] == "TinyLlama-1.1B-Chat"
            used = {l["gpu_id"] for d in (tuned, base) for l in d["job"]["leases"]}
            assert len(used) == 3                                           # three different GPUs
            ev = [e["type"] for e in await p.events(alice, tuned["job"]["id"])]
            assert ev.count("deploy.merging") == 1 and ev[-1] == "deploy.online"
            assert (await alice.get(f"/models/{model['id']}")).json()["deployed"] is True

            # generate: greedy by default; the fine-tuned model answers in the trained format, the base model does not
            prompt = "Ticket: my package has not arrived yet. Please fix this asap.\nAnswer:"
            a = (await alice.post(f"/deployments/{tuned['id']}/generate", json={"prompt": prompt})).json()
            b = (await alice.post(f"/deployments/{base['id']}/generate", json={"prompt": prompt})).json()
            assert '"department": "shipping"' in a["text"] and '"urgency": "high"' in a["text"] and "{" not in b["text"]
            # sampling parameters reach the model
            s = (await alice.post(f"/deployments/{base['id']}/generate", json={"prompt": prompt, "do_sample": True, "temperature": 0.7,
                                                                                "top_p": 0.9, "top_k": 40, "seed": 3})).json()
            assert "t=0.7" in s["text"] and "p=0.9" in s["text"] and "k=40" in s["text"] and "seed=3" in s["text"]

            cmp = (await alice.post("/compare", json={"prompt": prompt, "base_deployment_id": base["id"], "tuned_deployment_id": tuned["id"]})).json()
            assert "{" in cmp["finetuned"]["text"] and "{" not in cmp["base"]["text"] and cmp["sampled"] is False

            # held-out evaluation, stored with the model
            r = await alice.post(f"/models/{model['id']}/evaluate", json={"base_deployment_id": base["id"], "tuned_deployment_id": tuned["id"]})
            assert r.status_code == 202
            scores = (await wait_for(lambda: p_scores(alice, model["id"]), 60, "evaluation done"))
            assert scores["n"] == 48 and scores["base"]["valid_json"] == 0
            assert scores["finetuned"]["valid_json"] == 48 and scores["finetuned"]["dept_correct"] >= 40
            kinds = [e["type"] for e in await p.events(alice, tuned["job"]["id"])]
            assert kinds.count("eval.result") == 2

            # stop: GPUs go back, the merged model stays so the next deploy is instant (no second merge)
            assert (await alice.post(f"/deployments/{tuned['id']}/stop")).status_code == 200
            await wait_for(lambda: p_dep_status(alice, tuned["id"], "stopped"), 30, "deployment stopped")
            assert (await alice.get(f"/models/{model['id']}")).json()["deployed"] is False
            again = await p.deploy(alice, model_kind="finetuned", finetuned_model_id=model["id"], requested_gpus=1)
            assert [e["type"] for e in await p.events(alice, again["job"]["id"])].count("deploy.merging") == 0
            assert (await alice.post(f"/deployments/{tuned['id']}/generate", json={"prompt": prompt})).status_code == 409  # stopped
            costs = (await alice.get("/ledger/costs")).json()
            assert costs["total"] > 0 and len(costs["entries"]) >= 4      # training + deployments are billed
    run(scenario())


async def p_scores(user, model_id):
    s = (await user.get(f"/models/{model_id}")).json()["scores"]
    return s if s.get("status") in ("done", "failed") else None


async def p_dep_status(user, dep_id, status):
    d = (await user.get(f"/deployments/{dep_id}")).json()
    return d if d["status"] == status else None


def test_one_user_serves_while_another_trains_on_different_gpus(tmp_path):
    async def scenario():
        async with Platform(tmp_path, runner_spec_extra={"fake_step_delay": 0.3}) as p:
            alice, bob, lab = await p.user("alice"), await p.user("bob"), await p.user("lab")
            await p.provider(lab, "lab", gpus=4)
            await wait_for(lambda: asyncio.sleep(0, result=len(p.gpus_sync()) == 4), what="GPUs")
            model = await p.train_model(alice, requested_gpus=1, steps=3)
            dep = await p.deploy(alice, model_kind="finetuned", finetuned_model_id=model["id"], requested_gpus=2)
            served = {l["gpu_id"] for l in dep["job"]["leases"]}

            job = await p.start_job(bob, requested_gpus=2, steps=6)         # bob trains at the same time
            await p.wait_event(bob, job["id"], "worker.online", count=2)
            trained = {l["gpu_id"] for l in (await p.job(bob, job["id"]))["leases"]}
            assert not (served & trained) and len(trained) == 2
            r = await alice.post(f"/deployments/{dep['id']}/generate", json={"prompt": "Ticket: app crashes when I upload a photo.\nAnswer:"})
            assert r.status_code == 200 and "bug" in r.json()["text"]       # alice's model answers while bob trains

            # bob cannot see or use alice's deployment or model
            assert (await bob.get(f"/deployments/{dep['id']}")).status_code == 404
            assert (await bob.post(f"/deployments/{dep['id']}/generate", json={"prompt": "x"})).status_code == 404
            assert (await bob.post(f"/deployments/{dep['id']}/stop")).status_code == 404
            assert (await bob.post("/deployments", json={"model_kind": "finetuned", "finetuned_model_id": model["id"]})).status_code == 404
            assert (await bob.get("/deployments")).json() == []

            # a third deployment that needs more GPUs than are idle waits, then starts when bob's run ends
            waiting = await p.deploy(alice, wait=False, model_kind="base", base_model_id=1, requested_gpus=2)
            assert waiting["status"] == "queued" and waiting["job"]["queue_position"] == 1
            await p.wait_status(bob, job["id"], "completed", timeout=60)
            ready = await wait_for(lambda: p_dep_status(alice, waiting["id"], "running"), 60, "queued deployment starts")
            assert ready["num_gpus"] == 2
            assert (await alice.post(f"/deployments/{waiting['id']}/stop")).status_code == 200
    run(scenario())


def test_serving_survives_a_killed_worker_on_a_new_gpu(tmp_path):
    async def scenario():
        async with Platform(tmp_path) as p:
            alice, lab = await p.user("alice"), await p.user("lab")
            await p.provider(lab, "lab", gpus=3)
            await wait_for(lambda: asyncio.sleep(0, result=len(p.gpus_sync()) == 3), what="GPUs")
            dep = await p.deploy(alice, model_kind="base", base_model_id=1, requested_gpus=2)
            jid = dep["job"]["id"]
            before = {l["gpu_id"] for l in dep["job"]["leases"]}
            assert (await alice.post(f"/jobs/{jid}/kill-worker", json={"layer_start": 11})).status_code == 200
            starting = (await p.wait_event(alice, jid, "worker.replacement_starting"))[0]["payload"]
            await p.wait_event(alice, jid, "worker.replacement_online")
            leases = (await p.job(alice, jid))["leases"]
            new = [l for l in leases if l["active"] and l["layer_start"] == 11][0]
            assert new["gpu_id"] not in before and starting["gpu_class"] == "Apple M-series"
            r = await alice.post(f"/deployments/{dep['id']}/generate", json={"prompt": "hello"})
            assert r.status_code == 200                                      # still answering after the replacement
            assert (await alice.post(f"/deployments/{dep['id']}/stop")).status_code == 200
    run(scenario())


def test_deployment_validation_and_stopping_a_queued_one(tmp_path):
    async def scenario():
        async with Platform(tmp_path) as p:
            alice = await p.user("alice")
            assert (await alice.post("/deployments", json={"model_kind": "weird"})).status_code == 422
            assert (await alice.post("/deployments", json={"model_kind": "base", "base_model_id": 99})).status_code == 404
            assert (await alice.post("/deployments", json={"model_kind": "base", "base_model_id": 1, "requested_gpus": 99})).status_code == 422
            assert (await alice.post("/deployments", json={"model_kind": "finetuned", "finetuned_model_id": 5})).status_code == 404
            dep = await p.deploy(alice, wait=False, model_kind="base", base_model_id=1)   # no GPUs in the pool: queued
            assert dep["status"] == "queued"
            assert (await alice.post(f"/deployments/{dep['id']}/generate", json={"prompt": "hi"})).status_code == 409
            assert (await alice.post(f"/deployments/{dep['id']}/stop")).status_code == 200
            assert (await alice.get(f"/deployments/{dep['id']}")).json()["status"] == "stopped"
            assert (await alice.post(f"/deployments/{dep['id']}/stop")).status_code == 409
            bad = await alice.post("/deployments/1/generate", json={"prompt": "", "max_new_tokens": 9999})
            assert bad.status_code == 422
    run(scenario())
