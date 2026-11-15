"""RL sessions (Tinker-style calls), API keys for the SDK, and managed DPO jobs, over real HTTP with the fake runner."""
from __future__ import annotations

import asyncio
import json

import httpx

from tests.platform_helper import Platform, wait_for


def run(coro):
    return asyncio.run(coro)


def prefs(n=10, same=False):
    rows = [{"prompt": f"Ticket {i}: refund please\nAnswer:", "chosen": ' {"department": "billing"}',
             "rejected": ' {"department": "billing"}' if same else " I don't know."} for i in range(n)]
    return "\n".join(json.dumps(r) for r in rows)


async def upload(client, name, content):
    return await client.post("/datasets", data={"name": name}, files={"file": ("d.jsonl", content.encode(), "application/jsonl")})


async def ready_session(p, client, **body):
    r = await client.post("/sessions", json={"base_model_id": 1, **body})
    assert r.status_code == 201, r.text
    sid = r.json()["id"]

    async def is_ready():
        s = (await client.get(f"/sessions/{sid}")).json()
        return s if s["ready"] else None
    await wait_for(is_ready, 30, f"session {sid} ready")
    return sid


def test_api_keys_and_a_full_rl_session(tmp_path):
    async def scenario():
        async with Platform(tmp_path) as p:
            alice, bob = await p.user("alice"), await p.user("bob")
            await p.provider(bob, "bob-macs", gpus=2)
            await wait_for(lambda: asyncio.sleep(0, result=len(p.gpus_sync()) == 2), what="GPUs")

            key = (await alice.post("/api-keys", json={"label": "laptop"})).json()
            assert key["key"].startswith("syn_key_")
            sdk = httpx.AsyncClient(base_url=p.url, headers={"Authorization": f"Bearer {key['key']}"}, timeout=30)
            assert (await sdk.get("/auth/me")).json()["email"].startswith("alice")

            sid = await ready_session(p, sdk, rank=8, requested_gpus=2, name="grpo-demo")
            assert {s.mode for s in p.worker_specs} == {"train"}             # LoRA RPCs enabled on the workers
            call = lambda op, **args: sdk.post(f"/sessions/{sid}/call", json={"op": op, "args": args})

            toks = (await call("tokenize", texts=["hello world"])).json()["data"]["tokens"]
            samples = (await call("sample", prompts=toks, n=4, max_new_tokens=5, seed=3)).json()["data"]["samples"]
            assert len(samples) == 4 and all(len(s["logprobs"]) == 5 for s in samples)
            adv = (await call("advantages", rewards=[1, 0, 0, 1], group_size=4)).json()["data"]["advantages"]
            batch = [{"tokens": s["tokens"], "prompt_len": s["prompt_len"], "advantage": a, "old_logprobs": s["logprobs"]}
                     for s, a in zip(samples, adv)]
            assert (await call("forward_backward", loss="ppo", batch=batch, clip=0.2)).status_code == 200

            # a retried optim_step with the same idempotency key is applied once
            body = {"op": "optim_step", "args": {"lr": 1e-5}, "idempotency_key": "step-1"}
            first = (await sdk.post(f"/sessions/{sid}/call", json=body)).json()
            again = (await sdk.post(f"/sessions/{sid}/call", json=body)).json()
            assert first == again and first["data"]["step"] == 1
            assert (await call("state")).json()["data"]["step"] == 1

            # bad requests are 422 and the session keeps going
            assert (await call("forward_backward", loss="nope", batch=batch)).status_code == 422
            assert (await call("optim_step", lr=1e-5)).status_code == 422          # nothing accumulated
            assert (await sdk.post(f"/sessions/{sid}/call", json={"op": "save", "args": {}})).status_code == 422
            assert (await call("state")).status_code == 200

            # other users cannot see or drive it
            assert (await bob.get(f"/sessions/{sid}")).status_code == 404
            assert (await bob.post(f"/sessions/{sid}/call", json={"op": "state"})).status_code == 404

            saved = (await sdk.post(f"/sessions/{sid}/save", json={"name": "grpo step 1"})).json()
            closed = (await sdk.post(f"/sessions/{sid}/close", json={"save_as": "grpo final"})).json()
            models = {m["name"]: m for m in (await alice.get("/models")).json()}
            assert set(models) == {"grpo step 1", "grpo final"} and closed["model_id"] == models["grpo final"]["id"]
            assert saved["model_id"] == models["grpo step 1"]["id"] and models["grpo final"]["method"] == "session"
            await wait_for(lambda: asyncio.sleep(0, result=all(g.status == "idle" for g in p.gpus_sync())), what="GPUs back")
            s = (await sdk.get(f"/sessions/{sid}")).json()
            assert s["status"] == "stopped" and not s["ready"]
            assert (await call("state")).status_code == 409
            assert (await alice.get("/ledger/costs")).json()["total"] > 0     # billed like training

            # a revoked key stops working at once
            assert (await alice.delete(f"/api-keys/{key['id']}")).status_code == 204
            assert (await sdk.get("/sessions")).status_code == 401
            await sdk.aclose()
    run(scenario())


def test_an_idle_session_is_saved_and_closed(tmp_path):
    async def scenario():
        async with Platform(tmp_path, session_idle_seconds=1.0) as p:
            alice, bob = await p.user("alice"), await p.user("bob")
            await p.provider(bob, "bob", gpus=1)
            await wait_for(lambda: asyncio.sleep(0, result=len(p.gpus_sync()) == 1), what="GPU")
            sid = await ready_session(p, alice, name="forgotten")
            await alice.post(f"/sessions/{sid}/call", json={"op": "state"})

            async def closed():
                s = (await alice.get(f"/sessions/{sid}")).json()
                return s if s["status"] == "stopped" else None
            await wait_for(closed, 20, "idle session closed")
            names = [m["name"] for m in (await alice.get("/models")).json()]
            assert names == ["forgotten (auto-saved)"]
            ev = [e for e in (await alice.get("/events", params={"job_id": sid})).json() if e["type"] == "session.closed"]
            assert ev and ev[-1]["payload"]["reason"] == "idle"
    run(scenario())


def test_dpo_job_on_preference_data_continuing_from_an_sft_model(tmp_path):
    async def scenario():
        async with Platform(tmp_path, runner_spec_extra={"fake_step_delay": 0.02}) as p:
            alice, bob = await p.user("alice"), await p.user("bob")
            await p.provider(bob, "bob", gpus=2)
            await wait_for(lambda: asyncio.sleep(0, result=len(p.gpus_sync()) == 2), what="GPUs")

            assert (await upload(alice, "same", prefs(same=True))).status_code == 422
            bad = "\n".join(json.dumps({"prompt": "p", "chosen": "a"}) for _ in range(10))
            assert (await upload(alice, "half", bad)).status_code == 422   # first line is not preference -> text check fails
            pref = (await upload(alice, "prefs", prefs())).json()
            assert pref["format"] == "preference"

            sft = await p.train_model(alice, steps=3, name="SFT v1")
            assert sft["method"] == "sft"
            base = {"base_model_id": 1, "steps": 4}
            assert (await alice.post("/jobs", json={**base, "dataset_id": 1, "method": "dpo"})).status_code == 422
            assert (await alice.post("/jobs", json={**base, "dataset_id": pref["id"]})).status_code == 422
            assert (await alice.post("/jobs", json={**base, "dataset_id": pref["id"], "method": "dpo", "init_model_id": 999})).status_code == 404
            assert (await alice.post("/jobs", json={**base, "dataset_id": pref["id"], "method": "dpo", "beta": 0})).status_code == 422

            job = (await alice.post("/jobs", json={**base, "dataset_id": pref["id"], "method": "dpo", "beta": 0.2,
                                                   "init_model_id": sft["id"], "name": "DPO v1"})).json()
            done = await p.wait_status(alice, job["id"], "completed", "failed", timeout=60)
            assert done["status"] == "completed" and done["config"]["method"] == "dpo" and done["config"]["beta"] == 0.2
            steps = [e["payload"] for e in await p.events(alice, job["id"]) if e["type"] == "train.step"]
            assert len(steps) == 4 and steps[-1]["reward_margin"] > steps[0]["reward_margin"] and "reward_accuracy" in steps[-1]
            dpo = next(m for m in (await alice.get("/models")).json() if m["name"] == "DPO v1")
            assert dpo["method"] == "dpo" and dpo["parent_model_id"] == sft["id"]
            spec = json.loads((p.tmp / "jobs" / str(job["id"]) / "spec.json").read_text())
            assert spec["init_adapter"].endswith(f"/{sft['id']}/adapter")
    run(scenario())
