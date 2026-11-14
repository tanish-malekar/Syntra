"""The whole control plane: users, providers with real agents (fake workers), scheduler, orchestrator, events,
model registry and billing, driven over real HTTP. The swarm runner is a fake that speaks the same protocol."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from sqlmodel import Session, select

from app.models import FineTunedModel, Gpu, Lease, LedgerEntry
from tests.platform_helper import Platform, wait_for


def run(coro):
    return asyncio.run(coro)


def types(events):
    return [e["type"] for e in events]


def test_training_job_runs_end_to_end_and_lands_in_the_registry(tmp_path):
    async def scenario():
        async with Platform(tmp_path, runner_spec_extra={"fake_step_delay": 0.1}) as p:
            alice, bob = await p.user("alice"), await p.user("bob")
            await p.provider(bob, "bob-macs", gpus=3)
            await wait_for(lambda: len([g for g in p.app_gpus() if g.status == "idle"]) == 3, what="3 idle GPUs") if False else None
            await wait_for(lambda: asyncio.sleep(0, result=len(p.gpus_sync()) == 3), what="3 GPUs registered")

            plan = (await alice.get("/plans", params={"base_model_id": 1, "requested_gpus": 2})).json()
            assert plan["fits"] and [(s["layer_start"], s["layer_end"]) for s in plan["stages"]] == [(0, 11), (11, 22)]

            job = await p.start_job(alice, requested_gpus=2, steps=6, name="My ticket model")
            assert job["status"] in ("starting", "queued", "running") and len(job["leases"]) == 2
            done = await p.wait_status(alice, job["id"], "completed", "failed")
            assert done["status"] == "completed", done["result"]

            ev = await p.events(alice, job["id"])
            kinds = types(ev)
            assert kinds.count("lease.created") == 2 and kinds.count("worker.starting") == 2 and kinds.count("worker.online") == 2
            steps = [e["payload"] for e in ev if e["type"] == "train.step"]
            assert [s["step"] for s in steps] == [1, 2, 3, 4, 5, 6] and steps[-1]["loss"] < steps[0]["loss"]
            assert kinds[-1] == "train.completed"
            assert kinds.index("worker.online") > kinds.index("worker.starting")

            # stored for later: the registry holds the adapter, owned by alice only
            models = (await alice.get("/models")).json()
            assert len(models) == 1 and models[0]["name"] == "My ticket model" and models[0]["job_id"] == job["id"]
            adapter = Path(p.app.state.settings.registry_dir) / str(models[0]["id"]) / "adapter"
            assert (adapter / "adapter_config.json").exists()
            assert (await bob.get("/models")).json() == []
            assert (await bob.get(f"/models/{models[0]['id']}")).status_code == 404

            # GPUs are free again and both sides were billed
            assert {g["status"] for g in await p.pool(alice)} == {"idle"}
            costs, earnings = (await alice.get("/ledger/costs")).json(), (await bob.get("/ledger/earnings")).json()
            assert len(costs["entries"]) == 2 and costs["total"] == pytest.approx(earnings["total"], abs=1e-6) and costs["total"] > 0
            assert all(l["active"] is False for l in (await p.job(alice, job["id"]))["leases"])
    run(scenario())


def test_killing_a_worker_finds_a_new_gpu_and_the_run_continues(tmp_path):
    async def scenario():
        async with Platform(tmp_path, runner_spec_extra={"fake_step_delay": 0.5}) as p:
            alice, bob = await p.user("alice"), await p.user("bob")
            await p.provider(bob, "bob-macs", gpus=3)
            await wait_for(lambda: asyncio.sleep(0, result=len(p.gpus_sync()) == 3), what="GPUs")
            job = await p.start_job(alice, requested_gpus=2, steps=12)
            jid = job["id"]
            await p.wait_event(alice, jid, "worker.online", count=2)
            await p.wait_event(alice, jid, "train.step", count=2)
            before = {l["gpu_id"] for l in (await p.job(alice, jid))["leases"] if l["active"]}

            r = await alice.post(f"/jobs/{jid}/kill-worker", json={"layer_start": 11})
            assert r.status_code == 200
            down = (await p.wait_event(alice, jid, "worker.down"))[0]["payload"]
            assert down["reason"] == "killed_by_user" and (down["layer_start"], down["layer_end"]) == (11, 22)
            starting = (await p.wait_event(alice, jid, "worker.replacement_starting"))[0]
            assert starting["payload"]["gpu_class"] == "Apple M-series" and starting["payload"]["gpu_device"] == "mps"
            online = (await p.wait_event(alice, jid, "worker.replacement_online"))[0]["payload"]
            assert (online["layer_start"], online["layer_end"]) == (11, 22) and online["seconds_since_down"] >= 0

            done = await p.wait_status(alice, jid, "completed", "failed")
            assert done["status"] == "completed"
            leases = done["leases"]
            killed = next(l for l in leases if l["end_reason"] == "killed_by_user")
            new = next(l for l in leases if l["layer_start"] == 11 and l["id"] != killed["id"])
            assert new["gpu_id"] != killed["gpu_id"] and new["gpu_id"] not in before   # a different, previously idle GPU
            steps = [e["payload"]["step"] for e in await p.events(alice, jid) if e["type"] == "train.step"]
            assert steps[-1] == 12
            assert len((await alice.get("/ledger/costs")).json()["entries"]) == 3          # dead lease billed too
    run(scenario())


def test_two_users_run_at_the_same_time_on_different_gpus_and_extra_jobs_queue(tmp_path):
    async def scenario():
        async with Platform(tmp_path, runner_spec_extra={"fake_step_delay": 0.4}) as p:
            alice, bob, carol, owner = [await p.user(n) for n in ("alice", "bob", "carol", "owner")]
            await p.provider(owner, "lab", gpus=3)
            await wait_for(lambda: asyncio.sleep(0, result=len(p.gpus_sync()) == 3), what="GPUs")

            a = await p.start_job(alice, requested_gpus=2, steps=10)
            c = await p.start_job(carol, requested_gpus=1, steps=6)    # fits in the 3rd GPU, runs at the same time
            b = await p.start_job(bob, requested_gpus=2, steps=6)      # nothing left: waits in the queue
            assert (b["status"], b["queue_position"]) == ("queued", 1)
            assert (await p.events(bob, b["id"]))[0]["type"] == "job.queued"

            await p.wait_event(alice, a["id"], "worker.online", count=2)
            await p.wait_event(carol, c["id"], "worker.online", count=1)
            used_a = {l["gpu_id"] for l in (await p.job(alice, a["id"]))["leases"]}
            used_c = {l["gpu_id"] for l in (await p.job(carol, c["id"]))["leases"]}
            assert len(used_a) == 2 and len(used_c) == 1 and not (used_a & used_c)

            # nobody sees anybody else's jobs, events or leases
            assert (await bob.get(f"/jobs/{a['id']}")).status_code == 404
            assert (await alice.get("/events", params={"job_id": c["id"]})).json() == []
            assert [j["id"] for j in (await alice.get("/jobs")).json()] == [a["id"]]

            for user, job in ((carol, c), (alice, a), (bob, b)):
                done = await p.wait_status(user, job["id"], "completed", "failed", timeout=60)
                assert done["status"] == "completed"
            assert (await p.job(bob, b["id"]))["queue_position"] == 0
            kinds = types(await p.events(bob, b["id"]))
            assert kinds[0] == "job.queued" and "lease.created" in kinds and kinds[-1] == "train.completed"
            assert len((await bob.get("/models")).json()) == 1 and len((await alice.get("/models")).json()) == 1
    run(scenario())


def test_cancel_releases_gpus_and_a_crashing_runner_fails_cleanly(tmp_path):
    async def scenario():
        async with Platform(tmp_path, runner_spec_extra={"fake_step_delay": 1.0}) as p:
            alice, owner = await p.user("alice"), await p.user("owner")
            await p.provider(owner, "lab", gpus=2)
            await wait_for(lambda: asyncio.sleep(0, result=len(p.gpus_sync()) == 2), what="GPUs")
            job = await p.start_job(alice, requested_gpus=2, steps=30)
            await p.wait_event(alice, job["id"], "train.step", count=1)
            assert (await alice.post(f"/jobs/{job['id']}/cancel")).status_code == 200
            done = await p.wait_status(alice, job["id"], "stopped")
            assert not any(l["active"] for l in done["leases"])
            await wait_for(lambda: asyncio.sleep(0, result={g.status for g in p.gpus_sync()} == {"idle"}), what="GPUs idle after cancel")
            assert (await alice.get("/models")).json() == []                     # nothing registered for a cancelled run
    run(scenario())


def test_runner_failure_marks_the_job_failed_and_frees_the_gpus(tmp_path):
    async def scenario():
        async with Platform(tmp_path, runner_spec_extra={"fake_fail": True}) as p:
            alice, owner = await p.user("alice"), await p.user("owner")
            await p.provider(owner, "lab", gpus=2)
            await wait_for(lambda: asyncio.sleep(0, result=len(p.gpus_sync()) == 2), what="GPUs")
            job = await p.start_job(alice, steps=5)
            done = await p.wait_status(alice, job["id"], "failed")
            failed = (await p.wait_event(alice, job["id"], "train.failed"))[0]["payload"]
            assert "boom" in failed["error"] and "boom" in done["result"]["error"]
            assert (await alice.get("/models")).json() == []
            await wait_for(lambda: asyncio.sleep(0, result={g.status for g in p.gpus_sync()} == {"idle"}), what="GPUs idle")
    run(scenario())


def test_job_validation_datasets_and_plan_preview(tmp_path):
    async def scenario():
        async with Platform(tmp_path) as p:
            alice, bob, owner = await p.user("alice"), await p.user("bob"), await p.user("owner")
            base = {"base_model_id": 1, "dataset_id": 1}
            assert (await alice.post("/jobs", json={**base, "requested_gpus": 23})).status_code == 422
            assert (await alice.post("/jobs", json={**base, "steps": 100000})).status_code == 422
            assert (await alice.post("/jobs", json={**base, "base_model_id": 99})).status_code == 404
            assert (await alice.post("/jobs", json={**base, "dataset_id": 99})).status_code == 404

            # no GPUs yet: the preview says so and a job queues
            preview = (await alice.get("/plans", params={"base_model_id": 1})).json()
            assert preview["fits"] is False and "queue" in preview["reason"] and preview["idle_gpus"] == 0
            assert (await alice.get("/plans", params={"base_model_id": 1, "requested_gpus": 40})).status_code == 422

            # dataset upload: validation and visibility
            good = "\n".join(json.dumps({"text": f"Ticket {i}\nAnswer: x"}) for i in range(10))
            files = lambda content: {"file": ("d.jsonl", content.encode(), "application/jsonl")}
            for bad in ("not json\n" * 10, "\n".join(json.dumps({"txt": "x"}) for _ in range(10)), good.split("\n")[0]):
                assert (await alice.post("/datasets", data={"name": "bad"}, files=files(bad))).status_code == 422
            ok = await alice.post("/datasets", data={"name": "mine"}, files=files(good))
            assert ok.status_code == 201 and ok.json()["num_rows"] == 10
            names = [d["name"] for d in (await alice.get("/datasets")).json()]
            assert names == ["Support tickets (demo)", "mine"]
            assert [d["name"] for d in (await bob.get("/datasets")).json()] == ["Support tickets (demo)"]
            assert (await bob.post("/jobs", json={**base, "dataset_id": ok.json()["id"]})).status_code == 404

            big = "x" * (10 * 1024 * 1024 + 10)
            assert (await alice.post("/datasets", data={"name": "big"}, files=files(big))).status_code == 413

            # a queued job can be cancelled before it ever starts
            job = await p.start_job(alice, dataset_id=ok.json()["id"])
            assert job["status"] == "queued"
            assert (await alice.post(f"/jobs/{job['id']}/cancel")).status_code == 200
            assert (await p.job(alice, job["id"]))["status"] == "stopped"
    run(scenario())


def test_qlora_trains_and_deploys_with_4bit_base_weights_on_capable_agents_only(tmp_path):
    async def scenario():
        async with Platform(tmp_path, runner_spec_extra={"fake_step_delay": 0.05}) as p:
            alice, bob, carol = await p.user("alice"), await p.user("bob"), await p.user("carol")
            await p.provider(bob, "bob-old", gpus=2)                       # agent without 4-bit support
            await wait_for(lambda: asyncio.sleep(0, result=len(p.gpus_sync()) == 2), what="2 GPUs registered")
            base = {"base_model_id": 1, "dataset_id": 1}
            assert (await alice.post("/jobs", json={**base, "weight_bits": 8})).status_code == 422
            preview = (await alice.get("/plans", params={"base_model_id": 1, "weight_bits": 4})).json()
            assert preview["fits"] is False and "4-bit" in preview["reason"]

            await p.provider(carol, "carol-new", gpus=2, supports_4bit=True)
            await wait_for(lambda: asyncio.sleep(0, result=len(p.gpus_sync()) == 4), what="4 GPUs registered")
            assert sum(g.supports_4bit for g in p.gpus_sync()) == 2
            preview = (await alice.get("/plans", params={"base_model_id": 1, "weight_bits": 4})).json()
            assert preview["fits"] and len(preview["stages"]) == 1           # the whole model fits one GPU in 4-bit

            model = await p.train_model(alice, steps=4, weight_bits=4, requested_gpus=2)
            assert model["weight_bits"] == 4
            train_specs = [s for s in p.worker_specs if s.mode == "train"]
            assert len(train_specs) == 2 and {s.weight_bits for s in train_specs} == {4}
            carol_ids = {g.id for g in p.gpus_sync() if g.supports_4bit}
            job = await p.job(alice, model["job_id"])
            assert job["config"]["weight_bits"] == 4 and {l["gpu_id"] for l in job["leases"]} == carol_ids

            # deploy the QLoRA model in 4-bit, and the base model in float16
            dep = await p.deploy(alice, model_kind="finetuned", finetuned_model_id=model["id"], requested_gpus=1, weight_bits=4)
            assert dep["weight_bits"] == 4
            assert [s.weight_bits for s in p.worker_specs if s.mode == "serve"] == [4]
            base_dep = await p.deploy(alice, model_kind="base", base_model_id=1, requested_gpus=1)
            assert base_dep["weight_bits"] == 16
            assert [s.weight_bits for s in p.worker_specs if s.mode == "serve"] == [4, 16]
    run(scenario())
