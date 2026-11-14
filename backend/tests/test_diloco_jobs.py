"""DiLoCo full fine-tuning jobs: replica planning (each GPU holds the whole model), replica workers, replacement keeps the
slot, the full model lands in the registry ready to deploy without a merge; plus the replica file endpoints."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import httpx

from tests.platform_helper import Platform, wait_for


def test_diloco_job_plans_replicas_and_registers_a_full_model(tmp_path):
    async def scenario():
        async with Platform(tmp_path, runner_spec_extra={"fake_step_delay": 0.3}) as p:
            alice, lab, mac = await p.user("alice"), await p.user("lab"), await p.user("mac")
            await p.provider(mac, "small-mac", gpus=1, memory_mb=6000)                       # too small for a full replica
            await p.provider(lab, "lab", gpus=3, memory_mb=24000, chip="NVIDIA RTX A6000", device="cuda")
            await wait_for(lambda: asyncio.sleep(0, result=len(p.gpus_sync()) == 4), what="GPUs")
            base = {"base_model_id": 1, "dataset_id": 1, "method": "diloco", "steps": 6, "inner_steps": 2, "lr": 1e-5}
            assert (await alice.post("/jobs", json={**base, "weight_bits": 4})).status_code == 422
            assert (await alice.post("/jobs", json={**base, "lr": 0.005})).status_code == 422
            preview = (await alice.get("/plans", params={"base_model_id": 1, "requested_gpus": 2, "method": "diloco"})).json()
            assert preview["fits"] and [s["device"] for s in preview["stages"]] == ["cuda", "cuda"]  # the 6 GB Mac cannot hold a replica
            assert all(s["memory_needed_mb"] > 22 * 100 * 4 for s in preview["stages"])     # full-FT memory, not a layer slice

            job = (await alice.post("/jobs", json={**base, "requested_gpus": 2, "name": "Full FT"})).json()
            assert [(l["layer_start"], l["layer_end"]) for l in job["leases"]] == [(0, 1), (1, 2)]  # replica slots
            await wait_for(lambda: asyncio.sleep(0, result=len(p.worker_specs) >= 2), what="replicas started")
            assert {(s.kind, s.lease_id > 0, s.job_id) for s in p.worker_specs} == {("diloco", True, job["id"])}

            # a replica dies: the replacement gets the same slot (and with it the same data shard)
            await p.wait_event(alice, job["id"], "worker.online", count=2)
            assert (await alice.post(f"/jobs/{job['id']}/kill-worker", json={"layer_start": 1})).status_code == 200
            repl = (await p.wait_event(alice, job["id"], "worker.replacement_starting"))[0]["payload"]
            assert (repl["layer_start"], repl["layer_end"]) == (1, 2) and repl["gpu_device"] == "cuda"

            done = await p.wait_status(alice, job["id"], "completed", "failed", timeout=60)
            assert done["status"] == "completed", done["result"]
            model = next(m for m in (await alice.get("/models")).json() if m["name"] == "Full FT")
            assert model["method"] == "full"
            from sqlmodel import Session
            from app.models import FineTunedModel
            with Session(p.app.state.engine) as s:
                row = s.get(FineTunedModel, model["id"])
                assert row.adapter_path == "" and (Path(row.merged_path) / "model.safetensors").exists()
            dep = await p.deploy(alice, model_kind="finetuned", finetuned_model_id=model["id"], requested_gpus=1)
            kinds = [e["type"] for e in await p.events(alice, dep["job"]["id"])]
            assert "deploy.merging" not in kinds  # nothing to merge for a full model
    asyncio.run(scenario())


def test_replica_file_endpoints_are_lease_scoped(tmp_path):
    async def scenario():
        async with Platform(tmp_path, runner_spec_extra={"fake_step_delay": 5}) as p:
            alice, lab, other = await p.user("alice"), await p.user("lab"), await p.user("other")
            lab_agent = await p.provider(lab, "lab", gpus=1, memory_mb=24000, chip="NVIDIA RTX A6000", device="cuda")
            other_agent = await p.provider(other, "other", gpus=1, memory_mb=24000, chip="NVIDIA RTX A6000", device="cuda")
            await wait_for(lambda: asyncio.sleep(0, result=len(p.gpus_sync()) == 2), what="GPUs")
            job = (await alice.post("/jobs", json={"base_model_id": 1, "dataset_id": 1, "method": "diloco", "steps": 4,
                                                   "inner_steps": 2, "lr": 1e-5, "requested_gpus": 1})).json()
            lease = job["leases"][0]
            # which agent holds the lease?
            gpu = next(g for g in p.gpus_sync() if g.id == lease["gpu_id"])
            mine, theirs = (lab_agent, other_agent) if gpu.provider_id == (await lab.get("/providers")).json()[0]["id"] else (other_agent, lab_agent)
            c = lambda a: httpx.AsyncClient(base_url=p.url, headers={"Authorization": f"Bearer {a.s.token}"})
            async with c(mine) as m, c(theirs) as t:
                q = {"lease_id": lease["id"]}
                assert (await t.get(f"/agent/diloco/{job['id']}/state", params=q)).status_code == 404   # not their lease
                assert (await m.get(f"/agent/diloco/{job['id']}/state", params=q)).status_code == 409   # nothing published yet
                root = p.tmp / "jobs" / str(job["id"]) / "diloco"
                (root / "rounds" / "0").mkdir(parents=True)
                (root / "state.json").write_text(json.dumps({"round": 0, "rounds": 2, "status": "running", "deadline": 0}))
                st = (await m.get(f"/agent/diloco/{job['id']}/state", params=q)).json()
                assert st["slot"] == 0 and st["replicas"] == 1 and st["inner_steps"] == 2
                assert (await m.get(f"/agent/diloco/{job['id']}/data", params=q)).text.startswith('{"text"')
                body = b"delta-bytes"
                good = {**q, "sha256": hashlib.sha256(body).hexdigest(), "tokens": 128, "loss": 2.5}
                assert (await m.put(f"/agent/diloco/{job['id']}/delta/0", params={**good, "sha256": "0" * 64}, content=body)).status_code == 422
                assert (await m.put(f"/agent/diloco/{job['id']}/delta/1", params=good, content=body)).status_code == 409
                assert (await m.put(f"/agent/diloco/{job['id']}/delta/0", params=good, content=body)).status_code == 200
                assert (await m.put(f"/agent/diloco/{job['id']}/delta/0", params=good, content=body)).status_code == 200  # retry ok
                meta = json.loads((root / "rounds" / "0" / "delta-slot0.json").read_text())
                assert meta["tokens"] == 128 and (root / "rounds" / "0" / "delta-slot0.safetensors").read_bytes() == body
                assert not list((root / "rounds" / "0").glob("*.part"))
            await alice.post(f"/jobs/{job['id']}/cancel")
    asyncio.run(scenario())
