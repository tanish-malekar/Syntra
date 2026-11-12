"""Remote model delivery: only the agent holding a lease for a deployment of the model may download it; hashes verified."""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import httpx
import pytest
from sqlmodel import Session, select

from app.models import FineTunedModel, Gpu, Job, Lease
from tests.platform_helper import Platform, wait_for

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent"))
from syntra_agent.files import ensure_model_files  # noqa: E402


def test_agent_downloads_a_merged_model_only_while_it_holds_a_lease(tmp_path):
    async def scenario():
        async with Platform(tmp_path, runner_spec_extra={"fake_step_delay": 0.05}) as p:
            alice, lab, other = await p.user("alice"), await p.user("lab"), await p.user("other")
            await p.provider(lab, "lab", gpus=2)
            await p.provider(other, "other-lab", gpus=1)
            await wait_for(lambda: asyncio.sleep(0, result=len(p.gpus_sync()) == 3), what="GPUs")
            model = await p.train_model(alice, requested_gpus=1, steps=3)
            dep = await p.deploy(alice, model_kind="finetuned", finetuned_model_id=model["id"], requested_gpus=1)  # creates merged/

            with Session(p.app.state.engine) as s:
                merged = Path(s.get(FineTunedModel, model["id"]).merged_path)
                lease_gpu = s.exec(select(Gpu).where(Gpu.id == dep["job"]["leases"][0]["gpu_id"])).one()
                token_id = lease_gpu.agent_token_id
            (merged / "sub").mkdir(exist_ok=True)
            (merged / "sub" / "extra.bin").write_bytes(b"x" * 5000)

            def agent_http(agent):  # an httpx client carrying that agent's bearer token
                return httpx.AsyncClient(base_url=p.url, headers={"Authorization": f"Bearer {agent.s.token}"}, timeout=30)
            owner_agent = next(a for a in p.agents if a.gpu_ids and token_id and a is not None and any(
                g.agent_token_id == token_id for g in p.gpus_sync() if g.id in a.gpu_ids.values()))
            stranger = next(a for a in p.agents if a is not owner_agent)

            async with agent_http(owner_agent) as http:
                files = (await http.get(f"/files/models/{model['id']}/manifest")).json()["files"]
                assert {f["path"] for f in files} >= {"config.json", "model.safetensors", "sub/extra.bin"}
                for f in files:
                    assert f["sha256"] == hashlib.sha256((merged / f["path"]).read_bytes()).hexdigest()
                assert (await http.get(f"/files/models/{model['id']}/file/../../../etc/passwd")).status_code == 404
                assert (await http.get(f"/files/models/{model['id']}/file/nope.bin")).status_code == 404
                dest = await ensure_model_files(http, model["id"], tmp_path / "agent-models" / "m")
                assert (dest / "sub" / "extra.bin").read_bytes() == b"x" * 5000
                (dest / "model.safetensors").write_bytes(b"corrupted")          # a damaged copy is detected and repaired
                await ensure_model_files(http, model["id"], dest)
                assert (dest / "model.safetensors").read_bytes() == (merged / "model.safetensors").read_bytes()
            async with agent_http(stranger) as http:                              # another provider's agent: no lease, no files
                assert (await http.get(f"/files/models/{model['id']}/manifest")).status_code == 404
            assert (await httpx.AsyncClient(base_url=p.url).get(f"/files/models/{model['id']}/manifest")).status_code == 401

            # once the deployment stops, the lease ends and the files are no longer served
            assert (await alice.post(f"/deployments/{dep['id']}/stop")).status_code == 200
            await wait_for(lambda: asyncio.sleep(0, result={g.status for g in p.gpus_sync()} == {"idle"}), what="released")
            async with agent_http(owner_agent) as http:
                assert (await http.get(f"/files/models/{model['id']}/manifest")).status_code == 404
    asyncio.run(scenario())
