"""The `bloombee_sdk` SDK against the real backend (fake runner): GRPO, PPO and DPO loops, retries, errors."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk"))
import bloombee_sdk  # noqa: E402

from tests.platform_helper import Platform, wait_for  # noqa: E402


def test_sdk_grpo_ppo_dpo_save_and_close(tmp_path):
    async def scenario():
        async with Platform(tmp_path) as p:
            alice, bob = await p.user("alice"), await p.user("bob")
            await p.provider(bob, "bob", gpus=2, supports_4bit=True)
            await wait_for(lambda: asyncio.sleep(0, result=len(p.gpus_sync()) == 2), what="GPUs")
            key = (await alice.post("/api-keys", json={"label": "sdk"})).json()["key"]

            def use_sdk():
                with pytest.raises(bloombee_sdk.NotSignedIn):
                    bloombee_sdk.Client(api_key="", base_url=p.url)
                with pytest.raises(bloombee_sdk.NotSignedIn):
                    bloombee_sdk.Client(api_key="syn_key_wrong", base_url=p.url).models()
                client = bloombee_sdk.Client(api_key=key, base_url=p.url)
                with pytest.raises(bloombee_sdk.BloomBeeError, match="not ready"):  # never hang forever on a session that cannot start
                    client.training_session("TinyLlama-1.1B-Chat", weight_bits=4, gpus=5, timeout=2)
                gone = client.request("GET", "/sessions")
                assert gone and gone[0]["status"] == "stopped"  # the timed-out session was cancelled, not left queued
                with pytest.raises(bloombee_sdk.BadRequest):
                    client.training_session("no-such-model")
                with client.training_session("TinyLlama-1.1B-Chat", rank=8, weight_bits=4, gpus=1, name="sdk run") as tc:
                    seen = []
                    reward = lambda prompt, completion, s: float(len(completion) % 3)
                    hist = bloombee_sdk.rl.grpo(tc, ["a b", "c d", "e f"], reward, iterations=3, group_size=4,
                                          prompts_per_iteration=2, max_new_tokens=5, seed=1, on_iteration=seen.append)
                    assert len(hist) == 3 and seen == hist and all("reward_mean" in h for h in hist)
                    hist = bloombee_sdk.rl.ppo(tc, ["a b", "c d"], reward, iterations=2, samples_per_prompt=2, epochs=2, seed=2)
                    assert len(hist) == 2 and hist[-1]["step"] >= 5
                    pairs = [{"prompt": f"q{i}", "chosen": "good answer", "rejected": "bad"} for i in range(5)]
                    hist = bloombee_sdk.rl.dpo(tc, pairs, beta=0.1, batch_size=2, epochs=1)
                    assert len(hist) == 3
                    with pytest.raises(bloombee_sdk.BadRequest):
                        bloombee_sdk.rl.grpo(tc, ["a"], lambda *a: float("nan"), iterations=1, group_size=2)
                    with pytest.raises(bloombee_sdk.BadRequest):
                        tc.forward_backward([{"tokens": [1, 2], "prompt_len": 1}], loss="nope")
                    s = tc.sample(["x y"], n=2, max_new_tokens=3, seed=5)
                    assert len(s) == 2 and isinstance(s[0], bloombee_sdk.Sample) and len(tc.logprobs(s)) == 2
                    model_id = tc.save("sdk-grpo")
                    step = tc.state()["step"]
                assert {m["id"]: m["name"] for m in client.models()}[model_id] == "sdk-grpo"
                with pytest.raises(bloombee_sdk.SessionNotRunning):
                    tc.state()
                return step
            step = await asyncio.to_thread(use_sdk)
            assert step >= 8
            await wait_for(lambda: asyncio.sleep(0, result=all(g.status == "idle" for g in p.gpus_sync())), what="GPUs back")
            assert {s.weight_bits for s in p.worker_specs} == {4}
    asyncio.run(scenario())
