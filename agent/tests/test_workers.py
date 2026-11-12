from __future__ import annotations

import asyncio
import sys

import pytest

from syntra_agent.workers import AgentConfig, WorkerManager, WorkerSpec, build_bloombee_command, resolve_model
from pathlib import Path

FAKE_WORKER = Path(__file__).resolve().parent / "fake_worker.py"


def spec(worker_id="w1", **kw) -> WorkerSpec:
    base = dict(worker_id=worker_id, device="cuda", device_index=2, model={"name": "tinyllama", "hf_id": "TinyLlama/x"},
                layer_start=11, layer_end=22, dht_prefix="job7", initial_peers=["/ip4/127.0.0.1/tcp/31001/p2p/QmA"])
    base.update(kw)
    return WorkerSpec(**base)


def fake_builder(extra=()):
    def build(s, cfg):
        return [sys.executable, str(FAKE_WORKER), "--peer", f"12D3KooW{s.worker_id}", *extra], {}
    return build


async def collect(manager_events: list, want: str, timeout=5.0):
    async def wait():
        while not any(e["type"] == want for e in manager_events):
            await asyncio.sleep(0.02)
    await asyncio.wait_for(wait(), timeout)


def test_bloombee_command_for_training_on_cuda(tmp_path):
    cfg = AgentConfig(bloombee_dir=tmp_path / "BB", python="/py/bin/python", models_dir=tmp_path / "models")
    argv, env = build_bloombee_command(spec(host_maddr="/ip4/127.0.0.1/tcp/31003"), cfg)
    flag = lambda name: argv[argv.index(name) + 1]
    assert argv[:3] == ["/py/bin/python", "-m", "bloombee.cli.run_server"]
    assert argv[3] == "TinyLlama/x"                              # no local copy -> Hugging Face id
    assert flag("--block_indices") == "11:22" and flag("--dht_prefix") == "job7" and flag("--device") == "cuda"
    assert flag("--host_maddrs") == "/ip4/127.0.0.1/tcp/31003" and "--no_auto_relay" in argv
    assert argv[argv.index("--initial_peers") + 1] == "/ip4/127.0.0.1/tcp/31001/p2p/QmA"
    assert env["BLOOMBEE_ENABLE_LORA"] == "1" and env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["PYTHONPATH"].startswith(str(tmp_path / "BB" / "src"))


def test_bloombee_command_for_serving_on_mps_with_local_model(tmp_path):
    (tmp_path / "models" / "tinyllama").mkdir(parents=True)
    cfg = AgentConfig(bloombee_dir=tmp_path / "BB", models_dir=tmp_path / "models")
    argv, env = build_bloombee_command(spec(device="mps", device_index=0, mode="serve"), cfg)
    assert argv[3] == str(tmp_path / "models" / "tinyllama") and env["HF_HUB_OFFLINE"] == "1"
    assert env["BLOOMBEE_ENABLE_LORA"] == "0" and "CUDA_VISIBLE_DEVICES" not in env
    assert resolve_model({"name": "missing", "hf_id": "a/b"}, cfg) == "a/b"


def test_spec_from_command_ignores_unknown_keys():
    s = WorkerSpec.from_command({"cmd": "start_worker", "id": "r1", **spec().__dict__, "bogus": 1})
    assert s.worker_id == "w1" and s.layer_end == 22


def test_worker_lifecycle_started_online_then_killed(tmp_path):
    async def scenario():
        events: list = []
        async def on_event(e): events.append(e)
        m = WorkerManager(AgentConfig(work_dir=tmp_path), on_event, fake_builder())
        await m.start(spec())
        await collect(events, "worker.online")
        started = next(e for e in events if e["type"] == "worker.started")
        assert started["peer_id"] == "12D3KooWw1" and m.describe()[0]["online"] is True
        assert [e["type"] for e in events][:2] == ["worker.started", "worker.online"]
        assert await m.kill("w1") is True
        await collect(events, "worker.exited")
        exited = next(e for e in events if e["type"] == "worker.exited")
        assert exited["reason"] == "killed" and exited["returncode"] != 0
        assert m.pids() == [] and await m.kill("w1") is False
        await m.stop_all()
    asyncio.run(scenario())


def test_graceful_stop_and_natural_exit_are_distinguished(tmp_path):
    async def scenario():
        events: list = []
        async def on_event(e): events.append(e)
        m = WorkerManager(AgentConfig(work_dir=tmp_path), on_event, fake_builder())
        await m.start(spec("a"))
        await collect(events, "worker.online")
        assert await m.stop("a") is True
        await collect(events, "worker.exited")
        assert events[-1]["reason"] == "stopped" and events[-1]["returncode"] == 0

        events.clear()
        m2 = WorkerManager(AgentConfig(work_dir=tmp_path), on_event, fake_builder(["--exit-after", "0.3", "--exit-code", "3"]))
        await m2.start(spec("b"))
        await collect(events, "worker.exited")
        assert events[-1]["reason"] == "exited" and events[-1]["returncode"] == 3  # crashed on its own
        await m.stop_all(); await m2.stop_all()
    asyncio.run(scenario())


def test_duplicate_worker_id_is_rejected_and_logs_are_written(tmp_path):
    async def scenario():
        async def on_event(e): pass
        m = WorkerManager(AgentConfig(work_dir=tmp_path), on_event, fake_builder())
        await m.start(spec("dup"))
        with pytest.raises(ValueError):
            await m.start(spec("dup"))
        await asyncio.sleep(0.5)
        await m.stop_all()
        assert "Running a server on" in (tmp_path / "logs" / "dup.log").read_text()
    asyncio.run(scenario())


def test_explicit_model_path_wins_and_port_range_is_used(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    cfg = AgentConfig(bloombee_dir=tmp_path / "BB", models_dir=tmp_path / "models")
    assert resolve_model({"name": "tinyllama", "hf_id": "a/b", "path": str(merged)}, cfg) == str(merged)
    assert resolve_model({"name": "tinyllama", "hf_id": "a/b", "path": str(tmp_path / "gone")}, cfg) == "a/b"

    async def scenario():
        events = []
        async def on_event(e): events.append(e)
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0)); base = s.getsockname()[1]
        c = AgentConfig(work_dir=tmp_path, port_range=(base, base + 1))
        m = WorkerManager(c, on_event, fake_builder())
        w1 = await m.start(spec("p1"))
        w2 = await m.start(spec("p2"))
        assert w1.spec.host_maddr == f"/ip4/127.0.0.1/tcp/{base}" and w2.spec.host_maddr == f"/ip4/127.0.0.1/tcp/{base + 1}"
        with pytest.raises(RuntimeError, match="no free worker port"):
            await m.start(spec("p3"))
        await m.stop_all()
    asyncio.run(scenario())
