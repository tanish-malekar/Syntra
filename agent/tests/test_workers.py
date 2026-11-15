from __future__ import annotations

import asyncio
import sys

import pytest

from syntra_agent.workers import AgentConfig, WorkerManager, WorkerSpec, build_bloombee_command, resolve_model, supports_4bit
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


def test_4bit_base_weights_add_the_compress_flag_and_other_widths_are_refused(tmp_path):
    cfg = AgentConfig(bloombee_dir=tmp_path / "BB", models_dir=tmp_path / "models")
    assert "--compress_weight" not in build_bloombee_command(spec(), cfg)[0]
    assert "--compress_weight" in build_bloombee_command(spec(weight_bits=4), cfg)[0]
    assert "--compress_weight" in build_bloombee_command(spec(weight_bits=4, mode="serve"), cfg)[0]
    s = WorkerSpec.from_command({**spec().__dict__, "weight_bits": 4})
    assert s.weight_bits == 4
    with pytest.raises(ValueError):
        build_bloombee_command(spec(weight_bits=8), cfg)


def test_4bit_support_is_read_from_the_bloombee_checkout(tmp_path):
    cfg = AgentConfig(bloombee_dir=tmp_path / "BB")
    assert supports_4bit(cfg) is False                         # no checkout at all
    cli = tmp_path / "BB" / "src" / "bloombee" / "cli" / "run_server.py"
    cli.parent.mkdir(parents=True)
    cli.write_text("parser.add_argument('--quant_type')\n")
    assert supports_4bit(cfg) is False                         # an older BloomBee
    cli.write_text("parser.add_argument('--compress_weight', action='store_true')\n")
    assert supports_4bit(cfg) is True


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


def test_hanging_driver_does_not_block_the_agent():
    """nvidia-smi can hang for minutes after a CUDA process is killed: detect_now must give up waiting, not the heartbeat."""
    import asyncio
    import threading
    import time as _t
    from syntra_agent.client import Agent, AgentSettings
    from syntra_agent.hardware import GpuInfo
    from syntra_agent.workers import AgentConfig

    release = threading.Event()
    calls = []

    def detector():
        calls.append(1)
        if len(calls) == 2:
            release.wait(30)  # the second probe hangs
        return [GpuInfo(index=0, name="g", device="cuda", total_memory_mb=1000, free_memory_mb=900)]

    async def run():
        agent = Agent(AgentSettings(server="http://x", token="t"), AgentConfig(bloombee_dir=".", python="python"), detector=detector)
        assert await agent.detect_now(1) is not None
        t = _t.monotonic()
        assert await agent.detect_now(0.3) is None  # hung probe
        assert _t.monotonic() - t < 1
        assert await agent.detect_now(0.2) is None and len(calls) == 2  # still one probe in flight, no pile-up
        release.set()
        assert await agent.detect_now(2) is not None  # the slow probe's result is picked up

    asyncio.run(run())


def test_a_worker_that_crashes_while_starting_and_hangs_is_killed_and_reported(tmp_path):
    async def scenario():
        events: list = []
        async def on_event(e): events.append(e)
        m = WorkerManager(AgentConfig(work_dir=tmp_path, crash_grace=0.3), on_event, fake_builder(["--crash-and-hang"]))
        await m.start(spec())
        await collect(events, "worker.exited")
        exited = next(e for e in events if e["type"] == "worker.exited")
        assert exited["reason"] == "failed_to_start" and "No space left on device" in exited["error"]
        assert not any(e["type"] == "worker.online" for e in events) and m.pids() == []
        await m.stop_all()
    asyncio.run(scenario())


def test_a_worker_that_never_comes_online_is_killed_after_the_startup_timeout(tmp_path):
    async def scenario():
        events: list = []
        async def on_event(e): events.append(e)
        m = WorkerManager(AgentConfig(work_dir=tmp_path, startup_timeout=0.5), on_event, fake_builder(["--never-online"]))
        await m.start(spec())
        await collect(events, "worker.exited")
        exited = next(e for e in events if e["type"] == "worker.exited")
        assert exited["reason"] == "failed_to_start" and "not online" in exited["error"]
        await m.stop_all()
    asyncio.run(scenario())


def test_killing_a_worker_group_spares_the_multiprocessing_resource_tracker(tmp_path):
    """macOS leaks named semaphores until reboot if the resource_tracker dies with its parent (kern.posix.sem.max)."""
    import os
    import signal
    import subprocess
    import time
    from syntra_agent.workers import group_members, kill_process_group
    # a "worker" that starts a helper named like the tracker in its own group
    code = ("import subprocess, sys, time;"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)', 'multiprocessing.resource' + '_tracker']);"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)', 'p2pd']);"
            "time.sleep(30)")
    main = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        deadline = time.time() + 10
        while len(group_members(main.pid)) < 3 and time.time() < deadline:
            time.sleep(0.05)
        members = dict(group_members(main.pid))
        tracker = next(pid for pid, cmd in members.items() if "multiprocessing.resource_tracker" in cmd)
        helper = next(pid for pid, cmd in members.items() if "p2pd" in cmd)
        kill_process_group(main.pid)
        main.wait(5)
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                os.kill(helper, 0)
                time.sleep(0.05)
            except ProcessLookupError:
                break
        alive = lambda pid: subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip() not in ("", "Z")
        assert not alive(helper) and alive(tracker)
    finally:
        try:
            os.killpg(main.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_diloco_replica_command_carries_the_lease_and_keeps_the_token_out_of_argv(tmp_path):
    cfg = AgentConfig(bloombee_dir=tmp_path / "BB", python="/py", models_dir=tmp_path / "m", work_dir=tmp_path,
                      server="http://platform:8000", token="syn_agent_secret")
    argv, env = build_bloombee_command(spec(kind="diloco", job_id=7, lease_id=42, device="cuda", device_index=1), cfg)
    flag = lambda name: argv[argv.index(name) + 1]
    assert argv[:3] == ["/py", "-m", "syntra_agent.diloco_worker"] and flag("--job") == "7" and flag("--lease") == "42"
    assert flag("--server") == "http://platform:8000" and "syn_agent_secret" not in " ".join(argv)
    assert env["SYNTRA_AGENT_TOKEN"] == "syn_agent_secret" and env["CUDA_VISIBLE_DEVICES"] == "1"
    with pytest.raises(ValueError):
        build_bloombee_command(spec(kind="diloco"), AgentConfig(work_dir=tmp_path))  # no server/token
    with pytest.raises(ValueError):
        build_bloombee_command(spec(kind="mystery"), cfg)
