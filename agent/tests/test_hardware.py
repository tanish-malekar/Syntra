from __future__ import annotations

import json

from bloombee_agent import hardware as hw

SYSTEM_PROFILER = json.dumps({"SPHardwareDataType": [{"chip_type": "Apple M4 Pro", "physical_memory": "24 GB",
                                                       "machine_name": "MacBook Pro"}]})
VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               20000.
Pages active:                            300000.
Pages inactive:                           50000.
Pages speculative:                         2000.
Pages wired down:                        150000.
"""
NVIDIA_GPUS = """0, GPU-aaaa, NVIDIA RTX A6000, 49140, 5, 0
1, GPU-bbbb, NVIDIA RTX A6000, 49140, 46344, 100
2, GPU-cccc, NVIDIA RTX A6000, 49140, 5, 0
3, GPU-dddd, NVIDIA RTX A6000, 49140, 1500, 45
"""
NVIDIA_APPS = """GPU-bbbb, 3561272, 46332
GPU-aaaa, 777, 1200
"""


def test_parse_chip_name_and_memory_pages():
    assert hw.parse_system_profiler_chip(SYSTEM_PROFILER) == "Apple M4 Pro"
    assert hw.parse_system_profiler_chip("not json") is None
    assert hw.parse_vm_stat_free_bytes(VM_STAT) == (20000 + 50000 + 2000) * 16384
    assert hw.parse_vm_stat_free_bytes("garbage") is None


def test_parse_nvidia_output():
    gpus = hw.parse_nvidia_gpus(NVIDIA_GPUS)
    assert [g["index"] for g in gpus] == [0, 1, 2, 3] and gpus[1]["used_mb"] == 46344 and gpus[3]["util"] == 45
    apps = hw.parse_nvidia_apps(NVIDIA_APPS)
    assert apps[0] == {"uuid": "GPU-bbbb", "pid": 3561272, "used_mb": 46332}
    assert hw.parse_nvidia_gpus("") == [] and hw.parse_nvidia_apps("No devices were found") == []


def test_busy_external_is_detected_for_foreign_processes_and_load():
    gpus, apps = hw.parse_nvidia_gpus(NVIDIA_GPUS), hw.parse_nvidia_apps(NVIDIA_APPS)
    info = {g.index: g for g in hw.classify_nvidia(gpus, apps, own_pids={777})}
    assert not info[0].busy_external            # the only process on GPU 0 is ours
    assert info[1].busy_external and "3561272" in info[1].busy_detail  # someone else's 46 GB job
    assert not info[2].busy_external            # idle
    assert info[3].busy_external and "utilization" in info[3].busy_detail  # busy but no pid visible
    assert info[0].free_memory_mb == 49140 - 5 and info[1].free_memory_mb == 49140 - 46344


def test_own_pid_tree_includes_children():
    ps = "100 1\n101 100\n102 101\n200 1\n201 200\n"
    assert hw.own_pid_tree(ps, [100]) == {100, 101, 102}
    assert hw.own_pid_tree(ps, [100, 200]) == {100, 101, 102, 200, 201}


def test_simulate_splits_one_gpu_into_labelled_virtual_gpus():
    base = [hw.GpuInfo(0, "Apple M4 Pro", "mps", 24576, 10000)]
    sims = hw.simulate(base, 3, 6)
    assert [g.name for g in sims] == ["Apple M4 Pro (sim 1)", "Apple M4 Pro (sim 2)", "Apple M4 Pro (sim 3)"]
    assert all(g.simulated and g.total_memory_mb == 6144 and g.device == "mps" for g in sims)
    assert [g.index for g in sims] == [0, 1, 2] and hw.simulate([], 2, 6) == []


def test_clamshell_state_parsing():
    from bloombee_agent.hardware import parse_clamshell_closed
    assert parse_clamshell_closed('    | "AppleClamshellState" = Yes\n') is True
    assert parse_clamshell_closed('    | "AppleClamshellState" = No\n') is False
    assert parse_clamshell_closed("") is False
