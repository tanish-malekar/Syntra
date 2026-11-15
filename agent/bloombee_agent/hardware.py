"""GPU discovery: Apple-silicon Macs (system_profiler, sysctl, vm_stat) and NVIDIA GPUs (nvidia-smi).

The parsers are pure functions so they can be tested against captured command output.
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Iterable, Optional


@dataclass
class GpuInfo:
    index: int
    name: str
    device: str  # "mps" | "cuda"
    total_memory_mb: int
    free_memory_mb: int
    busy_external: bool = False  # someone else's workload is using this GPU
    busy_detail: str = ""
    simulated: bool = False


def run(cmd: list[str], timeout: float = 8.0) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
        return out.stdout
    except (OSError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------- Apple silicon ---------------------------
def parse_system_profiler_chip(json_text: str) -> Optional[str]:
    """`system_profiler SPHardwareDataType -json` -> 'Apple M4 Pro'."""
    try:
        item = json.loads(json_text)["SPHardwareDataType"][0]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    return item.get("chip_type") or item.get("cpu_type")


def parse_vm_stat_free_bytes(text: str) -> Optional[int]:
    """Memory the OS can hand out without swapping: free + inactive + speculative pages."""
    page = re.search(r"page size of (\d+) bytes", text)
    if not page:
        return None
    size = int(page.group(1))
    pages = 0
    for label in ("Pages free", "Pages inactive", "Pages speculative"):
        m = re.search(rf"{label}:\s+(\d+)\.", text)
        if m:
            pages += int(m.group(1))
    return pages * size


def detect_mac() -> list[GpuInfo]:
    chip = parse_system_profiler_chip(run(["system_profiler", "SPHardwareDataType", "-json"]) or "") or "Apple Silicon"
    mem = (run(["sysctl", "-n", "hw.memsize"]) or "0").strip()
    total_mb = int(mem) // (1024 * 1024) if mem.isdigit() else 0
    free = parse_vm_stat_free_bytes(run(["vm_stat"]) or "")
    free_mb = (free // (1024 * 1024)) if free is not None else total_mb
    return [GpuInfo(index=0, name=chip, device="mps", total_memory_mb=total_mb, free_memory_mb=min(free_mb, total_mb))]


# ---------------------------------------------------------------- NVIDIA ----------------------------------
NVIDIA_GPU_QUERY = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits"]
NVIDIA_APPS_QUERY = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"]


def _csv_rows(text: str) -> list[list[str]]:
    return [[c.strip() for c in line.split(",")] for line in text.splitlines() if line.strip()]


def parse_nvidia_gpus(text: str) -> list[dict]:
    rows = []
    for r in _csv_rows(text):
        if len(r) < 6:
            continue
        try:
            rows.append({"index": int(r[0]), "uuid": r[1], "name": r[2], "total_mb": int(float(r[3])),
                         "used_mb": int(float(r[4])), "util": int(float(r[5]))})
        except ValueError:
            continue
    return rows


def parse_nvidia_apps(text: str) -> list[dict]:
    apps = []
    for r in _csv_rows(text):
        if len(r) < 3 or not r[1].isdigit():
            continue
        try:
            apps.append({"uuid": r[0], "pid": int(r[1]), "used_mb": int(float(r[2]))})
        except ValueError:
            continue
    return apps


def own_pid_tree(ps_text: str, roots: Iterable[int]) -> set[int]:
    """All pids at or below `roots`, from `ps -e -o pid=,ppid=` output."""
    children: dict[int, list[int]] = {}
    for line in ps_text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, []))
    return seen


def classify_nvidia(gpus: list[dict], apps: list[dict], own_pids: set[int], util_threshold: int = 30) -> list[GpuInfo]:
    """Mark a GPU busy_external when a process that is not ours uses it (or it is busy and we run nothing on it)."""
    by_uuid = {g["uuid"]: g["index"] for g in gpus}
    out: list[GpuInfo] = []
    for g in gpus:
        mine = [a for a in apps if by_uuid.get(a["uuid"]) == g["index"] and a["pid"] in own_pids]
        foreign = [a for a in apps if by_uuid.get(a["uuid"]) == g["index"] and a["pid"] not in own_pids]
        busy, detail = False, ""
        if foreign:
            busy = True
            detail = "process %d uses %d MiB" % (foreign[0]["pid"], foreign[0]["used_mb"])
        elif g["util"] >= util_threshold and not mine:
            busy, detail = True, "utilization %d%%" % g["util"]
        out.append(GpuInfo(index=g["index"], name=g["name"], device="cuda", total_memory_mb=g["total_mb"],
                           free_memory_mb=max(0, g["total_mb"] - g["used_mb"]), busy_external=busy, busy_detail=detail))
    return out


def detect_nvidia(own_roots: Iterable[int] = ()) -> list[GpuInfo]:
    gpus = parse_nvidia_gpus(run(NVIDIA_GPU_QUERY) or "")
    if not gpus:
        return []
    apps = parse_nvidia_apps(run(NVIDIA_APPS_QUERY) or "")
    ps = run(["ps", "-e", "-o", "pid=,ppid="]) or ""
    return classify_nvidia(gpus, apps, own_pid_tree(ps, own_roots))


# ---------------------------------------------------------------- laptop lid ------------------------------
def parse_clamshell_closed(ioreg_text: str) -> bool:
    """`ioreg -r -k AppleClamshellState -d 1` -> True when the lid is closed."""
    m = re.search(r'"AppleClamshellState"\s*=\s*(Yes|No)', ioreg_text)
    return bool(m and m.group(1) == "Yes")


def lid_closed() -> bool:
    """True when this laptop's lid is closed (macOS clamshell state, Linux ACPI lid switch); False if unknown."""
    if platform.system() == "Darwin":
        return parse_clamshell_closed(run(["ioreg", "-r", "-k", "AppleClamshellState", "-d", "1"], timeout=2.0) or "")
    for f in Path("/proc/acpi/button/lid").glob("*/state"):
        try:
            return "closed" in f.read_text()
        except OSError:
            pass
    return False


# ---------------------------------------------------------------- front door --------------------------------
def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def detect(own_roots: Iterable[int] = ()) -> list[GpuInfo]:
    """The real GPUs on this machine."""
    if is_apple_silicon():
        return detect_mac()
    return detect_nvidia(own_roots)


def simulate(base: list[GpuInfo], count: int, memory_gb: float) -> list[GpuInfo]:
    """Split one real GPU into `count` labelled virtual providers' GPUs (demo/testing on a single machine)."""
    if not base:
        return []
    proto = base[0]
    mem = int(memory_gb * 1024)
    return [
        replace(proto, index=i, name=f"{proto.name} (sim {i + 1})", total_memory_mb=mem, free_memory_mb=mem,
                simulated=True, busy_external=False, busy_detail="")
        for i in range(count)
    ]
