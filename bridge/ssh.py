"""SSH helpers for reaching context3 (only reachable through the `context` jump host, key id_stonybrook)."""
from __future__ import annotations

import os
import subprocess

KEY = os.path.expanduser("~/.ssh/id_stonybrook")
BASE_OPTS = ["-o", "IdentitiesOnly=yes", "-i", KEY, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
PROXY = f"ssh -o IdentitiesOnly=yes -i {KEY} -o BatchMode=yes -W %h:%p context"


def context3_args(extra: list[str] | None = None) -> list[str]:
    """argv prefix for `ssh ... context3` (add the remote command or forwards via `extra`)."""
    return ["ssh", *BASE_OPTS, "-o", f"ProxyCommand={PROXY}", *(extra or []), "context3"]


def run_remote(script: str, timeout: float = 120) -> subprocess.CompletedProcess:
    """Run a bash script on context3 (sent on stdin, so quoting is never an issue)."""
    return subprocess.run(context3_args() + ["bash", "-s"], input=script, capture_output=True, text=True, timeout=timeout)


def tunnel(forwards_local: list[int], forwards_remote: list[int]) -> subprocess.Popen:
    """Long-lived SSH connection.

    -L P : a process on THIS machine connecting to 127.0.0.1:P reaches context3's 127.0.0.1:P
    -R P : a process on context3 connecting to 127.0.0.1:P reaches THIS machine's 127.0.0.1:P
    Using the same port number on both sides means a worker's announced address 127.0.0.1:P is valid everywhere.
    """
    extra = ["-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
    for p in forwards_local:
        extra += ["-L", f"{p}:127.0.0.1:{p}"]
    for p in forwards_remote:
        extra += ["-R", f"{p}:127.0.0.1:{p}"]
    return subprocess.Popen(context3_args(extra), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def push_dir(local_parent: str, name: str, remote_dir: str) -> None:
    """Copy local_parent/name into remote_dir on context3 (tar over ssh; no shell quoting of the proxy command)."""
    tar = subprocess.Popen(["tar", "czf", "-", "-C", local_parent, name], stdout=subprocess.PIPE)
    done = subprocess.run(context3_args() + [f"mkdir -p {remote_dir} && tar xzf - -C {remote_dir}"], stdin=tar.stdout,
                          capture_output=True, text=True, timeout=180)
    tar.wait()
    if done.returncode != 0 or tar.returncode != 0:
        raise RuntimeError("could not copy the agent to context3: " + done.stderr[-300:])
