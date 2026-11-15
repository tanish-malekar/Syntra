"""Tests for the swarm-runner side (torch needed): run with the BloomBee interpreter, e.g.
    ~/.syntra/venv/bin/python -m pytest backend/runner_tests -q
The runner modules are imported the way the runner imports them (as top-level modules from app/runner)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "runner"))
