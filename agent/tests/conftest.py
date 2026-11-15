import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))          # bloombee_agent
sys.path.insert(0, str(ROOT / "backend"))     # the platform backend, for the end-to-end test
