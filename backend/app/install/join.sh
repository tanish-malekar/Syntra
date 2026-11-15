#!/usr/bin/env bash
# Join the Syntra GPU pool from this machine. It only makes outgoing connections: no open ports, no SSH, any network.
#
#   curl -fsSL __SERVER__/join.sh | bash -s -- --token <TOKEN> [--region NAME] [--gpus 0,1]
#   curl -fsSL __SERVER__/join.sh | bash -s -- --install-only      (what the Syntra desktop app runs: install, then exit)
#
# First run installs everything under ~/.syntra (a Python venv, BloomBee, the Syntra agent); later runs start at once.
# Only one agent runs per machine: a second copy (terminal or desktop app) exits with code 4.
set -euo pipefail

SERVER="__SERVER__"
TOKEN="${SYNTRA_AGENT_TOKEN:-}"
REGION=""
GPUS=""
INSTALL_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --token) TOKEN="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --install-only) INSTALL_ONLY=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -n "$TOKEN" ] || [ "$INSTALL_ONLY" = 1 ] || { echo "Missing --token. Create one on the Provide GPUs page." >&2; exit 2; }

SYNTRA_HOME="${SYNTRA_HOME:-$HOME/.syntra}"
mkdir -p "$SYNTRA_HOME"

PY=""
for c in python3.12 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
  if command -v "$c" >/dev/null 2>&1; then PY="$(command -v "$c")"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3.12 is needed. On a Mac:  brew install python@3.12   then run this again." >&2
  exit 1
fi

step() { echo; echo "==> $*"; }

step "Python environment"
[ -x "$SYNTRA_HOME/venv/bin/python" ] || "$PY" -m venv "$SYNTRA_HOME/venv"
VPY="$SYNTRA_HOME/venv/bin/python"
"$VPY" -m pip install -q --upgrade pip
"$VPY" -m pip install -q "httpx>=0.27" "websockets>=13"

step "BloomBee (the distributed model engine)"
if [ -d "$SYNTRA_HOME/BloomBee/.git" ]; then
  git -C "$SYNTRA_HOME/BloomBee" pull -q --ff-only || true
else
  git clone -q --depth 1 -b mac-mps-ft https://github.com/tanish-malekar/BloomBee "$SYNTRA_HOME/BloomBee"
fi
if [ ! -f "$SYNTRA_HOME/.deps-installed" ]; then
  echo "Installing BloomBee's dependencies (a few minutes the first time)..."
  curl -fsSL "$SERVER/install/requirements.txt" -o "$SYNTRA_HOME/requirements.txt"
  "$VPY" -m pip install -q -r "$SYNTRA_HOME/requirements.txt"
  touch "$SYNTRA_HOME/.deps-installed"
fi

step "Syntra agent"
rm -rf "$SYNTRA_HOME/agent" && mkdir -p "$SYNTRA_HOME/agent"
curl -fsSL "$SERVER/install/agent.tgz" | tar xz -C "$SYNTRA_HOME/agent"
"$VPY" -c "import sys; sys.path.insert(0, '$SYNTRA_HOME/agent'); import syntra_agent.cli" || { echo "The agent did not install correctly." >&2; exit 1; }

if [ "$INSTALL_ONLY" = 1 ]; then
  step "Installed. The Syntra app starts the agent."
  exit 0
fi

step "Joining the pool (leave this window open; Ctrl-C to leave)"
ARGS=(--server "$SERVER" --token "$TOKEN" --relay --bloombee-dir "$SYNTRA_HOME/BloomBee" --python "$VPY"
      --models-dir "$SYNTRA_HOME/models" --work-dir "$SYNTRA_HOME/work" --control --state-dir "$SYNTRA_HOME")
[ -n "$REGION" ] && ARGS+=(--region "$REGION")
[ -n "$GPUS" ] && ARGS+=(--gpus "$GPUS")
cd "$SYNTRA_HOME/agent"
export PYTHONPATH="$SYNTRA_HOME/agent"
exec "$VPY" -m syntra_agent "${ARGS[@]}"
