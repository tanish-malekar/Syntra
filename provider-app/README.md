# BloomBee Provider (macOS menu bar app)

Shares this Mac's GPU with the BloomBee pool. The user never needs a terminal.

```bash
./build.sh          # -> dist/BloomBeeProvider.app (ad-hoc signed, menu bar only)
./build.sh --open   # build and start
swift test          # unit tests
```

## How it works

| Piece | What it does |
|---|---|
| Sign in | `POST /app/pair` with email + password + this Mac's hardware UUID. Returns an agent token for this machine, stored in `~/.bloombee/app-tokens.json` (mode 0600, like the agent's control secret). Pairing the same Mac again revokes its old token. |
| Install | First **Share** runs `join.sh --install-only`: Python 3.12 venv, BloomBee and the agent, all in `~/.bloombee`. The log goes to `~/.bloombee/install.log`. |
| Share | Starts `python -m bloombee_agent --control --relay ...` with the token in the environment (not on the command line). Restarts it after a crash, and gives up after 5 crashes in 2 minutes. |
| Status | The agent's local control API (`127.0.0.1`, port and secret in `~/.bloombee/agent.json`, mode 0600). |
| Pause / resume / quit | `POST /pause {"mode": "drain" or "now"}`, `/resume`, `/quit` on the control API. |
| Usage and earnings | `GET /agent/usage` with the token: GPU time and earnings per GPU, settled (from the ledger) plus a live estimate for the running job. |
| Sign out | `POST /agent/unpair`: the token stops working, and the agent notices and exits (code 3). |

The app polls the status every 3 s and the usage every ~10 s, and refreshes at once on wake.

## Edge cases handled
- **Another agent already running** (terminal or an earlier app run): the instance lock makes a second copy exit with code 4. The app attaches to the running agent's control API instead.
- **Lid closed**: the agent reports its GPUs offline and stops its workers at once. The platform moves the work elsewhere.
- **Network loss**: the agent retries with backoff, and the app shows "Disconnected: reconnecting" with the last error.
- **Token revoked on the website**: the agent exits with code 3. The app clears the Keychain entry and asks to sign in again.
- **Pause while a job is running**: the app asks "finish the current job first" (drain) or "pause now" (the job moves to another GPU).
- **On battery** (optional): pauses automatically, and resumes on power.
- **Quit**: GPUs go offline immediately rather than after the heartbeat timeout. An agent started from a terminal is left running.
- **Install fails** (no Python 3.12, no git, no network): the failing step is shown. Nothing is registered until the agent starts.

Linux machines keep the one-line `join.sh` command. It now also serves the same control API (`--control`).
