#!/usr/bin/env bash
# Emulate a slow wide-area link on macOS (dummynet + pf). NEEDS sudo. UNTESTED.
#
# Usage:
#   sudo scripts/mac_throttle.sh up   [bandwidth_mbit] [delay_ms] [loss_pct]   # default 100 Mbit, 40 ms, 0 %
#   sudo scripts/mac_throttle.sh down
#   scripts/mac_throttle.sh status
#
# Shapes TCP traffic on loopback (lo0) and the active LAN interface, so it works for several
# worker processes on one Mac as well as for two Macs on one network (run it on both).
# The rules live in their own pf anchor ("com.apple/bloombee") and are removed by `down`;
# your existing pf rules are not edited. Review before running.
set -euo pipefail

ANCHOR="com.apple/bloombee"
PIPE=1
TOKEN_FILE="/tmp/bloombee_pf_token"

cmd="${1:-status}"
bw="${2:-100}"
delay="${3:-40}"
loss="${4:-0}"

need_root() { [ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }; }

case "$cmd" in
  up)
    need_root
    dnctl pipe "$PIPE" config bw "${bw}Mbit/s" delay "${delay}ms" plr "$(awk "BEGIN{print $loss/100}")"
    # Shape traffic in both directions; peers on the same machine talk over lo0.
    printf '%s\n' \
      "dummynet in  quick proto tcp from any to any pipe $PIPE" \
      "dummynet out quick proto tcp from any to any pipe $PIPE" \
      | pfctl -a "$ANCHOR" -f - 2>&1 | grep -v "^No ALTQ" || true
    # Enable pf and remember the token so `down` only releases what we enabled.
    pfctl -E 2>&1 | awk '/Token/ {print $3}' > "$TOKEN_FILE" || true
    echo "throttle ON: ${bw} Mbit/s, ${delay} ms delay, ${loss}% loss (each direction)"
    ;;
  down)
    need_root
    pfctl -a "$ANCHOR" -F all 2>/dev/null || true
    dnctl -q flush || true
    if [ -s "$TOKEN_FILE" ]; then pfctl -X "$(cat "$TOKEN_FILE")" 2>/dev/null || true; rm -f "$TOKEN_FILE"; fi
    echo "throttle OFF"
    ;;
  status)
    echo "--- dnctl pipes"; dnctl list 2>/dev/null || echo "(need sudo to list)"
    echo "--- anchor rules"; pfctl -a "$ANCHOR" -sr 2>/dev/null || echo "(need sudo to list)"
    ;;
  *)
    echo "usage: $0 up|down|status"; exit 2 ;;
esac
