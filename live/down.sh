#!/usr/bin/env bash
# Stop live tasks (each supervisor takes its task down with it), and with no names the gateway too.
#
#   live/down.sh          # everything
#   live/down.sh peel     # one task; the gateway stays
set -u
RSW_LIVE="${RSW_LIVE:-$HOME/rss_ws/rsw_live}"
names=("$@"); [ $# -eq 0 ] && names=(bar peel towel cable)
pids=()
for name in "${names[@]}"; do
  f="$RSW_LIVE/run/$name/supervisor.pid"
  if [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; then
    kill -TERM "$(cat "$f")" && echo "$name: stopping" && pids+=("$(cat "$f")")
  fi
done
# Wait for each supervisor to take its task down (TERM, 8 s grace, KILL), so an up.sh straight
# after this starts a fresh one rather than finding the old one still alive.
for pid in "${pids[@]}"; do
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
  kill -0 "$pid" 2>/dev/null && echo "supervisor $pid still running after 30 s" >&2
done
if [ $# -eq 0 ] && [ -f "$RSW_LIVE/gw/caddy.pid" ] && kill -0 "$(cat "$RSW_LIVE/gw/caddy.pid")" 2>/dev/null; then
  kill -TERM "$(cat "$RSW_LIVE/gw/caddy.pid")" && echo "gateway: stopped"
fi
