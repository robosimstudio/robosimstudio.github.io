#!/usr/bin/env bash
# Stop live tasks (each supervisor takes its task down with it), and with no names the gateway too.
#
#   live/down.sh          # everything
#   live/down.sh peel     # one task; the gateway stays
set -u
RSW_LIVE="${RSW_LIVE:-$HOME/rss_ws/rsw_live}"
names=("$@"); [ $# -eq 0 ] && names=(bar phone peel towel)
for name in "${names[@]}"; do
  f="$RSW_LIVE/run/$name/supervisor.pid"
  if [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; then
    kill -TERM "$(cat "$f")" && echo "$name: stopping"
  fi
done
if [ $# -eq 0 ] && [ -f "$RSW_LIVE/gw/caddy.pid" ] && kill -0 "$(cat "$RSW_LIVE/gw/caddy.pid")" 2>/dev/null; then
  kill -TERM "$(cat "$RSW_LIVE/gw/caddy.pid")" && echo "gateway: stopped"
fi
