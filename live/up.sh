#!/usr/bin/env bash
# Bring every live task up behind the HTTPS gateway; what is already running is left alone.
#
#   live/up.sh            # the gateway + all tasks
#   live/up.sh towel      # just these tasks (the gateway too, if it is down)
#   live/down.sh [task]   # stop them
#
# Task table: name, GPU, local port, extra page options.
set -u
LIVE="$(cd "$(dirname "$0")" && pwd)"
export RSW_LIVE="${RSW_LIVE:-$HOME/rss_ws/rsw_live}"
export RSW_HOST="${RSW_HOST:-95-133-252-186.sslip.io}"
TASKS=(
  "bar   0 8101"
  "phone 1 8102"
  "peel  5 8103 --render-every 1 --attract 600"
  "towel 6 8104"
  "cable 4 8105"
)
mkdir -p "$RSW_LIVE"/{gw,www,logs,run}
cp "$LIVE/www/index.html" "$RSW_LIVE/www/index.html"

alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

if alive "$RSW_LIVE/gw/caddy.pid"; then
  "$RSW_LIVE/bin/caddy" reload --config "$LIVE/Caddyfile" --adapter caddyfile 2>/dev/null && echo "gateway: reloaded"
else
  (nohup setsid "$RSW_LIVE/bin/caddy" run --config "$LIVE/Caddyfile" --adapter caddyfile \
     --pidfile "$RSW_LIVE/gw/caddy.pid" >> "$RSW_LIVE/gw/caddy.log" 2>&1 &)
  echo "gateway: started (https://$RSW_HOST)"
fi

for row in "${TASKS[@]}"; do
  read -r name gpu port extra <<< "$row"
  [ $# -gt 0 ] && [[ " $* " != *" $name "* ]] && continue
  if alive "$RSW_LIVE/run/$name/supervisor.pid"; then
    echo "$name: already up (https://$name.$RSW_HOST)"
    continue
  fi
  # shellcheck disable=SC2086
  (TASK=$name GPU=$gpu PORT=$port PUBLIC_URL="https://$name.$RSW_HOST/" \
     nohup setsid "$LIVE/supervise.sh" $extra > /dev/null 2>&1 &)
  echo "$name: starting on GPU $gpu (https://$name.$RSW_HOST)"
done
