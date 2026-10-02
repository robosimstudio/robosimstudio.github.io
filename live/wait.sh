#!/usr/bin/env bash
# Wait until each named task's current run is drivable: a serve.log written after SINCE (epoch
# seconds; default: when this was called) that says "the hand is the phones' now". The supervisor
# moves the old log aside only once the new run starts, so the old run's ready line must not count.
#
#   since=$(date +%s); live/up.sh cable; SINCE=$since live/wait.sh cable
set -u
RSW_LIVE="${RSW_LIVE:-$HOME/rss_ws/rsw_live}"
since="${SINCE:-$(date +%s)}"; limit="${WAIT_LIMIT:-420}"; rc=0
for name in "$@"; do
  log="$RSW_LIVE/run/$name/serve.log"; t=0
  until { [ -f "$log" ] && [ "$(stat -c %Y "$log")" -ge "$since" ] && grep -q "the hand is the phones' now" "$log"; } || [ "$t" -ge "$limit" ]; do
    sleep 3; t=$((t + 3))
  done
  if [ "$t" -ge "$limit" ]; then echo "$name: not ready after ${limit} s"; rc=1; else echo "$name: ready (${t} s)"; fi
done
exit $rc
