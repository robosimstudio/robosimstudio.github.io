#!/usr/bin/env bash
# Keep one live task up: restart it when it exits, is not ready in time, stops logging, or its
# public link stops answering while the page still answers locally. Events (START / URL / READY /
# EXIT / HANG / UNREACHABLE) go to $RUN/supervisor.log; the live link is always in
# $RUN/current_url.txt. The phone demo's supervisor, for live/launch.py.
#
#   TASK=towel GPU=6 PORT=8104 nohup setsid live/supervise.sh > /dev/null 2>&1 &
#   TASK=towel GPU=6 PORT=8104 PUBLIC_URL=https://towel.<host>/ nohup setsid live/supervise.sh ...
#     behind the gateway (live/Caddyfile): no relay link, and the public check is that address
#   kill $(cat ~/rss_ws/rsw_live/run/towel/supervisor.pid)   # stop it (takes the task down with it)
set -u
LIVE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HOME/rss_ws/RoboSimStudio"
ROOT="${RSW_LIVE:-$HOME/rss_ws/rsw_live}"
TASK="${TASK:?set TASK=bar|phone|peel|towel}"; NAME="${NAME:-$TASK}"
GPU="${GPU:?set GPU}"; PORT="${PORT:?set PORT}"; PUBLIC_URL="${PUBLIC_URL:-}"
SHARE=(--share); [ -n "$PUBLIC_URL" ] && SHARE=()
case "$GPU" in [0-7]) ;; *) echo "GPU $GPU: no such card (0-7)" >&2; exit 2;; esac
RUN="$ROOT/run/$NAME"; LOG="$RUN/serve.log"; EV="$RUN/supervisor.log"; URLF="$RUN/current_url.txt"
mkdir -p "$RUN"; echo $$ > "$RUN/supervisor.pid"
say() { echo "$(date '+%F %T') [$NAME] $*" >> "$EV"; }
# The task is its own process group (setsid below): stopping the group takes its tunnel and
# manager processes too; killing only its PID left them holding the port.
stop_task() { kill -TERM -- "-$1" 2>/dev/null; sleep 8; kill -KILL -- "-$1" 2>/dev/null; }
trap 'say "SUPERVISOR stopped"; [ -n "${pid:-}" ] && stop_task "$pid"; exit 0' TERM INT

n=0
while true; do
  n=$((n + 1))
  [ -f "$LOG" ] && mv "$LOG" "$RUN/serve.prev.log"
  rm -f "$URLF"
  (cd "$REPO" && CUDA_VISIBLE_DEVICES="$GPU" exec setsid .venv/bin/python "$LIVE/launch.py" \
    --task "$TASK" --name "$NAME" --port "$PORT" "${SHARE[@]}" "$@") > "$LOG" 2>&1 &
  pid=$!
  started=$(date +%s); url=""; ready=0; fails=0
  say "START run=$n pid=$pid gpu=$GPU port=$PORT"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    cur=${PUBLIC_URL:-$(cat "$URLF" 2>/dev/null)}
    if [ -n "$cur" ] && [ "$cur" != "$url" ]; then
      url="$cur"; fails=0; say "URL $url"
      [ "$ready" = 1 ] && say "READY $url"
    fi
    if [ "$ready" = 0 ] && grep -q "the hand is the phones' now" "$LOG"; then
      ready=1; say "READY ${url:-<no link yet>}"
    fi
    if [ "$ready" = 0 ] && [ $((now - started)) -gt 420 ]; then
      say "HANG not ready after $((now - started)) s; restarting"; stop_task "$pid"; break
    fi
    # Once ready the page logs a status line every 30 s.
    age=$((now - $(stat -c %Y "$LOG")))
    if [ "$ready" = 1 ] && [ "$age" -gt 180 ]; then
      say "HANG no log for ${age} s; restarting"; stop_task "$pid"; break
    fi
    # A failed public check counts only while the page answers on its own port: then it is the
    # relay (or the gateway). A page busy with a long setup answers neither, and HANG covers a
    # real hang. Behind the gateway a restart cannot mend the public side, so it is only logged.
    if [ -n "$url" ] && [ "$ready" = 1 ]; then
      if curl -fsS -m 20 -o /dev/null "$url"; then
        fails=0
      elif curl -fsS -m 10 -o /dev/null "http://localhost:$PORT"; then
        fails=$((fails + 1))
      fi
      if [ "$fails" -ge 8 ] && [ -n "$PUBLIC_URL" ]; then
        [ "$fails" = 8 ] && say "UNREACHABLE $url for ~$((fails * 15)) s (the gateway, not the page)"
      elif [ "$fails" -ge 8 ]; then
        say "UNREACHABLE $url for ~$((fails * 15)) s; restarting"; stop_task "$pid"; break
      fi
    fi
  done
  wait "$pid" 2>/dev/null; code=$?
  say "EXIT run=$n code=$code last error: $(grep -E 'Error|Traceback|Killed' "$LOG" | tail -1 | cut -c1-200)"
  rm -f "$URLF"
  sleep 5
done
