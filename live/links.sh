#!/usr/bin/env bash
# Print the current public link of every live task, and whether it answers right now.
# The links change when viser's relay drops a tunnel or after 24 h; each page asks for a new one
# by itself, so run this again rather than reusing an old link.
check() {  # name, link file
  local u; u=$(cat "$2" 2>/dev/null)
  if [ -z "$u" ]; then printf '%-16s (no link: starting up or restarting)\n' "$1"; return; fi
  local code; code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' "$u")
  printf '%-16s %-48s %s\n' "$1" "$u" "$([ "$code" = 200 ] && echo ok || echo "no answer ($code), retry")"
}
check "Wipe the bar"   ~/rss_ws/phone_demo/run/bar/current_url.txt
check "Wipe a phone"   ~/rss_ws/phone_demo/run/phone/current_url.txt
check "Peel the film"  ~/rss_ws/phone_demo/run/peel/current_url.txt
check "Fold the towel" ~/rss_ws/rsw_live/run/towel/current_url.txt
