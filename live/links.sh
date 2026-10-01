#!/usr/bin/env bash
# Print the live tasks' addresses and whether each answers right now.
# The tasks sit behind the HTTPS gateway (live/Caddyfile) at fixed addresses; the conference phone
# demos (GPUs 2-4) keep their viser relay links, which change, so those are read fresh each time.
HOST="${RSW_HOST:-95-133-252-186.sslip.io}"
printf '%-16s %s\n' "Landing page" "https://$HOST/"
for row in "bar:Wipe the bar" "phone:Wipe a phone" "peel:Peel the film" "towel:Fold the towel"; do
  id=${row%%:*}; title=${row#*:}
  code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' "https://$HOST/status/$id")
  printf '%-16s %-44s %s\n' "$title" "https://$id.$HOST/" "$([ "$code" = 204 ] && echo up || echo "down or restarting ($code)")"
done
echo
echo "Conference phone demos (viser relay links, GPUs 2-4):"
for s in bar phone peel; do
  u=$(cat ~/rss_ws/phone_demo/run/$s/current_url.txt 2>/dev/null)
  printf '  %-6s %s\n' "$s" "${u:-(no link right now)}"
done
