#!/usr/bin/env bash
# Print the live tasks' addresses and whether each answers right now.
# The tasks sit behind the HTTPS gateway (live/Caddyfile) at fixed addresses.
HOST="${RSW_HOST:-95-133-252-186.sslip.io}"
printf '%-16s %s\n' "Landing page" "https://$HOST/"
for row in "bar:Wipe the bar" "peel:Peel the film" "towel:Fold the towel" "cable:Route the cable"; do
  id=${row%%:*}; title=${row#*:}
  code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' "https://$HOST/status/$id")
  printf '%-16s %-44s %s\n' "$title" "https://$id.$HOST/" "$([ "$code" = 204 ] && echo up || echo "down or restarting ($code)")"
done
