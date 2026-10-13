#!/usr/bin/env bash
# Rebuild www/viser/index.html: viser's static client (the version installed in $VENV) with
# viser-embed.patch, which adds ?embedded (no controls of its own; the page drives play/seek and
# draws the progress bar, see www/site.js). The recordings in www/rollouts must come from the same version.
#
#   scripts/build_viser.sh [venv]      # default ~/rss_ws/RoboSimStudio/.venv; needs node >= 24
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VENV=${1:-$HOME/rss_ws/RoboSimStudio/.venv}
PY=$VENV/bin/python
SRC=$("$PY" -c 'import viser, os; print(os.path.join(os.path.dirname(viser.__file__), "client"))')
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cp -r "$SRC" "$WORK/client"
rm -rf "$WORK/client/build" "$WORK/client/node_modules"
cd "$WORK/client"
patch -p1 < "$HERE/viser-embed.patch"
npm ci --no-audit --no-fund
npx vite build
cp build/index.html "$HERE/../www/viser/index.html"
"$PY" -c 'import viser; print(viser.__version__)' > "$HERE/../www/viser/VERSION"
echo "www/viser: viser $(cat "$HERE/../www/viser/VERSION") + embed patch"
