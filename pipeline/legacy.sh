#!/usr/bin/env bash
# Run an rsw stage against the 09-05 gallery worktree -- the tree the conduit and weave
# recordings rebuild in (see rsw/compat.py and demos/_bench.py LEGACY):
#
#   pipeline/legacy.sh export --demo conduit            # sim stages: CUDA_VISIBLE_DEVICES=N first
#   env -u CUDA_VISIBLE_DEVICES pipeline/legacy.sh track --demo weave --gpu 5
#
# The learning_ws venv holds that tree's pins; PYTHONPATH beats its editable .pth.
set -euo pipefail
W="${RSW_LEGACY:-$HOME/rss_ws/wt/gallery}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export RSW_RSS="$W"
export PYTHONPATH="$W:$HERE${PYTHONPATH:+:$PYTHONPATH}"
exec "${RSW_LEGACY_PY:-$HOME/learning_ws/RoboSimStudio/.venv/bin/python}" -m rsw "$@"
