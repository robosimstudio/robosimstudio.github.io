#!/usr/bin/env bash
# Build the drag demos (site/) and push them as a single orphan commit on gh-pages.
#
# NOTE: GitHub Pages for this repo is published from www/ by .github/workflows/pages.yml. A
# gh-pages branch pushed here is not served until the Pages source is switched to it.
#
#   scripts/deploy.sh            # from the repo root; main must be committed and clean
#
# The demo data (hundreds of MB of RTX stills and videos) never enters main's history:
# it is copied from build/site-data into the build, and gh-pages is force-pushed as one
# commit. A kept deploy clone lets that commit reuse blobs the remote already has.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${RSS_DATA_DIR:-$REPO_DIR/build/site-data}"
REMOTE="${DEPLOY_REMOTE:-https://github.com/robosimstudio/robosimstudio.github.io.git}"
CACHE="${DEPLOY_CACHE:-$HOME/.cache/rsw-deploy}"

cd "$REPO_DIR"
[ -z "$(git status --porcelain)" ] || { echo "deploy: main has uncommitted changes" >&2; exit 1; }
SHA="$(git rev-parse --short HEAD)"

cd site
npm ci --no-audit --no-fund
npm run typecheck
npm test
rm -rf dist
npm run build
mkdir -p dist/data
rsync -a --delete "$DATA/" dist/data/
touch dist/.nojekyll
node ../scripts/check-dist.mjs dist

if [ ! -d "$CACHE/.git" ]; then
  git clone --depth 1 --branch gh-pages "$REMOTE" "$CACHE" 2>/dev/null || { mkdir -p "$CACHE"; git -C "$CACHE" init -q -b gh-pages; }
fi
git -C "$CACHE" checkout -q --orphan deploy-tmp
rsync -a --delete --exclude .git dist/ "$CACHE/"
git -C "$CACHE" add -A
git -C "$CACHE" -c user.name="$(git -C "$REPO_DIR" config user.name || echo rsw)" -c user.email="$(git -C "$REPO_DIR" config user.email || echo rsw@localhost)" \
  commit -qm "deploy: site from main@$SHA"
git -C "$CACHE" branch -M gh-pages
git -C "$CACHE" push -f "$REMOTE" gh-pages
echo "deploy: pushed main@$SHA to gh-pages"
