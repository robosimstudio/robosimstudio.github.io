# robosimstudio.github.io

The RoboSimStudio project page, published by GitHub Pages at <https://robosimstudio.github.io/>.

The page (`www/`) is static: the title, authors and logos; the Real2Sim workflow as five steps on one
stage that sticks while the page scrolls through them, with a one-sentence lead above and the step rail down
the left (the input clip with the prompt to a coding agent,
the agent at work, the simulation it produced under a slider between its two views, a recorded rollout of
that simulation in 3D, with snapshots to pick which task: wipe the bar, peel the film, route the cable, and the
real robot running it); and the gallery. The 3D step needs no server: `www/viser/index.html` is viser's static
client with `scripts/viser-embed.patch` (built by `scripts/build_viser.sh`; version in `www/viser/VERSION`): with
`?embedded` it draws no controls of its own and the page draws the progress bar under the stage, talking to it by
`postMessage`. It plays `www/rollouts/<task>.viser`, a replay of the real simulation the visitor can orbit and
zoom. Each task's opening camera is in `VIEWS` at the top of `www/site.js`.

The rollouts are scripted (`rollouts.py` in [robosimstudio/demo](https://github.com/robosimstudio/demo)): the bar
wiped stroke after stroke, the film peeled off and fluttered, the cable routed and looped round a clip. To
re-record one, start its task with `RSW_RECORD` and `RSW_PROGRAM` and keep one viewer on the page (it only steps
with a viewer); the page runs the program once its pre-roll is done and writes the file at its last step:

```bash
cd ~/rss_ws/RoboSimStudio
RSW_RECORD=/tmp/bar.viser RSW_PROGRAM=1 CUDA_VISIBLE_DEVICES=5 \
    .venv/bin/python ~/website_ws/demo/launch.py --task bar --name bar_rec --port 8124 --attract 0
# open http://localhost:8124 in a browser and leave it; then: cp /tmp/bar.viser www/rollouts/
```

A recording only plays in the client of the viser version that wrote it: rebuild `www/viser/` (`scripts/build_viser.sh`)
and re-record when viser moves.

Clips under `www/video/`: `teaser` (from `~/rss_ws/teaser/out/rss_teaser.mp4`), `bar_real` / `bar_gen` / `bar_out` / `bar_mech` / `bar_deploy`
(the workflow: the phone video, the agent's hour as the 40x timelapse with its timer overlay from
`~/rss_ws/agentic surface cleaning - timelapse.mp4`, the path-traced twin and the multiphysics view from
`~/rss_ws/teaser/takes/bar_rtx.mp4` and `bar_mech.mp4` (frame-aligned, so the slider compares the same instant), the real robot rollout from `~/rss_ws/IMG_8647-2_2.5x 2.MOV`, 7.0 to 9.3 s and 10.8 to 21.5 s joined),
`bar` / `peel` / `cable` (the 3D block's posters), `gallery` (from `~/rss_ws/gallery/wall/task_wall_1080p.mp4`).
New clips: 1280 wide, H.264 CRF 26, no audio, `-movflags +faststart`; the poster is the frame at 2 s.

| path | what |
|---|---|
| `www/` | the page: `index.html`, `site.js`, `site.css`, fonts, logos, its clips under `video/`, the 3D replays under `rollouts/` and their player under `viser/` |
| `www/robosimstudio.pdf` | the paper, behind the Paper button until arXiv lists it: `make` in `~/rss_ws/RoboSimStudio-draft`, then copy its `main.pdf` here |
| `scripts/stamp.py` | run by the Pages workflow: every file link gets a content hash, so no cache pairs a new page with old files |
| `.github/workflows/pages.yml` | publishes `www/` on every push to `main` |
| `site/`, `pipeline/`, `scripts/deploy.sh` | the drag demos (see below); not part of the published page yet |

Edit `www/`, push to `main`, and the page is live a minute later. To preview locally:

```bash
python3 -m http.server -d www 8000     # http://localhost:8000/
```

## The drag demos (`site/` and `pipeline/`)

A second, data-heavy front end: drag the robot and feel RoboSimStudio's physics, rendered with its RTX
path tracer. Nothing is simulated in the browser; every state on screen was produced by RoboSimStudio
offline, and the page picks, ten times a second, the recorded frame that best continues what is on
screen toward your pointer (motion matching).

| path | what |
|---|---|
| `pipeline/` | the offline pipeline (`python -m rsw <stage>`), run with RoboSimStudio's venv: takes, export, db, plate, stills, attract, track, tiles, site |
| `pipeline/schema/CONTRACTS.md` | the data contract between the pipeline and the page |
| `site/` | the page: Vite + TypeScript + three.js; `src/core` is DOM-free and unit-tested |
| `scripts/deploy.sh` | build, check, and force-push the built site as one orphan commit on `gh-pages` |

Built data lives outside git (`build/site-data`, and takes/frames under `~/rss_ws/rsw_data`).

```bash
cd pipeline && PY=~/rss_ws/RoboSimStudio/.venv/bin/python
CUDA_VISIBLE_DEVICES=5 $PY -m rsw takes --demo wipe --seed 1 --takes 3 --seconds 240   # virtual visitors
CUDA_VISIBLE_DEVICES=5 $PY -m rsw export --demo wipe                                   # GLB + kinematics
$PY -m rsw db --demo wipe                                                              # the motion-matching database
env -u CUDA_VISIBLE_DEVICES $PY -m rsw plate --demo wipe --gpu 5                       # RTX: static plate
env -u CUDA_VISIBLE_DEVICES $PY -m rsw stills --demo wipe --gpu 5                      # RTX: the still set
env -u CUDA_VISIBLE_DEVICES $PY -m rsw attract --demo wipe --gpu 5                     # RTX: the idle loop
$PY -m rsw tiles --demo wipe && $PY -m rsw site --demos wipe,gears
cd ../site && npm ci && npm run dev                                                     # http://127.0.0.1:5173/robosimstudio-website/
```

RTX jobs pick their card with `device=cuda:N` and `CUDA_VISIBLE_DEVICES` unset (OVRTX skips every GPU when it
is set); simulation jobs use `CUDA_VISIBLE_DEVICES`.
