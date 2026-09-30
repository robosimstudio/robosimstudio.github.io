# RoboSimStudio — demos

A demo-only website: drag the robot and feel RoboSimStudio's physics, rendered with its RTX path tracer.
Live at <https://keplerc.github.io/robosimstudio-website/>.

Nothing is simulated in the browser. Every state on screen was produced by RoboSimStudio, and every
path-traced pixel by its RTX renderer:

- **Free-drag demos** (the sponge, the film, the cable, the towel): thousands of "virtual visitor" drags
  were simulated offline. While you drag, the page picks, ten times a second, the recorded frame whose hand
  position, velocity, near-future path and deformable state best continue what is on screen toward your
  ball (motion matching), and draws it live over an RTX plate of the static scene. Let go and the robot
  settles on a frame that has an RTX still, which dissolves in. Idle, it plays an RTX video of a recorded take.
- **Track demos** (gears, plates, cables): a recorded successful rollout plays as RTX video; grab the gripper
  to scrub it live, let go to land on the exact RTX frame.

## Layout

| path | what |
|---|---|
| `pipeline/` | the offline pipeline (`python -m rsw <stage>`), run with RoboSimStudio's venv: takes, export, db, plate, stills, attract, track, tiles, site |
| `pipeline/schema/CONTRACTS.md` | the data contract between the pipeline and the page |
| `site/` | the page: Vite + TypeScript + three.js; `src/core` is DOM-free and unit-tested |
| `scripts/deploy.sh` | build, check, and force-push the site as one orphan commit on `gh-pages` |

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
