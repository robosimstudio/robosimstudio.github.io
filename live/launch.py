"""Run one live task: the phone demo's touch page on one GPU, shared through viser's relay.

    cd ~/rss_ws/RoboSimStudio
    CUDA_VISIBLE_DEVICES=6 .venv/bin/python ~/rss_ws/robosimstudio-website/live/launch.py --task towel --port 8104 --share

A task is a ``task_<name>.py`` here (``main(opts, rest)``: build the scene, serve it on the
phone page with ``pages.serve``), or, without one, one of ``~/rss_ws/phone_demo/wipe_phone.py``'s
own scenes (bar, phone, peel), run unchanged (imported by path, never edited). A run folder per
task -- the log, and with ``--share`` the current public link (``current_url.txt``) and its QR
code -- goes under ``~/rss_ws/rsw_live/run/<task>``. A ``task_<name>.py`` is picked up the next
time that task starts, so develop under another name and rename when it is ready.

How the public link works: viser asks its relay (``https://share.viser.studio/?request_forward``)
for a forward. The relay answers with a random ``https://<words>.share.viser.studio`` URL and a
connection count (16); viser then opens that many outbound TLS connections to the relay, each
paired with a local connection to this page's port. A browser on the URL is piped through one
of them. Nothing listens on the internet here, so the firewall stays shut. The link lasts up to
24 h; ``wipe_phone.keep_shared`` asks for a new one when it drops and rewrites ``current_url.txt``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHONE = Path.home() / "rss_ws" / "phone_demo"
RUN_ROOT = Path(os.environ.get("RSW_LIVE", Path.home() / "rss_ws" / "rsw_live"))
ALLOWED_GPUS = {0, 1, 5, 6, 7}
"""2-4 run the conference phone demos: never these."""

sys.path.insert(0, str(PHONE))
sys.path.insert(0, str(HERE))


def _check_gpu() -> int:
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not v.isdigit() or int(v) not in ALLOWED_GPUS:
        raise SystemExit(f"run with CUDA_VISIBLE_DEVICES=<one of {sorted(ALLOWED_GPUS)}> (got {v!r})")
    return int(v)


PHONE_SCENES = ("bar", "phone", "peel")
"""Scenes ``wipe_phone.py`` runs itself, for a task with no ``task_<name>.py`` of its own."""


def tasks() -> list[str]:
    own = sorted(p.stem[len("task_"):] for p in HERE.glob("task_*.py"))
    return sorted(set(own) | set(PHONE_SCENES))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", choices=tasks(), required=True,
                   help="a task_<name>.py here, or one of the phone demo's own scenes")
    p.add_argument("--name", default="", help="run folder under run/ (default: the task)")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--share", action="store_true", help="a public link through viser's relay")
    p.add_argument("--attract", type=float, default=None, help="seconds idle before the autopilot; 0 = never")
    p.add_argument("--render-every", type=int, default=None, help="draw the pages every Nth control step")
    opts, rest = p.parse_known_args(argv)
    gpu = _check_gpu()
    opts.name = opts.name or opts.task
    print(f"[live] {opts.task} on GPU {gpu}, port {opts.port}, run folder {RUN_ROOT / 'run' / opts.name}", flush=True)

    import wipe_phone as wp

    wp.HERE = RUN_ROOT  # its run folder is HERE/run/<name>: keep ours out of phone_demo/
    if (HERE / f"task_{opts.task}.py").exists():
        module = importlib.import_module(f"task_{opts.task}")
        return module.main(opts, rest)
    args = ["--scene", opts.task, "--name", opts.name, "--port", str(opts.port)]
    if opts.share:
        args.append("--share")
    if opts.attract is not None:
        args += ["--attract", str(opts.attract)]
    if opts.render_every is not None:
        args += ["--render-every", str(opts.render_every)]
    return wp.main(args + rest)


if __name__ == "__main__":
    sys.exit(main())
