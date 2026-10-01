"""What every live task's page shares: the phone demo's ``PhoneSession`` options, and serving one.

A task module (``task_<name>.py``) builds its scene, subclasses ``wipe_phone.PhoneSession`` where
the scene needs it, and hands both to :func:`serve`. ``wipe_phone`` is the conference phone
demo's page (``~/rss_ws/phone_demo``), imported by path and never edited.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

RUN_ROOT = Path(os.environ.get("RSW_LIVE", Path.home() / "rss_ws" / "rsw_live"))


def phone_opts(opts, scene: str, *, attract: float, render_every: int = 2) -> None:
    """Fill in the options ``wipe_phone.main`` would have parsed, for a scene it does not know."""
    opts.scene = scene
    opts.attract = attract if opts.attract is None else opts.attract
    opts.render_every = render_every if opts.render_every is None else opts.render_every
    opts.distance = 1.0
    opts.world_load = 0.25
    opts.world_fps = 12.0
    opts.warm_world = True
    opts.watchdog = True
    opts.sponge_look = "shell"


def run_dir(opts) -> Path:
    run = RUN_ROOT / "run" / opts.name
    run.mkdir(parents=True, exist_ok=True)
    return run


def serve(session_cls, env, opts, *, cloth_look: str = "shell") -> int:
    """Put ``env`` on a phone page at ``opts.port`` and run it until it quits.

    With ``opts.share`` the page also gets a viser relay link (``current_url.txt`` in its run
    folder); behind the HTTPS gateway it needs none."""
    import wipe_phone as wp

    run = run_dir(opts)
    config = wp.StudioConfig(
        port=opts.port,
        share=False,
        panels=("phone",),
        render_every=opts.render_every,
        live=False,
        cameras=False,
        render_view="raster",
        out=str(run / "studio"),
        cloth_look=cloth_look,
    )
    session = session_cls(env, config=config, opts=opts)
    print(f"[phone] page: http://localhost:{opts.port}", flush=True)
    (run / "current_url.txt").unlink(missing_ok=True)
    if opts.share:
        threading.Thread(target=wp.keep_shared, args=(session.server, run, opts.port), daemon=True, name="share").start()
    session.serve()
    return 0


def stamp(t0: float) -> str:
    return f"{time.time() - t0:.1f} s"
