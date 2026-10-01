"""Run one live task: the phone demo's touch page on one GPU, shared through viser's relay.

    cd ~/rss_ws/RoboSimStudio
    CUDA_VISIBLE_DEVICES=6 .venv/bin/python ~/rss_ws/robosimstudio-website/live/launch.py --task towel --port 8104 --share

``bar``, ``phone`` and ``peel`` are ``~/rss_ws/phone_demo/wipe_phone.py``'s own scenes, run
unchanged (imported by path, never edited). ``towel`` is ``towel_scene.py`` here, on the same
page (``PhoneSession``). A run folder per task -- the log, the current public link
(``current_url.txt``) and its QR code -- goes under ``~/rss_ws/rsw_live/run/<task>``.

How the public link works: viser asks its relay (``https://share.viser.studio/?request_forward``)
for a forward. The relay answers with a random ``https://<words>.share.viser.studio`` URL and a
connection count (16); viser then opens that many outbound TLS connections to the relay, each
paired with a local connection to this page's port. A browser on the URL is piped through one
of them. Nothing listens on the internet here, so the firewall stays shut. The link lasts up to
24 h; ``wipe_phone.keep_shared`` asks for a new one when it drops and rewrites ``current_url.txt``.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
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


def run_towel(opts) -> int:
    import numpy as np

    import towel_scene as ts
    import wipe_phone as wp

    wp.WORDS["towel"] = {
        "title": "Fold the towel",
        "reink": "New towel",
        "hint": "drag me",
        "how": "Drag the orange ball: up lifts the towel's edge, sideways carries it over. "
        "Drag empty space to look around.",
        "done": "Folded!",
    }

    class TowelSession(wp.PhoneSession):
        """The phone page on the towel: the task's own grasp, then the hand is the visitors'."""

        def __init__(self, sim, *, config, opts) -> None:
            super().__init__(sim, config=config, opts=opts)
            self.setup_text = "Setting up -- the robot is taking the towel's edge..."
            if "stats" in self.phone_widgets:  # drawn by the base page with the sponge's words
                self.phone_widgets["stats"].content = self.setup_text

        def setup(self) -> None:
            env = self.env
            n, t0, fold = 0, time.time(), None
            for fold in ts.pre_roll(env):
                n += 1
                if n % 2 == 0:
                    self.refresh_view()
                if n % 150 == 0:  # the supervisor reads a silent log as a hang
                    print(f"[phone] pre-roll {n} steps ({fold.phase})", flush=True)
            self.wipe = ts.Towel(env, fold)
            self.map = ts.TowelMap(self.wipe)
            wp.VIEWS["towel"] = ts.view(self.wipe)
            print(f"[live] towel view {np.round(wp.VIEWS['towel'], 3).tolist()}, C {np.round(self.wipe.C, 3).tolist()}", flush=True)
            for client in self.server.get_clients().values():
                self.aim(client)
            print(f"[phone] pre-roll: {n} steps in {time.time() - t0:.1f} s; the hand is the phones' now", flush=True)
            self.controls.touched_at = time.time()

        def restart(self, why: str) -> None:
            if why.startswith("A new film"):
                why = "A fresh towel is going on."
            if self.wipe is not None:
                ts.release(self.env, self.wipe.fold)
            super().restart(why)

        def set_readouts(self, clean: float) -> None:
            w = self.phone_widgets
            w["progress"].value = round(100.0 * clean, 1)
            now = time.time()
            if self.map is not None and now - self._map_at >= wp.MAP_PERIOD:
                self._map_at = now
                w["map"].image = self.map.render()
            touch = self.map.caption() if self.map is not None else "setting up"
            w["stats"].content = f"**{100 * clean:.0f} %** folded · {touch} · sim {self._rate / 60:.2f}×"

    run = RUN_ROOT / "run" / opts.name
    run.mkdir(parents=True, exist_ok=True)
    t = time.time()
    env = ts.build_env()
    # Until the hand has the edge, look at the towel lying there from the bench's open side (+x;
    # ``towel_scene.view`` takes over once the carry's axes are known).
    q = env.state.particle_q.numpy()
    dh = env.handles.deformables.get("towel") or next(iter(env.handles.deformables.values()))
    mid = np.asarray(q[np.asarray(dh.particles)].mean(axis=0), dtype=np.float64)
    wp.VIEWS["towel"] = (tuple((mid + np.array([0.0, 0.0, -0.04])).tolist()), (0.66, 0.0, 0.46))
    print(f"[phone] scene built in {time.time() - t:.1f} s", flush=True)
    config = wp.StudioConfig(
        port=opts.port,
        share=False,
        panels=("phone",),
        render_every=opts.render_every,
        live=False,
        cameras=False,
        render_view="raster",
        out=str(run / "studio"),
        cloth_look="shell",
    )
    session = TowelSession(env, config=config, opts=opts)
    print(f"[phone] page: http://localhost:{opts.port}", flush=True)
    (run / "current_url.txt").unlink(missing_ok=True)
    if opts.share:
        threading.Thread(target=wp.keep_shared, args=(session.server, run, opts.port), daemon=True, name="share").start()
    session.serve()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", choices=("bar", "phone", "peel", "towel"), required=True)
    p.add_argument("--name", default="", help="run folder under run/ (default: the task)")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--share", action="store_true", help="a public link through viser's relay")
    p.add_argument("--attract", type=float, default=None, help="seconds idle before the autopilot; 0 = never")
    p.add_argument("--render-every", type=int, default=None, help="draw the pages every Nth control step")
    opts, rest = p.parse_known_args(argv)
    gpu = _check_gpu()
    opts.name = opts.name or opts.task
    print(f"[live] {opts.task} on GPU {gpu}, port {opts.port}, run folder {RUN_ROOT / 'run' / opts.name}", flush=True)

    if opts.task != "towel":
        import wipe_phone as wp

        wp.HERE = RUN_ROOT  # its run folder is HERE/run/<name>: keep ours out of phone_demo/
        args = ["--scene", opts.task, "--name", opts.name, "--port", str(opts.port)]
        if opts.share:
            args.append("--share")
        if opts.attract is not None:
            args += ["--attract", str(opts.attract)]
        if opts.render_every is not None:
            args += ["--render-every", str(opts.render_every)]
        return wp.main(args + rest)

    # the phone page's options, as wipe_phone.main would have parsed them for a towel
    opts.scene = "towel"
    opts.attract = 60.0 if opts.attract is None else opts.attract
    opts.render_every = 2 if opts.render_every is None else opts.render_every
    opts.distance = 1.0
    opts.world_load = 0.25
    opts.world_fps = 12.0
    opts.warm_world = True
    opts.watchdog = True
    opts.sponge_look = "shell"
    return run_towel(opts)


if __name__ == "__main__":
    sys.exit(main())
