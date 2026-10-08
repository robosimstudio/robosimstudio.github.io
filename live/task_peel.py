"""The peel task, faster: the phone demo's own ``--scene peel`` page on a cheaper stepping budget,
with an instant "New film".

``wipe_phone.py`` runs the peel page itself (``peel_scene.py``: ``rgbs/panel``, the routine's
pre-roll to the latched tab, the ``Peel`` driver, the bond map). This module changes three things
and nothing else, all at runtime and all from here (``wipe_phone``/``peel_scene`` are imported by
path, never edited); ``peel_fast.py`` has the measurements behind each:

* **the stepping budget** (``peel_fast.apply_budget``): the scene is built exactly as the task
  stages it -- settle, bonds, anchor -- and then steps at ``peel_fast.SUBSTEPS`` x
  ``peel_fast.ITERATIONS`` (20 x 10) instead of the task's 20 x 80: same 1/1200 s dt, an eighth of
  the VBD sweeps. 5.6 -> ~23 control steps/s on the page (0.09x -> ~0.38x real time);
* **the coupler's collision pass** keeps only its soft (particle-shape) half, the half VBD reads
  (``peel_fast.install_soft_contacts_only``, checked against the full pass when it installs);
* **"New film"** restores the state the first pre-roll ended in (``peel_fast.FilmSnapshot``)
  instead of resetting and replaying the routine's ~800-step hold-and-pinch. If a restore ever
  comes back unhealthy the page falls back to the original reset and pre-roll.

* **the film shows**: the task draws it at 10 % opacity, tuned so the path tracer matches the real
  clip's clear film (``rgbs/tasks/rgbs/peel/layout.py``); in the browser's raster view that is all
  but invisible from any angle, so this page draws it at :data:`FILM_SHOWN` (``show_film``).

``PEEL_PROFILE=1`` in the environment arms the studio's loop profiler and prints its table with
every 30 s status line.
"""

from __future__ import annotations

import dataclasses
import os
import time

import numpy as np
import peel_fast as pf

SETUP_TEXT = "Setting up -- the robot is taking the film's tab (about a minute)..."
"""Shown while the routine's pre-roll plays: once per process now, ~50 s build + ~50 s pre-roll."""
FILM_SHOWN = 0.35
"""The film's opacity on this page (the panel's black block once shipped it at 0.35)."""


def show_film() -> None:
    """Draw the film at :data:`FILM_SHOWN` in the browser: the viewer's garment, not the scene's spec."""
    from robosimstudio_studio.garments import ViewerStudio

    set0 = ViewerStudio.set_garments
    if getattr(set0, "_rsw_film", False):
        return

    def set_garments(self, garments):
        return set0(self, [dataclasses.replace(g, opacity=max(float(g.opacity), FILM_SHOWN)) if g.name == "film" else g
                           for g in garments])

    set_garments._rsw_film = True
    ViewerStudio.set_garments = set_garments


def make_session(wp, pl):
    """The phone page's session with the snapshot restart, for ``wipe_phone.main`` to build."""

    class PeelSession(wp.PhoneSession):
        """``PhoneSession`` whose "New film" restores the latched tab rather than re-running the routine."""

        def __init__(self, sim, *, config, opts) -> None:
            self.snapshot: pf.FilmSnapshot | None = None
            super().__init__(sim, config=config, opts=opts)
            self.setup_text = SETUP_TEXT
            if "stats" in self.phone_widgets:  # drawn by the base page with its own words
                self.phone_widgets["stats"].content = self.setup_text
            if os.environ.get("PEEL_PROFILE"):
                from robosimstudio.core.profile import parse_modes

                self.profile = parse_modes("wall,device")
                self.profile.start()

        def setup(self) -> None:
            if self.scene_kind != "peel":
                return super().setup()
            snap = self.snapshot
            if snap is not None:
                t0 = time.time()
                try:
                    snap.restore(self.env)
                    ok = snap.healthy(self.env)
                except Exception as exc:  # never strand the page: the caller falls back to the routine
                    print(f"[phone] snapshot restore failed ({type(exc).__name__}: {exc})", flush=True)
                    ok = False
                if not ok:
                    self.snapshot = None
                    raise RuntimeError("the film snapshot did not restore cleanly")
                routine = snap.routine
                self.wipe = pl.Peel(self.env, routine)
                self.map = pl.BondMap(self.env, routine, forward=-np.asarray(wp.VIEWS["peel"][1]))
                print(
                    f"[phone] new film #{snap.restores} from the snapshot in {time.time() - t0:.2f} s; "
                    "the hand is the phones' now",
                    flush=True,
                )
                self.controls.touched_at = time.time()
                return
            t0 = time.time()
            super().setup()  # the task's routine, to the latched tab
            self.snapshot = pf.FilmSnapshot(self.env, self.wipe.routine)
            print(f"[phone] film snapshot taken ({self.snapshot.n_latched} tab particles latched, "
                  f"setup {time.time() - t0:.1f} s)", flush=True)

        def restart(self, why: str) -> None:
            if self.snapshot is None or self.scene_kind != "peel":
                return super().restart(why)
            print(f"[phone] starting over: {why}", flush=True)
            self.phone_widgets["stats"].content = "A new film is going on..."
            self.set_paused(True)
            self.on_reset({}, {})
            wipe = self.wipe
            try:
                self.wipe = None
                self.setup()
            except Exception as exc:
                print(f"[phone] {exc}; resetting and replaying the routine instead", flush=True)
                self.wipe = wipe  # the original restart releases its latch
                self.snapshot = None
                return super().restart(why)
            self.set_paused(False)
            self.begin_run()

        def log_line(self) -> None:
            super().log_line()
            prof = self.profile
            if getattr(prof, "enabled", False) and prof.steps:
                if prof.device is not None:
                    prof.device.drain()
                print(prof.report(), flush=True)
                prof.reset()

    return PeelSession


def main(opts, rest) -> int:
    import peel_scene as pl
    import wipe_phone as wp

    built = pl.build_env

    def build_env():
        env = built()
        was = pf.apply_budget(env)
        print(f"[phone] stepping budget {pf.SUBSTEPS} substeps x {pf.ITERATIONS} VBD sweeps "
              f"(the task's {was[0]} x {was[1]}); {pf.install_soft_contacts_only(env)}", flush=True)
        return env

    pl.build_env = build_env
    show_film()
    wp.PhoneSession = make_session(wp, pl)
    args = ["--scene", "peel", "--name", opts.name, "--port", str(opts.port)]
    if opts.share:
        args.append("--share")
    if opts.attract is not None:
        args += ["--attract", str(opts.attract)]
    if opts.render_every is not None:
        args += ["--render-every", str(opts.render_every)]
    return wp.main(args + list(rest))
