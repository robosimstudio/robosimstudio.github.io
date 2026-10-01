"""The towel task: ``cloth_fold/towel_half`` (``towel_scene.py``) on the phone page.

The task's own ``FoldRoutine`` takes the towel's edge, then the carrying hand is the visitors'.
"""

from __future__ import annotations

import time

import numpy as np

import pages
import towel_scene as ts

WORDS = {
    "title": "Fold the towel",
    "reink": "New towel",
    "hint": "drag me",
    "how": "Drag the orange ball: up lifts the towel's edge, sideways carries it over. "
    "Drag empty space to look around.",
    "done": "Folded!",
}


def main(opts, rest) -> int:
    del rest
    import wipe_phone as wp

    wp.WORDS["towel"] = WORDS
    pages.phone_opts(opts, "towel", attract=60.0)

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

    t = time.time()
    env = ts.build_env()
    # Until the hand has the edge, look at the towel lying there from the bench's open side (+x;
    # ``towel_scene.view`` takes over once the carry's axes are known).
    q = env.state.particle_q.numpy()
    dh = env.handles.deformables.get("towel") or next(iter(env.handles.deformables.values()))
    mid = np.asarray(q[np.asarray(dh.particles)].mean(axis=0), dtype=np.float64)
    wp.VIEWS["towel"] = (tuple((mid + np.array([0.0, 0.0, -0.04])).tolist()), (0.66, 0.0, 0.46))
    print(f"[phone] scene built in {pages.stamp(t)}", flush=True)
    return pages.serve(TowelSession, env, opts)
