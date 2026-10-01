"""The cable task: route a SuperDex cable into three clips (``cable_scene.py``) on the phone page.

The Panda picks up the cable's free end by itself and lifts it; then the hand is the visitors'.
The page is the phone demo's (``wipe_phone.PhoneSession``), with four things of its own: the
cable drawn as a tube (the studio's viewer draws a deformable from its triangles, and a
centerline has none), a map of the route from above, the robot drawn through
``cable_scene.HostDraw`` (~6 ms a frame instead of ~11: what keeps the page at ~0.95x real time),
and no "World model" switch (the path tracer draws deformables from their triangles too, so the
cable would not be in its picture). The opening view comes closer on a landscape screen.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

import cable_scene as cs
import pages

WORDS = {
    "title": "Route the cable",
    "reink": "New cable",
    "hint": "drag me",
    "how": "Drag the orange ball: up lifts the cable, sideways carries it along the clips. "
    "Lay it down into all three. Drag empty space to look around.",
    "done": "Routed!",
}


def main(opts, rest) -> int:
    del rest
    import wipe_phone as wp

    wp.WORDS["cable"] = WORDS
    pages.phone_opts(opts, "cable", attract=25.0)
    opts.warm_world = False  # no World model on this page: see the module docstring

    t = time.time()
    env = cs.build_env()
    holds = cs.Holds(env)
    wp.VIEWS["cable"] = cs.view()
    print(f"[phone] scene built in {pages.stamp(t)}", flush=True)

    class CableSession(wp.PhoneSession):
        """The phone page on the cable: the robot takes the cable's end, then the hand is the visitors'."""

        def __init__(self, sim, *, config, opts) -> None:
            self.holds = holds
            self.tube = None
            self.cable_idx = np.asarray(sim["cable"].particles, dtype=np.int64)
            super().__init__(sim, config=config, opts=opts)
            self.setup_text = "Setting up -- the robot is picking up the cable's end..."
            if "stats" in self.phone_widgets:  # drawn by the base page with the sponge's words
                self.phone_widgets["stats"].content = self.setup_text
            self.viewer.show_particles = False
            self.host_draw = cs.HostDraw(self.viewer)  # ~700 device reads a frame down to a handful
            self.tube = cs.CableTube(self.server, len(self.cable_idx))

        def aim(self, client) -> None:
            """The opening view for this screen's shape: closer on a landscape one."""
            cam = client.camera
            try:
                aspect = float(cam.image_width) / max(float(cam.image_height), 1.0)
            except (TypeError, ValueError):
                aspect = 0.46
            look, offset = (np.asarray(v, dtype=np.float64) for v in cs.view(aspect if aspect > 0 else 0.46))
            cam.up_direction = (0.0, 0.0, 1.0)
            cam.fov = math.radians(55.0)
            cam.position = tuple(float(v) for v in look + offset * self.opts.distance)
            cam.look_at = tuple(float(v) for v in look)

        def _aim_when_sized(self, client) -> None:
            """A browser reports its canvas a moment after it connects: aim again once it has."""
            for _ in range(40):
                time.sleep(0.1)
                try:
                    if float(client.camera.image_width) > 0:
                        self.aim(client)
                        return
                except Exception:  # gone, or not reported yet
                    continue

        def _on_connect(self, client) -> None:
            """Every phone opens on the same view, with "New cable" and "Reset view"."""
            self.aim(client)
            threading.Thread(target=self._aim_when_sized, args=(client,), daemon=True, name="aim").start()
            row = client.gui.add_button_group("", options=(self.words["reink"], "Reset view"), order=2)

            def _row(event) -> None:
                if event.target.value == self.words["reink"]:
                    self.controls.stripe_request = True
                else:
                    self.aim(client)

            row.on_click(_row)
            client.add_notification(title=self.words["title"], body=self.words["how"], auto_close_seconds=8.0)

        def setup(self) -> None:
            env = self.env
            n, t0 = 0, time.time()
            for _ in cs.pre_roll(env, self.holds):
                n += 1
                if n % 3 == 0:
                    self.refresh_view()
                if n % 150 == 0:  # the supervisor reads a silent log as a hang
                    print(f"[phone] pre-roll {n} steps", flush=True)
            self.wipe = cs.Cable(env, self.holds)
            self.map = cs.CableMap(self.wipe)
            print(f"[phone] pre-roll: {n} steps in {time.time() - t0:.1f} s; the hand is the phones' now", flush=True)
            self.controls.touched_at = time.time()

        def restart(self, why: str) -> None:
            if why.startswith("A new film"):
                why = "A fresh cable is going on."
            # Let go first: the engine's restore must not meet a pin from before.
            self.holds.clear()
            # The ball would hang where the old hand was until the robot has the new cable's end.
            if self.controls is not None and self.controls.handle is not None:
                self.controls.handle.visible = False
                self.controls.hint.visible = False
            super().restart(why)

        def set_readouts(self, clean: float) -> None:
            w = self.phone_widgets
            w["progress"].value = round(100.0 * clean, 1)
            now = time.time()
            if self.map is not None and now - self._map_at >= wp.MAP_PERIOD:
                self._map_at = now
                w["map"].image = self.map.render()
            routed = self.wipe.routed if self.wipe is not None else 0
            where = self.map.caption() if self.map is not None else "setting up"
            w["stats"].content = f"**{routed}/3** clips · {where} · sim {self._rate / 60:.2f}×"

        def draw(self) -> None:
            super().draw()
            if self.tube is not None:
                self.tube.update(self.env.state.particle_q.numpy()[self.cable_idx])

    return pages.serve(CableSession, env, opts)
