"""The cable task: route a SuperDex cable into three clips (``cable_scene.py``) on the phone page.

The Panda picks up the cable's free end by itself and lifts it; then the hand is the visitors'.
The page is the phone demo's (``wipe_phone.PhoneSession``), with these things of its own: the
cable drawn as a tube (the studio's viewer draws a deformable from its triangles, and a
centerline has none), a map of the route from above, the robot drawn through
``cable_scene.HostDraw`` (~6 ms a frame instead of ~11: what keeps the page at ~0.95x real time),
and no "World model" switch (the path tracer draws deformables from their triangles too, so the
cable would not be in its picture). The opening view comes closer on a landscape screen.

The hand moves in 3D under a finger (``cable_scene.Cable.drag``): the orange ball lifts and carries
it, and the ball's shadow on the bench -- a second handle, drawn here -- slides it in and out. Left
alone, the autopilot loops physically -- route, rest, lift it out and lay it back, rest -- and never
resets the cable. A fresh cable ("New cable", or the watchdog finding the cable broken) is the
pre-roll's moment restored in milliseconds (``cable_scene.Snapshot``), with the page drawing and the
ball in the hand throughout; only when that cannot work does the page reset and replay the pick.
"""

from __future__ import annotations

import math
import os
import threading
import time

import numpy as np

import backdrop
import cable_scene as cs
import pages

WORDS = {
    "title": "Route the cable",
    "reink": "New cable",
    "hint": "drag me",
    "how": "Drag the orange ball: up lifts the cable, sideways carries it along the clips. "
    "Drag its shadow on the bench to move the hand in or out. Lay the cable into all three. "
    "Drag empty space to turn the view.",
    "done": "Routed!",
}

SHADOW_R = 0.034
"""Radius [m] of the ball's shadow on the bench, the handle that slides the hand over it."""

LOG_DRAG = bool(os.environ.get("CABLE_LOG_DRAG"))
"""Development: a log line at the start and end of every drag, and after each autopilot rest."""

RESCUE_GAP = 10.0
"""Seconds between two watchdog rescues from the snapshot: a cable that breaks again at once is
given the full reset and a fresh pick instead."""


def main(opts, rest) -> int:
    del rest
    import wipe_phone as wp

    wp.WORDS["cable"] = WORDS
    pages.phone_opts(opts, "cable", attract=25.0)
    opts.warm_world = False  # no World model on this page: see the module docstring

    t = time.time()
    env = cs.build_env()
    print(f"[phone] {backdrop.dress(env, 'cable')}", flush=True)
    holds = cs.Holds(env)
    wp.VIEWS["cable"] = cs.view()
    print(f"[phone] scene built in {pages.stamp(t)}", flush=True)

    class CableControls(wp.PhoneControls):
        """The page's finger handling, with the ball's shadow on the bench as a second handle, and
        the hand told where the camera was when a drag began (a finger's ray comes from there)."""

        shadow = None

        def build(self, gui) -> None:
            super().build(gui)
            self.shadow = self.server.scene.add_cylinder(
                "/phone/shadow",
                radius=SHADOW_R,
                height=0.002,
                color=wp.HANDLE_IDLE,
                opacity=0.55,
                radial_segments=40,
                cast_shadow=False,
                receive_shadow=False,
                position=(0.6, 0.0, 0.70),
            )
            self.shadow.visible = False

            @self.shadow.on_drag
            async def _(event) -> None:
                cam = event.client.camera
                view = (np.asarray(cam.position, dtype=np.float64), np.asarray(cam.look_at, dtype=np.float64), "shadow")
                self.events.append(
                    (event.phase, event.client_id, np.asarray(event.end_position, dtype=np.float64), view)
                )

        def _event(self, phase, cid, end, view, now, wipe) -> None:
            kind = view[2] if len(view) > 2 else "ball"
            view = view[:2]
            mine = cid == self.driver and self.anchor is not None
            super()._event(phase, cid, end, view, now, wipe)
            if phase == "start" and self.driver == cid and self.anchor is not None and self.anchor[0] is end:
                wipe.grab = (
                    np.asarray(view[0], dtype=np.float64),
                    np.asarray(view[1], dtype=np.float64),
                    np.asarray(end, dtype=np.float64),
                )
                wipe.grab_kind = kind
                if LOG_DRAG:
                    eye, look = (np.asarray(v, dtype=np.float64) for v in view)
                    f = look - eye
                    elev = math.degrees(math.atan2(-f[2], math.hypot(f[0], f[1])))
                    azim = math.degrees(math.atan2(-f[1], -f[0]))
                    print(
                        f"[drag] start {kind}: hand {np.round(wipe.local(wipe.cursor - wipe.ball), 3)} camera elevation "
                        f"{elev:.0f} deg azimuth {azim:.0f} deg",
                        flush=True,
                    )
            elif phase == "end" and mine and LOG_DRAG:
                print(
                    f"[drag] end: hand target {np.round(wipe.local(wipe.cursor - wipe.ball), 3)} "
                    f"tcp {np.round(wipe.local(wipe.tcp()), 3)} routed {wipe.routed}/3",
                    flush=True,
                )

        def refresh(self) -> None:
            super().refresh()
            wipe = self.session.wipe
            if wipe is None or self.shadow is None:
                return
            at = wipe.cursor if self.dragging else wipe.handle_point()
            self.shadow.position = tuple(float(v) for v in wipe.shadow_point(at))
            sliding = self.dragging and wipe.grab_kind == "shadow"
            self.shadow.color = wp.HANDLE_HELD if sliding else wp.HANDLE_IDLE
            self.shadow.visible = bool(self.handle is not None and self.handle.visible)

    class CableSession(wp.PhoneSession):
        """The phone page on the cable: the robot takes the cable's end, then the hand is the visitors'."""

        def __init__(self, sim, *, config, opts) -> None:
            self.holds = holds
            self.tube = None
            self.snapshot: cs.Snapshot | None = None
            self._rescued_at = -1e9
            self.cable_idx = np.asarray(sim["cable"].particles, dtype=np.int64)
            super().__init__(sim, config=config, opts=opts)
            self.setup_text = "Setting up -- the robot is picking up the cable's end..."
            if "stats" in self.phone_widgets:  # drawn by the base page with the sponge's words
                self.phone_widgets["stats"].content = self.setup_text
            self.viewer.show_particles = False
            self.host_draw = cs.HostDraw(self.viewer)  # ~700 device reads a frame down to a handful
            self.tube = cs.CableTube(self.server, len(self.cable_idx))

        def build_panels(self) -> None:
            super().build_panels()
            self.controls = CableControls(self)
            self.panels = [self.controls, self.world]

        def aim(self, client) -> None:
            """The opening view for this screen's shape: closer on a landscape one."""
            cam = client.camera
            try:
                aspect = float(cam.image_width) / max(float(cam.image_height), 1.0)
            except Exception:  # no camera message yet
                aspect = 0.46
            look, offset = (np.asarray(v, dtype=np.float64) for v in cs.view(aspect if aspect > 0 else 0.46))
            cam.up_direction = (0.0, 0.0, 1.0)
            cam.fov = math.radians(55.0)
            cam.position = tuple(float(v) for v in look + offset * self.opts.distance)
            cam.look_at = tuple(float(v) for v in look)

        @staticmethod
        def _sized(client) -> bool:
            try:
                return float(client.camera.image_width) > 0
            except Exception:  # gone, or no camera message yet
                return False

        def _aim_when_sized(self, client, state) -> None:
            """The fallback for a browser whose first sized camera message was missed."""
            for _ in range(80):
                time.sleep(0.05)
                if state["aimed"]:
                    return
                if self._sized(client):
                    state["aimed"] = True
                    self.aim(client)
                    return

        def _on_connect(self, client) -> None:
            """Every phone opens on the same view, with "New cable" and "Reset view". A browser that
            has not reported its canvas yet is aimed on its first camera message that has one (about
            50 ms after it connects, ahead of the scene's meshes), not on the portrait fallback:
            aimed with that, a desktop drew its first ~0.4 s in the phone's framing and then jumped
            (``cable3d/load_desk``)."""
            state = {"aimed": self._sized(client)}
            if state["aimed"]:
                self.aim(client)

            @client.camera.on_update
            def _(_cam) -> None:
                if not state["aimed"] and self._sized(client):
                    state["aimed"] = True
                    self.aim(client)

            threading.Thread(target=self._aim_when_sized, args=(client, state), daemon=True, name="aim").start()
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
            self.wipe.on_new = self.fresh_cable
            if LOG_DRAG:
                self.wipe.on_lap = lambda what, n: print(f"[auto] lap {what}: {n}/3", flush=True)
            self.map = cs.CableMap(self.wipe)
            if self.snapshot is not None:
                self.snapshot.release()
            self.snapshot = cs.Snapshot(env, self.holds)
            print(f"[phone] pre-roll: {n} steps in {time.time() - t0:.1f} s; the hand is the phones' now", flush=True)
            self.controls.touched_at = time.time()

        # -- a fresh cable --------------------------------------------------------------
        def fresh_cable(self, why: str) -> None:
            """The pre-roll's moment, put back: a fresh cable in the hand at the start, in
            milliseconds, with the page drawing throughout. Main thread (the page's drain or loop)."""
            snap = self.snapshot
            if snap is None or not snap.valid():
                self.full_restart(why)
                return
            t = time.perf_counter()
            try:
                snap.restore()
                self.wipe.reseed()
                ok = bool(np.isfinite(self.wipe.particles()).all())
            except Exception as exc:  # the engine refused the state: start over properly
                print(f"[phone] snapshot restore failed ({type(exc).__name__}: {exc})", flush=True)
                ok = False
            if not ok:
                self.snapshot = None
                self.full_restart(why)
                return
            controls = self.controls
            controls.autopilot = None  # its generator was walking the old cable
            controls._celebrated = False
            if controls.dragging:  # a finger mid-drag would pull the hand back to the old cable
                controls.dragging, controls.anchor = False, None
                controls.released_at = time.time()
            if why.startswith("New cable"):
                controls.touched_at = time.time()  # a visitor asked: the autopilot waits for them
            self.on_reset({}, {})
            print(
                f"[phone] fresh cable ({why.rstrip('.').lower()}) from the snapshot in "
                f"{1000 * (time.perf_counter() - t):.0f} ms (#{snap.restores})",
                flush=True,
            )
            if why.startswith("Something"):  # nobody asked: say why the cable jumped
                for client in self.server.get_clients().values():
                    client.add_notification(
                        title="A fresh cable",
                        body="The old one slipped, so here is a new one.",
                        auto_close_seconds=3.0,
                        with_close_button=False,
                    )

        def full_restart(self, why: str) -> None:
            """Reset the scene and replay the pick: the slow way, when the snapshot cannot be used."""
            if self.snapshot is not None:
                self.snapshot.release()
                self.snapshot = None
            if why.startswith("A new film"):
                why = "A fresh cable is going on."
            # Let go first: the engine's restore must not meet a pin from before.
            self.holds.clear()
            # The ball would hang where the old hand was until the robot has the new cable's end.
            if self.controls is not None and self.controls.handle is not None:
                self.controls.handle.visible = False
                self.controls.hint.visible = False
                if self.controls.shadow is not None:
                    self.controls.shadow.visible = False
            super().restart(why)

        def restart(self, why: str) -> None:
            """The page's loop asks for this when the watchdog finds the cable broken (or a reset
            was requested): the snapshot first, the full reset when it just failed."""
            now = time.time()
            if why.startswith("Something") and now - self._rescued_at < RESCUE_GAP:
                self.full_restart(why)
                return
            if why.startswith("Something"):
                self._rescued_at = now
            self.fresh_cable(why)

        def set_readouts(self, clean: float) -> None:
            w = self.phone_widgets
            w["progress"].value = round(100.0 * clean, 1)
            if clean < 0.95 and self.controls is not None:
                self.controls._celebrated = False  # taken out again: the next route is cheered too
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
