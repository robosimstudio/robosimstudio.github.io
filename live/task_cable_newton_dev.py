"""The cable-routing task: a Newton-native cable on the phone page (``cable_newton_scene.py``).

The Panda holds the cable's end from the start; the visitors' finger is its hand. Lead the cable
round the three orange clips, on the side the dashed line takes, and the readout counts them off.
"""

from __future__ import annotations

import time

import numpy as np

import cable_newton_scene as cs
import pages

SCENE = "cable_newton"
WORDS = {
    "title": "Route the cable",
    "reink": "New cable",
    "hint": "drag me",
    "how": "Drag the orange ball to lead the cable along the dashed line: in front of clip 1, "
    "behind 2, in front of 3. Drag empty space to look around.",
    "done": "Routed!",
}
ROUTE_COLOR = (235, 235, 240)
CLIP_TODO = (236, 238, 242)
CLIP_DONE = (40, 200, 110)
PACE_DEBT = 0.25
"""Seconds of lateness the loop may make up by running ahead of real time."""


def main(opts, rest) -> int:
    del rest
    import wipe_phone as wp

    wp.WORDS[SCENE] = WORDS
    pages.phone_opts(opts, SCENE, attract=30.0)

    class CableSession(wp.PhoneSession):
        """The phone page on the routing board: no pre-roll, the hand holds the cable from the start."""

        def __init__(self, sim, *, config, opts) -> None:
            super().__init__(sim, config=config, opts=opts)
            self.setup_text = "Setting up -- laying a fresh cable..."
            if "stats" in self.phone_widgets:
                self.phone_widgets["stats"].content = self.setup_text
            self.marks: dict = {}
            self.cap_state: list[bool] = []
            # Paced here rather than by the studio: its pacer drops any debt, so a loop whose
            # steps alternate cheap and dear (a frame drawn every second step) sleeps through
            # the cheap ones and still overruns on the dear ones -- 0.87x of real time on a loop
            # that averages 1.3x. This one carries up to PACE_DEBT of lateness forward.
            self.speed = 0.0
            self._pace = None

        def setup(self) -> None:
            env = self.env
            t0 = time.time()
            self.wipe = cs.Router(env)
            for n in range(30):  # let the lifted end settle into the hand
                self.wipe.drive()
                env.step_teleop()
                if n % 10 == 0:
                    self.refresh_view()
            self.map = cs.RouteMap(self.wipe)
            wp.VIEWS[SCENE] = cs.view(self.wipe)
            self.draw_route()
            for client in self.server.get_clients().values():
                self.aim(client)
            print(f"[phone] fresh cable in {time.time() - t0:.1f} s; the hand is the phones' now", flush=True)
            self.controls.touched_at = time.time()

        def draw_route(self) -> None:
            """The dashed route on the board and a numbered cap on each clip (scene-wide, once)."""
            if self.marks:
                return
            r = self.wipe
            o = r.origin
            path = cs.catmull_rom(cs.route_points(), per_leg=40)
            z = o[2] + cs.BOARD_T + 0.0015
            pts = np.column_stack([path[:, 0] + o[0], path[:, 1] + o[1], np.full(len(path), z)])
            s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            dashes = []
            dash, gap = 0.012, 0.008
            t = 0.0
            while t + dash < s[-1]:
                a = np.array([np.interp(t, s, pts[:, i]) for i in range(3)])
                b = np.array([np.interp(t + dash, s, pts[:, i]) for i in range(3)])
                dashes.append((a, b))
                t += dash + gap
            seg = np.asarray(dashes, dtype=np.float32)
            scene = self.server.scene
            self.marks["route"] = scene.add_line_segments(
                "/cable/route", points=seg, colors=ROUTE_COLOR, line_width=3.0
            )
            top = o[2] + cs.BOARD_T + 2 * (cs.CLIP_HALF + cs.CLIP_R)
            for k in range(len(cs.CLIP_Y)):
                c = cs.clip_xy(k) + o[:2]
                self.marks[f"cap{k}"] = scene.add_icosphere(
                    f"/cable/cap{k + 1}", radius=0.0125, color=CLIP_TODO, subdivisions=2,
                    position=(float(c[0]), float(c[1]), float(top + 0.002)),
                )
                self.marks[f"num{k}"] = scene.add_label(
                    f"/cable/num{k + 1}", str(k + 1), position=(float(c[0]), float(c[1]), float(top + 0.03)),
                    anchor="bottom-center", font_screen_scale=1.1,
                )

        def restart(self, why: str) -> None:
            if why.startswith("A new film"):
                why = "A fresh cable is going on."
            super().restart(why)

        def set_readouts(self, clean: float) -> None:
            self._timed("readouts", self._set_readouts, clean)

        def _set_readouts(self, clean: float) -> None:
            w = self.phone_widgets
            w["progress"].value = round(100.0 * clean, 1)
            now = time.time()
            if self.map is not None and now - self._map_at >= wp.MAP_PERIOD:
                self._map_at = now
                w["map"].image = self._timed("map", self.map.render)
            r = self.wipe
            routed = r.routed if r is not None else 0
            state = [k < routed for k in range(len(cs.CLIP_Y))]
            if state != self.cap_state:
                self.cap_state = state
                for k, done in enumerate(state):
                    cap = self.marks.get(f"cap{k}")
                    if cap is not None:
                        cap.color = CLIP_DONE if done else CLIP_TODO
            touch = self.map.caption() if self.map is not None else "setting up"
            w["stats"].content = f"**{routed}/3** clips routed · {touch} · sim {self._rate / cs.CONTROL_HZ:.2f}×"

        # -- where a second goes: timed per phase, reported with the log line ------------------
        _t = {"tick": 0.0, "draw": 0.0, "readouts": 0.0, "drain": 0.0, "step": 0.0, "refresh": 0.0, "map": 0.0}
        _n = {"tick": 0, "draw": 0, "readouts": 0, "drain": 0, "step": 0, "refresh": 0, "map": 0}

        def _timed(self, key, fn, *args):
            t0 = time.perf_counter()
            try:
                return fn(*args)
            finally:
                self._t[key] += time.perf_counter() - t0
                self._n[key] += 1

        def tick(self) -> bool:
            stepped = self._timed("tick", super().tick)
            if stepped:
                now = time.perf_counter()
                due = (self._pace if self._pace is not None else now) + 1.0 / cs.CONTROL_HZ
                self._pace = max(due, now - PACE_DEBT)
                if self._pace > now:
                    time.sleep(self._pace - now)
            else:
                self._pace = None
            return stepped

        def draw(self) -> None:
            self._timed("draw", super().draw)

        def refresh_view(self) -> None:
            self._timed("refresh", super().refresh_view)

        def build_panels(self) -> None:
            super().build_panels()
            drain = self.controls.drain
            self.controls.drain = lambda: self._timed("drain", drain)
            step = self.env.step_teleop
            self.env.step_teleop = lambda: self._timed("step", step)

        def log_line(self) -> None:
            super().log_line()
            r = self.wipe
            if r is not None:
                print(f"[cable] routed {r.routed}/3 (best {r.best}) · pull {r.pull:.1f} N", flush=True)
            parts = " · ".join(f"{k} {1000 * self._t[k] / max(self._n[k], 1):.1f} ms x{self._n[k]}" for k in self._t)
            print(f"[cable] loop: {parts}", flush=True)
            for k in self._t:
                self._t[k], self._n[k] = 0.0, 0

    t = time.time()
    env = cs.build_env()
    wp.VIEWS[SCENE] = cs.view_of(env)
    print(f"[phone] scene built in {pages.stamp(t)}", flush=True)
    return pages.serve(CableSession, env, opts)
