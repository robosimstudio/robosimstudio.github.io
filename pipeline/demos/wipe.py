"""Wipe the bar: the Surface Preparing twin, driven by virtual visitors.

Everything physical is the booth demo's (``~/rss_ws/phone_demo/wipe_phone.py``):
the scene is ``rgbs/scripts/surface_cleaning.py``'s, the pick is the task's own
``Wiper`` up to ``phase == "wipe"``, and the hand is ``Wipe`` -- OscSetpoint + IK
toward a cursor clamped to the ``Workspace`` box around the seat. A visitor's
drag is 2-D: ``a`` along the bar, ``b`` height over the seat (down presses).
"""

from __future__ import annotations

import numpy as np

from rsw import paths
from rsw.gestures import Box, GestureConfig
from rsw.ink import InkMirror

ID = "wipe"
TAKE_SECONDS = 240.0
LIFT_STEPS = 90


class Spec:
    id = ID
    deform_frame = "tcp"
    deform_name = "sponge"

    def build(self, device=None):
        paths.use_rss(paths.PHONE)
        import surface_cleaning as sc  # noqa: F401  (rgbs/scripts)
        import wipe_phone as wp  # noqa: F401  (phone_demo)

        self.sc, self.wp = sc, wp
        sim = sc.build_scene().build(control_mode="joint_position", device=device)
        return sim

    def pre_roll(self, sim):
        sc, wp = self.sc, self.wp
        wiper = sc.Wiper(sim, verbose=False)
        steps = wiper.steps()
        n = 0
        while wiper.phase != "wipe":
            next(steps)
            sim.step(wiper.action(), mode="joint_position")
            n += 1
            if n > 20000:
                raise RuntimeError("pre-roll never reached the wipe")
        w = wp.Wipe(sim, wiper, "bar")
        for _ in range(LIFT_STEPS):  # lift off the bar, as PhoneSession.setup
            w.drive()
            sim.step_teleop()
        w.new_stripe()
        self.pre_roll_steps = n
        self.dt = 1.0 / float(sim.request.physics.control_hz)
        self.ink = InkMirror(sim.scene.by_name["taint"], self.dt)
        self.sponge = np.asarray(w.sponge_idx, dtype=np.int64)
        self.lead_hist: list[float] = []
        return w

    # -------------------------------------------------------------- cursor
    def gesture_config(self, w) -> GestureConfig:
        s = w.space
        box = Box(a=(float(s.u_min), float(s.u_max)), b=(self.wp.PRESS_FLOOR, self.wp.LIFT_CEILING))
        # The foam settles in well under a second, so long holds can be short and
        # rare; most of the time goes to strokes along the bar, pressed or not.
        weights = {"reach": 0.36, "flick": 0.12, "hold": 0.18, "long_hold": 0.06, "osc": 0.18, "coast": 0.10}
        return GestureConfig(
            box=box, weights=weights, long_hold=(2.0, 5.0), b_density="contact", contact_band=(self.wp.PRESS_FLOOR, 0.001)
        )

    def hover_ab(self, w) -> np.ndarray:
        return np.array([0.0, float(self.wp.HOVER)])

    def to_world(self, w, ab: np.ndarray) -> np.ndarray:
        return w.space.at(float(ab[0]), float(ab[1]), 0.0)

    def drag_affine(self, static: dict) -> tuple[list, list, list]:
        """``space.at(a, b)``: along the bar from the seat, and straight up."""
        return static["seat"], static["along"], [0.0, 0.0, 1.0]

    def attract_program(self, w, hz: float = 60.0) -> np.ndarray:
        """The booth's autopilot (``Wipe.attract``): press, stroke out, back, out, lift,
        home -- at 60 mm/s -- as (a, b) cursor samples, with a hover second at each end."""
        seat, along = w.space.seat, w.space.along
        pts = [w.cursor.copy()] + [np.asarray(p, dtype=np.float64) for p in w.attract()]
        ab = np.array([[float((p - seat) @ along), float(p[2] - seat[2])] for p in pts])
        hold = np.repeat(self.hover_ab(w)[None, :], int(hz), axis=0)
        return np.concatenate([hold, ab, hold])

    def pick_attract(self, takes) -> list:
        """The autopilot take that cleans the stripe (>= 0.95 of cells under 0.1) and
        ends closest to where it started, so the idle loop's seam is invisible."""
        from rsw.record import load as load_take
        import json as _json

        best, best_key = None, None
        for t in takes:
            meta = _json.loads((t / "meta.json").read_text())
            if meta.get("gate"):
                continue
            d = load_take(t)
            inten = np.ones(d["ink_rate"].shape[1])
            for r in d["ink_rate"]:
                inten = np.maximum(0.0, inten - r / 60.0)
            cleaned = float(np.mean(inten <= 0.1))
            seam = float(np.linalg.norm(d["tcp"][-1, :3] - d["tcp"][0, :3])) + float(
                np.abs(d["deform"][-1] - d["deform"][0]).max())
            key = (cleaned < 0.95, seam)
            print(f"[attract] {t.parent.name}/{t.name}: cleaned {cleaned:.3f}, seam {seam * 1e3:.1f} mm")
            if best_key is None or key < best_key:
                best, best_key = t, key
        return [best] if best is not None else []

    # ------------------------------------------------------------- stepping
    def start_take(self, sim, w) -> None:
        w.new_stripe()
        self.ink.reset()
        self.lead_hist = []

    def step(self, sim, w, world: np.ndarray) -> None:
        w.cursor = np.asarray(world, dtype=np.float64)
        w.drive()
        sim.step_teleop()

    def observe(self, sim, w) -> dict:
        state = sim.state
        pos, quat = w.ik.ee_pose(state)
        sp, sq = w.law.setpoint
        pq = state.particle_q.numpy()[self.sponge]
        rate = self.ink.rates()
        self.ink.integrate(rate)
        taint = w.taint
        return {
            "t": float(sim.sim_time),
            "tcp": np.concatenate([np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)]),
            "setpoint": np.concatenate([sp, sq]),
            "joint_q": state.joint_q.numpy().astype(np.float32),
            "body_q": state.body_q.numpy().astype(np.float32),
            "deform": pq.astype(np.float32),
            "ink_rate": rate,
            "touching": bool(taint._touching[0]),
            "press": float(taint._press_now[0]),
        }

    def gate(self, sim, w, row: dict) -> str | None:
        if not (np.isfinite(row["deform"]).all() and np.isfinite(row["joint_q"]).all()):
            return "non-finite state"
        if float(np.linalg.norm(row["deform"].mean(axis=0) - row["tcp"][:3])) > 0.09:
            return "sponge left the hand"
        lead = float(np.linalg.norm(row["tcp"][:3] - row["setpoint"][:3]))
        self.lead_hist.append(lead)
        if len(self.lead_hist) >= 30 and min(self.lead_hist[-30:]) >= 0.95 * float(w.law.pos_lead):
            return "arm blocked (lead saturated 0.5 s)"
        return None

    def static_meta(self, sim, w) -> dict:
        s = w.space
        law = w.law
        return {
            "demo": ID,
            "control_hz": float(sim.request.physics.control_hz),
            "seat": s.seat.tolist(),
            "along": s.along.tolist(),
            "cross": s.cross.tolist(),
            "box": {"a": [float(s.u_min), float(s.u_max)], "b": [self.wp.PRESS_FLOOR, self.wp.LIFT_CEILING]},
            "hover_ab": self.hover_ab(w).tolist(),
            "hand_quat": w.quat.tolist(),
            "osc": {
                "pos_speed": float(law.pos_speed),
                "pos_lead": float(law.pos_lead),
                "rot_speed": float(law.rot_speed),
                "rot_lead": float(law.rot_lead),
                "control_dt": float(law.control_dt),
            },
            "sponge_particles": self.sponge.tolist(),
            "ink": {
                "cells_ut": self.ink.cells.tolist(),
                "reach2": self.ink.reach2,
                "radius": self.ink.radius,
            },
            "pre_roll_steps": int(self.pre_roll_steps),
        }

    # -------------------------------------------------------------- render
    def hero(self, static: dict) -> dict:
        """The website camera, chosen on ``rsw camera`` contact sheets (2026-09-30):
        from just in front of the slat wall (x 0.93; the wall stands at 0.957),
        facing the robot, so the sponge on the chrome bar fills the frame and a
        stroke reads left to right (+y is image right). The robot-side views all
        put an upper arm between the lens and the bar."""
        seat = np.asarray(static["seat"])
        along = np.asarray(static["along"])
        mid = seat + along * 0.5 * (static["box"]["a"][0] + static["box"]["a"][1])
        return {"look_at": (mid + np.array([0.0, 0.0, 0.02])).tolist(), "eye_offset": [0.33, -0.035, 0.34], "fov_y": 46.0}

    def ink_live(self, sim, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(shape indices, display colours) of the decals for an ink field."""
        taint = sim.scene.by_name["taint"]
        spec = taint.spec
        start = int(taint._decal_start(0))
        n = len(intensity)
        base = np.asarray(spec.taint_color, dtype=np.float32)
        clear = np.asarray(spec.clear_color, dtype=np.float32)
        i = np.asarray(intensity, dtype=np.float32)[:, None]
        return np.arange(start, start + n), base * i + clear * (1.0 - i)

    def parity(self) -> float:
        return self.ink.parity()
