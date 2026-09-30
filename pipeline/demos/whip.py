"""Whip the cable: the ceiling UR5e twin of IMG_8652.MOV, driven by virtual visitors.

Everything physical is ``~/rss_ws/cable_wiggle``'s: the scene (``scene.py``), the
hanging start with the cable in the jaws, the kinematic pin (``pin.py``), air drag
(``drag.py``), the 8000/300 servos commanded ``q + tau * dq`` (``run.py``). What a
visitor moves is the **held point** in the cable's plane (x, z at y = 0); the hand
leans with its own horizontal speed like a wrist cracking a whip, 27 deg (the clip's
mean) +-10 deg.

A take is planned before it is stepped, the way ``run.py`` plans the clip's drive:
cursor -> the servo law (open loop, cable gains) -> setpoint path -> warm-started IK
-> joint trajectory, whose gradient gives the servo lead.
"""

from __future__ import annotations

import math

import numpy as np

from rsw import paths
from rsw.gestures import Box, GestureConfig

ID = "whip"
LEAN0 = math.radians(27.0)
LEAN_GAIN = math.radians(10.0)
TAU = 0.0375
KP = 1296.0
"""Cartesian stiffness for the cable's servo law: cruise = sqrt(kp)/2 * 0.05 = 0.9 m/s,
the clip's own drive speed (the default 150 gives 0.31 m/s, too slow to whip)."""


def _compat(SC) -> None:
    """cable_wiggle (2026-09-14) predates the repo's per-world rod API: the base class
    now calls ``_pose_anchor(pipeline, world=...)``. One world here, so the argument is
    always 0; wrap rather than edit the twin."""
    import inspect

    cls = SC.EntityHangingRod
    orig = cls.__dict__.get("_pose_anchor")
    if orig is not None and "world" not in inspect.signature(orig).parameters:
        def _pose_anchor(self, pipeline, *, world: int = 0, _orig=orig):
            del world
            return _orig(self, pipeline)

        cls._pose_anchor = _pose_anchor


class Spec:
    id = ID
    # The cable is pinned to the jaws, so the gripper's own tool frame (FK in the
    # browser) carries it; stored as capsule centres, drawn as a tube.
    deform_frame = "tcp"
    deform_arm = "right"
    deform_name = "cable"
    deform_kind = "tube"
    tube_color = (0.015, 0.015, 0.017)
    plans_takes = True

    # ---------------------------------------------------------------- build
    def build(self, device=None):
        paths.use_rss(paths.CABLE)
        import drag as DR  # noqa: F401
        import pin  # noqa: F401
        import run as RUN  # noqa: F401
        import scene as SC  # noqa: F401
        from robosimstudio.specs import ActuationSpec

        self.SC, self.RUN, self.pin, self.DR = SC, RUN, pin, DR
        _compat(SC)
        self.params = SC.CableParams()
        self.tube_radius = float(self.params.radius)
        self.gap = 2 * self.params.radius + 0.005
        grip_q = 0.8 * (1.0 - self.gap / 0.085)
        # The held point's box, from ``rsw probe --demo whip`` (2026-09-30): IK holds to
        # ~2 (mm + deg/2) over x -0.06..0.36, z 1.30..1.65 except the corner (0.36, 1.30);
        # z >= 1.40 keeps the 1 m cable's end off the whiteboard's tray.
        self.box = Box(a=(-0.05, 0.33), b=(1.40, 1.62))
        P0 = np.array([0.155, 0.0, 1.485])
        from dataclasses import replace

        # The whiteboard is render-only scenery 10 cm behind the cable (the clip's).
        # A hanging chain driven in its plane can whirl out of it, and at 10 cm the
        # free end passed through the board within a second (smoke take, 2026-09-30);
        # 30 cm keeps a whirl in front of it.
        self.L = replace(SC.LAYOUT, board_y=0.30)
        scene, cable = SC.build_scene(self.params, self.L, arm_gains=ActuationSpec(target_ke=8000.0, target_kd=300.0),
                                      hang=(P0, LEAN0), grip=grip_q, physics=SC.PHYSICS)
        self.cable = cable
        sim = scene.build(control_mode=RUN.MODE, device=device)
        self.dt = 1.0 / 60.0
        return sim

    def pre_roll(self, sim):
        RUN, pin, DR = self.RUN, self.pin, self.DR
        arm = RUN.Arm(sim, 12)
        ent = self.cable.entity
        bodies = np.asarray(self.cable.bodies)
        k_quad = DR.quadratic_per_capsule(2 * self.params.radius, self.params.length / self.params.segments)
        drag = DR.Drag(sim, bodies, quad=k_quad, linear=0.08)
        arm.grip = float(arm.gripper.command_for_width(self.gap))
        pin.attach(sim, ent)
        segs = ent._hold_seg.numpy()
        self.seg = int(segs[0])
        sim.sync()
        Tg, Tee, Ts = arm.body_T(arm.ee_body), arm.ee(), arm.body_T(self.seg)
        self.T_rel = np.linalg.inv(Tg) @ Ts
        self.T_ee_in_g = np.linalg.inv(Tg) @ Tee
        self.up_sign = 1.0 if Ts[:3, 2][2] > 0 else -1.0
        self.R_seg_now = Ts[:3, :3].copy()
        # Settle to rest before any take: 150 steps at the twin's 0.02 left the free end
        # swinging out of plane at ~0.2 m/s; 10 s at 0.08 damps it.
        for _ in range(600):
            arm.step()
        drag.set(k_quad, 0.0)
        self.arm, self.ent, self.bodies, self.drag = arm, ent, bodies, drag
        self.plan_q = None
        sim.sync()
        self.hold_P0 = np.asarray(arm.body_T(self.seg)[:3, 3], dtype=np.float64)
        self.lead_hist: list[float] = []
        self.length0 = self._length(sim)
        return arm

    # ------------------------------------------------------------- geometry
    def _R_seg(self, lean: float) -> np.ndarray:
        """The held segment's orientation for a lean [rad] right of vertical (run.py's R_seg_for)."""
        want = self.up_sign * self.RUN.Drive.axis(lean)
        return self.RUN.rot_between(self.R_seg_now[:, 2], want) @ self.R_seg_now

    def _tcp_for(self, P: np.ndarray, lean: float) -> tuple[np.ndarray, np.ndarray]:
        Tg_des = self.RUN.T(P, self._R_seg(lean)) @ np.linalg.inv(self.T_rel)
        Tee = Tg_des @ self.T_ee_in_g
        return Tee[:3, 3], Tee[:3, :3]

    def _length(self, sim) -> float:
        bq = sim.pipeline.state_0.body_q.numpy()[self.bodies, :3]
        return float(np.linalg.norm(np.diff(bq, axis=0), axis=1).sum())

    # --------------------------------------------------------------- cursor
    def gesture_config(self, arm) -> GestureConfig:
        # Whipping is fast strokes, oscillations and long holds (the cable swings
        # on for seconds after the hand stops, and the still set needs it settled).
        weights = {"reach": 0.26, "flick": 0.16, "hold": 0.12, "long_hold": 0.16, "osc": 0.22, "coast": 0.08}
        return GestureConfig(box=self.box, weights=weights, v_reach=(0.05, 0.6), v_flick=(0.6, 2.0),
                             long_hold=(6.0, 12.0), osc_amp=(0.02, 0.12), osc_hz=(0.6, 2.5))

    def hover_ab(self, arm) -> np.ndarray:
        return np.array([0.5 * (self.box.a[0] + self.box.a[1]), 0.5 * (self.box.b[0] + self.box.b[1])])

    def to_world(self, arm, ab: np.ndarray) -> np.ndarray:
        return np.array([float(ab[0]), 0.0, float(ab[1])])

    # --------------------------------------------------------------- planning
    def plan(self, arm, cursor_world: np.ndarray) -> dict:
        """Open-loop servo law + IK over the whole take; returns per-step targets."""
        from robosimstudio.solvers.motion.osc import OscGains, OscSetpoint

        law = OscSetpoint(control_dt=self.dt, gains=OscGains(kp=KP))
        p = self.hold_P0.copy()
        law.reset(p, np.array([0.0, 0.0, 0.0, 1.0]))
        n = len(cursor_world)
        P = np.zeros((n, 3))
        for k in range(n):
            sp, _ = law.advance(p, np.array([0.0, 0.0, 0.0, 1.0]), cursor_world[k], np.array([0.0, 0.0, 0.0, 1.0]))
            p = sp  # open loop: the stiff servo tracks its setpoint to ~1.4 mm
            P[k] = sp
        vx = np.gradient(P[:, 0], self.dt)
        vx = np.convolve(vx, np.ones(9) / 9, mode="same")
        lean = LEAN0 + LEAN_GAIN * np.clip(vx / 0.5, -1.0, 1.0)
        coords = arm.ik.arm_coords
        q_des = np.zeros((n, len(coords)))
        worst = (0.0, 0.0)
        for k in range(n):
            pe, Re = self._tcp_for(P[k], lean[k])
            ep, ea = arm.solve(pe, Re, roll=False)
            worst = (max(worst[0], ep), max(worst[1], ea))
            q_des[k] = arm.q[coords]
        dq = np.gradient(q_des, self.dt, axis=0)
        self.plan_q = q_des + TAU * dq
        self.plan_P, self.plan_lean, self.k = P, lean, 0
        self.pos_speed, self.pos_lead = float(law.pos_speed), float(law.pos_lead)
        return {"ik_worst_mm": worst[0] * 1e3, "ik_worst_deg": math.degrees(worst[1])}

    # ------------------------------------------------------------- stepping
    def start_take(self, sim, arm) -> None:
        self.lead_hist = []

    def step(self, sim, arm, world: np.ndarray) -> None:
        coords = arm.ik.arm_coords
        arm.q[coords] = self.plan_q[min(self.k, len(self.plan_q) - 1)]
        arm.step()
        self.k += 1

    def observe(self, sim, arm) -> dict:
        state = sim.pipeline.state_0
        sim.sync()
        bq = state.body_q.numpy()
        pos, quat = arm.ik.ee_pose(sim.state)
        k = min(self.k - 1, len(self.plan_P) - 1)
        held = bq[self.seg].astype(np.float64)
        caps = bq[self.bodies, :3].astype(np.float32)
        return {
            "t": float(sim.sim_time),
            "tcp": np.concatenate([np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)]),
            "setpoint": np.concatenate([self.plan_P[k], [0.0, 0.0, 0.0, 1.0]]),
            "held": held,
            "lean": float(self.plan_lean[k]),
            "joint_q": state.joint_q.numpy().astype(np.float32),
            "body_q": bq.astype(np.float32),
            "deform": caps,
        }

    def gate(self, sim, arm, row: dict) -> str | None:
        caps = row["deform"]
        if not (np.isfinite(caps).all() and np.isfinite(row["joint_q"]).all()):
            return "non-finite state"
        L = float(np.linalg.norm(np.diff(caps, axis=0), axis=1).sum())
        if L > 1.03 * self.length0:
            return "cable exploded (stretch > 3 %)"
        k = min(self.k - 1, len(self.plan_P) - 1)
        # The pin is kinematic, so the held point IS the gripper's: this measures
        # the arm's lag. A dead stop from 0.9 m/s overshoots ~20 mm (smoke take,
        # 2026-09-30), which is physical; 60 mm would be a servo that gave up.
        if float(np.linalg.norm(row["held"][:3] - self.plan_P[k])) > 0.06:
            return "arm lost its plan (held point 60 mm off)"
        if float(caps[:, 1].max()) > self.L.board_y - self.params.radius:
            return "cable in the whiteboard"
        return None

    def static_meta(self, sim, arm) -> dict:
        return {
            "demo": ID,
            "control_hz": 60.0,
            "hold_P0": self.hold_P0.tolist(),
            "box": {"a": list(self.box.a), "b": list(self.box.b)},
            "hover_ab": self.hover_ab(arm).tolist(),
            "hand_quat": [0.0, 0.0, 0.0, 1.0],
            "osc": {"pos_speed": getattr(self, "pos_speed", 0.9), "pos_lead": getattr(self, "pos_lead", 0.05),
                    "rot_speed": 3.06, "rot_lead": 0.5, "control_dt": self.dt, "kp": KP},
            "cable_bodies": self.bodies.tolist(),
            "held_seg": self.seg,
            "radius": float(self.params.radius),
            "length": float(self.params.length),
            "lean0": LEAN0, "lean_gain": LEAN_GAIN, "tau": TAU,
        }

    def hero(self, static: dict) -> dict:
        """Three-quarters front: the ceiling UR5e, the held point's box and the whole
        hanging cable against the whiteboard (the clip's own camera leaves the hand out)."""
        return {"look_at": [0.12, 0.0, 1.35], "eye": [0.95, -2.35, 1.45], "fov_y": 46.0}

    # --------------------------------------------------------------- probe
    def probe_workspace(self, arm, xs=None, zs=None, leans_deg=(15.0, 27.0, 38.0)) -> dict:
        """IK over an (x, z, lean) grid from the hanging pose's branch: worst errors per cell."""
        xs = np.linspace(-0.06, 0.36, 15) if xs is None else xs
        zs = np.linspace(1.30, 1.65, 12) if zs is None else zs
        coords = arm.ik.arm_coords
        q_home = arm.q.copy()
        grid = np.zeros((len(zs), len(xs)))
        for i, z in enumerate(zs):
            for j, x in enumerate(xs):
                worst = 0.0
                arm.q = q_home.copy()
                for lean in leans_deg:
                    pe, Re = self._tcp_for(np.array([x, 0.0, z]), math.radians(lean))
                    ep, ea = arm.solve(pe, Re, iterations=60, roll=False)
                    worst = max(worst, ep * 1e3 + 0.5 * math.degrees(ea))
                grid[i, j] = worst
        arm.q = q_home
        del coords
        return {"xs": xs.tolist(), "zs": zs.tolist(), "score_mm_plus_halfdeg": grid.tolist()}
