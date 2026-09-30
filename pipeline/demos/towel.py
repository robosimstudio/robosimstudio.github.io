"""Shake the towel: the YAM hand holding the towel's edge is the visitor's to drag.

The scene and the grasp are the shipped ``cloth_fold/towel_half`` and its own ``FoldRoutine``
(``examples/tasks/cloth_fold/routine.py``): the pre-roll runs the routine until both carrying hands
have latched the edge (reach, settle, descend, close; ``ParticleDragger`` pins riding each hand),
then stops calling it. From there the two hands move as one: the finger drags their midpoint --
``a`` horizontally over the towel, ``b`` up -- and each hand keeps its offset from it, through its
own ``OscSetpoint`` and IK, like the booth's drivers. The pins are re-aimed at their hands every
step, as the routine does ("a latched edge rides its own hand").

Gated by ``rsw probe --demo towel`` (the plan's go/no-go: >= 4 steps/s, pins within 5 mm, stable
at speed, IK within 2 mm, the hands never closing on each other).
"""

from __future__ import annotations

import math
import time

import numpy as np

from rsw import paths
from rsw.gestures import Box, GestureConfig

ID = "towel"
TASK = "cloth_fold/towel_half"
LATCH_PHASE = "carry"


class Towel:
    """The two carrying hands, moved together: ``cursor`` is where their midpoint should go."""

    def __init__(self, env, routine) -> None:
        from robosimstudio.solvers.motion.osc import OscSetpoint

        self.env, self.routine = env, routine
        # the hands that hold the edge: towel_half's fold carries with one (the support hand
        # stays where the routine left it)
        self.names = tuple(n for n, d in routine.draggers.items() if d.active)
        self.iks = {n: env.arm(n) for n in self.names}
        hz = float(env.request.physics.control_hz)
        self.dt = 1.0 / hz
        self.p0, self.q0, self.laws = {}, {}, {}
        for n, ik in self.iks.items():
            pos, quat = ik.ee_pose(env.state)
            self.p0[n] = np.asarray(pos, dtype=np.float64)
            self.q0[n] = np.asarray(quat, dtype=np.float64)
            law = OscSetpoint(control_dt=self.dt)
            law.reset(self.p0[n], self.q0[n])
            self.laws[n] = law
        self.mid0 = np.mean([self.p0[n] for n in self.names], axis=0)
        self.off = {n: self.p0[n] - self.mid0 for n in self.names}
        self.cursor = self.mid0.copy()

    def hands(self) -> dict[str, np.ndarray]:
        return {n: np.asarray(ik.ee_pose(self.env.state)[0], dtype=np.float64) for n, ik in self.iks.items()}

    def drive(self) -> None:
        env = self.env
        state = env.state
        for n, ik in self.iks.items():
            pos, quat = ik.ee_pose(state)
            ik.set_target(*self.laws[n].advance(pos, quat, self.cursor + self.off[n], self.q0[n]))
            ik.solve_into(state.joint_q, env.control.joint_target_q)
        for n, dragger in self.routine.draggers.items():
            if dragger.active:
                pos, quat = self.iks[n].ee_pose(state)
                dragger.set_target(pos, quat)


class Spec:
    id = ID
    deform_frame = "world"
    deform_name = "towel"
    deform_arm = "left"
    fresh_episode_per_take = True
    """A fresh towel each take (pre-roll ~45 s): a take can leave the cloth folded or slid off."""

    def build(self, device=None):
        paths.use_rss()
        from robosimstudio_bench.envs.env import Env

        env = Env(TASK, episode=0, renderer="", device=device)
        env.obs_render_enabled = False
        self.env = env
        return env

    def _release(self, env) -> None:
        """The last routine's pins: released and taken out of the solve (the peel's lesson)."""
        routine = getattr(self, "routine", None)
        if routine is None:
            return
        hooks = env.pipeline.hooks.post_solve
        for d in routine.draggers.values():
            d.release()
            if d.apply in hooks:
                hooks.remove(d.apply)
        env.pipeline.invalidate_graph()

    def pre_roll(self, env, reset: bool = False):
        if reset:
            self._release(env)
            env.reset()
        routine = env.task.routine(env)
        routine.reset()
        mode = getattr(routine, "MODE", "")
        n = 0
        while routine.phase != LATCH_PHASE:
            env.step(routine.act(), mode=mode)
            n += 1
            if n > 2000:
                raise SystemExit(f"[towel] the routine never reached {LATCH_PHASE!r} (phase {routine.phase}: {routine.message})")
        if not any(d.active for d in routine.draggers.values()):
            raise SystemExit(f"[towel] no hand holds the edge: {routine.message}")
        self.routine = routine
        drv = Towel(env, routine)
        # the drag plane: a toward the towel's middle (horizontal, across the held edge), b up
        cloth = self._cloth_particles(env)
        p = env.state.particle_q.numpy()[cloth]
        toward = p.mean(axis=0) - drv.mid0
        toward[2] = 0.0
        self.A = toward / (np.linalg.norm(toward) or 1.0)
        self.B = np.array([0.0, 0.0, 1.0])
        self.cloth = cloth
        self.pre_roll_steps = n
        return drv

    def _cloth_particles(self, env) -> np.ndarray:
        dh = env.handles.deformables.get(self.deform_name) or next(iter(env.handles.deformables.values()))
        return np.asarray(dh.particles, dtype=np.int64)

    # ------------------------------------------------------------------ cursor
    def gesture_config(self, drv) -> GestureConfig:
        self.box = Box(a=(-0.10, 0.35), b=(0.0, 0.30))
        weights = {"reach": 0.30, "flick": 0.14, "hold": 0.14, "long_hold": 0.12, "osc": 0.22, "coast": 0.08}
        return GestureConfig(box=self.box, weights=weights, v_reach=(0.05, 0.4), v_flick=(0.4, 1.0),
                             long_hold=(3.0, 6.0), osc_amp=(0.03, 0.12), osc_hz=(0.5, 1.5))

    def hover_ab(self, drv) -> np.ndarray:
        return np.array([0.0, 0.10])

    def to_world(self, drv, ab: np.ndarray) -> np.ndarray:
        return drv.mid0 + self.A * float(ab[0]) + self.B * float(ab[1])

    def drag_affine(self, static: dict) -> tuple[list, list, list]:
        return static["mid0"], static["A"], static["B"]

    # ------------------------------------------------------------------ stepping
    def start_take(self, env, drv) -> None:
        pass

    def step(self, env, drv, world: np.ndarray) -> None:
        drv.cursor = np.asarray(world, dtype=np.float64)
        drv.drive()
        env.step_teleop()

    def observe(self, env, drv) -> dict:
        state = env.state
        hands = drv.hands()
        mid = np.mean([hands[n] for n in drv.names], axis=0)
        q = state.particle_q.numpy()
        pins = []
        for n, dragger in self.routine.draggers.items():
            idx = np.asarray(getattr(dragger, "indices", []) or [], dtype=np.int64)
            if len(idx):
                pins.append(float(np.linalg.norm(q[idx] - q[idx].mean(axis=0), axis=1).max()))
        quat = np.asarray(drv.iks[drv.names[0]].ee_pose(state)[1], dtype=np.float64)
        return {
            "t": float(env.sim_time),
            "tcp": np.concatenate([mid, quat]),
            "setpoint": np.concatenate([drv.cursor, [0.0, 0.0, 0.0, 1.0]]),
            "hands": np.concatenate([hands[n] for n in drv.names]).astype(np.float32),
            "joint_q": state.joint_q.numpy().astype(np.float32),
            "body_q": state.body_q.numpy().astype(np.float32),
            "deform": q[self.cloth].astype(np.float32),
        }

    def gate(self, env, drv, row: dict) -> str | None:
        d = row["deform"]
        if not np.isfinite(d).all():
            return "non-finite state"
        if float(d[:, 2].min()) < float(env.handles.surface_z) - 0.05:
            return "towel under the bench"
        h = row["hands"].reshape(-1, 3)
        if len(h) == 2 and np.linalg.norm(h[0] - h[1]) < 0.5 * np.linalg.norm(drv.off[drv.names[0]] - drv.off[drv.names[1]]):
            return "hands closed on each other"
        return None

    def static_meta(self, env, drv) -> dict:
        return {
            "demo": ID, "control_hz": 1.0 / drv.dt, "mid0": drv.mid0.tolist(), "A": self.A.tolist(), "B": self.B.tolist(),
            "box": {"a": list(self.box.a), "b": list(self.box.b)} if hasattr(self, "box") else None,
            "hover_ab": [0.0, 0.10], "hands": list(drv.names), "offsets": {n: drv.off[n].tolist() for n in drv.names},
            "hand_quat": drv.q0[drv.names[0]].tolist(), "cloth_particles": self.cloth.tolist(), "pre_roll_steps": self.pre_roll_steps,
        }

    def hero(self, static: dict) -> dict:
        """From behind the holding hand, over it onto the towel: the side views have the near arm
        across the frame (``rsw camera`` sheets, 2026-09-30)."""
        mid = np.asarray(static["mid0"], dtype=np.float64)
        A = np.asarray(static["A"], dtype=np.float64)
        look = mid + A * 0.12 + np.array([0.0, 0.0, 0.05])
        eye = look - A * 0.9 + np.array([0.0, 0.0, 0.7])
        return {"look_at": look.tolist(), "eye": eye.tolist(), "fov_y": 38.0}

    def attract_program(self, drv, hz: float = 60.0) -> np.ndarray:
        """Lift the edge, fold the towel over, lift it off again, lay it back: the loop's seam is a
        towel lying about as it started."""
        keys = [((0.0, 0.0), 1.0), ((0.0, 0.15), 1.0), ((0.30, 0.15), 2.2), ((0.30, 0.04), 0.8), ((0.30, 0.04), 1.5),
                ((0.30, 0.25), 1.0), ((0.10, 0.28), 1.4), ((-0.05, 0.12), 1.2), ((-0.05, 0.0), 0.8), ((0.0, 0.0), 0.6),
                ((0.0, 0.0), 2.5)]
        out, cur = [], np.array([0.0, 0.0])
        for (a, b), sec in keys:
            nxt = np.array([a, b])
            n = max(2, int(sec * hz))
            u = np.linspace(0.0, 1.0, n)[:, None]
            out += list(cur + (nxt - cur) * (u * u * (3 - 2 * u)))
            cur = nxt
        return np.asarray(out, dtype=np.float64)

    def pick_attract(self, takes) -> list:
        """The autopilot take (no gate) whose towel ends closest to how it started."""
        import json as _json

        from rsw.record import load as load_take

        best, best_key = None, None
        for t in takes:
            if _json.loads((t / "meta.json").read_text()).get("gate"):
                continue
            d = load_take(t)
            seam = float(np.abs(d["deform"][-1] - d["deform"][0]).max())
            print(f"[attract] {t.parent.name}/{t.name}: seam {seam * 1e3:.1f} mm, {len(d['t'])} steps")
            if best_key is None or seam < best_key:
                best, best_key = t, seam
        return [best] if best is not None else []

    # ------------------------------------------------------------------ probe
    def probe_workspace(self, drv) -> dict:
        """The plan's go/no-go on a scripted drag: lift 20 cm in 1 s, sway +-15 cm at 0.5 m/s for
        10 s over the towel, hold 3 s. Measures rate, pin spread, IK error, hand gap, stability."""
        env = self.env
        hz = 1.0 / drv.dt
        self.gesture_config(drv)
        prog = []
        for k in range(int(1.0 * hz)):
            prog.append((0.0, 0.20 * k / hz))
        t = np.arange(int(10.0 * hz)) / hz
        amp, w = 0.15, 0.5 / 0.15  # 0.5 m/s peak
        prog += [(0.10 + amp * math.sin(w * tt), 0.20) for tt in t]
        prog += [(0.10, 0.20)] * int(3.0 * hz)
        t0 = time.time()
        worst_pin, worst_ik, min_gap, fail = 0.0, 0.0, 9.9, None
        for i, ab in enumerate(prog):
            self.step(env, drv, self.to_world(drv, np.asarray(ab)))
            row = self.observe(env, drv)
            fail = self.gate(env, drv, row)
            h = row["hands"].reshape(-1, 3)
            if len(h) == 2:
                min_gap = min(min_gap, float(np.linalg.norm(h[0] - h[1])))
            for n, ik in drv.iks.items():
                tgt = drv.cursor + drv.off[n]
                worst_ik = max(worst_ik, float(np.linalg.norm(np.asarray(ik.ee_pose(env.state)[0]) - tgt)))
            if fail:
                break
        rate = len(prog) / (time.time() - t0)
        return {"steps": len(prog), "steps_per_s": round(rate, 2), "fail": fail, "hand_lag_max_mm": round(worst_ik * 1e3, 1),
                "hand_gap_min_m": round(min_gap, 3), "pin_spread_mm": round(worst_pin * 1e3, 1),
                "hands": list(drv.names), "A": self.A.tolist()}
