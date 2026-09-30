"""Peel the film: rgbs/panel's VBD film off a phone, driven by virtual visitors.

The booth demo's peel (``~/rss_ws/phone_demo/peel_scene.py``): the task's own
``RoutineFilmPeel`` plays up to ``phase == "peel"`` (the left hand pinches the
phone's corner, the right hand latches the film's tab), then a cursor takes the
right hand along the routine's peel arc: ``a`` = run down the face (0..0.36 m),
``b`` = sideways offset (+-20 mm). ``Peel.drive`` holds the hand to the routine's
50 mm/s. A new film is a new episode, so every take is: reset, pre-roll, peel.
"""

from __future__ import annotations

import numpy as np

from rsw import paths
from rsw.gestures import Box, GestureConfig

ID = "peel"


class Spec:
    id = ID
    deform_frame = "world"
    deform_name = "film"
    deform_arm = "right"
    fresh_episode_per_take = True

    def build(self, device=None):
        paths.use_rss(paths.PHONE)
        import peel_scene as pl  # noqa: F401  (phone_demo)
        from robosimstudio_bench.envs.env import Env

        self.pl = pl
        env = Env(pl.TASK, episode=0, renderer="", device=device)
        env.obs_render_enabled = False
        self.env = env
        self.dt = 1.0 / float(env.request.physics.control_hz)
        return env

    def _release(self, env, routine) -> None:
        if routine is not None and routine.dragger is not None:
            routine.dragger.release()
            hooks = env.pipeline.hooks.post_solve
            if routine.dragger.apply in hooks:
                hooks.remove(routine.dragger.apply)
            env.pipeline.invalidate_graph()

    def pre_roll(self, env, reset: bool = False):
        pl = self.pl
        if reset:
            self._release(env, getattr(self, "routine", None))
            env.reset()
        routine = None
        n = 0
        for routine in pl.pre_roll(env):
            n += 1
        self.routine = routine
        peel = pl.Peel(env, routine)
        self.film = routine.film
        self.film_particles = np.asarray(self.film.particles, dtype=np.int64)
        bonds = self.film.bonds
        self.n_bonds = int(bonds.per_world or len(bonds.idx_host))
        self.pre_roll_steps = n
        self.lead_hist: list[float] = []
        # the film's triangle edges and their lengths now (attached, at rest): a tear is
        # an edge stretched well past its rest length
        dh = env.handles.deformables[self.deform_name] if self.deform_name in env.handles.deformables else None
        if dh is None and env.handles.deformables:
            dh = next(iter(env.handles.deformables.values()))
        tri = np.asarray(dh.triangles, dtype=np.int64).reshape(-1, 3)
        e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        self.edges = np.unique(np.sort(e, axis=1), axis=0)
        p0 = env.state.particle_q.numpy()[self.film_particles]
        self.rest = np.linalg.norm(p0[self.edges[:, 0]] - p0[self.edges[:, 1]], axis=1)
        return peel

    # -------------------------------------------------------------- cursor
    def gesture_config(self, peel) -> GestureConfig:
        box = Box(a=(0.0, float(self.pl.RUN_MAX)), b=(-float(self.pl.ACROSS_HALF), float(self.pl.ACROSS_HALF)))
        # A visitor pulls down, drifts sideways, pauses (the film is quasi-static,
        # so short holds settle it), and sometimes pushes back up.
        weights = {"reach": 0.40, "flick": 0.10, "hold": 0.22, "long_hold": 0.06, "osc": 0.14, "coast": 0.08}
        return GestureConfig(box=box, weights=weights, v_reach=(0.01, 0.08), v_flick=(0.2, 0.8),
                             long_hold=(2.0, 5.0), osc_amp=(0.003, 0.02))

    def hover_ab(self, peel) -> np.ndarray:
        return np.array([0.0, 0.0])

    def attract_program(self, peel, hz: float = 60.0) -> np.ndarray:
        """The booth's autopilot (``Peel.attract``): the whole film at the routine's own speed,
        straight down the arc, with a second's rest before and two after. The idle loop's seam
        is a fresh film, as at the booth (``new_stripe`` lays a new one)."""
        step = float(peel.max_step)
        run_max = float(self.pl.RUN_MAX)
        runs = np.minimum(np.arange(0.0, run_max + step, step), run_max)
        ab = np.stack([runs, np.zeros_like(runs)], axis=1)
        return np.concatenate([np.repeat(ab[:1], int(hz), axis=0), ab, np.repeat(ab[-1:], int(2 * hz), axis=0)])

    def pick_attract(self, takes) -> list:
        """The autopilot take that peels furthest (all bonds broken, ideally) without a gate."""
        import json as _json

        from rsw.record import load as load_take

        best, best_key = None, None
        for t in takes:
            meta = _json.loads((t / "meta.json").read_text())
            if meta.get("gate"):
                continue
            d = load_take(t)
            act = d["bond_active"]
            broken = float(1.0 - act[-1].mean()) if act is not None and len(act) else 0.0
            key = (broken < 0.99, -broken)
            print(f"[attract] {t.parent.name}/{t.name}: broken {broken:.3f}, {len(d['t'])} steps")
            if best_key is None or key < best_key:
                best, best_key = t, key
        return [best] if best is not None else []

    def to_world(self, peel, ab: np.ndarray) -> np.ndarray:
        return peel.point(float(ab[0]), float(ab[1]))

    # ------------------------------------------------------------- stepping
    def start_take(self, env, peel) -> None:
        self.lead_hist = []

    def step(self, env, peel, world: np.ndarray) -> None:
        peel.cursor = np.asarray(world, dtype=np.float64)
        peel.drive()
        env.step_teleop()

    def observe(self, env, peel) -> dict:
        state = env.state
        pos, quat = peel.ik.ee_pose(state)
        sp, sq = peel.law.setpoint
        bonds = self.film.bonds
        active = np.asarray(bonds.active_rows(None), dtype=bool)[: self.n_bonds]
        rows = np.asarray(bonds.read(0).row, dtype=np.float32)
        front, _ = self.film.peel_front()
        return {
            "t": float(env.sim_time),
            "tcp": np.concatenate([np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)]),
            "setpoint": np.concatenate([sp, sq]),
            "target": np.asarray(peel.target, dtype=np.float64),
            "joint_q": state.joint_q.numpy().astype(np.float32),
            "body_q": state.body_q.numpy().astype(np.float32),
            "deform": state.particle_q.numpy()[self.film_particles].astype(np.float32),
            "bond_active": active,
            "rows": rows,
            "front": float(front),
        }

    def gate(self, env, peel, row: dict) -> str | None:
        if not (np.isfinite(row["deform"]).all() and np.isfinite(row["joint_q"]).all()):
            return "non-finite state"
        # A diagonal peel breaks bonds down one side before any full row (so "broken
        # ahead of the front" is a normal peel, measured 2026-09-30); a tear stretches.
        p = row["deform"]
        stretch = np.linalg.norm(p[self.edges[:, 0]] - p[self.edges[:, 1]], axis=1) / np.maximum(self.rest, 1e-6)
        if float(stretch.max()) > 1.3:
            return "film overstretched (an edge past 130 % of rest)"
        lead = float(np.linalg.norm(row["tcp"][:3] - row["setpoint"][:3]))
        self.lead_hist.append(lead)
        if len(self.lead_hist) >= 30 and min(self.lead_hist[-30:]) >= 0.95 * float(peel.law.pos_lead):
            return "arm blocked (lead saturated 0.5 s)"
        return None

    def hero(self, static: dict) -> dict:
        """The booth's peel view (``wipe_phone.peel_view``): straight onto the tilted
        phone face from over the arms' shoulders, aimed a little down the face; fov 34
        so the phone fills the frame. Chosen on ``rsw camera`` sheets (2026-09-30): every
        side or three-quarter view (+-y, 0.45-0.55 m out) sat inside an arm's links --
        the two YAMs flank the phone -- so the lift can only be read face-on."""
        tab, u, n = (np.asarray(static[k], dtype=np.float64) for k in ("tab", "u", "n"))
        look = tab - u * (0.0735 + 0.06)
        eye = look + 1.1 * (n * 0.40 + np.array([0.0, 0.0, 0.05]))
        return {"look_at": look.tolist(), "eye": eye.tolist(), "fov_y": 34.0}

    def static_meta(self, env, peel) -> dict:
        law = peel.law
        return {
            "demo": ID,
            "control_hz": float(env.request.physics.control_hz),
            "tab": peel.tab.tolist(),
            "u": peel.u.tolist(),
            "n": peel.n.tolist(),
            "y": peel.y.tolist(),
            "rise": float(peel.rise),
            "arc_lift": float(self.pl._rp.ARC_LIFT),
            "box": {"a": [0.0, float(self.pl.RUN_MAX)], "b": [-float(self.pl.ACROSS_HALF), float(self.pl.ACROSS_HALF)]},
            "hover_ab": [0.0, 0.0],
            "hand_quat": peel.quat.tolist(),
            "osc": {
                "pos_speed": float(law.pos_speed), "pos_lead": float(law.pos_lead),
                "rot_speed": float(law.rot_speed), "rot_lead": float(law.rot_lead),
                "control_dt": float(law.control_dt), "pre_limit": float(peel.routine.speed),
            },
            "film_particles": self.film_particles.tolist(),
            "n_bonds": self.n_bonds,
            "pre_roll_steps": int(self.pre_roll_steps),
        }
