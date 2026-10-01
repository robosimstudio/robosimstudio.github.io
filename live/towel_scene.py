"""Fold the towel, from a browser: ``cloth_fold/towel_half`` on the phone demo's touch page.

Everything physical is the shipped task's own: the VBD towel, the YAM pair, and its
``FoldRoutine`` for the grasp. The routine plays until the carrying hand has latched the
towel's edge (reach, settle, descend, close; a ``ParticleDragger`` pin rides the hand), then
stops. From there a finger drags that hand: up lifts the edge, sideways carries it (along the
screen's horizontal), and the hand follows through its own ``OscSetpoint`` and IK at the arm's
speed limit. The pin is re-aimed at the hand every step, as the routine does ("a latched
edge rides its own hand").

The drag box is where the website's recorded takes found the arm actually goes (2026-09-30:
``a`` toward the towel's middle -0.10..0.28 m, up 0..0.21 m; asked for more, the hand sat at
its reach with a full lead), plus a short sideways range.
"""

from __future__ import annotations

import numpy as np

from robosimstudio.solvers import OscSetpoint
from robosimstudio_bench.envs.env import Env

TASK = "cloth_fold/towel_half"
LATCH_PHASE = "carry"
BOX_A = (-0.10, 0.28)
"""Toward the towel's middle from where the hand latched [m]: the fold's carry."""
BOX_B = (0.0, 0.21)
"""Up from the latch height [m]."""
BOX_C = (-0.06, 0.06)
"""Sideways [m]: a swing, not a walk off the bench."""
BALL_TOWARD, BALL_UP = 0.055, 0.03
"""The ball's offset from the hand [m]: toward the opening camera and up, clear of the fingers."""
ATTRACT_KEYS = (
    ((0.0, 0.0), 1.0), ((0.0, 0.15), 1.0), ((0.30, 0.15), 2.2), ((0.30, 0.04), 0.8), ((0.30, 0.04), 1.5),
    ((0.30, 0.25), 1.0), ((0.10, 0.28), 1.4), ((-0.05, 0.12), 1.2), ((-0.05, 0.0), 0.8), ((0.0, 0.0), 0.6),
    ((0.0, 0.0), 2.5),
)
"""The website's autopilot (``demos/towel.py``): lift the edge, fold it over, lift it off again,
lay it back. (a, b) keys and the seconds to each, eased."""


def build_env() -> Env:
    env = Env(TASK, episode=0, renderer="tiled", verbose=True)
    env.obs_render_enabled = False
    return env


def pre_roll(env):
    """The task's routine up to the latch. A generator of control steps; yields the routine."""
    routine = env.task.routine(env)
    routine.reset()
    mode = getattr(routine, "MODE", "")
    n = 0
    while routine.phase != LATCH_PHASE:
        env.step(routine.act(), mode=mode)
        n += 1
        yield routine
        if n > 3000:
            raise RuntimeError(f"the routine never reached {LATCH_PHASE!r} (phase {routine.phase}: {routine.message})")
    if not any(d.active for d in routine.draggers.values()):
        raise RuntimeError(f"no hand holds the towel's edge: {routine.message}")


def release(env, routine) -> None:
    """Let go of the routine's pins and take them out of the solve: a reset does not know they are
    there, and a live pin would hold the next towel's edge (the peel demo's lesson)."""
    if routine is None:
        return
    hooks = env.pipeline.hooks.post_solve
    for d in routine.draggers.values():
        d.release()
        if d.apply in hooks:
            hooks.remove(d.apply)
    env.pipeline.invalidate_graph()


class Towel:
    """The carrying hand, driven by a finger. The phone page's driver interface."""

    scene = "towel"

    def __init__(self, env, fold) -> None:
        # ``fold``, not ``routine``: the phone page's restart reads ``routine.dragger`` (the peel's
        # one pin); this towel's pins are released by the towel session itself.
        self.env, self.fold = env, fold
        self.names = tuple(n for n, d in fold.draggers.items() if d.active)
        self.iks = {n: env.arm(n) for n in self.names}
        hz = float(env.request.physics.control_hz)
        self.p0, self.q0, self.laws = {}, {}, {}
        for n, ik in self.iks.items():
            pos, quat = ik.ee_pose(env.state)
            self.p0[n] = np.asarray(pos, dtype=np.float64)
            self.q0[n] = np.asarray(quat, dtype=np.float64)
            law = OscSetpoint(control_dt=1.0 / hz)
            law.reset(self.p0[n], self.q0[n])
            self.laws[n] = law
        self.mid0 = np.mean([self.p0[n] for n in self.names], axis=0)
        self.off = {n: self.p0[n] - self.mid0 for n in self.names}
        dh = env.handles.deformables.get("towel") or next(iter(env.handles.deformables.values()))
        self.cloth = np.asarray(dh.particles, dtype=np.int64)
        q = self.particles()
        toward = q.mean(axis=0) - self.mid0
        toward[2] = 0.0
        self.A = toward / (np.linalg.norm(toward) or 1.0)
        self.B = np.array([0.0, 0.0, 1.0])
        self.C = np.cross(self.B, self.A)
        along = (q - self.mid0) @ self.A
        self.length = float(along.max() - along.min())
        """The towel's extent along the carry [m]: folding it in half carries the edge this far."""
        self.surface_z = float(env.handles.surface_z)
        self.ball = self.C * BALL_TOWARD + self.B * BALL_UP
        """Where the orange ball sits from the hand: at the tool centre the fingers hide it."""
        self.cursor = self.mid0 + self.ball
        """The ball: the hands' midpoint should go to ``cursor - ball``."""
        self.reset_request = False
        self.hz = hz

    # -- geometry ---------------------------------------------------------------
    def point(self, a: float, b: float, c: float = 0.0) -> np.ndarray:
        return self.mid0 + self.A * a + self.B * b + self.C * c

    def coords(self, p: np.ndarray) -> np.ndarray:
        d = np.asarray(p, dtype=np.float64) - self.mid0
        return np.array([d @ self.A, d @ self.B, d @ self.C])

    def clamp(self, ball: np.ndarray) -> np.ndarray:
        """A ball position, with the hand it asks for kept inside the box."""
        a, b, c = self.coords(np.asarray(ball, dtype=np.float64) - self.ball)
        hand = self.point(float(np.clip(a, *BOX_A)), float(np.clip(b, *BOX_B)), float(np.clip(c, *BOX_C)))
        return hand + self.ball

    def particles(self) -> np.ndarray:
        return self.env.state.particle_q.numpy()[self.cloth].astype(np.float64)

    # -- the finger -------------------------------------------------------------
    def touch(self) -> np.ndarray:
        return self.cursor.copy()

    def drag(self, cursor0, delta, up, right, away) -> np.ndarray:
        """Up on the screen lifts (world z, whatever the elevation, as the bar's press does);
        across the screen carries along the screen's horizontal."""
        del away
        return self.clamp(cursor0 + (delta @ right) * right + (delta @ up) * self.B)

    def release(self) -> np.ndarray:
        return self.cursor  # the edge stays where the finger left it

    def idle_cursor(self) -> np.ndarray:
        return self.cursor

    # -- the loop ---------------------------------------------------------------
    def drive(self) -> None:
        env = self.env
        state = env.state
        for n, ik in self.iks.items():
            pos, quat = ik.ee_pose(state)
            ik.set_target(*self.laws[n].advance(pos, quat, self.cursor - self.ball + self.off[n], self.q0[n]))
            ik.solve_into(state.joint_q, env.control.joint_target_q)
        for n, dragger in self.fold.draggers.items():
            if dragger.active and n in self.iks:
                pos, quat = self.iks[n].ee_pose(state)
                dragger.set_target(pos, quat)

    def hand(self) -> np.ndarray:
        return np.mean([np.asarray(self.iks[n].ee_pose(self.env.state)[0], dtype=np.float64) for n in self.names], axis=0)

    def tcp(self) -> np.ndarray:
        return self.hand()

    def handle_point(self) -> np.ndarray:
        return self.hand() + self.ball

    def field(self) -> dict:
        """How far the edge has been carried toward the other one: 1 is folded in half."""
        a = float(self.coords(self.hand())[0])
        return {"cleared_fraction": float(np.clip(a / max(self.length, 1e-6), 0.0, 1.0))}

    def new_stripe(self) -> None:
        self.reset_request = True  # a new towel is a new episode: the session lays it out

    def attract(self):
        """The autopilot: lift, fold over, lift off, lay back, at the website's pace."""
        cur = self.coords(self.cursor - self.ball)[:2]
        for (a, b), sec in ATTRACT_KEYS:
            nxt = np.array([a, b], dtype=np.float64)
            n = max(2, int(sec * self.hz))
            for u in np.linspace(0.0, 1.0, n):
                s = u * u * (3.0 - 2.0 * u)
                ab = cur + (nxt - cur) * s
                yield self.clamp(self.point(float(ab[0]), float(ab[1])) + self.ball)
            cur = nxt

    def broken_state(self) -> bool:
        q = self.particles()
        if not np.isfinite(q).all():
            return True
        return float(q[:, 2].min()) < self.surface_z - 0.05  # through the bench


class TowelMap:
    """The towel from the side, as the opening camera sees it: the held edge on the right, the
    fold carrying it left; each particle at its height over the bench, grey lying, orange lifted.
    A short strip, so the phone's sheet stays short. It is the page's contact map."""

    PX_PER_M = 600.0
    HEIGHT = 0.30
    BENCH = np.array([22, 22, 26], dtype=np.float32)
    FLAT = np.array([150, 160, 172], dtype=np.float32)
    UP = np.array([255, 140, 0], dtype=np.float32)
    LINE = np.array([90, 92, 96], dtype=np.float32)

    def __init__(self, towel: Towel) -> None:
        self.towel = towel
        a = (towel.particles() - towel.mid0) @ towel.A
        pad = 0.03
        self.a0 = min(float(a.min()), BOX_A[0]) - pad
        self.a1 = max(float(a.max()), BOX_A[1]) + pad
        self.w = int(round((self.a1 - self.a0) * self.PX_PER_M))
        self.h = max(48, self.w // 4)  # a strip: the phone's sheet stays short
        self.z_px = (self.h - 8) / self.HEIGHT
        self.top = 0.0

    def caption(self) -> str:
        return f"edge {100 * self.top:.0f} cm up" if self.top > 0.01 else "the towel lies flat"

    def render(self) -> np.ndarray:
        t = self.towel
        q = t.particles()
        a = (q - t.mid0) @ t.A
        z = q[:, 2] - t.surface_z
        x = ((self.a1 - a) * self.PX_PER_M).astype(np.int64)  # mirrored: +a (the fold) to the left
        y = (self.h - 1 - 6 - z * self.z_px).astype(np.int64)
        lift = np.clip((z - 0.004) / 0.15, 0.0, 1.0)
        self.top = float(np.clip(t.coords(t.hand())[1], 0.0, None))
        img = np.empty((self.h, self.w, 3), dtype=np.float32)
        img[:] = self.BENCH
        img[self.h - 6 : self.h - 4] = self.LINE  # the bench top
        col = self.FLAT[None] * (1.0 - lift[:, None]) + self.UP[None] * lift[:, None]
        ok = (y >= 0) & (y < self.h) & (x >= 0) & (x < self.w)
        for dx in (0, 1):
            for dy in (0, 1):
                img[np.clip(y[ok] + dy, 0, self.h - 1), np.clip(x[ok] + dx, 0, self.w - 1)] = col[ok]
        return img.clip(0, 255).astype(np.uint8)


def view(towel: Towel) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(look-at, eye offset): from the bench's open side (+C; the arms stand on the far one), so
    the fold carries the edge right to left and a lift reads up; aimed a little below the towel
    so it sits above a phone's sheet."""
    look = towel.point(0.45 * towel.length, -0.04)
    offset = towel.C * 0.66 + np.array([0.0, 0.0, 0.46])
    return tuple(look.tolist()), tuple(offset.tolist())


__all__ = ["BOX_A", "BOX_B", "BOX_C", "TASK", "Towel", "TowelMap", "build_env", "pre_roll", "release", "view"]
