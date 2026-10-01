"""Fold the towel, from a browser: ``cloth_fold/towel_half`` on the phone demo's touch page.

The shipped task supplies the scene: the VBD towel, the YAM pair, and the ``FoldRoutine`` that
takes the towel's edge (reach, settle, descend, close; a ``ParticleDragger`` pin rides the
hand). The routine plays until the carrying hand has latched the edge, then stops, and this
module makes the rest of the episode a person's:

* **The grip is a gripper's, not a bar.** The routine latches the whole 30 cm edge row
  (186 particles held rigidly in the hand frame), so a lifted towel hung from it as a flat
  wall. The hand keeps only the patch its fingers cover -- :data:`GRIP_WIDTH` across the edge,
  :data:`GRIP_DEPTH` into it -- and the towel drapes from that point in folds.
* **The held patch is kinematic.** Its particles get zero mass, Newton's own idiom for a
  kinematic particle, so the VBD solve leaves them to the pin instead of fighting it every
  substep (the fight wrote 0.6 m/s chatter into the edge and jittered the sheet).
* **The fingers do not snag the cloth.** The held patch rides the hand kinematically, so the
  fingers' own colliders are switched off for the towel (:meth:`Grip.free_fingers`). Left on,
  they sliced through the free cloth next to the patch whenever the hand moved fast: scribbling
  between the drag box's corners tore or tangled the towel in 2 of 5 runs (a third came close).
  The rest of that damage came from the hand dragging the patch along the bench; the drag box
  now stops 2 cm above it (:data:`BOX_B`), and 12 of 12 scribbles left the towel whole.
* **The support hand lets go.** The routine's second hand presses the towel's far corner for
  its own fold. Left there, every lift stretched the towel between the two hands -- 2x at the
  grip, enough to invert the slab's core and tangle it. It is lifted :data:`SUPPORT_RISE` off.
* **A new towel is a restore, not a replay.** The state at the end of that handover is
  captured once (:class:`Rig`); "New towel" and the watchdog put it back in a few
  milliseconds, where re-running the routine took ~30 s with the page frozen on the old towel.

A finger drags the hand: up lifts the edge, sideways carries it along the screen's horizontal,
and the hand follows through its own ``OscSetpoint`` and IK at the arm's speed limit.

**The cloth** (:func:`towel_task`) is the suite's terry slab -- 5 mm, 500 g/m^2, tri_ke 1e4 --
with a softer core and the bending carried by the faces. Measured on this scene, 2026-10-01,
with a scripted lift, swing, carry and lay, and an abuse script (the hand jumped between the
drag box's corners, then whipped sideways at full height):

====================  ================  =========================================
core shear  [Pa]      edge_ke           what it did
====================  ================  =========================================
1e5 (shipped)         1e-4 (shipped)    a stiff plate: lifted, a flat wall
3e3                   0.1 (soft end)    silk-like: limp, many fine folds
3e3                   1.0 (default)     a towel: rounded tents, folds from the grip
3e3                   31.6 (stiff end)  canvas: one smooth large bend
====================  ================  =========================================

The bending damping (``edge_kd`` 1e-4 -> :data:`EDGE_KD`) is what keeps a soft towel from
whipping itself into knots: at edge_ke 0.3 the abuse script left 246 surface
self-intersections with the shipped damping and 41 with this. The page's softness slider
writes ``edge_ke`` live (:func:`edge_ke_for`), log-scaled from :data:`EDGE_KE_STIFF` to
:data:`EDGE_KE_SOFT`.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from robosimstudio.solvers import OscSetpoint
from robosimstudio_bench.envs.env import Env

TASK = "cloth_fold/towel_half"
LATCH_PHASE = "carry"
BOX_A = (-0.10, 0.28)
"""Toward the towel's middle from where the hand latched [m]: the fold's carry."""
BOX_B = (0.02, 0.21)
"""Up from the latch height [m]. Never quite down: dragged along the bench at the latch height,
the held patch slid through folds the bench's friction held still, and scribbling the hand round
the box tore or tangled the towel in 4 of 13 runs; 2 cm up, in 0 of 12."""
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
RESET_EVERY = 1
"""Autopilot laps per fresh towel. A fresh towel is a restore now (milliseconds, not the
routine's ~30 s), so every lap starts from a flat towel."""

# -- the cloth ------------------------------------------------------------------------------
CORE_SHEAR = 3.0e3
"""The slab core's shear modulus [Pa]. The shipped 1e5 couples the two faces into a plate."""
EDGE_KD = 1.0e-2
"""Bending damping. The shipped 1e-4 let a soft towel whip itself into knots."""
EDGE_KE_STIFF, EDGE_KE_SOFT = 10.0**1.5, 0.1
"""The softness slider's ends: canvas and a limp, silk-like sheet (``edge_ke`` of the faces)."""
EDGE_KE_DEFAULT = 1.0
"""A terry towel's bending on this mesh: soft tents and folds from the grip, robust to abuse."""
SOFTEN_AT_HANDOVER = True
"""Run the shipped routine's grasp on the shipped (stiff) towel and soften it at the handover.
Built soft, the routine's fingers closed on a soft core: in 7 runs of the abuse script 5 towels
tore (edges at 2.7-15x their length) and the worst of the others reached 2.45x. Softened at the
handover, the worst edge in 6 runs was 1.43x."""
SELF_CONTACT = (0.0025, 0.004)
"""VBD self-contact (radius, margin) [m]. The suite's (2, 3) mm: at edge_ke 0.3 the abuse script
left 246 mid-surface self-intersections, and 61 with these."""

# -- the grip -----------------------------------------------------------------------------------
GRIP_WIDTH = 0.07
"""Across the edge [m]: about what the YAM's fingers cover."""
GRIP_DEPTH = 0.012
"""Into the towel from its edge [m]: the two outer rows."""
SUPPORT_RISE = 0.15
"""How far the routine's support hand lifts off the towel at the handover [m]."""
HANDOVER_STEPS = 120
"""Control steps of the handover: the support hand lifts, the towel settles under the new grip."""
BVH_EVERY = 15
"""Control steps between rebuilds of the self-contact BVH. VBD only refits it, and a refit tree
loses quality as the towel folds: over three autopilot laps a rebuild every 30 steps ran 25.5
steps/s against 23.6. A rebuild costs ~0.4 ms."""

# -- the watchdog ----------------------------------------------------------------------------
TORN_STRETCH = 2.5
"""An edge this many times its rest length means the sheet has torn or tangled."""
TANGLE_LIMIT = 80
"""Mid-surface self-intersections that call a towel tangled. A fold makes none; a towel
whipped round the box corners for 10 s settles with 0-46, one that has torn with 94-482."""
TANGLE_EVERY = 5
"""Watchdog calls (one a second) between tangle counts (13 ms flat, ~50 ms crumpled)."""
SLIP = 0.03
"""How far the held patch may trail its hand before the grip counts as lost [m]."""
GRACE = 120
"""Control steps a new towel is left alone by the watchdog."""
STRIKES = 2
"""Watchdog calls in a row a tear, slip or pass-through must last. Scribbling the hand between the
box's corners stretched an edge past 2.5x for 5 steps in one run and the towel recovered."""


def edge_ke_for(softness: float) -> float:
    """Bending stiffness for a softness in [0, 1]: log-scaled, 0 is canvas and 1 is limp."""
    s = float(np.clip(softness, 0.0, 1.0))
    lo, hi = math.log10(EDGE_KE_STIFF), math.log10(EDGE_KE_SOFT)
    return float(10.0 ** (lo + (hi - lo) * s))


def softness_for(edge_ke: float) -> float:
    """The inverse of :func:`edge_ke_for`."""
    lo, hi = math.log10(EDGE_KE_STIFF), math.log10(EDGE_KE_SOFT)
    return float(np.clip((math.log10(edge_ke) - lo) / (hi - lo), 0.0, 1.0))


def towel_task():
    """The shipped task with the towel's cloth restated (see the module docstring)."""
    from robosimstudio_bench.taskbuild import get_task

    task = get_task(TASK)
    entities = []
    for spec in task.entities:
        if getattr(spec, "name", "") == "towel" and not SOFTEN_AT_HANDOVER:
            material = replace(spec.material, core_shear_modulus=CORE_SHEAR, edge_ke=EDGE_KE_DEFAULT, edge_kd=EDGE_KD)
            spec = replace(spec, material=material)
        entities.append(spec)
    radius, margin = SELF_CONTACT
    backends = tuple(
        (name, b.with_(particle_self_contact_radius=radius, particle_self_contact_margin=margin) if name == "cloth" else b)
        for name, b in task.physics.backends
    )
    return replace(task, entities=tuple(entities), physics=replace(task.physics, backends=backends))


def build_env() -> Env:
    env = Env(towel_task(), episode=0, renderer="tiled", verbose=True)
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


def vbd_solver(env):
    """The VBD solver, under the cosim wrapper and the one-way coupler."""
    solver = env.pipeline.solver
    for _ in range(5):
        if solver is None or type(solver).__name__ == "SolverVBD":
            return solver
        solver = getattr(solver, "soft", None) or getattr(solver, "newton_solver", None)
    return None


# ---------------------------------------------------------------------------------------------
class Grip:
    """The carrying hand's hold: the routine's ``ParticleDragger``, narrowed and made kinematic.

    The dragger, its arrays and its post-solve hook are the routine's and stay installed for the
    life of the page; what changes is which particles it holds (:meth:`narrow`) and that those
    weigh nothing to the solve (:meth:`hold`).
    """

    def __init__(self, env, name: str, dragger) -> None:
        self.env, self.name, self.dragger = env, name, dragger
        model = env.model
        self.mass = model.particle_mass.numpy().copy()
        self.inv_mass = model.particle_inv_mass.numpy().copy()
        """As built: nothing is held yet when the grip is made."""
        self.held = np.zeros(0, dtype=np.int64)

    def narrow(self, candidates, pos, quat, across, inward) -> np.ndarray:
        """Re-seize only the patch under the fingers: the candidates within half the grip width of
        the hand across the edge, and within the grip depth of the edge."""
        from robosimstudio.core.math import quat_to_mat

        q = self.env.state.particle_q.numpy().astype(np.float64)
        cand = np.asarray(candidates, dtype=np.int64)
        d = q[cand] - np.asarray(pos, dtype=np.float64)
        depth = d @ np.asarray(inward, dtype=np.float64)
        keep = cand[(np.abs(d @ np.asarray(across, dtype=np.float64)) <= 0.5 * GRIP_WIDTH)
                    & (depth - depth.min() <= GRIP_DEPTH + 1e-6)]
        d = self.dragger
        n = len(keep)
        index = np.zeros(d.max_particles, dtype=np.int32)
        offset = np.zeros((d.max_particles, 3), dtype=np.float32)
        index[:n] = keep
        offset[:n] = (q[keep] - np.asarray(pos, dtype=np.float64)) @ quat_to_mat(quat)
        d._grabbed.assign(index)
        d._offsets.assign(offset)
        d._count_host[0] = n
        d._count.assign(d._count_host)
        d.grabbed_indices = keep.astype(np.int32)
        d.set_target(pos, quat)
        self.hold(keep)
        return keep

    def hold(self, indices) -> None:
        """Zero the held particles' mass (and only theirs)."""
        self.held = np.asarray(indices, dtype=np.int64)
        mass, inv = self.mass.copy(), self.inv_mass.copy()
        mass[self.held] = 0.0
        inv[self.held] = 0.0
        self.env.model.particle_mass.assign(mass)
        self.env.model.particle_inv_mass.assign(inv)

    def free_fingers(self) -> list[int]:
        """Stop the carrying hand's fingers colliding with the cloth; return the shapes switched off.

        The fingers are the bodies below the hand's tool body. ``COLLIDE_PARTICLES`` is read per
        thread by Newton's soft-contact kernel, so the change needs no rebuild and no recapture.
        """
        import newton

        model = self.env.model
        tool = int(self.env.arm(self.name).ee_body)
        parent = {int(c): int(p) for p, c in zip(model.joint_parent.numpy(), model.joint_child.numpy())}
        below = set()
        grew = True
        while grew:
            grew = False
            for child, up in parent.items():
                if (up == tool or up in below) and child not in below:
                    below.add(child)
                    grew = True
        bodies = model.shape_body.numpy()
        flags = model.shape_flags.numpy()
        bit = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        off = [i for i in range(len(bodies)) if int(bodies[i]) in below and flags[i] & bit]
        flags[off] &= ~bit
        model.shape_flags.assign(flags)
        return off

    def installed(self) -> bool:
        return self.dragger.apply in self.env.pipeline.hooks.post_solve

    def capture(self) -> dict:
        d = self.dragger
        return {
            "grabbed": d._grabbed.numpy().copy(),
            "offsets": d._offsets.numpy().copy(),
            "count": int(d._count_host[0]),
            "target": d._target_host.copy(),
            "held": self.held.copy(),
        }

    def restore(self, saved: dict) -> None:
        d = self.dragger
        d._grabbed.assign(saved["grabbed"])
        d._offsets.assign(saved["offsets"])
        d._count_host[0] = saved["count"]
        d._count.assign(d._count_host)
        d._target_host[:] = saved["target"]
        d._target.assign(d._target_host)
        d.grabbed_indices = np.asarray(saved["held"], dtype=np.int32)
        self.hold(saved["held"])
        if not self.installed():  # somebody reset the env, which retires episode hooks
            self.env.pipeline.hooks.post_solve.append(d.apply)
            self.env.pipeline.invalidate_graph()

    def lag(self) -> float:
        """How far the held patch trails where the hand says it belongs [m]."""
        from robosimstudio.core.math import quat_to_mat

        d = self.dragger
        n = int(d._count_host[0])
        if n == 0:
            return float("inf")
        q = self.env.state.particle_q.numpy()[np.asarray(d.grabbed_indices, dtype=np.int64)].astype(np.float64)
        off = d._offsets.numpy()[:n].astype(np.float64)
        target = d._target_host[0].astype(np.float64)
        goal = target[:3] + off @ quat_to_mat(target[3:7]).T
        return float(np.linalg.norm(q - goal, axis=1).max())


class Rig:
    """Everything the page builds once: the routine's grip, the support hand's park, and the
    state a fresh towel starts from.

    :meth:`handover` runs once after the pre-roll; :meth:`restore` is "New towel".
    """

    STATE = ("particle_q", "particle_qd", "joint_q", "joint_qd", "body_q", "body_qd")
    CONTROL = ("joint_target_q", "joint_target_qd", "joint_f", "joint_act")

    def __init__(self, env, routine) -> None:
        self.env, self.routine = env, routine
        names = [n for n, d in routine.draggers.items() if d.active]
        if not names:
            raise RuntimeError("no hand holds the towel's edge")
        self.name = names[0]
        self.grip = Grip(env, self.name, routine.draggers[self.name])
        self.support = env.handles.support_arm if env.handles.support_arm != self.name else ""
        self.vbd = vbd_solver(env)
        self.tunable = env[routine.cloth_name].tunables()[0]
        self.edge_ke = EDGE_KE_DEFAULT
        self.snapshot: dict | None = None
        self.fingers: list[int] = []
        """The finger shapes that no longer touch the cloth (:meth:`Grip.free_fingers`)."""
        dh = env.handles.deformables.get(routine.cloth_name) or next(iter(env.handles.deformables.values()))
        self.cloth = np.asarray(dh.particles, dtype=np.int64)
        tris = env.model.tri_indices.numpy()
        mine = np.isin(tris, self.cloth).all(axis=1)
        edges = np.concatenate([tris[mine][:, [0, 1]], tris[mine][:, [1, 2]], tris[mine][:, [2, 0]]])
        self.edges = np.unique(np.sort(edges, axis=1), axis=0)
        self.rest_len: np.ndarray | None = None
        # The mid-surface's triangles, in the cloth's own numbering: a slab's particles are its
        # mid-surface offset to either face (``prism_slab``: bottom 0..n-1, top n..2n-1), so its
        # bottom face's triangles are the mid-surface's; a membrane's triangles are the sheet.
        local = np.full(int(self.cloth.max()) + 1, -1, dtype=np.int64)
        local[self.cloth] = np.arange(len(self.cloth))
        faces = local[tris[mine]]
        side = int(round(math.sqrt(len(self.cloth))))
        if side * side != len(self.cloth):
            faces = faces[(faces < len(self.cloth) // 2).all(axis=1)]
        self.faces = faces

    # -- once ------------------------------------------------------------------------------
    def handover(self):
        """Narrow the grip, lift the support hand off, let the towel settle; then capture.

        A generator of control steps, so the page keeps drawing.
        """
        env = self.env
        ik = env.arm(self.name)
        pos, quat = ik.ee_pose(env.state)
        q = env.state.particle_q.numpy()[self.cloth].astype(np.float64)
        inward = q.mean(axis=0) - np.asarray(pos, dtype=np.float64)
        inward[2] = 0.0
        inward /= np.linalg.norm(inward) or 1.0
        across = np.cross(np.array([0.0, 0.0, 1.0]), inward)
        self.rest_len = self._lengths(env.state.particle_q.numpy())  # the towel as it lay
        if SOFTEN_AT_HANDOVER:
            self.tunable.write("core_shear_modulus", CORE_SHEAR)
            self.tunable.write("edge_kd", EDGE_KD)
            self.set_edge_ke(self.edge_ke)
        self.grip.narrow(self.grip.dragger.grabbed_indices, pos, quat, across, inward)
        self.fingers = self.grip.free_fingers()
        park = SupportPark(env, self.support) if self.support else None
        towel = self.towel()
        for _ in range(HANDOVER_STEPS):
            if park is not None:
                park.step()
            towel.drive()
            env.step_teleop()
            yield
        self.capture()

    def capture(self) -> None:
        p = self.env.pipeline
        self.snapshot = {
            "states": [
                {f: getattr(st, f).numpy().copy() for f in self.STATE if getattr(st, f, None) is not None}
                for st in (p.state_0, p.state_1)
            ],
            "control": {f: getattr(p.control, f).numpy().copy() for f in self.CONTROL if getattr(p.control, f, None) is not None},
            "grip": self.grip.capture(),
        }

    # -- every new towel ---------------------------------------------------------------------
    def restore(self) -> None:
        """Put the handover state back: a fresh towel in the hand, in milliseconds.

        The solvers' own buffers first (MuJoCo's warm start and applied forces, VBD's rigid
        history; VBD rebaselines its particle history every step), then every state and control
        array, then the grip, then a fresh self-contact BVH for the flat towel.
        """
        if self.snapshot is None:
            raise RuntimeError("no handover captured yet")
        p = self.env.pipeline
        p.reset_solvers(flags=0)
        for st, saved in zip((p.state_0, p.state_1), self.snapshot["states"]):
            for f, a in saved.items():
                getattr(st, f).assign(a)
        for f, a in self.snapshot["control"].items():
            getattr(p.control, f).assign(a)
        self.grip.restore(self.snapshot["grip"])
        if self.vbd is not None:
            self.vbd.rebuild_bvh(p.state_0)
        if hasattr(p, "reset_finite_flags"):
            p.reset_finite_flags()
        p.mark_dirty()

    def towel(self) -> Towel:
        return Towel(self)

    # -- the cloth ---------------------------------------------------------------------------
    def set_edge_ke(self, value: float) -> float:
        value = float(np.clip(value, min(EDGE_KE_STIFF, EDGE_KE_SOFT), max(EDGE_KE_STIFF, EDGE_KE_SOFT)))
        self.tunable.write("edge_ke", value)
        self.edge_ke = value
        return value

    def _lengths(self, q) -> np.ndarray:
        return np.linalg.norm(q[self.edges[:, 0]] - q[self.edges[:, 1]], axis=1)

    def stretch(self) -> float:
        """The most any edge of the sheet is stretched over its length at the handover."""
        if self.rest_len is None:
            return 1.0
        return float((self._lengths(self.env.state.particle_q.numpy()) / np.maximum(self.rest_len, 1e-9)).max())


class SupportPark:
    """The routine's support hand, lifted straight up off the towel through its own OSC and IK."""

    def __init__(self, env, name: str) -> None:
        self.env = env
        self.ik = env.arm(name)
        pos, quat = self.ik.ee_pose(env.state)
        self.quat = np.asarray(quat, dtype=np.float64)
        self.target = np.asarray(pos, dtype=np.float64) + np.array([0.0, 0.0, SUPPORT_RISE])
        self.law = OscSetpoint(control_dt=1.0 / float(env.request.physics.control_hz))
        self.law.reset(np.asarray(pos, dtype=np.float64), self.quat)

    def step(self) -> None:
        pos, quat = self.ik.ee_pose(self.env.state)
        self.ik.set_target(*self.law.advance(pos, quat, self.target, self.quat))
        self.ik.solve_into(self.env.state.joint_q, self.env.control.joint_target_q)


# ---------------------------------------------------------------------------------------------
class Towel:
    """The carrying hand, driven by a finger. The phone page's driver interface."""

    scene = "towel"

    def __init__(self, rig: Rig) -> None:
        # ``fold``, not ``routine``: the phone page's restart reads ``routine.dragger`` (the peel's
        # one pin); this towel's grip belongs to the rig and outlives every towel.
        env = rig.env
        self.rig, self.env, self.fold = rig, env, rig.routine
        self.names = (rig.name,)
        self.iks = {rig.name: env.arm(rig.name)}
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
        self.cloth = rig.cloth
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
        self.lap_reset = False
        """Whether the reset asked for is the autopilot's, at the end of a lap."""
        self.hz = hz
        self.laps = 0
        self.lap_done = False
        self.steps = 0
        self.checks = 0
        self.strikes = 0
        self.why = ""
        """Why :meth:`broken_state` last said broken, for the page's notice."""
        self.detail = ""
        """By how much, for the log."""

    # -- geometry ---------------------------------------------------------------------------
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

    # -- the finger --------------------------------------------------------------------------
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

    # -- the loop ---------------------------------------------------------------------------
    def drive(self) -> None:
        env = self.env
        state = env.state
        for n, ik in self.iks.items():
            pos, quat = ik.ee_pose(state)
            ik.set_target(*self.laws[n].advance(pos, quat, self.cursor - self.ball + self.off[n], self.q0[n]))
            ik.solve_into(state.joint_q, env.control.joint_target_q)
            # A latched edge rides its own hand.
            self.rig.grip.dragger.set_target(pos, quat)
        self.steps += 1
        if self.rig.vbd is not None and self.steps % BVH_EVERY == 0:
            self.rig.vbd.rebuild_bvh(env.pipeline.state_0)

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
        """"New towel", or the autopilot's lap just ended (the page calls this for both)."""
        self.lap_reset = self.lap_done
        if self.lap_done:
            self.lap_done = False
            self.laps += 1
            if self.laps % RESET_EVERY:
                return  # the lap laid the edge back: keep this towel a while
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
        self.lap_done = True  # read by the new_stripe the page calls next

    def broken_state(self) -> bool:
        """Whether this towel is past saving: the page lays a fresh one (a restore) when it is.

        Called once a second. Nothing in a new towel's first :data:`GRACE` steps; a blow-up or a
        fall at once; a tear, a slip or the bench passed through only when it is still there on
        the next call (a whip can stretch an edge past the limit for a few steps and recover);
        the tangle count every :data:`TANGLE_EVERY` calls. :attr:`why` says which, for the page's
        notice, and :attr:`detail` by how much, for the log.
        """
        self.why = self.detail = ""
        if self.steps < GRACE:
            self.strikes = 0
            return False
        q = self.particles()
        if not np.isfinite(q).all() or not np.isfinite(self.env.state.joint_q.numpy()).all():
            self.why = "the simulation blew up"
            return True
        if q[:, 2].min() < self.surface_z - 0.25:
            self.why, self.detail = "the towel fell off the bench", f"{q[:, 2].min() - self.surface_z:+.2f} m"
            return True
        why = detail = ""
        lag = self.rig.grip.lag()
        stretch = self.rig.stretch()
        if self._through_bench(q):
            why, detail = "the towel went through the bench", f"lowest {q[:, 2].min() - self.surface_z:+.3f} m"
        elif lag > SLIP:
            why, detail = "the towel slipped out of the hand", f"{100 * lag:.1f} cm from the fingers"
        elif stretch > TORN_STRETCH:
            why, detail = "the towel tore", f"an edge at {stretch:.2f}x its length"
        self.strikes = self.strikes + 1 if why else 0
        if self.strikes >= STRIKES:
            self.why, self.detail = why, detail
            return True
        if not why:
            self.checks += 1
            if self.checks % TANGLE_EVERY == 0:
                crossings = self.tangles(q)
                if crossings > TANGLE_LIMIT:
                    self.why, self.detail = "the towel tangled itself", f"{crossings} self-intersections"
                    return True
        return False

    def _through_bench(self, q: np.ndarray) -> bool:
        from robosimstudio.scene.builtin.benches import BENCH_0728 as cell

        inside = (np.abs(q[:, 0] - cell.center[0]) < cell.half[0] - 0.01) & (
            np.abs(q[:, 1] - cell.center[1]) < cell.half[1] - 0.01
        )
        return bool((q[inside, 2] < self.surface_z - 0.02).any())

    def tangles(self, q: np.ndarray | None = None) -> int:
        """Self-intersections of the sheet's mid-surface (see :func:`self_intersections`)."""
        return self_intersections(self.mid_surface(self.particles() if q is None else q), self.rig.faces)

    def mid_surface(self, q: np.ndarray) -> np.ndarray:
        """The sheet's mid-surface: a slab's two faces averaged, a membrane as it is."""
        n = len(q)
        side = int(round(math.sqrt(n)))
        if side * side == n:
            return q
        return 0.5 * (q[: n // 2] + q[n // 2 :])


def self_intersections(mid: np.ndarray, faces: np.ndarray, reach: float = 0.02) -> int:
    """Edges of the mid-surface that pass through a triangle they do not belong to.

    A fold presses two layers together and makes none; a towel that has been whipped through
    itself makes dozens. Segment-triangle tests (Moller-Trumbore) over the pairs within
    ``reach`` of each other.
    """
    from scipy.spatial import cKDTree

    faces = np.asarray(faces, dtype=np.int64)
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    q = np.asarray(mid, dtype=np.float64)
    pairs = cKDTree(0.5 * (q[edges[:, 0]] + q[edges[:, 1]])).sparse_distance_matrix(
        cKDTree(q[faces].mean(axis=1)), reach, output_type="coo_matrix"
    )
    e, f = edges[pairs.row], faces[pairs.col]
    apart = ~((f == e[:, :1]).any(1) | (f == e[:, 1:]).any(1))
    e, f = e[apart], f[apart]
    p0, p1 = q[e[:, 0]], q[e[:, 1]]
    a, b, c = q[f[:, 0]], q[f[:, 1]], q[f[:, 2]]
    d, e1, e2 = p1 - p0, b - a, c - a
    h = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = p0 - a
    u = inv * np.einsum("ij,ij->i", s, h)
    qv = np.cross(s, e1)
    v = inv * np.einsum("ij,ij->i", d, qv)
    t = inv * np.einsum("ij,ij->i", e2, qv)
    return int((ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)).sum())


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
    look = towel.point(0.25 * towel.length, -0.04)  # nearer the hand: a phone held upright sees ~0.38 m across
    offset = towel.C * 0.66 + np.array([0.0, 0.0, 0.46])
    return tuple(look.tolist()), tuple(offset.tolist())


__all__ = [
    "BOX_A", "BOX_B", "BOX_C", "EDGE_KE_DEFAULT", "TASK", "Rig", "Towel", "TowelMap", "build_env", "edge_ke_for",
    "pre_roll", "self_intersections", "softness_for", "towel_task", "view",
]
