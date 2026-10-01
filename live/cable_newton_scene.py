"""Route the cable, from a browser: a Newton-native cable woven round three clips by a Panda.

The cable is a chain of capsules on two-axis bending joints with passive stiffness, solved by
MuJoCo-Warp in the **same island as the arm** -- no second solver, no coupler, no latch. One end
is fixed in a socket on the left of a routing board; the other is held in the Panda's jaws by a
MuJoCo weld (an equality constraint between the hand and the last capsule). Everything else is
contact: the cable lies on the board, wraps the clips' posts and slides round them.

Why this formulation (measured on one RTX PRO 6000, 2026-10-01, see the task's report): it is
the fastest Newton cable that still reads as one. MuJoCo-Warp is fast only while the whole model
stays on its dense path, **nv <= 60** -- a 24-capsule chain on ball joints (nv 81 with the arm)
ran at 64 control steps/s, the same chain at nv 57 at 212. So the bends are two hinges per joint
(no twist: a round cable never shows one) and the arm is the Panda (9 dof). The VBD capsule
chain (``rss.rod``) on its own island, proxy-coupled to the arm, is 11 steps/s at the budget
that holds its joints together.

A finger drags the hand across the board at a fixed carry height (sideways on the screen is
along the route, up the screen is away from you). The cable trails behind and wraps whatever
clip the hand goes round; a clip counts as routed when the cable passes it on its marked side,
close by, in order from the socket.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import newton
import numpy as np
import warp as wp

import robosimstudio as rss
from robosimstudio.core.math import quat_to_mat
from robosimstudio.domains.rod import RodSpec
from robosimstudio.solvers import OscSetpoint
from robosimstudio.specs import ArrangementSpec, CollisionSpec, SimRequirements, StaticSpec, backend
from robosimstudio.specs.asset import ShapeSpec

# ----------------------------------------------------------------------------------- layout
# Work-surface frame of bench_075 (origin on the bench top, x toward the open side, away from
# the Panda, which stands at x = -0.6). The clips stand on one line, the route weaves round it.
LINE_X = -0.17
"""x of the clips' line [m]: ~0.43 m in front of the Panda's base, where it reaches top-down."""
SOCKET_Y = -0.36
CLIP_Y = (-0.22, -0.11, 0.0)
CLIP_SIDE = (+1, -1, +1)
"""Which side of each clip the route passes: +1 in front (toward you), -1 behind (robot side)."""
END_Y = 0.10
"""Where the route ends, right of the last clip [m]: the routed path is ~0.47 m of the 0.50."""
ROUTE_TOL = 0.035
"""How far from a clip's surface the cable may pass and still count as routed by it [m]. The
fresh cable passes clip 1 170 mm out; a routed one 0-30 mm (the slack settles by clip 1)."""
ROUTE_OFFSET = 0.022
"""How far off the clips' line the route runs past each clip [m] (cable centre to line)."""

BOARD_T = 0.012
"""Routing board thickness [m]; the cable lies on its top."""
BOARD_X = (LINE_X - 0.12, LINE_X + 0.22)
BOARD_Y = (SOCKET_Y - 0.07, END_Y + 0.06)

CABLE_L = 0.50
CABLE_R = 0.004
SEGMENTS = 24
"""21 mm capsules. 24 two-axis joints are 48 dof; with the Panda's 9 the model is nv 57, inside
MuJoCo-Warp's dense limit (60), which is what keeps it fast."""
BEND_KE = 0.004
"""Per-joint bending stiffness [N m/rad]: EI ~ 8e-5 N m^2 over a 20 mm segment, a USB lead."""
BEND_KD = 1.0e-4
ARMATURE = 2.0e-6
"""Joint armature [kg m^2]: keeps a 1.5 g capsule's stiffest mode slow next to the step."""
DENSITY = 1300.0
JOINT = "pair"
CAP_STRETCH = 1.0
"""Capsule cylinder length over segment length (1: the round ends overlap the neighbours')."""
GIMBAL_MASS = 1.0e-4
GIMBAL_INERTIA = 1.0e-9
"""The link between a bend's two hinges [kg, kg m^2]: only there to carry the first hinge."""

CLIP_R = 0.009
CLIP_HALF = 0.022
"""Each clip is a capsule post, 18 mm thick and 62 mm tall (rounded top)."""
CARRY_Z = 0.032
"""TCP height above the board while carrying [m]: fingertips ~1 cm clear of the board."""
KEEP_OUT = 0.034
"""Closest the TCP may come to a clip's axis [m] -- the jaws are 13 mm either side of it."""

SOCKET_HALF = (0.022, 0.016, 0.014)

MU_CABLE = 0.15
MU_BOARD = 0.35
MU_CLIP = 0.05

CONTROL_HZ = 60.0
SUBSTEPS = 4
"""4.2 ms MuJoCo steps (with impratio 1, see ``physics``). Against 8: both survive the stress
(yank, flicks, 120 s of random drags); under a hard yank the cable sinks <= 2.8 mm into a post at
4 and <= 1.5 at 8, and 4 runs twice as fast (~160 against ~88 control steps/s headless)."""

CABLE_COLOR = (0.08, 0.08, 0.09)
BOARD_COLOR = (0.20, 0.23, 0.27)
CLIP_COLOR = (0.80, 0.84, 0.90)
"""Pale steel: orange is the drag ball's alone."""
SOCKET_COLOR = (0.62, 0.64, 0.68)

CONTACT_TIMECONST = 0.009
"""MuJoCo solref time constant for every contact the cable makes [s] -- just above 2 dt at 4
substeps (MuJoCo clamps anything shorter). MuJoCo's soft contact gives way by
``(1 - d) F / (d k m)`` and a capsule here is 1.5 g: at the default (0.02 s, d 0.95) a 1 N pull
sinks 13 mm into a post, and a taut cable walks through it. At 9 ms and d 0.99, ~0.5 mm."""
CONTACT_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)


def physics(*, substeps: int = SUBSTEPS, **overrides) -> ArrangementSpec:
    # Dense Jacobians: nv is 57, inside MuJoCo-Warp's dense limit (60), and 15 % faster than
    # sparse here. Elliptic cones: pyramidal ones went NaN within 3 s of the route. impratio 1,
    # not RoboSimStudio's 10: at 10 the friction rows are ten times stiffer than the normals on
    # 1.5 g capsules, and a calm, stalled scene went NaN in the float32 solve 15-40 s into a
    # random drag (3 of 3 seeds at 4 substeps, 2 of 3 at 6); at 1, 5 of 5 seeds ran yank +
    # flicks + 120 s of random drags clean at 4 substeps (all measured).
    params = dict(solver="newton", integrator="implicitfast", cone="elliptic", impratio=1.0, iterations=20,
                  ls_iterations=10, njmax=1024, nconmax=512, jacobian="dense")
    params.update(overrides)
    return ArrangementSpec(
        control_hz=CONTROL_HZ,
        substeps=substeps,
        backends=(("*", backend("mujoco", **params)),),
        collision=CollisionSpec(),
    )


PHYSICS = physics()
"""One MuJoCo-Warp island for the arm and the cable: no coupling to pay for or tune."""


def surface_xy(dx: float, y: float) -> np.ndarray:
    return np.array([LINE_X + dx, y], dtype=np.float64)


def clip_xy(k: int) -> np.ndarray:
    return surface_xy(0.0, CLIP_Y[k])


def route_points() -> np.ndarray:
    """The route a routed cable follows, as 2-D board points: socket, past each clip on its side, end."""
    pts = [surface_xy(0.0, SOCKET_Y + SOCKET_HALF[1])]
    for k, y in enumerate(CLIP_Y):
        pts.append(surface_xy(CLIP_SIDE[k] * ROUTE_OFFSET, y))
    pts.append(surface_xy(0.0, END_Y))
    return np.asarray(pts)


def catmull_rom(points: np.ndarray, per_leg: int = 24) -> np.ndarray:
    """A centripetal-free (uniform) Catmull-Rom curve through ``points``."""
    p = np.asarray(points, dtype=np.float64)
    ext = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for t in np.linspace(0.0, 1.0, per_leg, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(p[-1])
    return np.asarray(out)


def resample(curve: np.ndarray, length: float, n: int) -> np.ndarray:
    """``n + 1`` points at equal arc length along ``curve``, cut (or extended straight) to ``length``."""
    c = np.asarray(curve, dtype=np.float64)
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < length:
        d = (c[-1] - c[-2]) / max(seg[-1], 1e-12)
        c = np.vstack([c, c[-1] + d * (length - s[-1])])
        s = np.append(s, length)
    targets = np.linspace(0.0, length, n + 1)
    return np.stack([np.interp(targets, s, c[:, i]) for i in range(c.shape[1])], axis=1)


def turtle(start: np.ndarray, heading: float, moves, step: float = 0.002) -> np.ndarray:
    """A 2-D path from ``start`` [m] and ``heading`` [rad]: ``("line", length)`` and
    ``("arc", radius, degrees)`` moves, positive degrees turning toward +x from +y (clockwise
    seen from above with x toward you)."""
    pts = [np.asarray(start, dtype=np.float64)]
    h = float(heading)
    for move in moves:
        if move[0] == "line":
            n = max(1, int(move[1] / step))
            for _ in range(n):
                pts.append(pts[-1] + (move[1] / n) * np.array([math.cos(h), math.sin(h)]))
        else:
            radius, deg = float(move[1]), float(move[2])
            length = abs(math.radians(deg)) * radius
            n = max(2, int(length / step))
            dh = math.radians(deg) / n
            for _ in range(n):
                h -= dh
                pts.append(pts[-1] + (length / n) * np.array([math.cos(h + 0.5 * dh), math.sin(h + 0.5 * dh)]))
    return np.asarray(pts)


START_MOVES = (
    ("arc", 0.02, 90.0), ("line", 0.08), ("arc", 0.02, -180.0), ("line", 0.06), ("arc", 0.02, 180.0),
    ("line", 0.09), ("arc", 0.03, -90.0), ("line", 0.3),
)
"""The fresh cable's lie: out of the socket, folded toward you in an N between the socket and the
first clip, the end running right along the front to where the hand holds it, between clips 1 and
2 and 17 cm in front of them (the wrist stands clear of clip 1 in the phones' view) -- the slack
in view, no leg crossing another or nearer a clip than 40 mm."""


def start_nodes() -> np.ndarray:
    """The fresh cable's ``SEGMENTS + 1`` nodes in the work-surface frame. All on the board but the
    last two segments, which rise to the hand at the carry height."""
    exit_xy = surface_xy(0.0, SOCKET_Y + SOCKET_HALF[1])
    # heading measured from +x toward +y: +y (to the right) is pi/2
    path = turtle(exit_xy, math.pi / 2, START_MOVES)
    xy = resample(path, CABLE_L, SEGMENTS)
    z = np.full(len(xy), BOARD_T + CABLE_R + 0.0005)
    lift = BOARD_T + CARRY_Z
    z[-2] = 0.5 * (z[-3] + lift)
    z[-1] = lift
    nodes = np.column_stack([xy, z])
    # the rise shortens the plan view; stretch the last two legs back to their rest length
    seg = CABLE_L / SEGMENTS
    for i in (len(nodes) - 2, len(nodes) - 1):
        d = nodes[i] - nodes[i - 1]
        nodes[i] = nodes[i - 1] + d / np.linalg.norm(d) * seg
    return nodes


# ------------------------------------------------------------------------------- the cable
@dataclass(frozen=True, kw_only=True)
class ChainCableSpec(RodSpec):
    """A capsule chain on two-axis bending joints for MuJoCo-Warp, one end in a socket, one in a hand."""

    entity: str = "newton_chain_cable"
    armature: float = ARMATURE
    socket_stiffness: float = 0.05
    """The first joint's bend stiffness [N m/rad]: the strain relief where the cable leaves the socket."""
    joint: str = "pair"
    """``"pair"``: two hinges per node on a gimbal link; ``"ball"``: a ball joint; ``"alt"``: one
    hinge per node, alternating axes."""
    hold_arm: str = "right"
    """The arm whose hand holds the last capsule (weld). Empty: nothing holds it."""
    finger_q: float = 0.0005
    """Finger travel while holding [m] (each of the Panda's two prismatic fingers)."""
    mu: float = MU_CABLE


def _quat_z_to(axis: np.ndarray) -> wp.quat:
    """The rotation taking +z onto ``axis``."""
    z = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    a = np.cross([0.0, 0.0, 1.0], z)
    s, c = float(np.linalg.norm(a)), float(z[2])
    if s < 1e-9:
        return wp.quat_identity() if c > 0 else wp.quat(1.0, 0.0, 0.0, 0.0)
    return wp.quat_from_axis_angle(wp.vec3(*(a / s)), float(math.atan2(s, c)))


@rss.register_entity("newton_chain_cable")
class ChainCable(rss.Entity):
    """The cable: built as an articulation MuJoCo solves beside the arm.

    ``domain = "rigid"`` routes it to the arrangement's ``"*"`` backend -- the same MuJoCo
    island as the robot -- and makes it an ``objects`` node, which touches the surface, the
    fixtures and the robot."""

    domain = "rigid"

    def __init__(self, spec: ChainCableSpec, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.bodies = np.zeros(0, dtype=np.int32)
        """The capsules, socket to hand."""
        self.all_bodies = np.zeros(0, dtype=np.int32)
        self.joints = np.zeros(0, dtype=np.int32)
        self.half = np.zeros(0, dtype=np.float64)
        self.eq = -1
        self.hand_body = -1
        self.rest_q: np.ndarray | None = None
        """The chain's joint coordinates at rest (its slice of ``joint_q``)."""
        self.q_span = (0, 0)
        self.start_arm_q: np.ndarray | None = None
        """The arm's coordinates holding the fresh cable, found once by IK."""
        self.arm_coords: list[int] = []
        self.finger_coords: list[int] = []
        self.finger_q: list[float] = []

    def sim_requirements(self) -> SimRequirements:
        return SimRequirements(domain="rigid")

    # ------------------------------------------------------------------ build
    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec: ChainCableSpec = self.spec
        nodes = np.asarray([self.placed((*p, 0.0, 0.0, 0.0, 1.0))[:3] for p in spec.rest_polyline()])
        r = float(spec.radius)
        cfg = newton.ModelBuilder.ShapeConfig(density=float(spec.density), mu=float(spec.mu), margin=0.0, gap=0.002)
        b0, j0 = builder.body_count, builder.joint_count
        prev, q_prev, l_prev = -1, None, 0.0
        joints, half, caps, capsule_bodies = [], [], [], []
        ident = wp.transform_identity()
        g_inertia = wp.mat33(GIMBAL_INERTIA, 0.0, 0.0, 0.0, GIMBAL_INERTIA, 0.0, 0.0, 0.0, GIMBAL_INERTIA)
        for i in range(len(nodes) - 1):
            p0, p1 = nodes[i], nodes[i + 1]
            length = float(np.linalg.norm(p1 - p0))
            q = _quat_z_to(p1 - p0)
            # A bend is two hinges at the node: about the capsule's own x on a massless-ish gimbal
            # link, then about its y. Revolute joints only, so the arm's LM IK still builds (it
            # refuses D6 and more than 64 coordinates: 48 + the Panda's 9 is 57).
            gimbal = (builder.add_link(xform=wp.transform(wp.vec3(*p0), q), mass=GIMBAL_MASS, inertia=g_inertia,
                                       label=f"{spec.name}_gimbal{i}") if spec.joint == "pair" else -1)
            body = builder.add_link(xform=wp.transform(wp.vec3(*(0.5 * (p0 + p1))), q), label=f"{spec.name}_seg{i}")
            # The cylinder spans node to node and the round ends overlap the neighbours': the
            # cable reads as one cord, not a string of beads (neighbours never collide).
            caps.append(builder.add_shape_capsule(
                body, radius=r, half_height=0.5 * length * CAP_STRETCH, cfg=cfg, color=spec.color,
                label=f"{spec.name}_cap{i}",
            ))
            if prev < 0:
                parent_xf = wp.transform(wp.vec3(*p0), q)
            else:
                parent_xf = wp.transform(wp.vec3(0.0, 0.0, 0.5 * l_prev), wp.quat_inverse(q_prev) * q)
            k = float(spec.socket_stiffness if i == 0 else spec.bend_ke)
            common = dict(armature=float(spec.armature), damping=float(spec.bend_kd),
                          custom_attributes={"mujoco:dof_passive_stiffness": k})
            child_xf = wp.transform(wp.vec3(0.0, 0.0, -0.5 * length), wp.quat_identity())
            if spec.joint == "pair":
                joints.append(builder.add_joint_revolute(prev, gimbal, parent_xform=parent_xf, child_xform=ident,
                                                         axis=(1.0, 0.0, 0.0), label=f"{spec.name}_bx{i}", **common))
                joints.append(builder.add_joint_revolute(gimbal, body, parent_xform=ident, child_xform=child_xf,
                                                         axis=(0.0, 1.0, 0.0), label=f"{spec.name}_by{i}", **common))
            elif spec.joint == "ball":
                joints.append(builder.add_joint_ball(
                    prev, body, parent_xform=parent_xf, child_xform=child_xf, armature=float(spec.armature),
                    damping=float(spec.bend_kd), label=f"{spec.name}_ball{i}",
                    custom_attributes={"mujoco:dof_passive_stiffness": [k, k, k]}))
            else:  # "alt": one hinge per node, the axis alternating
                axis = (1.0, 0.0, 0.0) if i % 2 == 0 else (0.0, 1.0, 0.0)
                joints.append(builder.add_joint_revolute(prev, body, parent_xform=parent_xf, child_xform=child_xf,
                                                         axis=axis, label=f"{spec.name}_b{i}", **common))
            half.append(0.5 * length)
            capsule_bodies.append(body)
            prev, q_prev, l_prev = body, q, length
        builder.add_articulation(joints, label=f"{spec.name}_chain")
        # Neighbours meet at a node with overlapping caps; the gimbal between them hides the
        # parent-child relation MuJoCo would otherwise exclude, so say it.
        for a, b in zip(caps[:-1], caps[1:]):
            builder.add_shape_collision_filter_pair(a, b)
        self.all_bodies = np.arange(b0, builder.body_count, dtype=np.int32)
        self.bodies = np.asarray(capsule_bodies, dtype=np.int32)
        self.joints = np.arange(j0, builder.joint_count, dtype=np.int32)
        self.half = np.asarray(half)
        self.q_span = (int(builder.joint_q_start[j0]), int(builder.joint_coord_count))
        self._hold(builder)

    def _hold(self, builder: newton.ModelBuilder) -> None:
        """Weld the last capsule to the hand, and keep the jaws from fighting the weld."""
        emb = self.embodiment
        arm = self.spec.hold_arm
        hand = getattr(emb, "hands", {}).get(arm) if emb is not None else None
        if hand is None:
            return
        from newton._src.solvers.mujoco.enums import EqType
        from newton._src.solvers.mujoco.equality import _add_equality_constraint
        from robosimstudio.scene.entity import label_index

        labels = list(builder.body_label)
        self.hand_body = label_index(labels, hand.tcp_body)
        fingers = [label_index(labels, n) for side in hand.finger_bodies for n in side]
        last = int(self.bodies[-1])
        # The pose is written once the arm has been posed (on_scene_ready); until then identity.
        self.eq = _add_equality_constraint(
            builder, EqType.WELD, body1=self.hand_body, body2=last, anchor=(0.0, 0.0, 0.0),
            relpose=wp.transform_identity(), label=f"{self.spec.name}_in_hand",
        )
        # The whole hand -- palm, fingers, pads -- passes through the cable. The weld is the hold,
        # so the jaws need no contact with it; a finger pressing a capsule into the board is a
        # light body squeezed between a servo and a static, a fight the cable can only lose.
        root = self._ancestor(builder, self.hand_body, 2)
        prefix = labels[root] if root >= 0 else labels[self.hand_body]
        hand_bodies = {b for b in range(builder.body_count) if labels[b].startswith(prefix)} | {self.hand_body, *fingers}
        mine = set(int(b) for b in self.bodies)
        shape_body = builder.shape_body
        hand_shapes = [s for s in range(builder.shape_count) if int(shape_body[s]) in hand_bodies]
        cable_shapes = [s for s in range(builder.shape_count) if int(shape_body[s]) in mine]
        for a in hand_shapes:
            for b in cable_shapes:
                builder.add_shape_collision_filter_pair(a, b)

    @staticmethod
    def _ancestor(builder: newton.ModelBuilder, body: int, levels: int) -> int:
        """The body ``levels`` joints up the tree from ``body`` (-1 past the root)."""
        parent_of = {int(c): int(p) for c, p in zip(builder.joint_child, builder.joint_parent)}
        for _ in range(levels):
            body = parent_of.get(int(body), -1)
            if body < 0:
                return -1
        return body

    # ------------------------------------------------------------------ bind
    def bind(self, model: newton.Model) -> None:
        super().bind(model)
        q = model.joint_q.numpy()
        self.rest_q = q[self.q_span[0] : self.q_span[1]].copy()
        emb = self.embodiment
        if emb is not None and self.spec.hold_arm in getattr(emb, "arms", {}):
            index = emb.arms[self.spec.hold_arm]
            self.arm_coords = list(index.arm_coords)
            self.finger_coords = list(index.gripper_coords) + list(getattr(index, "gripper_follower_coords", []))

    def owned_bodies(self) -> np.ndarray:
        return self.all_bodies

    def apply_model_materials(self, model: newton.Model) -> None:
        """Stiff, mass-independent-enough contacts for everything the cable touches, and a stiff weld.

        Written here, after finalize and before the solver copies the model: the clips and the
        board are the scene's statics, built by the generic static entity, which has no way to
        say a MuJoCo solref."""
        from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW

        mj = getattr(model, "mujoco", None)
        if mj is None:
            return
        labels = [label.split("/")[-1] for label in model.shape_label]
        shape_body = model.shape_body.numpy()
        mine = set(self.all_bodies.tolist())
        cable = [s for s in range(model.shape_count) if int(shape_body[s]) in mine]
        fixtures = [s for s in range(model.shape_count) if labels[s].startswith("clip") or labels[s] == "board"]
        solref = mj.solref.numpy()
        mode = mj.solref_mode.numpy()
        solimp = mj.geom_solimp.numpy()
        for s in cable + fixtures:
            solref[s] = (CONTACT_TIMECONST, 1.0)
            mode[s] = SOLREF_MODE_RAW
            solimp[s] = CONTACT_SOLIMP
        mj.solref.assign(solref)
        mj.solref_mode.assign(mode)
        mj.geom_solimp.assign(solimp)
        if self.eq >= 0:
            eq_ref = mj.eq_solref.numpy()
            eq_imp = mj.eq_solimp.numpy()
            eq_ref[self.eq] = (CONTACT_TIMECONST, 1.0)
            eq_imp[self.eq] = CONTACT_SOLIMP
            mj.eq_solref.assign(eq_ref)
            mj.eq_solimp.assign(eq_imp)

    # ------------------------------------------------------------------ ready / reset
    def on_scene_ready(self, pipeline) -> None:
        if self.eq < 0:
            return
        self._pose_arm(pipeline)
        self._place(pipeline)
        self._write_relpose(pipeline)

    def reset(self, pipeline, episode, *, world: int = 0) -> None:
        del episode, world
        self._place(pipeline)

    def _pose_arm(self, pipeline) -> None:
        """IK once: the hand top-down over the last capsule, jaws across it.

        The robosuite Panda's fingers close along its tool x, which is world y at the arm's own
        roll -- along a cable running left to right, and the hand's long side to the camera. Its
        IK leaves the roll free (it keeps whatever the seed had), so the wrist is turned a
        quarter turn here, once: the jaws then close along world x, across the cable, and the
        hand is end-on to a camera on the bench's open side."""
        emb = self.embodiment
        ik = emb.iks[self.spec.hold_arm]
        model = pipeline.model
        state = pipeline.state_0
        q = state.joint_q.numpy().astype(np.float64)
        q[self.q_span[0] : self.q_span[1]] = self.rest_q
        bq = _fk_body_q(model, q)
        last = bq[int(self.bodies[-1])]
        target = np.asarray(last[:3], dtype=np.float64)
        quat = top_down_quat()
        solved = np.asarray(ik.solve_q_full(q, target, quat, iterations=200), dtype=np.float64)
        wrist = self.arm_coords[-1]
        lo = float(model.joint_limit_lower.numpy()[wrist])
        hi = float(model.joint_limit_upper.numpy()[wrist])
        turns = [solved[wrist] + d for d in (0.5 * math.pi, -0.5 * math.pi)]
        turns = [t for t in turns if lo + 0.2 < t < hi - 0.2] or turns
        solved[wrist] = max(turns, key=lambda t: min(t - lo, hi - t))
        pos, rot = ik.fk_ee(solved)
        jaw = quat_to_mat(np.asarray(rot, dtype=np.float64))[:, 0]
        print(f"[cable] hand posed on the cable's end: {np.linalg.norm(np.asarray(pos) - target) * 1000:.2f} mm, "
              f"jaws along ({jaw[0]:+.2f}, {jaw[1]:+.2f}, {jaw[2]:+.2f})", flush=True)
        self.start_arm_q = solved[self.arm_coords].copy()
        # The jaws shut on the cable: the Panda's pads stand 4.5 mm off its centre when the
        # prismatic pair reads zero, so an 8 mm cable is held at ~zero travel.
        self.finger_q = [float(self.spec.finger_q) * (1.0 if i % 2 == 0 else -1.0) for i in range(len(self.finger_coords))]

    def _place(self, pipeline) -> None:
        """The fresh cable at rest, the arm holding its end, targets to match."""
        for state in (pipeline.state_0, pipeline.state_1):
            q = state.joint_q.numpy()
            q[self.q_span[0] : self.q_span[1]] = self.rest_q
            if self.start_arm_q is not None:
                q[self.arm_coords] = self.start_arm_q
                for c, v in zip(self.finger_coords, self.finger_q):
                    q[c] = v
            state.joint_q.assign(q)
            qd = state.joint_qd.numpy()
            qd[:] = 0.0
            state.joint_qd.assign(qd)
        if self.start_arm_q is not None:
            t = pipeline.control.joint_target_q.numpy()
            t[self.arm_coords] = self.start_arm_q
            for c, v in zip(self.finger_coords, self.finger_q):
                t[c] = v
            pipeline.control.joint_target_q.assign(t)
        newton.eval_fk(pipeline.model, pipeline.state_0.joint_q, pipeline.state_0.joint_qd, pipeline.state_0)
        wp.copy(pipeline.state_1.body_q, pipeline.state_0.body_q)
        wp.copy(pipeline.state_1.body_qd, pipeline.state_0.body_qd)

    def _write_relpose(self, pipeline) -> None:
        """The weld's relative pose, read off the posed bodies: exactly satisfied at the start."""
        bq = pipeline.state_0.body_q.numpy()
        hand = bq[self.hand_body]
        cap = bq[int(self.bodies[-1])]
        th = wp.transform(wp.vec3(*hand[:3]), wp.quat(*hand[3:7]))
        tc = wp.transform(wp.vec3(*cap[:3]), wp.quat(*cap[3:7]))
        rel = wp.transform_multiply(wp.transform_inverse(th), tc)
        model = pipeline.model
        arr = model.mujoco.equality_constraint_relpose
        vals = arr.numpy()
        vals[self.eq] = np.array([*rel.p, *rel.q], dtype=np.float32)
        arr.assign(vals)
        pipeline.program.notify(int(newton.ModelFlags.CONSTRAINT_PROPERTIES))

    # ------------------------------------------------------------------ reading
    def centerline(self, body_q: np.ndarray) -> np.ndarray:
        """``SEGMENTS + 1`` points along the cable: each capsule's start, then the last one's end."""
        bq = np.asarray(body_q)[self.bodies]
        pos = bq[:, :3].astype(np.float64)
        axis = np.stack([quat_to_mat(q)[:, 2] for q in bq[:, 3:7]])
        starts = pos - axis * self.half[:, None]
        return np.vstack([starts, pos[-1] + axis[-1] * self.half[-1]])


def _fk_body_q(model: newton.Model, joint_q: np.ndarray) -> np.ndarray:
    state = model.state()
    q = wp.array(np.asarray(joint_q, dtype=np.float32), dtype=wp.float32, device=model.device)
    qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)
    newton.eval_fk(model, q, qd, state)
    return state.body_q.numpy()


def top_down_quat() -> np.ndarray:
    """Tool z straight down (the Panda's own home, untilted). Its IK leaves the roll about tool z
    free, so only "down" is honoured; ``ChainCable._pose_arm`` sets the roll."""
    return np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0])


# --------------------------------------------------------------------------- the scene
def _static(name: str, shape: ShapeSpec, xyz, *, color, friction=None, collide=True, quat=(0.0, 0.0, 0.0, 1.0)):
    return StaticSpec(name=name, shape=shape, xform=(*[float(v) for v in xyz], *quat), color=color,
                      friction=friction, collide=collide)


BENCH_ORIGIN = np.array([0.6, 0.0, 0.75])
"""bench_075's work-surface origin in the world (checked against the built scene by ``Router``)."""
VIEW_LOOK = np.array([LINE_X + 0.24, 0.5 * (SOCKET_Y + END_Y), BOARD_T])
"""Where the phones' camera looks, in the work-surface frame: ~25 cm in front of the board, so
the board sits in the top half of a phone's screen, above the sheet."""
VIEW_OFFSET = 1.2 * np.array([math.cos(math.radians(52.0)), 0.0, math.sin(math.radians(52.0))])
"""The phones' eye from there [m]: the bench's open side, 52 degrees down, 1.2 m off. With the
hand end-on (``ChainCable._pose_arm``) the arm rises behind the board rather than over it, and
a phone held upright sees the route from socket to end."""


def front_camera(width: int = 1280, height: int = 720) -> rss.CameraSpec:
    """The phones' view as a scene camera (the World model renders through it; so can a take)."""
    look = BENCH_ORIGIN + VIEW_LOOK
    eye = look + VIEW_OFFSET
    return rss.CameraSpec(name="front", width=width, height=height, fov_y=55.0, near=0.05, far=8.0,
                          eye=tuple(float(v) for v in eye), look_at=tuple(float(v) for v in look),
                          up=(0.0, 0.0, 1.0), channels=("rgb",))


def build_scene(*, physics: ArrangementSpec = PHYSICS) -> rss.Scene:
    scene = rss.Scene(cell="bench_075", physics=physics, name="cable_newton")
    scene.add(front_camera())
    scene.add(rss.robot("panda"))
    bx, by = 0.5 * (BOARD_X[0] + BOARD_X[1]), 0.5 * (BOARD_Y[0] + BOARD_Y[1])
    hx, hy = 0.5 * (BOARD_X[1] - BOARD_X[0]), 0.5 * (BOARD_Y[1] - BOARD_Y[0])
    scene.add(_static("board", ShapeSpec(kind="box", half_extents=(hx, hy, 0.5 * BOARD_T)), (bx, by, 0.5 * BOARD_T),
                      color=BOARD_COLOR, friction=MU_BOARD))
    sx, sy, sz = SOCKET_HALF
    # The socket the cable comes out of: drawn, not collided (the first capsule starts inside it).
    scene.add(_static("socket", ShapeSpec(kind="box", half_extents=SOCKET_HALF),
                      (LINE_X, SOCKET_Y - 0.004, BOARD_T + sz), color=SOCKET_COLOR, collide=False))
    for k in range(len(CLIP_Y)):
        c = clip_xy(k)
        scene.add(_static(f"clip{k + 1}", ShapeSpec(kind="capsule", radius=CLIP_R, half_height=CLIP_HALF),
                          (c[0], c[1], BOARD_T + CLIP_HALF + CLIP_R), color=CLIP_COLOR, friction=MU_CLIP))
    nodes = start_nodes()
    scene.add(ChainCableSpec(
        name="cable", nodes=tuple(tuple(float(v) for v in p) for p in nodes), radius=CABLE_R, density=DENSITY,
        bend_ke=BEND_KE, bend_kd=BEND_KD, color=CABLE_COLOR, joint=JOINT,
    ))
    return scene


def build_env(*, physics: ArrangementSpec | None = None):
    return build_scene(physics=physics or PHYSICS).build(control_mode="joint_position")


# ------------------------------------------------------------------------------ scoring
@dataclass
class ClipState:
    routed: bool
    s: float
    """Arc length [m] from the socket where the cable passes the clip (nan when it does not)."""
    off: float
    """Signed distance of that pass from the clip's axis, positive on the clip's marked side."""


def score(points: np.ndarray) -> tuple[int, list[ClipState]]:
    """How many clips are routed, in order from the socket, and each clip's reading.

    ``points`` is the cable's centreline in the work-surface frame. A clip is routed when the
    cable's first pass across the clip's line after the previous clip's pass is on the marked
    side, close by (within ROUTE_TOL of the post's surface) and below its top."""
    p = np.asarray(points, dtype=np.float64)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    top = BOARD_T + 2 * (CLIP_HALF + CLIP_R)
    states: list[ClipState] = []
    count, s_prev, chain = 0, -1.0, True
    for k, y in enumerate(CLIP_Y):
        found = None
        for i in range(len(p) - 1):
            y0, y1 = p[i, 1], p[i + 1, 1]
            if (y0 - y) * (y1 - y) > 0 or y0 == y1:
                continue
            t = (y - y0) / (y1 - y0)
            si = s[i] + t * seg[i]
            if si <= s_prev:
                continue
            q = p[i] + t * (p[i + 1] - p[i])
            found = (si, q)
            break
        if found is None:
            states.append(ClipState(False, float("nan"), float("nan")))
            chain = False
            continue
        si, q = found
        off = CLIP_SIDE[k] * (q[0] - LINE_X)
        ok = (off > CLIP_R) and (off < CLIP_R + CABLE_R + ROUTE_TOL) and (q[2] < top)
        states.append(ClipState(bool(ok), float(si), float(off)))
        if chain and ok:
            count += 1
            s_prev = si
        else:
            chain = False
    return count, states


# --------------------------------------------------------------------------------- driver
BALL_TOWARD, BALL_UP = 0.13, 0.025
"""The orange ball's offset from the hand [m]: toward the camera, just past the front of the hand
(end-on, it reaches 10 cm toward you), so the hand never hides it."""

WELD_ROWS_SEARCH = 24
PULL_LIMIT = 2.5
"""The most the hand may pull on the cable [N]. Dragging it round all three clips takes ~1."""

ATTRACT_SPEED = 0.11
"""Autopilot hand speed [m/s] -- a third of the arm's own cruise, so the cable is seen to follow."""
ATTRACT_HOLD = 4.0
"""Seconds the autopilot shows a routed cable before laying a fresh one."""


class Router:
    """The hand a finger drives, over the board at the carry height. The phone page's driver."""

    scene = "cable_newton"

    def __init__(self, env) -> None:
        self.env = env
        self.cable: ChainCable = env.scene.by_name["cable"]
        self.ik = env.arm("right")
        hz = float(env.request.physics.control_hz)
        self.hz = hz
        pos, quat = self.ik.ee_pose(env.state)
        self.q0 = np.asarray(quat, dtype=np.float64)
        """The start orientation, held: top-down, jaws across the cable (the IK leaves roll free)."""
        self.law = OscSetpoint(control_dt=1.0 / hz)
        self.law.reset(np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64))
        self.surface_z = float(env.handles.surface_z)
        self.origin = np.asarray(env.scene.by_name["board"].placed((0.0, 0.0, 0.0, 0, 0, 0, 1))[:3], dtype=np.float64)
        """The work-surface frame's origin in the world (the bench top's centre)."""
        if not np.allclose(self.origin, BENCH_ORIGIN, atol=1e-6):
            print(f"[cable] the bench's origin moved: {self.origin} (front camera assumes {BENCH_ORIGIN})", flush=True)
        self.carry_z = self.origin[2] + BOARD_T + CARRY_Z
        self.A = np.array([0.0, 1.0, 0.0])
        """Along the route (screen right)."""
        self.C = np.array([1.0, 0.0, 0.0])
        """Toward the viewer (off the clips' line, in front)."""
        self.ball = self.C * BALL_TOWARD + np.array([0.0, 0.0, BALL_UP])
        hand = np.asarray(pos, dtype=np.float64)
        hand[2] = self.carry_z
        self.cursor = hand + self.ball
        self.reset_request = False
        self.lap_done = False
        self.routed = 0
        self.best = 0
        self.states: list[ClipState] = []
        self.fingers = list(zip(self.cable.finger_coords, self.cable.finger_q))
        self.pull = 0.0
        """The cable's pull on the hand [N], read off the weld (one step old)."""
        solver = env.pipeline.program.mujoco_solver
        self._data = getattr(solver, "mjw_data", None)
        self._weld_rows: np.ndarray | None = None
        self._pull_host = None
        self._pull_event = None
        if self._data is not None and self.cable.eq >= 0:
            table = solver.mjc_eq_to_newton_eq.numpy()[0]
            hits = np.flatnonzero(table == self.cable.eq)
            if len(hits):
                n = WELD_ROWS_SEARCH
                kind = self._data.efc.type[0, :n].numpy()
                ident = self._data.efc.id[0, :n].numpy()
                rows = np.flatnonzero((kind == 0) & (ident == int(hits[0])))[:3]
                if len(rows) == 3:
                    self._weld_rows = rows
                    device = env.model.device
                    self._pull_host = wp.zeros(n, dtype=wp.float32, device="cpu", pinned=True)
                    self._pull_event = wp.Event(device)

    # -- geometry -----------------------------------------------------------------
    def world(self, dx: float, y: float) -> np.ndarray:
        return np.array([self.origin[0] + LINE_X + dx, self.origin[1] + y, self.carry_z])

    def clamp(self, ball: np.ndarray) -> np.ndarray:
        """A ball position whose hand stays over the board, at the carry height, off the clips."""
        hand = np.asarray(ball, dtype=np.float64) - self.ball
        x0, x1 = self.origin[0] + BOARD_X[0] + 0.03, self.origin[0] + BOARD_X[1] - 0.03
        y0, y1 = self.origin[1] + BOARD_Y[0] + 0.07, self.origin[1] + BOARD_Y[1] - 0.03
        hand[0] = float(np.clip(hand[0], x0, x1))
        hand[1] = float(np.clip(hand[1], y0, y1))
        hand[2] = self.carry_z
        for k in range(len(CLIP_Y)):
            c = self.origin[:2] + clip_xy(k)
            d = hand[:2] - c
            n = float(np.linalg.norm(d))
            if n < KEEP_OUT:
                d = d / n if n > 1e-6 else np.array([CLIP_SIDE[k], 0.0])
                hand[:2] = c + d * KEEP_OUT
        return hand + self.ball

    # -- the finger -----------------------------------------------------------------
    def touch(self) -> np.ndarray:
        return self.cursor.copy()

    def drag(self, cursor0, delta, up, right, away) -> np.ndarray:
        """Across the screen moves along the screen's horizontal; up the screen moves away from you."""
        move = (delta @ right) * right + (delta @ up) * away
        move[2] = 0.0
        return self.clamp(cursor0 + move)

    def release(self) -> np.ndarray:
        return self.cursor

    def idle_cursor(self) -> np.ndarray:
        return self.cursor

    # -- the loop -----------------------------------------------------------------
    def drive(self) -> None:
        env, ik = self.env, self.ik
        state = env.state
        pos, quat = ik.ee_pose(state)
        pos = np.asarray(pos, dtype=np.float64)
        set_pos, set_quat = self.law.advance(pos, quat, self.cursor - self.ball, self.q0)
        # A force-limited hand. A taut cable stops the hand while the servo keeps winding up
        # behind it -- the Panda's servo lags 0.15 s, so a blocked hand builds a 50 mm lead --
        # and that pull lands on 1.5 g capsules wrapped round a post: they sink in and the solve
        # blew up (measured, before this: NaN within a minute of random drags). The weld's own
        # constraint force is the cable's tension at the hand; over PULL_LIMIT the lead is cut
        # in proportion. It caps a sustained pull, not the jolt of a cable snapped taut.
        self.pull = self.weld_force()
        if self.pull > PULL_LIMIT:
            set_pos = pos + (set_pos - pos) * (PULL_LIMIT / self.pull)
            self.law.reset(set_pos, set_quat)
        ik.set_target(set_pos, set_quat)
        ik.solve_into(state.joint_q, env.control.joint_target_q)
        if self.fingers:
            t = env.control.joint_target_q
            arr = t.numpy()
            changed = False
            for c, v in self.fingers:
                if abs(arr[c] - v) > 1e-6:
                    arr[c] = v
                    changed = True
            if changed:
                t.assign(arr)

    def weld_force(self) -> float:
        """The translational force in the hand's weld [N], one control step old.

        MuJoCo puts equality rows first and their layout is fixed, so the rows are found once.
        The read is a copy into pinned memory queued behind the last step, collected on the
        next: a synchronous read here serialised the loop and cost ~0.8 ms a step."""
        if self._weld_rows is None:
            return 0.0
        wp.synchronize_event(self._pull_event)
        force = self._pull_host.numpy()[self._weld_rows]
        n = len(self._pull_host)
        wp.copy(self._pull_host, self._data.efc.force[0, :n])
        wp.record_event(self._pull_event)
        return float(np.linalg.norm(force))

    def hand(self) -> np.ndarray:
        return np.asarray(self.ik.ee_pose(self.env.state)[0], dtype=np.float64)

    def tcp(self) -> np.ndarray:
        return self.hand()

    def handle_point(self) -> np.ndarray:
        return self.hand() + self.ball

    def points(self) -> np.ndarray:
        return self.cable.centerline(self.env.state.body_q.numpy())

    def field(self) -> dict:
        self.routed, self.states = score(self.points() - self.origin)
        self.best = max(self.best, self.routed)
        return {"cleared_fraction": self.routed / len(CLIP_Y)}

    def new_stripe(self) -> None:
        """"New cable", or the autopilot's lap just ended: a fresh cable is a new episode."""
        self.lap_done = False
        self.reset_request = True

    def attract(self):
        """The autopilot: route the cable round the three clips at a walking pace, show it, start over."""
        wps = [self.cursor - self.ball]
        for k, y in enumerate(CLIP_Y):
            side = CLIP_SIDE[k]
            wps.append(self.world(side * 0.040, y - 0.035))
            wps.append(self.world(side * 0.040, y + 0.035))
        wps.append(self.world(0.0, END_Y))
        path = catmull_rom(np.asarray(wps)[:, :2], per_leg=12)
        cur = wps[0]
        for p2 in path[1:]:
            nxt = np.array([p2[0], p2[1], self.carry_z])
            n = max(1, int(np.linalg.norm(nxt - cur) / (ATTRACT_SPEED / self.hz)))
            for t in np.linspace(0.0, 1.0, n + 1)[1:]:
                yield self.clamp(cur + (nxt - cur) * t + self.ball)
            cur = nxt
        for _ in range(int(ATTRACT_HOLD * self.hz)):
            yield self.clamp(cur + self.ball)
        self.lap_done = True

    def broken_state(self) -> bool:
        bq = self.env.state.body_q.numpy()
        pts = self.cable.centerline(bq)
        if not np.isfinite(pts).all():
            return True
        if float(pts[:, 2].min()) < self.surface_z - 0.02:
            return True  # through the bench
        return float(np.linalg.norm(pts[-1] - self.hand())) > 0.06  # the hand lost the end


# ----------------------------------------------------------------------------------- map
class RouteMap:
    """The board from above, as the phone's sheet shows it: the route dashed, each clip pale,
    green once routed, the cable in white and the hand as the orange ball. Its contact map."""

    PX_PER_M = 520.0
    BG = np.array([30, 34, 40], dtype=np.float32)
    ROUTE = np.array([110, 116, 126], dtype=np.float32)
    CLIP = np.array([200, 210, 225], dtype=np.float32)
    DONE = np.array([60, 200, 120], dtype=np.float32)
    CABLE = np.array([235, 235, 240], dtype=np.float32)
    HAND = np.array([255, 140, 0], dtype=np.float32)

    def __init__(self, router: Router) -> None:
        self.r = router
        self.y0, self.y1 = BOARD_Y
        self.x0, self.x1 = LINE_X - 0.08, LINE_X + 0.20
        self.w = int(round((self.y1 - self.y0) * self.PX_PER_M))
        self.h = int(round((self.x1 - self.x0) * self.PX_PER_M))
        self.route = catmull_rom(route_points(), per_leg=40)

    def _px(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # screen right = +y, screen down = +x (toward you)
        u = ((y - self.y0) * self.PX_PER_M).astype(np.int64)
        v = ((x - self.x0) * self.PX_PER_M).astype(np.int64)
        return u, v

    def _dots(self, img, x, y, col, rad=1) -> None:
        u, v = self._px(np.asarray(x), np.asarray(y))
        for du in range(-rad, rad + 1):
            for dv in range(-rad, rad + 1):
                if du * du + dv * dv > rad * rad + 1:
                    continue
                uu, vv = u + du, v + dv
                ok = (uu >= 0) & (uu < self.w) & (vv >= 0) & (vv < self.h)
                img[vv[ok], uu[ok]] = col

    def caption(self) -> str:
        n = self.r.routed
        return "routed!" if n == len(CLIP_Y) else f"next: clip {n + 1}, {'in front' if CLIP_SIDE[n] > 0 else 'behind'}"

    def render(self) -> np.ndarray:
        img = np.empty((self.h, self.w, 3), dtype=np.float32)
        img[:] = self.BG
        dash = (np.arange(len(self.route)) // 6) % 2 == 0
        self._dots(img, self.route[dash, 0], self.route[dash, 1], self.ROUTE, rad=1)
        o = self.r.origin
        pts = self.r.points() - o
        dense = resample(pts, float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))), 160)
        self._dots(img, dense[:, 0], dense[:, 1], self.CABLE, rad=2)
        for k in range(len(CLIP_Y)):
            c = clip_xy(k)
            col = self.DONE if k < self.r.routed else self.CLIP
            self._dots(img, [c[0]], [c[1]], col, rad=int(CLIP_R * self.PX_PER_M) + 1)
        sx, sy = LINE_X, SOCKET_Y
        self._dots(img, [sx], [sy], np.array([170, 172, 178], dtype=np.float32), rad=7)
        h = self.r.hand() - o
        self._dots(img, [h[0]], [h[1]], self.HAND, rad=5)
        return img.clip(0, 255).astype(np.uint8)


def view_of(env) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(look-at, eye offset): from the bench's open side, high, so the weave reads left to right
    and the Panda reaches in from behind the board; aimed a little below the board's middle so
    the board sits above a phone's sheet."""
    o = np.asarray(env.scene.by_name["board"].placed((0.0, 0.0, 0.0, 0, 0, 0, 1))[:3], dtype=np.float64)
    return tuple((o + VIEW_LOOK).tolist()), tuple(VIEW_OFFSET.tolist())


def view(router: Router) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return view_of(router.env)


__all__ = [
    "CLIP_SIDE", "CLIP_Y", "PHYSICS", "ChainCable", "ChainCableSpec", "RouteMap", "Router", "build_env",
    "build_scene", "score", "start_nodes", "view", "view_of",
]
