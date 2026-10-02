"""Route the cable, from a browser: a SuperDex rod laid into three clips by a Panda.

The cable is SuperDex's Cosserat rod (``mochi_rod``: a centerline with a twist per node, stiff in
bending and in stretch the way a jacketed cable is), co-simulated beside Newton through
RoboSimStudio's SuperDex bridge: ``("rod", backend("superdex"))`` turns ``rss.RodSpec`` into
``EntityRodSuperdex``. The arm, the bench, the socket and the clips are Newton's (MuJoCo, on the
GPU). Every rigid shape the cable can touch is mirrored into the SuperDex scene as a collider and
posed there each engine step. This file uses RoboSimStudio's own API from outside the repository,
and changes three things at runtime, in this process only:

* **The mirrors are padded by the cable's radius.** SuperDex tests a rod's *centerline* samples
  against a collider's SDF, at the collider's threshold (1 mm, plus
  ``penaltyThresholdExtraPadding`` when the colliding actor is a point cloud). The library's
  mirrors leave the padding at 0, so a 2.5 mm cable rested with its axis 0.93 mm over the bench,
  1.57 mm of it inside the wood (measured: ``cable_sdx/probe1``). The rest height is exactly
  ``padding + 0.93 mm``, so :data:`MIRROR_PAD` puts the cable's skin on every surface: the bench,
  the clip prongs, the socket and the fingers.
* **The bench is mirrored as an exact plane**, not as the SDF of a 0.6 x 1.3 x 0.75 m box, which
  SuperDex bakes with ~0.2 m voxels; through it a cable was measured 11.6 mm under the top.
* **The hand holds the cable with engine node constraints, re-aimed every engine substep.** A
  mirror is a *static* actor in SuperDex, and the engine refuses one as a constraint target
  (``create_deformable_node_to_rigid_constraint``: "Invalid rigid actor type"). So the last two
  nodes are pinned (``pin_nodes``: a 30 kN/m node position constraint) to targets computed from
  the finger pads' own interpolated poses for each substep: the held end follows the jaws inside
  the substep, not once per control step. The socket end is pinned where it was built. See
  :class:`Holds` for why not boundary conditions.

Everything else is contact and friction: the cable drapes, slides on the bench and catches on the
clips' prongs. A clip is two Y prongs (an upright post, a horn leaning out) whose horns funnel a
cable lowered near the line down into the 8.4 mm channel between the posts.

An end-held cable is routed the way a hose is laid: lift it so it hangs off the bench, carry it
along the clips, lower it in. Dragged low it piles up and snags (measured: lowering past each
clip in turn routes none of them; lifting 20 cm and lowering at the far end seats clips 1, 2, 3
in that order).

**A finger moves the hand in 3D** (:meth:`Cable.drag`). The orange ball moves on the world plane
through the grab point that faces the camera most, met with the finger's ray so it stays under
the finger: from the opening view the vertical plane along the route (up lifts, across carries,
the hand's distance from the clips' line kept), from above the bench's plane, from the side the
vertical plane across the route. The ball's **shadow** on the bench is a second handle: it slides
the hand in, out and along at its height, so moving in and out needs no turn of the view. The
plane facing the camera itself (viser's drag plane, used as is) was tried first and lost: from
the opening view it leans back 32 deg, so up the screen is up *and in* and down is down *and out*,
and a hand taken high over the line and lowered straight down the screen came down 10-12 cm in
front of the clips -- 2/3 by mouse and 2/3 by touch where these planes routed 3/3 with the same
strokes (``cable3d/AB_*``), 2 of 7 scripted visitor strokes against 5 of 7 headless
(``cable3d/bench_visitor``). The hand is kept inside the measured reach, above the bench, over the
clips' tips (:data:`CLIP_KEEP`) and within the cable's length of the socket.

**Left alone it loops physically** (:meth:`Cable.attract`): route, rest, lift the cable out of the
clips and lay it back where a lap starts, rest, again. No reset, so nothing snaps: four laps
measured 3/3 after every route, 0/3 after every unroute, the cable back in the pre-roll's shape
each time (``cable3d/bench_laps``). A fresh cable -- "New cable", or the watchdog -- is the
pre-roll's moment restored from :class:`Snapshot` in 1-4 ms, exact to 0.000 mm.

:class:`Cable` is the phone page's driver, the interface ``wipe_phone.PhoneControls`` consumes;
:class:`CableTube`, :class:`CableMap` and :class:`HostDraw` are what the page draws and how it
draws it fast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import robosimstudio as rss
from robosimstudio.core.math import quat_to_mat
from robosimstudio.solvers import OscSetpoint
from robosimstudio.specs import ArrangementSpec, CollisionSpec, StaticSpec, backend
from robosimstudio.specs.asset import ShapeSpec

# ---------------------------------------------------------------------------------------------
# The cable
# ---------------------------------------------------------------------------------------------
CABLE_RADIUS = 0.0025
"""A 5 mm cable."""
CABLE_LENGTH = 0.45
CABLE_SEGMENTS = 45
"""1 cm elements: 46 nodes, 4 DoFs each (position and twist)."""
CABLE_DENSITY = 1500.0
"""kg/m^3: 29 g/m, 13 g for the cable (copper in a PVC jacket)."""
CABLE_YOUNG = 1.5e7
"""Effective modulus [Pa]: EI = 4.6e-4 N m^2, a cable that springs back from a bend, not a rope."""
CABLE_MU = 0.0625
"""``RodSpec.mu_cable``. The SuperDex rod's own Coulomb coefficient is its square root (0.25),
and a pair's coefficient is the geometric mean of both sides': 0.5 on the bench (1.0), 0.25 on the
clips (:data:`CLIP_FRICTION`). The cable grips the bench and slides down the prongs."""

MIRROR_PAD = CABLE_RADIUS - 0.00095
"""Padding on every mirrored collider: rest height = padding + 0.93 mm, so the cable's axis rests
one radius (2.48 mm) off every surface. See the module docstring."""

SDX_SUBSTEPS = 2
"""SuperDex steps per 1/60 s control step (8.3 ms). Contact is found at step endpoints. 3 and 4
cost 1.7-2x the tick (median 3.8-4.6 ms against 2.2-2.5 ms) and fixed nothing measured here."""

# ---------------------------------------------------------------------------------------------
# The layout, in the work-surface frame: the origin is the middle of bench_075's top (world
# (0.6, 0, 0.75)), +x points at the visitors' camera, the Panda stands at x = -0.6. The Panda
# reaches top-down to 0.1 mm over x -0.28..+0.14 and y -0.30..+0.30 (cable_sdx/probe2:
# OscSetpoint + solve_into, 2 cm and 10 cm over the bench), and every corner of the hand's box
# below at 1.4, 12 and 22 cm, holding the cable (cable3d/bench_reach).
# ---------------------------------------------------------------------------------------------
ROUTE_X = -0.06
"""The clips stand on this line along the bench (x = ROUTE_X), and the hand starts over it."""
CLIP_Y = (-0.10, -0.01, 0.08)
"""Clip centres along the route [m], 9 cm apart; the first 11 cm out of the socket, where the
socket's own heading has died out of a taut cable."""
SOCKET_Y = -0.21
"""Where the cable leaves the socket."""
SOCKET_YAW = math.radians(12.0)
"""The socket faces 12 deg off the route, toward the camera. The cable starts out straight along
that heading, in front of the clips; routing it bends it at the socket. At 20 deg, with clip 1
9 cm out, a taut cable still carried the socket's heading into clip 1's front prong and caught
on its outside (cable_sdx bench, "sloppy")."""

PRONG_R = 0.003
POST_HALF = 0.006
"""Each prong is a Y: a vertical post (3 mm radius, 12 mm of barrel, its top at 18 mm) ..."""
HORN_HALF = 0.008
HORN_LEAN = math.radians(26.0)
"""... and a horn leaning outward from the post's top (16 mm of barrel; the tips stand 35 mm tall,
the mouth 30 mm wide). The horns funnel a cable lowered near the line into the channel. The posts
are upright because an outward lean undercuts the bench: a cable dragged across the bench into
a leaning prong wedged under it (cable_sdx bench, "sloppy")."""
CHANNEL_HALF = 0.0042
"""Clear half-gap between the prongs at their foot [m]: a 5 mm cable in an 8.4 mm channel."""
CLIP_FRICTION = 0.25
SOCKET_HALF = (0.016, 0.012, 0.009)
"""Socket box half-extents [m]: along the cable, across it, up."""
CLIP_COLOR = (0.13, 0.42, 0.85)
CABLE_COLOR = (38, 40, 44)
"""A black cable: it reads on the wipe's light jigsaw mat (``backdrop``) and against the blue clips,
and the page's orange is the ball's alone. (White read on the old dark bench.)"""

# ---------------------------------------------------------------------------------------------
# The hand
# ---------------------------------------------------------------------------------------------
LAY_Z = 0.014
"""Lowest tool centre over the bench [m]: the pads (9.3 mm past it) clear the wood."""
Z_MAX = 0.22
"""Highest tool centre over the bench [m]."""
GRIP_GAP = 0.0075
"""Jaw gap holding the cable [m]: each pad 0.25 mm outside the cable's contact band."""
SEAT_DEPTH = 0.004
"""The held end sits this far past the tool centre along the approach axis [m], between the pads'
lower halves."""
BOX_Y = (-0.12, 0.25)
"""How far along the route the tool centre may go [m]: inside the measured reach, from clip 1 to
past the far end (further toward the socket a phone held upright loses the hand off its edge)."""
BOX_X = (ROUTE_X - 0.12, ROUTE_X + 0.12)
"""How far across the route the tool centre may go [m]: 12 cm inward (toward the robot, behind the
clips) and 12 cm outward (toward the visitors' camera, in front of them)."""
CLIP_CLEAR = 0.05
"""Lowest tool centre over a clip [m]: the fingertips (9.3 mm below it) pass over the prongs' tips."""
CLIP_KEEP = (0.032, 0.014)
"""Half-extents [m] (across the route, along it) of the zone over each clip where lower than
:data:`CLIP_CLEAR` the closed fingers (about +-14 x +-9 mm) would land on its prongs (+-17 x +-3 mm
at the horns' tips). Swept low across the clips' line, the hand otherwise ran into clip 3 and stuck
there 248 mm short of its target (``cable3d/bench_reach``)."""
CLIP_RAMP = 0.03
"""The floor rises to :data:`CLIP_CLEAR` over this distance [m] outside that zone, so a hand dragged
low across a clip rides up and over it rather than popping up."""
REACH_FROM_SOCKET = CABLE_LENGTH
"""How far the tool centre may get from the socket [m]: the cable's length, so a hand pulled past
it slides over a sphere instead, with the cable taut. Taut matters: a bow of amplitude A costs
only pi^2 A^2 / 4L of length, so 2.5 cm of slack let the cable lie in a 6 cm bow in front of the
clips, and lowering it there routed none of them."""
BALL_TOWARD, BALL_UP = 0.06, 0.02
"""The orange ball's offset from the tool centre [m]: toward the camera and up, clear of the hand."""
START = (ROUTE_X, 0.17, 0.04)
"""Where the pre-roll leaves the tool centre: the cable's end lifted 4 cm, at the far end of the
route plane, the rest of the cable still lying in front of the clips."""

AUTO_HIGH = 0.20
"""The autopilot's lift [m]: high enough that most of the cable hangs off the bench."""
AUTO_OUT = ROUTE_X + 0.11
"""Where the autopilot lays the cable back down [m, x]: 11 cm in front of the clips' line."""
HOLD_ROUTED, HOLD_START = 3.0, 2.0
"""Seconds the autopilot rests with the cable routed, and back where a lap starts."""
VIEW_ELEV = math.atan2(0.48, 0.78)
"""The opening view looks down on the route at this angle (32 deg) from the bench's open side."""


def surface_frame(sim) -> np.ndarray:
    """World position of the work-surface frame's origin."""
    return np.array([0.6, 0.0, float(sim.handles.surface_z)])


def _rz(angle: float) -> np.ndarray:
    """xyzw quaternion of a turn about +z."""
    return np.array([0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)])


def _quat_mul(a, b) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _quat_from_axis(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = math.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2.0)])


@dataclass(frozen=True)
class Clip:
    """One clip: two Y prongs either side of the route line, a channel between them."""

    index: int
    center: np.ndarray
    """Foot of the channel, work-surface frame."""
    along: np.ndarray
    """Channel direction (unit, horizontal)."""

    @property
    def across(self) -> np.ndarray:
        return np.array([-self.along[1], self.along[0], 0.0])


def clips() -> tuple[Clip, ...]:
    return tuple(Clip(i, np.array([ROUTE_X, y, 0.0]), np.array([0.0, 1.0, 0.0])) for i, y in enumerate(CLIP_Y))


def socket_exit() -> tuple[np.ndarray, np.ndarray]:
    """(where the cable leaves the socket, its heading there), work-surface frame."""
    d = np.array([math.sin(SOCKET_YAW), math.cos(SOCKET_YAW), 0.0])
    return np.array([ROUTE_X, SOCKET_Y, 0.0]), d


def cable_nodes() -> np.ndarray:
    """The cable as built, work-surface frame: straight out of the socket along its heading, on the
    bench. Straight because a rod's built shape is its rest shape (its natural curvature)."""
    p0, d = socket_exit()
    s = np.linspace(0.0, CABLE_LENGTH, CABLE_SEGMENTS + 1)
    nodes = p0[None, :] + s[:, None] * d[None, :]
    nodes[:, 2] = CABLE_RADIUS + 0.0002
    return nodes


def prong_specs(clip: Clip) -> list[StaticSpec]:
    """The two Y prongs of one clip, as static capsules: an upright post, a horn leaning out."""
    out = []
    up = np.array([0.0, 0.0, 1.0])
    for side, sign in (("a", 1.0), ("b", -1.0)):
        foot = clip.center + clip.across * sign * (CHANNEL_HALF + PRONG_R) + up * PRONG_R
        post = foot + up * POST_HALF
        knee = foot + up * 2.0 * POST_HALF
        q = _quat_from_axis(clip.along, -sign * HORN_LEAN)  # tips lean toward sign * across
        horn = knee + (quat_to_mat(q) @ up) * HORN_HALF
        for part, centre, half, quat in (("post", post, POST_HALF, (0.0, 0.0, 0.0, 1.0)), ("horn", horn, HORN_HALF, q)):
            out.append(
                StaticSpec(
                    name=f"clip{clip.index + 1}{side}{part}",
                    shape=ShapeSpec(kind="capsule", radius=PRONG_R, half_height=half),
                    xform=(*[float(v) for v in centre], *[float(v) for v in quat]),
                    friction=CLIP_FRICTION,
                    color=CLIP_COLOR,
                )
            )
    return out


def clip_base_spec(clip: Clip) -> StaticSpec:
    """A thin plate under each clip, for the eye: it collides with nothing the cable reaches."""
    return StaticSpec(
        name=f"clip{clip.index + 1}base",
        shape=ShapeSpec(kind="box", half_extents=(0.014, 0.008, 0.0006)),
        xform=(float(clip.center[0]), float(clip.center[1]), 0.0002, 0.0, 0.0, 0.0, 1.0),
        collide=False,
        color=(0.10, 0.30, 0.62),
    )


def socket_spec() -> StaticSpec:
    p0, d = socket_exit()
    hx, hy, hz = SOCKET_HALF
    centre = p0 - d * (hx + 0.0035)  # 3.5 mm off the face: outside its 2.48 mm contact band
    q = _rz(math.pi / 2.0 - SOCKET_YAW)  # the box's local x along the cable's heading
    return StaticSpec(
        name="socket",
        shape=ShapeSpec(kind="box", half_extents=(hx, hy, hz)),
        xform=(float(centre[0]), float(centre[1]), hz, *[float(v) for v in q]),
        friction=0.5,
        color=(0.16, 0.16, 0.18),
    )


PHYSICS = ArrangementSpec(
    control_hz=60.0,
    substeps=4,
    backends=(
        (
            "*",
            backend(
                "mujoco",
                use_mujoco_contacts=True,
                nconmax=512,
                njmax=1024,
                iterations=50,
                ls_iterations=20,
                cone="elliptic",
                impratio=10.0,
            ),
        ),
        ("rod", backend("superdex", substeps=SDX_SUBSTEPS)),
    ),
    collision=CollisionSpec(),
)
"""The arm on MuJoCo at 4 substeps (a rigid-only island: GENERIC's 16 are for cloth), the cable on
SuperDex."""


# ---------------------------------------------------------------------------------------------
# The mirrors, padded
# ---------------------------------------------------------------------------------------------
def install_mirror_padding(pad: float = MIRROR_PAD) -> None:
    """Make the SuperDex bridge build its mirrors with ``pad`` of contact padding. This process only.

    A copy of ``robosimstudio.engines.superdex.mirror.build_mirror_actors`` with one more
    ``ContactParams`` field, put where the bridge looks it up. Idempotent."""
    import robosimstudio.engines.superdex.bridge as bridge_mod
    import robosimstudio.engines.superdex.mirror as mirror_mod

    if getattr(bridge_mod.build_mirror_actors, "_cable_pad", None) == pad:
        return

    def build_mirror_actors(
        physics, scenes, model, *, shapes, layer="mirror", contact_smoothing=5.0e-4, friction_falloff=1.0e-2
    ):
        per_world = list(scenes) if isinstance(scenes, (list, tuple)) else [scenes]
        out = []
        for geom in mirror_mod.iter_mirrors(model, shapes):
            local = geom.local
            placed = local.copy()
            if _is_bench_top(geom):
                # The bench as an exact plane at its top face. Its SDF bakes at a quarter of the
                # box's mean edge -- ~0.2 m voxels on a 0.6 x 1.3 x 0.75 m table -- and a cable
                # pushed down by a stalled solve was measured 11.6 mm under it.
                handle = physics.create_plane_shape(normal=[0.0, 0.0, 1.0], distance=0.0)
                top = quat_to_mat(np.asarray(local[3:7], dtype=np.float64)) @ np.array([0.0, 0.0, float(geom.scale[2])])
                placed[:3] = np.asarray(local[:3], dtype=np.float64) + top
                collider = physics.ColliderType.AUTO
            else:
                handle = mirror_mod.make_collider_shape(physics, geom)
                if handle is None:
                    continue
                collider = physics.ColliderType.AUTO if geom.kind in ("plane", "sphere") else physics.ColliderType.SDF
            actor = per_world[geom.world % len(per_world)].create_rigid_actor(
                name=f"mirror:{geom.shape}" if len(per_world) == 1 else f"mirror:{geom.shape}#w{geom.world}",
                layer=layer,
                shape=handle,
                is_static=True,
                collider_type=collider,
                world_from_local=physics.TransformRT(
                    rotation=physics.Quaternion(*[float(v) for v in placed[3:7]]),
                    translation=[float(v) for v in placed[:3]],
                ),
                contact=physics.ContactParams(
                    coulomb_friction_coefficient=geom.mu,
                    penalty_smoothing_half_distance=float(contact_smoothing),
                    friction_falloff_vel=float(friction_falloff),
                    penalty_threshold_extra_padding=float(pad),
                ),
            )
            out.append(
                mirror_mod.MirrorActor(
                    shape=geom.shape,
                    body=geom.body,
                    actor=actor,
                    local=local,
                    two_way=geom.two_way,
                    world=geom.world,
                    prev=placed.copy(),
                )
            )
        return out

    build_mirror_actors._cable_pad = pad
    bridge_mod.build_mirror_actors = build_mirror_actors


def _is_bench_top(geom) -> bool:
    """A world-fixed box this big is the cell's table: the cable only ever meets its top."""
    return geom.body < 0 and geom.kind == "box" and float(np.max(geom.scale)) > 0.2


def build_scene() -> rss.Scene:
    scene = rss.Scene(cell="bench_075", physics=PHYSICS, name="cable_route")
    scene.add(rss.robot("panda"))
    scene.add(socket_spec())
    for clip in clips():
        scene.add(clip_base_spec(clip))
        for spec in prong_specs(clip):
            scene.add(spec)
    scene.add(
        rss.RodSpec(
            name="cable",
            nodes=tuple(tuple(float(v) for v in p) for p in cable_nodes()),
            radius=CABLE_RADIUS,
            density=CABLE_DENSITY,
            young_modulus=CABLE_YOUNG,
            mu_cable=CABLE_MU,
            color=tuple(c / 255.0 for c in CABLE_COLOR),
        )
    )
    return scene


def build_env():
    install_mirror_padding()
    return build_scene().build(control_mode="joint_position")


# ---------------------------------------------------------------------------------------------
# Holding nodes: engine node constraints, re-aimed every SuperDex substep
# ---------------------------------------------------------------------------------------------
PIN_STIFFNESS, PIN_DAMPING = 3.0e4, 5.0
"""The node constraints' spring [N/m] and damper [N s/m]: 5 N of cable tension moves a held node
0.17 mm."""


class Holds:
    """Held cable nodes: the socket end (fixed) and the held end (riding the finger pads).

    Both are node position constraints the bridge makes (``pin_nodes``), so a reset's
    ``restore_rest`` removes them with its own pins. The held ones are re-aimed by wrapping the
    bridge's per-substep ``_drive_world``: before each engine step their targets are set from the
    two pad mirrors' poses for *that* substep (the bridge interpolates those across the control
    step), so the cable's end follows the jaws inside the step. Twist stays free.

    Boundary conditions (``add_boundary_condition_dofs_world``, cleared and added again each
    substep) hold a node exactly and were tried first: their moving targets broke the engine's
    preconditioned CG ("Zero Preconditioner-dot product", "Linear solver didn't converge", once a
    "Solution explosion"), 10-82 such log lines per stress run against 0 with these pins, three
    runs each (cable_sdx bench, "stress"). Pinned, the held end stays 4.0 mm from the tool centre
    (its seat) through fast drags."""

    def __init__(self, sim, name: str = "cable") -> None:
        ref = sim[name]
        self.name = name
        self.bridge = ref.entity.bridge
        self.entry = next(e for e in self.bridge.actors if e.get("entity", e["name"]) == name)
        self.start = int(self.entry["start"])
        self.count = int(self.entry["count"])
        self.fixed: dict[int, np.ndarray] = {}
        self.held: dict[int, np.ndarray] = {}
        """Node -> position in the pad frame."""
        self.goal: dict[int, np.ndarray] = {}
        """Node -> where in the pad frame a held node is being drawn to (its seat in the jaws)."""
        self.seat_rate = 0.12
        """Fraction of the way to :attr:`goal` per engine substep: ~15 substeps, 1/8 s."""
        self.pins: dict[str, int] = {}
        """"fixed"/"held" -> the bridge's pin id."""
        self.pad_a, self.pad_b = self._pads(sim)
        self._orig = self.bridge._drive_world
        self.bridge._drive_world = self._drive_world

    def _pads(self, sim):
        """The two finger-pad mirrors: the hand's tip bodies (``..._tip`` on the Franka hand)."""
        names = getattr(sim.model, "body_key", None) or getattr(sim.model, "body_label", None) or []
        tips = [m for m in self.bridge.mirrors if m.body >= 0 and str(names[m.body]).endswith("_tip")]
        if len(tips) == 2:
            return tips[0], tips[1]
        moving = sorted({m.body for m in self.bridge.mirrors if m.body >= 0})
        pads = [m for m in self.bridge.mirrors if m.body >= 0]
        return next(m for m in pads if m.body == moving[1]), next(m for m in pads if m.body == moving[-1])

    @staticmethod
    def _frame(pose_a: np.ndarray, pose_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Midway between the pads, turned with pad a."""
        origin = 0.5 * (np.asarray(pose_a[:3]) + np.asarray(pose_b[:3]))
        return origin, quat_to_mat(np.asarray(pose_a[3:7], dtype=np.float64))

    def pad_frame(self, state=None) -> tuple[np.ndarray, np.ndarray]:
        """The pad frame now, from the Newton state (the composition the bridge uses)."""
        from robosimstudio.engines.pose import compose_many

        body_q = (state if state is not None else self.bridge.pipeline.state_0).body_q.numpy()
        poses = compose_many(
            np.asarray([body_q[self.pad_a.body], body_q[self.pad_b.body]], dtype=np.float64),
            np.asarray([self.pad_a.local, self.pad_b.local], dtype=np.float64),
        )
        return self._frame(poses[0], poses[1])

    def nodes_world(self) -> np.ndarray:
        """Every node's position now, from the engine (not the Newton copy)."""
        return np.asarray(self.entry["read"](), dtype=np.float64).reshape(-1, 3)

    def _pin(self, key: str, targets: dict[int, np.ndarray]) -> None:
        nodes = sorted(targets)
        self.pins[key] = self.bridge.pin_nodes(
            self.name,
            np.asarray([self.start + n for n in nodes], dtype=np.int64),
            np.asarray([targets[n] for n in nodes], dtype=np.float64),
            stiffness=PIN_STIFFNESS,
            damping=PIN_DAMPING,
        )

    # -- taking and letting go
    def fix(self, nodes) -> None:
        """Pin ``nodes`` where they are, for good: the socket end."""
        q = self.nodes_world()
        self.fixed = {int(n): q[int(n)].copy() for n in nodes}
        self._pin("fixed", self.fixed)

    def grip(self, nodes, state=None, *, seat=None) -> None:
        """Take ``nodes`` where they are; with ``seat`` (node -> world point), draw them there."""
        origin, rot = self.pad_frame(state)
        q = self.nodes_world()
        self.held = {int(n): rot.T @ (q[int(n)] - origin) for n in nodes}
        self.goal = {int(n): rot.T @ (np.asarray(p, dtype=np.float64) - origin) for n, p in (seat or {}).items()}
        self._pin("held", {int(n): q[int(n)] for n in nodes})

    def clear(self) -> None:
        """Everything let go, and the engine told so now: a restore must not meet a stale pin."""
        self.fixed, self.held, self.goal = {}, {}, {}
        for pin in self.pins.values():
            self.bridge.unpin(pin)
        self.pins = {}

    # -- per engine substep
    def _drive_world(self, world, posed, step_dt, teleport) -> None:
        self.apply(posed)
        self._orig(world, posed, step_dt, teleport)

    def apply(self, posed) -> None:
        if not self.held or "held" not in self.pins:
            return
        pa = pb = None
        for mirror, pose in posed:
            if mirror is self.pad_a:
                pa = pose
            elif mirror is self.pad_b:
                pb = pose
        if pa is None or pb is None:
            return
        origin, rot = self._frame(pa, pb)
        for n, goal in self.goal.items():
            if n in self.held:
                self.held[n] = self.held[n] + (goal - self.held[n]) * self.seat_rate
        nodes = sorted(self.held)
        try:
            self.bridge.set_pin_targets(self.pins["held"], np.asarray([origin + rot @ self.held[n] for n in nodes]))
        except (KeyError, IndexError):  # a reset took the pin away (restore_rest unpins all)
            self.held, self.goal = {}, {}
            self.pins.pop("held", None)


def grip_quat(sim) -> np.ndarray:
    """Top-down, the jaws turned to close across the route (the home pose closes along world y)."""
    _pos, home = sim.arm().home_pose
    return _quat_mul(_rz(math.pi / 2.0), home)


def routed_count(q: np.ndarray, origin: np.ndarray, route) -> tuple[int, list[bool]]:
    """How many clips hold the cable, counted in order from the socket; and which ones do.

    A clip holds the cable when a cable segment crosses the clip's channel plane between the prongs
    (within the channel's half-width plus 2 mm of the line) and below 12 mm. The count takes the
    clips in route order and counts each that holds the cable further along it than the last one
    counted: a cable doubling back through a clip is not routed there. A clip that does not hold it
    is passed over, not the end of the count -- laid in from the side, a cable can sit in clips 2
    and 3 and miss 1, and the page's map then shows two clips holding it (``cable3d/bench_side``,
    "behind_then_out": the first rule read that 0/3)."""
    loc = q - origin
    held: list[float | None] = []
    for clip in route:
        rel = loc - clip.center
        a = rel @ clip.along
        hit = None
        for i in range(len(loc) - 1):
            if (a[i] <= 0.0 < a[i + 1]) or (a[i + 1] <= 0.0 < a[i]):
                t = a[i] / (a[i] - a[i + 1])
                p = rel[i] + t * (rel[i + 1] - rel[i])
                if abs(p @ clip.across) <= CHANNEL_HALF + 0.002 and p[2] <= 0.012:
                    hit = i + t
                    break
        held.append(hit)
    count, last = 0, -1.0
    for hit in held:
        if hit is not None and hit > last:
            count, last = count + 1, hit
    return count, [h is not None for h in held]


def pre_roll(sim, holds: Holds, *, verbose: bool = False):
    """Hold the socket end; the Panda takes the cable's free end, lifts it and brings it to
    :data:`START`. A generator of control steps (335 of them, 5.6 s of sim; ~3 s on the page)."""
    ik, g = sim.arm(), sim.gripper()
    hz = float(sim.request.physics.control_hz)
    origin = surface_frame(sim)
    holds.clear()
    holds.fix((0, 1))
    q = holds.nodes_world()
    end = q[-1].copy()
    quat = grip_quat(sim)
    pos, cur = ik.ee_pose(sim.state)
    law = OscSetpoint(control_dt=1.0 / hz)
    law.reset(pos, cur)
    surface_z = float(sim.handles.surface_z)

    def go(target, steps, grip):
        g.command = grip
        for _ in range(steps):
            p, qq = ik.ee_pose(sim.state)
            ik.set_target(*law.advance(p, qq, target, quat))
            ik.solve_into(sim.state.joint_q, sim.control.joint_target_q)
            g.apply(sim.control.joint_target_q)
            sim.step_teleop()
            yield

    above = end.copy()
    above[2] = surface_z + 0.08
    yield from go(above, 100, g.command_for_width(0.035))
    low = end.copy()
    low[2] = surface_z + LAY_Z
    yield from go(low, 60, g.command_for_width(0.035))
    yield from go(low, 45, g.command_for_width(GRIP_GAP))
    # Take the last two nodes and draw them into the jaws: the end SEAT_DEPTH past the tool
    # centre, its neighbour one element back along the route -- the cable leaves the fingers
    # heading back toward the socket.
    tcp, tquat = ik.ee_pose(sim.state)
    out = quat_to_mat(tquat) @ np.array([0.0, 0.0, 1.0])  # the approach axis, out of the fingers
    q = holds.nodes_world()
    end_at = tcp + out * SEAT_DEPTH
    seg = float(np.linalg.norm(q[-1] - q[-2]))
    back = np.array([0.0, -1.0, 0.0])
    holds.grip((holds.count - 1, holds.count - 2), seat={holds.count - 1: end_at, holds.count - 2: end_at + back * seg})
    yield from go(low, 20, g.command_for_width(GRIP_GAP))
    up = low.copy()
    up[2] = surface_z + START[2]
    yield from go(up, 40, g.command_for_width(GRIP_GAP))
    yield from go(origin + np.asarray(START), 70, g.command_for_width(GRIP_GAP))
    if verbose:
        tcp = ik.ee_pose(sim.state)[0]
        print(
            f"[cable] picked: end {np.round(holds.nodes_world()[-1] - origin, 4)} tcp {np.round(tcp - origin, 4)}",
            flush=True,
        )


# ---------------------------------------------------------------------------------------------
# The hand, driven by a finger
# ---------------------------------------------------------------------------------------------
class Cable:
    """The hand holding the cable's end. The phone page's driver interface."""

    scene = "cable"

    def __init__(self, sim, holds: Holds) -> None:
        self.sim = sim
        self.holds = holds
        self.ik = sim.arm()
        self.hz = float(sim.request.physics.control_hz)
        self.origin = surface_frame(sim)
        self.surface_z = float(sim.handles.surface_z)
        self.quat = grip_quat(sim)
        self.law = OscSetpoint(control_dt=1.0 / self.hz)
        self.ball = np.array([BALL_TOWARD, 0.0, BALL_UP])
        exit_p, _d = socket_exit()
        self.socket = self.origin + exit_p
        self.idx = np.asarray(sim["cable"].particles, dtype=np.int64)
        self.clips = clips()
        self.reset_request = False
        self.on_new = None
        """``(why) -> None``: what "New cable" calls (the page's instant fresh cable). Without one,
        :meth:`new_stripe` asks the page for a full restart through :attr:`reset_request`."""
        self.grab: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        """(eye, look-at, grab point) of the drag in progress: where the finger's ray comes from."""
        self.grab_kind = "ball"
        """What the finger took: ``ball`` (moves the hand on the world plane facing the camera) or
        ``shadow`` (the ball's shadow on the bench: slides the hand over the bench)."""
        self.on_lap = None
        """``(what, routed) -> None``, told at the end of each autopilot rest: ``"routed"`` or
        ``"unrouted"`` and how many clips hold the cable then."""
        self.reseed()

    def reseed(self) -> None:
        """Start over from wherever the hand is now: after the pre-roll, and after a restore."""
        pos, quat = self.ik.ee_pose(self.sim.state)
        self.law.reset(pos, quat)
        self.cursor = self.clamp(np.asarray(pos, dtype=np.float64)) + self.ball
        """The orange ball. The hand goes to ``cursor - ball``."""
        self.reset_request = False
        self.grab = None
        self.grab_kind = "ball"
        self.routed = 0
        self.held_by: list[bool] = [False] * len(self.clips)

    # -- geometry
    def world(self, p) -> np.ndarray:
        return self.origin + np.asarray(p, dtype=np.float64)

    def local(self, p) -> np.ndarray:
        return np.asarray(p, dtype=np.float64) - self.origin

    def clamp(self, hand: np.ndarray) -> np.ndarray:
        """A tool-centre target kept inside the box over the clips (the measured reach, above the
        bench, 12 cm either side of the clips' line) and within the cable's length of the socket."""
        loc = self.local(hand)
        loc[0] = float(np.clip(loc[0], *BOX_X))
        loc[1] = float(np.clip(loc[1], *BOX_Y))
        loc[2] = float(np.clip(loc[2], self.floor_at(loc[0], loc[1]), Z_MAX))
        sock = self.local(self.socket)
        d = loc - sock
        if float(np.linalg.norm(d)) > REACH_FROM_SOCKET:
            # Pulled past the cable's length: slide in toward the socket over the bench, keeping the
            # height -- a taut cable stops the hand going further out.
            span = math.sqrt(max(REACH_FROM_SOCKET**2 - d[2] * d[2], 0.0))
            flat = float(np.hypot(d[0], d[1]))
            if flat > 1e-9:
                loc[:2] = sock[:2] + d[:2] * (span / flat)
            loc[2] = max(loc[2], self.floor_at(loc[0], loc[1]))
        return self.world(loc)

    def floor_at(self, x: float, y: float) -> float:
        """The lowest tool centre at (``x``, ``y``) [m over the bench]: :data:`LAY_Z`, rising to
        :data:`CLIP_CLEAR` over each clip (:data:`CLIP_KEEP`, :data:`CLIP_RAMP`)."""
        rise = 0.0
        for clip in self.clips:
            dx = max(abs(x - float(clip.center[0])) - CLIP_KEEP[0], 0.0)
            dy = max(abs(y - float(clip.center[1])) - CLIP_KEEP[1], 0.0)
            u = 1.0 - min(math.hypot(dx, dy) / CLIP_RAMP, 1.0)
            rise = max(rise, u * u * (3.0 - 2.0 * u))
        return LAY_Z + (CLIP_CLEAR - LAY_Z) * rise

    def ball_at(self, x: float, y: float, z: float) -> np.ndarray:
        """The ball for a tool centre at (``x``, ``y``, ``z``) in the work-surface frame."""
        return self.clamp(self.world((x, y, z))) + self.ball

    def particles(self) -> np.ndarray:
        return self.sim.state.particle_q.numpy()[self.idx].astype(np.float64)

    # -- the finger
    def touch(self) -> np.ndarray:
        return self.cursor.copy()

    def drag(self, cursor0, delta, up, right, away) -> np.ndarray:
        """The hand follows the finger in 3D. The page passes ``delta`` on viser's drag plane (the
        plane facing the camera through the grab point); the finger's ray through it is met with a
        world plane through the grab point instead -- the bench's for the shadow, and for the ball
        the world-axis plane that faces the camera most (see the module docstring for why not the
        drag plane itself) -- so what was grabbed stays under the finger. Turning the view turns
        the ball's plane. Clamped to :meth:`clamp`'s box."""
        del up, right, away
        if self.grab is None:  # no camera known: the drag plane itself
            move = np.asarray(delta, dtype=np.float64)
        elif self.grab_kind == "shadow":
            move = self._axis_move(delta, np.array([0.0, 0.0, 1.0]))
        else:
            move = self._axis_move(delta)
        return self.clamp(cursor0 - self.ball + move) + self.ball

    def _axis_move(self, delta, normal=None) -> np.ndarray:
        """The pointer's ray met with a world-axis plane through the grab point -- ``normal``'s, or
        the one that faces the camera most: from the front a vertical plane along the route, from
        the side one across it, from above the bench's own plane."""
        eye, _look, origin = self.grab
        sight = origin - eye
        if normal is not None:
            n = np.asarray(normal, dtype=np.float64)
        else:
            n = np.zeros(3)
            n[int(np.argmax(np.abs(sight)))] = 1.0
        ray = origin + np.asarray(delta, dtype=np.float64) - eye
        den = float(n @ ray)
        if abs(den) < 1e-6:
            return np.zeros(3)
        t = float(n @ (origin - eye)) / den
        if t <= 0.0:
            return np.zeros(3)
        move = eye + t * ray - origin
        dist = float(np.linalg.norm(move))
        return move if dist < 1.0 else move / dist  # a ray grazing the plane: no flight to infinity

    def release(self) -> np.ndarray:
        self.grab, self.grab_kind = None, "ball"
        return self.cursor  # the hand holds the cable where the finger left it

    def shadow_point(self, ball: np.ndarray) -> np.ndarray:
        """Where the ball's shadow lies: straight under it, on the bench."""
        return np.array([float(ball[0]), float(ball[1]), self.surface_z + 0.0015])

    def idle_cursor(self) -> np.ndarray:
        return self.cursor

    # -- the loop
    def drive(self) -> None:
        state = self.sim.state
        pos, quat = self.ik.ee_pose(state)
        set_pos, set_quat = self.law.advance(pos, quat, self.clamp(self.cursor - self.ball), self.quat)
        # The way there keeps over the clips too, not only the target: a target that jumps across
        # one (a quick flick) would otherwise draw the hand straight through it at bench height.
        loc = self.local(set_pos)
        floor = self.floor_at(float(loc[0]), float(loc[1]))
        if loc[2] < floor:
            loc[2] = floor
            set_pos = self.world(loc)
            self.law.reset(set_pos, set_quat)
        self.ik.set_target(set_pos, set_quat)
        self.ik.solve_into(state.joint_q, self.sim.control.joint_target_q)

    def tcp(self) -> np.ndarray:
        return np.asarray(self.ik.ee_pose(self.sim.state)[0], dtype=np.float64)

    def handle_point(self) -> np.ndarray:
        return self.tcp() + self.ball

    def field(self) -> dict:
        if self.reset_request:
            # A fresh cable is on its way: the page would otherwise cheer a route in the second
            # before its reset ("Routed! Tap New cable to go again").
            return {"cleared_fraction": 0.0}
        self.routed, self.held_by = routed_count(self.particles(), self.origin, self.clips)
        return {"cleared_fraction": self.routed / float(len(self.clips))}

    def new_stripe(self) -> None:
        """For "New cable": a fresh cable, from the page's snapshot when it has one."""
        if self.on_new is not None:
            self.on_new("New cable.")
        else:
            self.reset_request = True

    # -- the autopilot
    def attract(self):
        """The autopilot, for as long as nobody drives: route the cable, rest, take it out of the
        clips and lay it back where a lap starts, rest, and again -- every move physical, no reset.
        A generator of ball positions, one per control step, that never ends."""
        hand = self.local(self.cursor - self.ball)
        start = np.asarray(START, dtype=np.float64)
        if self.routed > 0 or float(np.linalg.norm(hand - start)) > 0.03:
            hand = yield from self._unroute(hand)  # a visitor left it somewhere: tidy up first
            hand = yield from self._hold(hand, HOLD_START)
        while True:
            hand = yield from self._route(hand)
            hand = yield from self._hold(hand, HOLD_ROUTED)
            if self.on_lap is not None:
                self.on_lap("routed", self.routed)
            hand = yield from self._unroute(hand)
            hand = yield from self._hold(hand, HOLD_START)
            if self.on_lap is not None:
                self.on_lap("unrouted", self.routed)

    def _glide(self, keys, hand):
        """Ease the tool centre through ``keys`` ((x, y, z), seconds), each leg from the last. The
        legs run between the keys as given and every point is clamped on the way, so a leg aimed
        past the cable's length rides the sphere with the cable taut. Returns where it ended."""
        cur = np.asarray(hand, dtype=np.float64)
        for key, sec in keys:
            nxt = np.asarray(key, dtype=np.float64)
            n = max(2, int(sec * self.hz))
            for u in np.linspace(0.0, 1.0, n):
                s = u * u * (3.0 - 2.0 * u)
                p = cur + (nxt - cur) * s
                yield self.ball_at(float(p[0]), float(p[1]), float(p[2]))
            cur = nxt
        return self.local(self.clamp(self.world(cur)))

    def _hold(self, hand, sec: float):
        ball = self.ball_at(*[float(v) for v in hand])
        for _ in range(max(1, int(sec * self.hz))):
            yield ball
        return hand

    def _route(self, hand):
        """Lift the cable off the bench, carry it to the far end (the reach clamp stops it where the
        cable goes taut), lower it into the clips along that sphere.

        Straight up and down over the line: lowered from behind it, along the opening view's drag
        plane, the cable kept the socket's heading at clip 1 and lay outside it every lap (2/3,
        ``cable3d/bench_laps_plane``); lowered straight down, 3/3 every lap."""
        far = BOX_Y[1]
        keys = (
            ((ROUTE_X, float(hand[1]), AUTO_HIGH), 1.4),
            ((ROUTE_X, far, AUTO_HIGH), 1.0),
            ((ROUTE_X, far, AUTO_HIGH), 0.5),
            ((ROUTE_X, far, 0.07), 2.4),
            ((ROUTE_X, far, LAY_Z), 1.2),
        )
        return (yield from self._glide(keys, hand))

    def _unroute(self, hand):
        """Lift the cable straight up out of the clips, swing it out in front of them, lay it down
        there and bring the end back to :data:`START`: where the pre-roll leaves a fresh cable."""
        far = BOX_Y[1]
        keys = (
            ((float(hand[0]), float(hand[1]), AUTO_HIGH), 1.6),
            ((AUTO_OUT, far, AUTO_HIGH), 1.4),
            ((AUTO_OUT, far, 0.025), 2.0),
            (START, 1.6),
        )
        return (yield from self._glide(keys, hand))

    def broken_state(self) -> bool:
        q = self.particles()
        if not np.isfinite(q).all() or self.holds.bridge.diverged_worlds:
            return True
        if float(q[:, 2].min()) < self.surface_z - 0.02:
            return True  # through the bench
        return bool(self.holds.held) and float(np.linalg.norm(q[-1] - self.tcp())) > 0.05


class Snapshot:
    """The moment the pre-roll ends -- a fresh cable straight out of the socket, its end in the hand
    at :data:`START` -- kept to come back to. "New cable" and the watchdog restore it in a few
    milliseconds instead of resetting and replaying the pick (~2.3 s of a page with no ball, an arm
    snapped home and the robot reaching down again).

    Newton's half is every state and control array, as ``towel_scene.Rig`` keeps a towel. SuperDex's
    half is the engine's own captured state (what the bridge's reset restores, but taken here, with
    the cable already in the hand). The engine restores a state only onto the composition it was
    captured with, so the pins stay put across a restore -- never unpinned and pinned again -- and a
    full restart (new pins) needs a new snapshot: :meth:`valid` says which.
    """

    STATE = ("particle_q", "particle_qd", "joint_q", "joint_qd", "body_q", "body_qd")
    CONTROL = ("joint_target_q", "joint_target_qd", "joint_f", "joint_act")

    def __init__(self, sim, holds: Holds) -> None:
        self.sim, self.holds = sim, holds
        p = sim.pipeline
        self.states = [
            {f: getattr(st, f).numpy().copy() for f in self.STATE if getattr(st, f, None) is not None}
            for st in (p.state_0, p.state_1)
        ]
        self.control = {
            f: getattr(p.control, f).numpy().copy() for f in self.CONTROL if getattr(p.control, f, None) is not None
        }
        bridge = holds.bridge
        self.engine = [scene.capture_state() for scene in bridge.scenes]
        self.pins = dict(holds.pins)
        self.fixed = {n: v.copy() for n, v in holds.fixed.items()}
        self.held = {n: v.copy() for n, v in holds.held.items()}
        self.goal = {n: v.copy() for n, v in holds.goal.items()}
        self.restores = 0

    def valid(self) -> bool:
        """The pins are the ones it was captured with (a full restart makes new ones)."""
        return self.holds.pins == self.pins and bool(self.held)

    def restore(self) -> None:
        holds, p = self.holds, self.sim.pipeline
        bridge = holds.bridge
        # MuJoCo's warm start and applied forces, not the host engine's: its reset is a return to
        # the build-time rest that also takes every pin away.
        newton_solver = getattr(p.program.solver, "newton_solver", None)
        if newton_solver is not None:
            try:
                newton_solver.reset(p.state_0, flags=0)
            except TypeError:
                newton_solver.reset(p.state_0)
        for st, saved in zip((p.state_0, p.state_1), self.states, strict=True):
            for f, a in saved.items():
                getattr(st, f).assign(a)
        for f, a in self.control.items():
            getattr(p.control, f).assign(a)
        # The engine, as the bridge's own restore does it: the state, a zero-length step so its
        # queries read the restored nodes, the write-back's previous positions, a teleport for the
        # mirrors (no sweep from wherever the hand was), the divergence flags and the wrench feed.
        for scene, saved in zip(bridge.scenes, self.engine, strict=True):
            scene.restore_state(saved, release_immediately=False)
            scene.step(0.0)
        for entry in bridge.actors:
            entry["prev"] = np.ascontiguousarray(np.asarray(entry["read"]()).reshape(-1, 3), dtype=entry["prev"].dtype)
        bridge.rig.mark_teleport()
        bridge._diverged_logged.clear()
        bridge._diverged_worlds = [False] * bridge.num_worlds
        if bridge.feedback_enabled and bridge.apply_wrenches is not None:
            bridge.apply_wrenches(bridge._acc.zero())
        holds.fixed = {n: v.copy() for n, v in self.fixed.items()}
        holds.held = {n: v.copy() for n, v in self.held.items()}
        holds.goal = {n: v.copy() for n, v in self.goal.items()}
        q = holds.nodes_world()
        bridge.set_pin_targets(self.pins["fixed"], np.asarray([self.fixed[n] for n in sorted(self.fixed)]))
        bridge.set_pin_targets(self.pins["held"], np.asarray([q[n] for n in sorted(self.held)]))
        if hasattr(p, "reset_finite_flags"):
            p.reset_finite_flags()
        p.mark_dirty()
        self.restores += 1

    def release(self) -> None:
        """Give the engine its captured states back (a snapshot that will not be used again)."""
        for scene, saved in zip(self.holds.bridge.scenes, self.engine, strict=False):
            try:
                scene.release_state(saved)
            except Exception:
                pass
        self.engine = []


# ---------------------------------------------------------------------------------------------
# What the page draws
# ---------------------------------------------------------------------------------------------
def _cylinder_mesh(sides: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """A unit cylinder: radius 1, from z = 0 to z = 1, open ends."""
    a = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    ring = np.stack([np.cos(a), np.sin(a)], axis=1)
    verts = np.concatenate([np.c_[ring, np.zeros(sides)], np.c_[ring, np.ones(sides)]]).astype(np.float32)
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces += [(i, j, sides + j), (i, sides + j, sides + i)]
    return verts, np.asarray(faces, dtype=np.uint32)


def _sphere_mesh(rings: int = 6, sides: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """A unit sphere."""
    verts = [(0.0, 0.0, 1.0)]
    for r in range(1, rings):
        th = math.pi * r / rings
        for s in range(sides):
            ph = 2.0 * math.pi * s / sides
            verts.append((math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)))
    verts.append((0.0, 0.0, -1.0))
    faces = []
    for s in range(sides):
        faces.append((0, 1 + s, 1 + (s + 1) % sides))
    for r in range(rings - 2):
        for s in range(sides):
            a, b = 1 + r * sides + s, 1 + r * sides + (s + 1) % sides
            faces += [(a, a + sides, b), (b, a + sides, b + sides)]
    last = len(verts) - 1
    base = 1 + (rings - 2) * sides
    for s in range(sides):
        faces.append((last, base + (s + 1) % sides, base + s))
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


def _z_to(d: np.ndarray) -> np.ndarray:
    """wxyz quaternions turning +z onto each row of unit vectors ``d``."""
    z = np.array([0.0, 0.0, 1.0])
    w = 1.0 + d @ z
    xyz = np.cross(np.broadcast_to(z, d.shape), d)
    q = np.c_[w, xyz]
    flip = w < 1e-6  # d = -z: half a turn about x
    q[flip] = (0.0, 1.0, 0.0, 0.0)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


class CableTube:
    """The cable on the page: a cylinder per element and a ball per node, posed from the nodes.

    The studio's viewer draws a deformable from its triangles, and a centerline has none."""

    def __init__(self, server, n_nodes: int, *, radius: float = CABLE_RADIUS, color=CABLE_COLOR) -> None:
        cv, cf = _cylinder_mesh()
        sv, sf = _sphere_mesh()
        n = int(n_nodes)
        ident = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32)
        zeros = np.zeros((n, 3), dtype=np.float32)
        self.radius = float(radius)
        self.segments = server.scene.add_batched_meshes_simple(
            "/cable/segments",
            cv,
            cf,
            batched_wxyzs=ident[: n - 1],
            batched_positions=zeros[: n - 1],
            batched_scales=np.ones((n - 1, 3), dtype=np.float32) * self.radius,
            batched_colors=color,
            lod="off",
        )
        self.joints = server.scene.add_batched_meshes_simple(
            "/cable/joints",
            sv,
            sf,
            batched_wxyzs=ident,
            batched_positions=zeros,
            batched_scales=np.full(n, self.radius, dtype=np.float32),
            batched_colors=color,
            lod="off",
        )
        self.plug = server.scene.add_box(
            "/cable/plug", dimensions=(0.007, 0.014, 0.012), color=(40, 40, 44), position=(0.0, 0.0, -1.0)
        )

    def update(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64)
        if len(q) < 2 or not np.isfinite(q).all():
            return
        d = np.diff(q, axis=0)
        length = np.linalg.norm(d, axis=1)
        u = d / np.maximum(length, 1e-9)[:, None]
        scales = np.c_[np.full(len(d), self.radius), np.full(len(d), self.radius), length].astype(np.float32)
        self.segments.batched_positions = q[:-1].astype(np.float32)
        self.segments.batched_wxyzs = _z_to(u).astype(np.float32)
        self.segments.batched_scales = scales
        self.joints.batched_positions = q.astype(np.float32)
        # A dark plug on the held end, along its last element: the thing in the jaws.
        mid = q[-1] - 0.5 * 0.014 * u[-1]
        self.plug.position = tuple(float(v) for v in mid)
        self.plug.wxyz = _plug_wxyz(u[-1])


def _plug_wxyz(u: np.ndarray) -> tuple[float, float, float, float]:
    """The plug box's local y along the cable's last element."""
    y = np.array([0.0, 1.0, 0.0])
    w = 1.0 + float(y @ u)
    if w < 1e-6:
        return (0.0, 0.0, 0.0, 1.0)
    xyz = np.cross(y, u)
    q = np.array([w, *xyz])
    q /= np.linalg.norm(q)
    return tuple(float(v) for v in q)


class CableMap:
    """The route from above, as the opening camera sees it (the socket on the left, the far side up):
    the clips, the cable coloured by height (grey on the bench, orange lifted), the hand as a
    ring. It is the page's contact map, and it is where a visitor reads
    how far in or out the hand is. It spans the hand's whole reach across the route at half the
    scale along it, so it stays a ~4:1 strip and the phone's sheet stays short."""

    PX_PER_M = 900.0
    DEPTH = 0.5
    """Scale across the route (in and out), relative to along it."""
    BG = (22, 22, 26)
    BENCH = (52, 54, 60)
    LOW = np.array([150, 160, 172], dtype=np.float32)
    HIGH = np.array([255, 140, 0], dtype=np.float32)

    def __init__(self, cable: Cable) -> None:
        self.cable = cable
        self.y0, self.y1 = SOCKET_Y - 0.045, BOX_Y[1] + 0.03
        self.x0, self.x1 = BOX_X[0] - 0.015, BOX_X[1] + 0.015
        self.w = int(round((self.y1 - self.y0) * self.PX_PER_M))
        self.h = max(48, int(round((self.x1 - self.x0) * self.PX_PER_M * self.DEPTH)))
        self.top = 0.0

    def _px(self, x, y):
        """Image (column, row) of work-surface (x, y): along the route to the right, toward the
        camera down."""
        return (np.asarray(y) - self.y0) * self.PX_PER_M, (np.asarray(x) - self.x0) * self.PX_PER_M * self.DEPTH

    def caption(self) -> str:
        return f"cable {100 * self.top:.0f} cm up" if self.top > 0.015 else "cable on the bench"

    def render(self) -> np.ndarray:
        from PIL import Image, ImageDraw

        c = self.cable
        img = Image.new("RGB", (self.w, self.h), self.BG)
        draw = ImageDraw.Draw(img)
        lx, ly = self._px([ROUTE_X, ROUTE_X], [self.y0, self.y1])
        draw.line([(lx[0], ly[0]), (lx[1], ly[1])], fill=(60, 64, 72), width=1)
        sx, sy = self._px(ROUTE_X, SOCKET_Y - 0.016)
        draw.rectangle([sx - 14, sy - 6, sx + 14, sy + 6], fill=(90, 92, 98))
        col = (60, 120, 230)  # the clips' own blue: the map does not judge the route
        for clip in c.clips:
            for s in (1.0, -1.0):
                p = clip.center + clip.across * s * (CHANNEL_HALF + PRONG_R)
                px, py = self._px(p[0], p[1])
                draw.ellipse([px - 3.5, py - 3.5, px + 3.5, py + 3.5], fill=col)
        q = c.local(c.particles())
        z = q[:, 2]
        self.top = float(max(0.0, z.max() - CABLE_RADIUS))
        px, py = self._px(q[:, 0], q[:, 1])
        lift = np.clip((z - CABLE_RADIUS - 0.004) / 0.12, 0.0, 1.0)
        for i in range(len(q) - 1):
            col = self.LOW * (1.0 - lift[i]) + self.HIGH * lift[i]
            draw.line([(px[i], py[i]), (px[i + 1], py[i + 1])], fill=tuple(int(v) for v in col), width=4)
        hand = c.local(c.tcp())
        hx, hy = self._px(hand[0], hand[1])
        draw.ellipse([hx - 7, hy - 7, hx + 7, hy + 7], outline=(255, 255, 255), width=2)
        return np.asarray(img, dtype=np.uint8)


class HostDraw:
    """A cheaper ``log_state`` for the page's viewer: the shape transforms composed on the host.

    Newton's viewer launches a transform kernel per shape batch and reads each batch's world
    transforms and scales back on its own, and the studio's PBR path reads the model's whole colour
    array back per batch as well: about 700 device-to-host copies a frame for the Panda's ~87
    visual batches, ~9 ms of a 16.7 ms control step (cable_sdx/profile_draw). Here ``body_q`` is
    read once, every batch's transforms are composed in numpy, the constant scales and colours are
    read once per model, and the same ``log_instances``/``log_capsules`` calls follow. Falls back to
    the viewer's own path when it would differ: a layer transform that moved, or a new model."""

    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.orig = viewer.log_state
        self._model = None
        self._cache: dict[int, tuple] = {}
        self._colors = None
        self._offsets = None
        viewer.log_state = self.log_state
        orig_pbr = viewer._log_pbr_instances

        def pbr(name, mesh, xforms, scales, _orig=orig_pbr):
            model = viewer.model
            if self._colors is None or model is None or getattr(model, "shape_color", None) is None:
                return _orig(name, mesh, xforms, scales)
            real = model.shape_color
            model.shape_color = self._colors  # read back once per model, not once per batch
            try:
                return _orig(name, mesh, xforms, scales)
            finally:
                model.shape_color = real

        viewer._log_pbr_instances = pbr

    class _Host:
        """What ``.numpy()`` is asked of: a host copy that does not move."""

        def __init__(self, array) -> None:
            self.array = array

        def numpy(self):
            return self.array

    def _batch(self, batch) -> tuple:
        key = id(batch)
        hit = self._cache.get(key)
        if hit is None:
            parents = np.asarray(batch.parents.numpy(), dtype=np.int64)
            local = np.asarray(batch.xforms.numpy(), dtype=np.float64).reshape(-1, 7)
            scales = np.asarray(batch.scales.numpy(), dtype=np.float32).reshape(-1, 3)
            offset = np.zeros((len(parents), 3))
            if self._offsets is not None and len(self._offsets):
                worlds = np.asarray(batch.worlds.numpy(), dtype=np.int64)
                ok = (worlds >= 0) & (worlds < len(self._offsets))
                offset[ok] = self._offsets[worlds[ok]]
            hit = (parents, local, scales, parents >= 0, offset if offset.any() else None)
            self._cache[key] = hit
        return hit

    def log_state(self, state) -> None:
        v = self.viewer
        if v.model is None:
            return
        layer = tuple(float(x) for x in v.layer.xform)
        if layer != (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0) or v.model_changed:
            self._cache.clear()
            self._colors = None
            return self.orig(state)
        if v.model is not self._model:
            self._model, self._cache = v.model, {}
            color = getattr(v.model, "shape_color", None)
            self._colors = self._Host(color.numpy()) if color is not None else None
            offsets = v.world_offsets
            self._offsets = None if offsets is None else np.asarray(offsets.numpy(), dtype=np.float64).reshape(-1, 3)
        import newton
        from robosimstudio.engines.pose import compose_many

        v._sync_shape_colors_from_model()
        hidden_layer = v._layer_force_hidden()
        body_q = np.asarray(state.body_q.numpy(), dtype=np.float64).reshape(-1, 7)
        for shapes in v._shape_instances.values():
            visible = v._should_show_shape(shapes.flags, shapes.static, shapes.geo_type) and not hidden_layer
            parents, local, scales, moving, offset = self._batch(shapes)
            world = local.copy()
            if moving.any():
                world[moving] = compose_many(body_q[parents[moving]], local[moving])
            if offset is not None:
                world[:, :3] += offset
            world = world.astype(np.float32)
            colors = shapes.colors if shapes.colors_changed else None
            log = v.log_capsules if shapes.geo_type == newton.GeoType.CAPSULE else v.log_instances
            log(shapes.name, shapes.mesh, world, scales, colors, None, hidden=not visible)
            shapes.colors_changed = False
        v._log_gaussian_shapes(state)
        v._log_non_shape_state(state)
        v.model_changed = False


def view(aspect: float = 0.46) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(look-at, eye offset), world metres, for a screen ``aspect`` (width / height) wide: from the
    bench's open side (+x) at :data:`VIEW_ELEV`, aimed a little below the route so it sits above a
    phone's sheet. A phone held upright needs the whole route across its narrow width (~0.5 m at
    0.9 m); a landscape screen sees twice as wide, so its eye comes closer.

    Framed for the hand's whole box, not only the route (``cable3d/frame_eval.py``: the ball at each
    corner and its shadow, projected through a 390 x 664 phone and a 1280 x 800 desktop): the look
    point sits 2-4 cm toward the far end, where the hand works, and on a desktop 2 cm higher and 6 %
    further back, so the ball lifted at the far end clears the floating panel at the top right."""
    look = np.array([0.6 + ROUTE_X + 0.03, 0.04, 0.75 + 0.03])
    offset = 0.916 * np.array([math.cos(VIEW_ELEV), 0.0, math.sin(VIEW_ELEV)])
    if aspect > 1.0:
        look = look + np.array([0.0, 0.02, 0.05])
        offset = offset * 0.66
    return tuple(look.tolist()), tuple(offset.tolist())


__all__ = [
    "Cable",
    "CableMap",
    "CableTube",
    "Holds",
    "HostDraw",
    "Snapshot",
    "build_env",
    "build_scene",
    "clips",
    "install_mirror_padding",
    "pre_roll",
    "routed_count",
    "view",
]
