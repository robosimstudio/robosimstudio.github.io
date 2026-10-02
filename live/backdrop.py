"""The wipe's room, for every live task: one place a visitor recognises from tab to tab.

**What the wipe's background is.** "Wipe the bar" is the phone demo's ``--scene bar``:
``rgbs/scripts/surface_cleaning.py: build_scene()``, an ``rss.Scene`` on ``bench_075`` stated with
``render=RGBS_LOOK`` (``rgbs/tasks/rgbs/look.py``). That look is the house look plus
``environment="lab"`` and ``scenery="rgbs_lab"``, and the scenery is a registered builder
(``robosimstudio.core.render.scenery``) that calls ``robosimstudio.core.render.room.lab``. Placed
against whatever cell the scene stands in, it draws:

* **the room**: a dark matte floor, 6 x 6 m, and two pale walls at the cell's +-y (``house_room``);
* **the jigsaw mat**: a 0.5 mm box over the whole bench top carrying ``room_grid_mat`` (a 25 mm
  grid on 300 mm interlocking tiles), and a black apron along the bench's near (-x) edge;
* **the far side**: an oak ledge along the bench's +x edge, the slat wall 6 cm behind it (oak
  slats, 60 mm on a 105 mm pitch, on dark 30 %-opaque glass), two white rig posts on the ledge,
  and through the glass a dim lounge (far wall, round table, two chairs, a bench).

None of it is a spec: no body, no shape, no collider, so the physics never sees it. Two kits draw
it from the one ``env.look.scenery`` name: the studio page (``robosimstudio_studio/scenery.py``,
``ViserKit``: a GLB per box under ``/studio/scenery``, sent once and kept by viser for late
joiners) and the path tracer (``rtx.py``, ``_StageKit``: USD meshes), so "Render Mode (Slow)" shows
the same room. (The mat seen in the bar is the task's own 10 mm ``prep_mat`` static, which lies over
the scenery mat with the same finish.)

**Who has it.** The peel (``rgbs/panel``) states the same ``RGBS_LOOK`` and has the room already.
The towel (``cloth_fold/towel_half``, on ``bench_0728``) and the cable (an ``rss.Scene`` on
``bench_075``) state no look and get the library's default room, ``house_room_closed`` (concrete
and pale plaster). Both look at their robots from the bench's open +x side, the opposite way to
the bar's camera, and both robots stand at or behind the bench's -x edge. So here the lab is drawn
**turned half a turn about the bench centre** -- the mat still on the top, the apron at the near
(+x) edge, the ledge, slats, glass and lounge behind the robots -- with the slat wall, and the room
round it, **set back** (:attr:`Place.back`) far enough to clear the robots. The ledge goes with the
wall for the towel, whose arm bases stand where it would be, and stays on the bench's edge for the
cable, whose Panda stands 0.3 m behind it. Nothing is copied: the builder is ``room.lab`` itself,
handed a kit that moves what it draws (:class:`_Placed`).

:func:`dress` stands a built scene in it: it points the scene's stated look at the turned lab (the
look is ``room.LAB``, which is ``RGBS_LOOK`` field for field but for the scenery's name). Call it
before the page is built; the page draws the look it finds.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

try:
    from robosimstudio.core.render.room import LAB, lab
    from robosimstudio.core.render.scenery import register_scenery
except Exception as _exc:  # noqa: BLE001 -- the room is decoration: a page must still start without it
    LAB = lab = register_scenery = None
    UNAVAILABLE = f"{type(_exc).__name__}: {_exc}"
else:
    UNAVAILABLE = ""

__all__ = ["LEDGE", "NEAR", "PLACES", "UNAVAILABLE", "WIPE_SCENERY", "Place", "dress", "faces", "look", "robot_rear", "scenery_name"]

WIPE_SCENERY = "rgbs_lab"
"""What the bar and the peel state (``RGBS_LOOK.scenery``): ``room.lab``, as measured, unturned."""

NEAR = ("grid_mat", "bench_apron")
"""The pieces ``room.lab`` lays on the bench itself. They turn with the rest but are never set back:
the mat stays on the top and the apron on the bench's near edge."""

LEDGE = ("ledge", "rig_post_")
"""The oak ledge along the bench's far edge and the two rig posts standing on it (name prefixes):
set back by :attr:`Place.ledge_back`. Everything else -- glass, slats, lounge, floor, side walls --
by :attr:`Place.back`."""


@dataclass(frozen=True)
class Place:
    """How the wipe's lab stands round one page's bench."""

    turn: bool
    """Half a turn about the bench centre: the ledge and slat wall go to the bench's -x side, behind
    robots seen from a camera at +x."""
    back: float
    """How much further from the bench than in the wipe the slat wall stands [m] -- the glass, the
    slats and the lounge behind them, and with them the floor and side walls, so the room keeps its
    proportions. The wipe's own robots stand on the camera's side and need none."""
    ledge_back: float | None = None
    """The same for the ledge and its rig posts; ``None`` is :attr:`back`. Zero keeps the ledge on
    the bench's far edge, as in the wipe, where nothing of the robot stands there."""
    why: str = ""

    def setback(self, name: str) -> float:
        if name in NEAR:
            return 0.0
        if name.startswith(LEDGE):
            return self.back if self.ledge_back is None else self.ledge_back
        return self.back


PLACES: dict[str, Place] = {
    "towel": Place(
        turn=True,
        back=0.40,
        why="yam_bimanual's arm bases stand on bench_0728's -x edge, where the ledge would be, and the arms' "
        "geometry reaches back to x +0.06 (bounding spheres, every pose a drag reached): the ledge, "
        "slats and lounge go back together, the ledge's face to x -0.10",
    ),
    "cable": Place(
        turn=True,
        back=0.50,
        ledge_back=0.0,
        why="the Panda stands 0.6 m behind bench_075's centre (x 0.0, 132 mm over the top), its links' "
        "bounding spheres reaching x -0.165: the ledge stays on the bench's edge, in front of it, and "
        "the slat wall goes behind it, its face at x -0.25",
    ),
}
"""The pages that stand in a turned lab. The bar and the peel are not here: they state the wipe's
own look already."""


def scenery_name(place: str) -> str:
    return f"rsw_lab_{place}"


def _turn_quat(q_xyzw) -> tuple[float, float, float, float]:
    """``q`` turned half a turn about world z: (0, 0, 1, 0) x q, Hamilton, xyzw."""
    x, y, z, w = (float(v) for v in q_xyzw)
    # (0, 0, 1, 0) * (x, y, z, w) = (-y, x, w, -z)
    return (-y, x, w, -z)


class _Placed:
    """A scenery kit that moves what a builder draws: each piece set back along the builder's +x
    (toward its far side) by :meth:`Place.setback`, then, with ``place.turn``, half a turn about the
    cell's centre. Boxes stay boxes, so both kits compute their own UVs on the moved geometry (world
    metres: the mat's grid lands on the same world lines as the wipe's)."""

    def __init__(self, kit, place: Place) -> None:
        self.kit = kit
        self.place = place
        self.cell = getattr(kit, "cell", None)
        self.surface_z = float(getattr(kit, "surface_z", 0.0) or 0.0)
        center = getattr(self.cell, "center", None) or (0.0, 0.0, 0.0)
        self.pivot = np.array([float(center[0]), float(center[1])], dtype=np.float64)

    def _move(self, name: str, points) -> np.ndarray:
        p = np.array(points, dtype=np.float64).reshape(-1, 3)
        p[:, 0] += self.place.setback(name)
        if self.place.turn:
            p[:, :2] = 2.0 * self.pivot - p[:, :2]
        return p

    def box(self, name: str, center, half, *, color, finish, quat=None) -> None:
        c = self._move(name, center)[0]
        if self.place.turn and quat is not None:
            quat = _turn_quat(quat)
        self.kit.box(name, tuple(float(v) for v in c), half, color=color, finish=finish, quat=quat)

    def mesh(self, name: str, points, faces, *, color, finish, uvs=None) -> None:
        self.kit.mesh(name, self._move(name, points).astype(np.float32), faces, color=color, finish=finish, uvs=uvs)


def lab_towel(kit) -> None:
    """The wipe's lab round the towel's bench (``PLACES["towel"]``)."""
    lab(_Placed(kit, PLACES["towel"]))


def lab_cable(kit) -> None:
    """The wipe's lab round the cable's bench (``PLACES["cable"]``)."""
    lab(_Placed(kit, PLACES["cable"]))


if not UNAVAILABLE:
    register_scenery(scenery_name("towel"), lab_towel)
    register_scenery(scenery_name("cable"), lab_cable)


def look(place: str):
    """The look a page in ``place`` renders under: the wipe's (``RGBS_LOOK`` is ``room.LAB`` with the
    scenery named ``rgbs_lab``, which calls the same ``room.lab``) with the turned lab as scenery."""
    if place not in PLACES:
        raise KeyError(f"no turned lab for {place!r}; there is one for {sorted(PLACES)}")
    return dataclasses.replace(LAB, scenery=scenery_name(place))


def _cell(env):
    from robosimstudio.scene.registry import get_cell

    request = env.scene.request
    name = getattr(request, "scene", "") or getattr(request, "cell", "")
    return get_cell(name) if name else None


def faces(env, place: str) -> dict[str, float] | None:
    """World x of the bench-side faces of the ledge and of the slat wall in ``place``, or ``None``
    for a page without a turned lab. (``room.lab``: the ledge spans 4 cm from the bench's far edge,
    the slats start ``LAB_WALL_SETBACK`` past it.)"""
    from robosimstudio.core.render.room import LAB_WALL_SETBACK

    p = PLACES.get(place)
    cell = _cell(env)
    if p is None or cell is None:
        return None
    cx, hx = float(cell.center[0]), float(cell.half[0])
    out = {}
    for what, name in (("ledge", "ledge"), ("slats", "slat_0")):
        x = cx + hx + p.setback(name) + (LAB_WALL_SETBACK if what == "slats" else 0.0)
        out[what] = 2.0 * cx - x if p.turn else x
    return out


MAX_PART_RADIUS = 1.0
"""Shapes with a bounding radius over this [m] are not parts of an arm: the Panda's grip-site
cylinder reports 10 m."""


def robot_rear(env) -> float | None:
    """World x of the robot's rearmost point, conservatively: every robot shape's centre less its
    bounding radius, at the pose the state holds now."""
    model = env.model
    radius = getattr(model, "shape_collision_radius", None)
    if radius is None:
        return None
    from robosimstudio.core.math import transform_matrix

    body = np.asarray(model.shape_body.numpy())
    xf = np.asarray(model.shape_transform.numpy(), dtype=np.float64)
    r = np.asarray(radius.numpy(), dtype=np.float64)
    bq = np.asarray(env.state.body_q.numpy(), dtype=np.float64)
    emb = getattr(env.scene, "embodiment", None)
    shapes = getattr(getattr(emb, "range", None), "shape_indices", None)
    idx = np.asarray(shapes if shapes is not None and len(shapes) else np.nonzero(body >= 0)[0], dtype=np.int64)
    rear = np.inf
    for s in idx:
        if float(r[s]) > MAX_PART_RADIUS:
            continue
        b = int(body[s])
        world = transform_matrix(bq[b]) @ transform_matrix(xf[s]) if b >= 0 else transform_matrix(xf[s])
        rear = min(rear, float(world[0, 3]) - float(r[s]))
    return rear if np.isfinite(rear) else None


def dress(env, place: str) -> str:
    """Stand the built scene ``env`` in the wipe's room. Returns a line for the log; never raises.

    Only the look the scene states changes (``request.render``), which the page's backdrop and
    lighting and the path tracer read and nothing physical does. A page in a place that states the
    wipe's look already (the bar, the peel) is left as it is, and anything going wrong leaves the
    page in the room it had: the room is decoration, and must never keep a page from starting.
    """
    if UNAVAILABLE:
        return f"backdrop: unavailable ({UNAVAILABLE}); {place} keeps its own room"
    try:
        stated = getattr(env.look, "scenery", "") or ""
        if place not in PLACES:
            if stated == WIPE_SCENERY:
                return f"backdrop: {place} states the wipe's own look already ({stated})"
            return f"backdrop: no turned lab for {place!r}; it keeps {stated or 'no scenery'}"
        new = look(place)
        request = env.scene.request
        dressed = dataclasses.replace(request, render=new)
        env.scene.request = dressed
        if getattr(env, "task", None) is request:  # an Env recompiles from its task: keep the look across that
            env.task = dressed
    except Exception as exc:  # noqa: BLE001 -- never keep a page from starting
        return f"backdrop: skipped for {place} ({type(exc).__name__}: {exc}); it keeps its own room"
    return f"backdrop: the wipe's lab, {new.scenery} (was {stated or 'none'}){_clearance(env, place)}"


def _clearance(env, place: str) -> str:
    """How far the robot's rearmost point stands in front of the slat wall, for the log."""
    try:
        face, rear = faces(env, place), robot_rear(env)
    except Exception as exc:  # noqa: BLE001 -- a log line, not a reason to stop
        return f"; clearance not measured ({type(exc).__name__}: {exc})"
    if face is None or rear is None:
        return ""
    gap = rear - face["slats"] if PLACES[place].turn else face["slats"] - rear
    out = (f"; robot's rearmost point {rear:+.3f} m, ledge face {face['ledge']:+.3f} m, slats {face['slats']:+.3f} m "
           f"({100 * gap:.0f} cm clear of the slats)")
    if gap < 0.0:
        out += " -- THE ROBOT REACHES INTO THE SLAT WALL"
    return out
