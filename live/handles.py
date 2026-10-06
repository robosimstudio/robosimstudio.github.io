"""Take the robot's hand: the drag handle of every page, in place of the orange ball.

The pages used to drive the hand with an orange ball (``wipe_phone.PhoneControls``: an icosphere and
a "drag me" label, dragged on a plane facing the viewer's camera; the hand chases it at its real
speed limit). Visitors found the ball distracting. Here nothing extra stands in the scene: the
visitor presses the robot itself. Same drag events, same driver mapping (``Wipe.drag`` and its
kin), so only what one presses and what one sees change.

* **What takes the drag.** The driving arm's hand meshes (the IK's end-effector body and what hangs
  from it), and an invisible ball round the tool (:data:`ZONE_RADIUS`): the sponge on the bar, the
  gripper elsewhere -- a target a thumb can hit on a phone. The cable's shadow handle on the bench
  stays, drawn as a quiet shadow under the hand (it moves the hand in and out).
* **Before touching.** Each viewer gets a hint until their first drag (and again once nobody has
  dragged for :data:`HINT_REPEAT`): what to do, in words (:data:`TASKS`), over the tool, and a
  finger acting the drag out along its path drawn as arrows.
* **While holding.** A light-blue shell lights the gripper up, a small dot marks where the hand is
  heading and a line pulls from the hand to it; it clears on release.

:func:`install` patches ``PhoneControls`` (the cable's subclass too). Call before the page is built.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np

LIGHT_BLUE = (110, 195, 255)
"""Held: the gripper's shell, the target dot, the pull line, the cable's shadow while it slides."""
HINT_ORANGE = (255, 140, 0)
"""The hint's arrows."""
SHADOW = (40, 40, 46)

ZONE_RADIUS = 0.05
"""The invisible ball round each tool point that takes the drag [m]."""
DOT_RADIUS = 0.008
SHELL_OFFSET, SHELL_OPACITY = 0.0015, 0.55
"""The gripper's shell: pushed out along the normals [m] so it never fights the gripper's own
surface (whose meshes carry baked colours), and how opaque."""
HINT_REPEAT = 45.0
"""A viewer who has dragged sees the hint again once nobody has dragged for this long [s]."""
FINGER_RADIUS, FINGER_FORWARD = 0.012, 0.07
"""The hint's finger [m], drawn this far in front of the tool toward the viewer's camera."""
LABEL_UP, LABEL_SIDE = 0.07, 0.06
"""The hint's words stand this far over the tool, or to its left [m]."""
LABEL_SCALE = 1.6
ARROW_HEAD = 0.009


@dataclass(frozen=True)
class Leg:
    """One stroke of a hint's gesture, from where the finger lands [m, world]."""

    to: tuple[float, float, float]
    seconds: float
    arrow: bool = True
    both_ends: bool = False


def _bar(wipe) -> tuple[str, list[Leg]]:
    a, z = np.asarray(wipe.space.along, dtype=np.float64), np.array([0.0, 0.0, 1.0])
    press, swing = -0.03 * z, 0.07 * a
    return "Drag the sponge to wipe the bar", [
        Leg(tuple(press), 0.5),
        Leg(tuple(press - swing), 0.55, arrow=False),
        Leg(tuple(press + swing), 1.1, both_ends=True),
        Leg(tuple(press), 0.55, arrow=False),
    ]


def _peel(wipe) -> tuple[str, list[Leg]]:
    d = -np.asarray(wipe.u, dtype=np.float64) + float(wipe.rise) * np.asarray(wipe.n, dtype=np.float64)
    d /= max(np.linalg.norm(d), 1e-9)
    return "Drag the hand to peel the film", [Leg(tuple(0.07 * d), 1.4)]


def _towel(wipe) -> tuple[str, list[Leg]]:
    up, over = 0.05 * np.asarray(wipe.B, dtype=np.float64), 0.10 * np.asarray(wipe.A, dtype=np.float64)
    return "Drag the hand to fold the towel", [Leg(tuple(up), 0.6), Leg(tuple(up + over), 1.3)]


def _cable(wipe) -> tuple[str, list[Leg]]:
    z, along = np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])  # toward the clips, from the start
    return "Drag the hand to lay the cable in the clips", [
        Leg(tuple(0.04 * z), 0.6),
        Leg(tuple(0.04 * z + 0.09 * along), 1.2),
        Leg(tuple(0.01 * z + 0.09 * along), 0.5),
    ]


TASKS = {"wipe": _bar, "peel": _peel, "towel": _towel, "cable": _cable}
"""The driver's class (lower case) -> its hint: the words and the gesture."""
BESIDE = {"peel"}
"""Pages whose hint words stand to the left of the hand, not over it: the peel's opening view has the
hand at its top edge."""


def _quat_to_wxyz_from_z(axis) -> tuple[float, float, float, float]:
    """The rotation (w, x, y, z) taking +z onto ``axis``."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    c = float(z @ a)
    if c < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)
    q = np.array([1.0 + c, *np.cross(z, a)])
    q /= np.linalg.norm(q)
    return tuple(float(x) for x in q)


def _xyzw_to_wxyz(q) -> tuple[float, float, float, float]:
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def torus(major: float, minor: float, n_major: int = 48, n_minor: int = 10):
    """A torus round +z: (vertices, faces)."""
    u = np.linspace(0.0, 2 * np.pi, n_major, endpoint=False)
    v = np.linspace(0.0, 2 * np.pi, n_minor, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    r = major + minor * np.cos(vv)
    verts = np.stack([r * np.cos(uu), r * np.sin(uu), minor * np.sin(vv)], axis=-1).reshape(-1, 3)
    faces = []
    for i in range(n_major):
        for j in range(n_minor):
            a, b = i * n_minor + j, ((i + 1) % n_major) * n_minor + j
            c, d = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor, i * n_minor + (j + 1) % n_minor
            faces += [(a, b, c), (a, c, d)]
    return verts.astype(np.float32), np.asarray(faces, dtype=np.uint32)


def disc(radius: float, n: int = 32):
    """A flat disc round +z: (vertices, faces)."""
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    verts = np.concatenate([[[0.0, 0.0, 0.0]], np.stack([radius * np.cos(a), radius * np.sin(a), np.zeros(n)], axis=1)])
    faces = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
    return verts.astype(np.float32), np.asarray(faces, dtype=np.uint32)


def _matrix(xform) -> np.ndarray:
    from robosimstudio.core.math import transform_matrix

    return transform_matrix(np.asarray(xform, dtype=np.float64))


def _quietly(fn) -> None:
    try:
        fn()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------------------------
# The driver, whichever page it is
# ---------------------------------------------------------------------------------------------
def arms(wipe) -> list:
    """The IK arms the finger drives (the towel's driver keeps a dict of them)."""
    iks = getattr(wipe, "iks", None)
    return list(iks.values()) if iks else [wipe.ik]


def target(wipe) -> np.ndarray:
    """Where the hand is heading: the cursor, less the ball's offset on pages that kept one."""
    return np.asarray(wipe.cursor, dtype=np.float64) - np.asarray(getattr(wipe, "ball", 0.0), dtype=np.float64)


def _state(wipe):
    owner = getattr(wipe, "sim", None) or getattr(wipe, "env", None)
    return owner.state


def tool_points(wipe) -> list[np.ndarray]:
    """Where the invisible grab balls sit: the sponge on the bar, each gripper's tool centre elsewhere."""
    if hasattr(wipe, "sponge_idx"):
        return [np.asarray(wipe.handle_point(), dtype=np.float64)]
    return [np.asarray(ik.ee_pose(_state(wipe))[0], dtype=np.float64) for ik in arms(wipe)]


# ---------------------------------------------------------------------------------------------
# The page's handle
# ---------------------------------------------------------------------------------------------
class Handle:
    """The extra nodes and bindings of one page's controls (``PhoneControls``)."""

    def __init__(self, controls) -> None:
        self.c = controls
        self.server = controls.server
        self.session = controls.session
        self.hands: list[set[int]] = []
        """Per arm: the shapes of its hand."""
        self.bound: dict[str, int] = {}
        """Node name -> id of the handle we bound."""
        self.zones: list = []
        self.shells: list = []
        self.done: set[int] = set()
        """Viewers who have dragged: their hint stays down until nobody drags for HINT_REPEAT."""
        self.coaches: dict[int, Coach] = {}
        self.last_drag = 0.0
        self.last_bind = 0.0
        self.scale: dict[int, float] = {}
        """Viewer -> the depth factor of the drag they are in (a grab ball's face to its centre)."""
        self.probe = os.environ.get("RSW_PROBE", "")
        self.probe_at = 0.0
        self.ready = False
        scene = self.server.scene
        self.dot = scene.add_icosphere("/phone/target", radius=DOT_RADIUS, color=LIGHT_BLUE, opacity=0.95, subdivisions=2,
                                       cast_shadow=False, receive_shadow=False, visible=False)
        self.tether = scene.add_line_segments("/phone/tether", points=np.zeros((1, 2, 3), dtype=np.float32),
                                              colors=LIGHT_BLUE, line_width=3.0, visible=False)

    # ------------------------------------------------------------ input
    def _drag(self, event) -> None:
        cam = event.client.camera
        eye = np.asarray(cam.position, dtype=np.float64)
        view = (eye, np.asarray(cam.look_at, dtype=np.float64))
        end = np.asarray(event.end_position, dtype=np.float64)
        if any(event.target is z for z in self.zones):
            # the ball's face is nearer the camera than the tool: scale the drag to the tool's depth
            if event.phase == "start":
                centre = np.asarray(event.target.position, dtype=np.float64)
                start = np.asarray(event.start_position, dtype=np.float64)
                self.scale[event.client_id] = float(np.linalg.norm(centre - eye) / max(np.linalg.norm(start - eye), 1e-6))
            end = eye + (end - eye) * self.scale.get(event.client_id, 1.0)
        if event.phase == "start":
            self.done.add(event.client_id)
        self.last_drag = time.time()
        self.c.events.append((event.phase, event.client_id, end, view))

    def _drag_hand(self, rows):
        def on(event) -> None:
            if rows is not None and event.instance_index is not None and int(event.instance_index) not in rows:
                return  # the same mesh on an arm nobody drives
            self._drag(event)
        return on

    # ------------------------------------------------------------ the hand
    def _find_hands(self, wipe) -> None:
        """Each driven arm's hand: the IK's end-effector body and every body hanging from it, or its
        parent link's when the end effector carries no meshes of its own."""
        model = self.session.env.model
        parent = np.asarray(model.joint_parent.numpy())
        child = np.asarray(model.joint_child.numpy())
        kids: dict[int, list[int]] = {}
        for p, c in zip(parent, child):
            kids.setdefault(int(p), []).append(int(c))
        up = {int(c): int(p) for p, c in zip(parent, child)}
        body = np.asarray(model.shape_body.numpy())
        flags = np.asarray(model.shape_flags.numpy())

        def subtree(b):
            out, todo = set(), [b]
            while todo:
                x = todo.pop()
                out.add(x)
                todo += kids.get(x, [])
            return out

        def visible(bodies):
            return {int(s) for s in np.nonzero(np.isin(body, list(bodies)))[0] if int(flags[s]) & 1}  # 1: VISIBLE

        labels = list(getattr(model, "body_label", None) or getattr(model, "body_key", None) or [])
        self.hands = []
        for ik in arms(wipe):
            ee = int(ik.ee_body)
            shapes = visible(subtree(ee))
            while len(shapes) < 2 and up.get(ee, -1) >= 0:  # a bare TCP frame: take the link that carries it
                ee = up[ee]
                shapes = visible(subtree(ee))
            self.hands.append(shapes)
            names = sorted({str(labels[b]).split("/")[-1] for b in subtree(ee) if b < len(labels)})
            print(f"[handles] a hand: {len(shapes)} shapes on {names}", flush=True)

    def _bind(self) -> None:
        """Give the hands' meshes the drag. The viewer re-creates its nodes at times: re-bind those."""
        from robosimstudio_studio.garments import DEFAULT_FINISH, group_by_material

        viewer = self.session.viewer
        mine_all = set().union(*self.hands) if self.hands else set()
        model = viewer.model
        colors = model.shape_color.numpy() if getattr(model, "shape_color", None) is not None else None
        for qualified, nodes in list(getattr(viewer, "_pbr_handles", {}).items()):
            shapes = viewer._batch_shapes(qualified)
            if not shapes or not any(int(s) in mine_all for s in shapes):
                continue
            for slot, rows in enumerate(group_by_material(shapes, colors, viewer._finishes, DEFAULT_FINISH).values()):
                node = f"{qualified}/m{slot}"
                handle = nodes.get(node)
                if handle is None or self.bound.get(node) == id(handle):
                    continue
                mine = [k for k, r in enumerate(rows) if int(shapes[r]) in mine_all]
                if not mine:
                    continue
                handle.on_drag(self._drag_hand(None if len(mine) == len(rows) else set(mine)))
                self.bound[node] = id(handle)

    def _build(self, wipe) -> None:
        """The grab balls and the gripper shells, once the driver exists."""
        import trimesh

        scene = self.server.scene
        self._find_hands(wipe)
        for i, _p in enumerate(tool_points(wipe)):
            zone = scene.add_icosphere(f"/phone/zone_{i}", radius=ZONE_RADIUS, color=LIGHT_BLUE, opacity=0.0, subdivisions=2,
                                       cast_shadow=False, receive_shadow=False)
            zone.on_drag(self._drag)
            self.zones.append(zone)
        model, state = self.session.env.model, _state(wipe)
        bq = state.body_q.numpy()
        body, xf, scale = model.shape_body.numpy(), model.shape_transform.numpy(), model.shape_scale.numpy()
        for ik, shapes in zip(arms(wipe), self.hands):
            pos, quat = ik.ee_pose(state)
            inv = np.linalg.inv(_matrix([*pos, *quat]))
            verts, faces, n = [], [], 0
            for s in sorted(shapes):
                src = model.shape_source[s]
                if src is None or not hasattr(src, "vertices"):
                    continue
                m = inv @ _matrix(bq[int(body[s])]) @ _matrix(xf[s])
                v = (np.asarray(src.vertices, dtype=np.float64) * scale[s]) @ m[:3, :3].T + m[:3, 3]
                verts.append(v)
                faces.append(np.asarray(src.indices, dtype=np.int64).reshape(-1, 3) + n)
                n += len(v)
            if not verts:
                self.shells.append(None)
                continue
            welded, inverse = np.unique(np.round(np.concatenate(verts), 6), axis=0, return_inverse=True)
            f = inverse.reshape(-1)[np.concatenate(faces)]
            normals = np.asarray(trimesh.Trimesh(welded, f, process=False).vertex_normals)
            self.shells.append(scene.add_mesh_simple(
                f"/phone/shell_{len(self.shells)}", (welded + SHELL_OFFSET * normals).astype(np.float32), f.astype(np.uint32),
                color=LIGHT_BLUE, opacity=SHELL_OPACITY, cast_shadow=False, receive_shadow=False, visible=False))
        self._wrap_subclass()
        self.ready = True

    def _wrap_subclass(self) -> None:
        """A page whose controls subclass ``PhoneControls`` (the cable's) draws after the base refresh:
        run :meth:`_after` after it, so the cable's shadow handle stays a quiet shadow."""
        c = self.c
        cls_refresh = type(c).refresh
        if getattr(cls_refresh, "_rsw_handle", False):
            return

        def refresh() -> None:
            cls_refresh(c)
            self._after()

        c.refresh = refresh

    def _after(self) -> None:
        shadow, wipe = getattr(self.c, "shadow", None), self.session.wipe
        if shadow is None or wipe is None:
            return
        at = target(wipe) if self.c.dragging else np.asarray(wipe.tcp(), dtype=np.float64)
        shadow.position = tuple(float(v) for v in wipe.shadow_point(at))
        sliding = self.c.dragging and getattr(wipe, "grab_kind", "") == "shadow"
        shadow.color = LIGHT_BLUE if sliding else SHADOW
        shadow.opacity = 0.7 if sliding else 0.32
        shadow.visible = True

    # ------------------------------------------------------------ drawing
    def refresh(self) -> None:
        c, wipe = self.c, self.session.wipe
        if wipe is None:
            return
        now = time.time()
        if not self.ready:
            self._build(wipe)
        if now - self.last_bind > 1.0:
            self.last_bind = now
            try:
                self._bind()
            except Exception as exc:  # noqa: BLE001 -- the hand is a convenience: never stop the page
                print(f"[handles] binding the hand: {type(exc).__name__}: {exc}", flush=True)
        for zone, p in zip(self.zones, tool_points(wipe)):
            zone.position = tuple(float(v) for v in p)
        holding = bool(c.dragging)
        aim = target(wipe)
        self.dot.visible = self.tether.visible = holding
        if holding:
            self.dot.position = tuple(float(v) for v in aim)
            self.tether.points = np.asarray([[wipe.tcp(), aim]], dtype=np.float32)
        state = _state(wipe)
        for shell, ik in zip(self.shells, arms(wipe)):
            if shell is None:
                continue
            shell.visible = holding
            if holding:
                pos, quat = ik.ee_pose(state)
                shell.position, shell.wxyz = tuple(float(v) for v in pos), _xyzw_to_wxyz(quat)
        self._hints(wipe, now)
        self._probe(wipe, now)

    def _hints(self, wipe, now) -> None:
        """Each viewer's hint, until they drag (and again after HINT_REPEAT with nobody dragging); down
        for everyone while someone drives."""
        clients = self.server.get_clients()
        for cid in list(self.coaches):
            if cid not in clients:
                self.coaches.pop(cid).remove()
        self.done &= set(clients)
        make = TASKS.get(type(wipe).__name__.lower())
        quiet = now - self.last_drag > HINT_REPEAT
        for cid, client in clients.items():
            want = make is not None and not self.c.dragging and (cid not in self.done or quiet)
            coach = self.coaches.get(cid)
            if want and coach is None:
                text, legs = make(wipe)
                coach = self.coaches[cid] = Coach(client, text, legs, now, beside=type(wipe).__name__.lower() in BESIDE)
            elif not want and coach is not None:
                self.coaches.pop(cid).remove()
                coach = None
            if coach is not None:
                coach.update(tool_points(wipe)[0], now)

    def _probe(self, wipe, now) -> None:
        """For scripted recordings: where the things one presses are, and the hint's path, in world metres."""
        if not self.probe or now - self.probe_at < 0.2:
            return
        self.probe_at = now
        cams = {}
        for cid, client in self.server.get_clients().items():
            cam = client.camera
            cams[str(cid)] = {"position": list(map(float, cam.position)), "look_at": list(map(float, cam.look_at)),
                              "up": list(map(float, cam.up_direction)), "fov": float(cam.fov),
                              "size": [float(cam.image_width), float(cam.image_height)]}
        make = TASKS.get(type(wipe).__name__.lower())
        legs = [list(leg.to) for leg in make(wipe)[1]] if make else []
        shadow = getattr(self.c, "shadow", None)
        data = {"tool": tool_points(wipe)[0].tolist(), "tcp": np.asarray(wipe.tcp()).tolist(), "target": target(wipe).tolist(),
                "dragging": bool(self.c.dragging), "legs": legs, "cams": cams,
                "shadow": None if shadow is None else list(map(float, shadow.position))}
        tmp = self.probe + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.probe)


class Coach:
    """One viewer's hint: the words over the tool and a finger acting the drag out over its path drawn as
    arrows. Nodes on that viewer's own scene only."""

    def __init__(self, client, text: str, legs: list[Leg], now: float, *, beside: bool = False) -> None:
        self.client, self.t0, self.legs, self.beside = client, now, legs, beside
        sc, base = client.scene, f"/phone/coach_{client.client_id}"
        self.label = sc.add_label(f"{base}/label", text, position=(0.0, 0.0, 0.0),
                                  anchor="center-right" if beside else "bottom-center", font_screen_scale=LABEL_SCALE)
        v, f = disc(FINGER_RADIUS)
        self.finger = sc.add_mesh_simple(f"{base}/finger", v, f, color=(255, 255, 255), opacity=0.0, side="double",
                                         cast_shadow=False, receive_shadow=False)
        v, f = torus(FINGER_RADIUS, 0.0016, 40, 6)
        self.rim = sc.add_mesh_simple(f"{base}/rim", v, f, color=(35, 35, 38), opacity=0.0, cast_shadow=False, receive_shadow=False)
        eye = np.asarray(client.camera.position, dtype=np.float64)
        self.path = sc.add_line_segments(f"{base}/path", points=self._arrows(eye), colors=HINT_ORANGE, line_width=2.5)
        self.nodes = [self.label, self.finger, self.rim, self.path]
        self.period = 0.35 + sum(leg.seconds for leg in legs) + 0.6

    def _arrows(self, eye) -> np.ndarray:
        """The legs as line segments with heads, the heads in the plane facing ``eye``."""
        segs, start = [], np.zeros(3)
        for leg in self.legs:
            end = np.asarray(leg.to, dtype=np.float64)
            d = end - start
            n = float(np.linalg.norm(d))
            if leg.arrow and n > 1e-6:
                d /= n
                side = np.cross(d, eye)  # across the arrow, roughly as seen
                side = side - (side @ d) * d
                side /= max(np.linalg.norm(side), 1e-9)
                segs.append((start, end))
                tips = [(end, -d)] + ([(start, d)] if leg.both_ends else [])
                for tip, back in tips:
                    segs += [(tip, tip + ARROW_HEAD * (back + side)), (tip, tip + ARROW_HEAD * (back - side))]
            start = end
        return np.asarray(segs or [(np.zeros(3), np.zeros(3))], dtype=np.float32)

    def _at(self, phase: float) -> tuple[np.ndarray, float]:
        """Where the finger is (from where it lands) and how opaque, ``phase`` seconds into a loop."""
        if phase < 0.35:
            return np.zeros(3), phase / 0.35
        t, start = phase - 0.35, np.zeros(3)
        for leg in self.legs:
            end = np.asarray(leg.to, dtype=np.float64)
            if t < leg.seconds:
                s = t / leg.seconds
                return start + (end - start) * (s * s * (3 - 2 * s)), 1.0
            t -= leg.seconds
            start = end
        return start, max(0.0, 1.0 - t / 0.6)

    def update(self, tool: np.ndarray, now: float) -> None:
        eye = np.asarray(self.client.camera.position, dtype=np.float64)
        if self.beside:  # to the left of the hand, as this viewer sees it
            look = np.asarray(self.client.camera.look_at, dtype=np.float64)
            right = np.cross(look - eye, np.array([0.0, 0.0, 1.0]))
            right /= max(np.linalg.norm(right), 1e-9)
            self.label.position = tuple(float(v) for v in tool - LABEL_SIDE * right)
        else:
            self.label.position = tuple(float(v) for v in tool + np.array([0.0, 0.0, LABEL_UP]))
        toward = eye - tool
        toward /= max(np.linalg.norm(toward), 1e-9)
        origin = tool + FINGER_FORWARD * toward
        offset, alpha = self._at((now - self.t0) % self.period)
        at, q = tuple(float(v) for v in origin + offset), _quat_to_wxyz_from_z(toward)
        for node, full in ((self.finger, 0.92), (self.rim, 0.85)):
            node.position, node.wxyz, node.opacity = at, q, full * alpha
        self.path.position = tuple(float(v) for v in origin)

    def remove(self) -> None:
        for node in self.nodes:
            _quietly(node.remove)


class _Sink:
    """Takes attribute writes and keeps them."""


HOW = {
    "bar": "Drag the sponge: down presses it onto the bar, sideways wipes. Drag empty space to look around.",
    "peel": "Drag the robot's hand down the phone's face to peel the film off. It plays in slow motion.",
}
"""The phone demo's words for its scenes, which told visitors to drag the orange ball (the towel's and the
cable's own say it in their modules)."""


def install(wp) -> None:
    """Patch ``wp.PhoneControls``: the robot's hand is the handle; the ball and its label are never drawn."""
    cls = wp.PhoneControls
    if getattr(cls, "_rsw_handle", False):
        return
    for scene, how in HOW.items():
        if scene in getattr(wp, "WORDS", {}):
            wp.WORDS[scene]["how"] = how
    build0, refresh0 = cls.build, cls.refresh

    def build(self, gui) -> None:
        build0(self, gui)
        self.handle_kit = Handle(self)

    def refresh(self) -> None:
        real = (self.handle, self.hint)
        real[0].visible = real[1].visible = False
        self.handle, self.hint = _Sink(), _Sink()  # today's refresh still sets the readouts; its ball goes nowhere
        try:
            refresh0(self)
        finally:
            self.handle, self.hint = real
        self.handle_kit.refresh()

    refresh._rsw_handle = True
    cls.build, cls.refresh = build, refresh
    cls._rsw_handle = True
