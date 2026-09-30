"""``rsw export``: the scene as the browser draws it, walked the way RTX draws it.

RTX draws *model shapes* (``decorate_stage`` binds ``handles.finishes[s]``, else
``finish_for(colour)``, to every VISIBLE shape instance) and *deformables* from
``handles.deformables`` with ``planar_uvs`` of the first state it renders. So:

* every VISIBLE shape becomes a mesh, geometry from ``shape_geometry`` (Newton's
  own primitive meshing), material from the finish (``gltf.Glb.material``);
* a body that moves in any recorded frame gets a named pose node ``body:<i>``
  carrying its shapes in body coordinates; everything else is baked to world at
  the canonical frame and merged per material under ``static``;
* live shapes (the wipe's ink decals) are left out: the browser draws the ink;
* deformable topology and UVs go to ``deform_topology.npz`` for ``rsw db``
  (never through gltfpack, which reorders vertices);
* the robot's kinematic tree goes to ``kinematics.json`` with FK test vectors taken
  from recorded frames, checked here against the recorded ``body_q``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from . import paths
from .gltf import Glb
from .record import load

PARK_Z = 5.0


# ------------------------------------------------------------------ recordings
def takes(demo: str, rounds=("r1", "smoke")) -> list[Path]:
    out = []
    for r in rounds:
        out += sorted(p for p in (paths.DATA / demo / "takes" / r).glob("s*/take_*") if (p / "meta.json").exists())
    return out


def _spec(demo: str):
    from demos import load as load_spec

    return load_spec(demo)


def is_track(demo: str) -> bool:
    return getattr(_spec(demo), "kind", "free") == "track"


def canonical(demo: str) -> dict[str, np.ndarray]:
    """Frame 0 of the demo's canonical take: the hover pose every render warms up on.

    Pinned in ``export.json`` at export time, because the RTX backend fixes a
    deformable's UVs from the first state it renders and the exported mesh's UVs
    come from the same frame; a later stage that picked "the first take" again
    would pick a different one once more takes exist."""
    if is_track(demo):
        rec = _spec(demo).recording()
        return {"body_q": rec["body_q"][0], "joint_q": None, "deform": None, "_take": rec["_take"]}
    pinned = paths.DATA / demo / "export" / "export.json"
    t = None
    if pinned.exists():
        t = json.loads(pinned.read_text()).get("canonical_take")
    t = Path(t) if t else takes(demo)[0]
    d = load(t)
    return {k: v[0] for k, v in d.items()} | {"_take": str(t)}


def moving_bodies(demo: str, pos_tol: float = 5e-4, ang_tol: float = 2e-3, limit: int = 12) -> tuple[list[int], np.ndarray]:
    """Bodies whose pose varies by more than 0.5 mm / 0.11 deg over the recorded takes."""
    dev_p = None
    dev_q = None
    ref = None
    sources = [_spec(demo).recording()["body_q"]] if is_track(demo) else [load(t)["body_q"] for t in takes(demo)[:limit]]
    for bq in sources:
        bq = np.asarray(bq, dtype=np.float64)
        if ref is None:
            ref = bq[0]
        dp = np.linalg.norm(bq[:, :, :3] - ref[None, :, :3], axis=2).max(axis=0)
        dq = 1.0 - np.abs(np.sum(bq[:, :, 3:7] * ref[None, :, 3:7], axis=2)).min(axis=0)
        dev_p = dp if dev_p is None else np.maximum(dev_p, dp)
        dev_q = dq if dev_q is None else np.maximum(dev_q, dq)
    ang = 2.0 * np.arccos(np.clip(1.0 - dev_q, -1.0, 1.0))
    moving = [int(b) for b in np.flatnonzero((dev_p > pos_tol) | (ang > ang_tol))]
    return moving, dev_p


# ------------------------------------------------------------------- geometry
def _quat_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _apply(pose7, pts: np.ndarray) -> np.ndarray:
    return pts @ _quat_matrix(pose7[3:7]).T + np.asarray(pose7[:3], dtype=np.float64)


def _compose(a7, b7) -> np.ndarray:
    ra = _quat_matrix(a7[3:7])
    p = np.asarray(a7[:3]) + ra @ np.asarray(b7[:3])
    ax, ay, az, aw = a7[3:7]
    bx, by, bz, bw = b7[3:7]
    q = np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                  aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])
    return np.concatenate([p, q / np.linalg.norm(q)])


def _inv(a7) -> np.ndarray:
    qi = np.array([-a7[3], -a7[4], -a7[5], a7[6]])
    return np.concatenate([-(_quat_matrix(qi) @ np.asarray(a7[:3])), qi])


def _shade(points: np.ndarray, faces: np.ndarray, flat: bool):
    """(points, faces, normals, source): flat for boxes/planes, crease-split smooth otherwise.

    ``source[i]`` is the input vertex output vertex ``i`` was split from, so per-vertex data
    (the mesh's own UVs) follows the split. It used not to: the UV count stopped matching the
    split mesh, the plates fell back to box-projected UVs, and their rim art sampled white."""
    import trimesh

    m = trimesh.Trimesh(vertices=points, faces=faces, process=False)
    if flat:
        src = faces.reshape(-1)
        fn = np.repeat(m.face_normals, 3, axis=0)
        return points[src], np.arange(len(src)).reshape(-1, 3), fn, src
    try:
        sm = trimesh.graph.smooth_shade(m, angle=np.radians(35.0))
        sf = np.asarray(sm.faces)
        if sf.shape == faces.shape:
            src = np.full(len(sm.vertices), -1, dtype=np.int64)
            src[sf.reshape(-1)] = faces.reshape(-1)
            if (src >= 0).all() and np.allclose(np.asarray(sm.vertices), points[src]):
                return np.asarray(sm.vertices), sf, np.asarray(sm.vertex_normals), src
    except Exception:  # noqa: BLE001
        pass
    return points, faces, np.asarray(m.vertex_normals), np.arange(len(points))


def export_scene(demo: str, sim, out: Path, moving: list[int], canon: dict) -> dict:
    import newton
    from robosimstudio.core.render.finish import finish_for
    from robosimstudio.core.render.uv import box_projected_uvs, planar_uvs
    from robosimstudio.core.render.visual import shape_geometry

    model = sim.model
    h = sim.handles
    body_q = np.asarray(canon["body_q"], dtype=np.float64)
    shape_body = model.shape_body.numpy()
    shape_type = model.shape_type.numpy()
    shape_scale = model.shape_scale.numpy()
    shape_flags = model.shape_flags.numpy()
    shape_xform = model.shape_transform.numpy()
    colors = model.shape_color.numpy()
    labels = list(getattr(model, "shape_label", None) or [])
    live = {int(s) for s in np.asarray(h.live_shapes).reshape(-1)}
    visible = int(newton.ShapeFlags.VISIBLE)
    mesh_types = {int(newton.GeoType.MESH), int(getattr(newton.GeoType, "CONVEX_MESH", newton.GeoType.MESH))}
    flat_types = {int(newton.GeoType.BOX), int(newton.GeoType.PLANE)}
    moving_set = set(moving)

    glb = Glb()
    per_body: dict[int, list[int]] = {}
    static_groups: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]] = {}
    skipped = {"invisible": 0, "live": 0, "parked": 0, "nogeom": 0}
    tri_count = {"moving": 0, "static": 0}
    for s in range(model.shape_count):
        if not (int(shape_flags[s]) & visible):
            skipped["invisible"] += 1
            continue
        if s in live:
            skipped["live"] += 1
            continue
        b = int(shape_body[s])
        if b >= 0 and float(body_q[b][2]) > PARK_Z:
            skipped["parked"] += 1
            continue
        # A plane's "infinite" stand-in is 100 m; gltfpack quantizes positions on one
        # scene-wide grid, so a 200 m bound made 14-bit steps of 12 mm and the robot
        # looked voxelized. 4 m outruns every camera here.
        geom = shape_geometry(model.shape_source[s], int(shape_type[s]), shape_scale[s], plane_extent=4.0)
        if geom is None:
            skipped["nogeom"] += 1
            continue
        pts, faces = geom
        kind = int(shape_type[s])
        n_src = len(pts)
        pts, faces, normals, from_src = _shade(np.asarray(pts, dtype=np.float64), np.asarray(faces, dtype=np.int64), kind in flat_types)
        finish = h.finishes.get(s) or finish_for(tuple(float(v) for v in colors[s][:3]))
        uvs = None
        if finish.texture or finish.roughness_texture:
            src = model.shape_source[s]
            src_uv = getattr(src, "uvs", None) if kind in mesh_types else None
            if src_uv is not None and len(src_uv) == n_src:
                uvs = np.asarray(src_uv, dtype=np.float64)[from_src]
            else:
                uvs = box_projected_uvs(pts[faces.reshape(-1)], np.arange(faces.size).reshape(-1, 3))
                pts, normals = pts[faces.reshape(-1)], normals[faces.reshape(-1)]
                faces = np.arange(faces.size).reshape(-1, 3)
            scale = np.asarray(finish.uv_scale, dtype=np.float64)
            offset = np.asarray(finish.uv_offset, dtype=np.float64)
            uvs = uvs * scale + offset
        mat = glb.material(colors[s][:3], finish, textured=uvs is not None, name=labels[s] if s < len(labels) else f"shape_{s}")
        local = shape_xform[s]
        if b in moving_set:
            p = _apply(local, pts)
            n = normals @ _quat_matrix(local[3:7]).T
            mesh = glb.mesh(labels[s] if s < len(labels) else f"shape_{s}", p, faces, material=mat, normals=n, uvs=uvs)
            per_body.setdefault(b, []).append(glb.node("", mesh=mesh))
            tri_count["moving"] += len(faces)
        else:
            world = _compose(body_q[b], local) if b >= 0 else np.asarray(local, dtype=np.float64)
            p = _apply(world, pts)
            n = normals @ _quat_matrix(world[3:7]).T
            static_groups.setdefault(mat, []).append((p, faces, n, uvs))
            tri_count["static"] += len(faces)

    static_children = []
    for mat, items in static_groups.items():
        base = 0
        P, F, N, U = [], [], [], []
        textured = any(u is not None for *_, u in items)
        for p, f, n, u in items:
            P.append(p)
            F.append(f + base)
            N.append(n)
            U.append(u if u is not None else np.zeros((len(p), 2)))
            base += len(p)
        mesh = glb.mesh(
            f"static_{mat}", np.concatenate(P), np.concatenate(F), material=mat, normals=np.concatenate(N),
            uvs=np.concatenate(U) if textured else None,
        )
        static_children.append(glb.node("", mesh=mesh))
    glb.node("static", children=static_children, root=True)
    nodes = {}
    for b, kids in sorted(per_body.items()):
        name = f"body:{b}"
        glb.node(name, children=kids, pose7=body_q[b], extras={"body": b}, root=True)
        nodes[name] = b

    raw = out / "scene.raw.glb"
    raw.write_bytes(glb.to_bytes())
    packed = out / "scene.glb"
    cmd = ["npx", "--yes", "gltfpack@1.3.0", "-i", str(raw), "-o", str(packed), "-cc", "-kn", "-km", "-ke", "-vp", "16"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(paths.SITE))
    if r.returncode != 0:
        raise RuntimeError(f"gltfpack failed: {r.stderr[-2000:]}")

    # Deformables: topology and UVs at the canonical state, for rsw db.
    deform = {}
    dcolors = sim._deformable_colors() if hasattr(sim, "_deformable_colors") else {}
    for name, dh in h.deformables.items():
        parts = np.asarray(dh.particles, dtype=np.int64)
        tris = np.asarray(dh.triangles, dtype=np.int64).reshape(-1, 3)
        if not len(parts) or not len(tris):
            continue
        fin = h.deformable_finishes.get(name) or finish_for(tuple(dcolors.get(name, (0.85, 0.85, 0.88))))
        deform[name] = {"particles": parts, "tris": tris, "finish": fin, "color": tuple(dcolors.get(name, (0.85, 0.85, 0.88)))}
    canon_particles = canon.get("deform")
    topo = {}
    for name, d in deform.items():
        rest = None
        if canon_particles is not None and len(canon_particles) == len(d["particles"]):
            rest = np.asarray(canon_particles, dtype=np.float64)
        if rest is None:
            rest = sim.state.particle_q.numpy()[d["particles"]].astype(np.float64)
        topo[name] = {
            "particles": d["particles"],
            "tris": d["tris"],
            "uv": planar_uvs(rest),
            "rest": rest,
            "color": np.asarray(d["color"]),
            "finish": _finish_dict(d["finish"]),
        }
    np.savez(out / "deform_topology.npz", **{f"{n}.{k}": np.asarray(v) if not isinstance(v, dict) else np.asarray(json.dumps(v))
                                              for n, t in topo.items() for k, v in t.items()})
    info = {
        "canonical_take": canon["_take"],
        "nodes": nodes,
        "moving": sorted(per_body),
        "skipped": skipped,
        "triangles": tri_count,
        "materials": len(glb.doc["materials"]),
        "raw_bytes": raw.stat().st_size,
        "glb_bytes": packed.stat().st_size,
        "deformables": {n: {"particles": int(len(t["particles"])), "tris": int(len(t["tris"])), "finish": t["finish"],
                            "color": [float(c) for c in t["color"]]} for n, t in topo.items()},
    }
    (out / "export.json").write_text(json.dumps(info, indent=1))
    (out / "look_rtx.json").write_text(json.dumps(look_rtx(sim), indent=1))
    return info


def look_rtx(sim) -> dict:
    """The RTX light rig this scene renders under, for the browser's matching lights."""
    from robosimstudio.solvers.render import rtx

    look = sim.look
    cell = getattr(sim.handles, "scene_name", "")
    fields = {}
    for k in ("environment", "environment_map", "key_intensity", "ambient", "scenery", "light_direction", "fill_lights"):
        v = getattr(look, k, None)
        fields[k] = v if isinstance(v, (int, float, str, type(None))) else (list(v) if hasattr(v, "__iter__") else str(v))
    center = [0.0, 0.0]
    kw = rtx.RendererRTX.look_kwargs(look)
    try:  # the cell the lab panel is centred on, as Sim hands it to the renderer
        cell_obj = sim._scene_facts([])["cell"]()
        if cell_obj is not None:
            center = [float(v) for v in getattr(cell_obj, "center", (0.0, 0.0, 0.0))[:2]]
    except Exception as e:  # noqa: BLE001
        print(f"[export] no cell centre ({e}); panel at the origin")
    return {
        "look": fields,
        "cell": cell,
        "surface_z": float(sim.handles.surface_z),
        "panel_center": [*center, float(sim.handles.surface_z) + 1.7],
        "panel_size": [1.4, 0.9],
        "panel_per_key": float(rtx.LAB_PANEL_PER_KEY),
        "dome_per_ambient": float(rtx.LAB_DOME_PER_AMBIENT),
        "tonemap": rtx.TONEMAP_OPERATOR,
        "palette_is_display": bool(rtx.PALETTE_IS_DISPLAY),
        "kwargs": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in kw.items()},
    }


def _finish_dict(f) -> dict:
    return {
        "roughness": float(f.roughness),
        "metallic": float(f.metallic),
        "opacity": float(f.opacity),
        "texture": f.texture or "",
        "roughness_texture": f.roughness_texture or "",
        "uv_scale": [float(v) for v in f.uv_scale],
        "uv_offset": [float(v) for v in f.uv_offset],
    }


# ------------------------------------------------------------------ kinematics
_JT = None


def _joint_type_names() -> dict[int, str]:
    import newton

    jt = newton.JointType
    names = {}
    for key, name in (("PRISMATIC", "prismatic"), ("REVOLUTE", "revolute"), ("BALL", "ball"), ("FIXED", "fixed"),
                      ("FREE", "free"), ("DISTANCE", "distance"), ("D6", "d6"), ("CABLE", "cable")):
        if hasattr(jt, key):
            names[int(getattr(jt, key))] = name
    return names


def export_kinematics(sim, out: Path, frames: dict[str, np.ndarray], n_tests: int = 200) -> dict:
    """The robot's joints (every articulation holding an arm's tool body), FK-checked on recorded frames."""
    model = sim.model
    h = sim.handles
    jtype = model.joint_type.numpy()
    jparent = model.joint_parent.numpy()
    jchild = model.joint_child.numpy()
    jXp = model.joint_X_p.numpy()
    jXc = model.joint_X_c.numpy()
    jaxis = model.joint_axis.numpy()
    qstart = model.joint_q_start.numpy()
    qdstart = model.joint_qd_start.numpy()
    art_start = model.articulation_start.numpy()
    names = _joint_type_names()
    labels = list(getattr(model, "body_label", None) or [])

    def art_of(j: int) -> int:
        return int(np.searchsorted(art_start, j, side="right") - 1)

    arts = set()
    for arm in h.arms.values():
        js = np.flatnonzero(jchild == int(arm.ee_body))
        if len(js):
            arts.add(art_of(int(js[0])))
    joints = [j for j in range(model.joint_count) if art_of(j) in arts]
    bodies = sorted({int(jchild[j]) for j in joints})
    q_index: list[int] = []
    jout = []
    for j in joints:
        t = names.get(int(jtype[j]), f"type{int(jtype[j])}")
        if t not in ("revolute", "prismatic", "fixed", "free", "ball"):
            raise RuntimeError(f"joint {j} has unsupported type {t}")
        q0 = int(qstart[j])
        nq = int(qstart[j + 1] - qstart[j]) if j + 1 < len(qstart) else 0
        ax_idx = int(qdstart[j])
        axis = jaxis[ax_idx] if t in ("revolute", "prismatic") and ax_idx < len(jaxis) else np.zeros(3)
        jout.append({
            "type": t, "parent": int(jparent[j]), "child": int(jchild[j]),
            "X_p": [float(v) for v in jXp[j]], "X_c": [float(v) for v in jXc[j]],
            "axis": [float(v) for v in axis], "q_start": q0, "nq": nq,
        })
        q_index += list(range(q0, q0 + nq))
    q_index = sorted(set(q_index))
    for jo in jout:  # CONTRACTS §2: q_start is Newton's own joint_q_start into q_full, for every joint
        del jo["nq"]
    tcp = {}
    for name, arm in h.arms.items():
        rot = getattr(arm, "ee_offset_rot", None)
        q = [0.0, 0.0, 0.0, 1.0] if rot is None else [float(v) for v in rot]
        tcp[name] = {"body": int(arm.ee_body), "offset": [*(float(v) for v in arm.ee_offset), *q]}

    jq0 = np.asarray(frames["joint_q"][0], dtype=np.float64)
    doc = {
        "q_index": q_index,
        "q_full": [float(v) for v in jq0],
        "joints": jout,
        "bodies": [{"index": b, "label": labels[b] if b < len(labels) else f"body_{b}"} for b in bodies],
        "roots": {},
        "tcp": tcp,
        "tests": [],
    }
    # FK parity on recorded frames: our FK (same math as the browser) vs recorded body_q.
    jq = np.asarray(frames["joint_q"], dtype=np.float64)
    bq = np.asarray(frames["body_q"], dtype=np.float64)
    pick = np.linspace(0, len(jq) - 1, min(n_tests, len(jq))).astype(int)
    worst_p, worst_a = 0.0, 0.0
    for i in pick:
        qr = jq[i][q_index]
        poses = fk(doc, qr)
        err_p = max(float(np.linalg.norm(poses[b][:3] - bq[i][b][:3])) for b in bodies)
        err_a = max(2 * float(np.arccos(min(1.0, abs(float(np.dot(poses[b][3:7], bq[i][b][3:7])))))) for b in bodies)
        worst_p, worst_a = max(worst_p, err_p), max(worst_a, err_a)
        doc["tests"].append({"q": [float(v) for v in qr], "poses": {str(b): [float(v) for v in bq[i][b]] for b in bodies}})
    doc["parity"] = {"max_pos_m": worst_p, "max_ang_rad": worst_a, "frames": int(len(pick))}
    (out / "kinematics.json").write_text(json.dumps(doc))
    return doc["parity"] | {"bodies": len(bodies), "nq": len(q_index)}


def _joint_motion(t: str, axis, q: np.ndarray, qs: int) -> np.ndarray:
    if t == "revolute":
        a = np.asarray(axis, dtype=np.float64)
        a = a / (np.linalg.norm(a) or 1.0)
        s = np.sin(q[qs] / 2.0)
        return np.array([0.0, 0.0, 0.0, a[0] * s, a[1] * s, a[2] * s, np.cos(q[qs] / 2.0)])
    if t == "prismatic":
        a = np.asarray(axis, dtype=np.float64)
        return np.array([*(a * q[qs]), 0.0, 0.0, 0.0, 1.0])
    if t == "free":
        return np.asarray(q[qs : qs + 7], dtype=np.float64)
    if t == "ball":
        return np.array([0.0, 0.0, 0.0, *q[qs : qs + 4]])
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def fk(doc: dict, q_robot: np.ndarray) -> dict[int, np.ndarray]:
    """The browser's FK (site/src/core/kinematics.ts): q_robot fills q_full at q_index."""
    q = np.asarray(doc["q_full"], dtype=np.float64).copy()
    q[np.asarray(doc["q_index"], dtype=np.int64)] = q_robot
    poses: dict[int, np.ndarray] = {}
    ident = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    for j in doc["joints"]:
        parent = j["parent"]
        base = ident if parent < 0 else poses.get(parent, np.asarray(doc["roots"].get(str(parent), ident)))
        xj = _joint_motion(j["type"], j["axis"], q, j["q_start"])
        x = _compose(_compose(_compose(base, np.asarray(j["X_p"])), xj), _inv(np.asarray(j["X_c"])))
        poses[j["child"]] = x
    return poses
