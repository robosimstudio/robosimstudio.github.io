"""Render-side stages: the static plate, the ink map, the still set, the attract video.

All RTX frames come from recorded states (never re-simulated). Each session warms
up on the demo's canonical frame first (deformable UVs, see render.py).
"""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from queue import Queue

import numpy as np

from . import export, paths, render, rssb
from .record import load


# ---------------------------------------------------------------------- helpers
def _static(demo: str, take: Path) -> dict:
    return json.loads((Path(take).parent / "static.json").read_text())


def _exp(demo: str) -> dict:
    return json.loads((paths.DATA / demo / "export" / "export.json").read_text())


def _sponge(static: dict) -> np.ndarray | None:
    for key in ("sponge_particles", "film_particles", "cloth_particles"):
        if key in static:
            return np.asarray(static[key], dtype=np.int64)
    return None


def session(demo: str, gpu: int, width=1920, height=1080, accumulate=64):
    from demos import load as load_spec

    spec = load_spec(demo)
    canon = export.canonical(demo)
    st = spec.static() if getattr(spec, "kind", "free") == "track" else _static(demo, Path(canon["_take"]))
    cam = render.hero_camera(spec.hero(st), width, height)
    sess = render.RtxSession(demo, gpu, [cam], accumulate=accumulate, spec=spec)
    return spec, canon, st, cam, sess


def _ink_colors(spec, sess, intensity):
    return spec.ink_live(sess.sim, intensity) if hasattr(spec, "ink_live") else None


def _write_frame(spec, sess, st, body_q, joint_q, deform, intensity=None):
    live = None
    if intensity is not None and hasattr(spec, "ink_live"):
        live = spec.ink_live(sess.sim, intensity)
    idx = _sponge(st)
    parts = {"d": (idx, deform)} if (idx is not None and deform is not None) else None
    return sess.write(body_q, joint_q, parts, live=live)


def _png(path: Path, rgb: np.ndarray) -> None:
    render.save_png(path, rgb)


def _png_writer(q: Queue):
    while True:
        item = q.get()
        if item is None:
            return
        path, rgb = item
        _png(path, rgb)


# ------------------------------------------------------------------------ plate
def ink_cells_world(spec, sim, canon) -> tuple[np.ndarray, dict]:
    """World positions of the ink cells' top faces, and the bar geometry, at the canonical frame."""
    taint = sim.scene.by_name["taint"]
    sp = taint.spec
    bar = np.asarray(canon["body_q"][int(taint.bar_body)], dtype=np.float64)
    R = export._quat_matrix(bar[3:7])
    cells = np.asarray(taint._cell_ut_np, dtype=np.float64)
    top = float(sp.bar_radius) + float(sp.decal_lift) + float(sp.decal_thickness)
    local = np.stack([top * np.cos(cells[:, 1]), top * np.sin(cells[:, 1]), cells[:, 0]], 1)
    world = local @ R.T + bar[:3]
    geo = {"bar_pose": bar.tolist(), "radius_top": top, "pitch": float(sp.cell_pitch), "cells": cells.tolist(),
           "taint": [float(v) for v in sp.taint_color], "clear": [float(v) for v in sp.clear_color]}
    return world, geo


def project(cam, pts: np.ndarray) -> np.ndarray:
    W = np.asarray(cam.world_matrix, dtype=np.float64)  # cam->world, +z fwd +y down
    R, t = W[:3, :3], W[:3, 3]
    pc = (pts - t) @ R  # world->cam
    K = cam.intrinsics
    return np.stack([K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2], pc[:, 2]], 1)


def inkmap(cam, geo: dict, bbox: list[int]) -> np.ndarray:
    """Per pixel of ``bbox``: the ink cell its ray hits on the bar's decal surface, or 0xFFFF.

    Pixel (i, j) looks along K^-1 (i, j, 1): with cx = W/2 - 0.5 the pixel centres
    sit on integer coordinates (checked against an RTX checkerboard render)."""
    x0, y0, w, h = bbox
    K = cam.intrinsics
    W = np.asarray(cam.world_matrix, dtype=np.float64)
    jj, ii = np.mgrid[y0 : y0 + h, x0 : x0 + w]
    d_cam = np.stack([(ii - K[0, 2]) / K[0, 0], (jj - K[1, 2]) / K[1, 1], np.ones_like(ii, dtype=np.float64)], -1)
    d_w = d_cam @ W[:3, :3].T
    o_w = W[:3, 3]
    bar = np.asarray(geo["bar_pose"])
    Rb = export._quat_matrix(bar[3:7])
    o = (o_w - bar[:3]) @ Rb
    d = d_w @ Rb
    r = geo["radius_top"]
    a = d[..., 0] ** 2 + d[..., 1] ** 2
    b = 2 * (o[0] * d[..., 0] + o[1] * d[..., 1])
    c = o[0] ** 2 + o[1] ** 2 - r * r
    disc = b * b - 4 * a * c
    ok = disc >= 0
    t = np.where(ok, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(a, 1e-12)), np.inf)
    p = o + d * t[..., None]
    u, ang = p[..., 2], np.arctan2(p[..., 1], p[..., 0])
    cells = np.asarray(geo["cells"])
    pitch = geo["pitch"]
    out = np.full((h, w), 0xFFFF, dtype=np.uint16)
    du = np.abs(u[..., None] - cells[None, None, :, 0])
    da = np.abs(((ang[..., None] - cells[None, None, :, 1]) + np.pi) % (2 * np.pi) - np.pi) * r
    inside = ok[..., None] & (du <= pitch / 2) & (da <= pitch / 2)
    hit = inside.any(axis=-1)
    out[hit] = np.argmax(inside, axis=-1)[hit].astype(np.uint16)
    return out


def plate(demo: str, gpu: int) -> dict:
    spec, canon, st, cam, sess = session(demo, gpu, accumulate=64)
    out = paths.demo_dir(demo, "plate")
    exp = _exp(demo)
    n_cells = len(st["ink"]["cells_ut"]) if "ink" in st else 0
    full = np.ones(n_cells) if n_cells else None
    # warm up on the canonical frame (fixes deformable UVs), then hide what moves
    s0 = _write_frame(spec, sess, st, canon["body_q"], canon["joint_q"], canon["deform"], full)
    sess.render(s0, passes=1)
    bq = np.asarray(canon["body_q"], dtype=np.float32).copy()
    for b in exp["moving"]:
        bq[b, 2] += render.HIDE_Z
    hidden = np.asarray(canon["deform"], dtype=np.float32) + np.array([0, 0, render.HIDE_Z], dtype=np.float32)
    s1 = _write_frame(spec, sess, st, bq, canon["joint_q"], hidden, full)
    img = sess.render(s1, passes=6)[cam.name]
    _png(out / "plate.png", img["rgb"])
    np.save(out / "plate_depth.npy", img["depth"].astype(np.float32))
    info = {"camera": render.camera_json(cam)}
    if n_cells:
        cells_w, geo = ink_cells_world(spec, sess.sim, canon)
        px = project(cam, cells_w)
        m = 24
        x0, y0 = int(max(0, np.floor(px[:, 0].min()) - m)), int(max(0, np.floor(px[:, 1].min()) - m))
        x1, y1 = int(min(cam.width, np.ceil(px[:, 0].max()) + m)), int(min(cam.height, np.ceil(px[:, 1].max()) + m))
        bbox = [x0, y0, x1 - x0, y1 - y0]
        s2 = _write_frame(spec, sess, st, bq, canon["joint_q"], hidden, np.zeros(n_cells))
        clean = sess.render(s2, passes=6)[cam.name]["rgb"]
        _png(out / "plate_clean.png", clean)
        imap = inkmap(cam, geo, bbox)
        rssb.write(out / "inkmap.rssb.gz", {"cell": imap}, {"bbox": bbox})
        info.update(bbox=bbox, ink=geo, inkmap_hits=int((imap != 0xFFFF).sum()))
    sess.close()
    (out / "plate.json").write_text(json.dumps(info, indent=1))
    return info


# --------------------------------------------------------------------- envprobe
FACES = (  # forward, up: six 90-degree views that tile the sphere
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)), ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),
)


def equirect_from_faces(faces: list[np.ndarray], cams: list[dict], width: int = 1024) -> np.ndarray:
    """An equirect in three.js's convention (Y-up env space; image row 0 = up) of what the six
    faces see. The page rotates world Z-up into that space with ``environmentRotation``
    (pi/2 about x), which three.js applies inverted: an env direction e is the world direction
    (e_x, -e_z, e_y). Each direction samples the face it points into most, bilinearly."""
    h = width // 2
    v = 1.0 - (np.arange(h) + 0.5) / h
    u = (np.arange(width) + 0.5) / width
    theta = (v - 0.5) * np.pi
    phi = (u - 0.5) * 2.0 * np.pi
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    e = np.stack([np.cos(th) * np.cos(ph), np.sin(th), np.cos(th) * np.sin(ph)], axis=-1)
    w = np.stack([e[..., 0], -e[..., 2], e[..., 1]], axis=-1)
    fwd = np.array([f for f, _ in FACES])
    pick = np.argmax(w @ fwd.T, axis=-1)
    out = np.zeros((h, width, 3), dtype=np.float32)
    for k, (img, cam) in enumerate(zip(faces, cams)):
        m = pick == k
        if not m.any():
            continue
        c2w = np.asarray(cam["cam_to_world_cv"], dtype=np.float64).reshape(4, 4, order="F")
        pc = w[m] @ c2w[:3, :3]  # world direction -> camera (CV: x right, y down, z forward)
        x = cam["fx"] * pc[:, 0] / pc[:, 2] + cam["cx"]
        y = cam["fy"] * pc[:, 1] / pc[:, 2] + cam["cy"]
        H, W = img.shape[:2]
        x = np.clip(x, 0, W - 1.001)
        y = np.clip(y, 0, H - 1.001)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        ax, ay = (x - x0)[:, None], (y - y0)[:, None]
        f = img.astype(np.float32)
        out[m] = ((f[y0, x0] * (1 - ax) + f[y0, x0 + 1] * ax) * (1 - ay)
                  + (f[y0 + 1, x0] * (1 - ax) + f[y0 + 1, x0 + 1] * ax) * ay)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def envprobe(demo: str, gpu: int, size: int = 512, lift: float = 0.10) -> dict:
    """What the robot's surfaces reflect, as RTX renders it: six 90-degree faces from the middle
    of the moving bodies (they and the deformable hidden, as for the plate), resampled into an
    equirect the live layer uses as its environment. The generic RoomEnvironment reflected a
    room this scene does not have; RTX lights the arm with its own dome, panel and walls."""
    from demos import load as load_spec
    from robosimstudio.core.cameras import CameraSpec

    spec = load_spec(demo)
    canon = export.canonical(demo)
    st = spec.static() if getattr(spec, "kind", "free") == "track" else _static(demo, Path(canon["_take"]))
    exp = _exp(demo)
    bq0 = np.asarray(canon["body_q"], dtype=np.float64)
    point = bq0[exp["moving"], :3].mean(axis=0) + np.array([0.0, 0.0, lift])
    f = size / 2.0  # 90 degrees
    cams = [CameraSpec(name=f"f{k}", width=size, height=size, fx=f, fy=f, cx=size / 2 - 0.5, cy=size / 2 - 0.5,
                       near=0.02, far=30.0, eye=tuple(point), look_at=tuple(point + np.asarray(fw)), up=up)
            for k, (fw, up) in enumerate(FACES)]
    sess = render.RtxSession(demo, gpu, cams, accumulate=64, spec=spec)
    s0 = _write_frame(spec, sess, st, canon["body_q"], canon["joint_q"], canon["deform"])
    sess.render(s0, passes=1)
    bq = np.asarray(canon["body_q"], dtype=np.float32).copy()
    for b in exp["moving"]:
        bq[b, 2] += render.HIDE_Z
    hidden = None if canon["deform"] is None else np.asarray(canon["deform"], dtype=np.float32) + np.array([0, 0, render.HIDE_Z], dtype=np.float32)
    s1 = _write_frame(spec, sess, st, bq, canon["joint_q"], hidden)
    img = sess.render(s1, passes=6)
    sess.close()
    out = paths.demo_dir(demo, "envprobe")
    faces = [render._u8(img[c.name]["rgb"]) for c in cams]
    for k, fimg in enumerate(faces):
        _png(out / f"face{k}.png", fimg)
    camj = [render.camera_json(c) for c in cams]
    eq = equirect_from_faces(faces, camj)
    _png(out / "env.png", eq)
    info = {"point": point.tolist(), "size": size, "faces": [list(map(list, fu)) for fu in FACES], "mean": float(eq.mean())}
    (out / "envprobe.json").write_text(json.dumps(info, indent=1))
    return info


# ----------------------------------------------------------------------- stills
def _frame_source(demo: str):
    """db frame -> (take path, 60 Hz step)."""
    db = json.loads((paths.DATA / demo / "db" / "db.json").read_text())
    tab = db["takes"]
    tp = db["take_paths"]

    def where(f: int):
        for (s, n, _k), p in zip(tab, tp):
            if s <= f < s + n:
                return Path(p), 2 * (f - s)
        raise IndexError(f)

    return db, where


def stills(demo: str, gpu: int, shard: int = 0, nshards: int = 1, accumulate: int = 64, passes: int = 3) -> dict:
    spec, canon, st, cam, sess = session(demo, gpu, accumulate=accumulate)
    db, where = _frame_source(demo)
    out = paths.demo_dir(demo, "stills")
    frames = db["stills"][shard::nshards]
    n_cells = len(st["ink"]["cells_ut"]) if "ink" in st else 0
    full = np.ones(n_cells) if n_cells else None
    s0 = _write_frame(spec, sess, st, canon["body_q"], canon["joint_q"], canon["deform"], full)
    sess.render(s0, passes=1)
    pinfo = json.loads((paths.DATA / demo / "plate" / "plate.json").read_text()) if n_cells else {}
    q: Queue = Queue(maxsize=8)
    th = threading.Thread(target=_png_writer, args=(q,), daemon=True)
    th.start()
    cache: dict[str, dict] = {}
    done = 0
    # A still is filed under its index in db.json's still list, and a rebuilt database can renumber
    # frames. Each image's source (take, 60 Hz step) is kept in a sidecar; one whose source no longer
    # matches the database is rendered again, one that matches is kept.
    side_path = out / f"sources.{shard}.json"
    sources = still_sources(demo)
    done_src = {}
    for sp in out.glob("sources.*.json"):
        done_src.update(json.loads(sp.read_text()))
    mine = {}
    for f in frames:
        sid = db["stills"].index(f)
        dst = out / f"{sid:04d}.png"
        mine[str(sid)] = sources[sid]
        if dst.exists() and done_src.get(str(sid)) == sources[sid]:
            continue
        take, step = where(f)
        if str(take) not in cache:
            cache.clear()
            cache[str(take)] = load(take)
        d = cache[str(take)]
        s = _write_frame(spec, sess, st, d["body_q"][step], d["joint_q"][step], d["deform"][step], full)
        rgb = sess.render(s, passes=passes)[cam.name]["rgb"]
        if n_cells:
            x0, y0, w, h = pinfo["bbox"]
            s2 = _write_frame(spec, sess, st, d["body_q"][step], d["joint_q"][step], d["deform"][step], np.zeros(n_cells))
            clean = sess.render(s2, passes=passes)[cam.name]["rgb"]
            q.put((out / f"{sid:04d}.clean.png", clean[y0 : y0 + h, x0 : x0 + w].copy()))
        q.put((dst, rgb))
        done += 1
        if done % 20 == 0:
            print(f"[stills] {demo} shard {shard}: {done}/{len(frames)}", flush=True)
    q.put(None)
    th.join()
    sess.close()
    side_path.write_text(json.dumps(mine))
    return {"rendered": done, "shard": shard}


def still_sources(demo: str) -> list[list]:
    """Each still's source in the current database: [take path, 60 Hz step], by still index."""
    db, where = _frame_source(demo)
    out = []
    for f in db["stills"]:
        take, step = where(int(f))
        out.append([str(take), int(step)])
    return out


# ---------------------------------------------------------------------- attract
def attract(demo: str, gpu: int, accumulate: int = 32, passes: int = 1) -> dict:
    spec, canon, st, cam, sess = session(demo, gpu, accumulate=accumulate)
    db, where = _frame_source(demo)
    f0, f1 = db["attract_frames"]
    out = paths.demo_dir(demo, "attract")
    n_cells = len(st["ink"]["cells_ut"]) if "ink" in st else 0
    full = np.ones(n_cells) if n_cells else None
    s0 = _write_frame(spec, sess, st, canon["body_q"], canon["joint_q"], canon["deform"], full)
    sess.render(s0, passes=1)
    take, _ = where(f0)
    d = load(take)
    ff = paths.ffmpeg()
    mp4 = out / "attract.mp4"
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{cam.width}x{cam.height}", "-r", "30",
           "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-c:v", "libx264", "-preset", "slow",
           "-crf", "21", "-g", "15", "-bf", "0", "-sc_threshold", "0", "-profile:v", "high", "-colorspace", "bt709",
           "-color_primaries", "bt709", "-color_trc", "iec61966-2-1", "-movflags", "+faststart", str(mp4)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    inten = np.ones(n_cells) if n_cells else None
    rates = d.get("ink_rate")
    frames_png = paths.demo_dir(demo, "attract", "frames")
    for k, f in enumerate(range(f0, f1)):
        _, step = where(f)
        s = _write_frame(spec, sess, st, d["body_q"][step], d["joint_q"][step], d["deform"][step], inten)
        rgb = sess.render(s, passes=passes)[cam.name]["rgb"]
        proc.stdin.write(np.ascontiguousarray(rgb).tobytes())
        if k % 15 == 0:
            _png(frames_png / f"{k:05d}.png", rgb)
        if inten is not None:
            for st_ in (step, step + 1):
                if st_ < len(rates):
                    live = inten > 0
                    inten[live] = np.maximum(0.0, inten[live] - rates[st_][live] / 60.0)
    proc.stdin.close()
    proc.wait()
    sess.close()
    return {"frames": f1 - f0, "mp4": str(mp4), "bytes": mp4.stat().st_size}


# ------------------------------------------------------------------------ tiles
def tiles(demo: str, start_s: float = 0.0, seconds: float = 5.0) -> dict:
    """The picker's loop (480x270) and poster, cut from the demo's RTX video."""
    src = paths.DATA / demo / "attract" / "attract.mp4"
    if not src.exists():
        src = paths.DATA / demo / "track" / "rollout.mp4"
    out = paths.demo_dir(demo, "tiles")
    ff = paths.ffmpeg()
    subprocess.run([ff, "-y", "-loglevel", "error", "-ss", f"{start_s:.2f}", "-t", f"{seconds:.2f}", "-i", str(src),
                    "-vf", "scale=480:270:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-preset", "slow", "-crf", "27",
                    "-g", "30", "-an", "-movflags", "+faststart", str(out / "tile.mp4")], check=True)
    png = out / "poster.png"
    subprocess.run([ff, "-y", "-loglevel", "error", "-ss", f"{start_s:.2f}", "-i", str(src), "-frames:v", "1",
                    "-vf", "scale=960:540:flags=lanczos", str(png)], check=True)
    from PIL import Image

    Image.open(png).convert("RGB").save(out / "poster.webp", format="WEBP", quality=80, method=6)
    return {"tile": (out / "tile.mp4").stat().st_size, "poster": (out / "poster.webp").stat().st_size}
