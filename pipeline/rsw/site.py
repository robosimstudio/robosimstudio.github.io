"""``rsw site``: assemble ``build/site-data/demos/<id>/`` for the browser.

Every file but ``manifest.json`` is renamed with a content hash (GitHub Pages sends
``max-age=600`` for everything). PNG frames become WebP here. The manifest follows
pipeline/schema/CONTRACTS.md §5.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import paths

OUT = paths.REPO / "build" / "site-data"

TITLES = {
    "wipe": "Wipe the bar",
    "peel": "Peel the film",
    "whip": "Whip the cable",
    "towel": "Shake the towel",
    "gears": "Gear train",
    "plates": "Plate rack",
    "weave": "Cable weave",
    "conduit": "Thread the conduit",
    "scissors": "Hang the scissors",
    "usbc": "Plug in USB-C",
    "syringe": "Dispose of the syringe",
    "crate": "Flip the crate",
}


MATCH_WEIGHTS = {"fd": 1.0, "hv": 0.6, "dm": 0.25, "dv": 0.1}
"""Relative weights of the standardized feature groups, tuned with ``site/tests/e2e/harness.py``
on the wipe (2026-09-30). The hand position is weighted by a physical scale instead
(``hp_scale``: that many metres of mismatch cost one unit): standardized by its own spread
(38 mm on the wipe), a 27 mm-away frame with a matching future beat a nearby one, and every such
pick was a visible jump (mean 27 mm; 3.4 mm at a 4 mm scale, settles 5/12 -> 12/12)."""
HP_SCALE = 0.004


def _weigh(features: dict, hp_scale: float, overrides: dict | None = None) -> dict:
    out = json.loads(json.dumps(features))
    rel = {**MATCH_WEIGHTS, **(overrides or {})}
    for g in out["groups"]:
        if g["name"] == "hp":
            g["w"] = round((float(g["sigma"]) / hp_scale) ** 2, 3)
        elif g["name"] in rel:
            g["w"] = rel[g["name"]]
    return out


def _hash_name(src: Path, stem: str, suffix: str) -> str:
    h = hashlib.sha256(src.read_bytes()).hexdigest()[:10]
    return f"{stem}.{h}{suffix}"


def _put(src: Path, dst_dir: Path, stem: str, suffix: str | None = None) -> str:
    suffix = suffix or "".join(src.suffixes)
    name = _hash_name(src, stem, suffix)
    dst = dst_dir / name
    if not dst.exists():
        shutil.copyfile(src, dst)
    return name


def _texture_file(key: str) -> Path | None:
    """A finish texture on disk: a path, or a procedural one RTX has already rendered into the
    texture cache (a suite registers its generators only when its task loads, so a bare
    ``texture_path`` cannot render them here)."""
    if Path(key).is_file():
        return Path(key)
    from robosimstudio.core.render.finish import texture_cache_dir

    found = sorted(Path(texture_cache_dir()).glob(f"{key}_v*.png"))
    return found[-1] if found else None


def _deform_textures(fin: dict, out: Path) -> dict:
    """The deformable's finish textures, shipped beside it, for the live layer."""
    tex = {}
    for field, key in (("map", "texture"), ("roughness_map", "roughness_texture")):
        name = fin.get(key)
        if not name:
            continue
        src = _texture_file(str(name))
        if src is None:
            print(f"[site] warning: texture {name!r} not found; the live {key} is left out")
            continue
        tex[field] = _put(src, out, f"tex_{Path(src).stem}", ".png")
    return tex


def _attach_env(look: dict, D: Path, out: Path, tmp: Path) -> None:
    """The RTX environment probe (``rsw envprobe``), shipped for the live layer's reflections."""
    src = D / "envprobe" / "env.png"
    if not src.exists():
        return
    to_webp([(src, tmp / "env.webp")], 92)
    look["env_map"] = _put(tmp / "env.webp", out, "env", ".webp")


def _webp(args) -> None:
    src, dst, q = args
    from PIL import Image

    # keyed by name (s0000.webp), so a re-rendered source must win over the cached conversion
    if Path(dst).exists() and Path(dst).stat().st_mtime >= Path(src).stat().st_mtime:
        return
    Image.open(src).convert("RGB").save(dst, format="WEBP", quality=q, method=6)


def to_webp(pairs: list[tuple[Path, Path]], quality: int, workers: int = 32) -> None:
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_webp, [(str(s), str(d), quality) for s, d in pairs]))


def _roi(cam: dict, pts: np.ndarray, margin: float = 0.22) -> list[int]:
    W = np.asarray(cam["cam_to_world_cv"], dtype=np.float64).reshape(4, 4).T
    R, t = W[:3, :3], W[:3, 3]
    pc = (pts - t) @ R
    u = cam["fx"] * pc[:, 0] / pc[:, 2] + cam["cx"]
    v = cam["fy"] * pc[:, 1] / pc[:, 2] + cam["cy"]
    x0, x1, y0, y1 = np.percentile(u, 1), np.percentile(u, 99), np.percentile(v, 1), np.percentile(v, 99)
    mx, my = (x1 - x0) * margin + 60, (y1 - y0) * margin + 80
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(cam["width"], x1 + mx), min(cam["height"], y1 + my)
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def assemble_free(demo: str) -> dict:
    from demos import load as load_spec

    spec = load_spec(demo)
    D = paths.DATA / demo
    out = OUT / "demos" / demo
    if out.exists():
        shutil.rmtree(out)  # every file is re-derived; stale hashed names must not linger
    out.mkdir(parents=True)
    exp = json.loads((D / "export" / "export.json").read_text())
    db = json.loads((D / "db" / "db.json").read_text())
    plate = json.loads((D / "plate" / "plate.json").read_text())
    canon_take = Path(exp["canonical_take"])
    static = json.loads((canon_take.parent / "static.json").read_text())
    cam = plate["camera"]
    files: dict = {}

    files["glb"] = _put(D / "export" / "scene.glb", out, "scene", ".glb")
    files["kin"] = _put(D / "export" / "kinematics.json", out, "kinematics", ".json")
    files["deform"] = _put(D / "db" / "deform.rssb.gz", out, "deform", ".rssb.gz")
    files["db"] = [_put(D / "db" / c["file"], out, f"db{i}", ".rssb.gz") for i, c in enumerate(db["chunks"])]

    tmp = D / "webp"
    tmp.mkdir(exist_ok=True)
    to_webp([(D / "plate" / "plate.png", tmp / "plate.webp")], 92)
    files["plate"] = _put(tmp / "plate.webp", out, "plate", ".webp")

    still_dir = D / "stills"
    db_time = (D / "db" / "db.json").stat().st_mtime
    stale = [p.name for p in still_dir.glob("*.png") if p.stat().st_mtime <= db_time]
    if stale:
        raise SystemExit(f"[site] {demo}: {len(stale)} stills predate db.json (e.g. {stale[0]}): the database was "
                         "rebuilt after they were rendered, so they may show other frames -- re-run rsw stills")
    pairs = []
    for i in range(len(db["stills"])):
        src = still_dir / f"{i:04d}.png"
        if src.exists():
            pairs.append((src, tmp / f"s{i:04d}.webp"))
            clean = still_dir / f"{i:04d}.clean.png"
            if clean.exists():
                pairs.append((clean, tmp / f"s{i:04d}.clean.webp"))
    to_webp(pairs, 82)
    sdir = out / "stills"
    sdir.mkdir(exist_ok=True)
    still_files, still_clean, still_frames = [], [], []
    for i, f in enumerate(db["stills"]):
        w = tmp / f"s{i:04d}.webp"
        if not w.exists():
            continue
        still_files.append("stills/" + _put(w, sdir, f"{i:04d}", ".webp"))
        c = tmp / f"s{i:04d}.clean.webp"
        still_clean.append("stills/" + _put(c, sdir, f"{i:04d}c", ".webp") if c.exists() else None)
        still_frames.append(int(f))

    manifest: dict = {
        "id": demo,
        "title": TITLES.get(demo, demo),
        "kind": "free",
        "fps": 30,
        "frames": int(db["frames"]),
        "camera": cam,
        "nodes": exp["nodes"],
        "moving": exp["moving"],
        "look": look_from_rtx(json.loads((D / "export" / "look_rtx.json").read_text()), D),
        "ctrl": {
            **static["osc"],
            "plant": {"alpha": db["plant"]["alpha"], "beta": db["plant"]["beta"]},
            "hand_quat": static["hand_quat"],
        },
        "drag": {
            **dict(zip(("origin", "A", "B"), spec.drag_affine(static))),
            **dict(zip(("a", "b"), spec.drag_box(static) if hasattr(spec, "drag_box") else (static["box"]["a"], static["box"]["b"]))),
            "hover": static["hover_ab"],
            "coast_tau": 0.3,
            "v_max": 1.5,
            **({"one_way_slack": float(spec.one_way_slack)} if getattr(spec, "one_way_slack", None) is not None else {}),
        },
        "features": _weigh(db["features"], float(getattr(spec, "hp_scale", HP_SCALE)), getattr(spec, "match_weights", None)),
        "match": {"every_frames": 3, "accept_ratio": 0.85, "accept_margin": 0.05, "min_gap_frames": 6,
                  "same_take_window": 15, "min_runway": 10, "start_guard": 30},
        "settle": {"enter_speed": 0.005, "enter_s": 0.15, "exit_speed": 0.015, "max_wait_s": 1.5, "dissolve_s": 0.4},
        "stills": {"files": still_files, "clean": still_clean, "frames": still_frames},
        "files": files,
    }
    if db.get("ink"):
        files["plate_clean"] = None
        x0, y0, w, h = plate["bbox"]
        from PIL import Image

        crop = Image.open(D / "plate" / "plate_clean.png").crop((x0, y0, x0 + w, y0 + h))
        crop.save(tmp / "plate_clean.png")
        to_webp([(tmp / "plate_clean.png", tmp / "plate_clean.webp")], 92)
        files["plate_clean"] = _put(tmp / "plate_clean.webp", out, "plate_clean", ".webp")
        files["inkmap"] = _put(D / "plate" / "inkmap.rssb.gz", out, "inkmap", ".rssb.gz")
        manifest["ink"] = {
            "cells": plate["ink"]["cells"], "pitch": plate["ink"]["pitch"], "radius": plate["ink"]["radius_top"],
            "taint": plate["ink"]["taint"], "clear": plate["ink"]["clear"], "bbox": plate["bbox"],
            "r0": db["ink"]["r0"], "rmax": db["ink"]["rmax"],
        }
    from . import rssb

    dmeta = rssb.read(D / "db" / "deform.rssb.gz")[1]
    if dmeta.get("kind") == "mesh":
        tex = _deform_textures(dmeta.get("finish", {}), out)
        if tex:
            manifest["deform_tex"] = tex
    if "n_bonds" in static:
        manifest["peel"] = {"n_bonds": int(static["n_bonds"]), "columns": len(db.get("rows_shape", [])) or None}
    if (D / "attract" / "attract.mp4").exists():
        files["video"] = _put(D / "attract" / "attract.mp4", out, "attract", ".mp4")
        manifest["attract"] = {"frames": db["attract_frames"]}
    if (D / "tiles" / "tile.mp4").exists():
        files["tile"] = _put(D / "tiles" / "tile.mp4", out, "tile", ".mp4")
        files["poster"] = _put(D / "tiles" / "poster.webp", out, "poster", ".webp")

    _attach_env(manifest["look"], D, out, tmp)
    hands = [rssb.read(D / "db" / c["file"])[0]["hand"][::15] for c in db["chunks"]]
    manifest["roi"] = _roi(cam, np.concatenate(hands))
    (out / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    return {"demo": demo, "files": len(list(out.rglob("*"))), "bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file())}


def assemble_track(demo: str) -> dict:
    D = paths.DATA / demo
    out = OUT / "demos" / demo
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    exp = json.loads((D / "export" / "export.json").read_text())
    tr = json.loads((D / "track" / "track.json").read_text())
    files: dict = {}
    files["glb"] = _put(D / "export" / "scene.glb", out, "scene", ".glb")
    files["state"] = _put(D / "track" / "state.rssb.gz", out, "state", ".rssb.gz")
    files["paths"] = _put(D / "track" / "paths.rssb.gz", out, "paths", ".rssb.gz")
    files["video"] = _put(D / "track" / "rollout.mp4", out, "rollout", ".mp4")
    tmp = D / "webp"
    tmp.mkdir(exist_ok=True)
    to_webp([(D / "plate" / "plate.png", tmp / "plate.webp")], 92)
    files["plate"] = _put(tmp / "plate.webp", out, "plate", ".webp")
    if (D / "tiles" / "tile.mp4").exists():
        files["tile"] = _put(D / "tiles" / "tile.mp4", out, "tile", ".mp4")
        files["poster"] = _put(D / "tiles" / "poster.webp", out, "poster", ".webp")
    from . import rssb

    pa = rssb.read(D / "track" / "paths.rssb.gz")[0]
    ee_px = pa["ee_px"].astype(np.float64)
    # as the page does: a hand beside or behind the camera projects to nonsense, and counts for
    # nothing (the conduit's right arm rises past the close camera between strokes)
    w2c = np.linalg.inv(np.asarray(tr["camera"]["cam_to_world_cv"], dtype=np.float64).reshape(4, 4, order="F"))
    ew = pa["ee_world"].astype(np.float64)
    depth = np.einsum("ij,nhj->nhi", w2c[:3, :3], ew)[..., 2] + w2c[2, 3]
    ok = (depth > 0.08) & (ee_px[..., 0] > -960) & (ee_px[..., 0] < 2880) & (ee_px[..., 1] > -540) & (ee_px[..., 1] < 1620)
    ee_px[~ok] = np.nan
    # an arm whose hand hardly travels on screen in the take gets no handle (the weave's left
    # arm holds still while the right one routes the cable)
    travel = np.nansum(np.linalg.norm(np.diff(ee_px, axis=0), axis=2), axis=0)
    show = [int(h) for h in np.flatnonzero(travel > 60.0)] or list(range(ee_px.shape[1]))
    px = ee_px[:, show].reshape(-1, 2)
    px = px[np.isfinite(px).all(axis=1)]
    x0, x1 = np.percentile(px[:, 0], [1, 99])
    y0, y1 = np.percentile(px[:, 1], [1, 99])
    mx, my = (x1 - x0) * 0.25 + 120, (y1 - y0) * 0.25 + 140
    cam = tr["camera"]
    roi = [int(max(0, x0 - mx)), int(max(0, y0 - my)), 0, 0]
    roi[2] = int(min(cam["width"], x1 + mx) - roi[0])
    roi[3] = int(min(cam["height"], y1 + my) - roi[1])
    manifest = {
        "id": demo, "title": TITLES.get(demo, demo), "kind": "track", "fps": 30, "frames": tr["frames"],
        "camera": cam, "nodes": exp["nodes"], "moving": exp["moving"],
        "look": look_from_rtx(json.loads((D / "export" / "look_rtx.json").read_text()), D),
        "track": {"handles": tr["handles"], "show": show, "travel_px": [round(float(t)) for t in travel]}, "roi": roi, "files": files,
    }
    _attach_env(manifest["look"], D, out, tmp)
    (out / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    return {"demo": demo, "files": len(list(out.rglob("*"))), "bytes": sum(q.stat().st_size for q in out.rglob("*") if q.is_file())}


def look_from_rtx(rig: dict, D: Path) -> dict:
    """Initial three.js lights mirroring the RTX lab rig: the overhead panel as the
    shadow-casting key, the untextured dome as a hemisphere. Intensities are
    replaced by the calibration fit (``calib/look.json``) when it exists."""
    fitted = D / "calib" / "look.json"
    if fitted.exists():
        return json.loads(fitted.read_text())
    pc = rig["panel_center"]
    amb = rig["look"].get("ambient") or [1.0, 1.0, 1.0]
    amb = float(np.mean(amb)) if isinstance(amb, list) else float(amb)
    key = float(rig["look"].get("key_intensity") or 1.0)
    return {
        "exposure": 1.0,
        "key": {"pos": pc, "target": [pc[0], pc[1], rig["surface_z"] + 0.15], "intensity": 2.4 * key, "color": [1.0, 0.97, 0.92]},
        "hemi": {"sky": [0.94, 0.95, 0.97], "ground": [0.42, 0.41, 0.40], "intensity": 0.55 * amb},
        "env_intensity": 0.35,
        # soft, as the wipe's fit found (PCF 12.45 -> VSM 11.98): RTX's key is a large panel
        "shadow_map": "vsm",
        "shadow_radius": 28,
        "shadow_opacity": 0.21,
    }


def write_index(demos: list[str]) -> None:
    items = []
    for d in demos:
        m = OUT / "demos" / d / "manifest.json"
        if not m.exists():
            continue
        man = json.loads(m.read_text())
        items.append({"id": d, "title": man.get("title", d), "kind": man["kind"],
                      "tile": man["files"].get("tile"), "poster": man["files"].get("poster")})
    (OUT / "demos.json").write_text(json.dumps({"demos": items}, indent=1))
