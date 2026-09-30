"""``python -m rsw <stage> ...`` -- run from ``pipeline/`` with RoboSimStudio's venv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import paths


def _takes(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="rsw takes")
    p.add_argument("--demo", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--takes", type=int, default=1)
    p.add_argument("--seconds", type=float, default=240.0)
    p.add_argument("--round", default="r1")
    p.add_argument("--attract", action="store_true", help="the demo's scripted autopilot instead of gestures")
    p.add_argument("--program", default="gestures", help="gestures, or a named program of the spec (<name>_program)")
    a = p.parse_args(argv)
    from .takes import run_session

    out = paths.demo_dir(a.demo, "takes", "attract" if a.attract else a.round)
    run_session(a.demo, out, a.seed, a.takes, a.seconds, attract=a.attract, program=a.program)
    return 0


def _export(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="rsw export")
    p.add_argument("--demo", required=True)
    a = p.parse_args(argv)
    import json

    from demos import load

    from . import export

    out = paths.demo_dir(a.demo, "export")
    spec = load(a.demo)
    sim = spec.build()
    canon = export.canonical(a.demo)
    moving, dev = export.moving_bodies(a.demo)
    # bodies the page draws another way (the whip's capsules: its tube) stay out of the GLB, or
    # they stand there at their canonical pose -- nothing on the page poses a free body
    # (they stay in ``moving``: the plate and the probe hide what moves)
    drawn = set(int(b) for b in (spec.tube_body_indices() if hasattr(spec, "tube_body_indices") else []))
    print(f"[export] canonical frame from {canon['_take']}; moving bodies {moving}; drawn as a tube {sorted(drawn)}")
    info = export.export_scene(a.demo, sim, out, moving, canon, skip_bodies=drawn)
    print("[export]", json.dumps({k: v for k, v in info.items() if k != "deformables"}))
    if export.is_track(a.demo):
        return 0
    frames = export.load(Path(canon["_take"]))
    kin = export.export_kinematics(sim, out, frames)
    print("[export] kinematics", kin)
    return 0


def _camera(argv: list[str]) -> int:
    """Render candidate hero cameras on the canonical and a pressing frame, as a contact sheet."""
    p = argparse.ArgumentParser(prog="rsw camera")
    p.add_argument("--demo", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--candidates", default="", help="json list of {eye_offset, fov_y, look_dz}")
    p.add_argument("--size", default="640x360")
    a = p.parse_args(argv)
    import json

    import numpy as np

    from . import export, render

    w, h = (int(v) for v in a.size.split("x"))
    static = render.load_static(a.demo)
    from demos import load

    spec = load(a.demo)
    base = spec.hero(static)
    cands = json.loads(a.candidates) if a.candidates else [
        {"eye_offset": [-0.55, -0.05, 0.42], "fov_y": 42.0},
        {"eye_offset": [-0.62, -0.02, 0.52], "fov_y": 45.0},
        {"eye_offset": [-0.45, -0.35, 0.40], "fov_y": 42.0},
        {"eye_offset": [-0.45, 0.30, 0.40], "fov_y": 42.0},
        {"eye_offset": [-0.35, -0.05, 0.30], "fov_y": 50.0},
        {"eye_offset": [-0.70, -0.10, 0.60], "fov_y": 36.0},
    ]
    cams = []
    for i, c in enumerate(cands):
        hero = dict(base, **{k: v for k, v in c.items() if k != "look_dz"})
        if "eye_offset" in c:
            hero.pop("eye", None)
        if "look_dz" in c:
            hero["look_at"] = (np.asarray(base["look_at"]) + np.array([0, 0, c["look_dz"]])).tolist()
        cams.append(render.hero_camera(hero, w, h, name=f"c{i}"))
    sess = render.RtxSession(a.demo, a.gpu, cams, accumulate=16, spec=spec)
    if export.is_track(a.demo):
        rec = spec.recording()
        d = {"body_q": rec["body_q"], "joint_q": [None] * len(rec["body_q"]), "deform": None}
        press = int(len(rec["body_q"]) * 0.55)
        idx = None
    else:
        take = export.takes(a.demo)[0]
        d = export.load(take)
        touch = np.flatnonzero(d["touching"]) if "touching" in d else np.arange(len(d["t"]))
        press = int(touch[len(touch) // 2])
        from .stages import _sponge

        idx = _sponge(static)
    images, labels = [], []
    for fi in (0, press):
        parts = {"d": (idx, d["deform"][fi])} if (idx is not None and d["deform"] is not None) else None
        st = sess.write(d["body_q"][fi], d["joint_q"][fi], parts)
        out = sess.render(st, passes=3)
        for i, c in enumerate(cams):
            images.append(out[c.name]["rgb"])
            labels.append(f"c{i} f{fi} {cands[i]}")
    sess.close()
    sheet = render.contact_sheet(images, labels, cols=len(cams))
    dst = paths.demo_dir(a.demo, "camera") / "candidates.png"
    render.save_png(dst, sheet)
    print("[camera]", dst)
    return 0


def _db(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="rsw db")
    p.add_argument("--demo", required=True)
    p.add_argument("--rounds", default="r1")
    p.add_argument("--stills", type=int, default=450)
    a = p.parse_args(argv)
    import json

    from .db import build

    s = build(a.demo, rounds=tuple(a.rounds.split(",")), n_stills=a.stills)
    print("[db]", json.dumps({k: s[k] for k in ("frames", "pca", "plant", "still_cover_mm", "ink")}, default=str))
    return 0


def _render_stage(argv: list[str], which: str) -> int:
    p = argparse.ArgumentParser(prog=f"rsw {which}")
    p.add_argument("--demo", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    a = p.parse_args(argv)
    import json

    from . import stages

    if which == "plate":
        r = stages.plate(a.demo, a.gpu)
        r = {k: v for k, v in r.items() if k not in ("ink", "camera")}
    elif which == "track":
        from . import track

        r = track.build(a.demo, a.gpu)
    elif which == "stills":
        r = stages.stills(a.demo, a.gpu, a.shard, a.nshards)
    elif which == "envprobe":
        r = stages.envprobe(a.demo, a.gpu)
    else:
        r = stages.attract(a.demo, a.gpu)
    print(f"[{which}]", json.dumps(r, default=str))
    return 0


def _taskcams(argv: list[str]) -> int:
    """The task's own camera rig, as built (run with CUDA_VISIBLE_DEVICES=<card>): a track demo's
    hero can be one of them (``BenchTrack.HERO_CAMERA``), as the gallery chose on contact sheets."""
    p = argparse.ArgumentParser(prog="rsw taskcams")
    p.add_argument("--demo", required=True)
    a = p.parse_args(argv)
    import json
    import math

    from demos import load

    sim = load(a.demo).build(device="cuda:0")
    cams = []
    for c in sim.cameras:
        if int(getattr(c, "parent_body", -1) or -1) >= 0 or getattr(c, "parent_frame", None):
            continue  # rides on a body: no fixed pose to film from
        # today's spec states fy; the 09-05 tree's leaves it None and states fov_y
        fy = getattr(c, "fy", None)
        fov = math.degrees(2 * math.atan(0.5 * c.height / float(fy))) if fy else float(c.fov_y)
        cams.append({"name": c.name, "eye": [float(v) for v in c.eye], "look_at": [float(v) for v in c.look_at],
                     "up": [float(v) for v in c.up], "width": int(c.width), "height": int(c.height),
                     "fov_y": fov, "xform": list(c.xform) if getattr(c, "xform", None) is not None else None})
    out = paths.demo_dir(a.demo, "camera") / "task_cameras.json"
    out.write_text(json.dumps(cams, indent=1))
    print("[taskcams]", [c["name"] for c in cams], "->", out)
    return 0


def _probe(argv: list[str]) -> int:
    """Demo-specific feasibility probes (run with CUDA_VISIBLE_DEVICES=<card>)."""
    p = argparse.ArgumentParser(prog="rsw probe")
    p.add_argument("--demo", required=True)
    a = p.parse_args(argv)
    import json

    from demos import load

    spec = load(a.demo)
    sim = spec.build()
    drv = spec.pre_roll(sim)
    out = paths.demo_dir(a.demo, "probe")
    res = spec.probe_workspace(drv)
    (out / "workspace.json").write_text(json.dumps(res))
    import numpy as np

    if "score_mm_plus_halfdeg" not in res:  # a demo-specific report (the towel's go/no-go)
        print("[probe]", json.dumps(res))
        return 0
    g = np.asarray(res["score_mm_plus_halfdeg"])
    print("[probe] rows z, cols x; cell = worst IK mm + 0.5*deg over the leans")
    print("       " + " ".join(f"{x:6.2f}" for x in res["xs"]))
    for z, row in zip(res["zs"], g):
        print(f"z={z:5.3f} " + " ".join(f"{v:6.1f}" for v in row))
    return 0


def _tiles(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="rsw tiles")
    p.add_argument("--demo", required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=5.0)
    a = p.parse_args(argv)
    from . import stages

    print("[tiles]", stages.tiles(a.demo, a.start, a.seconds))
    return 0


def _site(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="rsw site")
    p.add_argument("--demos", required=True, help="comma-separated, picker order")
    a = p.parse_args(argv)
    import json

    from . import site

    ids = a.demos.split(",")
    for d in ids:
        m = json.loads((paths.DATA / d / "db" / "db.json").read_text()) if (paths.DATA / d / "db" / "db.json").exists() else None
        r = site.assemble_free(d) if m is not None else site.assemble_track(d)
        print("[site]", r)
    site.write_index(ids)
    return 0


def _doctor(argv: list[str]) -> int:
    from . import gpu

    print("RSS:", paths.RSS, paths.RSS.exists())
    print("DATA:", paths.DATA)
    print("ffmpeg:", paths.ffmpeg())
    print("playwright python:", paths.PLAYWRIGHT_PY, paths.PLAYWRIGHT_PY.exists())
    for c in gpu.cards():
        tag = "DENY" if c.index in gpu.DENIED else ("ok" if c.index in gpu.ALLOWED else "-")
        print(f"GPU {c.index} {tag:4s} used {c.used_mib:6d} MiB util {c.util:3d}% pids {list(c.pids)}")
    print("free allowed GPUs:", gpu.free())
    return 0


STAGES = {"takes": _takes, "export": _export, "camera": _camera, "db": _db, "probe": _probe, "tiles": _tiles, "site": _site, "doctor": _doctor,
          "plate": lambda a: _render_stage(a, "plate"), "stills": lambda a: _render_stage(a, "stills"),
          "envprobe": lambda a: _render_stage(a, "envprobe"), "taskcams": _taskcams,
          "attract": lambda a: _render_stage(a, "attract"), "track": lambda a: _render_stage(a, "track")}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in STAGES:
        print(f"usage: python -m rsw {{{','.join(STAGES)}}} ...", file=sys.stderr)
        return 2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    return STAGES[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
