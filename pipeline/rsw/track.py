"""``rsw track``: a recorded, successful rollout as a track demo.

* an RTX video of every frame at 30 fps (the 60 Hz recording's even steps),
  1080p, keyframe every 8 frames and no B-frames, so a paused seek to an exact
  frame is cheap and exact;
* ``state.rssb.gz``: the moving bodies' poses per video frame (q16d);
* ``paths.rssb.gz``: each handle's tool point per frame in world and RTX pixels,
  an "active" flag (moving faster than 5 mm/s) and cumulative arc length;
* the plate (moving bodies hidden) comes from ``rsw plate`` like a free demo.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np

from . import paths, render, rssb
from .stages import project, session


def build(demo: str, gpu: int, accumulate: int = 24, crf: int = 22) -> dict:
    spec, canon, st, cam, sess = session(demo, gpu, accumulate=accumulate)
    rec = spec.recording()
    out = paths.demo_dir(demo, "track")
    exp = json.loads((paths.DATA / demo / "export" / "export.json").read_text())
    moving = [int(b) for b in exp["moving"]]
    bq = np.asarray(rec["body_q"], dtype=np.float32)
    step = max(1, int(round(float(rec.get("hz", 60.0)) / 30.0)))  # 60 Hz recordings: every other step
    sel = np.arange(0, len(bq), step)
    N = len(sel)

    # warm up on frame 0 (the canonical state)
    sess.render(sess.write(bq[0], None, None), passes=1)
    ff = paths.ffmpeg()
    mp4 = out / "rollout.mp4"
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{cam.width}x{cam.height}",
           "-r", "30", "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-g", "8", "-keyint_min", "8", "-bf", "0",
           "-sc_threshold", "0", "-x264-params", "open-gop=0", "-profile:v", "high", "-colorspace", "bt709",
           "-color_primaries", "bt709", "-color_trc", "iec61966-2-1", "-movflags", "+faststart", str(mp4)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    frames_dir = paths.demo_dir(demo, "track", "frames")
    for k, i in enumerate(sel):
        rgb = sess.render(sess.write(bq[i], None, None), passes=1)[cam.name]["rgb"]
        proc.stdin.write(np.ascontiguousarray(rgb).tobytes())
        if k % 30 == 0:
            render.save_png(frames_dir / f"{k:05d}.png", rgb)
        if k % 300 == 0:
            print(f"[track] {demo}: {k}/{N}", flush=True)
    proc.stdin.close()
    proc.wait()
    sess.close()

    # state: moving bodies at the video's frames
    poses = bq[sel][:, moving, :].reshape(N, -1)
    rssb.write(out / "state.rssb.gz", {"bodies": np.asarray(moving, dtype=np.uint32), "body_q": ("q16d", poses)},
               {"frames": int(N), "fps": 30})

    # paths: each handle's tool point, in world and in RTX pixels
    ee = np.asarray(rec["ee"], dtype=np.float64)[sel]  # [N, H, 7]
    H = ee.shape[1]
    world = ee[:, :, :3]
    px = np.stack([project(cam, world[:, h])[:, :2] for h in range(H)], axis=1)
    speed = np.zeros((N, H))
    speed[1:] = np.linalg.norm(np.diff(world, axis=0), axis=2) * 30.0
    arc = np.concatenate([np.zeros((1, H)), np.cumsum(np.linalg.norm(np.diff(world, axis=0), axis=2), axis=0)])
    rssb.write(out / "paths.rssb.gz", {
        "ee_world": world.astype(np.float32), "ee_px": px.astype(np.float32),
        "active": (speed > 0.005).astype(np.uint8), "arc": arc.astype(np.float32),
    }, {"handles": [h for h, _ in spec.handles]})
    info = {"frames": int(N), "mp4_bytes": mp4.stat().st_size, "camera": render.camera_json(cam),
            "handles": [h for h, _ in spec.handles], "moving": moving,
            "px_range": [float(px[..., 0].min()), float(px[..., 0].max()), float(px[..., 1].min()), float(px[..., 1].max())]}
    (out / "track.json").write_text(json.dumps(info, indent=1))
    return {k: v for k, v in info.items() if k != "camera"}

