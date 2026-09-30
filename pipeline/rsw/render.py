"""RTX rendering of recorded states: plate, stills, attract and track videos, previews.

One process per card, and for RTX the card is chosen with ``device="cuda:N"``
with CUDA_VISIBLE_DEVICES *unset*: OVRTX enumerates graphics devices on its own
and skips every GPU when that variable is set (rtx.py, ``_init_ovrtx``). After
the first frame the process checks it holds a context on that card only.

Nothing is simulated here. A frame is written into a fresh ``model.state()``
through ``StateReplay`` and rendered. Every session warms up on the demo's
canonical frame first, because the RTX backend fixes deformable UVs from the
first state it sees (``_build_deformable_cache``) and the exported scene's
deformable UVs come from that same frame.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import gpu

HIDE_Z = -1000.0


def hero_camera(spec_hero: dict, width: int = 1920, height: int = 1080, name: str = "hero"):
    """A CameraSpec whose principal point makes RTX's aperture offset exactly zero.

    ``rtx.usd_film_back`` offsets the aperture by ``(W/2 - 0.5 - cx)``, so
    ``cx = W/2 - 0.5`` is the symmetric frustum a three.js PerspectiveCamera
    reproduces pixel for pixel.
    """
    from robosimstudio.core.cameras import CameraSpec

    look = np.asarray(spec_hero["look_at"], dtype=np.float64)
    eye = np.asarray(spec_hero.get("eye") or (look + np.asarray(spec_hero["eye_offset"])), dtype=np.float64)
    f = 0.5 * height / np.tan(np.radians(float(spec_hero["fov_y"])) / 2.0)
    return CameraSpec(
        name=name, width=width, height=height, fx=float(f), fy=float(f),
        cx=width / 2.0 - 0.5, cy=height / 2.0 - 0.5, near=0.02, far=30.0,
        eye=tuple(float(v) for v in eye), look_at=tuple(float(v) for v in look), up=(0.0, 0.0, 1.0),
    )


def camera_json(cam) -> dict:
    wm = np.asarray(cam.world_matrix, dtype=np.float64)
    return {
        "width": int(cam.width), "height": int(cam.height),
        "fx": float(cam.fx), "fy": float(cam.fy), "cx": float(cam.cx), "cy": float(cam.cy),
        "near": float(cam.near), "far": float(cam.far),
        "eye": [float(v) for v in cam.eye], "look_at": [float(v) for v in cam.look_at], "up": [float(v) for v in cam.up],
        "cam_to_world_cv": [float(v) for v in wm.T.reshape(-1)],  # column-major
    }


class RtxSession:
    def __init__(self, demo: str, card: int, cameras, *, accumulate: int = 64, sim=None, spec=None) -> None:
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise SystemExit("RTX jobs pick their card with device=cuda:N; unset CUDA_VISIBLE_DEVICES")
        gpu.check(card)
        from robosimstudio.core.render.offline import StateReplay

        from demos import load

        self.card = card
        self.spec = spec or load(demo)
        self.sim = sim or self.spec.build(device=f"cuda:{card}")
        self.cameras = list(cameras)
        self.renderer = self.sim.make_renderer(
            "rtx", cameras=self.cameras, channels=["rgb", "depth"], accumulate=int(accumulate), quality="final"
        )
        self.replay = StateReplay(self.sim.model, strict=False)
        self.particle_base = self.sim.model.particle_q.numpy().copy() if self.sim.model.particle_count else None
        self.checked = False

    def write(self, body_q, joint_q=None, particles: dict[str, np.ndarray] | None = None, live=None):
        """Load one recorded frame. ``particles`` maps a global index array's key to positions."""
        step: dict[str, np.ndarray] = {"body_q": np.asarray(body_q, dtype=np.float32)}
        if joint_q is not None:
            step["joint_q"] = np.asarray(joint_q, dtype=np.float32)
        if self.particle_base is not None:
            pq = self.particle_base.copy()
            for idx, pos in (particles or {}).values():
                pq[np.asarray(idx, dtype=np.int64)] = pos
            step["particle_q"] = pq
        state = self.replay.write(step)
        if live is not None:
            shapes, colors = live
            sc = self.sim.model.shape_color.numpy()
            sc[np.asarray(shapes, dtype=np.int64), :3] = colors
            self.sim.model.shape_color.assign(sc)
        return state

    def render(self, state, passes: int = 1) -> dict[str, dict[str, np.ndarray]]:
        out = None
        for _ in range(max(1, passes)):
            out = self.renderer.render(state)
        if not self.checked:
            gpu.assert_only_on(self.card)
            self.checked = True
        return {c.name: {"rgb": _u8(out[c.name]["rgb"]), "depth": np.asarray(out[c.name]["depth"])} for c in self.cameras}

    def close(self) -> None:
        self.renderer.close()


def _u8(rgb) -> np.ndarray:
    a = np.asarray(rgb)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return a[..., :3]


def save_png(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(rgb).save(path)


def contact_sheet(images: list[np.ndarray], labels: list[str], cols: int = 3) -> np.ndarray:
    from PIL import Image, ImageDraw

    h, w = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, (im, lab) in enumerate(zip(images, labels)):
        x, y = (i % cols) * w, (i // cols) * h
        sheet.paste(Image.fromarray(im), (x, y))
        draw.rectangle([x, y, x + 8 * len(lab) + 8, y + 18], fill=(0, 0, 0))
        draw.text((x + 4, y + 3), lab, fill=(255, 255, 255))
    return np.asarray(sheet)


def load_static(demo: str) -> dict:
    from .export import _spec, is_track, takes

    if is_track(demo):
        return _spec(demo).static()
    t = takes(demo)[0]
    return json.loads((t.parent / "static.json").read_text())
