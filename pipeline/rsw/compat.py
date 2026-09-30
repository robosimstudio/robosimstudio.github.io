"""What the pipeline needs from RoboSimStudio that an older tree does not have.

Two demos (``conduit``, ``weave``) replay recordings that only rebuild in the 09-05 gallery
worktree (``~/rss_ws/wt/gallery``, f492a8d2): today's scenes have moved on. That tree predates
``core/render/uv.py`` (09-10: the path tracer's private UV helpers made public) and the studio's
``robosimstudio_studio/finish_gltf.py``, and its path tracer did not decode colours: before
c4485d60 (09-10) ``model.shape_color`` went into ``displayColor`` as it is, so it *was* the
linear albedo. Each helper here is today's code where today's tree is loaded, and a verbatim copy
(or, for the colour, the old transfer) where it is not.
"""

from __future__ import annotations

import numpy as np

try:
    from robosimstudio.core.render.uv import box_projected_uvs, planar_uvs  # noqa: F401
except ImportError:  # the 09-05 tree: the path tracer's helpers, copied from today's uv.py

    def box_projected_uvs(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
        n = np.abs(np.cross(b - a, c - a))
        axis = np.argmax(n, axis=1)
        planes = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
        uv = np.zeros((len(tris) * 3, 2), dtype=np.float32)
        for k in range(3):
            corner = p[tris[:, k]]
            for ax, (u, v) in planes.items():
                mask = axis == ax
                uv[k::3][mask] = corner[mask][:, [u, v]]
        return uv

    def planar_uvs(points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        centred = p - p.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        uv = centred @ vt[:2].T
        uv -= uv.min(axis=0)
        return uv.astype(np.float32)


try:
    from robosimstudio.core.render.palette import srgb_to_linear

    LEGACY_COLOR = False
except ImportError:  # the 09-05 tracer used shape_color as linear albedo

    def srgb_to_linear(rgb) -> tuple[float, float, float]:
        return tuple(float(v) for v in tuple(rgb)[:3])

    LEGACY_COLOR = True


try:
    from robosimstudio_studio.finish_gltf import _roughness_image, _texture_image, material_key  # noqa: F401
except ImportError:  # copies of today's finish_gltf helpers

    def material_key(color, finish) -> tuple:
        rgb = tuple(round(float(c), 6) for c in tuple(color)[:3])
        return (rgb, round(float(finish.roughness), 6), round(float(finish.metallic), 6), round(float(finish.opacity), 6),
                finish.texture, finish.roughness_texture, tuple(float(v) for v in finish.uv_scale),
                tuple(float(v) for v in finish.uv_offset))

    def _texture_image(name: str):
        if not name:
            return None
        try:
            from PIL import Image
            from robosimstudio.core.render.finish import texture_path

            path = texture_path(name)
            return Image.open(path).convert("RGB") if path else None
        except Exception:  # noqa: BLE001
            return None

    def _roughness_image(name: str):
        """The tracer reads roughness from the map's red channel, glTF from green."""
        if not name:
            return None
        try:
            from PIL import Image
            from robosimstudio.core.render.finish import texture_path

            path = texture_path(name)
            if not path:
                return None
            red = Image.open(path).convert("RGB").split()[0]
            return Image.merge("RGB", (Image.new("L", red.size, 0), red, Image.new("L", red.size, 255)))
        except Exception:  # noqa: BLE001
            return None
