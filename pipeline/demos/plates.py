"""Plate rack: the bimanual YAM puts six stoneware plates away, one per shelf slot.

A track demo from ``~/rss_ws/plate_rack/out/v6_final`` (30/30 plates across five
takes, no penetration on the drawn geometry). The take is recorded at 30 Hz
(``record_every`` 2 of 60 Hz); the scene is rebuilt as ``plate_rack/render.py``
does, from ``meta.json``'s layout and seed. Replayed, never re-simulated.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from rsw import paths

ID = "plates"
TAKE = paths.PLATES / "out" / "v6_final"


class Spec:
    id = ID
    kind = "track"
    handles = (("left", None), ("right", None))

    def build(self, device=None):
        paths.use_rss(paths.PLATES)
        import demo as D  # noqa: F401  (plate_rack)
        import scene as S  # noqa: F401  (plate_rack)

        meta = json.loads((TAKE / "meta.json").read_text())
        L = replace(S.LAYOUT, **{k: tuple(v) if isinstance(v, list) else v for k, v in meta["layout"].items()})
        scene, _ = S.build_scene(L, seed=meta.get("seed"))
        sim = scene.build(control_mode=D.MODE, device=device)
        self.sim, self.D = sim, D
        return sim

    def _tcp(self, body_q: np.ndarray) -> np.ndarray:
        """Each arm's tool centre per frame: pose[ee_body] composed with ee_offset."""
        from rsw.export import _quat_matrix

        out = np.zeros((len(body_q), len(self.handles), 7), dtype=np.float64)
        for h, (name, _) in enumerate(self.handles):
            arm = self.sim.handles.arms[name]
            off = np.asarray(arm.ee_offset, dtype=np.float64)
            for i, row in enumerate(body_q[:, int(arm.ee_body)]):
                out[i, h, :3] = row[:3] + _quat_matrix(row[3:7]) @ off
                out[i, h, 3:] = row[3:7]
        return out

    def recording(self) -> dict:
        d = np.load(TAKE / "take.npz")
        body_q = d["body_q"]
        ee = self._tcp(body_q.astype(np.float64)) if hasattr(self, "sim") else np.zeros((len(body_q), 2, 7))
        return {"body_q": body_q, "joint_q": d["joint_q"], "ee": ee, "hz": 30.0, "_take": str(TAKE)}

    def static(self) -> dict:
        return {}

    def hero(self, static: dict) -> dict:
        """The plate rack's own "front" camera (the whole bench from the far front corner),
        pulled in a little for 16:9."""
        eye = np.array([1.40, -0.78, 1.38])
        look = np.array([0.52, 0.0, 0.86])
        eye = look + (eye - look) * 0.92
        return {"look_at": look.tolist(), "eye": eye.tolist(), "fov_y": 38.0}
