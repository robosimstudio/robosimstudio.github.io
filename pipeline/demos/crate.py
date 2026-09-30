"""Crate flip: two YAMs tip a crate over, the real2sim twin of ``real2sim/crate_reference``.

A track demo from ``~/rss_ws/real2sim/crate_run/verified`` (the validated take: 15.1 s, 454 frames
at 30 Hz; ``real2sim/README.md`` has the physics and the validation). The scene is the live one
the take was made in, ``examples/live/scene.py``, run as ``real2sim/render_crate.py`` runs it; the
take records body poses only, so the hands are the arms' tool frames composed from them.
"""

from __future__ import annotations

import numpy as np

from rsw import paths

ID = "crate"
TAKE = paths.WS / "real2sim" / "crate_run" / "verified"
SCENE = paths.RSS / "examples" / "live" / "scene.py"


class Spec:
    id = ID
    kind = "track"
    handles = (("left", None), ("right", None))

    def build(self, device=None):
        import runpy

        paths.use_rss(SCENE.parent)
        ns = runpy.run_path(str(SCENE))  # builds its own sim on the CPU; the scene is what we need
        sim = ns["scene"].build(control_mode="joint_position", use_cuda_graph=False, device=device or "cuda:0")
        n_rec = np.load(TAKE / "trajectory.npz")["body_q"].shape[1]
        if sim.model.body_count != n_rec:
            raise SystemExit(f"[crate] examples/live/scene.py builds {sim.model.body_count} bodies, the take has {n_rec}: "
                             "the live scene has changed since the take was recorded")
        self.sim = sim
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
        d = np.load(TAKE / "trajectory.npz")
        body_q = d["body_q"]
        ee = self._tcp(body_q.astype(np.float64)) if hasattr(self, "sim") else np.zeros((len(body_q), 2, 7))
        return {"body_q": body_q, "joint_q": d["joint_q"], "ee": ee, "hz": 30.0, "_take": str(TAKE)}

    def static(self) -> dict:
        return {}

    def hero(self, static: dict) -> dict:
        """The scene's ``pivot`` camera: from the side, onto the edge the crate tips over, a little
        tighter (``rsw camera`` sheets, 2026-09-30: the ``reference`` camera has the right arm over
        the crate, and a front-right one looks through the slatted wall)."""
        return {"look_at": [0.55, 0.0, 0.92], "eye": [0.68, -1.65, 1.18], "fov_y": 40.0}
