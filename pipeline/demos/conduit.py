"""Conduit: the bimanual YAM threads a USB-C cable through a lidded conduit (``rgbs/thread``).

A track demo: a recorded, successful rollout, replayed and never re-simulated. The take is the
gallery's (``~/rss_ws/gallery/runs/rgbs__thread__R__a2``: the shipped ``RoutineTunnel``, reward
1.0, 3593 steps, recorded 09-10 in the 09-05 tree), so its stages run in that tree
(``pipeline/legacy.sh``). Why not today's: 12d4d641 (09-11) rebuilt the tunnel against its clip --
a free ``box`` body and a 38-capsule cable, 59 bodies against the take's 54, though the
``model_key`` still matches -- and the shipped routine has failed since (09-30 ``t001``: never
latched, reward 0; ``t002`` with a9c6e5a6's pinch fix, run by ``rsw/tunnel_fix.py``: stalled
strokes, 21 mm advance).

The conduit's roof is built but not drawn -- the task's own ``tunnel/box.py`` ``CUTAWAY`` idea --
so the cable shows in the bore.
"""

from __future__ import annotations

import numpy as np

from rsw import paths

from ._bench import BenchTrack

ID = "conduit"


class Spec(BenchTrack):
    id = ID
    RUN = paths.GALLERY / "runs" / "rgbs__thread__R__a2"
    LEGACY = True
    CUT = ("_roof",)

    def static(self) -> dict:
        s = np.load(self.RUN / self.EPISODE / "steps.npz")
        p = np.load(self.RUN / self.EPISODE / "privileged.npz")
        z = p["body_q"][0, :, 2]
        on_bench = float(np.median(z[z < z.min() + 0.01]))  # the cable's capsules lie on the bench at t=0
        return {"cable_start": s["obj_cable"][0, :3].tolist(), "cable_end": s["obj_cable"][-1, :3].tolist(), "bench_z": on_bench}

    def hero(self, static: dict) -> dict:
        """Between the fed cable and the conduit -- the 09-05 tree's ``tunnel/layout.py`` puts the
        20 cm conduit at x 0.52, y 0.15 (entry at y 0.05), fed along +y -- from across the bench,
        facing the two arms, looking down into the roofless bore. Chosen on ``rsw camera`` sheets
        (2026-09-30): further out than 0.3 m the camera is behind that tree's room wall (black)."""
        return {"look_at": [0.52, 0.10, static["bench_z"] + 0.02], "eye_offset": [0.26, 0.0, 0.40], "fov_y": 38.0}
