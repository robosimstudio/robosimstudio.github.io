"""Conduit: the bimanual YAM threads a USB-C cable through a lidded conduit (``rgbs/thread``).

A track demo: a recorded, successful rollout, replayed and never re-simulated.

The 09-01 success (``RoboSimStudio/outputs/rgbs_final/thread``, reward 0.98) does not replay:
the scene has since gained a ``box`` body and a 38-capsule cable (59 bodies against its 54),
though its ``model_key`` still matches. And today's shipped routine fails (09-30 ``t001``: the
pinch never latched, reward 0), which RoboSimStudio ``a9c6e5a6`` fixes on a side branch; the take
is recorded with that fixed routine (``rsw/tunnel_fix.py``), nothing in the checkout edited.

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
    RUN = paths.DATA / "conduit" / "runs" / "t002"
    CUT = ("_roof",)

    def static(self) -> dict:
        s = np.load(self.RUN / self.EPISODE / "steps.npz")
        p = np.load(self.RUN / self.EPISODE / "privileged.npz")
        z = p["body_q"][0, :, 2]
        on_bench = float(np.median(z[z < z.min() + 0.01]))  # the cable's capsules lie on the bench at t=0
        return {"cable_start": s["obj_cable"][0, :3].tolist(), "cable_end": s["obj_cable"][-1, :3].tolist(), "bench_z": on_bench}

    def hero(self, static: dict) -> dict:
        """The conduit's middle (``tunnel/layout.py`` TUNNEL_X 0.55, TUNNEL_Y -0.080), from
        across the bench, facing the two arms, looking down into the roofless bore."""
        return {"look_at": [0.55, -0.08, static["bench_z"] + 0.05], "eye_offset": [0.55, 0.0, 0.45], "fov_y": 40.0}
