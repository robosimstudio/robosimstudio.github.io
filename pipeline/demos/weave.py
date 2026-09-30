"""Cable weave: the bimanual YAM weaves a cable through five pegs (``cable_route/weave5``).

A track demo: a recorded, successful rollout, replayed and never re-simulated. The take is the
gallery's (``~/rss_ws/gallery/runs/cable_route__weave5__GT__a1``: the suite's ground-truth graph,
reward 1.0, 3352 steps, recorded densely from the step observer in the 09-05 tree), so its stages
run in that tree (``pipeline/legacy.sh``). Today's suite has no weave5 -- the paper's seeded
instances replaced it -- and its nearest variant, ``port_medium`` (four pegs into a port), seated
three of four and missed the port with the shipped routine (09-30 ``t001``, reward 0.5).
"""

from __future__ import annotations

import numpy as np

from rsw import paths

from ._bench import BenchTrack

ID = "weave"
N_ARM_BODIES = 20
"""The two YAMs' bodies come first; the cable's capsules follow."""


class Spec(BenchTrack):
    id = ID
    RUN = paths.GALLERY / "runs" / "cable_route__weave5__GT__a1"
    LEGACY = True

    def static(self) -> dict:
        p = np.load(self.RUN / self.EPISODE / "privileged.npz")["body_q"]
        cable_end = p[-1, N_ARM_BODIES:, :3]
        lo, hi = cable_end.min(axis=0), cable_end.max(axis=0)
        return {"board_center": (0.5 * (lo + hi)).tolist(), "board_extent": (hi - lo).tolist(),
                "bench_z": float(np.median(p[0, N_ARM_BODIES:, 2]))}

    def hero(self, static: dict) -> dict:
        """The routed cable's middle, from across the bench facing the two arms, high enough to
        read the weave around the five pegs and wide enough (fov 45) for the whole 80 cm board.
        Chosen on ``rsw camera`` sheets (2026-09-30); from behind the arms their links cover it."""
        c = np.asarray(static["board_center"], dtype=np.float64)
        return {"look_at": [float(c[0]), float(c[1]), static["bench_z"] + 0.02], "eye_offset": [0.25, 0.0, 0.55], "fov_y": 45.0}
