"""Cable weave: the bimanual YAM weaves a cable through the pegs and seats its end in the port.

A track demo: a recorded, successful rollout of ``cable_route``'s own ``RoutineWeave``,
replayed and never re-simulated. The gallery's ``cable_route/weave5`` no longer exists; the
suite's port variants are the same act (``port_medium``: four pegs, 720 mm cable, 60 capsules;
``port_easy``: two pegs, 600 mm, 50). ``port_medium`` episode 0 (09-30 ``t001``) seated three of
four pegs and missed the port (reward 0.5), so the take is the first episode that succeeds.
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
    RUN = paths.DATA / "cable" / "runs" / "t002"
    EPISODE = "0001"

    def static(self) -> dict:
        p = np.load(self.RUN / self.EPISODE / "privileged.npz")["body_q"]
        cable_end = p[-1, N_ARM_BODIES:, :3]
        lo, hi = cable_end.min(axis=0), cable_end.max(axis=0)
        return {"board_center": (0.5 * (lo + hi)).tolist(), "board_extent": (hi - lo).tolist(),
                "bench_z": float(np.median(p[0, N_ARM_BODIES:, 2]))}

    def hero(self, static: dict) -> dict:
        """The routed cable's middle, from across the bench facing the two arms, high enough to
        read the weave around the pegs."""
        c = np.asarray(static["board_center"], dtype=np.float64)
        return {"look_at": [float(c[0]), float(c[1]), static["bench_z"] + 0.02], "eye_offset": [0.55, 0.05, 0.55], "fov_y": 42.0}
