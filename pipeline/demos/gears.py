"""Gear train: the Panda seats the gears so they mesh, presses the button, the train turns.

A track demo: a recorded, successful rollout, replayed and never re-simulated.
The 09-14 film takes (``slides/out/runs/gear_train_watch_*``) no longer rebuild:
commit 3eb38c6e replaced ``gear_mesh/train`` with the paper's seeded instances
(``gearmesh-medium-train-001..100``), so the take is re-recorded on today's code
with the suite's own routine (``rsw_data/gears/runs``).
"""

from __future__ import annotations

import numpy as np

from rsw import paths

from ._bench import BenchTrack

ID = "gears"


class Spec(BenchTrack):
    id = ID
    RUN = paths.DATA / "gears" / "runs" / "t001"

    def static(self) -> dict:
        s = np.load(self.RUN / self.EPISODE / "steps.npz")
        gears = np.stack([s[k][-1, :3] for k in ("obj_gear_a", "obj_gear_b", "obj_gear_c")])
        return {"gears_final": gears.tolist(), "press": s["right_ee_pose"][-1, :3].tolist()}

    def hero(self, static: dict) -> dict:
        """The board's middle (the seated gears and the button the episode ends pressing),
        from the robot's right front, close: chosen on ``rsw camera`` sheets (2026-09-30);
        the first set framed the Panda and left the 30 cm board a sliver."""
        g = np.asarray(static["gears_final"])
        press = np.asarray(static["press"])
        look = 0.5 * (g.mean(axis=0) + np.array([press[0], press[1], g[:, 2].mean()])) + np.array([0.0, 0.0, 0.02])
        return {"look_at": look.tolist(), "eye_offset": [0.25, -0.30, 0.25], "fov_y": 38.0}
