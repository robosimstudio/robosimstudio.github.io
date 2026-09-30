"""Gear train: the Panda seats the gears so they mesh, presses the button, the train turns.

A track demo: a recorded, successful rollout, replayed and never re-simulated.
The 09-14 film takes (``slides/out/runs/gear_train_watch_*``) no longer rebuild:
commit 3eb38c6e replaced ``gear_mesh/train`` with the paper's seeded instances
(``gearmesh-medium-train-001..100``), so the take is re-recorded on today's code
with the suite's own routine (``rsw_data/gears/runs``).
"""

from __future__ import annotations

import json

import numpy as np

from rsw import paths

ID = "gears"
RUN = paths.DATA / "gears" / "runs" / "t001"


class Spec:
    id = ID
    kind = "track"
    handles = (("right", "right_ee_pose"),)

    def meta(self) -> dict:
        return json.loads((RUN / "run_meta.json").read_text())

    def build(self, device=None):
        paths.use_rss()
        from robosimstudio_bench.envs.env import Env

        m = self.meta()
        env = Env(m["task"], scene=m["scene"], embodiment=m["embodiment"], gripper=m.get("gripper") or "",
                  episode=0, renderer="", device=device)
        env.obs_render_enabled = False
        ep = json.loads((RUN / "0000" / "meta.json").read_text())
        key = getattr(env, "model_key", None)
        if key and ep.get("model_key") and key != ep["model_key"]:
            print(f"[gears] warning: model_key {key} != recorded {ep['model_key']}")
        return env

    def recording(self) -> dict:
        p = np.load(RUN / "0000" / "privileged.npz")
        s = np.load(RUN / "0000" / "steps.npz")
        ee = np.stack([s[key] for _, key in self.handles], axis=1)
        return {"body_q": p["body_q"], "ee": ee, "sim_time": s["sim_time"], "_take": str(RUN)}

    def static(self) -> dict:
        s = np.load(RUN / "0000" / "steps.npz")
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
