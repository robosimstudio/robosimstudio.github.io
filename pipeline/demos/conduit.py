"""Conduit: the bimanual YAM threads a USB-C cable through a lidded conduit (``rgbs/thread``).

A track demo: a recorded, successful rollout, replayed and never re-simulated. The take is the
09-01 success (``RoboSimStudio/outputs/rgbs_final/thread``: reward 0.98, 3588 steps); today's
model has the same ``model_key`` (58cbd4ac40931933), so it replays as is. A re-record on 09-30
(``rsw_data/conduit/runs/t001``) failed its one episode (the tip reached 15 % of the bore), which
the simulator's non-determinism allows; it is kept for the record, not shown.

The conduit's roof is built but not drawn -- the task's own ``tunnel/box.py`` ``CUTAWAY`` idea --
so the cable shows in the bore. ``CUTAWAY`` itself cannot be set from here (the env loads task
modules as its own private copies), so the roof's VISIBLE flag is cleared on the built model,
before any renderer or exporter walks it. Nothing about the physics changes, and nothing is
simulated anyway.
"""

from __future__ import annotations

import json

import numpy as np

from rsw import paths

ID = "conduit"
RUN = paths.RSS / "outputs" / "rgbs_final" / "thread"
CUT = ("_roof",)
"""Suffixes of the shape labels to build but not draw."""


class Spec:
    id = ID
    kind = "track"
    handles = (("right", "right_ee_pose"), ("left", "left_ee_pose"))

    def meta(self) -> dict:
        return json.loads((RUN / "run_meta.json").read_text())

    def build(self, device=None):
        paths.use_rss()
        import newton
        from robosimstudio_bench.envs.env import Env

        m = self.meta()
        env = Env(m["task"], scene=m["scene"], embodiment=m["embodiment"], gripper=m.get("gripper") or "",
                  episode=0, renderer="", device=device)
        env.obs_render_enabled = False
        ep = json.loads((RUN / "0000" / "meta.json").read_text())
        key = getattr(env, "model_key", None)
        if key and ep.get("model_key") and key != ep["model_key"]:
            print(f"[conduit] warning: model_key {key} != recorded {ep['model_key']}")
        model = env.model
        labels = list(getattr(model, "shape_label", None) or [])
        flags = model.shape_flags.numpy()
        cut = [i for i, lab in enumerate(labels) if any(str(lab).endswith(s) for s in CUT)]
        if not cut:
            raise SystemExit(f"[conduit] no shape label ends with {CUT}: the cutaway would hide nothing")
        for i in cut:
            flags[i] = int(flags[i]) & ~int(newton.ShapeFlags.VISIBLE)
        model.shape_flags.assign(flags)
        print(f"[conduit] cutaway: {[labels[i] for i in cut]}")
        return env

    def recording(self) -> dict:
        p = np.load(RUN / "0000" / "privileged.npz")
        s = np.load(RUN / "0000" / "steps.npz")
        ee = np.stack([s[key] for _, key in self.handles], axis=1)
        return {"body_q": p["body_q"], "ee": ee, "sim_time": s["sim_time"], "_take": str(RUN)}

    def static(self) -> dict:
        s = np.load(RUN / "0000" / "steps.npz")
        p = np.load(RUN / "0000" / "privileged.npz")
        z = p["body_q"][0, :, 2]
        on_bench = float(np.median(z[z < z.min() + 0.01]))  # the cable's segments lie on the bench at t=0
        return {
            "cable_start": s["obj_cable"][0, :3].tolist(),
            "cable_end": s["obj_cable"][-1, :3].tolist(),
            "bench_z": on_bench,
        }

    def hero(self, static: dict) -> dict:
        """The conduit's middle (y = TUNNEL_Y -0.080, where the head's 285 mm pull is centred),
        from across the bench, facing the two arms, looking down into the roofless bore."""
        a, b = np.asarray(static["cable_start"]), np.asarray(static["cable_end"])
        look = np.array([0.5 * (a[0] + b[0]) + 0.02, -0.08, static["bench_z"] + 0.03])
        return {"look_at": look.tolist(), "eye_offset": [0.55, 0.0, 0.45], "fov_y": 40.0}
