"""A track demo made from a ``scripts/bench_run.py --record`` run: rebuild its scene, replay it.

``RUN`` is the run's folder (``run_meta.json``, ``<EPISODE>/{meta.json, privileged.npz, steps.npz}``).
The scene is rebuilt from the run's own task/scene/embodiment and episode, and checked against
the episode's recorded body count -- the ``model_key`` alone does not pin it: the 09-01 conduit
take shares today's key with 54 bodies against today's 59. ``CUT`` names shape labels (by
suffix) to build but not draw, the way a suite's own ``CUTAWAY`` looks inside a closed part.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rsw import paths


class BenchTrack:
    kind = "track"
    RUN: Path
    EPISODE = "0000"
    """The episode's folder in the run (``--episodes 1:3`` records ``0001`` and ``0002``)."""
    CUT: tuple[str, ...] = ()

    # ------------------------------------------------------------------ the run
    def meta(self) -> dict:
        return json.loads((self.RUN / "run_meta.json").read_text())

    def episode(self) -> dict:
        return json.loads((self.RUN / self.EPISODE / "meta.json").read_text())

    @property
    def handles(self) -> tuple[tuple[str, str], ...]:
        """One per recorded end-effector pose, the right hand first."""
        keys = np.load(self.RUN / self.EPISODE / "steps.npz").files
        arms = sorted((k[: -len("_ee_pose")] for k in keys if k.endswith("_ee_pose")), key=lambda a: (a != "right", a))
        return tuple((a, f"{a}_ee_pose") for a in arms)

    def recording(self) -> dict:
        p = np.load(self.RUN / self.EPISODE / "privileged.npz")
        s = np.load(self.RUN / self.EPISODE / "steps.npz")
        ee = np.stack([s[key] for _, key in self.handles], axis=1)
        return {"body_q": p["body_q"], "ee": ee, "sim_time": s["sim_time"], "_take": str(self.RUN)}

    # ------------------------------------------------------------------ the scene
    def build(self, device=None):
        paths.use_rss()
        from robosimstudio_bench.envs.env import Env

        m, ep = self.meta(), self.episode()
        env = Env(m["task"], scene=m["scene"], embodiment=m["embodiment"], gripper=m.get("gripper") or "",
                  episode=int(ep.get("episode_id", 0)), renderer="", device=device)
        env.obs_render_enabled = False
        key = getattr(env, "model_key", None)
        if key and ep.get("model_key") and key != ep["model_key"]:
            print(f"[{self.id}] warning: model_key {key} != recorded {ep['model_key']}")
        n_rec = np.load(self.RUN / self.EPISODE / "privileged.npz")["body_q"].shape[1]
        if env.model.body_count != n_rec:
            raise SystemExit(f"[{self.id}] the scene has {env.model.body_count} bodies, the recording {n_rec}: "
                             "it was made against another build of this task and cannot be replayed")
        if self.CUT:
            self._cutaway(env.model)
        return env

    def _cutaway(self, model) -> None:
        import newton

        labels = list(getattr(model, "shape_label", None) or [])
        flags = model.shape_flags.numpy()
        cut = [i for i, lab in enumerate(labels) if any(str(lab).endswith(s) for s in self.CUT)]
        if not cut:
            raise SystemExit(f"[{self.id}] no shape label ends with {self.CUT}: the cutaway would hide nothing")
        for i in cut:
            flags[i] = int(flags[i]) & ~int(newton.ShapeFlags.VISIBLE)
        model.shape_flags.assign(flags)
        print(f"[{self.id}] cutaway: {[labels[i] for i in cut]}")
