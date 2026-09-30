"""Run virtual-visitor sessions: build once, pre-roll once, then many takes.

A session is one process on one card. Takes inside it are physically continuous
(the hand carries on from where the last take left it); each starts with a
second at the demo's hover pose and then plays a seeded gesture program.
Nothing is re-simulated later: every frame the site shows is read from here.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from . import gpu, prov
from .gestures import Program
from .record import Recorder

SESSION_FATAL = ("sponge left the hand", "non-finite state", "cable exploded")


def _visible_gpu() -> int:
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not v or "," in v:
        raise SystemExit("run takes with CUDA_VISIBLE_DEVICES=<one allowed card>")
    idx = int(v)
    gpu.check(idx)
    return idx


def run_session(demo: str, out_root: Path, seed: int, n_takes: int, take_seconds: float, hover_s: float = 1.0,
                attract: bool = False, program: str = "gestures") -> Path:
    from demos import load

    card = _visible_gpu()
    spec = load(demo)
    sdir = Path(out_root) / f"s{seed:05d}"
    sdir.mkdir(parents=True, exist_ok=True)
    beat = sdir / "heartbeat"
    t0 = time.time()
    sim = spec.build()
    t_build = time.time() - t0
    beat.write_text(f"{time.time():.0f} built\n")
    t1 = time.time()
    drv = spec.pre_roll(sim)
    t_pre = time.time() - t1
    static = spec.static_meta(sim, drv)
    static.update(seed=seed, gpu=card, t_build=t_build, t_pre_roll=t_pre)
    (sdir / "static.json").write_text(json.dumps(static, indent=1))
    print(f"[rsw] {demo} s{seed}: built {t_build:.0f}s, pre-roll {t_pre:.0f}s on GPU {card}", flush=True)

    cfg = spec.gesture_config(drv)
    hz = cfg.hz
    for k in range(n_takes):
        tdir = sdir / f"take_{k:03d}"
        if (tdir / "meta.json").exists():
            continue
        if k > 0 and getattr(spec, "fresh_episode_per_take", False):
            t_r = time.time()
            drv = spec.pre_roll(sim, reset=True)
            print(f"[rsw] {demo} s{seed} take {k}: fresh episode, pre-roll {time.time() - t_r:.0f}s", flush=True)
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "static.json").write_text(json.dumps(spec.static_meta(sim, drv), indent=1))
        spec.start_take(sim, drv)
        if attract:
            cursor = spec.attract_program(drv, hz)
            prim = np.full(len(cursor), 254, dtype=np.uint8)
        elif program != "gestures":
            # a demo's own visitor pattern (the peel's staircase: pull, hold, pull...)
            cursor = getattr(spec, f"{program}_program")(drv, hz, np.random.default_rng(seed * 1000 + k))
            prim = np.full(len(cursor), 253, dtype=np.uint8)
        else:
            prog = Program(cfg, seed=seed * 1000 + k, start=spec.hover_ab(drv))
            cursor, prim = prog.seconds(take_seconds)
            pre = np.repeat(spec.hover_ab(drv)[None, :], int(hover_s * hz), axis=0)
            cursor = np.concatenate([pre, cursor])
            prim = np.concatenate([np.full(len(pre), 255, dtype=np.uint8), prim])
        if getattr(spec, "plans_takes", False):
            t_p = time.time()
            info = spec.plan(drv, np.array([spec.to_world(drv, ab) for ab in cursor]))
            print(f"[rsw] {demo} s{seed} take {k}: planned {len(cursor)} steps in {time.time() - t_p:.0f}s {info}", flush=True)
        rec = Recorder(tdir)
        fail, parity, tw, last_beat = None, 0.0, time.time(), time.time()
        for i, ab in enumerate(cursor):
            world = spec.to_world(drv, ab)
            spec.step(sim, drv, world)
            row = spec.observe(sim, drv)
            row["cursor"] = np.asarray(ab, dtype=np.float32)
            row["cursor_world"] = np.asarray(world, dtype=np.float32)
            row["prim"] = np.uint8(prim[i])
            fail = spec.gate(sim, drv, row)
            row["gate"] = np.uint8(0 if fail is None else 1)
            rec.step(row)
            if hasattr(spec, "parity") and i % 30 == 29:
                parity = max(parity, spec.parity())
            if time.time() - last_beat > 30:
                last_beat = time.time()
                rate = (i + 1) / (time.time() - tw)
                beat.write_text(f"{last_beat:.0f} take {k} step {i + 1}/{len(cursor)} {rate:.1f} steps/s\n")
                print(f"[rsw] {demo} s{seed} take {k}: {i + 1}/{len(cursor)} steps, {rate:.1f} steps/s", flush=True)
            if fail:
                break
        wall = time.time() - tw
        meta = {
            "demo": demo,
            "seed": seed,
            "take": k,
            "gate": fail,
            "wall_s": wall,
            "steps_per_s": rec.n / max(wall, 1e-9),
            "ink_parity_max": parity,
            "kind": "attract" if attract else "visitor",
            "prov": prov.record() if k == 0 else None,
        }
        rec.close(meta)
        print(f"[rsw] {demo} s{seed} take {k}: {rec.n} steps in {wall:.0f}s ({rec.n / wall:.1f}/s)"
              f"{', gate: ' + fail if fail else ''}{f', ink parity {parity:.2e}' if parity else ''}", flush=True)
        if fail and any(fail.startswith(f) for f in SESSION_FATAL):
            break
    (sdir / "DONE").write_text(f"{time.time():.0f}\n")
    return sdir
