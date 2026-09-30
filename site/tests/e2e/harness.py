"""The matching harness: synthetic human drags on the real page, scored from ``debugState()``.

Each trial grabs the ball and runs a seeded gesture string -- a Fitts reach with overshoot and
correction, a scrub back and forth, a hold with hand tremor, a flick and release -- sampling the
demo every frame-ish (~30 Hz). Scores (the plan's targets in brackets):

  follow error  |hand - target| while held, all and slow (target < 0.1 m/s)  [median 8, p90 25 mm]
  transitions   per second held                                                [1-4]
  jump          the hand discontinuity each transition's blend hides          [mean 4, p95 12 mm]
  stall         held samples where the hand is stuck (< 0.5 mm in 0.25 s) while
                more than 15 mm from the target                                [<= 3 %]
  settle        release -> the RTX still dissolving in, and success within 2.5 s [median 0.8 s, >= 95 %]

    ~/rss_ws/trailer/.venv-tools/bin/python site/tests/e2e/harness.py --demo wipe --trials 12
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

GPU_ARGS = ["--use-angle=vulkan", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan", "--ignore-gpu-blocklist"]
STATE = "() => JSON.stringify(window.__rsw.debugState())"


class Trial:
    def __init__(self, page, rng: np.random.Generator):
        self.page, self.rng = page, rng
        self.samples: list[dict] = []
        self.x = self.y = 0.0
        self.held = False
        self.t0 = time.time()

    def sample(self) -> None:
        s = json.loads(self.page.evaluate(STATE))
        s["held"] = self.held
        s["wall"] = time.time() - self.t0
        self.samples.append(s)

    def move(self, x: float, y: float, seconds: float, tremor: float = 0.0) -> None:
        x0, y0 = self.x, self.y
        n = max(1, int(seconds * 30))
        for i in range(1, n + 1):
            u = i / n
            k = u * u * u * (10 - 15 * u + 6 * u * u)
            jx, jy = (self.rng.normal(0, tremor, 2) if tremor else (0.0, 0.0))
            self.page.mouse.move(x0 + (x - x0) * k + jx, y0 + (y - y0) * k + jy)
            self.page.wait_for_timeout(28)
            self.sample()
        self.x, self.y = x, y

    def wait(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            self.page.wait_for_timeout(30)
            self.sample()


def ball_center(page) -> tuple[float, float]:
    b = page.locator(".ball").bounding_box()
    return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2


def run_trial(page, rng, span: float) -> tuple[list[dict], float]:
    """One visitor, in the demo's own drag coordinates (a along its main axis, b across):
    grab, reach a random point of the box (10 % overshoot, then the correction), scrub along a,
    hold with tremor, flick along a and release, wait. ``span`` is unused, kept for the CLI."""
    del span
    st = json.loads(page.evaluate(STATE))
    ax = st["axesPx"]
    sc = float(st["scale"])
    (alo, ahi), (blo, bhi) = st["box"]["a"], st["box"]["b"]
    a0, b0 = st["target"]
    tr = Trial(page, rng)
    bx, by = ball_center(page)

    def px(a: float, b: float) -> tuple[float, float]:
        return (bx + ((a - a0) * ax["A"][0] + (b - b0) * ax["B"][0]) * sc,
                by + ((a - a0) * ax["A"][1] + (b - b0) * ax["B"][1]) * sc)

    tr.x, tr.y = bx + 60, by + 60
    page.mouse.move(tr.x, tr.y)
    tr.move(bx, by, 0.4)
    page.mouse.down()
    tr.held = True
    ga, gb = rng.uniform(alo, ahi), rng.uniform(blo, bhi)
    oa, ob = a0 + 1.1 * (ga - a0), b0 + 1.1 * (gb - b0)
    tr.move(*px(oa, ob), rng.uniform(0.35, 0.9))
    tr.move(*px(ga, gb), 0.25)
    tr.wait(0.3)
    amp, hz = rng.uniform(0.15, 0.4) * (ahi - alo), rng.uniform(0.5, 1.5)
    ca = float(np.clip(ga, alo + amp, ahi - amp)) if ahi - alo > 2 * amp else 0.5 * (alo + ahi)
    for _ in range(3):
        tr.move(*px(ca + amp, gb), 0.5 / hz)
        tr.move(*px(ca - amp, gb), 0.5 / hz)
    tr.move(*px(ca, gb), 0.25 / hz)
    tr.move(tr.x, tr.y, rng.uniform(1.0, 2.0), tremor=1.5)
    fa = ca + rng.choice([-1.0, 1.0]) * rng.uniform(0.2, 0.4) * (ahi - alo)
    tr.move(*px(fa, gb), rng.uniform(0.12, 0.22))
    page.mouse.up()
    tr.held = False
    released = tr.samples[-1]["wall"]
    tr.wait(3.0)
    return tr.samples, released


def score(trials: list[tuple[list[dict], float]]) -> dict:
    err_all, err_slow, jumps, trans_rate, stall_n, held_n, settle_t, settle_ok = [], [], [], [], 0, 0, [], 0
    for samples, released in trials:
        held = [s for s in samples if s["held"] and s.get("hand")]
        for i, s in enumerate(held):
            e = float(np.linalg.norm(np.asarray(s["hand"]) - np.asarray(s["targetWorld"])))
            err_all.append(e)
            if i > 0:
                dt = max(1e-3, s["wall"] - held[i - 1]["wall"])
                v = float(np.linalg.norm(np.asarray(s["targetWorld"]) - np.asarray(held[i - 1]["targetWorld"]))) / dt
                if v < 0.1:
                    err_slow.append(e)
            if s["transitions"] > (held[i - 1]["transitions"] if i else s["transitions"]):
                jumps.append(s["lastJump"])
            j = next((k for k in range(i, -1, -1) if s["wall"] - held[k]["wall"] >= 0.25), None)
            if j is not None:
                moved = float(np.linalg.norm(np.asarray(s["hand"]) - np.asarray(held[j]["hand"])))
                held_n += 1
                if moved < 5e-4 and e > 0.015:
                    stall_n += 1
        if len(held) > 1:
            trans_rate.append((held[-1]["transitions"] - held[0]["transitions"]) / max(1e-3, held[-1]["wall"] - held[0]["wall"]))
        after = [s for s in samples if s["wall"] >= released]
        hit = next((s for s in after if s["mode"] == "still"), None)
        if hit is not None and hit["wall"] - released <= 2.5:
            settle_ok += 1
            settle_t.append(hit["wall"] - released)
    mm = lambda xs, q: round(float(np.percentile(np.asarray(xs) * 1e3, q)), 1) if xs else None  # noqa: E731
    return {
        "follow_mm": {"median": mm(err_all, 50), "p90": mm(err_all, 90), "slow_median": mm(err_slow, 50), "slow_p90": mm(err_slow, 90)},
        "transitions_per_s": round(float(np.median(trans_rate)), 2) if trans_rate else None,
        "jump_mm": {"mean": round(float(np.mean(jumps)) * 1e3, 1) if jumps else None, "p95": mm(jumps, 95), "n": len(jumps)},
        "stall_fraction": round(stall_n / max(1, held_n), 4),
        "settle": {"success": f"{settle_ok}/{len(trials)}", "median_s": round(float(np.median(settle_t)), 2) if settle_t else None},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:5173/robosimstudio-website/")
    p.add_argument("--demo", default="wipe")
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    p.add_argument("--weights", default="", help='json {group: weight} applied before the trials')
    a = p.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    trials = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(args=GPU_ARGS)
        page = b.new_page(viewport={"width": 1600, "height": 900})
        page.goto(f"{a.url}#{a.demo}")
        page.wait_for_function("() => { const b = document.querySelector('.ball'); return b && b.style.display !== 'none' && b.style.width; }", timeout=90000)
        page.wait_for_timeout(4000)  # every chunk in
        if a.weights:
            page.evaluate(f"() => window.__rsw.setWeights({a.weights})")
        span = 0.12 * 1600
        for k in range(a.trials):
            trials.append(run_trial(page, rng, span))
            print(f"[harness] {a.demo} trial {k + 1}/{a.trials}: {len(trials[-1][0])} samples", flush=True)
        b.close()
    report = {"demo": a.demo, "trials": a.trials, "seed": a.seed, "weights": a.weights or "manifest", **score(trials)}
    print(json.dumps(report, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps({"report": report, "trials": [[s, r] for s, r in trials]}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
