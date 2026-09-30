"""Fit the live layer's lights to RTX: the same state, the same camera, pixel for pixel.

For a handful of still-set frames, the page pins the live layer to that exact database
frame (``showFrame``) over the RTX plate; the screenshot is compared with the RTX still of
the same frame. The loss is the mean absolute error over the pixels where either image
differs from the plate (the moving parts and their shadows). Coordinate descent over
exposure, key, hemisphere, environment and shadow strength; the result is written to
``rsw_data/<demo>/calib/look.json``, which ``rsw site`` puts in the manifest.

    ~/rss_ws/trailer/.venv-tools/bin/python site/tests/e2e/calib.py --demo wipe
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

GPU_ARGS = ["--use-angle=vulkan", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan", "--ignore-gpu-blocklist"]
DATA = Path.home() / "rss_ws" / "rsw_data"
SITE = Path.home() / "rss_ws" / "robosimstudio-website" / "build" / "site-data" / "demos"

FACTORS = [0.5, 0.7, 0.85, 1.18, 1.4, 2.0]
"""Each round tries the current value times these, so a parameter can walk past any fixed grid
(the first wipe run pinned exposure, key and hemi at their grid minimum)."""


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:5173/robosimstudio-website/")
    p.add_argument("--demo", default="wipe")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--rounds", type=int, default=2)
    a = p.parse_args(argv)
    man = json.loads((SITE / a.demo / "manifest.json").read_text())
    db = json.loads((DATA / a.demo / "db" / "db.json").read_text())
    stills = db["stills"]
    pick = np.linspace(0, len(stills) - 1, a.frames).astype(int)
    plate = np.asarray(Image.open(DATA / a.demo / "plate" / "plate.png").convert("RGB"), dtype=np.float32)
    refs = [(int(stills[i]), np.asarray(Image.open(DATA / a.demo / "stills" / f"{i:04d}.png").convert("RGB"), dtype=np.float32)) for i in pick]
    look = dict(man["look"])
    prior = DATA / a.demo / "calib" / "look.json"
    if prior.exists():  # continue from the last fit
        look = json.loads(prior.read_text())
    params = {
        "exposure": look.get("exposure", 1.0),
        "key": look["key"]["intensity"],
        "hemi": look["hemi"]["intensity"],
        "env": look.get("env_intensity", 0.35),
        "shadow": look.get("shadow_opacity", 0.28),
    }

    def as_look(pr: dict) -> dict:
        return {"exposure": pr["exposure"], "key": {"intensity": pr["key"]}, "hemi": {"intensity": pr["hemi"]},
                "env_intensity": pr["env"], "shadow_opacity": pr["shadow"]}

    with sync_playwright() as pw:
        b = pw.chromium.launch(args=GPU_ARGS)
        page = b.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        page.goto(f"{a.url}#{a.demo}")
        page.add_style_tag(content="#picker,#wordmark,#overlay,#loading{display:none!important}")
        page.wait_for_function("() => { const b = document.querySelector('.ball'); return b && b.style.width; }", timeout=120000)
        page.wait_for_timeout(2500)  # remaining chunks

        def shot(f: int) -> np.ndarray:
            page.evaluate(f"() => window.__rsw.showFrame({f})")
            page.wait_for_timeout(60)
            png = page.screenshot(clip={"x": 0, "y": 0, "width": 1920, "height": 1080})
            return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float32)

        def loss(pr: dict) -> float:
            page.evaluate(f"() => window.__rsw.setLook({json.dumps(as_look(pr))})")
            errs = []
            for f, ref in refs:
                live = shot(f)
                mask = (np.abs(ref - plate).max(axis=2) > 10) | (np.abs(live - plate).max(axis=2) > 10)
                if mask.sum() < 100:
                    continue
                errs.append(float(np.abs(live - ref)[mask].mean()))
            return float(np.mean(errs)) if errs else 1e9

        best = loss(params)
        print(f"[calib] start {params} loss {best:.2f}", flush=True)
        for r in range(a.rounds):
            for k in list(params):
                for v in [params[k] * f for f in FACTORS] + ([0.0] if k == "env" else []):
                    trial = dict(params, **{k: v})
                    lv = loss(trial)
                    if lv < best - 1e-3:
                        best, params = lv, trial
                print(f"[calib] round {r} {k}: {params[k]} loss {best:.2f}", flush=True)
        # one side-by-side for the record
        f, ref = refs[len(refs) // 2]
        page.evaluate(f"() => window.__rsw.setLook({json.dumps(as_look(params))})")
        live = shot(f)
        b.close()
    out = DATA / a.demo / "calib"
    out.mkdir(parents=True, exist_ok=True)
    fitted = dict(man["look"])
    fitted["exposure"] = params["exposure"]
    fitted["key"] = dict(man["look"]["key"], intensity=params["key"])
    fitted["hemi"] = dict(man["look"]["hemi"], intensity=params["hemi"])
    fitted["env_intensity"] = params["env"]
    fitted["shadow_opacity"] = params["shadow"]
    fitted["calib_loss"] = best
    (out / "look.json").write_text(json.dumps(fitted, indent=1))
    Image.fromarray(np.concatenate([live, ref], axis=1).clip(0, 255).astype(np.uint8)).resize((1920, 540)).save(out / "compare.png")
    print(f"[calib] best {params} loss {best:.2f} -> {out / 'look.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
