"""Orbit the view with the mouse, drag the robot from there, and watch it glide home.

  python site/tests/e2e/orbit.py --demo wipe --out /tmp/rsw_orbit

Shots: home, orbited, orbited while dragging (free demos), zoomed, and back home;
report.json holds the view's state at each and any console errors."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

GPU_ARGS = ["--use-angle=vulkan", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan", "--ignore-gpu-blocklist"]
VIEW_JS = "() => { const v = window.__rswView; return JSON.stringify({home: v.home, look: v.look, yaw: v.yaw, pitch: v.pitch, zoom: v.zoom}); }"


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:5173/robosimstudio-website/")
    p.add_argument("--demo", default="wipe")
    p.add_argument("--out", default="/tmp/rsw_orbit")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep: dict = {"demo": a.demo, "states": {}, "console": []}
    with sync_playwright() as pw:
        b = pw.chromium.launch(args=GPU_ARGS)
        page = b.new_page(viewport={"width": 1600, "height": 900})
        page.on("console", lambda m: rep["console"].append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: rep["console"].append(f"pageerror: {e}"))
        page.goto(f"{a.url}#{a.demo}")
        free = True
        try:
            page.wait_for_function("() => { const b = document.querySelector('.ball'); return b && b.style.display !== 'none' && b.style.width; }", timeout=30000)
        except Exception:
            free = False
            page.wait_for_selector(".handle", state="attached", timeout=60000)
        page.wait_for_timeout(2500)

        def state(name: str) -> None:
            rep["states"][name] = json.loads(page.evaluate(VIEW_JS))
            page.screenshot(path=str(out / f"{len(rep['states'])}_{name}.png"))

        state("home")
        cx, cy = 800, 450
        page.mouse.move(cx, cy)
        page.mouse.down(button="right")
        for i in range(1, 31):
            page.mouse.move(cx + 12 * i, cy - 3 * i)
            page.wait_for_timeout(16)
        page.mouse.up(button="right")
        page.wait_for_timeout(400)
        state("orbited")
        if free:
            bb = page.locator(".ball").bounding_box()
            if bb:
                bx, by = bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2
                page.mouse.move(bx, by)
                page.mouse.down()
                for i in range(1, 25):
                    page.mouse.move(bx + 5 * i, by + 2 * i)
                    page.wait_for_timeout(20)
                page.wait_for_timeout(300)
                state("orbited_drag")
                page.mouse.up()
        page.mouse.move(cx, cy)
        for _ in range(6):
            page.mouse.wheel(0, -120)
            page.wait_for_timeout(30)
        page.wait_for_timeout(300)
        state("zoomed")
        t0 = time.time()
        page.wait_for_function("() => window.__rswView.home && window.__rswView.look === 0", timeout=20000)
        rep["home_after_s"] = round(time.time() - t0, 2)
        page.wait_for_timeout(1500)
        state("back_home")
        b.close()
    rep["errors"] = [c for c in rep["console"] if "error" in c.lower()]
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("states", "home_after_s", "errors")}, indent=1))
    return 1 if rep["errors"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
