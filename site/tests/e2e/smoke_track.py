"""Track-demo smoke test: play, grab the gripper handle, scrub, let go, land on the RTX frame.

    ~/rss_ws/trailer/.venv-tools/bin/python site/tests/e2e/smoke_track.py --demo gears --out /tmp/shots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

GPU_ARGS = ["--use-angle=vulkan", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan", "--ignore-gpu-blocklist"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:5173/robosimstudio-website/")
    p.add_argument("--demo", default="gears")
    p.add_argument("--out", default="/tmp/rsw_track")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(args=GPU_ARGS)
        page = b.new_page(viewport={"width": 1600, "height": 900})
        page.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
        page.goto(f"{a.url}#{a.demo}")
        page.wait_for_selector(".handle", state="attached", timeout=60000)
        page.wait_for_timeout(3000)
        page.screenshot(path=str(out / "1_playing.png"))
        box = page.locator(".handle").first.bounding_box()
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        t0 = page.evaluate("() => document.getElementById('vid').currentTime")
        page.mouse.move(cx, cy)
        page.mouse.down()
        for i in range(1, 31):
            page.mouse.move(cx - 4 * i, cy + 2 * i)
            page.wait_for_timeout(20)
        page.screenshot(path=str(out / "2_scrub.png"))
        page.mouse.up()
        page.wait_for_timeout(1500)
        page.screenshot(path=str(out / "3_paused.png"))
        state = page.evaluate("""() => ({ video: getComputedStyle(document.getElementById('vid')).opacity,
            t: document.getElementById('vid').currentTime, paused: document.getElementById('vid').paused })""")
        b.close()
    report = {"t_before": t0, "after": state, "console": [l for l in logs if "vite" not in l][-20:]}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
