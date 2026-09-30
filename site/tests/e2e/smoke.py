"""End-to-end smoke test: load a demo, drag the ball, let go, screenshot each phase.

Run with the Playwright venv against a running dev/preview server:

    ~/rss_ws/trailer/.venv-tools/bin/python site/tests/e2e/smoke.py --demo wipe --out /tmp/shots
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

GPU_ARGS = ["--use-angle=vulkan", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan", "--ignore-gpu-blocklist"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:5173/robosimstudio-website/")
    p.add_argument("--demo", default="wipe")
    p.add_argument("--out", default="/tmp/rsw_shots")
    p.add_argument("--mobile", action="store_true")
    p.add_argument("--drag", default="180,60", help="dx,dy in CSS px")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []
    trace: list = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(args=GPU_ARGS)
        if a.mobile:
            dev = dict(pw.devices["iPhone 13"])
            dev.pop("default_browser_type", None)
            ctx = b.new_context(**dev)
        else:
            ctx = b.new_context(viewport={"width": 1600, "height": 900}, device_scale_factor=1)
        page = ctx.new_page()
        page.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
        t0 = time.time()
        page.goto(f"{a.url}#{a.demo}")
        gl = page.evaluate("""() => { const c = document.createElement('canvas').getContext('webgl2');
            const e = c && c.getExtension('WEBGL_debug_renderer_info');
            return e ? c.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'none'; }""")
        page.wait_for_selector(".ball", state="attached", timeout=30000)
        try:
            page.wait_for_function("() => { const b = document.querySelector('.ball'); return b && b.style.display !== 'none' && b.style.width; }", timeout=60000)
        except Exception:
            page.screenshot(path=str(out / "0_timeout.png"))
            print("\n".join(logs[-40:]))
            raise
        ready = time.time() - t0
        page.wait_for_timeout(1500)
        page.screenshot(path=str(out / "1_attract.png"))
        box = page.locator(".ball").bounding_box()
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        dx, dy = (float(v) for v in a.drag.split(","))
        if a.mobile:
            cdp = ctx.new_cdp_session(page)
            def touch(kind, x, y):
                pts = [] if kind == "touchEnd" else [{"x": x, "y": y, "id": 1}]
                cdp.send("Input.dispatchTouchEvent", {"type": kind, "touchPoints": pts})
            touch("touchStart", cx, cy)
            for i in range(1, 41):
                touch("touchMove", cx + dx * i / 40, cy + dy * i / 40)
                page.wait_for_timeout(16)
            page.wait_for_timeout(400)
            page.screenshot(path=str(out / "2_dragging.png"))
            touch("touchEnd", 0, 0)
        else:
            page.mouse.move(cx, cy)
            page.mouse.down()
            for i in range(1, 41):
                page.mouse.move(cx + dx * i / 40, cy + dy * i / 40)
                page.wait_for_timeout(16)
            page.wait_for_timeout(400)
            page.screenshot(path=str(out / "2_dragging.png"))
            page.wait_for_timeout(600)
            page.screenshot(path=str(out / "3_dragging_late.png"))
            trace.append(("before-up", json.loads(page.evaluate("() => JSON.stringify(window.__rsw?.debugState?.() ?? null)"))))
            page.mouse.up()
        for k in range(6):
            page.wait_for_timeout(500)
            trace.append((f"t+{0.5 * (k + 1):.1f}s", json.loads(page.evaluate("() => JSON.stringify(window.__rsw?.debugState?.() ?? null)"))))
        page.screenshot(path=str(out / "4_released.png"))
        state = page.evaluate("""() => ({ still: getComputedStyle(document.getElementById('still')).opacity,
            video: getComputedStyle(document.getElementById('vid')).opacity })""")
        b.close()
    report = {"gl_renderer": gl, "ready_s": round(ready, 2), "layers": state, "trace": trace, "console": logs[-40:]}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
