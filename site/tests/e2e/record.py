"""Screen-record a scripted visit to the site: sharp frames from the DevTools screencast.

The same recorder as ``~/rss_ws/slides/capture.py``: every composited frame arrives as a JPEG with
its own timestamp and is resampled to a constant 30 fps by ffmpeg's concat demuxer (Playwright's
own recorder is a sub-megabit VP8 stream). Headless Chromium draws no pointer, so the recording
injects a faint ring that follows it (the page itself has none).

    ~/rss_ws/trailer/.venv-tools/bin/python site/tests/e2e/record.py \\
        --url https://keplerc.github.io/robosimstudio-website/ --out /tmp/rsw_visit
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

GPU_ARGS = ["--use-angle=vulkan", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan", "--ignore-gpu-blocklist"]
FFMPEG = str(Path.home() / "rss_ws/RoboSimStudio/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2")
W, H = 1920, 1080

POINTER_JS = """
(() => {
  const ring = document.createElement('div');
  ring.style.cssText = 'position:fixed;left:0;top:0;width:34px;height:34px;margin:-17px 0 0 -17px;border-radius:50%;'
    + 'border:2px solid rgba(255,255,255,.75);box-shadow:0 0 6px rgba(0,0,0,.35);pointer-events:none;z-index:99999;'
    + 'transition:transform .08s,background .08s;opacity:0';
  const put = () => document.body && !ring.isConnected && document.body.appendChild(ring);
  addEventListener('pointermove', (e) => { put(); ring.style.opacity = 1; ring.style.left = e.clientX + 'px'; ring.style.top = e.clientY + 'px'; }, true);
  addEventListener('pointerdown', () => { ring.style.transform = 'scale(.72)'; ring.style.background = 'rgba(255,255,255,.22)'; }, true);
  addEventListener('pointerup', () => { ring.style.transform = ''; ring.style.background = ''; }, true);
})();
"""


class Screencast:
    """DevTools ``Page.startScreencast`` into a directory of JPEGs with timestamps."""

    def __init__(self, ctx, page, frames_dir: Path, quality: int = 90):
        self.dir = frames_dir
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        self.stamps: list[float] = []
        self.cdp = ctx.new_cdp_session(page)
        self.quality = quality
        self.cdp.on("Page.screencastFrame", self._on_frame)

    def _on_frame(self, params) -> None:
        i = len(self.stamps)
        (self.dir / f"f{i:06d}.jpg").write_bytes(base64.b64decode(params["data"]))
        self.stamps.append(float(params["metadata"]["timestamp"]))
        self.cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})

    def start(self) -> None:
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": self.quality, "maxWidth": W, "maxHeight": H, "everyNthFrame": 1})

    def stop(self) -> None:
        self.cdp.send("Page.stopScreencast")

    def encode(self, mp4: Path, fps: int = 30, crf: int = 20) -> None:
        lines = ["ffconcat version 1.0"]
        for i, t in enumerate(self.stamps):
            nxt = self.stamps[i + 1] if i + 1 < len(self.stamps) else t + 1.0 / fps
            lines += [f"file 'f{i:06d}.jpg'", f"duration {max(1e-3, nxt - t):.5f}"]
        (self.dir / "frames.ffconcat").write_text("\n".join(lines) + "\n")
        subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(self.dir / "frames.ffconcat"),
                        "-vf", f"fps={fps},scale={W}:{H}:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-preset", "slow",
                        "-crf", str(crf), "-movflags", "+faststart", str(mp4)], check=True)


class Hand:
    """A visitor's pointer: min-jerk glides sampled at 60 Hz, waits that keep frames flowing."""

    def __init__(self, page):
        self.page = page
        self.x = self.y = 0.0

    def wait(self, s: float) -> None:
        self.page.wait_for_timeout(int(s * 1000))

    def to(self, x: float, y: float, s: float = 0.6) -> None:
        x0, y0 = self.x, self.y
        n = max(1, int(s * 60))
        t_end = time.time() + s
        for i in range(1, n + 1):
            u = i / n
            k = u * u * u * (10 - 15 * u + 6 * u * u)  # min-jerk
            self.page.mouse.move(x0 + (x - x0) * k, y0 + (y - y0) * k)
            left = t_end - (n - i) / 60 - time.time()
            if left > 0:
                self.page.wait_for_timeout(int(left * 1000))
        self.x, self.y = x, y

    def jump(self, x: float, y: float) -> None:
        self.page.mouse.move(x, y)
        self.x, self.y = x, y

    def down(self) -> None:
        self.page.mouse.down()

    def up(self) -> None:
        self.page.mouse.up()


def axes(page) -> dict:
    """The free demo's drag axes on screen: CSS px per metre along a and b, and the box."""
    st = json.loads(page.evaluate("() => JSON.stringify(window.__rsw.debugState())"))
    sc = float(st["scale"])
    return {"A": [v * sc for v in st["axesPx"]["A"]], "B": [v * sc for v in st["axesPx"]["B"]], "box": st["box"], "t": st["target"]}


def pick(page, hand: "Hand", demo: str) -> None:
    tile = page.locator(f"#picker [data-id='{demo}']").first
    tile.scroll_into_view_if_needed()
    tb = tile.bounding_box()
    hand.to(tb["x"] + tb["width"] / 2, tb["y"] + tb["height"] / 2, 0.7)
    page.mouse.click(hand.x, hand.y)


def free_drag(page, hand: "Hand", moves: list[tuple[float, float, float]], settle: float = 2.5) -> None:
    """Grab the ball and move it by (da, db) metres along the demo's own axes, each over s seconds."""
    ax = axes(page)
    bx, by = center(page, ".ball")
    hand.to(bx, by, 0.6)
    hand.down()
    for da, db, sec in moves:
        hand.to(hand.x + da * ax["A"][0] + db * ax["B"][0], hand.y + da * ax["A"][1] + db * ax["B"][1], sec)
    hand.up()
    hand.to(hand.x + 150, hand.y + 120, 0.5)
    hand.wait(settle)


def scrub(page, hand: "Hand", dx: float, dy: float) -> None:
    hx, hy = center(page, ".handle")
    hand.to(hx, hy, 0.6)
    hand.down()
    hand.to(hx + dx, hy + dy, 1.6)
    hand.wait(0.3)
    hand.to(hx + dx * 0.3, hy + dy * 0.3, 1.2)
    hand.up()
    hand.to(hand.x + 180, hand.y + 150, 0.5)
    hand.wait(2.5)


def center(page, sel: str) -> tuple[float, float]:
    b = page.locator(sel).first.bounding_box()
    return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="https://keplerc.github.io/robosimstudio-website/")
    p.add_argument("--out", default="/tmp/rsw_visit")
    p.add_argument("--tour", action="store_true", help="every free drag, then two rollouts")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    beats: list[tuple[float, str]] = []
    logs: list[str] = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(args=GPU_ARGS)
        ctx = b.new_context(viewport={"width": W, "height": H}, device_scale_factor=1)
        ctx.add_init_script(POINTER_JS)
        page = ctx.new_page()
        page.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
        cast = Screencast(ctx, page, out / "frames")
        hand = Hand(page)
        t0 = time.time()

        def beat(name: str) -> None:
            beats.append((round(time.time() - t0, 2), name))

        cast.start()
        if a.tour:
            page.goto(f"{a.url}#wipe")
            page.wait_for_function("() => { const b = document.querySelector('.ball'); return b && b.style.display !== 'none' && b.style.width; }", timeout=60000)
            hand.jump(W * 0.62, H * 0.80)
            beat("wipe: attract"); hand.wait(3.0)
            beat("wipe: drag")
            free_drag(page, hand, [(0.0, -0.02, 0.5), (0.08, -0.02, 1.3), (-0.12, -0.02, 1.6), (0.10, 0.0, 1.3), (0.0, 0.06, 0.6)])
            for demo, moves in (("peel", [(0.06, 0.0, 1.5), (0.06, 0.004, 1.5), (0.05, -0.004, 1.4)]),
                                ("whip", [(0.10, 0.0, 0.35), (-0.14, 0.0, 0.4), (0.12, 0.02, 0.4), (-0.08, 0.0, 0.35)]),
                                ("towel", [(0.0, 0.12, 1.2), (0.18, 0.04, 1.6), (0.0, -0.10, 1.0), (-0.12, 0.05, 1.4)])):
                beat(f"{demo}: pick"); pick(page, hand, demo)
                page.wait_for_function("() => { const b = document.querySelector('.ball'); return b && b.style.display !== 'none' && b.style.width; }", timeout=60000)
                hand.to(W * 0.55, H * 0.55, 0.6); hand.wait(3.0)
                beat(f"{demo}: drag"); free_drag(page, hand, moves, settle=3.0)
            for demo, (dx, dy) in (("gears", (-120, 60)), ("crate", (-160, 20))):
                beat(f"{demo}: pick"); pick(page, hand, demo)
                page.wait_for_selector(".handle", state="attached", timeout=60000)
                hand.to(W * 0.55, H * 0.55, 0.6); hand.wait(4.0)
                beat(f"{demo}: scrub"); scrub(page, hand, dx, dy)
            cast.stop()
            b.close()
            mp4 = out / "visit.mp4"
            cast.encode(mp4)
            (out / "timeline.json").write_text(json.dumps({"beats": beats, "frames": len(cast.stamps),
                                                            "console": [x for x in logs if "error" in x.lower()][:20]}, indent=1))
            print(json.dumps({"mp4": str(mp4), "bytes": mp4.stat().st_size, "frames": len(cast.stamps), "beats": beats}, indent=1))
            return 0
        page.goto(f"{a.url}#wipe")
        page.wait_for_function("() => { const b = document.querySelector('.ball'); return b && b.style.display !== 'none' && b.style.width; }", timeout=60000)
        beat("wipe: attract video (RTX)")
        hand.jump(W * 0.62, H * 0.80)
        hand.wait(3.5)
        bx, by = center(page, ".ball")
        hand.to(bx, by, 0.7)
        hand.down()
        beat("wipe: grab the ball -- live")
        hand.to(bx + 10, by + 55, 0.5)  # press onto the bar
        for dx, s in ((230, 1.3), (-330, 1.6), (260, 1.3), (-120, 0.8)):
            hand.to(hand.x + dx, hand.y + 4 * math.copysign(1, dx), s)
        hand.to(hand.x + 20, hand.y - 70, 0.6)  # lift off
        hand.wait(0.3)
        hand.up()
        beat("wipe: let go -- settles, RTX still dissolves in")
        hand.to(hand.x + 140, hand.y + 120, 0.6)
        hand.wait(2.6)
        bx, by = center(page, ".ball")
        hand.to(bx, by, 0.5)
        hand.down()
        beat("wipe: flick")
        hand.to(bx - 70, by + 40, 0.25)
        hand.to(bx + 160, by + 30, 0.28)
        hand.up()
        hand.to(hand.x + 120, hand.y + 150, 0.5)
        hand.wait(3.0)
        beat("gears: pick")
        tile = page.locator("#picker [data-id='gears'], #picker button:has-text('Gear train')").first
        tb = tile.bounding_box()
        hand.to(tb["x"] + tb["width"] / 2, tb["y"] + tb["height"] / 2, 0.8)
        page.mouse.click(hand.x, hand.y)
        page.wait_for_selector(".handle", state="attached", timeout=60000)
        beat("gears: RTX rollout video")
        hand.to(W * 0.55, H * 0.55, 0.8)
        hand.wait(4.5)
        hx, hy = center(page, ".handle")
        hand.to(hx, hy, 0.6)
        hand.down()
        beat("gears: grab the gripper -- scrub live")
        hand.to(hx - 120, hy + 60, 1.6)
        hand.wait(0.4)
        hand.to(hx + 60, hy - 20, 1.8)
        hand.to(hx - 40, hy + 30, 1.0)
        hand.up()
        beat("gears: let go -- lands on the RTX frame, plays on")
        hand.to(hand.x + 200, hand.y + 180, 0.6)
        hand.wait(4.0)
        cast.stop()
        b.close()
    mp4 = out / "visit.mp4"
    cast.encode(mp4)
    (out / "timeline.json").write_text(json.dumps({"beats": beats, "frames": len(cast.stamps),
                                                    "console": [x for x in logs if "error" in x.lower()][:20]}, indent=1))
    print(json.dumps({"mp4": str(mp4), "bytes": mp4.stat().st_size, "frames": len(cast.stamps), "beats": beats}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
