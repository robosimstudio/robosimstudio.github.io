"""Record the front page's preview clip of one demo: viser's panel and pop-ups hidden (in this browser
only), the robot's hand dragged along a scripted stroke, a grey dot drawn where the finger is.

The page must have been started with ``RSW_PROBE=<file>``: ``handles.Handle`` then writes where the
hand is, and the viewer's camera, five times a second. Run a copy of the task on a spare GPU so the
live demo's driver seat stays free:

    RSW_PROBE=/tmp/probe_bar.json CUDA_VISIBLE_DEVICES=7 .venv/bin/python live/launch.py \\
        --task bar --name bar_rec --port 8124 --attract 0
    ~/rss_ws/trailer/.venv-tools/bin/python live/tools/record_preview.py bar http://localhost:8124 \\
        /tmp/probe_bar.json ~/rss_ws/rsw_live/www/video
        ->  OUTDIR/bar.mp4 (640 x 400, H.264, faststart) and OUTDIR/bar.jpg (its poster)

Frames come from DevTools' screencast with their own timestamps, so the clip runs in real time from
the first idle frame (the visitor's hint showing) to the last.
"""
import base64, json, math, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from playwright.sync_api import sync_playwright

TASK, URL, PROBE, OUT = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
W, H, FPS = 960, 600, 25
FFMPEG = os.path.expanduser("~/rss_ws/RoboSimStudio/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2")
ARGS = ["--use-angle=vulkan", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan", "--ignore-gpu-blocklist"]
HIDE = """div:has(> [data-testid="floating-panel-handle"]), .mantine-Notifications-root, .mantine-Notification-root
          { display: none !important; }"""
# strokes: (dx, dy, seconds) relative moves while holding the hand, in 960 x 600 pixels
STROKES = {
    "bar": [(90, 45, 1.5), (0, 0, 0.6), (-190, 0, 3.0), (190, 0, 3.0), (-120, 0, 2.0)],
    "peel": [(0, 60, 3.0), (0, 0, 1.0), (0, 70, 3.5), (0, 0, 1.0), (0, 60, 3.0)],
    "towel": [(0, -150, 1.6), (-180, 0, 2.6), (0, 120, 1.6), (0, 0, 1.2), (0, -120, 1.4), (150, 0, 2.2)],
    "cable": [(0, -170, 1.6), (130, 0, 2.2), (0, 200, 2.2), (0, 0, 1.5)],
}
IDLE = 3.0
"""Seconds of the hint before the finger arrives."""
REST = (900, 560)
FINGER = """(() => { const d = document.createElement('div');
  d.style.cssText = 'position:fixed;left:0;top:0;width:24px;height:24px;margin:-12px 0 0 -12px;border-radius:50%;' +
    'background:rgba(30,30,30,0.28);border:2px solid rgba(255,255,255,0.9);box-shadow:0 1px 4px rgba(0,0,0,0.4);' +
    'pointer-events:none;z-index:99999;transition:transform 0.12s ease, background 0.12s ease;';
  document.body.appendChild(d);
  addEventListener('pointermove', e => { d.style.left = e.clientX + 'px'; d.style.top = e.clientY + 'px'; }, true);
  addEventListener('pointerdown', () => { d.style.transform = 'scale(0.72)'; d.style.background = 'rgba(30,30,30,0.5)'; }, true);
  addEventListener('pointerup', () => { d.style.transform = ''; d.style.background = 'rgba(30,30,30,0.28)'; }, true);
})()"""


class Cast:
    """DevTools screencast: every frame with its timestamp, acknowledged as the script goes."""

    def __init__(self, page):
        self.page, self.frames, self.pending, self.on = page, [], [], False
        self.cdp = page.context.new_cdp_session(page)
        self.cdp.on("Page.screencastFrame", self._frame)

    def _frame(self, ev):
        self.pending.append(ev["sessionId"])
        if self.on:
            self.frames.append((float(ev["metadata"]["timestamp"]), base64.b64decode(ev["data"])))

    def start(self):
        self.on = True
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 90, "maxWidth": W, "maxHeight": H})

    def pump(self):
        while self.pending:
            try:
                self.cdp.send("Page.screencastFrameAck", {"sessionId": self.pending.pop()})
            except Exception:
                pass

    def stop(self):
        self.on = False
        self.cdp.send("Page.stopScreencast")

    def write(self, mp4: Path) -> float:
        fr = sorted(self.frames, key=lambda f: f[0])
        t0, t1 = fr[0][0], fr[-1][0]
        enc = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "mjpeg",
                                "-i", "-", "-vf", "scale=640:400:flags=lanczos", "-c:v", "libx264", "-preset", "slow", "-crf", "24",
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(mp4)], stdin=subprocess.PIPE)
        j = 0
        for k in range(int((t1 - t0) * FPS)):
            while j + 1 < len(fr) and fr[j + 1][0] <= t0 + k / FPS:
                j += 1
            enc.stdin.write(fr[j][1])
        enc.stdin.close()
        enc.wait()
        return t1 - t0


def wait(cast, sec):
    end = time.time() + sec
    while time.time() < end:
        cast.pump()
        cast.page.wait_for_timeout(15)


def glide(cast, a, b, sec):
    t0 = time.time()
    while True:
        t = min(1.0, (time.time() - t0) / max(sec, 1e-6))
        e = t * t * (3 - 2 * t)
        cast.page.mouse.move(a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e)
        cast.pump()
        if t >= 1.0:
            return
        cast.page.wait_for_timeout(8)


def probe():
    for _ in range(40):
        try:
            with open(PROBE) as f:
                return json.load(f)
        except Exception:
            time.sleep(0.05)
    raise SystemExit(f"no probe at {PROBE}: was the page started with RSW_PROBE?")


def project(p, cam):
    """World point -> CSS pixels through a three.js perspective camera (vertical fov, radians)."""
    eye, look, up = (np.asarray(cam[k], dtype=float) for k in ("position", "look_at", "up"))
    f = look - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = np.asarray(p, dtype=float) - eye
    s = (H / 2) / math.tan(cam["fov"] / 2)
    return W / 2 + s * (d @ r) / (d @ f), H / 2 - s * (d @ u) / (d @ f)


OUT.mkdir(parents=True, exist_ok=True)
with sync_playwright() as pw:
    browser = pw.chromium.launch(args=ARGS)
    ctx = browser.new_context(viewport={"width": W, "height": H})
    page = ctx.new_page()
    cast = Cast(page)
    page.goto(URL, wait_until="load")
    wait(cast, 6.0)
    page.add_style_tag(content=HIDE)
    page.evaluate(FINGER)
    page.mouse.move(REST[0] - 1, REST[1])
    page.mouse.move(*REST)
    wait(cast, 1.0)
    cast.start()
    wait(cast, IDLE)
    d = probe()
    grab = project(d["tool"], list(d["cams"].values())[-1])
    glide(cast, REST, grab, 1.0)
    wait(cast, 0.5)
    page.mouse.down()
    at = grab
    for dx, dy, sec in STROKES[TASK]:
        nxt = (at[0] + dx, at[1] + dy)
        glide(cast, at, nxt, sec)
        at = nxt
    wait(cast, 0.4)
    page.mouse.up()
    glide(cast, at, REST, 0.8)
    wait(cast, 1.5)
    cast.stop()
    mp4 = OUT / f"{TASK}.mp4"
    dur = cast.write(mp4)
    ctx.close()
    browser.close()
subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{0.6 * dur:.2f}", "-i", str(mp4), "-frames:v", "1", "-q:v", "4",
                str(OUT / f"{TASK}.jpg")], check=True)
print(f"{TASK}: grabbed at ({grab[0]:.0f}, {grab[1]:.0f}); {dur:.1f} s -> {mp4} ({mp4.stat().st_size / 1e3:.0f} KB)")
