"""Record the project page's clip of one demo: viser's panel and pop-ups hidden (in this browser only),
the camera brought in on the robot's hand and what it works, the hand dragged along a scripted stroke,
a grey dot drawn where the finger is. The clip plays at 1x simulated time.

The page must have been started with ``RSW_PROBE=<file>``: ``handles.Handle`` then writes where the
hand is, the viewer's camera and the simulation's clock five times a second, and sets the viewer's
camera from ``<file>.view``. Run a copy of the task on a spare GPU so the live demo's driver seat
stays free:

    RSW_PROBE=/tmp/probe_bar.json CUDA_VISIBLE_DEVICES=7 .venv/bin/python live/launch.py \\
        --task bar --name bar_rec --port 8124 --attract 0
    ~/rss_ws/trailer/.venv-tools/bin/python live/tools/record_preview.py bar http://localhost:8124 \\
        /tmp/probe_bar.json ~/rss_ws/rsw_live/www/video [--still]
        ->  OUTDIR/bar.mp4 (960 x 600, H.264, faststart) and OUTDIR/bar.jpg (its poster)
            (--still: only OUTDIR/bar_view.png, the framing, to tune VIEWS)

Timing: the simulations run slower than the wall clock (0.4-0.6x; the cable 1x). Each DevTools
screencast frame carries its wall time, the probe pairs wall time with simulated time, and the clip
is cut on simulated time: a second of the clip is a second of physics.
"""
import base64, json, math, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from playwright.sync_api import sync_playwright

TASK, URL, PROBE, OUT = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
STILL = "--still" in sys.argv[5:]
"""--still: only the framing, to tune VIEWS; --view '<json>' replaces this task's VIEWS entry (tuning)."""
W, H, FPS = 960, 600, 25
FFMPEG = os.path.expanduser("~/rss_ws/RoboSimStudio/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2")
ARGS = ["--use-angle=vulkan", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan", "--ignore-gpu-blocklist"]
HIDE = """div:has(> [data-testid="floating-panel-handle"]), .mantine-Notifications-root, .mantine-Notification-root
          { display: none !important; }"""
VIEWS = {
    # task: (look_at offset from the tool [m, world], along the hint's gesture (k, i, j): k * (leg j - leg i),
    #        leg -1 = where the finger lands, distance from the look_at point [m], and the view's direction:
    #        None = the page's own, or (leg, elevation, yaw) [deg] = from where that leg of the gesture heads)
    "bar": ((0.0, 0.0, -0.03), None, 0.40, None),
    "peel": ((0.0, 0.0, -0.01), (0.9, -1, 0), 0.30, (0, 40.0, 0.0)),  # from beyond the far end: the clear film lifts toward the camera
    "towel": ((0.0, 0.0, -0.05), (1.2, 0, 1), 0.42, None),
    "cable": ((0.0, -0.15, -0.03), None, 0.40, None),
}
STROKE_ZOOM = {"cable": (1.0, 1.0), "towel": (1.8, 1.1)}
"""How far a nearer camera may scale the strokes, across and up the screen (default 1.8 both, keeping the
finger in the frame; the cable's and the towel's lifts would carry the hand out of the top)."""
# strokes: (dx, dy, seconds) relative moves while holding the hand, in 960 x 600 pixels at the page's own
# camera distance; scaled by how much nearer the recording's camera is (at most STROKE_ZOOM, to keep the
# finger in the frame), so the hand travels about as far
STROKES = {
    "bar": [(90, 45, 1.5), (0, 0, 0.6), (-190, 0, 3.0), (190, 0, 3.0), (-190, 0, 3.0), (190, 0, 3.0), (-100, 0, 1.8)],
    "peel": [(0, 60, 3.0), (0, 0, 1.0), (0, 70, 3.5), (0, 0, 1.0), (0, 60, 3.0), (0, 0, 1.0), (0, 60, 3.0)],
    "towel": [(0, -150, 1.6), (-180, 0, 2.6), (0, 120, 1.6), (0, 0, 1.2), (0, -120, 1.4), (150, 0, 2.2), (0, 110, 1.4), (0, 0, 1.0)],
    "cable": [(0, -170, 1.6), (130, 0, 2.2), (0, 200, 2.2), (0, 0, 1.5)],
}
IDLE = 3.0
"""Seconds of the hint before the finger arrives (wall clock)."""
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


def probe():
    for _ in range(40):
        try:
            with open(PROBE) as f:
                return json.load(f)
        except Exception:
            time.sleep(0.05)
    raise SystemExit(f"no probe at {PROBE}: was the page started with RSW_PROBE?")


class Cast:
    """DevTools screencast: every frame with its wall time, acknowledged as the script goes; and the
    simulation's clock, read off the probe while it runs."""

    def __init__(self, page):
        self.page, self.frames, self.pending, self.on = page, [], [], False
        self.clock: list[tuple[float, float]] = []
        self.read_at = 0.0
        self.cdp = page.context.new_cdp_session(page)
        self.cdp.on("Page.screencastFrame", self._frame)

    def _frame(self, ev):
        self.pending.append(ev["sessionId"])
        if self.on:
            self.frames.append((float(ev["metadata"]["timestamp"]), base64.b64decode(ev["data"])))

    def start(self):
        self.on = True
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": W, "maxHeight": H})

    def pump(self):
        while self.pending:
            try:
                self.cdp.send("Page.screencastFrameAck", {"sessionId": self.pending.pop()})
            except Exception:
                pass
        if self.on and time.time() - self.read_at > 0.1:
            self.read_at = time.time()
            d = probe()
            if "sim_time" in d and (not self.clock or d["wall"] > self.clock[-1][0]):
                self.clock.append((float(d["wall"]), float(d["sim_time"])))

    def stop(self):
        self.on = False
        self.cdp.send("Page.stopScreencast")

    def write(self, mp4: Path) -> tuple[float, float]:
        """The frames cut on simulated time. Returns (clip seconds, simulated seconds per wall second)."""
        fr = sorted(self.frames, key=lambda f: f[0])
        wall = np.array([c[0] for c in self.clock])
        sim = np.array([c[1] for c in self.clock])
        if len(wall) < 2:
            raise SystemExit("the probe gave no simulation clock")
        rate = float((sim[-1] - sim[0]) / (wall[-1] - wall[0]))
        # a frame's simulated time, from the probe's pairs (extrapolated at the mean rate past either end)
        at = np.interp([f[0] for f in fr], wall, sim)
        at = np.where(np.array([f[0] for f in fr]) < wall[0], sim[0] - (wall[0] - np.array([f[0] for f in fr])) * rate, at)
        at = np.where(np.array([f[0] for f in fr]) > wall[-1], sim[-1] + (np.array([f[0] for f in fr]) - wall[-1]) * rate, at)
        enc = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "mjpeg",
                                "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "23",
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(mp4)], stdin=subprocess.PIPE)
        j = 0
        n = int((at[-1] - at[0]) * FPS)
        for k in range(n):
            while j + 1 < len(fr) and at[j + 1] <= at[0] + k / FPS:
                j += 1
            enc.stdin.write(fr[j][1])
        enc.stdin.close()
        enc.wait()
        return n / FPS, rate


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


def frame_view(page, cast):
    """Bring the camera in on the tool; returns how much nearer it is than the page's own."""
    d = probe()
    cam = list(d["cams"].values())[-1]
    tool = np.asarray(d["tool"], dtype=float)
    eye0, look0 = np.asarray(cam["position"], dtype=float), np.asarray(cam["look_at"], dtype=float)
    offset, along, dist, heading = (json.loads(sys.argv[sys.argv.index("--view") + 1]) if "--view" in sys.argv
                                    else VIEWS[TASK])
    legs = [np.zeros(3)] + [np.asarray(leg, dtype=float) for leg in d["legs"]]
    look = tool + np.asarray(offset, dtype=float)
    if along is not None:
        k, i, j = along
        look = look + k * (legs[j + 1] - legs[i + 1])
    if heading is None:
        back = eye0 - look0
    else:
        leg, elev, yaw = heading
        h = legs[leg + 1] - legs[leg]
        h = np.array([h[0], h[1], 0.0]) / max(np.hypot(h[0], h[1]), 1e-9)
        c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        h = np.array([c * h[0] - s_ * h[1], s_ * h[0] + c * h[1], 0.0])
        back = math.cos(math.radians(elev)) * h + np.array([0.0, 0.0, math.sin(math.radians(elev))])
    back /= np.linalg.norm(back)
    view = {"look_at": look.tolist(), "position": (look + dist * back).tolist()}
    tmp = PROBE + ".view.tmp"
    with open(tmp, "w") as f:
        json.dump(view, f)
    os.replace(tmp, PROBE + ".view")
    for _ in range(60):  # until the page reports the new camera
        wait(cast, 0.2)
        got = np.asarray(list(probe()["cams"].values())[-1]["position"], dtype=float)
        if np.linalg.norm(got - np.asarray(view["position"])) < 1e-3:
            break
    else:
        raise SystemExit("the page never took the recording camera (is handles.py's view reader live?)")
    wait(cast, 1.0)
    depth0 = float((tool - eye0) @ ((look0 - eye0) / np.linalg.norm(look0 - eye0)))
    return depth0 / float((tool - (look + dist * back)) @ -back)


OUT.mkdir(parents=True, exist_ok=True)
try:  # a camera left by an earlier take would greet this browser too: start from the page's own
    os.remove(PROBE + ".view")
except FileNotFoundError:
    pass
with sync_playwright() as pw:
    browser = pw.chromium.launch(args=ARGS)
    ctx = browser.new_context(viewport={"width": W, "height": H})
    page = ctx.new_page()
    cast = Cast(page)
    page.goto(URL, wait_until="load")
    wait(cast, 6.0)
    page.add_style_tag(content=HIDE)
    zoom = frame_view(page, cast)
    if STILL:
        page.screenshot(path=str(OUT / f"{TASK}_view.png"))
        print(f"{TASK}: framing -> {OUT / f'{TASK}_view.png'} (zoom {zoom:.2f}x)")
        ctx.close()
        browser.close()
        raise SystemExit(0)
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
        kx, ky = (min(zoom, c) for c in STROKE_ZOOM.get(TASK, (1.8, 1.8)))
        nxt = (at[0] + dx * kx, at[1] + dy * ky)
        glide(cast, at, nxt, sec)
        at = nxt
    wait(cast, 0.4)
    page.mouse.up()
    glide(cast, at, REST, 0.8)
    wait(cast, 1.5)
    cast.stop()
    mp4 = OUT / f"{TASK}.mp4"
    dur, rate = cast.write(mp4)
    ctx.close()
    browser.close()
subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{0.6 * dur:.2f}", "-i", str(mp4), "-frames:v", "1", "-q:v", "3",
                str(OUT / f"{TASK}.jpg")], check=True)
print(f"{TASK}: zoom {zoom:.2f}x, grabbed at ({grab[0]:.0f}, {grab[1]:.0f}); simulation at {rate:.2f}x real time; "
      f"{dur:.1f} s of physics -> {mp4} ({mp4.stat().st_size / 1e3:.0f} KB)")
