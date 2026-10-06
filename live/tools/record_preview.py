"""Record the front page's preview clip of one live demo: viser's panel and pop-ups hidden (in this
browser only), the demo driven by its orange ball along a scripted stroke.

    ~/rss_ws/trailer/.venv-tools/bin/python live/tools/record_preview.py TASK OUTDIR
        ->  OUTDIR/TASK.webm (raw screencast) and OUTDIR/TASK.json (when the drag started and ended)

Then cut and encode it where the front page reads it (~/rss_ws/rsw_live/www/video/, outside the repo):

    FF=~/rss_ws/RoboSimStudio/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
    $FF -ss <start-0.2> -to <end> -i TASK.webm -an -vf "scale=640:400:flags=lanczos,fps=25" -c:v libx264 \
        -preset slow -crf 24 -pix_fmt yuv420p -movflags +faststart ~/rss_ws/rsw_live/www/video/TASK.mp4
    $FF -ss <mid> -i ~/rss_ws/rsw_live/www/video/TASK.mp4 -frames:v 1 -q:v 4 ~/rss_ws/rsw_live/www/video/TASK.jpg

The drag takes the live demo's driver seat for ~15 s.
"""
import json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

TASK, OUT = sys.argv[1], Path(sys.argv[2])
W, H = 960, 600
HIDE = """div:has(> [data-testid="floating-panel-handle"]), .mantine-Notifications-root, .mantine-Notification-root
          { display: none !important; }"""
# strokes: (dx, dy, seconds) relative moves while holding the ball; None = release and re-grab
STROKES = {
    "bar": [(90, 45, 1.5), (0, 0, 0.6), (-190, 0, 3.0), (190, 0, 3.0), (-120, 0, 2.0)],
    "peel": [(0, 60, 3.0), (0, 0, 1.0), (0, 70, 3.5), (0, 0, 1.0), (0, 60, 3.0)],
    "towel": [(0, -150, 1.6), (-180, 0, 2.6), (0, 120, 1.6), (0, 0, 1.2), (0, -120, 1.4), (150, 0, 2.2)],
    "cable": [(0, -170, 1.6), (130, 0, 2.2), (0, 200, 2.2), (0, 0, 1.5)],
}

def ball(png):
    a = np.asarray(Image.open(png).convert("RGB")).astype(np.int32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = (r > 200) & (g > 90) & (g < 175) & (b < 80) & (r - g > 60)
    ys, xs = np.nonzero(m)
    return (float(np.median(xs)), float(np.median(ys))) if len(xs) > 30 else None

with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--use-angle=vulkan", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan", "--ignore-gpu-blocklist"])
    ctx = b.new_context(viewport={"width": W, "height": H}, record_video_dir=str(OUT / f"raw_{TASK}"), record_video_size={"width": W, "height": H})
    page = ctx.new_page()
    t0 = time.time()
    page.goto(f"https://{TASK}.95-133-252-186.sslip.io/", wait_until="load")
    page.add_style_tag(content=HIDE)
    time.sleep(4.0)
    start = time.time() - t0
    page.screenshot(path=str(OUT / f"{TASK}_start.png"))
    p = ball(OUT / f"{TASK}_start.png")
    if p is None:
        raise SystemExit(f"{TASK}: no ball")
    page.mouse.move(*p); page.mouse.down()
    x, y = p
    for dx, dy, secs in STROKES[TASK]:
        n = max(1, int(secs / 0.05))
        for i in range(1, n + 1):
            page.mouse.move(x + dx * i / n, y + dy * i / n); time.sleep(secs / n)
        x, y = x + dx, y + dy
    page.mouse.up()
    time.sleep(1.5)
    end = time.time() - t0
    page.screenshot(path=str(OUT / f"{TASK}_end.png"))
    video = page.video.path()
    ctx.close(); b.close()
    Path(video).rename(OUT / f"{TASK}.webm")
    (OUT / f"{TASK}.json").write_text(json.dumps({"start": start, "end": end}))
    print(TASK, "recorded", round(start, 1), "->", round(end, 1), "s")
