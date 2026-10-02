"""What every live task's page shares: the phone demo's ``PhoneSession`` options, and serving one.

A task module (``task_<name>.py``) builds its scene, subclasses ``wipe_phone.PhoneSession`` where
the scene needs it, and hands both to :func:`serve`. ``wipe_phone`` is the conference phone
demo's page (``~/rss_ws/phone_demo``), imported by path and never edited.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

RUN_ROOT = Path(os.environ.get("RSW_LIVE", Path.home() / "rss_ws" / "rsw_live"))


RENDER_LABEL = "Render Mode (Slow)"
"""The phone page's per-visitor path-traced view, "World model" there."""

HIDE_VISER_MENU = (
    "<style>[data-testid='floating-panel-handle'] .mantine-ActionIcon-root,"
    "button.mantine-ActionIcon-root:has(svg.tabler-icon-adjustments){display:none!important}</style>"
)
"""viser's own settings menu (the sliders icon in the panel's handle): no theme switch hides it, but
the panel renders raw HTML, so a style rule does."""

PING_TIMEOUT = 60.0
"""Seconds a viewer may take to answer the server's keepalive ping (the websockets default is 20)."""


def harden_websockets() -> None:
    """Every page's websocket server, as viser opens it: a keepalive with more slack, and a log line
    for every connection that ends -- who closed it, the code and reason, and for how long it was
    open -- so a visitor's "it crashed and refreshed" can be traced to a cause. Call before the page
    starts its server."""
    import time as _time

    import websockets.asyncio.server as wss

    if getattr(wss.serve, "_rsw", False):
        return
    serve = wss.serve

    def logged_serve(handler, *args, **kwargs):
        kwargs.setdefault("ping_timeout", PING_TIMEOUT)

        async def logged(conn) -> None:
            t0 = _time.time()
            try:
                await handler(conn)
            finally:
                req = getattr(conn, "request", None)
                headers = req.headers if req is not None else {}
                who = headers.get("X-Forwarded-For") or str(getattr(conn, "remote_address", "?"))
                ua = (headers.get("User-Agent") or "?")[:60]
                proto = conn.protocol
                rcvd, sent = proto.close_rcvd, proto.close_sent
                cause = (f"client sent {rcvd.code} {rcvd.reason!r}" if rcvd is not None
                         else f"server sent {sent.code} {sent.reason!r}" if sent is not None
                         else "dropped without a close frame")
                print(f"[ws] {who} closed after {_time.time() - t0:.0f} s: {cause} ({ua})", flush=True)

        return serve(logged, *args, **kwargs)

    logged_serve._rsw = True
    wss.serve = logged_serve


def polish(wp) -> None:
    """The phone page as the website shows it: the per-visitor path-traced view is called
    "Render Mode (Slow)", and viser's settings menu is gone. Patches ``wipe_phone.PhoneSession``
    in this process (never the file); call before the session is built."""
    cls = wp.PhoneSession
    if getattr(cls, "_rsw_polished", False):
        return
    on_connect, build_gui = cls._on_connect, cls.build_gui

    def _on_connect(self, client) -> None:
        gui = client.gui
        add_checkbox = gui.add_checkbox

        def add_checkbox_renamed(label, *args, **kwargs):
            if label == "World model":
                label, kwargs["hint"] = RENDER_LABEL, "A path-traced picture of your view, a few frames a second"
            return add_checkbox(label, *args, **kwargs)

        gui.add_checkbox = add_checkbox_renamed
        try:
            on_connect(self, client)
        finally:
            del gui.add_checkbox

    def _build_gui(self) -> None:
        build_gui(self)
        self.server.gui.add_html(HIDE_VISER_MENU, order=1000)

    cls._on_connect, cls.build_gui, cls._rsw_polished = _on_connect, _build_gui, True


def phone_opts(opts, scene: str, *, attract: float, render_every: int = 2) -> None:
    """Fill in the options ``wipe_phone.main`` would have parsed, for a scene it does not know."""
    opts.scene = scene
    opts.attract = attract if opts.attract is None else opts.attract
    opts.render_every = render_every if opts.render_every is None else opts.render_every
    opts.distance = 1.0
    opts.world_load = 0.25
    opts.world_fps = 12.0
    opts.warm_world = True
    opts.watchdog = True
    opts.sponge_look = "shell"


def run_dir(opts) -> Path:
    run = RUN_ROOT / "run" / opts.name
    run.mkdir(parents=True, exist_ok=True)
    return run


def serve(session_cls, env, opts, *, cloth_look: str = "shell") -> int:
    """Put ``env`` on a phone page at ``opts.port`` and run it until it quits.

    With ``opts.share`` the page also gets a viser relay link (``current_url.txt`` in its run
    folder); behind the HTTPS gateway it needs none."""
    import wipe_phone as wp

    run = run_dir(opts)
    config = wp.StudioConfig(
        port=opts.port,
        share=False,
        panels=("phone",),
        render_every=opts.render_every,
        live=False,
        cameras=False,
        render_view="raster",
        out=str(run / "studio"),
        cloth_look=cloth_look,
    )
    session = session_cls(env, config=config, opts=opts)
    print(f"[phone] page: http://localhost:{opts.port}", flush=True)
    (run / "current_url.txt").unlink(missing_ok=True)
    if opts.share:
        threading.Thread(target=wp.keep_shared, args=(session.server, run, opts.port), daemon=True, name="share").start()
    session.serve()
    return 0


def stamp(t0: float) -> str:
    return f"{time.time() - t0:.1f} s"
