"""What every live task's page shares: the phone demo's ``PhoneSession`` options, and serving one.

A task module (``task_<name>.py``) builds its scene, subclasses ``wipe_phone.PhoneSession`` where
the scene needs it, and hands both to :func:`serve`. ``wipe_phone`` is the conference phone
demo's page (``~/rss_ws/phone_demo``), imported by path and never edited.
"""

from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path

import numpy as np

RUN_ROOT = Path(os.environ.get("RSW_LIVE", Path.home() / "rss_ws" / "rsw_live"))


RENDER_LABEL = "Render Mode (Slow)"
"""The phone page's per-visitor path-traced view, "World model" there."""

HIDE_VISER_MENU = (
    "<style>[data-testid='floating-panel-handle'] .mantine-ActionIcon-root,"
    "button.mantine-ActionIcon-root:has(svg.tabler-icon-adjustments){display:none!important}</style>"
)
"""viser's own settings menu (the sliders icon in the panel's handle): no theme switch hides it, but
the panel renders raw HTML, so a style rule does."""

PANEL_WIDTH = "medium"
"""viser's floating panel width on a desktop (the phone page asks for "small")."""

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


def fast_catch_up() -> None:
    """A new viewer's catch-up in O(what is still buffered), not O(everything ever sent).

    viser's ``AsyncMessageBuffer.window_generator`` starts every new connection at message id 0 and
    steps through every id ever pushed, though nearly all were superseded long ago and dropped. On a
    page that has drawn for days that is hundreds of millions of Python steps: the cable's new
    viewers waited 26 s for their camera, buttons and robot (measured 10-06), while the loop held the
    GIL on the one thread that serves every viewer -- so everyone already watching froze as well.
    This version reads the ids still buffered (the dict keeps them in id order) whenever the gap to
    the newest message is far larger than the buffer, and otherwise behaves exactly as viser's own.
    Call before the page starts its server."""
    import asyncio

    from viser.infra import _async_message_buffer as amb

    cls = amb.AsyncMessageBuffer
    if getattr(cls.window_generator, "_rsw", False):
        return

    async def window_generator(self, client_id: int):
        last_sent_id = -1
        flush_wait = self.event_loop.create_task(self.flush_event.wait())
        caught_up = announce = False
        while not self.done:
            if announce:  # the window that finished the catch-up has been sent: tell whoever listens
                announce = False
                hook = getattr(self, "on_caught_up", None)
                if hook is not None:
                    try:
                        hook(client_id)
                    except Exception as exc:  # a page detail must never stall the stream
                        print(f"[live] on_caught_up: {type(exc).__name__}: {exc}", flush=True)
            window = []
            most_recent_message_id = self.message_counter - 1
            gap = most_recent_message_id - last_sent_id
            if self.persistent_messages and gap > 4 * len(self.message_from_id) + self.max_window_size:
                if self.atomic_counter == 0:
                    with self.buffer_lock:
                        pending = [k for k in self.message_from_id if last_sent_id < k <= most_recent_message_id]
                    taken = pending[: self.max_window_size]
                    for k in taken:
                        message = self.message_from_id.get(k, None)
                        if message is not None and message.excluded_self_client != client_id:
                            window.append(message)
                    # every id not taken up to the newest was superseded: skip past it
                    last_sent_id = taken[-1] if len(pending) > len(taken) else most_recent_message_id
            else:  # viser's own loop
                while (
                    last_sent_id < most_recent_message_id
                    and len(window) < self.max_window_size
                    and self.atomic_counter == 0
                ):
                    last_sent_id += 1
                    if self.persistent_messages:
                        message = self.message_from_id.get(last_sent_id, None)
                    else:
                        with self.buffer_lock:
                            message = self.message_from_id.pop(last_sent_id, None)
                            if message is not None:
                                self.id_from_redundancy_key.pop(message.redundancy_key(), None)
                    if message is not None and message.excluded_self_client != client_id:
                        window.append(message)

            if not caught_up and self.persistent_messages and last_sent_id >= most_recent_message_id:
                caught_up = announce = True  # fires once this window is out (or now, if it is empty)
            if len(window) > 0:
                yield window
            elif announce:
                continue
            else:
                await self.message_event.wait()
                self.message_event.clear()

            most_recent_message_id = self.message_counter - 1
            if len(window) == 0 or most_recent_message_id == last_sent_id:
                done, pending_tasks = await asyncio.wait([flush_wait], timeout=self.window_duration_sec)
                del pending_tasks
                if flush_wait in done and not self.done:
                    self.flush_event.clear()
                    flush_wait = self.event_loop.create_task(self.flush_event.wait())

    window_generator._rsw = True
    cls.window_generator = window_generator


READY_PING = (
    '<img src="data:," alt="" width="0" height="0" style="position:absolute" '
    "onerror=\"try{parent.postMessage({rsw:'ready'},'*')}catch(e){};this.remove()\">"
)
"""Sent to one viewer once the scene has gone out to it (``fast_catch_up``'s hook): the front page's
loading bar waits for this. viser renders a panel's HTML raw, so an image that fails to load can
run the one line that tells the embedding page."""


def ready_ping(server) -> None:
    """Tell each new viewer's embedding page when its scene has arrived (needs ``fast_catch_up``)."""

    def send(client_id: int) -> None:
        # viser registers a new viewer's handle from a worker thread, and a short catch-up can finish
        # first (the bar, towel and peel did; the cable's long one hid it): wait for the handle
        for _ in range(100):
            client = server.get_clients().get(client_id)
            if client is not None:
                break
            time.sleep(0.03)
        else:
            print(f"[live] ready ping: viewer {client_id} never registered", flush=True)
            return
        handle = client.gui.add_html(READY_PING, order=1001)
        threading.Timer(5.0, lambda: _quietly(handle.remove)).start()

    def ping(client_id: int) -> None:  # called on viser's event-loop thread: never wait there
        threading.Thread(target=send, args=(client_id,), daemon=True, name="ready-ping").start()

    server._websock_server._broadcast_buffer.on_caught_up = ping


def _quietly(fn) -> None:
    try:
        fn()
    except Exception:
        pass


CREASE_DEG = 35.0
"""Edges sharper than this keep hard normals when a mesh is welded for the browser."""


def lighter_meshes(crease_deg: float = CREASE_DEG) -> None:
    """Weld the meshes the page sends. STL links arrive unwelded, three private corners a triangle, so
    the Panda's 12-15k-triangle links went out at 0.8-1.1 MB each and the cable's first load was
    4.4 MB to the bar's 1.4. Welded, then split again along edges sharper than ``crease_deg`` so hard
    corners stay hard, the same triangles cost about a fifth of the vertices; the geometry is
    untouched (a quadric decimation to 5k frayed the open rims). The studio builds each prototype's
    GLB once (``ViewerStudio._glb``, cached), so this runs once a mesh at load, ~30 ms for a Panda
    link. Only the browser's copy changes. Call before the page is built."""
    import trimesh

    import robosimstudio_studio.garments as g

    cls = g.ViewerStudio
    if getattr(cls._glb, "_rsw", False):
        return
    glb = cls._glb

    def _glb(self, data, color, finish):
        try:
            idx = np.asarray(data["indices"])
            if not finish.texture and idx.size >= 1500:  # a textured finish unwelds again for its UVs
                pts = np.asarray(data["points"], dtype=np.float32).reshape(-1, 3)
                welded, inverse = np.unique(np.round(pts, 6), axis=0, return_inverse=True)
                if len(welded) < 0.6 * len(pts):
                    faces = inverse.reshape(-1)[idx.reshape(-1, 3)]
                    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
                    mesh = trimesh.graph.smooth_shade(trimesh.Trimesh(welded, faces, process=False),
                                                      angle=np.radians(crease_deg))
                    data = {**data, "points": np.asarray(mesh.vertices, dtype=np.float32),
                            "indices": np.asarray(mesh.faces, dtype=idx.dtype).reshape(-1)}
        except Exception as exc:  # never lose a mesh over its diet
            print(f"[live] mesh sent as built ({type(exc).__name__}: {exc})", flush=True)
        return glb(self, data, color, finish)

    _glb._rsw = True
    cls._glb = _glb


def opening_camera(server, look, eye, fov_deg: float = 55.0) -> None:
    """Where a new viewer's camera starts, before any of the page's own messages reach it: the page
    aims each viewer once its catch-up is through, and until then viser drew from its default spot
    outside the room (the walls)."""
    cam = server.initial_camera
    cam.up = (0.0, 0.0, 1.0)
    cam.fov = math.radians(fov_deg)
    cam.position = tuple(float(v) for v in eye)
    cam.look_at = tuple(float(v) for v in look)


def polish(wp) -> None:
    """The phone page as the website shows it: the per-visitor path-traced view is called
    "Render Mode (Slow)", viser's settings menu is gone, and nothing on the page checks or announces
    success or shows the simulation's speed -- no progress bar, no "Clean!"/"Peeled!" toasts, a
    readout line that is only the physical reading (contact pressure, bond load, ...). Patches
    ``wipe_phone`` in this process (never the file); call before the session is built."""
    cls = wp.PhoneSession
    if getattr(cls, "_rsw_polished", False):
        return
    on_connect, build_gui = cls._on_connect, cls.build_gui
    # PhoneControls cheers once the field is 95 % done unless it already has: make that always so.
    wp.PhoneControls._celebrated = property(lambda self: True, lambda self, value: None)

    def set_readouts(self, clean: float) -> None:
        del clean  # the task's completion: not shown
        if self.scene_kind != "bar":
            quiet_panel(self)
            return
        # the wipe keeps its contact map and one line: the sponge's force on the bar, and its peak pressure
        w = self.phone_widgets
        now = time.time()
        if self.map is not None and now - self._map_at >= wp.MAP_PERIOD:
            self._map_at = now
            w["map"].image = self.map.render()
        if self.map is None:
            w["stats"].content = self.setup_text
            return
        force = float(self.map.field._press_now[0])
        self._force_shown = 0.8 * getattr(self, "_force_shown", force) + 0.2 * force
        pressure = self.map.caption()  # "peak N kPa", or "not touching" under the field's pressure floor
        w["stats"].content = pressure if pressure == "not touching" else f"{self._force_shown:.1f} N · {pressure}"

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
        # The phone page's "small" floating panel is too narrow on a desktop for its button row: the
        # unlabelled row keeps viser's label column, and "Reset view" ran 31 px past the panel's edge.
        # (A phone shows the panel as a full-width sheet: unaffected.)
        self.server.gui.configure_theme(
            control_layout="floating",
            control_width=PANEL_WIDTH,
            dark_mode=False,
            show_logo=False,
            show_share_button=False,
            brand_color=wp.BRAND,
        )
        self.phone_widgets["progress"].visible = False  # completion is not shown
        self.server.gui.add_html(HIDE_VISER_MENU, order=1000)
        ready_ping(self.server)
        view = wp.VIEWS.get(self.scene_kind)
        if view is not None:  # the opening view, as aim() sets it, for the very first frame
            look, offset = (np.asarray(v, dtype=np.float64) for v in view)
            opening_camera(self.server, look, look + offset * self.opts.distance)

    cls._on_connect, cls.build_gui, cls.set_readouts = _on_connect, _build_gui, set_readouts
    cls._rsw_polished = True


def quiet_panel(session) -> None:
    """No readings on this page: the readout line and the map go once the task is running (the line
    still carries the "Setting up..." text before that)."""
    w = session.phone_widgets
    for key in ("stats", "map"):
        if key in w and w[key].visible:
            w[key].visible = False


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
