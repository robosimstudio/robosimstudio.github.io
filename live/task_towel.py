"""The towel task: ``cloth_fold/towel_half`` (``towel_scene.py``) on the phone page.

The task's own ``FoldRoutine`` takes the towel's edge once, at start-up; the rig then narrows
the grip to the fingers, lifts the support hand off and captures that moment. From there the
carrying hand is the visitors', "New towel" (and the watchdog) restore the captured moment in
milliseconds, and a shared **Softness** slider sets how limp the towel is. The page stands in the
wipe's room (``backdrop.py``: the bar's jigsaw mat, slat wall and lab, turned to face this camera).
"""

from __future__ import annotations

import collections
import time

import numpy as np

import backdrop
import pages
import towel_scene as ts

WORDS = {
    "title": "Fold the towel",
    "reink": "New towel",
    "hint": "drag me",
    "how": "Drag the robot's hand: up lifts the towel's edge, sideways carries it over. "
    "Drag empty space to look around.",
    "done": "Folded!",
}

SOFTNESS_MARKS = ((0, "canvas"), (100, "silk"))
"""The slider's two ends (0..100; the towel itself sits at 60): see ``towel_scene.edge_ke_for``.
A third mark at 60 collided with "silk" in the phone page's narrow sheet."""


SOFTNESS_IDLE_RESET = 60.0
"""Seconds since anyone moved the slider before Softness goes back to the towel's own, once the page
is idle (its autopilot is running). The setting is shared: the next visitor should meet the towel,
not whatever the last one left (a test left it at silk)."""


RESCUE_GAP = 10.0
"""Seconds between two watchdog rescues: a towel that breaks again at once is not rescued in a loop."""


LOOK = "fluffy"
"""How the towel is drawn (the studio's garment styles): ``fluffy`` is the core plus translucent pile
shells and a sprinkle of soft fuzz points (the user's "bubbles"), ``shell`` the core alone."""


FUZZ_CAP = 12_000
"""Fuzz points on the towel (the studio's own cap is 48,000). Every point is re-sent every frame:
48,000 cost 4.7 MB/s per viewer (measured); fewer, proportionally larger points keep the same cover."""


def lighter_fuzz(cap: int) -> str:
    """Fewer, larger fuzz points: same coverage, a fraction of the bytes. The studio reads these
    module constants when it lays out the fuzz and draws it, so setting them here is enough."""
    import robosimstudio_studio.garments as g

    base_cap = 48_000
    scale = (base_cap / max(cap, 1)) ** 0.5
    g.FUZZ_CAP = int(cap)
    g.FUZZ_POINT_SIZE = 0.0017 * scale
    g.FUZZ_TUFT_SIZE = 0.0024 * scale
    return f"{cap} fuzz points at {1000 * g.FUZZ_POINT_SIZE:.1f}/{1000 * g.FUZZ_TUFT_SIZE:.1f} mm"


STILL_MM = 1.0
"""The towel's drawing is re-sent only once a point has moved this far [mm] since the last one sent
(or it was shown or hidden). The studio re-sends the whole garment -- core, pile shells, fuzz --
every frame, and for a towel lying still that was all of its 0.5-4.7 MB/s a viewer (measured)."""


def send_only_motion(viewer, still: float = STILL_MM / 1000.0) -> None:
    """Wrap the viewer's per-garment draw so frames that would repeat the last one sent are skipped.
    viser keeps every node's last state for late joiners, so nothing is lost by not re-sending it."""
    draw_garment = viewer._draw_garment
    sent: dict[str, tuple[np.ndarray, bool]] = {}

    def draw(garment, points, hidden) -> None:
        last = sent.get(garment.name)
        if (last is not None and last[1] == hidden and last[0].shape == points.shape
                and float(np.abs(points - last[0]).max()) < still):
            return
        sent[garment.name] = (np.array(points, copy=True), hidden)
        draw_garment(garment, points, hidden)

    viewer._draw_garment = draw


def main(opts, rest) -> int:
    import argparse

    import wipe_phone as wp

    p = argparse.ArgumentParser()
    p.add_argument("--look", choices=("fluffy", "shell", "flat"), default=LOOK)
    p.add_argument("--fuzz-cap", type=int, default=FUZZ_CAP)
    a = p.parse_known_args(rest)[0]
    look = a.look
    if look == "fluffy":
        print(f"[live] towel look: fluffy, {lighter_fuzz(a.fuzz_cap)}", flush=True)

    wp.WORDS["towel"] = WORDS
    pages.phone_opts(opts, "towel", attract=60.0)

    class TowelSession(wp.PhoneSession):
        """The phone page on the towel: the task's own grasp once, then the hand is the visitors'."""

        def __init__(self, sim, *, config, opts) -> None:
            self.rig: ts.Rig | None = None
            self.softness = round(100.0 * ts.softness_for(ts.EDGE_KE_DEFAULT))
            """The slider's value, 0..100, as applied to the towel. One for everybody."""
            self.softness_slider = None
            self.softness_requests: collections.deque = collections.deque()
            """Values the slider's viser callbacks asked for; applied on the main thread."""
            self.softness_default = self.softness
            self._softness_moved_at = time.time()
            self._rescued_at = -1e9
            """When the watchdog last laid a fresh towel (wall clock)."""
            super().__init__(sim, config=config, opts=opts)
            send_only_motion(self.viewer)
            self.setup_text = "Setting up -- the robot is taking the towel's edge..."
            if "stats" in self.phone_widgets:  # drawn by the base page with the sponge's words
                self.phone_widgets["stats"].content = self.setup_text
            self.each_step(self.apply_softness)

        # -- the page -------------------------------------------------------------------
        def build_gui(self) -> None:
            super().build_gui()
            slider = self.server.gui.add_slider(
                "Softness",
                min=0,
                max=100,
                step=1,
                initial_value=int(self.softness),
                marks=SOFTNESS_MARKS,
                hint="How limp the towel is: its bending stiffness, canvas to silk. Everyone shares it.",
                order=2.5,
            )

            @slider.on_update
            def _(event) -> None:
                # The studio's threading contract: a viser callback only enqueues. The page's own
                # writes (client_id None) are already applied.
                if event.client_id is not None:
                    self.softness_requests.append(int(event.target.value))

            self.softness_slider = slider

        def apply_softness(self) -> None:
            """Main thread, every control step: the latest slider value, live on the towel; back to
            the towel's own once nobody is around (see ``SOFTNESS_IDLE_RESET``)."""
            if self.rig is None:
                return
            if self.softness_requests:
                value = int(self.softness_requests[-1])
                self.softness_requests.clear()
                self._softness_moved_at = time.time()
                self._set_softness(value)
            elif (self.softness != self.softness_default and self.controls.autopilot is not None
                  and time.time() - self._softness_moved_at > SOFTNESS_IDLE_RESET):
                self._set_softness(self.softness_default)

        def _set_softness(self, value: int) -> None:
            value = int(min(max(value, 0), 100))
            ke = self.rig.set_edge_ke(ts.edge_ke_for(value / 100.0))
            if value != self.softness:
                print(f"[phone] softness {self.softness} -> {value} (edge_ke {ke:.3g})", flush=True)
            self.softness = value
            if self.softness_slider is not None and int(self.softness_slider.value) != value:
                self.softness_slider.value = value  # every phone's slider shows what the towel is

        # -- the loop -------------------------------------------------------------------
        def setup(self) -> None:
            env = self.env
            n, t0, fold = 0, time.time(), None
            for fold in ts.pre_roll(env):
                n += 1
                if n % 2 == 0:
                    self.refresh_view()
                if n % 150 == 0:  # the supervisor reads a silent log as a hang
                    print(f"[phone] pre-roll {n} steps ({fold.phase})", flush=True)
            self.rig = ts.Rig(env, fold)
            for _ in self.rig.handover():
                n += 1
                if n % 2 == 0:
                    self.refresh_view()
            self._set_softness(self.softness)
            self._lay_out()
            look, offset = (np.asarray(v, dtype=np.float64) for v in wp.VIEWS["towel"])
            pages.opening_camera(self.server, look, look + offset * self.opts.distance)  # the carry's view
            for client in self.server.get_clients().values():
                self.aim(client)
            print(
                f"[phone] pre-roll: {n} steps in {time.time() - t0:.1f} s, grip {len(self.rig.grip.held)} particles; "
                "the hand is the phones' now",
                flush=True,
            )
            self.controls.touched_at = time.time()

        def _lay_out(self) -> None:
            self.wipe = self.rig.towel()
            self.map = ts.TowelMap(self.wipe)
            wp.VIEWS["towel"] = ts.view(self.wipe)

        def restart(self, why: str) -> None:
            """A fresh towel: the rig's captured handover, put back. Milliseconds, so the page
            never freezes on the old towel (the routine's replay took ~30 s, the page stalled)."""
            if self.rig is None or self.rig.snapshot is None:
                if self.rig is not None:  # give the held particles their mass back for the replay
                    self.rig.grip.hold([])
                    self.rig = None
                super().restart(why)
                return
            broken = why.startswith("Something") and self.wipe is not None
            reason = self.wipe.why if broken else ""
            detail = self.wipe.detail if broken else ""
            if broken:
                now = time.time()
                if now - self._rescued_at < RESCUE_GAP:
                    return  # the fresh towel is breaking again at once: leave it to "New towel"
                self._rescued_at = now
            if why.startswith("A new film"):  # the base page's words for a reset request
                why = "New towel."
            t = time.perf_counter()
            lap = bool(getattr(self.wipe, "lap_reset", False))
            self.rig.restore()
            self._set_softness(self.softness)  # the material lives on the model, not the state
            controls = self.controls
            controls.autopilot = None  # its generator belongs to the old towel
            if not lap:  # a person asked (or the towel broke under one): idle starts now
                controls.touched_at = time.time()
            self._lay_out()
            self.on_reset({}, {})
            self.begin_run()
            print(
                f"[phone] new towel ({reason or why.rstrip('.').lower()}{': ' + detail if detail else ''}) "
                f"in {1000 * (time.perf_counter() - t):.0f} ms",
                flush=True,
            )
            if broken:
                for client in self.server.get_clients().values():
                    client.add_notification(
                        title="A fresh towel",
                        body=f"{reason[:1].upper()}{reason[1:]}, so here is a new one." if reason else "Here is a new one.",
                        auto_close_seconds=3.0,
                        with_close_button=False,
                    )

        def set_readouts(self, clean: float) -> None:
            del clean  # how far the fold has come: not shown (no success on the page)
            pages.quiet_panel(self)  # and no readings: only the wipe keeps its force and pressure

    t = time.time()
    env = ts.build_env()
    print(f"[phone] {backdrop.dress(env, 'towel')}", flush=True)
    # Until the hand has the edge, look at the towel lying there from the bench's open side (+x;
    # ``towel_scene.view`` takes over once the carry's axes are known).
    q = env.state.particle_q.numpy()
    dh = env.handles.deformables.get("towel") or next(iter(env.handles.deformables.values()))
    mid = np.asarray(q[np.asarray(dh.particles)].mean(axis=0), dtype=np.float64)
    wp.VIEWS["towel"] = (tuple((mid + np.array([0.0, 0.0, -0.04])).tolist()), (0.66, 0.0, 0.46))
    print(f"[phone] scene built in {pages.stamp(t)}", flush=True)
    return pages.serve(TowelSession, env, opts, cloth_look=look)
