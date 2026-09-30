// A track demo: a recorded, successful rollout.
//
//   PLAYING  the RTX rollout video.
//   SCRUB    the finger took a gripper handle (or dragged the stage): live 3D of
//            the moving bodies over the RTX plate, at the dragged time.
//   PAUSED   let go: snap to a frame, seek the video there exactly (keyframe every
//            8 frames), dissolve it in. Space plays and pauses; arrows step.

import type { Demo, Stage } from "../app/App.ts";
import { toStage, type Placement } from "../app/layout.ts";
import { fetchRssb } from "../data/fetch.ts";
import { Camera } from "../core/camera.ts";
import { Scrubber } from "../core/scrubber.ts";

type Mode = "playing" | "scrub" | "paused" | "calib";
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Manifest = any;

const RESUME_AFTER_S = 8;

export class TrackDemo implements Demo {
  private stage: Stage;
  private m: Manifest;
  private base: string;
  private mode: Mode = "playing";
  private bodies: Uint32Array = new Uint32Array(0);
  private poses: Float32Array = new Float32Array(0);
  /** Each handle's pixel per frame in the view being shown (the scrubber reads it in place). */
  private eePx: Float32Array = new Float32Array(0);
  /** ... as the pipeline projected them for the hero camera, and the points in the world. */
  private eePxHero: Float32Array = new Float32Array(0);
  private eeWorld: Float32Array | null = null;
  private projVersion = -1;
  private H = 1;
  private N = 0;
  private t = 0;
  private scrubber: Scrubber | null = null;
  private handles: HTMLDivElement[] = [];
  /** The handles a visitor can take: an arm that never moves in the take has none. */
  private show: number[] = [];
  private grab: { handle: number; relative: boolean; lastPx: [number, number] } | null = null;
  private lastInput = 0;
  private videoAlpha = 1;
  private seekToken = 0;
  private ready = false;
  private disposed = false;

  constructor(stage: Stage, id: string, manifest: Manifest) {
    this.stage = stage;
    this.m = manifest;
    this.base = stage.base(id);
  }

  /** The camera the page is seen through: the hero's, or the visitor's orbit about it. */
  private get camera(): Camera {
    return this.stage.view.camera;
  }

  canOrbit(): boolean {
    return this.ready;
  }

  roi(): [number, number, number, number] {
    return this.m.roi;
  }

  async start(): Promise<void> {
    const { stage, m, base } = this;
    stage.live.setCamera(m.camera);
    stage.live.setLook(m.look ?? {}, base);
    stage.bg.src = base + m.files.plate;
    stage.still.style.opacity = "0";
    stage.bgInk.style.display = "none";
    stage.stillInk.style.display = "none";
    const v = stage.video;
    v.loop = true;
    v.src = base + m.files.video;
    v.style.opacity = "1";
    void v.play().catch(() => undefined);
    const [_, state, paths] = await Promise.all([
      stage.live.loadGlb(base + m.files.glb, m.nodes, m.look?.shadow_opacity ?? 0.28),
      fetchRssb(base + m.files.state),
      fetchRssb(base + m.files.paths),
    ]);
    if (this.disposed) return;
    this.bodies = state.arrays.bodies.data as Uint32Array;
    this.poses = state.arrays.body_q.data as Float32Array;
    this.N = state.arrays.body_q.shape[0];
    const px = paths.arrays.ee_px;
    this.eePx = px.data as Float32Array;
    this.H = px.shape[1];
    // A hand beside or behind the camera projects to nonsense (the conduit's right arm rises
    // past the close camera between strokes: x -17566 px), and behind it mirrors onto the frame.
    // Such samples carry no handle and are no scrub target (NaN drops out of the search).
    const world = paths.arrays.ee_world?.data as Float32Array | undefined;
    if (world) {
      const p: [number, number, number] = [0, 0, 0];
      for (let i = 0; i < this.N * this.H; i++) {
        this.camera.project([world[3 * i], world[3 * i + 1], world[3 * i + 2]], p);
        const u = this.eePx[2 * i], v = this.eePx[2 * i + 1];
        if (!(p[2] > 0.08) || u < -960 || u > 2880 || v < -540 || v > 1620) {
          this.eePx[2 * i] = Number.NaN;
          this.eePx[2 * i + 1] = Number.NaN;
        }
      }
    }
    this.eePxHero = Float32Array.from(this.eePx);
    this.eeWorld = world ?? null;
    this.projVersion = this.stage.view.version;
    this.scrubber = new Scrubber(this.eePx, this.N, this.H, { fps: 30 });
    const track = (this.m.track ?? {}) as { show?: number[] };
    this.show = (track.show ?? [...Array(this.H).keys()]).filter((h) => h >= 0 && h < this.H);
    if (world) {
      // orbit about where the grippers work
      const c = [0, 0, 0];
      let n = 0;
      for (let i = 0; i < this.N; i++)
        for (const h of this.show) {
          const o = (i * this.H + h) * 3;
          if (!Number.isFinite(world[o])) continue;
          c[0] += world[o];
          c[1] += world[o + 1];
          c[2] += world[o + 2];
          n++;
        }
      if (n) this.stage.view.setFocus([c[0] / n, c[1] / n, c[2] / n]);
      this.projVersion = this.stage.view.version;
    }
    for (let k = 0; k < this.show.length; k++) {
      const d = document.createElement("div");
      d.className = "handle";
      stage.overlay.appendChild(d);
      this.handles.push(d);
    }
    this.ready = true;
  }

  pointerDown(e: PointerEvent, rtx: [number, number]): void {
    if (!this.ready) return;
    this.lastInput = performance.now();
    this.reproject();
    const f = this.currentFrame();
    let best = -1;
    let bestD = Infinity;
    for (const h of this.show) {
      const u = this.eePx[(f * this.H + h) * 2];
      if (Number.isNaN(u)) continue;
      const v = this.eePx[(f * this.H + h) * 2 + 1];
      const d = Math.hypot(rtx[0] - u, rtx[1] - v) * this.stage.placement().s;
      if (d < bestD) {
        bestD = d;
        best = h;
      }
    }
    const near = bestD <= (e.pointerType === "touch" ? 40 : 24);
    this.grab = { handle: near ? best : (this.show[0] ?? 0), relative: !near, lastPx: [rtx[0], rtx[1]] };
    this.t = f;
    this.enterScrub();
  }

  pointerMove(e: PointerEvent, rtx: [number, number]): void {
    if (!this.grab || !this.scrubber) return;
    this.lastInput = performance.now();
    if (this.grab.relative) {
      const dx = (rtx[0] - this.grab.lastPx[0]) * this.stage.placement().s;
      this.t = Math.min(this.N - 1, Math.max(0, this.t + this.scrubber.relative(dx, this.stage.placement().stageW, this.N - 1)));
    } else {
      this.t = this.scrubber.nearest(rtx, this.t, this.grab.handle);
    }
    this.grab.lastPx = [rtx[0], rtx[1]];
    this.scrubber.flick.push(e.timeStamp / 1000, this.t, 0);
  }

  pointerUp(e: PointerEvent): void {
    if (!this.grab || !this.scrubber) return;
    this.grab = null;
    this.lastInput = performance.now();
    if (!this.scrubber.flick.release(e.timeStamp / 1000)) this.snap();
  }

  /** A second finger turned the scrub into an orbit: stop on the frame reached, no coast. */
  pointerCancel(): void {
    if (!this.grab) return;
    this.grab = null;
    this.lastInput = performance.now();
    this.snap();
  }

  key(e: KeyboardEvent): void {
    if (e.code === "Space") {
      e.preventDefault();
      if (this.mode === "playing") this.pauseAt(this.currentFrame());
      else this.play();
    } else if (e.code === "ArrowRight" || e.code === "ArrowLeft") {
      const f = Math.min(this.N - 1, Math.max(0, Math.round(this.currentFrame()) + (e.code === "ArrowRight" ? 1 : -1)));
      this.pauseAt(f);
    }
  }

  /** Calibration: the live layer alone at rollout frame ``f`` (video and handles hidden). */
  showFrame(f: number): void {
    if (!this.ready) return;
    this.stage.video.pause();
    this.videoAlpha = 0;
    this.stage.video.style.opacity = "0";
    this.mode = "calib";
    this.t = f;
    for (const h of this.handles) h.style.visibility = "hidden";
    this.poseAt(f);
  }

  /** Calibration: retune the lights live. */
  setLook(look: Record<string, unknown>): void {
    this.stage.live.applyLook(look);
    this.stage.live.render();
  }

  private currentFrame(): number {
    if (this.mode === "playing") return Math.min(this.N - 1, Math.max(0, Math.floor(this.stage.video.currentTime * 30 + 1e-3)));
    return Math.round(this.t);
  }

  private enterScrub(): void {
    this.stage.video.pause();
    this.videoAlpha = 0;
    this.mode = "scrub";
    this.seekToken++;
  }

  private snap(): void {
    this.pauseAt(Math.round(this.t));
  }

  private pauseAt(f: number): void {
    this.t = f;
    this.mode = "paused";
    const token = ++this.seekToken;
    const v = this.stage.video;
    v.pause();
    const done = () => {
      if (token !== this.seekToken) return;
      this.videoAlpha = 1;
    };
    v.addEventListener("seeked", done, { once: true });
    v.currentTime = (f + 0.5) / 30;
  }

  private play(): void {
    const v = this.stage.video;
    v.currentTime = (Math.round(this.t) + 0.5) / 30;
    this.videoAlpha = 1;
    this.mode = "playing";
    void v.play().catch(() => undefined);
  }

  frame(_now: number, dt: number): void {
    if (!this.ready || this.mode === "calib") return;
    const v = this.stage.video;
    const cur = Number(v.style.opacity || "0");
    // hold when there: "want > cur, else fade down" dipped a full video to 0.89 every other
    // frame, a flicker that showed the plate (robot-less) through the playing RTX video
    if (this.videoAlpha > cur) v.style.opacity = String(Math.min(this.videoAlpha, cur + dt / 0.3));
    else if (this.videoAlpha < cur) v.style.opacity = String(Math.max(this.videoAlpha, cur - dt / 0.15));
    if (this.scrubber?.flick.active && !this.grab) {
      if (!this.scrubber.flick.coast(dt)) this.snap();
      else this.t = Math.min(this.N - 1, Math.max(0, this.scrubber.flick.x[0]));
    }
    // looking around is activity too: a paused frame stays paused while it is being looked at
    if (this.mode === "paused" && performance.now() - Math.max(this.lastInput, this.stage.view.lastInputAt) > RESUME_AFTER_S * 1000) this.play();
    const f = this.currentFrame();
    if (this.mode !== "playing") this.poseAt(this.t);
    // off the hero view the video is hidden: its frame, drawn live
    else if (this.stage.view.look > 0) this.poseAt(Math.min(this.N - 1, Math.max(0, v.currentTime * 30)));
    this.reproject();
    this.drawHandles(f);
  }

  /** The handles' pixels for the view being shown: the pipeline's own at the hero view, else
   * re-projected from the world, with the same rule for a hand beside or behind the camera. */
  private reproject(): void {
    const view = this.stage.view;
    if (view.version === this.projVersion) return;
    this.projVersion = view.version;
    const px = this.eePx;
    if (view.home) {
      px.set(this.eePxHero);
      return;
    }
    const w = this.eeWorld;
    if (!w) {
      px.fill(Number.NaN); // nothing to re-project: no handles off the hero view
      return;
    }
    const cam = this.camera;
    const p: [number, number, number] = [0, 0, 0];
    for (let i = 0; i < this.N * this.H; i++) {
      cam.project([w[3 * i], w[3 * i + 1], w[3 * i + 2]], p);
      const ok = p[2] > 0.08 && p[0] > -960 && p[0] < 2880 && p[1] > -540 && p[1] < 1620;
      px[2 * i] = ok ? p[0] : Number.NaN;
      px[2 * i + 1] = ok ? p[1] : Number.NaN;
    }
  }

  private poseAt(t: number): void {
    const i0 = Math.max(0, Math.min(this.N - 1, Math.floor(t)));
    const i1 = Math.min(this.N - 1, i0 + 1);
    const a = t - i0;
    const nb = this.bodies.length;
    for (let k = 0; k < nb; k++) {
      const o0 = (i0 * nb + k) * 7;
      const o1 = (i1 * nb + k) * 7;
      const P = this.poses;
      const p = [0, 1, 2].map((j) => P[o0 + j] * (1 - a) + P[o1 + j] * a);
      let q = [3, 4, 5, 6].map((j) => P[o0 + j]);
      const q1 = [3, 4, 5, 6].map((j) => P[o1 + j]);
      const dot = q[0] * q1[0] + q[1] * q1[1] + q[2] * q1[2] + q[3] * q1[3];
      const sgn = dot < 0 ? -1 : 1;
      q = q.map((x, j) => x * (1 - a) + sgn * q1[j] * a);
      const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
      this.stage.live.setBody(this.bodies[k], p, q.map((x) => x / n));
    }
    this.stage.live.render();
  }

  private drawHandles(f: number): void {
    const pl = this.stage.placement();
    this.show.forEach((h, k) => {
      const u = this.eePx[(f * this.H + h) * 2];
      const v = this.eePx[(f * this.H + h) * 2 + 1];
      const el = this.handles[k];
      if (Number.isNaN(u)) {
        el.style.visibility = "hidden";
        return;
      }
      el.style.visibility = "";
      const [x, y] = toStage(pl, u, v);
      el.style.transform = `translate(${x}px, ${y}px)`;
    });
  }

  resize(p: Placement): void {
    this.stage.live.resize(1920 * p.s, 1080 * p.s, window.devicePixelRatio || 1);
  }

  dispose(): void {
    this.disposed = true;
    this.stage.live.reset();
    for (const h of this.handles) h.remove();
    this.stage.video.pause();
    this.stage.video.removeAttribute("src");
    this.stage.video.load();
    this.stage.bgInk.style.display = "";
    this.stage.stillInk.style.display = "";
  }
}
