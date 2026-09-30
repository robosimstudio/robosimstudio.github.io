// The visitor's own view: an orbit about a point on the hero camera's optical
// axis, which glides back to the hero pose after a few idle seconds.
//
// Only at the hero pose do the RTX layers line up with the scene, so only there
// are they shown. Anywhere else the live layer draws the whole scene; `look` is
// that cross-fade (0: RTX, 1: live), and it runs only at the hero pose on the
// way back, so the RTX frames return exactly aligned.

import { Camera, type CameraJson } from "./camera.ts";
import type { V3 } from "./math.ts";

export interface HeroCamera extends CameraJson {
  look_at?: number[];
}

export const VIEW = {
  /** No input this long [s]: glide back to the hero pose. */
  idleHomeS: 4,
  /** RTX out, live in, as the camera leaves the hero pose [s]. */
  lookInS: 0.15,
  /** Live out, RTX in, once it is back [s]. */
  lookOutS: 0.45,
  /** Orbit per CSS pixel dragged [rad]: a 360 px drag turns 80 degrees (0.006 felt loose). */
  radPerPx: 0.004,
  /** The eye's elevation over the pivot stays within these [rad]: above the floor, short of the pole. */
  elevMin: (4 * Math.PI) / 180,
  elevMax: (86 * Math.PI) / 180,
  /** Distance to the pivot, relative to the hero's (log): at 0.35 the wipe's camera sat inside an arm. */
  zoomMin: Math.log(0.55),
  zoomMax: Math.log(2.5),
};

type M3 = [number, number, number, number, number, number, number, number, number]; // row-major

export class View {
  /** About world +z through the pivot [rad]. */
  yaw = 0;
  /** Elevation added to the hero's [rad]. */
  pitch = 0;
  /** log of the distance to the pivot, relative to the hero's. */
  zoom = 0;
  /** 0: the RTX layers show; 1: the live layer draws everything. */
  look = 0;
  /** Bumped whenever the pose changes. */
  version = 0;
  /** performance.now() of the last input of any kind (orbit, zoom, a drag). */
  lastInputAt = Number.NEGATIVE_INFINITY;
  private hero: HeroCamera | null = null;
  private heroM = new Float64Array(16);
  private pivot: V3 = [0, 0, 0];
  private d0: V3 = [0, 0, 1]; // eye - pivot at the hero pose
  private right0: V3 = [1, 0, 0]; // the pitch axis: horizontal, across the view
  private elev0 = 0;
  private elevSign = 1; // +1 when turning about right0 by +a raises the eye
  private glide: { from: [number, number, number]; t: number; dur: number } | null = null;
  private cam: Camera | null = null;
  private camVersion = -1;
  private m = new Float64Array(16);

  /** A new demo: its hero camera, at home, RTX showing. */
  setHero(c: HeroCamera): void {
    this.hero = c;
    this.heroM = Float64Array.from(c.cam_to_world_cv);
    const h = this.heroM;
    const eye: V3 = [h[12], h[13], h[14]];
    const fwd: V3 = [h[8], h[9], h[10]];
    const la = c.look_at;
    const along = la && la.length === 3 ? (la[0] - eye[0]) * fwd[0] + (la[1] - eye[1]) * fwd[1] + (la[2] - eye[2]) * fwd[2] : 1;
    this.setPivotAt(Math.max(0.2, along));
    this.yaw = 0;
    this.pitch = 0;
    this.zoom = 0;
    this.look = 0;
    this.glide = null;
    this.version++;
  }

  /** Orbit about the point of the optical axis nearest ``p`` (where the action is). Home only. */
  setFocus(p: ArrayLike<number>): void {
    if (!this.hero || !this.home) return;
    const h = this.heroM;
    const along = (p[0] - h[12]) * h[8] + (p[1] - h[13]) * h[9] + (p[2] - h[14]) * h[10];
    if (Number.isFinite(along)) this.setPivotAt(Math.max(0.2, along));
    this.version++;
  }

  private setPivotAt(along: number): void {
    const h = this.heroM;
    this.pivot = [h[12] + h[8] * along, h[13] + h[9] * along, h[14] + h[10] * along];
    this.d0 = [h[12] - this.pivot[0], h[13] - this.pivot[1], h[14] - this.pivot[2]];
    const r = Math.hypot(this.d0[0], this.d0[1], this.d0[2]);
    this.elev0 = Math.asin(clamp(this.d0[2] / r, -1, 1));
    // levelled: the hero's right with any roll taken out (straight down, fall back to its x)
    const rn = Math.hypot(h[0], h[1]);
    this.right0 = rn > 1e-6 ? [h[0] / rn, h[1] / rn, 0] : [1, 0, 0];
    const cz = this.right0[0] * this.d0[1] - this.right0[1] * this.d0[0]; // (right0 x d0).z
    this.elevSign = cz >= 0 ? 1 : -1;
  }

  get home(): boolean {
    return this.yaw === 0 && this.pitch === 0 && this.zoom === 0;
  }

  /** Any input: holds the view where it is (and stops a glide). */
  touch(now = performance.now()): void {
    this.lastInputAt = now;
    this.glide = null;
  }

  /** A drag of (dx, dy) CSS pixels: right turns the scene with the finger, down raises the eye. */
  orbit(dx: number, dy: number, now = performance.now()): void {
    if (!this.hero) return;
    this.touch(now);
    this.yaw -= dx * VIEW.radPerPx;
    const lo = Math.min(0, VIEW.elevMin - this.elev0);
    const hi = Math.max(0, VIEW.elevMax - this.elev0);
    this.pitch = clamp(this.pitch + dy * VIEW.radPerPx, lo, hi);
    this.version++;
  }

  /** Scale the distance to the pivot by ``f`` (> 1 backs off). */
  dolly(f: number, now = performance.now()): void {
    if (!this.hero || !(f > 0)) return;
    this.touch(now);
    this.zoom = clamp(this.zoom + Math.log(f), VIEW.zoomMin, VIEW.zoomMax);
    if (Math.abs(this.zoom) < 1e-9) this.zoom = 0;
    this.version++;
  }

  /** Start back now (Escape). */
  goHome(): void {
    if (!this.home && !this.glide) this.startGlide();
  }

  private startGlide(): void {
    this.yaw = wrapPi(this.yaw); // the short way round
    const span = Math.max(Math.abs(this.yaw), Math.abs(this.pitch), Math.abs(this.zoom));
    this.glide = { from: [this.yaw, this.pitch, this.zoom], t: 0, dur: 0.6 + 0.35 * span };
  }

  /** Advance the glide and the cross-fade; true when the pose moved. */
  update(now: number, dt: number): boolean {
    let moved = false;
    if (!this.home && !this.glide && (now - this.lastInputAt) / 1000 >= VIEW.idleHomeS) this.startGlide();
    const g = this.glide;
    if (g) {
      g.t = Math.min(1, g.t + dt / g.dur);
      const k = 1 - smootherstep(g.t);
      if (g.t >= 1) {
        this.yaw = 0;
        this.pitch = 0;
        this.zoom = 0;
        this.glide = null;
      } else {
        this.yaw = g.from[0] * k;
        this.pitch = g.from[1] * k;
        this.zoom = g.from[2] * k;
      }
      this.version++;
      moved = true;
    }
    const want = this.home ? 0 : 1;
    if (this.look < want) this.look = Math.min(want, this.look + dt / VIEW.lookInS);
    else if (this.look > want) this.look = Math.max(want, this.look - dt / VIEW.lookOutS);
    return moved;
  }

  /** CV camera-to-world of the current pose, column-major (the hero's, exactly, at home). */
  camToWorld(): Float64Array {
    this.refresh();
    return this.m;
  }

  /** The current pose as a projection camera (hero intrinsics). */
  get camera(): Camera {
    this.refresh();
    return this.cam!;
  }

  private refresh(): void {
    if (this.cam && this.camVersion === this.version) return;
    if (!this.hero) throw new Error("View: no hero camera");
    this.compose();
    this.cam = new Camera({ ...this.hero, cam_to_world_cv: Array.from(this.m) });
    this.camVersion = this.version;
  }

  private compose(): void {
    const m = this.m;
    const h = this.heroM;
    if (this.home) {
      m.set(h);
      return;
    }
    // R = Rz(yaw) R(right0, a): pitch about the levelled right axis, then yaw about +z, both
    // through the pivot; the axes turn with it, the eye's distance scales by exp(zoom)
    const R = mul3(axisAngle([0, 0, 1], this.yaw), axisAngle(this.right0, this.pitch * this.elevSign));
    for (let c = 0; c < 3; c++) {
      const w = apply3(R, [h[4 * c], h[4 * c + 1], h[4 * c + 2]]);
      m[4 * c] = w[0];
      m[4 * c + 1] = w[1];
      m[4 * c + 2] = w[2];
      m[4 * c + 3] = 0;
    }
    const s = Math.exp(this.zoom);
    const d = apply3(R, this.d0);
    m[12] = this.pivot[0] + s * d[0];
    m[13] = this.pivot[1] + s * d[1];
    m[14] = this.pivot[2] + s * d[2];
    m[15] = 1;
  }
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

function wrapPi(a: number): number {
  const t = (a + Math.PI) % (2 * Math.PI);
  return (t < 0 ? t + 2 * Math.PI : t) - Math.PI;
}

function smootherstep(t: number): number {
  return t * t * t * (t * (6 * t - 15) + 10);
}

function axisAngle(a: V3, ang: number): M3 {
  const c = Math.cos(ang);
  const s = Math.sin(ang);
  const t = 1 - c;
  const [x, y, z] = a;
  return [
    t * x * x + c, t * x * y - s * z, t * x * z + s * y,
    t * x * y + s * z, t * y * y + c, t * y * z - s * x,
    t * x * z - s * y, t * y * z + s * x, t * z * z + c,
  ];
}

function mul3(a: M3, b: M3): M3 {
  const o = new Array(9).fill(0) as M3;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) o[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
  return o;
}

function apply3(R: M3, v: ArrayLike<number>): V3 {
  return [
    R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
    R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
    R[6] * v[0] + R[7] * v[1] + R[8] * v[2],
  ];
}
