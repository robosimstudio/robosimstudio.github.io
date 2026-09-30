// Free-drag input: a visitor's finger in RTX pixels ↔ the demo's 2-D drag
// coordinates (a, b) from manifest.drag (CONTRACTS.md §5):
//
//   world = origin + a A + b B,   a ∈ [a_lo, a_hi], b ∈ [b_lo, b_hi]
//
// Pixels map back to (a, b) through the 2x2 Jacobian of the projection. Where
// the drag plane is seen nearly edge-on, a singular value of that Jacobian goes
// to zero and a few pixels would mean metres; every inverse here clamps the
// singular values to at least SIGMA_MIN_PX, so a pixel moves (a, b) by at most
// 1/SIGMA_MIN_PX metres.
//
// `Flick` turns the last 100 ms of a drag into a release velocity and coasts it.

import type { Camera } from "./camera.ts";
import type { V3 } from "./math.ts";

export interface DragJson {
  origin: number[];
  A: number[];
  B: number[];
  a: number[];
  b: number[];
  hover: number[];
  coast_tau: number;
  v_max: number;
}

/** Smallest singular value [px per unit of a or b, i.e. px/m] an inverse Jacobian may use. */
export const SIGMA_MIN_PX = 150;
/** The drag coast stops below this speed [m/s]. */
export const DRAG_V_STOP = 0.01;

export class DragMapping {
  readonly origin: V3;
  readonly A: V3;
  readonly B: V3;
  readonly aLo: number;
  readonly aHi: number;
  readonly bLo: number;
  readonly bHi: number;
  readonly hover: [number, number];
  readonly coastTau: number;
  readonly vMax: number;
  private readonly w: V3 = [0, 0, 0];
  private readonly px: V3 = [0, 0, 0];
  private readonly pa: V3 = [0, 0, 0];
  private readonly pb: V3 = [0, 0, 0];

  constructor(d: DragJson) {
    this.origin = [d.origin[0], d.origin[1], d.origin[2]];
    this.A = [d.A[0], d.A[1], d.A[2]];
    this.B = [d.B[0], d.B[1], d.B[2]];
    this.aLo = d.a[0];
    this.aHi = d.a[1];
    this.bLo = d.b[0];
    this.bHi = d.b[1];
    this.hover = [d.hover[0], d.hover[1]];
    this.coastTau = d.coast_tau;
    this.vMax = d.v_max;
  }

  /** The world point of (a, b); unclamped, so it stays differentiable at the box faces. */
  toWorld(a: number, b: number, out: V3 = [0, 0, 0]): V3 {
    const { origin: o, A, B } = this;
    out[0] = o[0] + a * A[0] + b * B[0];
    out[1] = o[1] + a * A[1] + b * B[1];
    out[2] = o[2] + a * A[2] + b * B[2];
    return out;
  }

  clamp(a: number, b: number): [number, number] {
    return [Math.min(this.aHi, Math.max(this.aLo, a)), Math.min(this.bHi, Math.max(this.bLo, b))];
  }

  /** ∂(u, v)/∂(a, b) at (a, b), analytic: row-major [du/da, du/db, dv/da, dv/db] in px per unit. */
  jacobian(a: number, b: number, camera: Camera): [number, number, number, number] {
    const p = camera.toCam(this.toWorld(a, b, this.w), this.px);
    const pa = camera.dirToCam(this.A, this.pa);
    const pb = camera.dirToCam(this.B, this.pb);
    const iz = 1 / p[2];
    const x = p[0] * iz;
    const y = p[1] * iz;
    // u = fx X/Z + cx  →  du/da = fx (X_a - (X/Z) Z_a) / Z, and likewise for v and b
    return [
      camera.fx * (pa[0] - x * pa[2]) * iz,
      camera.fx * (pb[0] - x * pb[2]) * iz,
      camera.fy * (pa[1] - y * pa[2]) * iz,
      camera.fy * (pb[1] - y * pb[2]) * iz,
    ];
  }

  /** (a, b) whose projection is ``targetPx``: 3 Gauss-Newton steps from ``init``, then clamped. */
  solveAbsolute(targetPx: ArrayLike<number>, init: ArrayLike<number>, camera: Camera): [number, number] {
    let a = init[0];
    let b = init[1];
    for (let it = 0; it < 3; it++) {
      const px = camera.project(this.toWorld(a, b, this.w), this.px);
      const ru = targetPx[0] - px[0];
      const rv = targetPx[1] - px[1];
      const J = this.jacobian(a, b, camera);
      const [da, db] = pinvSolve2(J[0], J[1], J[2], J[3], ru, rv, SIGMA_MIN_PX);
      a += da;
      b += db;
    }
    return this.clamp(a, b);
  }

  /** (a, b) moved by a finger displacement ``dpx`` [px] through the pseudo-inverse, scaled by ``gain``, clamped. */
  relative(dpx: ArrayLike<number>, a: number, b: number, camera: Camera, gain = 1): [number, number] {
    const J = this.jacobian(a, b, camera);
    const [da, db] = pinvSolve2(J[0], J[1], J[2], J[3], dpx[0], dpx[1], SIGMA_MIN_PX);
    return this.clamp(a + gain * da, b + gain * db);
  }
}

/**
 * ``V diag(1 / max(σ_i, sMin)) Uᵀ r`` for the 2x2 matrix [[m00, m01], [m10, m11]]:
 * the pseudo-inverse solve with its singular values clamped from below. Uses the
 * closed-form SVD M = Rot(φ) diag(sx, sy) Rot(θ) (Blinn, "Consider the lowly 2x2
 * matrix"), where sy carries the sign of det M.
 */
export function pinvSolve2(m00: number, m01: number, m10: number, m11: number, r0: number, r1: number, sMin: number): [number, number] {
  const E = (m00 + m11) / 2;
  const F = (m00 - m11) / 2;
  const G = (m10 + m01) / 2;
  const H = (m10 - m01) / 2;
  const Q = Math.hypot(E, H);
  const R = Math.hypot(F, G);
  const sx = Q + R;
  const sy = Q - R;
  const a1 = Math.atan2(G, F);
  const a2 = Math.atan2(H, E);
  const th = (a2 - a1) / 2;
  const ph = (a2 + a1) / 2;
  // z = diag(1/sx, 1/sy) Rot(φ)ᵀ r, clamped; then x = Rot(θ)ᵀ z
  const cp = Math.cos(ph);
  const sp = Math.sin(ph);
  const z0 = (cp * r0 + sp * r1) / Math.max(sx, sMin);
  const z1 = ((sy < 0 ? -1 : 1) * (-sp * r0 + cp * r1)) / Math.max(Math.abs(sy), sMin);
  const ct = Math.cos(th);
  const st = Math.sin(th);
  return [ct * z0 + st * z1, -st * z0 + ct * z1];
}

export interface FlickParams {
  /** Coast decay time constant [s]: v *= exp(-dt / tau). */
  tau: number;
  /** Release speed cap [units/s]. */
  vMax: number;
  /** The coast ends below this speed [units/s]. */
  vStop: number;
  /** The box the coast stays in: one or two dimensions. */
  lo: ArrayLike<number>;
  hi: ArrayLike<number>;
  /** Velocity fit window before release [s], default 0.1. */
  window?: number;
}

const FLICK_SAMPLES = 128;

/**
 * Release velocity and coast. During a drag, ``push`` every pointer sample; on
 * release, the velocity is the least-squares slope of the samples from the last
 * ``window`` seconds, capped at ``vMax``. ``coast(dt)`` then integrates the decaying
 * velocity exactly, zeroes any component pushing into a face of the box, and stops
 * once the speed drops below ``vStop``.
 */
export class Flick {
  readonly dims: number;
  /** Coast position and velocity (per dimension). */
  readonly x = new Float64Array(2);
  readonly v = new Float64Array(2);
  active = false;
  private readonly p: Required<FlickParams>;
  private readonly ts = new Float64Array(FLICK_SAMPLES);
  private readonly xs = new Float64Array(2 * FLICK_SAMPLES);
  private head = 0;
  private count = 0;

  constructor(p: FlickParams) {
    if (p.lo.length !== p.hi.length || p.lo.length < 1 || p.lo.length > 2) throw new Error("Flick: lo/hi must both have 1 or 2 entries");
    this.dims = p.lo.length;
    this.p = { ...p, window: p.window ?? 0.1 };
  }

  /** The drag's own coaster: manifest coast_tau and v_max, stops below 0.01 m/s. */
  static forDrag(drag: DragMapping): Flick {
    return new Flick({ tau: drag.coastTau, vMax: drag.vMax, vStop: DRAG_V_STOP, lo: [drag.aLo, drag.bLo], hi: [drag.aHi, drag.bHi] });
  }

  /** A pointer sample at time ``t`` [s]; a new drag stops any coast. */
  push(t: number, a: number, b = 0): void {
    const i = this.head;
    this.ts[i] = t;
    this.xs[2 * i] = a;
    this.xs[2 * i + 1] = b;
    this.head = (i + 1) % FLICK_SAMPLES;
    if (this.count < FLICK_SAMPLES) this.count++;
    this.active = false;
  }

  /** End the drag at time ``t`` [s]: fit the release velocity, start coasting from the last sample. */
  release(t: number): boolean {
    const { dims, ts, xs, x, v } = this;
    const from = t - this.p.window;
    let n = 0;
    let st = 0;
    let s0 = 0;
    let s1 = 0;
    for (let j = 0; j < this.count; j++) {
      const i = (this.head - 1 - j + FLICK_SAMPLES) % FLICK_SAMPLES;
      if (ts[i] < from) break;
      n++;
      st += ts[i];
      s0 += xs[2 * i];
      s1 += xs[2 * i + 1];
    }
    v.fill(0);
    if (this.count > 0) {
      const last = (this.head - 1 + FLICK_SAMPLES) % FLICK_SAMPLES;
      for (let d = 0; d < dims; d++) x[d] = Math.min(this.p.hi[d], Math.max(this.p.lo[d], xs[2 * last + d]));
    }
    if (n >= 2) {
      const tm = st / n;
      const m0 = s0 / n;
      const m1 = s1 / n;
      let stt = 0;
      let st0 = 0;
      let st1 = 0;
      for (let j = 0; j < n; j++) {
        const i = (this.head - 1 - j + FLICK_SAMPLES) % FLICK_SAMPLES;
        const dt = ts[i] - tm;
        stt += dt * dt;
        st0 += dt * (xs[2 * i] - m0);
        st1 += dt * (xs[2 * i + 1] - m1);
      }
      if (stt > 0) {
        v[0] = st0 / stt;
        if (dims > 1) v[1] = st1 / stt;
      }
    }
    const speed = Math.hypot(v[0], v[1]);
    if (speed > this.p.vMax) {
      v[0] *= this.p.vMax / speed;
      v[1] *= this.p.vMax / speed;
    }
    this.count = 0;
    this.active = Math.hypot(v[0], v[1]) >= this.p.vStop;
    if (!this.active) v.fill(0);
    return this.active;
  }

  /** Advance the coast by ``dt`` [s]; false once it has stopped. */
  coast(dt: number): boolean {
    if (!this.active) return false;
    const { dims, x, v } = this;
    const { tau, lo, hi } = this.p;
    const e = Math.exp(-dt / tau);
    const travel = tau * (1 - e); // ∫_0^dt e^{-s/tau} ds
    for (let d = 0; d < dims; d++) {
      let xd = x[d] + v[d] * travel;
      let vd = v[d] * e;
      if (xd <= lo[d]) {
        xd = lo[d];
        if (vd < 0) vd = 0;
      } else if (xd >= hi[d]) {
        xd = hi[d];
        if (vd > 0) vd = 0;
      }
      x[d] = xd;
      v[d] = vd;
    }
    if (Math.hypot(v[0], v[1]) < this.p.vStop) this.stop();
    return this.active;
  }

  stop(): void {
    this.active = false;
    this.v.fill(0);
  }
}
