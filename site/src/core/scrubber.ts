// Track demos (CONTRACTS.md §6): the visitor scrubs a pre-rendered rollout by
// dragging a handle along its path. `ee_px` [N, H, 2] holds each handle's RTX
// pixel per frame; the path between frames is the straight segment.
//
// `nearest` picks the fractional frame t* minimizing
//
//   |px - p(t*)|² + λ² Δt²,   Δt = (t* - t) / fps  [s]
//
// over the segments within ±window frames of the current t. The time term keeps
// the answer on the stretch of path being dragged, so where a path crosses itself
// (a figure-8) the playhead passes the crossing instead of jumping to the other
// visit. On each segment the cost is a quadratic in the fraction α ∈ [0, 1], so
// its minimum is closed-form.

import { Flick } from "./drag.ts";

/** Release coast of a scrub [s]. */
export const SCRUB_COAST_TAU = 0.35;

export interface ScrubberOptions {
  /** Video frame rate, default 30. */
  fps?: number;
  /** Release speed cap [frames/s], default unlimited (the clip ends clamp it). */
  vMax?: number;
  /** The coast stops below this speed [frames/s], default 0.5. */
  vStop?: number;
}

export class Scrubber {
  readonly eePx: Float32Array;
  /** Frames and handles. */
  readonly n: number;
  readonly h: number;
  readonly fps: number;
  /** Release coast of the playhead, in frames (tau 0.35 s, kept inside [0, n-1]). */
  readonly flick: Flick;

  constructor(eePx: Float32Array, n: number, h: number, opts: ScrubberOptions = {}) {
    if (eePx.length !== n * h * 2) throw new Error(`Scrubber: ee_px holds ${eePx.length} values, [${n}, ${h}, 2] needs ${n * h * 2}`);
    this.eePx = eePx;
    this.n = n;
    this.h = h;
    this.fps = opts.fps ?? 30;
    this.flick = new Flick({
      tau: SCRUB_COAST_TAU,
      vMax: opts.vMax ?? Number.POSITIVE_INFINITY,
      vStop: opts.vStop ?? 0.5,
      lo: [0],
      hi: [Math.max(0, n - 1)],
    });
  }

  /** The fractional frame whose ``handle`` position best explains the finger at ``px``, near frame ``t``. */
  nearest(px: ArrayLike<number>, t: number, handle: number, window = 60, lambdaPxPerS = 40, fps = this.fps): number {
    const { eePx, h } = this;
    if (!(handle >= 0 && handle < h)) throw new RangeError(`Scrubber: handle ${handle} is not in 0..${h - 1}`);
    const c = lambdaPxPerS / fps; // px per frame of offset
    const c2 = c * c;
    const x = px[0];
    const y = px[1];
    const k0 = Math.max(0, Math.floor(t - window));
    const k1 = Math.min(this.n - 2, Math.ceil(t + window) - 1);
    let best = Number.POSITIVE_INFINITY;
    let bestT = Math.min(Math.max(t, 0), this.n - 1);
    for (let k = k0; k <= k1; k++) {
      const i0 = (k * h + handle) * 2;
      const i1 = i0 + 2 * h;
      const rx = x - eePx[i0];
      const ry = y - eePx[i0 + 1];
      const ex = eePx[i1] - eePx[i0];
      const ey = eePx[i1 + 1] - eePx[i0 + 1];
      const dk = k - t;
      // d/dα [ |r - α e|² + c² (dk + α)² ] = 0
      let al = (rx * ex + ry * ey - c2 * dk) / (ex * ex + ey * ey + c2);
      al = al < 0 ? 0 : al > 1 ? 1 : al;
      const dx = rx - al * ex;
      const dy = ry - al * ey;
      const dt = dk + al;
      const f = dx * dx + dy * dy + c2 * dt * dt;
      if (f < best) {
        best = f;
        bestT = k + al;
      }
    }
    return bestT;
  }

  /** Frames to move for a drag of ``dxPx`` on an empty stage ``visW`` px wide spanning ``T`` frames. */
  relative(dxPx: number, visW: number, T = this.n - 1): number {
    return (dxPx / visW) * T;
  }
}
