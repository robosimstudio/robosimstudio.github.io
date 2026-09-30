// Motion-matching search over the frame store's features.
//
// The DB stores standardized features; the query-time weight of each group
// (manifest `features.groups`, in column order) is folded in once:
// W[i, d] = feat[i, d] * sqrt(w_g), so a search minimizes the plain squared
// distance ||W[i] - q||² to a query already in that weighted space (`weigh`).
//
// Exactness: the result equals a brute-force scan over all admissible frames,
// ties going to the lowest frame index. Frames are grouped in blocks of BLOCK
// consecutive frames that never cross a take; each block keeps the per-dimension
// min/max of its rows, so Σ_d gap(q_d, [lo_d, hi_d])² bounds every frame in it
// from below, term by term. Floating-point rounding is monotone, so the bound
// holds for the computed sums too, and a block (or a frame, once its partial
// sum over whole feature groups) that already exceeds the best cost is skipped
// without changing the answer. The block with the lowest bound is scanned first
// so the best cost is tight before the rest are visited.

import type { FrameStore } from "./store.ts";

export interface FeatureGroup {
  name: string;
  dims: number;
  w: number;
}

/** Peel keys never go backwards: a candidate must reach at least the current state's. */
export interface Monotone {
  /** Bonds broken per frame, [N] (store.broken). */
  broken: Uint16Array;
  /** Peel front per frame and column, [N, C] (store.front), or null. */
  front: Uint8Array | null;
  /** The keys of the state being played. */
  cur: { broken: number; front: ArrayLike<number> | null };
}

export interface SearchOptions {
  /** The frame being played: frames within ±sameTakeWindow of it in its take are excluded. */
  current?: number;
  /** Exclude frames with fewer frames than this left in their take. Default 10. */
  minRunway?: number;
  /** Default 15. */
  sameTakeWindow?: number;
  /** Exclude a take's first ``startMargin`` frames. Default 30. */
  startMargin?: number;
  monotone?: Monotone;
  /** Extra cost per frame (settle mode). Must be >= 0, which keeps the lower bounds valid. */
  bias?: (i: number) => number;
}

export interface Match {
  /** Best admissible frame, or -1 when none is admissible. */
  frame: number;
  cost: number;
}

export const BLOCK = 16;

export class Matcher {
  readonly store: FrameStore;
  readonly D: number;
  /** sqrt(w_g) per feature column. */
  readonly scale: Float64Array;
  /** Weighted features [n, D], rebuilt by ``sync``. */
  W = new Float32Array(0);
  /** Frames indexed (store.n at the last sync). */
  n = 0;
  /** Group g covers columns [ends[g-1], ends[g]); the early exit is checked at each end. */
  private readonly ends: Int32Array;
  private readonly qd: Float64Array;
  private nb = 0;
  private bStart = new Int32Array(0);
  private bEnd = new Int32Array(0);
  private bTake = new Int32Array(0);
  private bTakeStart = new Int32Array(0);
  private bTakeLast = new Int32Array(0);
  private lo = new Float32Array(0);
  private hi = new Float32Array(0);
  private lb = new Float64Array(0);
  private ok = new Uint8Array(0);
  private brokenSrc: Uint16Array | null = null;
  private brokenMax = new Uint16Array(0);

  constructor(store: FrameStore, groups: FeatureGroup[]) {
    this.store = store;
    this.D = groups.reduce((s, g) => s + g.dims, 0);
    this.scale = new Float64Array(this.D);
    this.ends = new Int32Array(groups.length);
    let d = 0;
    groups.forEach((g, k) => {
      if (!(g.w >= 0)) throw new Error(`Matcher: group "${g.name}" has weight ${g.w}`);
      this.scale.fill(Math.sqrt(g.w), d, d + g.dims);
      d += g.dims;
      this.ends[k] = d;
    });
    this.qd = new Float64Array(this.D);
    this.sync();
  }

  /** New group weights (one per group, in order): rescales every column and rebuilds W. */
  setWeights(ws: number[]): void {
    if (ws.length !== this.ends.length) throw new Error(`Matcher: ${ws.length} weights for ${this.ends.length} groups`);
    let d = 0;
    ws.forEach((w, k) => {
      if (!(w >= 0)) throw new Error(`Matcher: weight ${w} for group ${k}`);
      this.scale.fill(Math.sqrt(w), d, this.ends[k]);
      d = this.ends[k];
    });
    this.sync();
  }

  /** Rebuild W and the blocks from the store (after ``store.append``; ``search`` does it lazily). */
  sync(): void {
    const s = this.store;
    const D = this.D;
    if (s.dims.feat !== D) throw new Error(`Matcher: feature groups span ${D} columns, the store's feat rows are ${s.dims.feat}`);
    const n = s.n;
    const feat = s.feat;
    const W = new Float32Array(n * D);
    for (let i = 0; i < n; i++) for (let d = 0, o = i * D; d < D; d++) W[o + d] = feat[o + d] * this.scale[d];
    let nb = 0;
    for (const t of s.takes) nb += Math.ceil(t.len / BLOCK);
    this.bStart = new Int32Array(nb);
    this.bEnd = new Int32Array(nb);
    this.bTake = new Int32Array(nb);
    this.bTakeStart = new Int32Array(nb);
    this.bTakeLast = new Int32Array(nb);
    this.lo = new Float32Array(nb * D);
    this.hi = new Float32Array(nb * D);
    let b = 0;
    s.takes.forEach((t, k) => {
      const end = t.start + t.len;
      for (let i0 = t.start; i0 < end; i0 += BLOCK, b++) {
        const i1 = Math.min(i0 + BLOCK, end);
        this.bStart[b] = i0;
        this.bEnd[b] = i1;
        this.bTake[b] = k;
        this.bTakeStart[b] = t.start;
        this.bTakeLast[b] = end - 1;
        for (let d = 0; d < D; d++) {
          let l = Number.POSITIVE_INFINITY;
          let h = Number.NEGATIVE_INFINITY;
          for (let i = i0; i < i1; i++) {
            const x = W[i * D + d];
            if (x < l) l = x;
            if (x > h) h = x;
          }
          this.lo[b * D + d] = l;
          this.hi[b * D + d] = h;
        }
      }
    });
    this.W = W;
    this.n = n;
    this.nb = nb;
    this.lb = new Float64Array(nb);
    this.ok = new Uint8Array(nb);
    this.brokenSrc = null;
  }

  /** Standardized features → the weighted query space (``out`` may be ``std``). */
  weigh<T extends { [i: number]: number }>(std: ArrayLike<number>, out: T): T {
    for (let d = 0; d < this.D; d++) out[d] = std[d] * this.scale[d];
    return out;
  }

  /** Frame ``i``'s weighted feature row (a view). */
  row(i: number): Float32Array {
    return this.W.subarray(i * this.D, (i + 1) * this.D);
  }

  /** ||W[i] - q||² for one frame, no exclusions or bias. */
  costAt(i: number, q: ArrayLike<number>): number {
    if (this.n !== this.store.n) this.sync();
    const W = this.W;
    const o = i * this.D;
    let s = 0;
    for (let d = 0; d < this.D; d++) {
      const t = W[o + d] - q[d];
      s += t * t;
    }
    return s;
  }

  /** The admissible frame minimizing ||W[i] - q||² (+ bias(i)). */
  search(query: ArrayLike<number>, opts: SearchOptions = {}): Match {
    if (this.n !== this.store.n) this.sync();
    const D = this.D;
    const q = this.qd;
    for (let d = 0; d < D; d++) q[d] = query[d];
    const minRunway = opts.minRunway ?? 10;
    const win = opts.sameTakeWindow ?? 15;
    const margin = opts.startMargin ?? 30;
    const cur = opts.current ?? Number.NaN;
    const curTake = Number.isNaN(cur) ? -1 : this.store.takeOf(cur);
    const mono = opts.monotone;
    const blockBroken = mono ? this.blockMaxBroken(mono.broken) : null;
    const minBroken = mono ? mono.cur.broken : 0;
    const broken = mono ? mono.broken : null;
    const front = mono && mono.front && mono.cur.front ? mono.front : null;
    const curFront = mono && mono.cur.front ? mono.cur.front : null;
    const C = curFront ? curFront.length : 0;
    const bias = opts.bias;
    const { W, lo, hi, lb, ok, ends, bStart, bEnd, bTake, bTakeStart, bTakeLast, nb } = this;
    const G = ends.length;

    // Pass 1: each block's admissible range and lower bound.
    let first = -1;
    let firstLb = Number.POSITIVE_INFINITY;
    for (let b = 0; b < nb; b++) {
      const i0 = Math.max(bStart[b], bTakeStart[b] + margin);
      const i1 = Math.min(bEnd[b] - 1, bTakeLast[b] - minRunway);
      if (
        i0 > i1 ||
        (blockBroken !== null && blockBroken[b] < minBroken) ||
        (bTake[b] === curTake && cur - win <= i0 && i1 <= cur + win)
      ) {
        ok[b] = 0;
        continue;
      }
      ok[b] = 1;
      let s = 0;
      for (let d = 0, o = b * D; d < D; d++) {
        const x = q[d];
        const l = lo[o + d];
        if (x < l) s += (l - x) * (l - x);
        else {
          const h = hi[o + d];
          if (x > h) s += (x - h) * (x - h);
        }
      }
      lb[b] = s;
      if (s < firstLb) {
        firstLb = s;
        first = b;
      }
    }

    // Pass 2: the lowest-bound block first, then every block whose bound can still win.
    let best = Number.POSITIVE_INFINITY;
    let bestFrame = -1;
    for (let k = -1; k < nb; k++) {
      const b = k < 0 ? first : k;
      if (b < 0 || (k >= 0 && b === first) || !ok[b] || lb[b] > best) continue;
      const i0 = Math.max(bStart[b], bTakeStart[b] + margin);
      const i1 = Math.min(bEnd[b] - 1, bTakeLast[b] - minRunway);
      const inTake = bTake[b] === curTake;
      for (let i = i0; i <= i1; i++) {
        if (inTake && i >= cur - win && i <= cur + win) continue;
        if (broken !== null) {
          if (broken[i] < minBroken) continue;
          if (front !== null && curFront !== null) {
            let c = 0;
            while (c < C && front[i * C + c] >= curFront[c]) c++;
            if (c < C) continue;
          }
        }
        const o = i * D;
        let s = 0;
        let d = 0;
        let g = 0;
        for (; g < G; g++) {
          for (const e = ends[g]; d < e; d++) {
            const t = W[o + d] - q[d];
            s += t * t;
          }
          if (s > best) break;
        }
        if (g < G) continue;
        if (bias) s += bias(i);
        if (s < best || (s === best && i < bestFrame)) {
          best = s;
          bestFrame = i;
        }
      }
    }
    return { frame: bestFrame, cost: best };
  }

  /** Per-block max of ``broken``, cached for the array it was built from. */
  private blockMaxBroken(broken: Uint16Array): Uint16Array {
    if (broken !== this.brokenSrc) {
      const m = new Uint16Array(this.nb);
      for (let b = 0; b < this.nb; b++) {
        let x = 0;
        for (let i = this.bStart[b]; i < this.bEnd[b]; i++) if (broken[i] > x) x = broken[i];
        m[b] = x;
      }
      this.brokenMax = m;
      this.brokenSrc = broken;
    }
    return this.brokenMax;
  }
}
