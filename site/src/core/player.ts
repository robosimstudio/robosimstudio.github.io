// A playhead over the frame store. `phi` is a fractional frame index that always
// lies inside one take: it is clamped to the take's last frame instead of running
// into the next take, and sampling interpolates only between frames of that take.

import type { FloatChannel, FrameStore } from "./store.ts";

export class StatePlayer {
  readonly store: FrameStore;
  /** Fractional frame index, ``first <= phi <= last``. */
  phi = 0;
  /** Index into ``store.takes`` of the take being played. */
  take = 0;
  /** The take's first and last frame. */
  first = 0;
  last = 0;
  /** True once ``phi`` has reached the take's last frame. */
  atEnd = false;
  private readonly crossed: number[] = [];

  constructor(store: FrameStore, frame = 0) {
    this.store = store;
    this.set(frame);
  }

  /** Jump to frame ``i`` (fractional allowed), inside the take holding ``floor(i)``. */
  set(i: number): void {
    const k = this.store.takeOf(i);
    if (k < 0) throw new RangeError(`StatePlayer: frame ${i} is outside the store (0..${this.store.n - 1})`);
    const t = this.store.takes[k];
    this.take = k;
    this.first = t.start;
    this.last = t.start + t.len - 1;
    this.phi = Math.min(i, this.last);
    this.atEnd = this.phi >= this.last;
  }

  /**
   * Move forward ``dFrames`` (≥ 0), stopping at the take's last frame. Returns every
   * integer frame k with phi_old < k <= phi_new, in order, so per-frame effects (ink
   * removal, peel keys) see each frame once. The array is reused by the next call.
   */
  advance(dFrames: number): number[] {
    if (!(dFrames >= 0)) throw new RangeError(`StatePlayer.advance: dFrames must be >= 0, got ${dFrames}`);
    const out = this.crossed;
    out.length = 0;
    const to = Math.min(this.phi + dFrames, this.last);
    for (let k = Math.floor(this.phi) + 1; k <= to; k++) out.push(k);
    this.phi = to;
    this.atEnd = to >= this.last;
    return out;
  }

  /** The frame whose integer channels (still, broken, front) describe the playhead. */
  get frame(): number {
    return Math.floor(this.phi);
  }

  /** Frames left before the take ends. */
  runway(): number {
    return this.last - this.phi;
  }

  /**
   * ``channel`` at the playhead: linear between floor(phi) and the next frame. The next
   * frame is only read when phi is strictly before ``last``, so it is always in the take.
   */
  sample<T extends { [i: number]: number }>(channel: FloatChannel, out: T): T {
    const data = this.store.float(channel);
    const dim = this.store.dims[channel] as number;
    const i0 = Math.floor(this.phi);
    const f = this.phi - i0;
    const a = i0 * dim;
    if (f === 0) {
      for (let d = 0; d < dim; d++) out[d] = data[a + d];
    } else {
      for (let d = 0; d < dim; d++) out[d] = data[a + d] + f * (data[a + dim + d] - data[a + d]);
    }
    return out;
  }
}
