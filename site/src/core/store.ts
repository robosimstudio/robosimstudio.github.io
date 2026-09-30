// The free-drag database (CONTRACTS.md §3) held in memory: N frames of named
// channels, one row per frame, plus the takes that partition the frames.
//
// Chunk 0 alone is enough to start; later chunks append. Frame indices are global:
// a chunk's `meta.frame0` must equal the frames already held (append in order), and
// its `takes` rows (start, len, kind) carry global starts that tile the chunk's
// frames exactly. A take never spans chunks, and frames never interpolate or
// continue across a take.

import type { NumArray, Rssb } from "./rssb.ts";

export interface Take {
  /** Global index of the take's first frame. */
  start: number;
  len: number;
  /** TAKE_VISITOR or TAKE_ATTRACT. */
  kind: number;
}

export const TAKE_VISITOR = 0;
export const TAKE_ATTRACT = 1;

/** Channels decoded to Float32Array rows (q16d, raw f32 and f16 all land here). */
export const FLOAT_CHANNELS = ["q", "hand", "setpoint", "cursor", "deform", "feat"] as const;
export type FloatChannel = (typeof FLOAT_CHANNELS)[number];
const REQUIRED: readonly FloatChannel[] = ["q", "hand", "setpoint", "cursor", "feat"];
const FIXED_DIM: Partial<Record<FloatChannel, number>> = { hand: 3, setpoint: 3, cursor: 2 };

/** Demo-specific integer channels and their contract dtypes. */
const INT_CHANNELS = { ink: Uint8Array, broken: Uint16Array, front: Uint8Array, still: Int32Array } as const;
type IntChannel = keyof typeof INT_CHANNELS;
export type Channel = FloatChannel | IntChannel;

const EMPTY = new Float32Array(0);
const product = (s: number[]): number => s.reduce((a, b) => a * b, 1);

export class FrameStore {
  /** Frames held: global indices 0..n-1. */
  n = 0;
  /** Sorted by start; they tile 0..n-1. */
  readonly takes: Take[] = [];
  /** Row width of each channel present (``dims.q`` = nq, ``dims.feat`` = D, ...). */
  readonly dims: Partial<Record<Channel, number>> = {};
  q: Float32Array = EMPTY;
  hand: Float32Array = EMPTY;
  setpoint: Float32Array = EMPTY;
  cursor: Float32Array = EMPTY;
  feat: Float32Array = EMPTY;
  deform: Float32Array | null = null;
  ink: Uint8Array | null = null;
  broken: Uint16Array | null = null;
  front: Uint8Array | null = null;
  still: Int32Array | null = null;
  /** Wipe takeover: ink intensity × 255 for every attract frame, [attract frames, cells]. */
  inkState: Uint8Array | null = null;
  /** Per take, the ``inkState`` row of its first frame, or -1. */
  private readonly stateRow: number[] = [];

  constructor(chunk0: Rssb) {
    this.append(chunk0);
  }

  /** Add the next chunk. Everything is validated before anything changes. */
  append(chunk: Rssb): void {
    const a = chunk.arrays;
    const first = this.takes.length === 0;
    const frame0 = chunk.meta?.frame0 ?? 0;
    const fail: (msg: string) => never = (msg) => {
      throw new Error(`FrameStore chunk @${frame0}: ${msg}`);
    };
    if (frame0 !== this.n) fail(`starts at frame ${frame0} but the store holds ${this.n} frames (append chunks in order)`);
    if (!a.q) fail('has no "q"');
    const n = a.q.shape[0];

    const rows = (name: Channel, want: new (n: number) => NumArray): number | null => {
      const arr = a[name];
      const had = first ? undefined : this.dims[name] !== undefined;
      if (!arr) {
        if ((REQUIRED as readonly string[]).includes(name)) fail(`has no "${name}"`);
        if (had) fail(`lacks "${name}", which chunk 0 had`);
        return null;
      }
      if (had === false) fail(`has "${name}", which chunk 0 lacked`);
      if (!(arr.data instanceof want)) fail(`"${name}" decodes to ${arr.data.constructor.name}, expected ${want.name}`);
      if (arr.shape[0] !== n) fail(`"${name}" has ${arr.shape[0]} rows, "q" has ${n}`);
      const dim = product(arr.shape.slice(1));
      const fixed = FIXED_DIM[name as FloatChannel];
      if (fixed !== undefined && dim !== fixed) fail(`"${name}" rows are ${dim} wide, expected ${fixed}`);
      if (!first && dim !== this.dims[name]) fail(`"${name}" rows are ${dim} wide, chunk 0's were ${this.dims[name]}`);
      return dim;
    };
    const dims: Partial<Record<Channel, number>> = {};
    for (const name of FLOAT_CHANNELS) {
      const d = rows(name, Float32Array);
      if (d !== null) dims[name] = d;
    }
    for (const name of Object.keys(INT_CHANNELS) as IntChannel[]) {
      const d = rows(name, INT_CHANNELS[name]);
      if (d !== null) dims[name] = d;
    }

    const tk = a.takes;
    if (!tk || !(tk.data instanceof Uint32Array) || tk.shape.length !== 2 || tk.shape[1] !== 3)
      fail('"takes" must be raw u32 [T, 3]');
    const takes: Take[] = [];
    for (let t = 0; t < tk.shape[0]; t++) takes.push({ start: tk.data[3 * t], len: tk.data[3 * t + 1], kind: tk.data[3 * t + 2] });
    takes.sort((x, y) => x.start - y.start);
    let at = frame0;
    for (const t of takes) {
      if (t.start !== at || t.len < 1) fail(`take (${t.start}, ${t.len}) does not continue at frame ${at}: takes must tile the chunk`);
      at += t.len;
    }
    if (at !== frame0 + n) fail(`takes cover frames ${frame0}..${at - 1}, the chunk holds ${frame0}..${frame0 + n - 1}`);

    // ink_state rows follow the chunk's attract frames in order
    const attract = takes.reduce((s, t) => s + (t.kind === TAKE_ATTRACT ? t.len : 0), 0);
    const st = a.ink_state;
    if (st) {
      if (!(st.data instanceof Uint8Array)) fail('"ink_state" must be raw u8');
      if (st.shape[0] !== attract) fail(`"ink_state" has ${st.shape[0]} rows for ${attract} attract frames`);
      if (product(st.shape.slice(1)) !== (dims.ink ?? -1)) fail('"ink_state" rows must be as wide as "ink" rows');
    } else if (dims.ink !== undefined && attract > 0) fail(`has ${attract} attract frames and "ink" but no "ink_state"`);

    // commit
    let row = this.inkState ? this.inkState.length / (dims.ink as number) : 0;
    for (const t of takes) {
      this.takes.push(t);
      this.stateRow.push(st && t.kind === TAKE_ATTRACT ? row : -1);
      if (st && t.kind === TAKE_ATTRACT) row += t.len;
    }
    Object.assign(this.dims, dims);
    this.q = concat(this.q, a.q.data as Float32Array);
    this.hand = concat(this.hand, a.hand.data as Float32Array);
    this.setpoint = concat(this.setpoint, a.setpoint.data as Float32Array);
    this.cursor = concat(this.cursor, a.cursor.data as Float32Array);
    this.feat = concat(this.feat, a.feat.data as Float32Array);
    if (a.deform) this.deform = concat(this.deform, a.deform.data as Float32Array);
    if (a.ink) this.ink = concat(this.ink, a.ink.data as Uint8Array);
    if (a.broken) this.broken = concat(this.broken, a.broken.data as Uint16Array);
    if (a.front) this.front = concat(this.front, a.front.data as Uint8Array);
    if (a.still) this.still = concat(this.still, a.still.data as Int32Array);
    if (st) this.inkState = concat(this.inkState, st.data as Uint8Array);
    this.n += n;
  }

  /** A float channel's rows (length n × dims[name]); throws if the demo has none. */
  float(name: FloatChannel): Float32Array {
    const data = this[name];
    if (!data) throw new Error(`FrameStore: this demo has no "${name}" channel`);
    return data;
  }

  /** Index into ``takes`` of the take holding frame ``i`` (fractional: its floor), or -1. */
  takeOf(i: number): number {
    if (!(i >= 0 && i < this.n)) return -1;
    const t = this.takes;
    let lo = 0;
    let hi = t.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (t[mid].start <= i) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  /** Ink intensity × 255 of attract frame ``i`` (for the wipe takeover), or null for other frames. */
  inkStateRow(i: number): Uint8Array | null {
    const k = this.takeOf(i);
    if (k < 0 || this.stateRow[k] < 0 || !this.inkState) return null;
    const cells = this.dims.ink as number;
    const row = this.stateRow[k] + Math.floor(i) - this.takes[k].start;
    return this.inkState.subarray(row * cells, (row + 1) * cells);
  }
}

/** ``a`` followed by ``b``; chunk 0's arrays are adopted without a copy. */
function concat<T extends NumArray>(a: T | null, b: T): T {
  if (!a || a.length === 0) return b;
  const out = new (a.constructor as new (n: number) => T)(a.length + b.length);
  out.set(a);
  out.set(b, a.length);
  return out;
}
