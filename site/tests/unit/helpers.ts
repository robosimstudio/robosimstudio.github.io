// Shared test fixtures: a seeded RNG, synthetic DB chunks that go through the real
// RSSB writer and reader (so store/player/matcher tests see decoded data), and a
// look-at camera.

import type { CameraJson } from "../../src/core/camera.ts";
import type { V3 } from "../../src/core/math.ts";
import { buildRssb, type DType, type Enc, parseRssb, type Rssb } from "../../src/core/rssb.ts";

/** mulberry32: a small, fast, seeded uniform generator on [0, 1). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal samples (Box-Muller) from a uniform generator. */
export function gauss(u: () => number): () => number {
  return () => Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u());
}

export interface ChannelFill {
  dim: number;
  dtype?: DType;
  enc?: Enc;
  /** Value of column d at global frame i. */
  f: (i: number, d: number) => number;
}

/** Contract dtypes of each DB array. */
const DTYPE: Record<string, [DType, Enc]> = {
  q: ["f32", "q16d"],
  hand: ["f32", "raw"],
  setpoint: ["f32", "raw"],
  cursor: ["f32", "raw"],
  deform: ["f32", "q16d"],
  feat: ["f16", "raw"],
  ink: ["u8", "raw"],
  broken: ["u16", "raw"],
  front: ["u8", "raw"],
  still: ["i32", "raw"],
};

const DEFAULT_FILL: Record<string, ChannelFill> = {
  q: { dim: 3, f: (i, d) => Math.sin(i * 0.01 + d) },
  hand: { dim: 3, f: (i, d) => i * 10 + d },
  setpoint: { dim: 3, f: (i, d) => i * 10 + d + 0.5 },
  cursor: { dim: 2, f: (i, d) => i + d / 4 },
  feat: { dim: 4, f: (i, d) => ((i * 7 + d * 3) % 11) / 4 },
};

/**
 * A DB chunk holding ``takes`` ((start, len, kind), global starts, contiguous).
 * Channels default to simple functions of the global frame index; ``fill`` adds or
 * overrides channels (``null`` drops one). ``inkState`` gives an attract frame's state row.
 */
export function dbChunk(
  takes: [number, number, number][],
  fill: Record<string, ChannelFill | null> = {},
  inkState?: (i: number, c: number) => number,
): Rssb {
  const frame0 = takes[0][0];
  const n = takes.reduce((s, t) => s + t[1], 0);
  const arrays: Parameters<typeof buildRssb>[0] = {
    takes: { data: takes.flat(), shape: [takes.length, 3], dtype: "u32" },
  };
  const channels = { ...DEFAULT_FILL, ...fill };
  for (const [name, ch] of Object.entries(channels)) {
    if (!ch) continue;
    const [dtype, enc] = DTYPE[name] ?? ["f32", "raw"];
    const data = new Float64Array(n * ch.dim);
    for (let k = 0; k < n; k++) for (let d = 0; d < ch.dim; d++) data[k * ch.dim + d] = ch.f(frame0 + k, d);
    const shape = ch.dim === 1 && (name === "broken" || name === "still") ? [n] : [n, ch.dim];
    arrays[name] = { data, shape, dtype: ch.dtype ?? dtype, enc: ch.enc ?? enc };
  }
  if (inkState) {
    const cells = (channels.ink as ChannelFill).dim;
    const rows: number[] = [];
    for (const [start, len, kind] of takes)
      if (kind === 1) for (let i = start; i < start + len; i++) for (let c = 0; c < cells; c++) rows.push(inkState(i, c));
    arrays.ink_state = { data: rows, shape: [rows.length / cells, cells], dtype: "u8" };
  }
  return parseRssb(buildRssb(arrays, { frame0 }));
}

/** A CV camera at ``eye`` looking at ``target`` (world z up): x right, y down, z forward. */
export function lookAt(eye: V3, target: V3): CameraJson {
  const unit = (a: V3): V3 => {
    const n = Math.hypot(a[0], a[1], a[2]);
    return [a[0] / n, a[1] / n, a[2] / n];
  };
  const z = unit([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
  const x = unit([z[1], -z[0], 0]); // z × up, up = +z
  const y: V3 = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]; // z × x
  return { width: 1920, height: 1080, fx: 1400, fy: 1390, cx: 955.5, cy: 541.25, cam_to_world_cv: [...x, 0, ...y, 0, ...z, 0, ...eye, 1] };
}
