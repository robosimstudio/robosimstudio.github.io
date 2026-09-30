import { describe, expect, it } from "vitest";
import { type FeatureGroup, type Match, Matcher, type SearchOptions } from "../../src/core/matcher.ts";
import { FrameStore, type Take } from "../../src/core/store.ts";
import { dbChunk, gauss, rng } from "./helpers.ts";

// the free-drag layout: fd 9, hp 3, hv 3, dm m, dv m, pg -- 27 columns
const GROUPS: FeatureGroup[] = [
  { name: "fd", dims: 9, w: 1.0 },
  { name: "hp", dims: 3, w: 0.5 },
  { name: "hv", dims: 3, w: 0.8 },
  { name: "dm", dims: 5, w: 2.0 },
  { name: "dv", dims: 5, w: 0.3 },
  { name: "pg", dims: 2, w: 1.5 },
];
const D = 27;
const FRONT = 3;

/** Takes of random lengths (each at least ``min``) tiling n frames. */
function randomTakes(n: number, count: number, u: () => number, min = 200): [number, number, number][] {
  const lens = Array.from({ length: count }, () => min + u());
  const scale = (n - count * min) / lens.reduce((s, l) => s + l - min, 0);
  const takes: [number, number, number][] = [];
  let start = 0;
  lens.forEach((l, k) => {
    const len = k === count - 1 ? n - start : min + Math.floor((l - min) * scale);
    takes.push([start, len, k === 0 ? 1 : 0]);
    start += len;
  });
  return takes;
}

function randomStore(n: number, takes: number, seed: number, feat: (i: number, d: number) => number): FrameStore {
  const u = rng(seed);
  const broken = Array.from({ length: n }, () => Math.floor(u() * 500));
  const front = Array.from({ length: n * FRONT }, () => Math.floor(u() * 256));
  return new FrameStore(
    dbChunk(randomTakes(n, takes, u), {
      feat: { dim: D, f: feat },
      broken: { dim: 1, f: (i) => broken[i] },
      front: { dim: FRONT, f: (i, c) => front[i * FRONT + c] },
    }),
  );
}

/** Every admissible frame in index order, summing in the matcher's order. */
function brute(store: FrameStore, W: Float32Array, q: ArrayLike<number>, o: SearchOptions): Match {
  const minRunway = o.minRunway ?? 10;
  const win = o.sameTakeWindow ?? 15;
  const margin = o.startMargin ?? 30;
  const cur = o.current;
  const curTake = cur === undefined ? -1 : store.takeOf(cur);
  let cost = Number.POSITIVE_INFINITY;
  let frame = -1;
  store.takes.forEach((t, k) => {
    for (let i = t.start; i < t.start + t.len; i++) {
      if (i - t.start < margin || t.start + t.len - 1 - i < minRunway) continue;
      if (k === curTake && Math.abs(i - (cur as number)) <= win) continue;
      const m = o.monotone;
      if (m) {
        if (m.broken[i] < m.cur.broken) continue;
        const cf = m.cur.front;
        if (m.front && cf && Array.from(cf).some((v, c) => (m.front as Uint8Array)[i * cf.length + c] < v)) continue;
      }
      let s = 0;
      for (let d = 0; d < D; d++) {
        const x = W[i * D + d] - q[d];
        s += x * x;
      }
      if (o.bias) s += o.bias(i);
      if (s < cost) {
        cost = s;
        frame = i;
      }
    }
  });
  return { frame, cost };
}

/** A deterministic, non-negative per-frame penalty. */
const hashBias = (scale: number) => (i: number) => (((Math.imul(i, 2654435761) >>> 0) % 1000) / 1000) * scale;

const g = gauss(rng(11));
const white = Array.from({ length: 20000 * D }, () => g());
const store = randomStore(20000, 7, 12, (i, d) => white[i * D + d]);
const matcher = new Matcher(store, GROUPS);
const rowOf = (i: number) => Float64Array.from(matcher.row(i));
const mid = (t: Take) => t.start + (t.len >> 1);

describe("Matcher", () => {
  it("weights each group by sqrt(w)", () => {
    expect(store.takes).toHaveLength(7);
    const sw = GROUPS.flatMap((gr) => new Array(gr.dims).fill(Math.sqrt(gr.w)));
    for (let i = 0; i < store.n; i += 97)
      for (let d = 0; d < D; d++) expect(matcher.W[i * D + d]).toBe(Math.fround(store.feat[i * D + d] * sw[d]));
    expect(Array.from(matcher.weigh(new Float64Array(D).fill(1), new Float64Array(D)))).toEqual(sw);
    const q = rowOf(123);
    q[0] += 0.5;
    expect(matcher.costAt(123, q)).toBeCloseTo(0.25, 12);
  });

  it("equals brute force on 2000 random queries", () => {
    const u = rng(5);
    const pick = () => Math.floor(u() * store.n);
    let empty = 0;
    for (let k = 0; k < 2000; k++) {
      const src = pick();
      const kind = u();
      const q = new Float64Array(D);
      for (let d = 0; d < D; d++) {
        const row = matcher.W[src * D + d];
        q[d] = kind < 0.3 ? row : kind < 0.8 ? row + 0.05 * g() : matcher.scale[d] * g();
      }
      const o: SearchOptions = {};
      const r = u();
      if (r < 0.25) o.current = src;
      else if (r < 0.5) o.current = pick() + 0.5;
      if (u() < 0.3) {
        o.minRunway = Math.floor(u() * 40);
        o.sameTakeWindow = Math.floor(u() * 40);
        o.startMargin = Math.floor(u() * 40);
      }
      if (u() < 0.25) o.bias = hashBias(u() * 5);
      if (u() < 0.25)
        o.monotone = {
          broken: store.broken as Uint16Array,
          front: u() < 0.8 ? (store.front as Uint8Array) : null,
          cur: { broken: Math.floor(u() * 300), front: [0, 0, 0].map(() => Math.floor(u() * 128)) },
        };
      const want = brute(store, matcher.W, q, o);
      expect(matcher.search(q, o)).toEqual(want); // same frame, bit-identical cost
      if (want.frame < 0) empty++;
    }
    expect(empty).toBe(0);
  });

  it("excludes frames without runway, near the playhead, and at take starts", () => {
    const t = store.takes[3];
    const last = t.start + t.len - 1;
    // runway 9 < 10: never chosen, however well it matches
    const f = last - 9;
    expect(matcher.search(rowOf(f)).frame).not.toBe(f);
    expect(matcher.search(rowOf(f), { minRunway: 9 })).toEqual({ frame: f, cost: 0 });
    // the same-take window is ±15 around the current frame, inclusive
    const m = mid(t);
    expect(matcher.search(rowOf(m), { current: m + 15 }).frame).not.toBe(m);
    expect(matcher.search(rowOf(m), { current: m + 16 }).frame).toBe(m);
    expect(matcher.search(rowOf(m), { current: mid(store.takes[4]) }).frame).toBe(m);
    // a take's first 30 frames are out
    expect(matcher.search(rowOf(t.start + 29)).frame).not.toBe(t.start + 29);
    expect(matcher.search(rowOf(t.start + 30)).frame).toBe(t.start + 30);
    expect(matcher.search(rowOf(t.start), { startMargin: 0 }).frame).toBe(t.start);
    // nothing admissible at all
    expect(matcher.search(rowOf(m), { minRunway: 1e9 })).toEqual({ frame: -1, cost: Number.POSITIVE_INFINITY });
  });

  it("keeps the peel monotone", () => {
    const broken = store.broken as Uint16Array;
    const front = store.front as Uint8Array;
    const m = mid(store.takes[2]);
    const keys = { broken: broken[m], front: Array.from(front.subarray(m * FRONT, (m + 1) * FRONT)) };
    // equal keys are allowed ...
    expect(matcher.search(rowOf(m), { monotone: { broken, front, cur: keys } }).frame).toBe(m);
    // ... one bond more is not, and neither is one column further on
    const hit = matcher.search(rowOf(m), { monotone: { broken, front, cur: { ...keys, broken: keys.broken + 1 } } });
    expect(hit.frame).not.toBe(m);
    expect(broken[hit.frame]).toBeGreaterThanOrEqual(keys.broken + 1);
    const ahead = [keys.front[0], keys.front[1] + 1, keys.front[2]];
    const hit2 = matcher.search(rowOf(m), { monotone: { broken, front, cur: { broken: 0, front: ahead } } });
    expect(hit2.frame).not.toBe(m);
    for (let c = 0; c < FRONT; c++) expect(front[hit2.frame * FRONT + c]).toBeGreaterThanOrEqual(ahead[c]);
  });

  it("adds the bias to every candidate's cost", () => {
    const m = mid(store.takes[5]);
    const bias = (i: number) => (i === m ? 1e6 : 0.5);
    const got = matcher.search(rowOf(m), { bias });
    expect(got.frame).not.toBe(m);
    expect(got).toEqual(brute(store, matcher.W, rowOf(m), { bias }));
    expect(got.cost).toBeCloseTo(matcher.costAt(got.frame, rowOf(m)) + 0.5, 12);
  });

  it("indexes appended chunks on the next search", () => {
    const small = randomStore(3000, 2, 21, (i, d) => white[i * D + d]);
    const mt = new Matcher(small, GROUPS);
    small.append(dbChunk([[3000, 500, 0]], { feat: { dim: D, f: (i, d) => white[i * D + d] }, broken: { dim: 1, f: () => 0 }, front: { dim: FRONT, f: () => 0 } }));
    const q = Float64Array.from({ length: D }, (_, d) => Math.fround(small.feat[3200 * D + d] * mt.scale[d]));
    expect(mt.search(q)).toEqual({ frame: 3200, cost: 0 });
    expect(mt.n).toBe(3500);
  });

  it("times a 60k x 27 search (reported, not enforced)", () => {
    // smooth per-take features, like the real DB's hand/PCA trajectories at the control rate
    const N = 60000;
    const u = rng(99);
    const waves = Array.from({ length: D * 3 }, () => ({ f: 0.002 + u() * 0.05, p: u() * 6.3, a: 0.4 + u() }));
    const smooth = (i: number, d: number) => {
      let s = 0;
      for (let k = 0; k < 3; k++) {
        const w = waves[d * 3 + k];
        s += w.a * Math.sin(w.f * i + w.p + 17 * Math.floor(i / 7000));
      }
      return s;
    };
    const big = randomStore(N, 12, 98, smooth);
    const mt = new Matcher(big, GROUPS);
    const queries = Array.from({ length: 600 }, () => {
      const src = Math.floor(u() * N);
      return { q: Float64Array.from(mt.row(src), (x) => x + 0.1 * g()), current: src };
    });
    const time = (from: number, to: number) => {
      const t0 = performance.now();
      for (let k = from; k < to; k++) mt.search(queries[k].q, { current: queries[k].current });
      return (performance.now() - t0) / (to - from);
    };
    time(0, 100); // warm up the JIT
    const perQuery = time(100, 600);
    const t0 = performance.now();
    for (let k = 0; k < 20; k++) brute(big, mt.W, queries[k].q, { current: queries[k].current });
    const bruteMs = (performance.now() - t0) / 20;
    console.log(`[matcher] 60k x 27 smooth: ${perQuery.toFixed(3)} ms/search (brute force ${bruteMs.toFixed(2)} ms)`);
    for (let k = 0; k < 20; k++) expect(mt.search(queries[k].q, { current: queries[k].current })).toEqual(brute(big, mt.W, queries[k].q, { current: queries[k].current }));
    expect(perQuery).toBeGreaterThan(0);
  });
});
