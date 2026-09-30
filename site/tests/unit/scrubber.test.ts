import { describe, expect, it } from "vitest";
import { SCRUB_COAST_TAU, Scrubber } from "../../src/core/scrubber.ts";
import { gauss, rng } from "./helpers.ts";

// two handles over 241 frames: handle 0 sits still, handle 1 draws a figure-8 twice
// (period T = 120 frames), crossing itself at the centre at t = 0, 60, 120, 180, 240
const N = 241;
const T = 120;
const eight = (s: number): [number, number] => [960 + 300 * Math.sin((2 * Math.PI * s) / T), 540 + 200 * Math.sin((4 * Math.PI * s) / T)];
const eePx = new Float32Array(N * 2 * 2);
for (let i = 0; i < N; i++) {
  eePx.set([100, 100], i * 4);
  eePx.set(eight(i), i * 4 + 2);
}
const sc = new Scrubber(eePx, N, 2);

/** Drag the finger along the path from s0 to s1 in steps of ``ds`` frames. */
function sweep(s0: number, s1: number, ds: number, lambda: number, noise = 0): number[] {
  const g = gauss(rng(6));
  let t = s0;
  const ts: number[] = [];
  for (let s = s0 + ds; s <= s1; s += ds) {
    const [x, y] = eight(s);
    t = sc.nearest([x + noise * g(), y + noise * g()], t, 1, 60, lambda);
    ts.push(t);
  }
  return ts;
}

describe("Scrubber.nearest", () => {
  it("passes a figure-8's crossings in order on a forward sweep", () => {
    for (const noise of [0, 1]) {
      const ts = sweep(20, 220, 0.5, 40, noise);
      for (let k = 1; k < ts.length; k++) expect(ts[k]).toBeGreaterThan(ts[k - 1]);
      // a jump to another visit would be off by 60 frames or more
      ts.forEach((t, k) => expect(Math.abs(t - (20.5 + 0.5 * k))).toBeLessThan(noise ? 0.5 : 0.02));
    }
  });

  it("would jump at the crossing without the time term", () => {
    // frames 0, 60 and 120 share one pixel: purely spatially they tie, and the earliest wins
    expect(sc.nearest([960, 540], 59.5, 1, 60, 0)).toBe(0);
    expect(sc.nearest([960, 540], 59.5, 1)).toBeCloseTo(60, 1);
  });

  it("finds fractional frames between samples", () => {
    const [x0, y0] = [eePx[40 * 4 + 2], eePx[40 * 4 + 3]];
    const [x1, y1] = [eePx[41 * 4 + 2], eePx[41 * 4 + 3]];
    const t = sc.nearest([x0 + 0.25 * (x1 - x0), y0 + 0.25 * (y1 - y0)], 40.25, 1);
    expect(t).toBeCloseTo(40.25, 9);
  });

  it("follows time alone on a handle that does not move", () => {
    expect(sc.nearest([500, 500], 100.3, 0)).toBeCloseTo(100.3, 12);
    expect(sc.nearest([500, 500], 500, 0)).toBe(N - 1);
  });

  it("only searches within the window", () => {
    // the finger is on frame 200's point, but the window around t = 20 ends at frame 80
    const t = sc.nearest(eight(200), 20, 1, 60);
    expect(t).toBeLessThanOrEqual(80);
  });
});

describe("Scrubber drags and flicks", () => {
  it("maps an empty-stage drag to frames", () => {
    expect(sc.relative(100, 400, 300)).toBe(75);
    expect(sc.relative(-100, 400)).toBe(-60); // default span: the whole clip
  });

  it("coasts the playhead with tau 0.35 s inside the clip", () => {
    for (let t = 0; t <= 0.1 + 1e-9; t += 1 / 60) sc.flick.push(t, 200 + 90 * t);
    expect(sc.flick.release(0.1)).toBe(true);
    expect(sc.flick.v[0]).toBeCloseTo(90, 6);
    sc.flick.coast(SCRUB_COAST_TAU);
    expect(sc.flick.v[0]).toBeCloseTo(90 * Math.exp(-1), 6);
    while (sc.flick.coast(1 / 60));
    expect(sc.flick.x[0]).toBe(N - 1); // clamped at the last frame, velocity zeroed
  });

  it("rejects an ee_px array of the wrong size and unknown handles", () => {
    expect(() => new Scrubber(new Float32Array(10), 3, 2)).toThrow(/\[3, 2, 2\] needs 12/);
    expect(() => sc.nearest([0, 0], 10, 2)).toThrow(RangeError);
  });
});
