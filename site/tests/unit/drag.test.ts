import { describe, expect, it } from "vitest";
import { Camera } from "../../src/core/camera.ts";
import { DRAG_V_STOP, type DragJson, DragMapping, Flick, pinvSolve2, SIGMA_MIN_PX } from "../../src/core/drag.ts";
import { gauss, lookAt, rng } from "./helpers.ts";

// the wipe's box: a along the bar (world x), b height over the seat (world z)
const DRAG: DragJson = { origin: [0.1, 0.2, 0.75], A: [1, 0, 0], B: [0, 0, 1], a: [-0.3, 0.3], b: [-0.02, 0.1], hover: [0, 0.05], coast_tau: 0.3, v_max: 1.5 };
const cam = new Camera(lookAt([0.3, -1.1, 1.3], [0.1, 0.2, 0.75]));
const drag = new DragMapping(DRAG);
const u = rng(9);
const pixel = (a: number, b: number) => cam.project(drag.toWorld(a, b));

describe("DragMapping", () => {
  it("maps (a, b) to world and clamps to the box", () => {
    const w = drag.toWorld(0.1, 0.05);
    [0.2, 0.2, 0.8].forEach((x, i) => expect(w[i]).toBeCloseTo(x, 15));
    expect(drag.toWorld(0.5, 0)[0]).toBeCloseTo(0.6, 15); // toWorld itself does not clamp
    expect(drag.clamp(0.5, -1)).toEqual([0.3, -0.02]);
    expect(drag.clamp(-0.31, 0.2)).toEqual([-0.3, 0.1]);
    expect(drag.clamp(0.01, 0.02)).toEqual([0.01, 0.02]);
  });

  it("has the analytic Jacobian of the projection", () => {
    const h = 1e-6;
    for (let k = 0; k < 20; k++) {
      const [a, b] = [u() * 0.6 - 0.3, u() * 0.12 - 0.02];
      const J = drag.jacobian(a, b, cam);
      const da = [0, 1].map((r) => (pixel(a + h, b)[r] - pixel(a - h, b)[r]) / (2 * h));
      const db = [0, 1].map((r) => (pixel(a, b + h)[r] - pixel(a, b - h)[r]) / (2 * h));
      const num = [da[0], db[0], da[1], db[1]];
      J.forEach((x, i) => expect(Math.abs(x - num[i])).toBeLessThan(1e-4 * Math.max(1, Math.abs(num[i]))));
    }
  });

  it("solves Gauss-Newton onto a reachable pixel within 0.01 px", () => {
    for (let k = 0; k < 100; k++) {
      const [a, b] = [u() * 0.6 - 0.3, u() * 0.12 - 0.02];
      const target = pixel(a, b);
      const [sa, sb] = drag.solveAbsolute(target, drag.hover, cam);
      const got = pixel(sa, sb);
      expect(Math.hypot(got[0] - target[0], got[1] - target[1])).toBeLessThan(0.01);
      expect(sa).toBeCloseTo(a, 6);
      expect(sb).toBeCloseTo(b, 6);
    }
  });

  it("clamps an unreachable pixel's solution into the box", () => {
    const far = pixel(0.8, 0.05); // half a metre past the end of the bar
    const [a, b] = drag.solveAbsolute(far, drag.hover, cam);
    expect(a).toBe(0.3);
    expect(b).toBeGreaterThan(-0.02);
    expect(b).toBeLessThan(0.1);
  });

  it("moves the projected point by about dpx on a relative drag", () => {
    for (const [a, b] of [[0, 0.03], [-0.2, 0.08], [0.25, -0.01]]) {
      const dpx = [6, -4];
      const before = pixel(a, b);
      const [na, nb] = drag.relative(dpx, a, b, cam);
      const after = pixel(na, nb);
      expect(Math.abs(after[0] - before[0] - dpx[0])).toBeLessThan(0.05);
      expect(Math.abs(after[1] - before[1] - dpx[1])).toBeLessThan(0.05);
      const [ha, hb] = drag.relative(dpx, a, b, cam, 0.5);
      expect(ha - a).toBeCloseTo((na - a) / 2, 12);
      expect(hb - b).toBeCloseTo((nb - b) / 2, 12);
    }
  });
});

describe("pinvSolve2", () => {
  const g = gauss(rng(10));
  it("inverts a well-conditioned 2x2 matrix exactly", () => {
    for (let k = 0; k < 200; k++) {
      const m = [g(), g(), g(), g()].map((x) => x * 2000);
      const [s1, s2] = svals(m);
      if (s2 < SIGMA_MIN_PX) continue;
      const x = [g(), g()];
      const r = [m[0] * x[0] + m[1] * x[1], m[2] * x[0] + m[3] * x[1]];
      const got = pinvSolve2(m[0], m[1], m[2], m[3], r[0], r[1], SIGMA_MIN_PX);
      expect(Math.abs(got[0] - x[0]) + Math.abs(got[1] - x[1])).toBeLessThan(1e-9 * (s1 / s2));
    }
  });

  it("clamps small singular values instead of blowing up", () => {
    // diag(1000, 10): the second direction is divided by 150, not 10
    const [x0, x1] = pinvSolve2(1000, 0, 0, 10, 50, 10, SIGMA_MIN_PX);
    expect(x0).toBeCloseTo(0.05, 12);
    expect(x1).toBeCloseTo(10 / 150, 12);
    // a reflection (det < 0) inverts correctly
    const [y0, y1] = pinvSolve2(0, 400, 300, 0, 40, 60, SIGMA_MIN_PX);
    expect(y0).toBeCloseTo(0.2, 12);
    expect(y1).toBeCloseTo(0.1, 12);
    // rank one: bounded by |r| / 150
    const z = pinvSolve2(300, 600, 150, 300, 30, 40, SIGMA_MIN_PX);
    expect(Math.hypot(z[0], z[1])).toBeLessThanOrEqual(50 / 150 + 1e-12);
  });
});

/** Singular values of a row-major 2x2, from the eigenvalues of MᵀM. */
function svals(m: number[]): [number, number] {
  const p = m[0] * m[0] + m[2] * m[2];
  const q = m[0] * m[1] + m[2] * m[3];
  const r = m[1] * m[1] + m[3] * m[3];
  const disc = Math.sqrt(((p - r) / 2) ** 2 + q * q);
  return [Math.sqrt((p + r) / 2 + disc), Math.sqrt(Math.max(0, (p + r) / 2 - disc))];
}

describe("Flick", () => {
  const drive = (f: Flick, from: number, to: number, pos: (t: number) => [number, number]) => {
    for (let t = from; t <= to + 1e-9; t += 1 / 120) f.push(t, ...pos(t));
  };

  it("fits the release velocity to the last 100 ms only", () => {
    const f = Flick.forDrag(drag);
    // fast for 0.2 s, then slow: the release sees only the slow part
    drive(f, 0, 0.2, (t) => [-0.25 + t, 0]);
    drive(f, 0.2 + 1 / 120, 0.35, (t) => [-0.05 + 0.2 * (t - 0.2), 0.01 - 0.05 * (t - 0.2)]);
    expect(f.release(0.35)).toBe(true);
    expect(f.v[0]).toBeCloseTo(0.2, 9);
    expect(f.v[1]).toBeCloseTo(-0.05, 9);
    expect(f.x[0]).toBeCloseTo(-0.05 + 0.2 * 0.15, 9);
  });

  it("does not coast after the finger rested before letting go", () => {
    const f = Flick.forDrag(drag);
    drive(f, 0, 0.2, (t) => [-0.25 + t, 0]);
    expect(f.release(0.45)).toBe(false);
    expect(Array.from(f.v)).toEqual([0, 0]);
  });

  it("caps the release speed at v_max", () => {
    const f = Flick.forDrag(drag);
    drive(f, 0, 0.1, (t) => [-0.3 + 4 * t, 3 * t]);
    f.release(0.1);
    expect(Math.hypot(f.v[0], f.v[1])).toBeCloseTo(DRAG.v_max, 12);
    expect(f.v[1] / f.v[0]).toBeCloseTo(0.75, 9);
  });

  it("coasts with v *= exp(-dt/tau), independent of the step size, and stops below 0.01 m/s", () => {
    const f = Flick.forDrag(drag);
    drive(f, 0, 0.1, (t) => [-0.2 + 0.3 * t, 0]);
    f.release(0.1);
    const [x0, v0] = [f.x[0], f.v[0]];
    for (let k = 0; k < 12; k++) f.coast(0.025);
    const T = 0.3;
    expect(f.v[0]).toBeCloseTo(v0 * Math.exp(-T / DRAG.coast_tau), 12);
    expect(f.x[0]).toBeCloseTo(x0 + v0 * DRAG.coast_tau * (1 - Math.exp(-T / DRAG.coast_tau)), 12);
    let t = T;
    while (f.coast(1 / 60)) t += 1 / 60;
    expect(t).toBeCloseTo(DRAG.coast_tau * Math.log(v0 / DRAG_V_STOP), 1);
    expect(f.active).toBe(false);
    expect(Array.from(f.v)).toEqual([0, 0]);
  });

  it("zeroes only the velocity component pushing into a face", () => {
    const f = Flick.forDrag(drag);
    drive(f, 0, 0.1, (t) => [0.2 + 0.9 * t, 0.02 + 0.2 * t]);
    f.release(0.1);
    for (let k = 0; k < 6; k++) f.coast(1 / 60);
    expect(f.x[0]).toBe(0.3);
    expect(f.v[0]).toBe(0);
    expect(f.v[1]).toBeGreaterThan(0.1);
    expect(f.active).toBe(true);
  });

  it("works in one dimension", () => {
    const f = new Flick({ tau: 0.35, vMax: Number.POSITIVE_INFINITY, vStop: 0.5, lo: [0], hi: [100] });
    for (let t = 0; t <= 0.1; t += 0.01) f.push(t, 50 + 30 * t);
    expect(f.release(0.1)).toBe(true);
    expect(f.v[0]).toBeCloseTo(30, 9);
    f.coast(0.35);
    expect(f.x[0]).toBeCloseTo(53 + 30 * 0.35 * (1 - Math.exp(-1)), 9);
  });
});
