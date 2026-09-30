import { describe, expect, it } from "vitest";
import { type Q, qFromAxisAngle, type V3 } from "../../src/core/math.ts";
import { type OscConstants, OscSetpoint, predict } from "../../src/core/osc.ts";

const C: OscConstants = { control_dt: 1 / 60, pos_speed: 0.3, pos_lead: 0.02, rot_speed: 2, rot_lead: 0.3 };
const I: Q = [0, 0, 0, 1];
const step = C.pos_speed * C.control_dt; // 5 mm
const norm = (v: ArrayLike<number>) => Math.hypot(v[0], v[1], v[2]);
const angle = (a: Q, b: Q) => 2 * Math.acos(Math.min(1, Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])));

describe("OscSetpoint", () => {
  it("cruises toward a far goal at pos_speed * dt per step", () => {
    const s = new OscSetpoint(C);
    s.reset([0, 0, 0], I);
    const goal: V3 = [1, 0, 0];
    for (let k = 1; k <= 10; k++) {
      const { p } = s.advance(s.p, s.q, goal, I); // the hand tracks the setpoint perfectly
      expect(p[0]).toBeCloseTo(k * step, 14);
      expect(p[1]).toBe(0);
    }
  });

  it("turns toward a far orientation at rot_speed * dt per step", () => {
    const s = new OscSetpoint(C);
    s.reset([0, 0, 0], I);
    const goal = qFromAxisAngle([0, 0, 1], 2);
    for (let k = 1; k <= 5; k++) {
      const { q } = s.advance(s.p, s.q, s.p, goal);
      expect(angle(q, I)).toBeCloseTo(k * C.rot_speed * C.control_dt, 9);
    }
  });

  it("holds the lead clamp on a blocked arm: no creep", () => {
    const s = new OscSetpoint(C);
    s.reset([0, 0, 0], I);
    const hand: V3 = [0, 0, 0];
    let p: V3 = [0, 0, 0];
    for (let k = 0; k < 200; k++) p = s.advance(hand, I, [1, 0.5, 0], I).p;
    expect(norm(p)).toBeCloseTo(C.pos_lead, 15);
    const again = s.advance(hand, I, [1, 0.5, 0], I).p;
    expect(norm([again[0] - p[0], again[1] - p[1], again[2] - p[2]])).toBeLessThan(1e-15);
  });

  it("reaches a goal within one step exactly", () => {
    const s = new OscSetpoint(C);
    s.reset([0.1, 0.2, 0.3], I);
    const goal: V3 = [0.103, 0.2, 0.301];
    expect(s.advance(s.p, I, goal, I).p).toEqual(goal);
  });
});

describe("predict", () => {
  const plant = { alpha: 0.2, beta: 0.3 }; // poles 0.6 and 0.5: stable, no overshoot

  it("converges on a static goal", () => {
    const goal: V3 = [0.05, -0.02, 0.01];
    const hand: V3 = [0, 0, 0];
    const pr = predict(C, plant, hand, hand, hand, () => goal, [6, 12, 20, 600]);
    expect(pr.disp).toHaveLength(4);
    const d = pr.disp.map((v) => norm([goal[0] - v[0], goal[1] - v[1], goal[2] - v[2]]));
    for (let k = 1; k < d.length; k++) expect(d[k]).toBeLessThan(d[k - 1]);
    expect(d[3]).toBeLessThan(1e-9);
    for (let i = 0; i < 3; i++) expect(pr.end[i]).toBeCloseTo(goal[i], 9);
  });

  it("rate-limits the goal itself with pre_limit", () => {
    const limited = { ...C, pre_limit: 0.05 };
    const hand: V3 = [0, 0, 0];
    const pr = predict(limited, plant, hand, hand, hand, () => [1, 0, 0], [20]);
    // the goal can have moved at most 20 steps at 50 mm/s, and the hand lags it
    expect(norm(pr.disp[0])).toBeLessThanOrEqual(20 * 0.05 * C.control_dt + 1e-12);
    expect(pr.disp[0][0]).toBeGreaterThan(0);
  });
});
