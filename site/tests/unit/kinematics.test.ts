import { describe, expect, it } from "vitest";
import { type KinematicsJson, Kinematics, quatAngle } from "../../src/core/kinematics.ts";
import type { Q, Tf } from "../../src/core/math.ts";
import { rng } from "./helpers.ts";

// ---- an independent reference: 4x4 row-major matrices, rotations by Rodrigues' formula
type M = number[];
const mmul = (a: M, b: M): M => {
  const o = new Array(16).fill(0);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) for (let k = 0; k < 4; k++) o[r * 4 + c] += a[r * 4 + k] * b[k * 4 + c];
  return o;
};
const chain = (...ms: M[]): M => ms.reduce(mmul);
const trans = (x: number, y: number, z: number): M => [1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1];
function rot(axis: number[], th: number): M {
  const n = Math.hypot(axis[0], axis[1], axis[2]);
  const [x, y, z] = axis.map((v) => v / n);
  const c = Math.cos(th);
  const s = Math.sin(th);
  const t = 1 - c;
  return [c + x * x * t, x * y * t - z * s, x * z * t + y * s, 0, y * x * t + z * s, c + y * y * t, y * z * t - x * s, 0, z * x * t - y * s, z * y * t + x * s, c + z * z * t, 0, 0, 0, 0, 1];
}
function fromPose7(p: number[]): M {
  const [px, py, pz, x, y, z, w] = p;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), px,
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), py,
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), pz,
    0, 0, 0, 1,
  ];
}
function rigidInv(m: M): M {
  const o = [m[0], m[4], m[8], 0, m[1], m[5], m[9], 0, m[2], m[6], m[10], 0, 0, 0, 0, 1];
  for (let r = 0; r < 3; r++) o[r * 4 + 3] = -(o[r * 4] * m[3] + o[r * 4 + 1] * m[7] + o[r * 4 + 2] * m[11]);
  return o;
}
/** Shepperd's matrix → xyzw quaternion. */
function quatOf(m: M): Q {
  const [a, b, c, , d, e, f, , g, h, i] = m;
  const tr = a + e + i;
  if (tr > 0) {
    const s = 2 * Math.sqrt(tr + 1);
    return [(h - f) / s, (c - g) / s, (d - b) / s, s / 4];
  }
  if (a > e && a > i) {
    const s = 2 * Math.sqrt(1 + a - e - i);
    return [s / 4, (b + d) / s, (c + g) / s, (h - f) / s];
  }
  if (e > i) {
    const s = 2 * Math.sqrt(1 + e - a - i);
    return [(b + d) / s, s / 4, (f + h) / s, (c - g) / s];
  }
  const s = 2 * Math.sqrt(1 + i - a - e);
  return [(c + g) / s, (f + h) / s, s / 4, (d - b) / s];
}
const pose7 = (p: number[], axis: number[], ang: number): number[] => {
  const n = Math.hypot(axis[0], axis[1], axis[2]);
  const s = Math.sin(ang / 2) / n;
  return [p[0], p[1], p[2], axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(ang / 2)];
};
const ID7 = [0, 0, 0, 0, 0, 0, 1];
const matOf = (t: Tf): M => fromPose7([...t.p, ...t.q]);

// ---- the fixture: a mounted 3-dof arm (revolute z, revolute y with offsets, prismatic x),
// a finger driven by a coordinate the DB does not store, and a fixed tool flange.
const MOUNT = 9;
const ARM = {
  root: pose7([0.5, -0.2, 0.8], [0, 0, 1], Math.PI / 6),
  xp: [pose7([0, 0, 0.1], [0, 0, 1], 0), pose7([0.02, 0, 0.3], [1, 0, 0], 0.35), pose7([0.4, 0, 0], [0, 0, 1], 0), pose7([0.1, 0, 0], [0, 1, 0], Math.PI / 2), pose7([0, 0, 0.03], [0, 0, 1], 0.4)],
  xc: [ID7, pose7([0.05, 0.01, 0], [0, 0, 1], 0.17), pose7([0, 0, 0.02], [1, 1, 0], 0.2), ID7, pose7([0.01, 0, 0], [0, 1, 0], -0.3)],
  tcp: pose7([0, 0, 0.05], [1, 0, 0], 0.1),
};
const qFull = [0.7, 0, 0, -0.3, 0, 0.01];
const armJson = (tests: KinematicsJson["tests"] = []): KinematicsJson => ({
  q_index: [1, 2, 4],
  q_full: qFull,
  joints: [
    { type: "revolute", parent: MOUNT, child: 1, X_p: ARM.xp[0], X_c: ARM.xc[0], axis: [0, 0, 1], q_start: 1 },
    { type: "revolute", parent: 1, child: 2, X_p: ARM.xp[1], X_c: ARM.xc[1], axis: [0, 1, 0], q_start: 2 },
    { type: "prismatic", parent: 2, child: 3, X_p: ARM.xp[2], X_c: ARM.xc[2], axis: [1, 0, 0], q_start: 4 },
    { type: "prismatic", parent: 3, child: 4, X_p: ARM.xp[3], X_c: ARM.xc[3], axis: [0, 0, 1], q_start: 5 },
    { type: "fixed", parent: 4, child: 5, X_p: ARM.xp[4], X_c: ARM.xc[4], axis: [0, 0, 0], q_start: 0 },
  ],
  bodies: [1, 2, 3, 4, 5].map((index) => ({ index, label: `link${index}` })),
  roots: { [MOUNT]: ARM.root },
  tcp: { right: { body: 5, offset: ARM.tcp } },
  tests,
});

/** The same arm by hand: pose[child] = pose[parent] · X_p · X_j · X_c⁻¹ with matrices. */
function reference(q: number[]): Map<number, M> {
  const X = (k: number, joint: M) => chain(fromPose7(ARM.xp[k]), joint, rigidInv(fromPose7(ARM.xc[k])));
  const b1 = mmul(fromPose7(ARM.root), X(0, rot([0, 0, 1], q[0])));
  const b2 = mmul(b1, X(1, rot([0, 1, 0], q[1])));
  const b3 = mmul(b2, X(2, trans(q[2], 0, 0)));
  const b4 = mmul(b3, X(3, trans(0, 0, qFull[5]))); // not in q_index: keeps its canonical value
  const b5 = mmul(b4, X(4, trans(0, 0, 0)));
  return new Map([[1, b1], [2, b2], [3, b3], [4, b4], [5, b5]]);
}

const maxAbsDiff = (a: M, b: M) => Math.max(...a.map((x, i) => Math.abs(x - b[i])));
const u = rng(3);
const randQ = () => [u() * 6 - 3, u() * 3 - 1.5, u() * 0.4 - 0.1];

describe("Kinematics.fk", () => {
  it("matches an independent matrix computation", () => {
    const kin = new Kinematics(armJson());
    for (let k = 0; k < 20; k++) {
      const q = randQ();
      const poses = kin.fk(q);
      const ref = reference(q);
      expect([...poses.keys()].sort()).toEqual([1, 2, 3, 4, 5]);
      for (const [body, m] of ref) expect(maxAbsDiff(matOf(poses.get(body) as Tf), m)).toBeLessThan(1e-12);
      const tcp = kin.tcp("right", poses);
      expect(maxAbsDiff(matOf(tcp), mmul(ref.get(5) as M, fromPose7(ARM.tcp)))).toBeLessThan(1e-12);
    }
  });

  it("refills a reused output map in place", () => {
    const kin = new Kinematics(armJson());
    const out = kin.fk([0, 0, 0]);
    const link3 = out.get(3) as Tf;
    const again = kin.fk([0.3, -0.2, 0.1], out);
    expect(again).toBe(out);
    expect(again.get(3)).toBe(link3);
    expect(maxAbsDiff(matOf(link3), reference([0.3, -0.2, 0.1]).get(3) as M)).toBeLessThan(1e-12);
  });

  it("drives free and ball joints, renormalizing their quaternions", () => {
    const free = pose7([0.1, 0.2, 0.3], [1, 2, 3], 0.8);
    const ball = pose7([0, 0, 0], [0, 1, 1], -1.1).slice(3);
    const kin = new Kinematics({
      q_index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      q_full: new Array(11).fill(0),
      joints: [
        { type: "free", parent: -1, child: 0, X_p: ID7, X_c: ID7, axis: [0, 0, 0], q_start: 0 },
        { type: "ball", parent: 0, child: 1, X_p: pose7([0, 0, 0.5], [0, 0, 1], 0), X_c: ID7, axis: [0, 0, 0], q_start: 7 },
      ],
      bodies: [],
      roots: {},
      tcp: {},
      tests: [],
    });
    const scaled = [...free.slice(0, 3), ...free.slice(3).map((x) => x * 1.001), ...ball.map((x) => x * 0.999)];
    const poses = kin.fk(scaled);
    const b0 = fromPose7(free);
    const b1 = chain(b0, trans(0, 0, 0.5), fromPose7([0, 0, 0, ...ball]));
    expect(maxAbsDiff(matOf(poses.get(0) as Tf), b0)).toBeLessThan(1e-12);
    expect(maxAbsDiff(matOf(poses.get(1) as Tf), b1)).toBeLessThan(1e-12);
    expect(Math.hypot(...(poses.get(1) as Tf).q)).toBeCloseTo(1, 14);
  });

  it("rejects kinematics it cannot evaluate", () => {
    const orphan = armJson();
    orphan.roots = {};
    expect(() => new Kinematics(orphan)).toThrow(/parent body 9 is neither produced/);
    const outside = armJson();
    outside.q_index = [1, 2, 6];
    expect(() => new Kinematics(outside)).toThrow(/q_index entry 6 is outside q_full/);
    expect(() => new Kinematics(armJson()).fk([0, 0])).toThrow(/2 robot coordinates, expected 3/);
    expect(() => new Kinematics(armJson()).tcp("left", new Map())).toThrow(/no tcp "left"/);
  });
});

describe("Kinematics.selfTest", () => {
  const vectors = (qs: number[][]) =>
    qs.map((q) => {
      const ref = reference(q);
      return { q, poses: Object.fromEntries([...ref].map(([b, m]) => [String(b), [m[3], m[7], m[11], ...quatOf(m)]])) };
    });

  it("reproduces the file's own test vectors", () => {
    const kin = new Kinematics(armJson(vectors([randQ(), randQ(), randQ(), [0, 0, 0]])));
    const { maxPosErr, maxAngErr } = kin.selfTest();
    expect(maxPosErr).toBeLessThan(1e-12);
    expect(maxAngErr).toBeLessThan(1e-9);
  });

  it("reports a wrong vector's position and angle error", () => {
    const tests = vectors([[0.2, 0.1, 0.05]]);
    const p = tests[0].poses["3"];
    const off = fromPose7(p);
    const bent = mmul(off, rot([0, 0, 1], 0.01));
    tests[0].poses["3"] = [p[0] + 0.001, p[1], p[2], ...quatOf(bent)];
    const { maxPosErr, maxAngErr } = new Kinematics(armJson(tests)).selfTest();
    expect(maxPosErr).toBeCloseTo(0.001, 12);
    expect(maxAngErr).toBeCloseTo(0.01, 9);
  });

  it("measures tiny rotations accurately", () => {
    const a: Q = [0, 0, 0, 1];
    expect(quatAngle(a, [Math.sin(5e-9), 0, 0, Math.cos(5e-9)])).toBeCloseTo(1e-8, 20);
  });
});
