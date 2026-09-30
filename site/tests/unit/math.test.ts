import { describe, expect, it } from "vitest";
import {
  type Q,
  qconj,
  qExp,
  qFromAxisAngle,
  qLog,
  qmul,
  qnormalize,
  qrot,
  qSlerp,
  type Tf,
  tfInv,
  tfMul,
  tfMulTo,
  type V3,
} from "../../src/core/math.ts";
import { gauss, rng } from "./helpers.ts";

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol = 1e-12) => {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol);
};
/** Equal as rotations (q and -q are the same). */
const sameRot = (a: Q, b: Q, tol = 1e-12) => {
  const s = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] < 0 ? -1 : 1;
  close(a, [s * b[0], s * b[1], s * b[2], s * b[3]], tol);
};

const g = gauss(rng(1));
const randQ = (): Q => qnormalize([g(), g(), g(), g()]);
const randV = (): V3 => [g(), g(), g()];
const randTf = (): Tf => ({ p: randV(), q: randQ() });

describe("quaternions", () => {
  it("qmul has identity, conjugate inverse and associativity", () => {
    for (let k = 0; k < 50; k++) {
      const [a, b, c] = [randQ(), randQ(), randQ()];
      close(qmul(a, [0, 0, 0, 1]), a);
      close(qmul([0, 0, 0, 1], a), a);
      close(qmul(a, qconj(a)), [0, 0, 0, 1]);
      close(qmul(qmul(a, b), c), qmul(a, qmul(b, c)));
    }
  });

  it("qrot turns x into y about z and composes like qmul", () => {
    close(qrot(qFromAxisAngle([0, 0, 1], Math.PI / 2), [1, 0, 0]), [0, 1, 0]);
    close(qrot(qFromAxisAngle([1, 0, 0], Math.PI / 2), [0, 1, 0]), [0, 0, 1]);
    for (let k = 0; k < 50; k++) {
      const [a, b, v] = [randQ(), randQ(), randV()];
      close(qrot(qmul(a, b), v), qrot(a, qrot(b, v)));
    }
  });

  it("qSlerp matches hand-computed values", () => {
    const I: Q = [0, 0, 0, 1];
    const z90: Q = [0, 0, Math.SQRT1_2, Math.SQRT1_2];
    const z45: Q = [0, 0, Math.sin(Math.PI / 8), Math.cos(Math.PI / 8)];
    close(qSlerp(I, z90, 0.5), z45);
    close(qSlerp(I, z90, 0), I);
    close(qSlerp(I, z90, 1), z90);
    // b in the other hemisphere is negated first: still the short 45° way
    close(qSlerp(I, [0, 0, -Math.SQRT1_2, -Math.SQRT1_2], 0.5), z45);
    // 120° about (1,1,1)/√3: dot = cos 60°, so t = 1/2 is 60° about that axis
    const s = 1 / Math.sqrt(3);
    const b: Q = [s * Math.sin(Math.PI / 3), s * Math.sin(Math.PI / 3), s * Math.sin(Math.PI / 3), 0.5];
    close(qSlerp(I, b, 0.5), [s * 0.5, s * 0.5, s * 0.5, Math.cos(Math.PI / 6)]);
    // dot > 0.9995 takes the normalized-lerp branch: exactly normalize(a + t (b - a))
    const tiny: Q = [Math.sin(0.01), 0, 0, Math.cos(0.01)];
    const nl = qnormalize([0.25 * Math.sin(0.01), 0, 0, 0.75 + 0.25 * Math.cos(0.01)]);
    close(qSlerp(I, tiny, 0.25), nl, 1e-15);
    // ...which stays within 2e-8 of the true slerp at that small angle
    close(qSlerp(I, tiny, 0.25), [Math.sin(0.0025), 0, 0, Math.cos(0.0025)], 2e-8);
  });

  it("qLog and qExp invert each other", () => {
    for (let k = 0; k < 100; k++) {
      const v = randV();
      const n = Math.hypot(...v);
      const w: V3 = n > 3 ? [(v[0] * 3) / n, (v[1] * 3) / n, (v[2] * 3) / n] : v; // |w| < π: unique log
      close(qLog(qExp(w)), w, 1e-12);
      const q = randQ();
      sameRot(qExp(qLog(q)), q);
    }
    close(qLog(qExp([1e-13, -2e-13, 0])), [1e-13, -2e-13, 0], 1e-25);
  });
});

describe("transforms", () => {
  it("tfInv undoes tfMul and composition is associative", () => {
    for (let k = 0; k < 50; k++) {
      const [a, b, c] = [randTf(), randTf(), randTf()];
      const id = tfMul(a, tfInv(a));
      close(id.p, [0, 0, 0]);
      sameRot(id.q, [0, 0, 0, 1]);
      const l = tfMul(tfMul(a, b), c);
      const r = tfMul(a, tfMul(b, c));
      close(l.p, r.p);
      sameRot(l.q, r.q);
      // a ∘ b maps a point as "first b, then a"
      const p = randV();
      const ab = tfMul(a, b);
      const once = qrot(ab.q, p).map((x, i) => x + ab.p[i]);
      const bp = qrot(b.q, p).map((x, i) => x + b.p[i]) as V3;
      close(once, qrot(a.q, bp).map((x, i) => x + a.p[i]));
    }
  });

  it("tfMulTo equals tfMul, also when out aliases an input", () => {
    for (let k = 0; k < 50; k++) {
      const [a, b] = [randTf(), randTf()];
      const want = tfMul(a, b);
      const out: Tf = { p: [0, 0, 0], q: [0, 0, 0, 1] };
      tfMulTo(out, a, b);
      close(out.p, want.p, 1e-14);
      close(out.q, want.q, 1e-15);
      const a2: Tf = { p: [...a.p], q: [...a.q] };
      tfMulTo(a2, a2, b);
      close(a2.p, want.p, 1e-14);
      const b2: Tf = { p: [...b.p], q: [...b.q] };
      tfMulTo(b2, a, b2);
      close(b2.q, want.q, 1e-15);
    }
  });
});
