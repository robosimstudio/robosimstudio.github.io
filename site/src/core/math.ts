// Small, allocation-light vector/quaternion/transform math in f64.
// Quaternions are xyzw, as Newton and RoboSimStudio store them.

export type V3 = [number, number, number];
export type Q = [number, number, number, number];
export interface Tf {
  p: V3;
  q: Q;
}

export const v3 = (x = 0, y = 0, z = 0): V3 => [x, y, z];
export const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const norm = (a: V3): number => Math.hypot(a[0], a[1], a[2]);
export const cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

export const qIdentity = (): Q => [0, 0, 0, 1];

export function qmul(a: Q, b: Q): Q {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

export const qconj = (q: Q): Q => [-q[0], -q[1], -q[2], q[3]];

export function qnormalize(q: Q): Q {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

export function qrot(q: Q, v: V3): V3 {
  // v' = v + 2w(u x v) + 2 u x (u x v), u = q.xyz
  const u: V3 = [q[0], q[1], q[2]];
  const w = q[3];
  const t = scale(cross(u, v), 2);
  return add(add(v, scale(t, w)), cross(u, t));
}

export function qFromAxisAngle(axis: V3, angle: number): Q {
  const n = norm(axis) || 1;
  const s = Math.sin(angle / 2) / n;
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)];
}

/** Shortest-arc slerp, a transcription of robosimstudio.core.math.quat_slerp. */
export function qSlerp(a: Q, bIn: Q, t: number): Q {
  let b = bIn;
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  if (d < 0) {
    b = [-b[0], -b[1], -b[2], -b[3]];
    d = -d;
  }
  let q: Q;
  if (d > 0.9995) {
    q = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]), a[3] + t * (b[3] - a[3])];
  } else {
    const th = Math.acos(Math.min(1, Math.max(-1, d)));
    const s0 = Math.sin((1 - t) * th) / Math.sin(th);
    const s1 = Math.sin(t * th) / Math.sin(th);
    q = [s0 * a[0] + s1 * b[0], s0 * a[1] + s1 * b[1], s0 * a[2] + s1 * b[2], s0 * a[3] + s1 * b[3]];
  }
  return qnormalize(q);
}

/** Rotation vector (axis * angle) of a unit quaternion, shortest arc. */
export function qLog(qIn: Q): V3 {
  const q = qIn[3] < 0 ? ([-qIn[0], -qIn[1], -qIn[2], -qIn[3]] as Q) : qIn;
  const s = Math.hypot(q[0], q[1], q[2]);
  if (s < 1e-12) return [2 * q[0], 2 * q[1], 2 * q[2]];
  const ang = 2 * Math.atan2(s, q[3]);
  return [(q[0] / s) * ang, (q[1] / s) * ang, (q[2] / s) * ang];
}

export function qExp(v: V3): Q {
  const ang = norm(v);
  if (ang < 1e-12) return qnormalize([v[0] / 2, v[1] / 2, v[2] / 2, 1]);
  return qFromAxisAngle(v, ang);
}

export const tf = (p: V3 = [0, 0, 0], q: Q = [0, 0, 0, 1]): Tf => ({ p, q });

/** a ∘ b: first b, then a. */
export function tfMul(a: Tf, b: Tf): Tf {
  return { p: add(a.p, qrot(a.q, b.p)), q: qmul(a.q, b.q) };
}

export function tfInv(a: Tf): Tf {
  const qi = qconj(a.q);
  return { p: scale(qrot(qi, a.p), -1), q: qi };
}

/** ``out = a ∘ b`` without allocating; ``out`` may alias ``a`` or ``b``. Returns ``out``. */
export function tfMulTo(out: Tf, a: Tf, b: Tf): Tf {
  const ax = a.q[0], ay = a.q[1], az = a.q[2], aw = a.q[3];
  const bx = b.q[0], by = b.q[1], bz = b.q[2], bw = b.q[3];
  const vx = b.p[0], vy = b.p[1], vz = b.p[2];
  // qrot(a.q, b.p): t = 2 (u x v); v' = v + w t + u x t
  const tx = 2 * (ay * vz - az * vy), ty = 2 * (az * vx - ax * vz), tz = 2 * (ax * vy - ay * vx);
  out.p[0] = a.p[0] + vx + aw * tx + (ay * tz - az * ty);
  out.p[1] = a.p[1] + vy + aw * ty + (az * tx - ax * tz);
  out.p[2] = a.p[2] + vz + aw * tz + (ax * ty - ay * tx);
  out.q[0] = aw * bx + ax * bw + ay * bz - az * by;
  out.q[1] = aw * by - ax * bz + ay * bw + az * bx;
  out.q[2] = aw * bz + ax * by - ay * bx + az * bw;
  out.q[3] = aw * bw - ax * bx - ay * by - az * bz;
  return out;
}

export const tfFrom7 = (a: ArrayLike<number>, o = 0): Tf => ({
  p: [a[o], a[o + 1], a[o + 2]],
  q: [a[o + 3], a[o + 4], a[o + 5], a[o + 6]],
});

/** ``target``, or the point ``limit`` from ``origin`` towards it (osc._step_toward). */
export function stepToward(origin: V3, target: V3, limit: number): V3 {
  const off = sub(target, origin);
  const d = norm(off);
  if (d <= limit || d < 1e-12) return [target[0], target[1], target[2]];
  return add(origin, scale(off, limit / d));
}

/** ``target``, or the orientation ``limit`` rad from ``origin`` towards it (osc._turn_toward). */
export function turnToward(origin: Q, target: Q, limit: number): Q {
  const d = Math.abs(origin[0] * target[0] + origin[1] * target[1] + origin[2] * target[2] + origin[3] * target[3]);
  const angle = 2 * Math.acos(Math.min(1, Math.max(-1, d)));
  if (angle <= limit || angle < 1e-9) return [target[0], target[1], target[2], target[3]];
  return qSlerp(origin, target, limit / angle);
}
