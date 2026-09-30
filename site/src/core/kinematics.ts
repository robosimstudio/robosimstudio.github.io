// Forward kinematics of a demo's robot from kinematics.<h>.json (CONTRACTS.md §2),
// with Newton's semantics: for each joint, in the file's (topological) order,
//
//   pose[child] = pose[parent] ∘ X_p ∘ X_j(q) ∘ X_c⁻¹
//
// X_j is a rotation about `axis` by q[q_start] (revolute), a translation axis * q[q_start]
// (prismatic), identity (fixed), q[q_start..+7] as a pose (free) or q[q_start..+4] as an
// xyzw rotation (ball). A parent of -1 is the world; a parent FK does not produce takes its
// world pose from `roots`. The DB stores only the robot's coordinates (`q_index` into
// Newton's joint_q); every other coordinate keeps its canonical value from `q_full`.

import { type Q, type Tf, type V3, qconj, qmul, tf, tfFrom7, tfInv, tfMulTo } from "./math.ts";

export type JointType = "revolute" | "prismatic" | "fixed" | "free" | "ball";

export interface KinJointJson {
  type: JointType;
  parent: number;
  child: number;
  X_p: number[];
  X_c: number[];
  axis: number[];
  q_start: number;
}

export interface KinematicsJson {
  q_index: number[];
  q_full: number[];
  joints: KinJointJson[];
  bodies: { index: number; label: string }[];
  roots: Record<string, number[]>;
  tcp: Record<string, { body: number; offset: number[] }>;
  tests: { q: number[]; poses: Record<string, number[]> }[];
}

const COORDS: Record<JointType, number> = { revolute: 1, prismatic: 1, fixed: 0, free: 7, ball: 4 };

interface Joint {
  type: JointType;
  child: number;
  /** True when an earlier joint produces the parent: its pose is read from the output map. */
  produced: boolean;
  parent: number;
  /** World pose of a parent FK does not produce (identity for -1). */
  fixedParent: Tf;
  Xp: Tf;
  XcInv: Tf;
  axis: V3;
  qStart: number;
}

export class Kinematics {
  readonly qIndex: Int32Array;
  /** Canonical full joint_q; entries outside ``qIndex`` keep these values. */
  readonly qFull: Float64Array;
  readonly bodies: { index: number; label: string }[];
  readonly roots: Map<number, Tf>;
  readonly tests: KinematicsJson["tests"];
  private readonly joints: Joint[];
  private readonly tcps: Map<string, { body: number; offset: Tf }>;
  private readonly q: Float64Array;
  private readonly xj: Tf = tf();
  private readonly tmp: Tf = tf();

  constructor(json: KinematicsJson) {
    this.qIndex = Int32Array.from(json.q_index);
    this.qFull = Float64Array.from(json.q_full);
    this.q = new Float64Array(this.qFull.length);
    this.bodies = json.bodies;
    this.tests = json.tests ?? [];
    this.roots = new Map(Object.entries(json.roots ?? {}).map(([k, v]) => [Number(k), tfFrom7(v)]));
    this.tcps = new Map(Object.entries(json.tcp ?? {}).map(([k, v]) => [k, { body: v.body, offset: tfFrom7(v.offset) }]));
    for (const k of this.qIndex)
      if (k < 0 || k >= this.qFull.length) throw new Error(`kinematics: q_index entry ${k} is outside q_full (${this.qFull.length})`);

    const produced = new Set<number>();
    this.joints = json.joints.map((j, i) => {
      if (!Object.hasOwn(COORDS, j.type)) throw new Error(`kinematics: joint ${i} has unknown type ${JSON.stringify(j.type)}`);
      if (j.q_start < 0 || j.q_start + COORDS[j.type] > this.qFull.length)
        throw new Error(`kinematics: joint ${i} (${j.type}) reads q[${j.q_start}..+${COORDS[j.type]}] outside q_full`);
      const isProduced = produced.has(j.parent);
      if (!isProduced && j.parent !== -1 && !this.roots.has(j.parent))
        throw new Error(`kinematics: joint ${i}'s parent body ${j.parent} is neither produced by an earlier joint nor in roots`);
      if (produced.has(j.child)) throw new Error(`kinematics: body ${j.child} is the child of two joints`);
      produced.add(j.child);
      const n = Math.hypot(j.axis[0], j.axis[1], j.axis[2]) || 1;
      return {
        type: j.type,
        child: j.child,
        produced: isProduced,
        parent: j.parent,
        fixedParent: isProduced || j.parent === -1 ? tf() : (this.roots.get(j.parent) as Tf),
        Xp: tfFrom7(j.X_p),
        XcInv: tfInv(tfFrom7(j.X_c)),
        axis: [j.axis[0] / n, j.axis[1] / n, j.axis[2] / n] as V3,
        qStart: j.q_start,
      };
    });
    for (const [arm, t] of this.tcps)
      if (!produced.has(t.body)) throw new Error(`kinematics: tcp "${arm}" hangs off body ${t.body}, which FK does not produce`);
  }

  /**
   * World pose of every body the joints produce, for the robot coordinates ``qRobot``
   * (order ``q_index``). Pass the previous result as ``out`` to reuse its poses:
   * the per-frame call then allocates nothing.
   */
  fk(qRobot: ArrayLike<number>, out: Map<number, Tf> = new Map()): Map<number, Tf> {
    if (qRobot.length !== this.qIndex.length)
      throw new Error(`kinematics: ${qRobot.length} robot coordinates, expected ${this.qIndex.length}`);
    const q = this.q;
    q.set(this.qFull);
    for (let k = 0; k < this.qIndex.length; k++) q[this.qIndex[k]] = qRobot[k];
    const xj = this.xj;
    const tmp = this.tmp;
    for (const j of this.joints) {
      jointTransform(j, q, xj);
      const parent = j.produced ? (out.get(j.parent) as Tf) : j.fixedParent;
      tfMulTo(tmp, parent, j.Xp);
      tfMulTo(tmp, tmp, xj);
      let child = out.get(j.child);
      if (!child) {
        child = tf();
        out.set(j.child, child);
      }
      tfMulTo(child, tmp, j.XcInv);
    }
    return out;
  }

  /** Tool centre of ``arm``: pose[body] ∘ offset, from poses returned by ``fk``. */
  tcp(arm: string, poses: Map<number, Tf>, out: Tf = tf()): Tf {
    const t = this.tcps.get(arm);
    if (!t) throw new Error(`kinematics: no tcp "${arm}" (have ${[...this.tcps.keys()].join(", ")})`);
    const body = poses.get(t.body);
    if (!body) throw new Error(`kinematics: poses lack body ${t.body} for tcp "${arm}"`);
    return tfMulTo(out, body, t.offset);
  }

  /** Worst position [m] and rotation [rad] error of ``fk`` against the file's own test vectors. */
  selfTest(): { maxPosErr: number; maxAngErr: number } {
    let maxPosErr = 0;
    let maxAngErr = 0;
    const poses = new Map<number, Tf>();
    for (const t of this.tests) {
      this.fk(t.q, poses);
      for (const [key, ref7] of Object.entries(t.poses)) {
        const got = poses.get(Number(key)) ?? this.roots.get(Number(key));
        if (!got) throw new Error(`kinematics selfTest: body ${key} is neither produced by FK nor a root`);
        const ref = tfFrom7(ref7);
        maxPosErr = Math.max(maxPosErr, Math.hypot(got.p[0] - ref.p[0], got.p[1] - ref.p[1], got.p[2] - ref.p[2]));
        maxAngErr = Math.max(maxAngErr, quatAngle(got.q, ref.q));
      }
    }
    return { maxPosErr, maxAngErr };
  }
}

/** Angle [rad] of the rotation between two quaternions; atan2 keeps tiny angles accurate. */
export function quatAngle(a: Q, b: Q): number {
  const r = qmul(qconj(a), b);
  return 2 * Math.atan2(Math.hypot(r[0], r[1], r[2]), Math.abs(r[3]));
}

/** X_j(q) into ``out``. Free and ball rotations are renormalized: q16d leaves them slightly off unit. */
function jointTransform(j: Joint, q: Float64Array, out: Tf): void {
  const p = out.p;
  const r = out.q;
  const s = j.qStart;
  p[0] = p[1] = p[2] = 0;
  r[0] = r[1] = r[2] = 0;
  r[3] = 1;
  switch (j.type) {
    case "revolute": {
      const h = q[s] / 2;
      const sh = Math.sin(h);
      r[0] = j.axis[0] * sh;
      r[1] = j.axis[1] * sh;
      r[2] = j.axis[2] * sh;
      r[3] = Math.cos(h);
      break;
    }
    case "prismatic":
      p[0] = j.axis[0] * q[s];
      p[1] = j.axis[1] * q[s];
      p[2] = j.axis[2] * q[s];
      break;
    case "free":
      p[0] = q[s];
      p[1] = q[s + 1];
      p[2] = q[s + 2];
      unitQuat(q, s + 3, r);
      break;
    case "ball":
      unitQuat(q, s, r);
      break;
    case "fixed":
      break;
  }
}

function unitQuat(q: Float64Array, o: number, out: Q): void {
  const n = Math.hypot(q[o], q[o + 1], q[o + 2], q[o + 3]) || 1;
  out[0] = q[o] / n;
  out[1] = q[o + 1] / n;
  out[2] = q[o + 2] / n;
  out[3] = q[o + 3] / n;
}
