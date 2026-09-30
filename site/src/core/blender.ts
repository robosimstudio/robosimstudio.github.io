// Inertialization: when playback jumps from one stream of frames to another, the
// difference between the two is kept as an offset on top of the new stream and
// decayed to zero with an exact critically damped spring (Holden's
// "spring-it-on" form), so the output keeps its value and velocity at the switch.
//
//   y = 2 ln2 / halflife
//   j1 = ov + y o
//   o'  = e^{-y dt} (o + j1 dt)
//   ov' = e^{-y dt} (ov - j1 y dt)
//   output = dst + o
//
// The decay is the closed-form solution, so it does not depend on how dt is sliced.

/** Default halflives [s] per channel group. */
export const HALFLIFE_JOINTS = 0.1;
export const HALFLIFE_HAND = 0.1;
export const HALFLIFE_PCA = 0.12;

export interface InertialSpec {
  dim: number;
  /** [s], default HALFLIFE_JOINTS. */
  halflife?: number;
  /** Per dimension: a continuous joint whose offset is wrapped to (-pi, pi]. */
  wrap?: ArrayLike<boolean>;
}

type Vec = ArrayLike<number>;

const TWO_PI = 2 * Math.PI;

/** ``x`` wrapped to (-pi, pi]. */
export const wrapPi = (x: number): number => x - TWO_PI * Math.ceil((x - Math.PI) / TWO_PI);

/** The decaying offset of one channel group (joints, hand, PCA coefficients...). */
export class Inertializer {
  readonly dim: number;
  readonly halflife: number;
  /** Position and velocity offsets: output = dst + o. */
  readonly o: Float64Array;
  readonly ov: Float64Array;
  private readonly wrap: Uint8Array | null;

  constructor(spec: InertialSpec) {
    this.dim = spec.dim;
    this.halflife = spec.halflife ?? HALFLIFE_JOINTS;
    this.o = new Float64Array(spec.dim);
    this.ov = new Float64Array(spec.dim);
    this.wrap = spec.wrap ? Uint8Array.from(spec.wrap, (w) => (w ? 1 : 0)) : null;
  }

  /**
   * Switch from the stream being left (``src``: its raw value and velocity at the switch)
   * to ``dst``. The pending offset is folded in, so a new transition re-bases from the
   * current output: o = (src + o) - dst, ov = (srcV + ov) - dstV.
   */
  transition(srcX: Vec, srcV: Vec, dstX: Vec, dstV: Vec): void {
    const { o, ov, wrap } = this;
    for (let d = 0; d < this.dim; d++) {
      const x = srcX[d] + o[d] - dstX[d];
      o[d] = wrap && wrap[d] ? wrapPi(x) : x;
      ov[d] = srcV[d] + ov[d] - dstV[d];
    }
  }

  /** Decay the offsets by ``dt`` seconds. */
  apply(dt: number): void {
    const { o, ov } = this;
    const y = (2 * Math.LN2) / this.halflife;
    const e = Math.exp(-y * dt);
    for (let d = 0; d < this.dim; d++) {
      const j1 = ov[d] + y * o[d];
      o[d] = e * (o[d] + j1 * dt);
      ov[d] = e * (ov[d] - j1 * y * dt);
    }
  }

  /** ``out = dstX + o``. */
  output<T extends { [i: number]: number }>(dstX: Vec, out: T): T {
    for (let d = 0; d < this.dim; d++) out[d] = dstX[d] + this.o[d];
    return out;
  }

  reset(): void {
    this.o.fill(0);
    this.ov.fill(0);
  }
}

/** A named set of channel groups that transition and decay together. */
export class Blender {
  readonly groups: Map<string, Inertializer>;

  constructor(specs: Record<string, InertialSpec>) {
    this.groups = new Map(Object.entries(specs).map(([name, s]) => [name, new Inertializer(s)]));
  }

  get(name: string): Inertializer {
    const g = this.groups.get(name);
    if (!g) throw new Error(`Blender: no channel group "${name}"`);
    return g;
  }

  transition(name: string, srcX: Vec, srcV: Vec, dstX: Vec, dstV: Vec): void {
    this.get(name).transition(srcX, srcV, dstX, dstV);
  }

  apply(dt: number): void {
    for (const g of this.groups.values()) g.apply(dt);
  }

  output<T extends { [i: number]: number }>(name: string, dstX: Vec, out: T): T {
    return this.get(name).output(dstX, out);
  }

  reset(): void {
    for (const g of this.groups.values()) g.reset();
  }
}
