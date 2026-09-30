// The wipe's ink field, integrated from the removal rates of the frames the page
// actually plays (CONTRACTS.md §3 `ink`), so a visitor sees their own stripe.
// Removal never depends on the ink except through the clamp at zero, which is
// what makes this replay exact (pipeline/rsw/ink.py integrates the same way):
//
//   rate = r0 * (exp(v/255 * ln(1 + rmax/r0)) - 1)   [1/s], v the stored u8 code
//   I    = max(0, I - rate * dt)                      (never increases)

/** Decoded removal rate [1/s] of code ``v`` (0 → 0, 255 → rmax). */
export function inkRate(v: number, r0: number, rmax: number): number {
  return r0 * (Math.exp((v / 255) * Math.log(1 + rmax / r0)) - 1);
}

export class InkField {
  readonly cells: number;
  /** Default step [s] for ``applyFrame``: one control step. */
  readonly dt: number;
  /** Intensity per cell, 1 = full ink, 0 = clean. */
  readonly I: Float32Array;
  /** Set whenever ``I`` changes; the renderer clears it after uploading. */
  changed = true;
  private readonly lut = new Float64Array(256);
  private lutR0 = Number.NaN;
  private lutRmax = Number.NaN;

  constructor(cells: number, dt: number) {
    this.cells = cells;
    this.dt = dt;
    this.I = new Float32Array(cells).fill(1);
  }

  /** One frame of removal: ``codes[offset + c]`` is cell c's rate code. */
  applyFrame(codes: Uint8Array, r0: number, rmax: number, dt = this.dt, offset = 0): void {
    if (!(dt >= 0)) throw new Error(`InkField: dt must be >= 0, got ${dt}`);
    if (r0 !== this.lutR0 || rmax !== this.lutRmax) {
      if (!(r0 > 0 && rmax >= 0)) throw new Error(`InkField: need r0 > 0 and rmax >= 0, got r0=${r0} rmax=${rmax}`);
      for (let v = 0; v < 256; v++) this.lut[v] = inkRate(v, r0, rmax);
      this.lutR0 = r0;
      this.lutRmax = rmax;
    }
    const { I, lut } = this;
    let changed = false;
    for (let c = 0; c < this.cells; c++) {
      const v = codes[offset + c];
      if (v === 0 || I[c] <= 0) continue;
      const next = I[c] - lut[v] * dt;
      I[c] = next > 0 ? next : 0;
      changed = true;
    }
    if (changed) this.changed = true;
  }

  /** Take over an attract frame's field: I = v / 255 (store.inkStateRow). */
  setFromState(row: Uint8Array): void {
    for (let c = 0; c < this.cells; c++) this.I[c] = row[c] / 255;
    this.changed = true;
  }

  /** Back to full ink. */
  reset(): void {
    this.I.fill(1);
    this.changed = true;
  }
}
