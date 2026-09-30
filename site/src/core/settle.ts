// When the visitor stops dragging, the scene should come to rest rather than keep
// matching into frames that are still in motion. This decides when that is:
//
//   ACTIVE → SETTLING  once the target has moved slower than 5 mm/s for 150 ms and the
//                      predicted hand displacement is under 4 mm, or when a coast ends;
//   SETTLING → ACTIVE  as soon as the target moves faster than 15 mm/s.
//
// The gap between the two speeds is hysteresis: a hand resting on the screen
// jitters around a few mm/s without flipping the mode.

export type SettleMode = "active" | "settling";

export interface SettleParams {
  /** [m/s] */
  enterSpeed: number;
  /** [s] */
  enterHold: number;
  /** [m] */
  enterDisp: number;
  /** [s] a target this long at rest settles even if the hand is still predicted to move
   * (a hand stuck short of it, with no frame that closes the gap, would never settle). */
  enterMaxHold: number;
  /** [m/s] */
  exitSpeed: number;
}

export const SETTLE_DEFAULTS: SettleParams = { enterSpeed: 0.005, enterHold: 0.15, enterDisp: 0.004, exitSpeed: 0.015, enterMaxHold: 0.6 };

export class Settle {
  mode: SettleMode = "active";
  readonly params: SettleParams;
  /** Time the target has been slower than enterSpeed [s]. */
  private slow = 0;

  constructor(params: Partial<SettleParams> = {}) {
    this.params = { ...SETTLE_DEFAULTS, ...params };
  }

  /**
   * One tick: ``targetSpeed`` [m/s] of the drag target, ``predictedDisp3`` the hand
   * displacement still to come (e.g. predict()'s last horizon) [m], ``dt`` [s].
   */
  update(targetSpeed: number, predictedDisp3: ArrayLike<number>, dt: number): SettleMode {
    const p = this.params;
    if (this.mode === "settling") {
      if (targetSpeed > p.exitSpeed) this.wake();
      return this.mode;
    }
    this.slow = targetSpeed < p.enterSpeed ? this.slow + dt : 0;
    // the 1e-9 absorbs the rounding of a hold summed from many small dt
    const held = this.slow >= p.enterHold - 1e-9;
    const quiet = Math.hypot(predictedDisp3[0], predictedDisp3[1], predictedDisp3[2]) < p.enterDisp;
    if ((held && quiet) || this.slow >= p.enterMaxHold - 1e-9) this.mode = "settling";
    return this.mode;
  }

  /** A release coast has run out: settle now. */
  coastEnded(): void {
    this.mode = "settling";
  }

  /** Back to active (a new drag). */
  wake(): void {
    this.mode = "active";
    this.slow = 0;
  }
}
