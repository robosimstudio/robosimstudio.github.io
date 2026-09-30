// The robot's Cartesian servo law, ported exactly from
// robosimstudio/solvers/motion/osc.py: OscSetpoint.advance steps the setpoint
// toward the goal at pos_speed, THEN clamps it to within pos_lead of the
// measured hand (the order matters: a blocked arm must not creep).
//
// The browser uses it to predict where the visitor's drag is taking the hand
// over the next ~0.33 s: the "desired future" the matcher looks for.

import { type Q, type V3, stepToward, turnToward } from "./math.ts";

export interface OscConstants {
  control_dt: number;
  pos_speed: number;
  pos_lead: number;
  rot_speed: number;
  rot_lead: number;
  /** Optional pre-limit on the goal itself [m/s] (the peel caps the hand at 50 mm/s). */
  pre_limit?: number;
}

export class OscSetpoint {
  c: OscConstants;
  p: V3 = [0, 0, 0];
  q: Q = [0, 0, 0, 1];
  constructor(c: OscConstants) {
    this.c = c;
  }
  reset(p: V3, q: Q): void {
    this.p = [p[0], p[1], p[2]];
    this.q = [q[0], q[1], q[2], q[3]];
  }
  advance(measP: V3, measQ: Q, goalP: V3, goalQ: Q): { p: V3; q: Q } {
    const dt = this.c.control_dt;
    let p = stepToward(this.p, goalP, this.c.pos_speed * dt);
    let q = turnToward(this.q, goalQ, this.c.rot_speed * dt);
    p = stepToward(measP, p, this.c.pos_lead);
    q = turnToward(measQ, q, this.c.rot_lead);
    this.p = p;
    this.q = q;
    return { p, q };
  }
}

/**
 * The arm's response to its setpoint, fitted offline per demo (``rsw db``):
 *   x_{k+1} = x_k + alpha (s_k - x_k) + beta (x_k - x_{k-1})
 * A second-order lag that is good enough to predict 20 control steps ahead.
 */
export interface Plant {
  alpha: number;
  beta: number;
}

export interface Prediction {
  /** Hand displacement after each horizon (control steps), relative to now. */
  disp: V3[];
  /** Hand position at the last horizon. */
  end: V3;
}

/**
 * Roll the servo + plant forward from the current hand toward a moving goal.
 * ``goalAt(k)`` gives the goal k control steps from now (a dragged target is
 * extrapolated at its velocity; a coasting one follows the coast law).
 */
export function predict(
  c: OscConstants,
  plant: Plant,
  setpoint: V3,
  hand: V3,
  handPrev: V3,
  goalAt: (k: number) => V3,
  horizons: number[],
): Prediction {
  let s: V3 = [setpoint[0], setpoint[1], setpoint[2]];
  let x: V3 = [hand[0], hand[1], hand[2]];
  let xp: V3 = [handPrev[0], handPrev[1], handPrev[2]];
  const dt = c.control_dt;
  const last = Math.max(...horizons);
  const disp: V3[] = [];
  let goalPrev: V3 | null = null;
  for (let k = 1; k <= last; k++) {
    let g = goalAt(k);
    if (c.pre_limit !== undefined) {
      // the goal itself is rate-limited (peel): step it from the previous goal
      g = stepToward(goalPrev ?? s, g, c.pre_limit * dt);
      goalPrev = g;
    }
    s = stepToward(s, g, c.pos_speed * dt);
    s = stepToward(x, s, c.pos_lead);
    const xn: V3 = [
      x[0] + plant.alpha * (s[0] - x[0]) + plant.beta * (x[0] - xp[0]),
      x[1] + plant.alpha * (s[1] - x[1]) + plant.beta * (x[1] - xp[1]),
      x[2] + plant.alpha * (s[2] - x[2]) + plant.beta * (x[2] - xp[2]),
    ];
    xp = x;
    x = xn;
    if (horizons.includes(k)) disp.push([x[0] - hand[0], x[1] - hand[1], x[2] - hand[2]]);
  }
  return { disp, end: x };
}
