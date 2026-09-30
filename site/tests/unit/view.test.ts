import { describe, expect, it } from "vitest";
import { View, VIEW } from "../../src/core/view.ts";

// A hero camera 1.2 m out along +x and 0.8 m up, looking at (0, 0, 0.9) with +z up.
function hero() {
  const eye = [1.2, 0, 1.7];
  const at = [0, 0, 0.9];
  const f = [at[0] - eye[0], at[1] - eye[1], at[2] - eye[2]];
  const fn = Math.hypot(f[0], f[1], f[2]);
  const z = f.map((v) => v / fn); // forward
  const up = [0, 0, 1];
  const xr = [z[1] * up[2] - z[2] * up[1], z[2] * up[0] - z[0] * up[2], z[0] * up[1] - z[1] * up[0]]; // forward x up = right
  const xn = Math.hypot(xr[0], xr[1], xr[2]);
  const x = xr.map((v) => v / xn);
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]; // down
  const m = [x[0], x[1], x[2], 0, y[0], y[1], y[2], 0, z[0], z[1], z[2], 0, eye[0], eye[1], eye[2], 1];
  return { width: 1920, height: 1080, fx: 1400, fy: 1400, cx: 959.5, cy: 539.5, cam_to_world_cv: m, look_at: at };
}

const elevation = (v: View, pivot: number[]) => {
  const m = v.camToWorld();
  const d = [m[12] - pivot[0], m[13] - pivot[1], m[14] - pivot[2]];
  return Math.asin(d[2] / Math.hypot(d[0], d[1], d[2]));
};

describe("View", () => {
  it("is the hero camera, exactly, at home", () => {
    const v = new View();
    const h = hero();
    v.setHero(h);
    expect(v.home).toBe(true);
    expect(Array.from(v.camToWorld())).toEqual(h.cam_to_world_cv);
  });

  it("orbits about the look-at point: it stays in the middle of the frame", () => {
    const v = new View();
    v.setHero(hero());
    v.orbit(150, -40, 0);
    v.dolly(1.3, 0);
    expect(v.home).toBe(false);
    const p = v.camera.project([0, 0, 0.9]);
    expect(p[0]).toBeCloseTo(959.5, 6);
    expect(p[1]).toBeCloseTo(539.5, 6);
    const m = v.camToWorld();
    const r0 = Math.hypot(1.2, 0, 0.8);
    expect(Math.hypot(m[12], m[13], m[14] - 0.9)).toBeCloseTo(r0 * 1.3, 9);
  });

  it("drag down raises the eye, and the elevation stays short of the floor and the pole", () => {
    const v = new View();
    v.setHero(hero());
    const e0 = elevation(v, [0, 0, 0.9]);
    v.orbit(0, 50, 0);
    expect(elevation(v, [0, 0, 0.9])).toBeGreaterThan(e0);
    v.orbit(0, 1e5, 0);
    expect(elevation(v, [0, 0, 0.9])).toBeCloseTo(VIEW.elevMax, 9);
    v.orbit(0, -1e5, 0);
    expect(elevation(v, [0, 0, 0.9])).toBeCloseTo(VIEW.elevMin, 9);
  });

  it("glides home after the idle time and lands on the hero pose exactly", () => {
    const v = new View();
    const h = hero();
    v.setHero(h);
    v.orbit(900, 30, 0); // more than half a turn: goes back the short way
    let t = 0;
    const dt = 1 / 60;
    for (; t < (VIEW.idleHomeS - 0.5) * 1000; t += dt * 1000) v.update(t, dt);
    expect(v.home).toBe(false); // not before the idle time
    for (let k = 0; k < 600 && !v.home; k++, t += dt * 1000) v.update(t, dt);
    expect(v.home).toBe(true);
    expect(Array.from(v.camToWorld())).toEqual(h.cam_to_world_cv);
  });

  it("any input holds the view and stops a glide", () => {
    const v = new View();
    v.setHero(hero());
    v.orbit(100, 0, 0);
    const dt = 1 / 60;
    let t = VIEW.idleHomeS * 1000 + 1;
    v.update(t, dt); // the glide starts
    const yaw = v.yaw;
    v.touch(t);
    for (let k = 0; k < 60; k++) v.update((t += dt * 1000), dt);
    expect(v.yaw).toBe(yaw);
  });

  it("fades the live layer in off home, and out only once back", () => {
    const v = new View();
    v.setHero(hero());
    v.orbit(20, 0, 0);
    v.update(0, 0.1);
    v.update(0, 0.1);
    expect(v.look).toBe(1);
    v.goHome();
    let k = 0;
    for (; k < 200; k++) {
      v.update(0, 1 / 60);
      if (v.home) break;
      expect(v.look).toBe(1); // fully live all the way back
    }
    expect(v.home).toBe(true);
    const arrived = 1 - 1 / 60 / VIEW.lookOutS; // the fade starts on the frame it lands
    expect(v.look).toBeCloseTo(arrived, 9);
    v.update(0, 0.1);
    expect(v.look).toBeCloseTo(arrived - 0.1 / VIEW.lookOutS, 9);
  });

  it("focuses on the point of the optical axis nearest the action", () => {
    const v = new View();
    v.setHero(hero());
    v.setFocus([0.3, 0.2, 1.0]);
    v.orbit(80, 20, 0);
    const m = v.camToWorld();
    // the new pivot projects to the middle of the frame
    const h = hero().cam_to_world_cv;
    const f = [h[8], h[9], h[10]];
    const along = (0.3 - 1.2) * f[0] + (0.2 - 0) * f[1] + (1.0 - 1.7) * f[2];
    const pivot = [1.2 + f[0] * along, 0 + f[1] * along, 1.7 + f[2] * along];
    const p = v.camera.project(pivot);
    expect(p[0]).toBeCloseTo(959.5, 6);
    expect(p[1]).toBeCloseTo(539.5, 6);
    expect(m[15]).toBe(1);
  });
});
