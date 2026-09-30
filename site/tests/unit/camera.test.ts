import { describe, expect, it } from "vitest";
import { Camera } from "../../src/core/camera.ts";
import type { V3 } from "../../src/core/math.ts";
import { lookAt, rng } from "./helpers.ts";

const eye: V3 = [1.2, -1.5, 1.1];
const cam = new Camera(lookAt(eye, [0.1, 0.2, 0.75]));
const u = rng(8);

describe("Camera", () => {
  it("projects a point on the optical axis to the principal point", () => {
    const z = cam.ray(cam.cx, cam.cy).d;
    const [px, py, depth] = cam.project([eye[0] + 2.5 * z[0], eye[1] + 2.5 * z[1], eye[2] + 2.5 * z[2]]);
    expect(px).toBeCloseTo(cam.cx, 9);
    expect(py).toBeCloseTo(cam.cy, 9);
    expect(depth).toBeCloseTo(2.5, 12);
  });

  it("round-trips project(ray(u, v) at depth d)", () => {
    const axis = cam.ray(cam.cx, cam.cy).d;
    for (let k = 0; k < 200; k++) {
      const [pu, pv, d] = [u() * 1920, u() * 1080, 0.2 + u() * 5];
      const { o, d: dir } = cam.ray(pu, pv);
      expect(Math.hypot(...dir)).toBeCloseTo(1, 14);
      const s = d / (dir[0] * axis[0] + dir[1] * axis[1] + dir[2] * axis[2]); // the ray reaches depth d here
      const p = cam.project([o[0] + s * dir[0], o[1] + s * dir[1], o[2] + s * dir[2]]);
      expect(Math.abs(p[0] - pu)).toBeLessThan(1e-6);
      expect(Math.abs(p[1] - pv)).toBeLessThan(1e-6);
      expect(p[2]).toBeCloseTo(d, 10);
    }
  });

  it("builds GL matrices that reproduce the CV projection inside a crop", () => {
    const crop = [700, 120, 608, 900]; // a phone-portrait ROI
    const [near, far] = [0.05, 20];
    const { projection: P, cameraToWorld: M } = cam.glMatrices(crop, near, far);
    // view = inverse of the rigid GL camera-to-world
    const R = [M[0], M[1], M[2], M[4], M[5], M[6], M[8], M[9], M[10]]; // columns: GL x, y, z axes
    const t = [M[12], M[13], M[14]];
    for (let k = 0; k < 200; k++) {
      const [pu, pv, d] = [crop[0] + u() * crop[2], crop[1] + u() * crop[3], 0.3 + u() * 4];
      const { o, d: dir } = cam.ray(pu, pv);
      const w: V3 = [o[0] + d * dir[0], o[1] + d * dir[1], o[2] + d * dir[2]];
      const rel = [w[0] - t[0], w[1] - t[1], w[2] - t[2]];
      const c = [0, 1, 2].map((a) => R[3 * a] * rel[0] + R[3 * a + 1] * rel[1] + R[3 * a + 2] * rel[2]);
      const clip = [0, 1, 2, 3].map((r) => P[r] * c[0] + P[4 + r] * c[1] + P[8 + r] * c[2] + P[12 + r]);
      const [nx, ny, nz] = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]];
      const cv = cam.project(w);
      expect(Math.abs(crop[0] + ((nx + 1) / 2) * crop[2] - cv[0])).toBeLessThan(1e-6);
      expect(Math.abs(crop[1] + ((1 - ny) / 2) * crop[3] - cv[1])).toBeLessThan(1e-6);
      expect(Math.abs(nz)).toBeLessThan(1);
      expect(clip[3]).toBeCloseTo(cv[2], 12); // w_clip is the CV depth
    }
  });

  it("rejects a malformed extrinsic", () => {
    expect(() => new Camera({ ...lookAt(eye, [0, 0, 0]), cam_to_world_cv: [1, 0, 0] })).toThrow(/16 numbers/);
  });
});
