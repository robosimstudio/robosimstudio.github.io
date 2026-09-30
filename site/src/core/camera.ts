// The RTX render camera from manifest.camera (CONTRACTS.md §5): pinhole
// intrinsics in RTX pixels and `cam_to_world_cv`, a column-major 4x4 rigid
// transform in OpenCV convention (+x right, +y down, +z forward).
//
// The page draws its overlays with an OpenGL camera that must line up with the
// video pixel for pixel, including when it shows only a crop of the frame:
// `glMatrices` builds the off-axis frustum whose NDC square is exactly that crop.

import type { V3 } from "./math.ts";

export interface CameraJson {
  width: number;
  height: number;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  cam_to_world_cv: number[];
}

export class Camera {
  readonly width: number;
  readonly height: number;
  readonly fx: number;
  readonly fy: number;
  readonly cx: number;
  readonly cy: number;
  /** CV camera-to-world, column-major. */
  readonly camToWorld: Float64Array;
  /** Camera x, y, z axes in world (the rotation's columns) and the camera centre. */
  private readonly ax: V3;
  private readonly ay: V3;
  private readonly az: V3;
  private readonly c: V3;

  constructor(cam: CameraJson) {
    if (cam.cam_to_world_cv?.length !== 16) throw new Error("Camera: cam_to_world_cv must hold 16 numbers");
    this.width = cam.width;
    this.height = cam.height;
    this.fx = cam.fx;
    this.fy = cam.fy;
    this.cx = cam.cx;
    this.cy = cam.cy;
    const m = Float64Array.from(cam.cam_to_world_cv);
    this.camToWorld = m;
    this.ax = [m[0], m[1], m[2]];
    this.ay = [m[4], m[5], m[6]];
    this.az = [m[8], m[9], m[10]];
    this.c = [m[12], m[13], m[14]];
  }

  /** The camera centre in world. */
  get center(): V3 {
    return [this.c[0], this.c[1], this.c[2]];
  }

  /** A world point in the CV camera frame (the rotation is orthonormal: its inverse is its transpose). */
  toCam(world: ArrayLike<number>, out: V3 = [0, 0, 0]): V3 {
    return this.rotT(world[0] - this.c[0], world[1] - this.c[1], world[2] - this.c[2], out);
  }

  /** A world direction in the CV camera frame. */
  dirToCam(dir: ArrayLike<number>, out: V3 = [0, 0, 0]): V3 {
    return this.rotT(dir[0], dir[1], dir[2], out);
  }

  private rotT(x: number, y: number, z: number, out: V3): V3 {
    const { ax, ay, az } = this;
    out[0] = ax[0] * x + ax[1] * y + ax[2] * z;
    out[1] = ay[0] * x + ay[1] * y + ay[2] * z;
    out[2] = az[0] * x + az[1] * y + az[2] * z;
    return out;
  }

  /** ``[u, v, depth]``: RTX pixel of a world point and its depth along the optical axis [m]. */
  project(world: ArrayLike<number>, out: V3 = [0, 0, 0]): V3 {
    this.toCam(world, out);
    const z = out[2];
    out[0] = (this.fx * out[0]) / z + this.cx;
    out[1] = (this.fy * out[1]) / z + this.cy;
    return out;
  }

  /** The viewing ray through RTX pixel (u, v): origin the camera centre, ``d`` a unit world direction. */
  ray(u: number, v: number): { o: V3; d: V3 } {
    const x = (u - this.cx) / this.fx;
    const y = (v - this.cy) / this.fy;
    const { ax, ay, az } = this;
    const d: V3 = [ax[0] * x + ay[0] * y + az[0], ax[1] * x + ay[1] * y + az[1], ax[2] * x + ay[2] * y + az[2]];
    const n = Math.hypot(d[0], d[1], d[2]);
    return { o: this.center, d: [d[0] / n, d[1] / n, d[2] / n] };
  }

  /**
   * OpenGL matrices (column-major) that render exactly the RTX crop ``[x0, y0, w, h]``
   * onto the viewport: an off-axis frustum with
   *   left = (x0 - cx) n / fx, right = (x0 + w - cx) n / fx,
   *   top = (cy - y0) n / fy, bottom = (cy - y0 - h) n / fy,
   * and the GL camera-to-world cam_to_world_cv · diag(1, -1, -1, 1) (+y up, looking down -z).
   */
  glMatrices(crop: ArrayLike<number>, near: number, far: number): { projection: number[]; cameraToWorld: number[] } {
    const x0 = crop[0];
    const y0 = crop[1];
    const w = crop[2];
    const h = crop[3];
    const n = near;
    const f = far;
    const l = ((x0 - this.cx) * n) / this.fx;
    const r = ((x0 + w - this.cx) * n) / this.fx;
    const t = ((this.cy - y0) * n) / this.fy;
    const b = ((this.cy - y0 - h) * n) / this.fy;
    const projection = [
      (2 * n) / (r - l), 0, 0, 0,
      0, (2 * n) / (t - b), 0, 0,
      (r + l) / (r - l), (t + b) / (t - b), -(f + n) / (f - n), -1,
      0, 0, (-2 * f * n) / (f - n), 0,
    ];
    const m = this.camToWorld;
    const cameraToWorld = [
      m[0], m[1], m[2], m[3],
      -m[4], -m[5], -m[6], -m[7],
      -m[8], -m[9], -m[10], -m[11],
      m[12], m[13], m[14], m[15],
    ];
    return { projection, cameraToWorld };
  }
}
