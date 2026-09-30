// The LIVE layer: raster moving parts over the RTX plate.
//
// What is drawn here is only what moves (robot links, carried objects, the
// deformable). Static geometry from the GLB is drawn twice, invisibly:
//   * a depth-only pre-pass, so a moving part behind the bar is hidden by it;
//   * a shadow catcher (ShadowMaterial), so a moving part darkens the plate
//     where its shadow falls. Static casters are off: the plate already holds
//     the static shadows, path traced.
// The camera is the RTX camera exactly (cx = W/2 - 0.5 makes the RTX frustum
// symmetric), so this canvas and the RTX frames line up pixel for pixel.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

export interface CameraJson {
  width: number;
  height: number;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  near: number;
  far: number;
  eye: number[];
  look_at: number[];
  up: number[];
  cam_to_world_cv: number[];
}

export interface LookJson {
  exposure?: number;
  key?: { pos: number[]; target: number[]; intensity: number; color?: number[] };
  hemi?: { sky: number[]; ground: number[]; intensity: number };
  env_intensity?: number;
  shadow_radius?: number;
  shadow_opacity?: number;
}

export interface DeformMaterial {
  color: number[]; // display sRGB 0..1
  finish: { roughness: number; metallic: number; opacity: number; uv_scale?: number[]; uv_offset?: number[] };
  /** The finish's colour / roughness textures (URLs), as RTX draws them: a colour texture
   * replaces the colour, a roughness texture the roughness constant (``finish_gltf`` rules). */
  map?: string;
  roughnessMap?: string;
  /** Live-only opacity (the look calibration's): RTX keeps a thin film's highlights at any
   * opacity, a raster blend scales them with it. */
  opacity?: number;
}

export class LiveRenderer {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(45, 16 / 9, 0.02, 30);
  readonly bodies = new Map<number, THREE.Object3D>();
  private staticRoot: THREE.Object3D | null = null;
  private key: THREE.DirectionalLight | null = null;
  private deform: { mesh: THREE.Mesh; pos: THREE.BufferAttribute; nrm: THREE.BufferAttribute; tris: Uint32Array } | null = null;
  private envTex: THREE.Texture | null = null;
  private hemi: THREE.HemisphereLight | null = null;
  private catchers: THREE.ShadowMaterial[] = [];
  width = 1920;
  height = 1080;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      premultipliedAlpha: true,
      powerPreference: "high-performance",
    });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    // RTX writes sRGB with no tone curve (rtx.TONEMAP_OPERATOR == "none").
    this.renderer.toneMapping = THREE.LinearToneMapping;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.camera.up.set(0, 0, 1);
  }

  setCamera(c: CameraJson): void {
    this.width = c.width;
    this.height = c.height;
    this.camera.fov = (2 * Math.atan(0.5 * c.height / c.fy) * 180) / Math.PI;
    this.camera.aspect = (c.width / c.height) * (c.fy / c.fx);
    this.camera.near = Math.max(0.01, c.near);
    this.camera.far = c.far;
    // Pose from the CV camera-to-world (x right, y down, z forward): three's
    // camera looks down -z with +y up, i.e. diag(1, -1, -1) on the CV basis.
    const m = c.cam_to_world_cv; // column-major
    const mat = new THREE.Matrix4().set(
      m[0], -m[4], -m[8], m[12],
      m[1], -m[5], -m[9], m[13],
      m[2], -m[6], -m[10], m[14],
      0, 0, 0, 1,
    );
    mat.decompose(this.camera.position, this.camera.quaternion, this.camera.scale);
    this.camera.updateProjectionMatrix();
    this.camera.updateMatrixWorld(true);
  }

  setLook(look: LookJson): void {
    this.renderer.toneMappingExposure = look.exposure ?? 1.0;
    const pm = new THREE.PMREMGenerator(this.renderer);
    this.envTex = pm.fromScene(new RoomEnvironment(), 0.04).texture;
    pm.dispose();
    this.scene.environment = this.envTex;
    this.scene.environmentIntensity = look.env_intensity ?? 0.6;
    this.scene.environmentRotation.set(Math.PI / 2, 0, 0); // RoomEnvironment is Y-up; the world is Z-up
    const h = look.hemi ?? { sky: [0.94, 0.95, 0.97], ground: [0.35, 0.34, 0.33], intensity: 0.9 };
    const hemi = new THREE.HemisphereLight(new THREE.Color(...(h.sky as [number, number, number])),
      new THREE.Color(...(h.ground as [number, number, number])), h.intensity);
    hemi.position.set(0, 0, 1);
    this.scene.add(hemi);
    this.hemi = hemi;
    const k = look.key ?? { pos: [0.5, 0, 2.6], target: [0.5, 0, 0.9], intensity: 2.2 };
    const key = new THREE.DirectionalLight(new THREE.Color(...((k.color ?? [1, 0.97, 0.92]) as [number, number, number])), k.intensity);
    key.position.set(k.pos[0], k.pos[1], k.pos[2]);
    key.target.position.set(k.target[0], k.target[1], k.target[2]);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.radius = look.shadow_radius ?? 4;
    key.shadow.bias = -0.0004;
    key.shadow.normalBias = 0.01;
    const sc = key.shadow.camera as THREE.OrthographicCamera;
    sc.left = -0.8;
    sc.right = 0.8;
    sc.top = 0.8;
    sc.bottom = -0.8;
    sc.near = 0.1;
    sc.far = 5;
    this.scene.add(key, key.target);
    this.key = key;
  }

  async loadGlb(url: string, nodes: Record<string, number>, shadowOpacity = 0.35): Promise<void> {
    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);
    const gltf = await loader.loadAsync(url);
    const root = gltf.scene;
    // glTF loaders assume Y-up; this file is Z-up world coordinates with no
    // axis conversion, and the camera is Z-up too, so nothing is rotated.
    this.scene.add(root);
    for (const [name, body] of Object.entries(nodes)) {
      // GLTFLoader sanitizes node names (drops ":" among others): "body:2" is "body2"
      const n = root.getObjectByName(THREE.PropertyBinding.sanitizeNodeName(name));
      if (!n) continue;
      n.matrixAutoUpdate = false;
      n.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.isMesh) {
          m.castShadow = true;
          m.receiveShadow = true;
          m.frustumCulled = false;
        }
      });
      this.bodies.set(body, n);
    }
    const stat = root.getObjectByName("static");
    if (stat) {
      this.staticRoot = stat;
      const meshes: THREE.Mesh[] = [];
      stat.traverse((o) => {
        if ((o as THREE.Mesh).isMesh) meshes.push(o as THREE.Mesh);
      });
      for (const m of meshes) {
        // depth-only: occludes moving parts, draws nothing over the plate
        m.material = new THREE.MeshBasicMaterial({ colorWrite: false });
        m.renderOrder = -10;
        m.castShadow = false;
        m.receiveShadow = false;
        // shadow catcher: only moving parts cast, so it darkens the plate under them
        const sm = new THREE.ShadowMaterial({ opacity: shadowOpacity, depthWrite: false });
        this.catchers.push(sm);
        const c = new THREE.Mesh(m.geometry, sm);
        c.receiveShadow = true;
        c.renderOrder = 10;
        c.position.copy(m.position);
        c.quaternion.copy(m.quaternion);
        c.scale.copy(m.scale);
        m.parent?.add(c);
      }
    }
  }

  /** Retune the lights without rebuilding the scene (look calibration). */
  applyLook(look: LookJson): void {
    if (look.exposure !== undefined) this.renderer.toneMappingExposure = look.exposure;
    if (look.env_intensity !== undefined) this.scene.environmentIntensity = look.env_intensity;
    if (look.hemi && this.hemi) this.hemi.intensity = look.hemi.intensity;
    if (look.key && this.key) this.key.intensity = look.key.intensity;
    if (look.shadow_radius !== undefined && this.key) this.key.shadow.radius = look.shadow_radius;
    if (look.shadow_opacity !== undefined) for (const c of this.catchers) c.opacity = look.shadow_opacity;
  }

  /** A deformable surface: indexed triangles over ``vertexCount`` vertices, positions set per frame. */
  addDeformable(tris: Uint32Array, uv: Float32Array, vertexCount: number, mat: DeformMaterial): void {
    const g = new THREE.BufferGeometry();
    const pos = new THREE.BufferAttribute(new Float32Array(vertexCount * 3), 3);
    pos.setUsage(THREE.DynamicDrawUsage);
    const nrm = new THREE.BufferAttribute(new Float32Array(vertexCount * 3), 3);
    nrm.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute("position", pos);
    g.setAttribute("normal", nrm);
    g.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
    g.setIndex(new THREE.BufferAttribute(tris, 1));
    const color = new THREE.Color().setRGB(mat.color[0], mat.color[1], mat.color[2], THREE.SRGBColorSpace);
    const opacity = mat.opacity ?? mat.finish.opacity;
    const m = new THREE.MeshStandardMaterial({
      color: mat.map ? new THREE.Color(1, 1, 1) : color,
      roughness: mat.roughnessMap ? 1.0 : mat.finish.roughness,
      metalness: mat.finish.metallic,
      transparent: opacity < 1,
      opacity,
      side: THREE.DoubleSide,
    });
    const tiled = (url: string, srgb: boolean): THREE.Texture => {
      const t = new THREE.TextureLoader().load(url, () => (m.needsUpdate = true));
      t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      const [su, sv] = mat.finish.uv_scale ?? [1, 1];
      const [ou, ov] = mat.finish.uv_offset ?? [0, 0];
      t.repeat.set(su, sv);
      t.offset.set(ou, ov);
      t.anisotropy = 4;
      return t;
    };
    if (mat.map) m.map = tiled(mat.map, true);
    if (mat.roughnessMap) m.roughnessMap = tiled(mat.roughnessMap, false);
    const mesh = new THREE.Mesh(g, m);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.frustumCulled = false;
    this.scene.add(mesh);
    this.deform = { mesh, pos, nrm, tris };
  }

  /** World positions [V*3] of the deformable; normals area-weighted as rtx._vertex_normals. */
  setDeformable(world: Float32Array): void {
    const d = this.deform;
    if (!d) return;
    (d.pos.array as Float32Array).set(world);
    const n = d.nrm.array as Float32Array;
    n.fill(0);
    const t = d.tris;
    for (let i = 0; i < t.length; i += 3) {
      const a = t[i] * 3, b = t[i + 1] * 3, c = t[i + 2] * 3;
      const e1x = world[b] - world[a], e1y = world[b + 1] - world[a + 1], e1z = world[b + 2] - world[a + 2];
      const e2x = world[c] - world[a], e2y = world[c + 1] - world[a + 1], e2z = world[c + 2] - world[a + 2];
      const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
      for (const v of [a, b, c]) {
        n[v] += nx;
        n[v + 1] += ny;
        n[v + 2] += nz;
      }
    }
    for (let i = 0; i < n.length; i += 3) {
      const l = Math.hypot(n[i], n[i + 1], n[i + 2]) || 1;
      n[i] /= l;
      n[i + 1] /= l;
      n[i + 2] /= l;
    }
    d.pos.needsUpdate = true;
    d.nrm.needsUpdate = true;
  }

  private tube: {
    mesh: THREE.Mesh;
    pos: THREE.BufferAttribute;
    nrm: THREE.BufferAttribute;
    radius: number;
    ring: number;
    samples: number;
    pts: Float32Array;
  } | null = null;

  /** A cable: a tube of ``radius`` through ``n`` centreline points, rebuilt per frame. */
  addTube(n: number, radius: number, color: number[], roughness = 0.45): void {
    const per = 4; // samples per segment, Catmull-Rom
    const samples = (n - 1) * per + 1;
    const ring = 10;
    const g = new THREE.BufferGeometry();
    const pos = new THREE.BufferAttribute(new Float32Array(samples * ring * 3), 3);
    const nrm = new THREE.BufferAttribute(new Float32Array(samples * ring * 3), 3);
    pos.setUsage(THREE.DynamicDrawUsage);
    nrm.setUsage(THREE.DynamicDrawUsage);
    const idx: number[] = [];
    for (let i = 0; i < samples - 1; i++)
      for (let j = 0; j < ring; j++) {
        const a = i * ring + j, b = i * ring + ((j + 1) % ring), c = (i + 1) * ring + j, d = (i + 1) * ring + ((j + 1) % ring);
        idx.push(a, c, b, b, c, d);
      }
    g.setAttribute("position", pos);
    g.setAttribute("normal", nrm);
    g.setIndex(idx);
    const m = new THREE.MeshStandardMaterial({
      color: new THREE.Color().setRGB(color[0], color[1], color[2], THREE.SRGBColorSpace),
      roughness,
      metalness: 0,
    });
    const mesh = new THREE.Mesh(g, m);
    mesh.castShadow = true;
    mesh.frustumCulled = false;
    this.scene.add(mesh);
    this.tube = { mesh, pos, nrm, radius, ring, samples, pts: new Float32Array(samples * 3) };
  }

  /** Centreline points [n*3] in world; frames by parallel transport, so the tube never twists. */
  setTube(c: Float32Array): void {
    const t = this.tube;
    if (!t) return;
    const n = c.length / 3;
    const per = (t.samples - 1) / (n - 1);
    const P = t.pts;
    const at = (i: number, k: number) => c[3 * Math.min(n - 1, Math.max(0, i)) + k];
    for (let s = 0; s < n - 1; s++)
      for (let u = 0; u < per; u++) {
        const x = u / per;
        const x2 = x * x, x3 = x2 * x;
        const o = (s * per + u) * 3;
        for (let k = 0; k < 3; k++) {
          const p0 = at(s - 1, k), p1 = at(s, k), p2 = at(s + 1, k), p3 = at(s + 2, k);
          P[o + k] = 0.5 * (2 * p1 + (-p0 + p2) * x + (2 * p0 - 5 * p1 + 4 * p2 - p3) * x2 + (-p0 + 3 * p1 - 3 * p2 + p3) * x3);
        }
      }
    for (let k = 0; k < 3; k++) P[(t.samples - 1) * 3 + k] = c[(n - 1) * 3 + k];
    const pos = t.pos.array as Float32Array;
    const nrm = t.nrm.array as Float32Array;
    const T = new THREE.Vector3(), N = new THREE.Vector3(), B = new THREE.Vector3(), prevT = new THREE.Vector3();
    for (let i = 0; i < t.samples; i++) {
      const a = Math.max(0, i - 1), b = Math.min(t.samples - 1, i + 1);
      T.set(P[3 * b] - P[3 * a], P[3 * b + 1] - P[3 * a + 1], P[3 * b + 2] - P[3 * a + 2]).normalize();
      if (i === 0) {
        N.set(0, 0, 1).cross(T);
        if (N.lengthSq() < 1e-6) N.set(1, 0, 0).cross(T);
        N.normalize();
      } else {
        // parallel transport: rotate the previous normal by the tangent's turn
        const q = new THREE.Quaternion().setFromUnitVectors(prevT, T);
        N.applyQuaternion(q).normalize();
      }
      B.crossVectors(T, N);
      prevT.copy(T);
      for (let j = 0; j < t.ring; j++) {
        const ang = (2 * Math.PI * j) / t.ring;
        const cs = Math.cos(ang), sn = Math.sin(ang);
        const nx = N.x * cs + B.x * sn, ny = N.y * cs + B.y * sn, nz = N.z * cs + B.z * sn;
        const o = (i * t.ring + j) * 3;
        nrm[o] = nx;
        nrm[o + 1] = ny;
        nrm[o + 2] = nz;
        pos[o] = P[3 * i] + t.radius * nx;
        pos[o + 1] = P[3 * i + 1] + t.radius * ny;
        pos[o + 2] = P[3 * i + 2] + t.radius * nz;
      }
    }
    t.pos.needsUpdate = true;
    t.nrm.needsUpdate = true;
  }

  /** Pose a body node: p (xyz) and q (xyzw) in world. */
  setBody(body: number, p: ArrayLike<number>, q: ArrayLike<number>): void {
    const n = this.bodies.get(body);
    if (!n) return;
    n.matrix.compose(new THREE.Vector3(p[0], p[1], p[2]), new THREE.Quaternion(q[0], q[1], q[2], q[3]), new THREE.Vector3(1, 1, 1));
    n.matrixWorldNeedsUpdate = true;
  }

  resize(cssW: number, cssH: number, dpr: number, maxPixels = 2.4e6): void {
    // The canvas covers the 1920x1080 plate box; its drawing buffer follows the
    // on-screen size, capped by a pixel budget and never above 1920x1080.
    const scale = Math.min(dpr * (cssW / this.width), Math.sqrt(maxPixels / (this.width * this.height)), 1);
    const w = Math.max(2, Math.round(this.width * scale));
    const h = Math.max(2, Math.round(this.height * scale));
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(w, h, false);
  }

  render(): void {
    this.renderer.render(this.scene, this.camera);
  }

  /** Empty the scene for the next demo: geometry, materials, textures, lights. */
  reset(): void {
    this.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh) {
        m.geometry?.dispose();
        const mats = Array.isArray(m.material) ? m.material : [m.material];
        for (const mat of mats) {
          for (const v of Object.values(mat)) if ((v as THREE.Texture)?.isTexture) (v as THREE.Texture).dispose();
          mat.dispose();
        }
      }
    });
    this.scene.clear();
    this.bodies.clear();
    this.staticRoot = null;
    this.key = null;
    this.deform = null;
    this.tube = null;
    this.hemi = null;
    this.catchers = [];
    this.envTex?.dispose();
    this.envTex = null;
    this.scene.environment = null;
    this.renderer.clear();
  }

  dispose(): void {
    this.reset();
    this.renderer.dispose();
  }
}
