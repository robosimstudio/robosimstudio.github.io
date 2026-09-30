// A free-drag demo: the visitor drags a ball, the robot chases it at its real
// speed limit, and what is shown is always a recorded simulation frame.
//
//   ATTRACT  the RTX video of a recorded virtual-visitor take; the ball follows
//            that take's own cursor. A touch hands over to LIVE at the video's
//            exact database frame (same state, same ink).
//   LIVE     raster moving parts over the RTX plate. Every 0.1 s the matcher
//            picks the recorded frame whose hand position, velocity, predicted
//            near-future path and deformable state best continue what is on
//            screen toward the ball; joins are inertialized in joint and PCA space.
//   SETTLING the ball stopped: the search is restricted to the still set
//            (settled frames with an RTX still) and the playhead freezes there.
//   STILL    that frame's RTX still, dissolved in, with the visitor's own ink.

import * as THREE from "three";
import type { Demo, Stage } from "../app/App.ts";
import { toStage, type Placement } from "../app/layout.ts";
import { fetchJson, fetchRssb } from "../data/fetch.ts";
import { Blender } from "../core/blender.ts";
import { Camera } from "../core/camera.ts";
import { DragMapping, Flick } from "../core/drag.ts";
import { InkField } from "../core/ink.ts";
import { Kinematics, type KinematicsJson } from "../core/kinematics.ts";
import { Matcher, type FeatureGroup } from "../core/matcher.ts";
import { OscSetpoint, predict, type OscConstants, type Plant } from "../core/osc.ts";
import { StatePlayer } from "../core/player.ts";
import { Settle } from "../core/settle.ts";
import { FrameStore, TAKE_ATTRACT } from "../core/store.ts";
import type { Q, Tf, V3 } from "../core/math.ts";
import { InkComposite } from "../render/InkComposite.ts";
import { Ball } from "../ui/Ball.ts";

type Mode = "boot" | "attract" | "live" | "settling" | "still";

interface Group extends FeatureGroup {
  mu: number[];
  sigma: number;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Manifest = any;

const FD_HORIZONS = [6, 12, 20];
const IDLE_TO_ATTRACT_S = 20;
const DISSOLVE_IN_S = 0.4;
const DISSOLVE_OUT_S = 0.15;
/** A fresh stripe, as the booth's ``new_stripe``: when a finger lands on a bar that is mostly clean
 * (under half the ink left), and when a finished wipe (under 5 % left) has rested this long. */
const REINK_ON_GRAB_BELOW = 0.5;
const REINK_DONE_BELOW = 0.05;
const REINK_REST_S = 2.5;
const REINK_GRAB_FADE_S = 0.35;
const REINK_REST_FADE_S = 0.8;

export class FreeDragDemo implements Demo {
  private mode: Mode = "boot";
  private stage: Stage;
  private m: Manifest;
  private base: string;
  private ball: Ball;
  private camera: Camera;
  private drag: DragMapping;
  private flick: Flick;
  private settle: Settle;
  private osc: OscSetpoint;
  private plant: Plant;
  private ctrl: OscConstants;
  private handQuat: Q;
  private store: FrameStore | null = null;
  private player: StatePlayer | null = null;
  private matcher: Matcher | null = null;
  private blender: Blender | null = null;
  private kin: Kinematics | null = null;
  private groups: Group[] = [];
  private ink: InkField | null = null;
  private bgInk: InkComposite | null = null;
  private stillInk: InkComposite | null = null;
  private reinkFade = 0; // [s] of the re-ink fade still to run, and its length
  private reinkLen = 0;
  private restClean = 0; // [s] at rest with a finished wipe
  private pca: { mean: Float32Array; basis: Float32Array; K: number; V: number; frame: string; arm: string; tube: boolean } | null = null;
  private moving: number[];
  private target: [number, number];
  private targetPrev: [number, number];
  private targetVel: [number, number] = [0, 0];
  private grab: { mode: "absolute" | "relative"; lastPx: [number, number]; offset: [number, number] } | null = null;
  private lastInput = performance.now();
  private framesSinceSearch = 0;
  private framesSinceTransition = 99;
  private stillIndex = -1;
  private stillAlpha = 0;
  private videoAlpha = 1;
  private disposed = false;
  private poses = new Map<number, Tf>();
  private buf: Record<string, Float64Array> = {};
  private prevHand: V3 = [0, 0, 0];
  private hovering = false;
  private dataReady = false;
  private stillCache = new Map<number, Promise<void>>();
  private lastSettle: unknown = null;

  constructor(stage: Stage, id: string, manifest: Manifest) {
    this.stage = stage;
    this.m = manifest;
    this.base = stage.base(id);
    this.camera = new Camera(manifest.camera);
    this.drag = new DragMapping(manifest.drag);
    this.flick = Flick.forDrag(this.drag);
    this.settle = new Settle({
      enterSpeed: manifest.settle?.enter_speed ?? 0.005,
      enterHold: manifest.settle?.enter_s ?? 0.15,
      exitSpeed: manifest.settle?.exit_speed ?? 0.015,
    });
    this.ctrl = manifest.ctrl as OscConstants;
    this.plant = manifest.ctrl.plant as Plant;
    this.handQuat = manifest.ctrl.hand_quat as Q;
    this.osc = new OscSetpoint(this.ctrl);
    this.moving = manifest.moving as number[];
    this.target = [manifest.drag.hover[0], manifest.drag.hover[1]];
    this.targetPrev = [...this.target];
    this.ball = new Ball(stage.overlay);
    this.ball.visible(false);
  }

  roi(): [number, number, number, number] {
    return this.m.roi;
  }

  async start(): Promise<void> {
    const { stage, m, base } = this;
    stage.live.setCamera(m.camera);
    stage.live.setLook(m.look ?? {});
    stage.bg.src = base + m.files.plate;
    stage.still.style.opacity = "0";
    const v = stage.video;
    v.loop = true;
    v.src = base + m.files.video;
    v.style.opacity = "1";
    this.videoAlpha = 1;
    void v.play().catch(() => undefined);
    this.mode = "attract";

    const [_, kinJson, deform, chunk0] = await Promise.all([
      stage.live.loadGlb(base + m.files.glb, m.nodes, m.look?.shadow_opacity ?? 0.28),
      fetchJson<KinematicsJson>(base + m.files.kin),
      fetchRssb(base + m.files.deform),
      fetchRssb(base + m.files.db[0]),
    ]);
    if (this.disposed) return;
    this.kin = new Kinematics(kinJson);
    this.store = new FrameStore(chunk0);
    this.groups = (m.features.groups as Group[]).map((g) => ({ ...g }));
    this.matcher = new Matcher(this.store, this.groups);
    const nq = this.store.dims.q as number;
    const K = this.store.dims.deform as number;
    this.blender = new Blender({ q: { dim: nq, halflife: 0.1 }, deform: { dim: K, halflife: 0.12 }, hand: { dim: 3, halflife: 0.1 } });
    this.player = new StatePlayer(this.store, m.attract?.frames?.[0] ?? 0);
    for (const [name, n] of [["q", nq], ["deform", K], ["hand", 3], ["qa", nq], ["qb", nq], ["da", K], ["db", K], ["ha", 3], ["hb", 3]] as const)
      this.buf[name] = new Float64Array(n);

    const dm = deform.arrays;
    const mean = dm.mean.data as Float32Array;
    const basis = dm.basis.data as Float32Array;
    const V = mean.length / 3;
    const dmeta = deform.meta as {
      frame: string; kind?: string; radius?: number;
      finish: { roughness: number; metallic: number; opacity: number }; color: number[]; arm?: string;
    };
    this.pca = { mean, basis, K, V, frame: dmeta.frame, arm: dmeta.arm ?? m.deform_arm ?? "right", tube: dmeta.kind === "tube" };
    if (dmeta.kind === "tube") stage.live.addTube(V, dmeta.radius ?? 0.0025, dmeta.color, dmeta.finish.roughness);
    else stage.live.addDeformable(dm.tris.data as Uint32Array, dm.uv.data as Float32Array, V, { color: dmeta.color, finish: dmeta.finish });

    if (m.ink) {
      const cells = (m.ink.cells as number[][]).length;
      this.ink = new InkField(cells, 1 / 30);
      const imap = await fetchRssb(base + m.files.inkmap);
      const cellMap = imap.arrays.cell.data as Uint16Array;
      this.bgInk = new InkComposite(stage.bgInk, m.ink.bbox, cellMap);
      this.stillInk = new InkComposite(stage.stillInk, m.ink.bbox, cellMap);
      await this.bgInk.setClean(base + m.files.plate_clean);
    }
    this.dataReady = true;
    this.ball.visible(true);
    this.ball.hint(true);
    void this.loadRest();
  }

  private async loadRest(): Promise<void> {
    const files = this.m.files.db as string[];
    for (let i = 1; i < files.length && !this.disposed; i++) {
      const c = await fetchRssb(this.base + files[i]);
      if (this.disposed || !this.store || !this.matcher) return;
      this.store.append(c);
      this.matcher.sync();
    }
  }

  // ------------------------------------------------------------------ input
  pointerDown(e: PointerEvent, rtx: [number, number]): void {
    this.lastInput = performance.now();
    if (!this.dataReady) return;
    if (this.mode === "attract") this.handover();
    if (this.mode === "still" || this.mode === "settling") this.wake();
    const bp = this.ballPx();
    const near = Math.hypot(rtx[0] - bp[0], rtx[1] - bp[1]) * this.stage.placement().s <= (e.pointerType === "touch" ? 44 : 26);
    this.grab = {
      mode: near ? "absolute" : "relative",
      lastPx: [rtx[0], rtx[1]],
      offset: near ? [bp[0] - rtx[0], bp[1] - rtx[1]] : [0, 0],
    };
    this.flick.stop();
    this.flick.push(performance.now() / 1000, this.target[0], this.target[1]);
    this.ball.held(true);
    this.ball.hint(false);
  }

  pointerMove(e: PointerEvent, rtx: [number, number]): void {
    if (!this.grab) {
      if (e.pointerType === "mouse" && this.dataReady) {
        const bp = this.ballPx();
        const near = Math.hypot(rtx[0] - bp[0], rtx[1] - bp[1]) * this.stage.placement().s <= 26;
        if (near !== this.hovering) {
          this.hovering = near;
          this.ball.el.style.filter = near ? "brightness(1.15)" : "";
        }
      }
      return;
    }
    this.lastInput = performance.now();
    if (this.grab.mode === "absolute") {
      const px: [number, number] = [rtx[0] + this.grab.offset[0], rtx[1] + this.grab.offset[1]];
      const ab = this.drag.solveAbsolute(px, this.target, this.camera);
      this.target = [ab[0], ab[1]];
    } else {
      const dpx: [number, number] = [rtx[0] - this.grab.lastPx[0], rtx[1] - this.grab.lastPx[1]];
      const ab = this.drag.relative(dpx, this.target[0], this.target[1], this.camera, e.pointerType === "touch" ? 1.3 : 1.0);
      this.target = [ab[0], ab[1]];
    }
    this.grab.lastPx = [rtx[0], rtx[1]];
    this.flick.push(e.timeStamp / 1000, this.target[0], this.target[1]);
  }

  pointerUp(e: PointerEvent): void {
    if (!this.grab) return;
    this.grab = null;
    this.ball.held(false);
    this.lastInput = performance.now();
    const coasting = this.flick.release(e.timeStamp / 1000);
    if (!coasting) this.settle.coastEnded();
  }

  // ---------------------------------------------------------------- handover
  private handover(): void {
    const { store, player } = this;
    if (!store || !player) return;
    const f0 = this.m.attract?.frames?.[0] ?? 0;
    const k = Math.max(0, Math.floor(this.stage.video.currentTime * 30 + 1e-3));
    const f = Math.min(f0 + k, (this.m.attract?.frames?.[1] ?? f0 + 1) - 1);
    player.set(f);
    this.blender?.reset();
    const cur = store.cursor.subarray(2 * f, 2 * f + 2);
    this.target = [cur[0], cur[1]];
    this.targetPrev = [...this.target];
    const sp = store.setpoint.subarray(3 * f, 3 * f + 3);
    this.osc.reset([sp[0], sp[1], sp[2]], this.handQuat);
    const h = store.hand.subarray(3 * f, 3 * f + 3);
    this.prevHand = [h[0], h[1], h[2]];
    const row = store.inkStateRow(f);
    if (this.ink && row) this.ink.setFromState(row);
    if (this.ink && this.inkLeft() < REINK_ON_GRAB_BELOW) this.startReink(REINK_GRAB_FADE_S);
    if (this.ink) this.bgInk?.draw(this.ink.I);
    this.stage.video.pause();
    this.videoAlpha = 0;
    this.mode = "live";
    this.framesSinceSearch = 99;
  }

  private inkLeft(): number {
    const I = this.ink!.I;
    let s = 0;
    for (let c = 0; c < I.length; c++) s += I[c];
    return s / Math.max(1, I.length);
  }

  private startReink(len: number): void {
    this.reinkLen = len;
    this.reinkFade = len;
  }

  /** Run the re-ink fade, and start one after a finished wipe has rested. */
  private stepReink(dt: number): void {
    if (!this.ink) return;
    if (this.reinkFade <= 0) {
      const resting = this.mode === "still" && !this.grab;
      this.restClean = resting && this.inkLeft() < REINK_DONE_BELOW ? this.restClean + dt : 0;
      if (this.restClean < REINK_REST_S) return;
      this.restClean = 0;
      this.startReink(REINK_REST_FADE_S);
    }
    const step = Math.min(dt, this.reinkFade) / this.reinkLen;
    this.reinkFade -= dt;
    const I = this.ink.I;
    for (let c = 0; c < I.length; c++) I[c] = Math.min(1, I[c] + step);
    this.ink.changed = true;
    this.bgInk?.draw(I);
    if (this.stillAlpha > 0) this.stillInk?.draw(I);
  }

  private wake(): void {
    this.settle.wake();
    this.stillIndex = -1;
    this.mode = "live";
    this.framesSinceSearch = 99;
  }

  private toAttract(): void {
    const v = this.stage.video;
    v.currentTime = 0;
    void v.play().catch(() => undefined);
    this.videoAlpha = 1;
    this.mode = "attract";
    this.stillIndex = -1;
  }

  // ------------------------------------------------------------------ frame
  frame(now: number, dt: number): void {
    const { stage } = this;
    if (this.mode === "boot" && this.dataReady) return; // pinned by showFrame (calibration)
    this.fade(dt);
    if (this.mode !== "attract" && this.dataReady) this.stepReink(dt);
    if (this.mode === "attract" || !this.dataReady) {
      this.drawBallAttract();
      return;
    }
    // target: the finger, a coast, or where the finger left it
    if (!this.grab && this.flick.active) {
      const still = !this.flick.coast(dt);
      this.target = [this.flick.x[0], this.flick.x[1]];
      if (still) this.settle.coastEnded();
    }
    this.targetVel = [(this.target[0] - this.targetPrev[0]) / Math.max(dt, 1e-3), (this.target[1] - this.targetPrev[1]) / Math.max(dt, 1e-3)];
    this.targetPrev = [...this.target];

    const player = this.player!;
    const store = this.store!;
    if (this.mode === "live" || this.mode === "settling") {
      // playback (a frozen settle holds its frame)
      const frozen = this.mode === "settling" && this.stillIndex >= 0;
      if (!frozen) {
        const crossed = player.advance(30 * dt);
        this.framesSinceSearch += crossed.length;
        this.framesSinceTransition += crossed.length;
        if (this.ink && store.ink) {
          for (const f of crossed) this.ink.applyFrame(store.ink, this.m.ink.r0, this.m.ink.rmax, 1 / 30, f * this.ink.cells);
          if (crossed.length) this.bgInk?.draw(this.ink.I);
        }
      }
      const A = this.drag.A, B = this.drag.B, va = this.targetVel[0], vb = this.targetVel[1];
      const speed = Math.hypot(va * A[0] + vb * B[0], va * A[1] + vb * B[1], va * A[2] + vb * B[2]);
      const pred = this.predictDisp();
      const mode = this.grab ? "active" : this.settle.update(speed, pred.slice(6, 9), dt);
      if (mode === "settling" && this.mode === "live") {
        this.mode = "settling";
        this.framesSinceSearch = 99;
      }
      if (this.framesSinceSearch >= (this.m.match?.every_frames ?? 3) || player.runway() < 4) this.search(pred);
      if (this.mode === "settling" && this.stillIndex >= 0) this.maybeDissolve();
    }
    if (idleFor(this.lastInput) > IDLE_TO_ATTRACT_S && !this.grab) this.toAttract();
    this.pose(dt);
    stage.live.render();
    this.drawBall();
    void now;
  }

  // ------------------------------------------------------------------ search
  private query(pred: number[]): Float64Array {
    const q = new Float64Array(this.matcher!.D);
    const hand = this.buf.hand;
    const deform = this.buf.deform;
    const hv = this.handVelocity();
    const dv = this.deformVelocity();
    let o = 0;
    for (const g of this.groups) {
      for (let d = 0; d < g.dims; d++) {
        let x = 0;
        if (g.name === "fd") x = pred[d];
        else if (g.name === "hp") x = hand[d];
        else if (g.name === "hv") x = hv[d];
        else if (g.name === "dm") x = deform[d];
        else if (g.name === "dv") x = dv[d];
        else if (g.name === "pg") x = (this.store!.broken?.[this.player!.frame] ?? 0) / (this.m.peel?.n_bonds ?? 1);
        q[o + d] = (x - g.mu[d]) / g.sigma;
      }
      o += g.dims;
    }
    return this.matcher!.weigh(q, q);
  }

  private search(pred: number[]): void {
    const matcher = this.matcher!;
    const player = this.player!;
    const store = this.store!;
    this.framesSinceSearch = 0;
    const q = this.query(pred);
    const cur = player.frame;
    const settling = this.mode === "settling";
    const still = store.still;
    // Settling means coming to rest where the ball is: among the still set, the
    // hand's distance from the ball dominates (4 mm costs as much as a unit of
    // feature mismatch). Measured without it: the settle picked a still 4 cm off.
    const ab = this.drag.clamp(this.target[0], this.target[1]);
    const rest = this.drag.toWorld(ab[0], ab[1]);
    const hand = store.hand;
    const W = 1 / (0.004 * 0.004);
    const settleBias = (i: number) => {
      if (!still || still[i] < 0) return 1e6;
      const dx = hand[3 * i] - rest[0], dy = hand[3 * i + 1] - rest[1], dz = hand[3 * i + 2] - rest[2];
      return W * (dx * dx + dy * dy + dz * dz);
    };
    const opts = {
      current: cur,
      minRunway: this.m.match?.min_runway ?? 10,
      sameTakeWindow: this.m.match?.same_take_window ?? 15,
      startMargin: this.m.match?.start_guard ?? 30,
      bias: settling && still ? settleBias : undefined,
      monotone: store.broken
        ? {
            broken: store.broken,
            front: store.front,
            cur: {
              broken: store.broken[cur],
              front: store.front ? store.front.subarray(cur * (store.dims.front as number), (cur + 1) * (store.dims.front as number)) : null,
            },
          }
        : undefined,
    };
    const r = matcher.search(q, opts);
    if (r.frame < 0) return;
    if (settling) {
      // diagnostics: the chosen still against the stills nearest the rest point
      const near: [number, number, number][] = [];
      if (still) {
        const cand: [number, number][] = [];
        for (let i = 0; i < store.n; i++) if (still[i] >= 0) cand.push([i, settleBias(i)]);
        cand.sort((x, y) => x[1] - y[1]);
        for (const [i, bi] of cand.slice(0, 3)) near.push([i, matcher.costAt(i, q), bi]);
      }
      this.lastSettle = { chosen: [r.frame, r.cost, settleBias(r.frame)], near };
    }
    const contCost = matcher.costAt(Math.min(cur + 1, store.n - 1), q) + (settling ? settleBias(Math.min(cur + 1, store.n - 1)) : 0);
    const ratio = this.m.match?.accept_ratio ?? 0.85;
    const margin = this.m.match?.accept_margin ?? 0.05;
    const gap = this.m.match?.min_gap_frames ?? 6;
    const forced = player.runway() < 4;
    // Settling takes any strictly better still-set frame (no live hysteresis): the
    // point is to stop, near the ball, on a frame that has an RTX still.
    const better = settling ? r.cost < contCost : r.cost < ratio * contCost - margin && this.framesSinceTransition >= gap;
    if (forced || better) this.transition(r.frame);
    // Freeze only on a still near the ball's rest point (6 mm): the playhead can be
    // passing a still frame of some other hold when settling starts.
    if (settling && still && still[player.frame] >= 0 && settleBias(player.frame) <= 2.25) this.freezeOnStill(player.frame);
  }

  private transition(f: number): void {
    const player = this.player!;
    const b = this.blender!;
    const { qa, qb, da, db, ha, hb } = this.buf;
    // the stream being left, and the one being entered, with velocities per second
    this.sampleAt(player.frame, qa, da, ha);
    this.sampleAt(Math.max(player.frame - 1, 0), qb, db, hb);
    const srcQ = Float64Array.from(qa), srcQv = qa.map((x, i) => (x - qb[i]) * 30);
    const srcD = Float64Array.from(da), srcDv = da.map((x, i) => (x - db[i]) * 30);
    const srcH = Float64Array.from(ha), srcHv = ha.map((x, i) => (x - hb[i]) * 30);
    this.sampleAt(f, qa, da, ha);
    this.sampleAt(Math.max(f - 1, 0), qb, db, hb);
    b.transition("q", srcQ, srcQv, qa, qa.map((x, i) => (x - qb[i]) * 30));
    b.transition("deform", srcD, srcDv, da, da.map((x, i) => (x - db[i]) * 30));
    b.transition("hand", srcH, srcHv, ha, ha.map((x, i) => (x - hb[i]) * 30));
    player.set(f);
    this.framesSinceTransition = 0;
  }

  private sampleAt(f: number, q: Float64Array, d: Float64Array, h: Float64Array): void {
    const s = this.store!;
    const nq = s.dims.q as number;
    const K = s.dims.deform as number;
    for (let i = 0; i < nq; i++) q[i] = s.q[f * nq + i];
    for (let i = 0; i < K; i++) d[i] = s.deform![f * K + i];
    for (let i = 0; i < 3; i++) h[i] = s.hand[f * 3 + i];
  }

  // -------------------------------------------------------------- settling
  private freezeOnStill(f: number): void {
    const idx = this.store!.still![f];
    if (idx < 0 || idx === this.stillIndex) return;
    this.stillIndex = idx;
    const url = this.base + this.m.stills.files[idx];
    const clean = this.m.stills.clean?.[idx];
    if (!this.stillCache.has(idx)) {
      const img = new Image();
      img.src = url;
      const p = img.decode().then(async () => {
        if (clean && this.stillInk) await this.stillInk.setClean(this.base + clean);
      });
      this.stillCache.set(idx, p.catch(() => undefined));
    }
  }

  private maybeDissolve(): void {
    const idx = this.stillIndex;
    const p = this.stillCache.get(idx);
    if (!p) return;
    void p.then(() => {
      if (this.stillIndex !== idx || this.mode !== "settling") return;
      const b = this.blender!;
      if (b.get("q").o.some((x) => Math.abs(x) > 2e-4) || b.get("hand").o.some((x) => Math.abs(x) > 2e-4)) return;
      this.stage.still.src = this.base + this.m.stills.files[idx];
      if (this.ink) this.stillInk?.draw(this.ink.I);
      this.mode = "still";
    });
  }

  private fade(dt: number): void {
    const { stage } = this;
    const wantStill = this.mode === "still" ? 1 : 0;
    const rate = wantStill > this.stillAlpha ? dt / DISSOLVE_IN_S : dt / DISSOLVE_OUT_S;
    this.stillAlpha = wantStill > this.stillAlpha ? Math.min(1, this.stillAlpha + rate) : Math.max(0, this.stillAlpha - rate);
    const s = this.stillAlpha;
    stage.still.style.opacity = String(s * s * (3 - 2 * s));
    stage.stillInk.style.opacity = stage.still.style.opacity;
    const wantVid = this.videoAlpha;
    const cur = Number(stage.video.style.opacity || "0");
    const vr = dt / (wantVid > cur ? 0.5 : DISSOLVE_OUT_S);
    stage.video.style.opacity = String(wantVid > cur ? Math.min(1, cur + vr) : Math.max(0, cur - vr));
  }

  // -------------------------------------------------------------- posing
  private pose(dt: number): void {
    const player = this.player!;
    const b = this.blender!;
    const { q, deform, hand } = this.buf;
    player.sample("q", q);
    player.sample("deform", deform);
    player.sample("hand", hand);
    b.apply(dt);
    b.output("q", q, q);
    b.output("deform", deform, deform);
    b.output("hand", hand, hand);
    const poses = this.kin!.fk(q, this.poses);
    for (const body of this.moving) {
      const t = poses.get(body);
      if (t) this.stage.live.setBody(body, t.p, t.q);
    }
    const world = this.deformWorld(deform, poses);
    if (this.pca!.tube) this.stage.live.setTube(world);
    else this.stage.live.setDeformable(world);
    this.prevHand = [hand[0], hand[1], hand[2]];
  }

  private worldBuf = new Float32Array(0);

  private deformWorld(c: Float64Array, poses: Map<number, Tf>): Float32Array {
    const p = this.pca!;
    if (this.worldBuf.length !== p.V * 3) this.worldBuf = new Float32Array(p.V * 3);
    const out = this.worldBuf;
    out.set(p.mean);
    const n3 = p.V * 3;
    for (let k = 0; k < p.K; k++) {
      const ck = c[k];
      if (ck === 0) continue;
      const row = p.basis.subarray(k * n3, (k + 1) * n3);
      for (let i = 0; i < n3; i++) out[i] += ck * row[i];
    }
    if (p.frame === "tcp") {
      const t = this.kin!.tcp(p.arm, poses);
      const m = new THREE.Matrix4().compose(
        new THREE.Vector3(t.p[0], t.p[1], t.p[2]),
        new THREE.Quaternion(t.q[0], t.q[1], t.q[2], t.q[3]),
        new THREE.Vector3(1, 1, 1),
      );
      const e = m.elements;
      for (let i = 0; i < n3; i += 3) {
        const x = out[i], y = out[i + 1], z = out[i + 2];
        out[i] = e[0] * x + e[4] * y + e[8] * z + e[12];
        out[i + 1] = e[1] * x + e[5] * y + e[9] * z + e[13];
        out[i + 2] = e[2] * x + e[6] * y + e[10] * z + e[14];
      }
    }
    return out;
  }

  private handVelocity(): number[] {
    const player = this.player!;
    const s = this.store!;
    const f = player.frame;
    const f0 = Math.max(s.takes[s.takeOf(f)].start, f - 1);
    if (f0 === f) return [0, 0, 0];
    return [0, 1, 2].map((i) => (s.hand[f * 3 + i] - s.hand[f0 * 3 + i]) * 30);
  }

  private deformVelocity(): number[] {
    const player = this.player!;
    const s = this.store!;
    const K = s.dims.deform as number;
    const f = player.frame;
    const f0 = Math.max(s.takes[s.takeOf(f)].start, f - 1);
    const m = this.groups.find((g) => g.name === "dv")?.dims ?? 0;
    const out: number[] = [];
    for (let i = 0; i < m; i++) out.push(f0 === f ? 0 : (s.deform![f * K + i] - s.deform![f0 * K + i]) * 30);
    return out;
  }

  /** The servo law + fitted plant rolled 20 control steps toward the target: fd (9). */
  private predictDisp(): number[] {
    const hand = this.buf.hand;
    const goal = this.drag.toWorld(this.target[0], this.target[1]);
    const va = this.targetVel;
    const tau = this.drag.coastTau;
    const goalAt = (k: number): V3 => {
      const t = k * this.ctrl.control_dt;
      // a dragged target keeps moving for a moment; a released one coasts to rest
      const s = this.grab ? t : tau * (1 - Math.exp(-t / tau));
      const ab = this.drag.clamp(this.target[0] + va[0] * s, this.target[1] + va[1] * s);
      return this.drag.toWorld(ab[0], ab[1]);
    };
    void goal;
    const sp = this.osc.p;
    const pr = predict(this.ctrl, this.plant, sp, [hand[0], hand[1], hand[2]], this.prevHand, goalAt, FD_HORIZONS);
    // keep the servo's own setpoint moving with the target, as the robot's would
    this.osc.advance([hand[0], hand[1], hand[2]], this.handQuat, goalAt(1), this.handQuat);
    return pr.disp.flat();
  }

  // ------------------------------------------------------------------- ball
  private ballPx(): [number, number] {
    const w = this.drag.toWorld(this.target[0], this.target[1]);
    const p = this.camera.project(w);
    return [p[0], p[1]];
  }

  private drawBall(): void {
    const pl = this.stage.placement();
    const w = this.drag.toWorld(this.target[0], this.target[1]);
    const p = this.camera.project(w);
    const [x, y] = toStage(pl, p[0], p[1]);
    const r = Math.min(26, Math.max(10, ((this.camera.fy * 0.0125) / Math.max(p[2], 0.05)) * pl.s));
    this.ball.place(x, y, r);
    const h = this.buf.hand;
    const ph = this.camera.project([h[0], h[1], h[2]]);
    const [hx, hy] = toStage(pl, ph[0], ph[1]);
    const lag = Math.hypot(w[0] - h[0], w[1] - h[1], w[2] - h[2]);
    this.ball.tetherTo(hx, hy, x, y, lag > 0.015);
  }

  private drawBallAttract(): void {
    if (!this.store) return;
    const f0 = this.m.attract?.frames?.[0] ?? 0;
    const f1 = this.m.attract?.frames?.[1] ?? f0 + 1;
    const k = Math.floor(this.stage.video.currentTime * 30 + 1e-3);
    const f = Math.min(f0 + Math.max(0, k), f1 - 1);
    const c = this.store.cursor.subarray(2 * f, 2 * f + 2);
    this.target = [c[0], c[1]];
    const pl = this.stage.placement();
    const w = this.drag.toWorld(c[0], c[1]);
    const p = this.camera.project(w);
    const [x, y] = toStage(pl, p[0], p[1]);
    const r = Math.min(26, Math.max(10, ((this.camera.fy * 0.0125) / Math.max(p[2], 0.05)) * pl.s));
    this.ball.place(x, y, r);
    this.ball.tetherTo(0, 0, 0, 0, false);
  }

  /** Calibration: pin the live layer to database frame ``f`` (no blending), stills and video hidden. */
  showFrame(f: number, inked = true): void {
    if (!this.player || !this.blender) return;
    this.mode = "boot";
    this.player.set(f);
    this.blender.reset();
    this.videoAlpha = 0;
    this.stillAlpha = 0;
    this.stage.video.pause();
    this.stage.video.style.opacity = "0";
    this.stage.still.style.opacity = "0";
    this.stage.stillInk.style.opacity = "0";
    if (this.ink) {
      this.ink.I.fill(inked ? 1 : 0);
      this.bgInk?.draw(this.ink.I);
    }
    this.ball.visible(false);
    this.pose(0);
    this.stage.live.render();
  }

  /** Calibration: retune the lights live. */
  setLook(look: Record<string, unknown>): void {
    this.stage.live.applyLook(look);
    this.stage.live.render();
  }

  /** For tests and the console: what the demo is doing right now. */
  debugState(): Record<string, unknown> {
    return {
      mode: this.mode,
      target: [...this.target],
      frame: this.player?.frame ?? -1,
      stillIndex: this.stillIndex,
      hand: this.buf.hand ? Array.from(this.buf.hand) : null,
      grab: this.grab?.mode ?? null,
      coasting: this.flick.active,
      settle: this.settle.mode,
      lastSettle: this.lastSettle,
    };
  }

  resize(p: Placement): void {
    // the live canvas fills the 1920x1080 plate box, drawn at its on-screen size
    this.stage.live.resize(1920 * p.s, 1080 * p.s, window.devicePixelRatio || 1);
  }

  dispose(): void {
    this.disposed = true;
    this.stage.live.reset();
    this.ball.dispose();
    this.stage.video.pause();
    this.stage.video.removeAttribute("src");
    this.stage.video.load();
  }
}

function idleFor(t: number): number {
  return (performance.now() - t) / 1000;
}

void TAKE_ATTRACT;
