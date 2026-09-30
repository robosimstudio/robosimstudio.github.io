// The page: one stage, one demo at a time, a picker of RTX tile loops.
//
// Input: one finger (or the left button) drives the demo -- drags the robot, or
// scrubs a rollout. Two fingers, the right or middle button, or a modified drag
// (Alt, Shift, Ctrl, Cmd) orbit the view; the wheel and a pinch zoom; Escape
// goes back to the hero view, as a few idle seconds do by themselves.

import { View } from "../core/view.ts";
import { LiveRenderer } from "../render/Live.ts";
import { DATA, fetchJson } from "../data/fetch.ts";
import { apply, place, type Placement, toRtx } from "./layout.ts";

export interface DemoIndexItem {
  id: string;
  title: string;
  kind: "free" | "track";
  tile?: string;
  poster?: string;
}

export interface Stage {
  root: HTMLElement;
  plate: HTMLElement;
  bg: HTMLImageElement;
  bgInk: HTMLCanvasElement;
  live: LiveRenderer;
  liveCanvas: HTMLCanvasElement;
  still: HTMLImageElement;
  stillInk: HTMLCanvasElement;
  video: HTMLVideoElement;
  overlay: HTMLElement;
  /** The RTX layers drawn over the live canvas (still, its ink, video): faded out off the hero view. */
  over: HTMLElement;
  /** The visitor's view: the hero camera, or an orbit about it. */
  view: View;
  curtain: HTMLElement;
  loading: HTMLElement;
  placement: () => Placement;
  base: (id: string) => string;
}

export interface Demo {
  start(): Promise<void>;
  frame(now: number, dt: number): void;
  pointerDown(e: PointerEvent, rtx: [number, number]): void;
  pointerMove(e: PointerEvent, rtx: [number, number]): void;
  pointerUp(e: PointerEvent, rtx: [number, number]): void;
  /** The drag ends without a release (a second finger turned it into an orbit): no flick. */
  pointerCancel?(): void;
  /** Whether the scene is loaded enough to be looked at from elsewhere. */
  canOrbit?(): boolean;
  key?(e: KeyboardEvent): void;
  resize(p: Placement): void;
  roi(): [number, number, number, number];
  dispose(): void;
}

export type DemoFactory = (stage: Stage, id: string, manifest: Record<string, unknown>) => Demo;

export class App {
  private stage: Stage;
  private demo: Demo | null = null;
  private demoId = "";
  private index: DemoIndexItem[] = [];
  private placementNow: Placement = { s: 1, ox: 0, oy: 0, stageW: 1920, stageH: 1080 };
  private last = performance.now();
  private activePointer: number | null = null;
  private pointers = new Map<number, { x: number; y: number }>();
  private gesture: "none" | "demo" | "orbit" = "none";
  private orbitPrev: { x: number; y: number; spread: number; n: number } | null = null;
  private poseVersion = -1;
  private loadToken = 0;
  private factories: Record<string, DemoFactory>;

  constructor(factories: Record<string, DemoFactory>) {
    this.factories = factories;
    const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
    const liveCanvas = $<HTMLCanvasElement>("live");
    this.stage = {
      root: $("stage"),
      plate: $("plate"),
      bg: $<HTMLImageElement>("bg"),
      bgInk: $<HTMLCanvasElement>("bg-ink"),
      live: new LiveRenderer(liveCanvas),
      liveCanvas,
      still: $<HTMLImageElement>("still"),
      stillInk: $<HTMLCanvasElement>("still-ink"),
      video: $<HTMLVideoElement>("vid"),
      overlay: $("overlay"),
      over: $("over"),
      view: new View(),
      curtain: $("curtain"),
      loading: $("loading"),
      placement: () => this.placementNow,
      base: (id: string) => `${DATA}demos/${id}/`,
    };
    (window as unknown as { __rswView: View }).__rswView = this.stage.view; // tests and the console
  }

  async start(): Promise<void> {
    const idx = await fetchJson<{ demos: DemoIndexItem[] }>(`${DATA}demos.json`, true);
    this.index = idx.demos;
    this.buildPicker();
    window.addEventListener("resize", () => this.relayout());
    window.addEventListener("hashchange", () => this.route());
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        this.stage.view.goHome();
        return;
      }
      this.demo?.key?.(e);
    });
    this.bindPointer();
    await this.route();
    requestAnimationFrame(this.tick);
  }

  private tick = (now: number): void => {
    const dt = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    const { view, live } = this.stage;
    if (this.demo) {
      view.update(now, dt);
      if (view.version !== this.poseVersion) {
        live.setPose(view.camToWorld());
        this.poseVersion = view.version;
      }
      live.setFreeLook(view.look);
      const k = view.look;
      this.stage.over.style.opacity = k > 0 ? String(1 - k * k * (3 - 2 * k)) : "";
      this.stage.root.classList.toggle("freelook", k > 0);
    }
    this.demo?.frame(now, dt);
    requestAnimationFrame(this.tick);
  };

  private async route(): Promise<void> {
    const want = location.hash.replace(/^#/, "");
    const id = this.index.some((d) => d.id === want) ? want : this.index[0]?.id;
    if (!id || id === this.demoId) return;
    await this.load(id);
  }

  private async load(id: string): Promise<void> {
    const token = ++this.loadToken;
    if (this.demo) {
      this.stage.curtain.classList.add("on");
      await new Promise((r) => setTimeout(r, 170));
      if (token !== this.loadToken) return;
    }
    this.demo?.dispose();
    this.demo = null;
    this.demoId = id;
    this.markActive(id);
    this.stage.loading.classList.add("on");
    const manifest = await fetchJson<Record<string, unknown>>(`${this.stage.base(id)}manifest.json`, true);
    if (token !== this.loadToken) return;
    const make = this.factories[manifest.kind as string];
    this.stage.view.setHero(manifest.camera as Parameters<View["setHero"]>[0]); // before the demo: it projects through it
    this.poseVersion = -1;
    const demo = make(this.stage, id, manifest);
    this.demo = demo;
    (window as unknown as { __rsw: unknown }).__rsw = demo;
    this.relayout();
    await demo.start();
    if (token !== this.loadToken) return;
    this.stage.loading.classList.remove("on");
    await presented(this.stage.video, 2500);
    if (token === this.loadToken) this.stage.curtain.classList.remove("on");
  }

  private relayout(): void {
    const roi = this.demo?.roi() ?? [0, 0, 1920, 1080];
    this.placementNow = place(this.stage.root, roi);
    apply(this.stage.plate, this.placementNow);
    this.demo?.resize(this.placementNow);
  }

  private bindPointer(): void {
    const root = this.stage.root;
    const view = this.stage.view;
    const rtx = (e: PointerEvent) => toRtx(this.placementNow, root.getBoundingClientRect(), e.clientX, e.clientY);
    const canOrbit = () => !!this.demo && (this.demo.canOrbit?.() ?? true);
    root.addEventListener("contextmenu", (e) => e.preventDefault()); // the right button orbits
    root.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        if (!canOrbit()) return;
        const px = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1);
        // a trackpad pinch arrives as a ctrl-wheel of small steps
        view.dolly(Math.exp(px * (e.ctrlKey ? 0.01 : 0.0015)));
      },
      { passive: false },
    );
    root.addEventListener("pointerdown", (e) => {
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      view.touch();
      const orbitButton = e.pointerType === "mouse" && (e.button === 1 || e.button === 2 || e.altKey || e.shiftKey || e.ctrlKey || e.metaKey);
      const twoFingers = e.pointerType !== "mouse" && this.pointers.size >= 2;
      if (this.gesture === "orbit") {
        root.setPointerCapture(e.pointerId);
        this.orbitPrev = this.orbitAnchor(); // a finger joined: carry on from here
        return;
      }
      if ((orbitButton && this.gesture === "none") || (twoFingers && canOrbit())) {
        if (this.gesture === "demo") {
          this.demo?.pointerCancel?.(); // the first finger's drag becomes the orbit
          this.activePointer = null;
        }
        if (!canOrbit()) return;
        this.gesture = "orbit";
        for (const id of this.pointers.keys()) if (root.hasPointerCapture?.(id) === false) root.setPointerCapture(id);
        root.setPointerCapture(e.pointerId);
        root.classList.add("dragging", "orbiting");
        this.orbitPrev = this.orbitAnchor();
        return;
      }
      if (this.gesture !== "none") return; // one finger drives
      this.gesture = "demo";
      this.activePointer = e.pointerId;
      root.setPointerCapture(e.pointerId);
      root.classList.add("dragging");
      this.demo?.pointerDown(e, rtx(e));
    });
    root.addEventListener("pointermove", (e) => {
      const p = this.pointers.get(e.pointerId);
      if (p) {
        p.x = e.clientX;
        p.y = e.clientY;
      }
      if (this.gesture === "orbit") {
        if (!p || !this.orbitPrev) return;
        const a = this.orbitAnchor();
        view.orbit(a.x - this.orbitPrev.x, a.y - this.orbitPrev.y);
        if (a.n >= 2 && this.orbitPrev.n >= 2 && a.spread > 8 && this.orbitPrev.spread > 8) view.dolly(this.orbitPrev.spread / a.spread);
        this.orbitPrev = a;
        return;
      }
      if (e.pointerId !== this.activePointer) {
        if (this.gesture === "none") this.demo?.pointerMove(e, rtx(e)); // hover
        return;
      }
      view.touch();
      const events = typeof e.getCoalescedEvents === "function" ? e.getCoalescedEvents() : [e];
      for (const ce of events.length ? events : [e]) this.demo?.pointerMove(ce, rtx(ce));
    });
    const up = (e: PointerEvent) => {
      this.pointers.delete(e.pointerId);
      if (this.gesture === "orbit") {
        view.touch();
        if (this.pointers.size === 0) {
          this.gesture = "none";
          this.orbitPrev = null;
          root.classList.remove("dragging", "orbiting");
        } else this.orbitPrev = this.orbitAnchor(); // a finger lifted: carry on from here
        return;
      }
      if (e.pointerId !== this.activePointer) return;
      view.touch();
      this.activePointer = null;
      this.gesture = "none";
      root.classList.remove("dragging");
      this.demo?.pointerUp(e, rtx(e));
    };
    root.addEventListener("pointerup", up);
    root.addEventListener("pointercancel", up);
  }

  /** The orbiting fingers' middle, how far apart they are, and how many. */
  private orbitAnchor(): { x: number; y: number; spread: number; n: number } {
    let x = 0;
    let y = 0;
    const n = this.pointers.size || 1;
    for (const p of this.pointers.values()) {
      x += p.x;
      y += p.y;
    }
    x /= n;
    y /= n;
    let spread = 0;
    for (const p of this.pointers.values()) spread += Math.hypot(p.x - x, p.y - y);
    return { x, y, spread: spread / n, n: this.pointers.size };
  }

  private buildPicker(): void {
    const nav = document.getElementById("picker")!;
    nav.replaceChildren();
    const io = new IntersectionObserver(
      (entries) => {
        for (const en of entries) {
          const v = en.target.querySelector("video");
          if (!v) continue;
          if (en.isIntersecting) void v.play().catch(() => undefined);
          else v.pause();
        }
      },
      { root: nav, threshold: 0.5 },
    );
    for (const d of this.index) {
      const b = document.createElement("button");
      b.className = "tile";
      b.dataset.id = d.id;
      b.setAttribute("aria-label", d.title);
      if (d.tile) {
        const v = document.createElement("video");
        v.muted = true;
        v.loop = true;
        v.playsInline = true;
        v.preload = "metadata";
        if (d.poster) v.poster = `${this.stage.base(d.id)}${d.poster}`;
        v.src = `${this.stage.base(d.id)}${d.tile}`;
        b.appendChild(v);
      } else if (d.poster) {
        const img = document.createElement("img");
        img.src = `${this.stage.base(d.id)}${d.poster}`;
        img.alt = "";
        b.appendChild(img);
      }
      const label = document.createElement("span");
      label.textContent = d.title;
      b.appendChild(label);
      b.addEventListener("click", () => {
        location.hash = d.id;
      });
      nav.appendChild(b);
      io.observe(b);
    }
  }

  private markActive(id: string): void {
    for (const t of document.querySelectorAll<HTMLElement>(".tile")) t.classList.toggle("active", t.dataset.id === id);
  }
}

/** Resolves once the video has put a frame on screen (or after ``maxMs``). */
function presented(v: HTMLVideoElement, maxMs: number): Promise<void> {
  return new Promise((resolve) => {
    const done = () => resolve();
    const timer = setTimeout(done, maxMs);
    const finish = () => {
      clearTimeout(timer);
      done();
    };
    // Firefox < 132 has no requestVideoFrameCallback
    const rvfc = (v as { requestVideoFrameCallback?: (cb: () => void) => number }).requestVideoFrameCallback;
    if (rvfc) rvfc.call(v, finish);
    else (v as HTMLVideoElement).addEventListener("playing", finish, { once: true });
  });
}
