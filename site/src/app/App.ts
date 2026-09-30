// The page: one stage, one demo at a time, a picker of RTX tile loops.

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
      curtain: $("curtain"),
      loading: $("loading"),
      placement: () => this.placementNow,
      base: (id: string) => `${DATA}demos/${id}/`,
    };
  }

  async start(): Promise<void> {
    const idx = await fetchJson<{ demos: DemoIndexItem[] }>(`${DATA}demos.json`, true);
    this.index = idx.demos;
    this.buildPicker();
    window.addEventListener("resize", () => this.relayout());
    window.addEventListener("hashchange", () => this.route());
    window.addEventListener("keydown", (e) => this.demo?.key?.(e));
    this.bindPointer();
    await this.route();
    requestAnimationFrame(this.tick);
  }

  private tick = (now: number): void => {
    const dt = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
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
    const rtx = (e: PointerEvent) => toRtx(this.placementNow, root.getBoundingClientRect(), e.clientX, e.clientY);
    root.addEventListener("pointerdown", (e) => {
      if (this.activePointer !== null) return; // one finger drives
      this.activePointer = e.pointerId;
      root.setPointerCapture(e.pointerId);
      root.classList.add("dragging");
      this.demo?.pointerDown(e, rtx(e));
    });
    root.addEventListener("pointermove", (e) => {
      if (e.pointerId !== this.activePointer) {
        if (this.activePointer === null) this.demo?.pointerMove(e, rtx(e)); // hover
        return;
      }
      const events = typeof e.getCoalescedEvents === "function" ? e.getCoalescedEvents() : [e];
      for (const ce of events.length ? events : [e]) this.demo?.pointerMove(ce, rtx(ce));
    });
    const up = (e: PointerEvent) => {
      if (e.pointerId !== this.activePointer) return;
      this.activePointer = null;
      root.classList.remove("dragging");
      this.demo?.pointerUp(e, rtx(e));
    };
    root.addEventListener("pointerup", up);
    root.addEventListener("pointercancel", up);
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
