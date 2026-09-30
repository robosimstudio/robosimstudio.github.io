// One transform places the 1920x1080 plate box on the stage; every layer
// (RTX plate, live canvas, RTX still, RTX video) fills that box, so they align
// by construction. Desktop and landscape: cover. Portrait: fit the region of
// interest (where the action is), never cutting it.

export interface Placement {
  s: number;
  ox: number;
  oy: number;
  stageW: number;
  stageH: number;
}

export const PLATE_W = 1920;
export const PLATE_H = 1080;

export function place(stage: HTMLElement, roi: [number, number, number, number]): Placement {
  const W = stage.clientWidth;
  const H = stage.clientHeight;
  const sCover = Math.max(W / PLATE_W, H / PLATE_H);
  const sContain = Math.min(W / PLATE_W, H / PLATE_H);
  const sRoi = Math.min(W / (1.08 * roi[2]), H / (1.08 * roi[3]));
  const s = Math.max(sContain, Math.min(sCover, sRoi));
  const cx = roi[0] + roi[2] / 2;
  const cy = roi[1] + roi[3] / 2;
  const clamp = (o: number, span: number, size: number) =>
    span * s >= size ? Math.min(0, Math.max(size - span * s, o)) : (size - span * s) / 2;
  const ox = clamp(W / 2 - cx * s, PLATE_W, W);
  const oy = clamp(H / 2 - cy * s, PLATE_H, H);
  return { s, ox, oy, stageW: W, stageH: H };
}

export function apply(plate: HTMLElement, p: Placement): void {
  plate.style.transform = `translate(${p.ox}px, ${p.oy}px) scale(${p.s})`;
}

/** Client coordinates -> RTX pixel coordinates of the plate. */
export function toRtx(p: Placement, stageRect: DOMRect, clientX: number, clientY: number): [number, number] {
  return [(clientX - stageRect.left - p.ox) / p.s, (clientY - stageRect.top - p.oy) / p.s];
}

/** RTX pixel coordinates -> stage CSS pixels. */
export function toStage(p: Placement, u: number, v: number): [number, number] {
  return [p.ox + u * p.s, p.oy + v * p.s];
}
