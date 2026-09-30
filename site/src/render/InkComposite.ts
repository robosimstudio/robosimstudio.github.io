// The visitor's own ink on an RTX image. The RTX plate and stills are rendered
// with the stripe fully inked; a clean crop of the stripe's box is rendered too.
// Over the inked image, this canvas draws the clean crop with alpha 1 - I(cell),
// where the static ink map says a pixel shows that cell -- so the picture is
// mix(clean, inked, I). Where the sponge covers the stripe both renders agree,
// so the blend is harmless there.

export class InkComposite {
  private ctx: CanvasRenderingContext2D;
  private clean: ImageData | null = null;
  private out: ImageData | null = null;
  private map: Uint16Array;
  private w: number;
  private h: number;
  private canvas: HTMLCanvasElement;

  constructor(canvas: HTMLCanvasElement, bbox: [number, number, number, number], cellMap: Uint16Array) {
    this.canvas = canvas;
    const [x0, y0, w, h] = bbox;
    this.w = w;
    this.h = h;
    this.map = cellMap;
    canvas.width = w;
    canvas.height = h;
    canvas.style.left = `${x0}px`;
    canvas.style.top = `${y0}px`;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    this.ctx = canvas.getContext("2d", { willReadFrequently: false })!;
  }

  /** The clean crop for the image underneath (the plate's, or a still's). */
  async setClean(url: string): Promise<void> {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = url;
    await img.decode();
    const c = document.createElement("canvas");
    c.width = this.w;
    c.height = this.h;
    const cx = c.getContext("2d")!;
    cx.drawImage(img, 0, 0, this.w, this.h);
    this.clean = cx.getImageData(0, 0, this.w, this.h);
    this.out = new ImageData(this.w, this.h);
  }

  draw(intensity: Float32Array): void {
    if (!this.clean || !this.out) return;
    const src = this.clean.data;
    const dst = this.out.data;
    const map = this.map;
    for (let i = 0, p = 0; i < map.length; i++, p += 4) {
      const cell = map[i];
      if (cell === 0xffff) {
        dst[p + 3] = 0;
        continue;
      }
      dst[p] = src[p];
      dst[p + 1] = src[p + 1];
      dst[p + 2] = src[p + 2];
      dst[p + 3] = Math.round(255 * (1 - intensity[cell]));
    }
    this.ctx.putImageData(this.out, 0, 0);
  }

  show(on: boolean): void {
    this.canvas.style.display = on ? "" : "none";
  }
}
