// The orange ball a visitor drags (the booth demo's handle), drawn in the DOM
// overlay at the projection of the drag target.

export class Ball {
  readonly el: HTMLDivElement;
  private halo: HTMLDivElement;
  private tether: HTMLDivElement;

  constructor(parent: HTMLElement) {
    this.el = document.createElement("div");
    this.el.className = "ball";
    this.halo = document.createElement("div");
    this.halo.className = "halo";
    this.el.appendChild(this.halo);
    this.tether = document.createElement("div");
    this.tether.className = "tether";
    parent.append(this.tether, this.el);
  }

  place(x: number, y: number, radiusPx: number): void {
    const d = Math.round(2 * radiusPx);
    this.el.style.width = `${d}px`;
    this.el.style.height = `${d}px`;
    this.el.style.transform = `translate(${x - d / 2}px, ${y - d / 2}px)`;
  }

  /** A faint line from the hand to the ball while the hand lags behind it. */
  tetherTo(hx: number, hy: number, bx: number, by: number, show: boolean): void {
    if (!show) {
      this.tether.style.opacity = "0";
      return;
    }
    const len = Math.hypot(bx - hx, by - hy);
    const ang = Math.atan2(by - hy, bx - hx);
    this.tether.style.opacity = "1";
    this.tether.style.width = `${len}px`;
    this.tether.style.transform = `translate(${hx}px, ${hy}px) rotate(${ang}rad)`;
  }

  held(on: boolean): void {
    this.el.classList.toggle("held", on);
  }

  hint(on: boolean): void {
    this.halo.style.display = on ? "" : "none";
  }

  visible(on: boolean): void {
    this.el.style.display = on ? "" : "none";
    if (!on) this.tether.style.opacity = "0";
  }

  dispose(): void {
    this.el.remove();
    this.tether.remove();
  }
}
