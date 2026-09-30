import { describe, expect, it } from "vitest";
import { Blender, HALFLIFE_PCA, Inertializer, wrapPi } from "../../src/core/blender.ts";

const out = () => new Float64Array(3);

describe("Inertializer", () => {
  it("is continuous at the switch: the output starts at the source", () => {
    const b = new Inertializer({ dim: 3 });
    const src = [0.3, -1.2, 2];
    const dst = [0.1, 0.4, 1.5];
    b.transition(src, [0, 0, 0], dst, [0, 0, 0]);
    const o = b.output(dst, out());
    for (let d = 0; d < 3; d++) expect(o[d]).toBeCloseTo(src[d], 15); // dst + (src - dst): equal to an ulp
  });

  it("keeps the source velocity at the switch", () => {
    const b = new Inertializer({ dim: 1, halflife: 0.1 });
    b.transition([0], [2], [0], [-1]); // offset velocity 3
    const h = 1e-7;
    b.apply(h);
    // output = dst(t) + o(t) with dst moving at -1: slope -1 + 3 = the source's 2
    expect((-1 * h + b.o[0]) / h).toBeCloseTo(2, 5);
  });

  it("decays below 1% of the offset after 5 halflives, however dt is sliced", () => {
    const a = new Inertializer({ dim: 2, halflife: 0.1 });
    const b = new Inertializer({ dim: 2, halflife: 0.1 });
    a.transition([1, -3], [0, 0], [0, 0], [0, 0]);
    b.transition([1, -3], [0, 0], [0, 0], [0, 0]);
    a.apply(0.5);
    for (let k = 0; k < 30; k++) b.apply(0.5 / 30);
    expect(Math.abs(a.o[0])).toBeLessThan(0.01);
    expect(Math.abs(a.o[1])).toBeLessThan(0.03);
    // exact closed form: e^{-y t} (1 + y t) of the start, y = 2 ln2 / halflife
    const yt = 2 * Math.LN2 * 5;
    expect(a.o[0]).toBeCloseTo(Math.exp(-yt) * (1 + yt), 12);
    expect(b.o[0]).toBeCloseTo(a.o[0], 12);
    expect(b.ov[1]).toBeCloseTo(a.ov[1], 12);
  });

  it("re-bases a second transition on the current output", () => {
    const b = new Inertializer({ dim: 2, halflife: 0.1 });
    b.transition([1, 1], [0, 0], [0, 0], [0, 0]);
    b.apply(0.05);
    // the output currently follows stream A = (0.2, 0.2); switch to stream B = (-1, 3)
    const shown = Array.from(b.output([0.2, 0.2], out()).subarray(0, 2));
    b.transition([0.2, 0.2], [0, 0], [-1, 3], [0, 0]);
    const after = Array.from(b.output([-1, 3], out()).subarray(0, 2));
    expect(after[0]).toBeCloseTo(shown[0], 15);
    expect(after[1]).toBeCloseTo(shown[1], 15);
  });

  it("wraps continuous joints to (-pi, pi] in the offset only", () => {
    const b = new Inertializer({ dim: 2, wrap: [true, false] });
    b.transition([3.1, 3.1], [0, 0], [-3.1, -3.1], [0, 0]);
    expect(b.o[0]).toBeCloseTo(6.2 - 2 * Math.PI, 15);
    expect(b.o[1]).toBeCloseTo(6.2, 15);
    const o = b.output([-3.1, -3.1], out());
    expect(o[0]).toBeCloseTo(3.1 - 2 * Math.PI, 15); // the same angle as the source, reached the short way
    expect(wrapPi(Math.PI)).toBe(Math.PI);
    expect(wrapPi(-Math.PI)).toBe(Math.PI);
    expect(wrapPi(7)).toBeCloseTo(7 - 2 * Math.PI, 15);
  });
});

describe("Blender", () => {
  it("keeps a halflife per channel group and decays them together", () => {
    const b = new Blender({ q: { dim: 2 }, hand: { dim: 3, halflife: 0.1 }, pca: { dim: 3, halflife: HALFLIFE_PCA } });
    b.transition("q", [1, 0], [0, 0], [0, 0], [0, 0]);
    b.transition("pca", [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]);
    b.apply(0.12);
    const q = b.get("q").o[0];
    const pca = b.get("pca").o[0];
    const yt = (h: number) => (2 * Math.LN2 * 0.12) / h;
    expect(q).toBeCloseTo(Math.exp(-yt(0.1)) * (1 + yt(0.1)), 12);
    expect(pca).toBeCloseTo(Math.exp(-yt(HALFLIFE_PCA)) * (1 + yt(HALFLIFE_PCA)), 12);
    expect(Array.from(b.output("hand", [1, 2, 3], out()))).toEqual([1, 2, 3]);
    b.reset();
    expect(b.get("q").o[0]).toBe(0);
    expect(() => b.get("deform")).toThrow(/no channel group "deform"/);
  });
});
