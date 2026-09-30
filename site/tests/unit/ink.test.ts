import { describe, expect, it } from "vitest";
import { InkField, inkRate } from "../../src/core/ink.ts";
import { rng } from "./helpers.ts";

const R0 = 0.05;
const RMAX = 4;
const DT = 1 / 60;

describe("InkField", () => {
  it("decodes rates by the contract's formula", () => {
    expect(inkRate(0, R0, RMAX)).toBe(0);
    expect(inkRate(255, R0, RMAX)).toBeCloseTo(RMAX, 12);
    const f = new InkField(256, DT);
    const codes = Uint8Array.from({ length: 256 }, (_, v) => v);
    f.applyFrame(codes, R0, RMAX);
    for (let v = 0; v < 256; v++) {
      const rate = R0 * (Math.exp((v / 255) * Math.log(1 + RMAX / R0)) - 1);
      expect(f.I[v]).toBe(Math.fround(Math.max(0, 1 - rate * DT)));
    }
  });

  it("never increases, and bottoms out at zero", () => {
    const u = rng(4);
    const f = new InkField(50, DT);
    const prev = new Float32Array(50);
    for (let k = 0; k < 600; k++) {
      prev.set(f.I);
      const codes = Uint8Array.from({ length: 50 }, () => (u() < 0.5 ? 0 : Math.floor(u() * 256)));
      f.applyFrame(codes, R0, RMAX, u() * 0.05);
      for (let c = 0; c < 50; c++) {
        expect(f.I[c]).toBeLessThanOrEqual(prev[c]);
        expect(f.I[c]).toBeGreaterThanOrEqual(0);
      }
    }
    expect(Math.min(...f.I)).toBe(0);
    expect(() => f.applyFrame(new Uint8Array(50), R0, RMAX, -DT)).toThrow(/dt must be >= 0/);
  });

  it("reads a row out of the flat per-frame array at an offset", () => {
    const ink = new Uint8Array([9, 9, 9, 0, 255, 0, 9, 9, 9]); // three frames of three cells
    const f = new InkField(3, DT);
    f.applyFrame(ink, R0, RMAX, DT, 3);
    expect(Array.from(f.I)).toEqual([1, Math.fround(1 - RMAX * DT), 1]);
  });

  it("takes over an attract frame's state and flags every change", () => {
    const f = new InkField(4, DT);
    f.setFromState(new Uint8Array([0, 51, 255, 128]));
    expect(Array.from(f.I)).toEqual([0, 0.2, 1, 128 / 255].map(Math.fround));
    expect(f.changed).toBe(true);
    f.changed = false;
    f.applyFrame(new Uint8Array([0, 0, 0, 0]), R0, RMAX);
    expect(f.changed).toBe(false);
    f.applyFrame(new Uint8Array([200, 0, 0, 0]), R0, RMAX); // cell 0 is already clean
    expect(f.changed).toBe(false);
    f.applyFrame(new Uint8Array([0, 10, 0, 0]), R0, RMAX);
    expect(f.changed).toBe(true);
    f.reset();
    expect(Array.from(f.I)).toEqual([1, 1, 1, 1]);
  });
});
