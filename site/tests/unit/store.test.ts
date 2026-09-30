import { describe, expect, it } from "vitest";
import { FrameStore, TAKE_ATTRACT, TAKE_VISITOR } from "../../src/core/store.ts";
import { dbChunk } from "./helpers.ts";

const wipe = {
  ink: { dim: 4, f: (i: number, c: number) => (i + c) % 256 },
  still: { dim: 1, f: (i: number) => (i % 5 === 0 ? i / 5 : -1) },
};
const state = (i: number, c: number) => (i * 3 + c) % 256;

describe("FrameStore", () => {
  it("holds chunk 0's channels, dims and takes", () => {
    const s = new FrameStore(dbChunk([[0, 5, TAKE_ATTRACT], [5, 10, TAKE_VISITOR], [15, 7, TAKE_VISITOR]], wipe, state));
    expect(s.n).toBe(22);
    expect(s.takes).toEqual([{ start: 0, len: 5, kind: 1 }, { start: 5, len: 10, kind: 0 }, { start: 15, len: 7, kind: 0 }]);
    expect(s.dims).toMatchObject({ q: 3, hand: 3, setpoint: 3, cursor: 2, feat: 4, ink: 4, still: 1 });
    expect(s.hand.length).toBe(22 * 3);
    expect(s.hand[7 * 3 + 2]).toBe(72);
    expect(s.still?.[10]).toBe(2);
    expect(s.still?.[11]).toBe(-1);
    expect(s.deform).toBeNull();
    expect(() => s.float("deform")).toThrow(/no "deform" channel/);
  });

  it("finds the take of any frame, fractional frames by their floor", () => {
    const s = new FrameStore(dbChunk([[0, 5, 1], [5, 10, 0], [15, 7, 0]]));
    const want = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2];
    expect(want.map((_, i) => s.takeOf(i))).toEqual(want);
    expect(s.takeOf(4.99)).toBe(0);
    expect(s.takeOf(5)).toBe(1);
    expect(s.takeOf(-1)).toBe(-1);
    expect(s.takeOf(22)).toBe(-1);
    expect(s.takeOf(Number.NaN)).toBe(-1);
  });

  it("appends later chunks at their global frame0", () => {
    const s = new FrameStore(dbChunk([[0, 5, 1], [5, 10, 0]], wipe, state));
    s.append(dbChunk([[15, 8, 0], [23, 4, 1], [27, 3, 0]], wipe, state));
    expect(s.n).toBe(30);
    expect(s.takes.map((t) => t.start)).toEqual([0, 5, 15, 23, 27]);
    expect(s.takeOf(25)).toBe(3);
    for (let i = 0; i < 30; i++) {
      expect(s.hand[i * 3]).toBe(i * 10);
      expect(s.cursor[i * 2 + 1]).toBe(i + 0.25);
      expect(s.ink?.[i * 4 + 1]).toBe((i + 1) % 256);
    }
    // ink_state rows follow the attract frames of every chunk
    for (const i of [0, 4, 23, 26]) expect(Array.from(s.inkStateRow(i) as Uint8Array)).toEqual([0, 1, 2, 3].map((c) => state(i, c)));
    expect(s.inkStateRow(5)).toBeNull();
    expect(s.inkStateRow(28)).toBeNull();
  });

  it("rejects chunks that are out of order, gapped or inconsistent, and stays unchanged", () => {
    const s = new FrameStore(dbChunk([[0, 5, 1], [5, 10, 0]]));
    expect(() => s.append(dbChunk([[20, 5, 0]]))).toThrow(/starts at frame 20 but the store holds 15 frames/);
    const gapped = dbChunk([[15, 5, 0], [20, 5, 0]]);
    (gapped.arrays.takes.data as Uint32Array)[3] = 21;
    expect(() => s.append(gapped)).toThrow(/does not continue at frame 20/);
    expect(() => s.append(dbChunk([[15, 5, 0]], { feat: { dim: 5, f: () => 0 } }))).toThrow(/"feat" rows are 5 wide, chunk 0's were 4/);
    expect(() => s.append(dbChunk([[15, 5, 0]], { deform: { dim: 6, f: () => 0 } }))).toThrow(/has "deform", which chunk 0 lacked/);
    expect(() => s.append(dbChunk([[15, 5, 0]], { cursor: null }))).toThrow(/has no "cursor"/);
    expect(() => s.append(dbChunk([[15, 5, 0]], { hand: { dim: 3, dtype: "f64", f: () => 0 } }))).toThrow(
      /"hand" decodes to Float64Array, expected Float32Array/,
    );
    expect(() => new FrameStore(dbChunk([[0, 5, 1]], wipe))).toThrow(/5 attract frames and "ink" but no "ink_state"/);
    expect(s.n).toBe(15);
    expect(s.takes).toHaveLength(2);
  });
});
