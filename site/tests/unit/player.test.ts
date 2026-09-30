import { describe, expect, it } from "vitest";
import { StatePlayer } from "../../src/core/player.ts";
import { FrameStore } from "../../src/core/store.ts";
import { dbChunk } from "./helpers.ts";

// hand[i] = (10 i, 10 i + 1, 10 i + 2): exact in f32, so interpolation is checkable exactly
const store = new FrameStore(dbChunk([[0, 5, 1], [5, 10, 0], [15, 7, 0]]));
const hand = (p: StatePlayer) => Array.from(p.sample("hand", new Float64Array(3)));

describe("StatePlayer", () => {
  it("reports every integer frame crossed, in order, once", () => {
    const p = new StatePlayer(store);
    p.set(5.5);
    expect(p.advance(2)).toEqual([6, 7]);
    expect(p.phi).toBe(7.5);
    expect(p.advance(0.25)).toEqual([]);
    expect(p.advance(1.25)).toEqual([8, 9]);
    expect(p.advance(0.5)).toEqual([]); // 9 was already reported
    expect(p.advance(0)).toEqual([]);
    p.set(15);
    expect(p.advance(4)).toEqual([16, 17, 18, 19]);
  });

  it("clamps at the take's last frame and reports it", () => {
    const p = new StatePlayer(store, 12);
    expect(p.take).toBe(1);
    expect(p.runway()).toBe(2);
    expect(p.atEnd).toBe(false);
    expect(p.advance(10)).toEqual([13, 14]);
    expect(p.phi).toBe(14);
    expect(p.atEnd).toBe(true);
    expect(p.runway()).toBe(0);
    expect(p.advance(3)).toEqual([]);
    expect(p.phi).toBe(14);
  });

  it("never interpolates across takes", () => {
    const p = new StatePlayer(store);
    p.set(13.5);
    expect(hand(p)).toEqual([135, 136, 137]);
    p.set(14);
    expect(hand(p)).toEqual([140, 141, 142]);
    p.set(14.9); // inside the last frame of take 1: clamped, not blended with frame 15
    expect(p.phi).toBe(14);
    expect(hand(p)).toEqual([140, 141, 142]);
    p.advance(5);
    expect(hand(p)).toEqual([140, 141, 142]);
    p.set(15.25);
    expect(p.take).toBe(2);
    expect(hand(p)).toEqual([152.5, 153.5, 154.5]);
    expect(p.frame).toBe(15);
  });

  it("rejects frames outside the store and backwards steps", () => {
    const p = new StatePlayer(store);
    expect(() => p.set(22)).toThrow(RangeError);
    expect(() => p.set(-0.5)).toThrow(RangeError);
    expect(() => p.advance(-1)).toThrow(RangeError);
    expect(() => p.advance(Number.NaN)).toThrow(RangeError);
  });
});
