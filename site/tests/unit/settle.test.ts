import { describe, expect, it } from "vitest";
import { Settle } from "../../src/core/settle.ts";

const still = [0, 0, 0.001];

describe("Settle", () => {
  it("settles after 150 ms under 5 mm/s with little motion to come", () => {
    const s = new Settle();
    expect(s.update(0.004, still, 0.05)).toBe("active");
    expect(s.update(0.004, still, 0.05)).toBe("active");
    expect(s.update(0.004, still, 0.05)).toBe("settling");
  });

  it("counts a hold summed from many control ticks", () => {
    const s = new Settle();
    for (let k = 1; k < 9; k++) expect(s.update(0, still, 1 / 60)).toBe("active");
    expect(s.update(0, still, 1 / 60)).toBe("settling"); // 9 ticks = 150 ms
  });

  it("waits while the hand is still predicted to move 4 mm or more", () => {
    const s = new Settle();
    for (let k = 0; k < 20; k++) expect(s.update(0, [0.003, 0.003, 0], 0.05)).toBe("active");
    expect(s.update(0, [0.002, 0.002, 0.001], 0.05)).toBe("settling");
  });

  it("restarts the hold when the target speeds up in between", () => {
    const s = new Settle();
    s.update(0.004, still, 0.1);
    s.update(0.006, still, 0.1); // above 5 mm/s: the 150 ms start over
    expect(s.update(0.004, still, 0.1)).toBe("active");
    expect(s.update(0.004, still, 0.05)).toBe("settling");
  });

  it("leaves settling only above 15 mm/s (hysteresis)", () => {
    const s = new Settle();
    s.coastEnded();
    expect(s.mode).toBe("settling");
    expect(s.update(0.01, [0.01, 0, 0], 0.05)).toBe("settling");
    expect(s.update(0.015, still, 0.05)).toBe("settling");
    expect(s.update(0.016, still, 0.05)).toBe("active");
    expect(s.update(0.004, still, 0.1)).toBe("active"); // the hold starts from zero again
  });

  it("takes custom thresholds", () => {
    const s = new Settle({ enterSpeed: 0.02, enterHold: 0.3 });
    expect(s.update(0.01, still, 0.2)).toBe("active");
    expect(s.update(0.01, still, 0.1)).toBe("settling");
    s.wake();
    expect(s.mode).toBe("active");
  });
});
