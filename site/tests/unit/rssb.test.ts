import { describe, expect, it } from "vitest";
import { buildRssb, type DType, f16Bits, f16Lut, parseRssb } from "../../src/core/rssb.ts";
import { gauss, rng } from "./helpers.ts";

const u = rng(7);
const g = gauss(u);

/** Reference f16 decoder, independent of the LUT: rebuild the f32 bit pattern. */
function refF16(h: number): number {
  const sign = (h & 0x8000) << 16;
  let e = (h >> 10) & 0x1f;
  let m = h & 0x3ff;
  let bits: number;
  if (e === 0x1f) bits = sign | 0x7f800000 | (m << 13);
  else if (e !== 0) bits = sign | ((e - 15 + 127) << 23) | (m << 13);
  else if (m === 0) bits = sign;
  else {
    // subnormal: shift the mantissa up to an implicit leading one
    e = 1;
    while (!(m & 0x400)) {
      m <<= 1;
      e--;
    }
    bits = sign | ((e - 15 + 127) << 23) | ((m & 0x3ff) << 13);
  }
  return new Float32Array(new Uint32Array([bits >>> 0]).buffer)[0];
}

/** Native references when the runtime has them (Node >= 24). */
const nativeGetF16 = (DataView.prototype as unknown as { getFloat16?: (o: number, le: boolean) => number }).getFloat16;
const f16round = (Math as unknown as { f16round?: (x: number) => number }).f16round;

const sameNumber = (a: number, b: number) => (Number.isNaN(a) ? Number.isNaN(b) : Object.is(a, b));

describe("RSSB raw arrays", () => {
  it("round-trip every dtype, shape and the meta block", () => {
    const cases: [DType, number[], number[]][] = [
      ["f64", [2, 3], [Math.PI, -1e300, 5e-324, 0, -0, 1 / 3]],
      ["f32", [5], [Math.fround(0.1), -3.5, 3.4028234663852886e38, 1.401298464324817e-45, 0]],
      ["f16", [2, 2], [65504, -2.5, 2 ** -24, 0.333251953125]],
      ["i32", [3], [-2147483648, 2147483647, 0]],
      ["u32", [3], [0, 4294967295, 123456789]],
      ["i16", [2, 2, 1], [-32768, 32767, -1, 7]],
      ["u16", [2], [0, 65535]],
      ["i8", [4], [-128, 127, -1, 0]],
      ["u8", [3], [0, 255, 17]],
      ["f32", [0, 4], []],
    ];
    const arrays = Object.fromEntries(cases.map(([dtype, shape, data], k) => [`a${k}`, { data, shape, dtype }]));
    const meta = { frame0: 1200, deform: { mode: "pca" }, name: "wipe ✓" };
    const r = parseRssb(buildRssb(arrays, meta));
    expect(r.meta).toEqual(meta);
    const ctor = { f64: Float64Array, f32: Float32Array, f16: Float32Array, i32: Int32Array, u32: Uint32Array, i16: Int16Array, u16: Uint16Array, i8: Int8Array, u8: Uint8Array };
    cases.forEach(([dtype, shape, data], k) => {
      const a = r.arrays[`a${k}`];
      expect(a.shape).toEqual(shape);
      expect(a.data).toBeInstanceOf(ctor[dtype]);
      expect(Array.from(a.data).every((x, i) => sameNumber(x, data[i]))).toBe(true);
    });
  });

  it("reads a container assembled by hand from the contract text", () => {
    // arrays: "v" raw f32 [2] at offset 0; "q" q16d [2, 2] at offset 16 over lo=[0, -1], hi=[1, 1]
    const json = JSON.stringify({
      arrays: {
        v: { dtype: "f32", shape: [2], offset: 0, bytes: 8, enc: "raw" },
        q: { dtype: "f32", shape: [2, 2], offset: 16, bytes: 8, enc: "q16d", lo: [0, -1], hi: [1, 1] },
      },
      meta: { k: 1 },
    });
    const D = Math.ceil((12 + json.length) / 16) * 16;
    const bytes = new Uint8Array(D + 32);
    bytes.set([0x52, 0x53, 0x53, 0x42, 1, 0, 0, 0, json.length & 0xff, json.length >> 8, 0, 0]);
    bytes.set(new TextEncoder().encode(json), 12);
    bytes.fill(0x20, 12 + json.length, D);
    bytes.set(new Uint8Array(new Float32Array([1.5, -2]).buffer), D);
    // values v = [[0, 65535], [65535, 0]]: deltas [0, 65535, 65535, 1] (mod 65536), as low then high byte planes
    bytes.set([0x00, 0xff, 0xff, 0x01, 0x00, 0xff, 0xff, 0x00], D + 16);
    const r = parseRssb(bytes);
    expect(r.meta).toEqual({ k: 1 });
    expect(Array.from(r.arrays.v.data)).toEqual([1.5, -2]);
    expect(Array.from(r.arrays.q.data)).toEqual([0, 1, 1, -1]);
  });

  it("writes q16d as the contract's byte planes", () => {
    const bytes = buildRssb({ q: { data: [0, 1, 1, 0], shape: [2, 2], dtype: "f32", enc: "q16d", lo: [0, 0], hi: [1, 1] } });
    const jsonLen = new DataView(bytes.buffer).getUint32(8, true);
    const D = Math.ceil((12 + jsonLen) / 16) * 16;
    expect(Array.from(bytes.subarray(D, D + 8))).toEqual([0x00, 0xff, 0xff, 0x01, 0x00, 0xff, 0xff, 0x00]);
    expect(Array.from(bytes.subarray(12 + jsonLen, D)).every((b) => b === 0x20)).toBe(true);
  });

  it("decodes arrays that are not aligned in memory", () => {
    const src = buildRssb({ x: { data: [1.25, -7, 3e9], shape: [3], dtype: "f64" }, y: { data: [-5, 9], shape: [2], dtype: "i16" } });
    const buf = new Uint8Array(src.length + 3);
    buf.set(src, 3);
    const r = parseRssb(buf.subarray(3));
    expect(Array.from(r.arrays.x.data)).toEqual([1.25, -7, 3e9]);
    expect(Array.from(r.arrays.y.data)).toEqual([-5, 9]);
  });
});

describe("RSSB f16", () => {
  it("decodes all 65536 bit patterns like the reference decoders", () => {
    const all = Array.from({ length: 65536 }, (_, h) => h);
    const bytes = buildRssb({ h: { data: all, shape: [65536], dtype: "u16" } });
    // same-length edit of the header: store the patterns as f16 instead of u16
    const head = new TextDecoder().decode(bytes.subarray(0, 200));
    const at = head.indexOf('"dtype":"u16"') + '"dtype":"'.length;
    bytes[at] = "f".charCodeAt(0);
    const dec = parseRssb(bytes).arrays.h.data as Float32Array;
    const dv = new DataView(new Uint16Array(all).buffer);
    let bad = 0;
    for (let h = 0; h < 65536; h++) {
      if (!sameNumber(dec[h], refF16(h))) bad++;
      if (nativeGetF16 && !sameNumber(dec[h], nativeGetF16.call(dv, 2 * h, true))) bad++;
    }
    expect(bad).toBe(0);
    expect(dec[0x7c00]).toBe(Number.POSITIVE_INFINITY);
    expect(dec[0xfc00]).toBe(Number.NEGATIVE_INFINITY);
    expect(dec[0x7e00]).toBeNaN();
    expect(Object.is(dec[0x8000], -0)).toBe(true);
    expect(dec[0x0001]).toBe(2 ** -24);
    expect(dec[0x7bff]).toBe(65504);
  });

  it("encodes every representable value back to its own pattern", () => {
    const lut = f16Lut();
    let bad = 0;
    for (let h = 0; h < 65536; h++) if (!Number.isNaN(lut[h]) && f16Bits(lut[h]) !== h) bad++;
    expect(bad).toBe(0);
    expect(f16Bits(Number.NaN)).toBe(0x7e00);
  });

  it("rounds to nearest even within half an ulp", () => {
    const halfUlp = (x: number) => {
      const a = Math.abs(x);
      return a < 2 ** -14 ? 2 ** -25 : 2 ** (Math.floor(Math.log2(a)) - 11);
    };
    const xs: number[] = [];
    for (let k = 0; k < 20000; k++) xs.push(g() * 2 ** Math.floor(u() * 42 - 28)); // |x| < 65504
    const r = parseRssb(buildRssb({ x: { data: xs, shape: [xs.length], dtype: "f16" } })).arrays.x.data;
    xs.forEach((x, i) => {
      expect(Math.abs(r[i] - x)).toBeLessThanOrEqual(halfUlp(x));
      if (f16round) expect(r[i]).toBe(f16round(x));
    });
    // ties go to the even mantissa, overflow goes to infinity
    const ties = [1 + 2 ** -11, 1 + 3 * 2 ** -11, 2 ** -25, 3 * 2 ** -25, 65519.99, 65520, -1e6];
    const want = [1, 1 + 2 ** -9, 0, 2 ** -23, 65504, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY];
    const t = parseRssb(buildRssb({ t: { data: ties, shape: [ties.length], dtype: "f16" } })).arrays.t.data;
    expect(Array.from(t)).toEqual(want);
  });
});

describe("RSSB q16d", () => {
  const roundTrip = (data: number[], shape: number[], lo?: number[], hi?: number[]) => {
    const bytes = buildRssb({ x: { data, shape, dtype: "f32", enc: "q16d", lo, hi } });
    const r = parseRssb(bytes);
    const spec = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + new DataView(bytes.buffer).getUint32(8, true)))).arrays.x;
    return { out: r.arrays.x.data as Float32Array, shape: r.arrays.x.shape, lo: spec.lo as number[], hi: spec.hi as number[] };
  };
  /** Worst excess over half a quantum + 1e-6 (+ the f32 storage rounding |x| 2^-24 when ``f32``). */
  const maxErr = (data: number[], out: Float32Array, C: number, lo: number[], hi: number[], f32 = false) => {
    let worst = Number.NEGATIVE_INFINITY;
    data.forEach((x, k) => {
      const c = k % C;
      const bound = (hi[c] - lo[c]) / 65535 / 2 + 1e-6 + (f32 ? Math.abs(x) * 2 ** -24 : 0);
      worst = Math.max(worst, Math.abs(out[k] - x) - bound);
    });
    return worst;
  };
  /** N steps of mean-reverting walks, one per scale (every falling step wraps its delta). */
  const walks = (N: number, scales: number[], revert: number) => {
    const data: number[] = [];
    const x = scales.map((s) => s * g());
    for (let n = 0; n < N; n++)
      for (let c = 0; c < scales.length; c++) {
        x[c] = revert * x[c] + scales[c] * g();
        data.push(x[c]);
      }
    return data;
  };

  it("round-trips random walks within half a quantum", () => {
    // joint-angle and PCA magnitudes (|x| < 16), where f32 storage rounding stays under 1e-6
    const scales = [1e-3, 0.05, 0.3, 0.8, 0.1, 0.5];
    const data = walks(5000, scales, 0.95);
    expect(Math.max(...data.map(Math.abs))).toBeLessThan(16);
    const { out, shape, lo, hi } = roundTrip(data, [5000, scales.length]);
    expect(shape).toEqual([5000, scales.length]);
    expect(maxErr(data, out, scales.length, lo, hi)).toBeLessThanOrEqual(0);
    // wide ranges: the decoded values are f32, so storage adds up to |x| 2^-24
    const wide = walks(5000, [2, 30, 400], 0.999);
    const w = roundTrip(wide, [5000, 3]);
    expect(maxErr(wide, w.out, 3, w.lo, w.hi, true)).toBeLessThanOrEqual(0);
  });

  it("round-trips step signals, 1-D arrays, constant channels and explicit ranges", () => {
    const steps: number[] = [];
    for (let n = 0; n < 1000; n++) steps.push(n % 200 < 100 ? -3 : 7, Math.floor(n / 250), 0.25);
    const s = roundTrip(steps, [1000, 3]);
    expect(maxErr(steps, s.out, 3, s.lo, s.hi)).toBeLessThanOrEqual(0);
    for (let k = 2; k < steps.length; k += 3) expect(s.out[k]).toBe(0.25); // lo == hi: exact

    const oneD = Array.from({ length: 777 }, (_, n) => Math.sin(n / 10));
    const d = roundTrip(oneD, [777]);
    expect(d.lo).toHaveLength(1);
    expect(maxErr(oneD, d.out, 1, d.lo, d.hi)).toBeLessThanOrEqual(0);

    const e = roundTrip([0.5, -0.5, 2], [3, 1], [-1], [1]); // 2 is outside [lo, hi]: saturates
    expect(Array.from(e.out)).toEqual([Math.fround(-1 + (2 * 49151) / 65535), Math.fround(-1 + (2 * 16384) / 65535), 1]);
  });
});

describe("RSSB errors", () => {
  const good = () => buildRssb({ x: { data: [1, 2], shape: [2], dtype: "f32" } });
  const withHeader = (edit: (spec: Record<string, unknown>) => void) => {
    const json = { arrays: { x: { dtype: "f32", shape: [2], offset: 0, bytes: 8, enc: "raw" } as Record<string, unknown> }, meta: {} };
    edit(json.arrays.x);
    const text = new TextEncoder().encode(JSON.stringify(json));
    const D = Math.ceil((12 + text.length) / 16) * 16;
    const b = new Uint8Array(D + 16);
    b.set([0x52, 0x53, 0x53, 0x42, 1, 0, 0, 0, text.length, 0, 0, 0]);
    b.set(text, 12);
    return b;
  };

  it("rejects a bad magic, and says when the bytes are still gzipped", () => {
    const b = good();
    b[0] = 0x51;
    expect(() => parseRssb(b)).toThrow(/bad magic/);
    const gz = new Uint8Array([0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0]);
    expect(() => parseRssb(gz)).toThrow(/still gzip-compressed/);
    expect(() => parseRssb(new Uint8Array(5))).toThrow(/shorter than the 12-byte header/);
  });

  it("rejects an unknown version and a truncated header", () => {
    const b = good();
    b[4] = 2;
    expect(() => parseRssb(b)).toThrow(/version 2 is not supported/);
    const t = good();
    new DataView(t.buffer).setUint32(8, 1 << 20, true);
    expect(() => parseRssb(t)).toThrow(/runs past the end/);
  });

  it("rejects off-contract array entries", () => {
    expect(() => parseRssb(withHeader((s) => (s.dtype = "f8")))).toThrow(/unknown dtype/);
    expect(() => parseRssb(withHeader((s) => (s.enc = "zstd")))).toThrow(/unknown enc/);
    expect(() => parseRssb(withHeader((s) => (s.offset = 8)))).toThrow(/multiple of 16/);
    expect(() => parseRssb(withHeader((s) => (s.bytes = 12)))).toThrow(/needs 8/);
    expect(() => parseRssb(withHeader((s) => (s.shape = [9])))).toThrow(/needs 36/);
    expect(() => parseRssb(withHeader((s) => Object.assign(s, { shape: [100], bytes: 400 })))).toThrow(/run past the end/);
    expect(() => parseRssb(withHeader((s) => Object.assign(s, { dtype: "u16", enc: "q16d", bytes: 4 })))).toThrow(/logical dtype f32/);
    expect(() => parseRssb(withHeader((s) => Object.assign(s, { enc: "q16d", bytes: 4, lo: [0, 0], hi: [1] })))).toThrow(
      /lo and hi arrays of length 1/,
    );
  });
});
