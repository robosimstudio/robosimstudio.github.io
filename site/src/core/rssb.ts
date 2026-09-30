// The RSSB container (pipeline/schema/CONTRACTS.md §1), read from already-gunzipped bytes.
//
//   offset 0   'RSSB'        4 bytes ASCII
//   offset 4   version = 1   u32 LE
//   offset 8   json_len      u32 LE
//   offset 12  JSON          json_len bytes UTF-8, 0x20-padded to a 16-byte boundary
//   D = align16(12 + json_len) starts the data section; every array sits at D + offset.
//
// JSON is {"arrays": {name: {dtype, shape, offset, bytes, enc, lo?, hi?}}, "meta": {...}}.
// enc "raw" stores the little-endian typed array (f16 is widened to f32 here);
// enc "q16d" stores an [N, C] f32 array quantized to u16 over [lo[c], hi[c]],
// delta-coded along axis 0 (mod 65536, from v[-1] = 0) and split into byte planes:
// the N*C low bytes, then the N*C high bytes.
//
// `buildRssb` is the matching writer. The site never writes containers; it exists
// so the reader has round-trip tests against an encoder that follows the contract.

export type DType = "f64" | "f32" | "f16" | "i32" | "u32" | "i16" | "u16" | "i8" | "u8";
export type Enc = "raw" | "q16d";

export type NumArray =
  | Float64Array
  | Float32Array
  | Int32Array
  | Uint32Array
  | Int16Array
  | Uint16Array
  | Int8Array
  | Uint8Array;

export interface RssbArray {
  /** Decoded values, row-major. f16 and q16d arrays decode to Float32Array. */
  data: NumArray;
  shape: number[];
}

/** A parsed container; ``meta`` is free-form JSON owned by the pipeline. */
export interface Rssb<M = any> {
  meta: M;
  arrays: Record<string, RssbArray>;
}

/** One array's header entry, exactly as the JSON stores it. */
export interface RssbArraySpec {
  dtype: DType;
  shape: number[];
  offset: number;
  bytes: number;
  enc: Enc;
  lo?: number[];
  hi?: number[];
}

export const RSSB_VERSION = 1;
const MAGIC = "RSSB";

type Ctor = { new (buffer: ArrayBufferLike, byteOffset: number, length: number): NumArray; new (length: number): NumArray };

const ITEM: Record<DType, number> = { f64: 8, f32: 4, f16: 2, i32: 4, u32: 4, i16: 2, u16: 2, i8: 1, u8: 1 };
const CTOR: Record<Exclude<DType, "f16">, Ctor> = {
  f64: Float64Array,
  f32: Float32Array,
  i32: Int32Array,
  u32: Uint32Array,
  i16: Int16Array,
  u16: Uint16Array,
  i8: Int8Array,
  u8: Uint8Array,
};

const align16 = (n: number): number => (n + 15) & ~15;
const product = (shape: number[]): number => shape.reduce((a, b) => a * b, 1);
const hostIsLittleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

// ------------------------------------------------------------------------ f16

let f16Table: Float32Array | null = null;

/** All 65536 half-precision bit patterns as f32, built once (subnormals, ±0, ±inf, NaN exact). */
export function f16Lut(): Float32Array {
  if (f16Table) return f16Table;
  const lut = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;
    if (e === 0) lut[h] = s * m * 2 ** -24; // zero and subnormals: m units of 2^-24
    else if (e === 31) lut[h] = m ? Number.NaN : s * Number.POSITIVE_INFINITY;
    else lut[h] = s * (1024 + m) * 2 ** (e - 25); // (1 + m/1024) 2^(e-15)
  }
  f16Table = lut;
  return lut;
}

/**
 * The half-precision bit pattern nearest to ``x`` (round half to even; overflow to
 * ±inf, NaN to the quiet NaN 0x7e00). Every intermediate here is exact in f64,
 * so there is no double rounding.
 */
export function f16Bits(x: number): number {
  if (Number.isNaN(x)) return 0x7e00;
  const sign = x < 0 || Object.is(x, -0) ? 0x8000 : 0;
  const a = Math.abs(x);
  if (a >= 65520) return sign | 0x7c00; // halfway between 65504 and 2^16 rounds (to even) up
  if (a < 2 ** -14) return sign | roundHalfEven(a * 2 ** 24); // subnormal; 1024 is the smallest normal
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e--;
  else if (2 ** (e + 1) <= a) e++;
  // a mantissa that rounds up to 1024 carries into the exponent, as the bit layout does
  return sign | (((e + 15) << 10) + roundHalfEven((a / 2 ** e - 1) * 1024));
}

function roundHalfEven(v: number): number {
  const f = Math.floor(v);
  const r = v - f;
  if (r > 0.5) return f + 1;
  if (r < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

// --------------------------------------------------------------------- reader

/**
 * Parse an RSSB container. Raw arrays are zero-copy views into ``bytes`` when their
 * start is aligned for the element type (always, for a freshly decompressed buffer),
 * so ``bytes`` must not be reused afterwards. Throws on anything off-contract.
 */
export function parseRssb<M = any>(bytes: Uint8Array): Rssb<M> {
  if (!hostIsLittleEndian) throw new Error("RSSB: big-endian hosts are not supported");
  if (bytes.length < 12) throw new Error(`RSSB: ${bytes.length} bytes is shorter than the 12-byte header`);
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  if (magic !== MAGIC) {
    const gz = bytes[0] === 0x1f && bytes[1] === 0x8b ? " (the bytes are still gzip-compressed)" : "";
    throw new Error(`RSSB: bad magic ${JSON.stringify(magic)}, expected "RSSB"${gz}`);
  }
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = dv.getUint32(4, true);
  if (version !== RSSB_VERSION) throw new Error(`RSSB: version ${version} is not supported (this reader reads ${RSSB_VERSION})`);
  const jsonLen = dv.getUint32(8, true);
  if (12 + jsonLen > bytes.length) throw new Error(`RSSB: json_len ${jsonLen} runs past the end of ${bytes.length} bytes`);
  let head: { arrays?: Record<string, RssbArraySpec>; meta?: M };
  try {
    head = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + jsonLen)));
  } catch (err) {
    throw new Error(`RSSB: header JSON does not parse: ${(err as Error).message}`);
  }
  if (!head || typeof head.arrays !== "object" || head.arrays === null) throw new Error('RSSB: header has no "arrays" object');
  const dataStart = align16(12 + jsonLen);
  const arrays: Record<string, RssbArray> = {};
  for (const [name, spec] of Object.entries(head.arrays)) arrays[name] = decodeArray(name, spec, bytes, dataStart);
  return { meta: head.meta ?? ({} as M), arrays };
}

function decodeArray(name: string, spec: RssbArraySpec, bytes: Uint8Array, dataStart: number): RssbArray {
  const where = `RSSB array "${name}"`;
  const { dtype, shape, offset, enc } = spec;
  if (!Object.hasOwn(ITEM, dtype)) throw new Error(`${where}: unknown dtype ${JSON.stringify(dtype)}`);
  if (!Array.isArray(shape) || !shape.every((d) => Number.isInteger(d) && d >= 0))
    throw new Error(`${where}: bad shape ${JSON.stringify(shape)}`);
  if (!Number.isInteger(offset) || offset < 0 || offset % 16 !== 0)
    throw new Error(`${where}: offset ${offset} is not a non-negative multiple of 16`);
  const count = product(shape);
  const stored = enc === "raw" ? count * ITEM[dtype] : enc === "q16d" ? 2 * count : -1;
  if (stored < 0) throw new Error(`${where}: unknown enc ${JSON.stringify(enc)}`);
  if (spec.bytes !== stored) throw new Error(`${where}: stores ${spec.bytes} bytes, ${enc} ${dtype}${JSON.stringify(shape)} needs ${stored}`);
  const start = dataStart + offset;
  if (start + stored > bytes.length) throw new Error(`${where}: bytes [${start}, ${start + stored}) run past the end (${bytes.length})`);
  const src = bytes.subarray(start, start + stored);

  if (enc === "q16d") {
    if (dtype !== "f32") throw new Error(`${where}: q16d arrays have logical dtype f32, not ${dtype}`);
    if (shape.length < 1) throw new Error(`${where}: a q16d array needs at least one axis`);
    const channels = product(shape.slice(1));
    const { lo, hi } = spec;
    if (!Array.isArray(lo) || !Array.isArray(hi) || lo.length !== channels || hi.length !== channels)
      throw new Error(`${where}: q16d needs lo and hi arrays of length ${channels}`);
    return { data: decodeQ16d(src, shape[0], channels, lo, hi), shape: shape.slice() };
  }
  if (dtype === "f16") {
    const lut = f16Lut();
    const out = new Float32Array(count);
    for (let i = 0, j = 0; i < count; i++, j += 2) out[i] = lut[src[j] | (src[j + 1] << 8)];
    return { data: out, shape: shape.slice() };
  }
  const C = CTOR[dtype];
  const byteStart = bytes.byteOffset + start;
  const data = byteStart % ITEM[dtype] === 0 ? new C(bytes.buffer, byteStart, count) : new C(src.slice().buffer, 0, count);
  return { data, shape: shape.slice() };
}

/** Undo q16d: per channel, v = cumsum(d) mod 65536, then x = lo + (hi - lo) * v / 65535. */
export function decodeQ16d(planes: Uint8Array, n: number, channels: number, lo: number[], hi: number[]): Float32Array {
  const m = n * channels;
  const out = new Float32Array(m);
  for (let c = 0; c < channels; c++) {
    const l = lo[c];
    const span = hi[c] - l;
    let v = 0;
    for (let k = c; k < m; k += channels) {
      v = (v + (planes[k] | (planes[m + k] << 8))) & 0xffff;
      out[k] = l + (span * v) / 65535;
    }
  }
  return out;
}

// --------------------------------------------------------------------- writer

export interface RssbInput {
  /** Logical values, row-major (q16d: the f32 values before quantization). */
  data: ArrayLike<number>;
  shape: number[];
  dtype: DType;
  enc?: Enc;
  /** q16d range per channel; each defaults to that channel's min / max. */
  lo?: number[];
  hi?: number[];
}

/** Write a container per the contract (the reader's test counterpart). */
export function buildRssb(arrays: Record<string, RssbInput>, meta: unknown = {}): Uint8Array {
  const specs: Record<string, RssbArraySpec> = {};
  const blobs: Uint8Array[] = [];
  let offset = 0;
  for (const [name, a] of Object.entries(arrays)) {
    const enc = a.enc ?? "raw";
    const count = product(a.shape);
    if (a.data.length !== count) throw new Error(`buildRssb "${name}": ${a.data.length} values for shape ${JSON.stringify(a.shape)}`);
    const spec: RssbArraySpec = { dtype: a.dtype, shape: a.shape.slice(), offset, bytes: 0, enc };
    let blob: Uint8Array;
    if (enc === "q16d") {
      if (a.dtype !== "f32") throw new Error(`buildRssb "${name}": q16d arrays have dtype f32`);
      const channels = product(a.shape.slice(1));
      const { planes, lo, hi } = encodeQ16d(a.data, a.shape[0], channels, a.lo, a.hi);
      blob = planes;
      spec.lo = lo;
      spec.hi = hi;
    } else if (a.dtype === "f16") {
      blob = new Uint8Array(2 * count);
      for (let i = 0; i < count; i++) {
        const h = f16Bits(a.data[i]);
        blob[2 * i] = h & 0xff;
        blob[2 * i + 1] = h >> 8;
      }
    } else {
      const typed = new CTOR[a.dtype](count);
      for (let i = 0; i < count; i++) typed[i] = a.data[i];
      blob = new Uint8Array(typed.buffer, 0, typed.byteLength);
    }
    spec.bytes = blob.length;
    specs[name] = spec;
    blobs.push(blob);
    offset = align16(offset + blob.length);
  }
  const json = new TextEncoder().encode(JSON.stringify({ arrays: specs, meta }));
  const dataStart = align16(12 + json.length);
  const out = new Uint8Array(dataStart + offset);
  out.set([0x52, 0x53, 0x53, 0x42], 0);
  const dv = new DataView(out.buffer);
  dv.setUint32(4, RSSB_VERSION, true);
  dv.setUint32(8, json.length, true);
  out.set(json, 12);
  out.fill(0x20, 12 + json.length, dataStart);
  let i = 0;
  for (const spec of Object.values(specs)) out.set(blobs[i++], dataStart + spec.offset);
  return out;
}

/** Quantize to u16 over [lo, hi] per channel, delta-code along axis 0, split into byte planes. */
export function encodeQ16d(
  data: ArrayLike<number>,
  n: number,
  channels: number,
  loIn?: number[],
  hiIn?: number[],
): { planes: Uint8Array; lo: number[]; hi: number[] } {
  const lo: number[] = [];
  const hi: number[] = [];
  for (let c = 0; c < channels; c++) {
    let l = Number.POSITIVE_INFINITY;
    let h = Number.NEGATIVE_INFINITY;
    for (let k = c; k < n * channels; k += channels) {
      l = Math.min(l, data[k]);
      h = Math.max(h, data[k]);
    }
    lo.push(loIn ? loIn[c] : n ? l : 0);
    hi.push(hiIn ? hiIn[c] : n ? h : 0);
  }
  const m = n * channels;
  const planes = new Uint8Array(2 * m);
  for (let c = 0; c < channels; c++) {
    const span = hi[c] - lo[c];
    let prev = 0;
    for (let k = c; k < m; k += channels) {
      const v = span > 0 ? Math.min(65535, Math.max(0, Math.round(((data[k] - lo[c]) / span) * 65535))) : 0;
      const d = (v - prev) & 0xffff;
      prev = v;
      planes[k] = d & 0xff;
      planes[m + k] = d >> 8;
    }
  }
  return { planes, lo, hi };
}
