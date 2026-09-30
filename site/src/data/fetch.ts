// Fetching demo data. Pages serves .gz as application/gzip with no
// Content-Encoding, so the page gunzips itself (DecompressionStream, fflate as
// fallback). The magic-byte check keeps a server that DID decode it working.

import { gunzipSync } from "fflate";
import { parseRssb } from "../core/rssb.ts";

export const DATA = `${import.meta.env.BASE_URL}data/`;

export async function fetchBytes(url: string, onProgress?: (done: number, total: number) => void): Promise<Uint8Array> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = Number(r.headers.get("Content-Length") ?? 0);
  if (!r.body || !onProgress) return new Uint8Array(await r.arrayBuffer());
  const reader = r.body.getReader();
  const parts: Uint8Array[] = [];
  let done = 0;
  for (;;) {
    const { value, done: end } = await reader.read();
    if (end) break;
    parts.push(value);
    done += value.length;
    onProgress(done, total);
  }
  const out = new Uint8Array(done);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

export async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes; // already decoded upstream
  if (typeof DecompressionStream !== "undefined") {
    const s = new Blob([bytes as BlobPart]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Uint8Array(await new Response(s).arrayBuffer());
  }
  return gunzipSync(bytes);
}

export async function fetchRssb(url: string, onProgress?: (done: number, total: number) => void) {
  return parseRssb(await gunzip(await fetchBytes(url, onProgress)));
}

export async function fetchJson<T>(url: string, noCache = false): Promise<T> {
  const r = await fetch(url, noCache ? { cache: "no-cache" } : undefined);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return (await r.json()) as T;
}
