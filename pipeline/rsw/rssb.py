"""The RSSB container (pipeline/schema/CONTRACTS.md §1): writer and reader.

A header, a JSON table of contents and 16-byte aligned arrays, gzip-compressed
as a whole. ``q16d`` arrays are quantized to u16, delta-coded along time and
stored as byte planes, which is what makes position streams compress.
"""

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"RSSB"
VERSION = 1
DTYPES = {"f64": np.float64, "f32": np.float32, "f16": np.float16, "i32": np.int32, "u32": np.uint32,
          "i16": np.int16, "u16": np.uint16, "i8": np.int8, "u8": np.uint8}
_NAME = {np.dtype(v): k for k, v in DTYPES.items()}


def _align(n: int, a: int = 16) -> int:
    return n + (-n % a)


def q16d_encode(x: np.ndarray, lo: np.ndarray | None = None, hi: np.ndarray | None = None):
    """(planes bytes, lo, hi, max abs quantization error) for an [N, C] float array."""
    x = np.asarray(x, dtype=np.float64)
    x2 = x.reshape(len(x), -1)
    lo = x2.min(axis=0) if lo is None else np.asarray(lo, dtype=np.float64)
    hi = x2.max(axis=0) if hi is None else np.asarray(hi, dtype=np.float64)
    span = np.where(hi > lo, hi - lo, 1.0)
    v = np.rint((x2 - lo) / span * 65535.0).clip(0, 65535).astype(np.int64)
    err = float(np.max(np.abs(lo + span * v / 65535.0 - x2))) if x2.size else 0.0
    d = np.diff(np.concatenate([np.zeros((1, v.shape[1]), dtype=np.int64), v]), axis=0) % 65536
    d = d.astype(np.uint16).reshape(-1)
    planes = np.concatenate([(d & 0xFF).astype(np.uint8), (d >> 8).astype(np.uint8)]).tobytes()
    return planes, lo, np.where(hi > lo, hi, lo + 1.0), err


def q16d_decode(planes: bytes, shape, lo, hi) -> np.ndarray:
    n = int(np.prod(shape))
    b = np.frombuffer(planes, dtype=np.uint8)
    d = b[:n].astype(np.int64) | (b[n : 2 * n].astype(np.int64) << 8)
    c = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    v = np.cumsum(d.reshape(shape[0], c), axis=0) % 65536
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    return (lo + (hi - lo) * v / 65535.0).astype(np.float32).reshape(shape)


def write(path: Path, arrays: dict[str, np.ndarray | tuple], meta: dict | None = None) -> dict:
    """Write a gzipped RSSB. A value is an array (raw) or ``("q16d", array[, lo, hi])``."""
    toc: dict[str, dict] = {}
    blobs: list[bytes] = []
    off = 0
    report: dict[str, float] = {}
    for name, value in arrays.items():
        if isinstance(value, tuple) and value[0] == "q16d":
            arr = np.asarray(value[1])
            planes, lo, hi, err = q16d_encode(arr, *(value[2:4] if len(value) > 2 else (None, None)))
            entry = {"dtype": "f32", "shape": list(arr.shape), "enc": "q16d", "lo": lo.tolist(), "hi": hi.tolist()}
            data = planes
            report[name] = err
        else:
            arr = np.ascontiguousarray(value)
            if arr.dtype == np.bool_:
                arr = arr.astype(np.uint8)
            entry = {"dtype": _NAME[arr.dtype], "shape": list(arr.shape), "enc": "raw"}
            data = arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes()
        entry["offset"] = off
        entry["bytes"] = len(data)
        toc[name] = entry
        pad = _align(len(data)) - len(data)
        blobs.append(data + b"\x00" * pad)
        off += len(data) + pad
    js = json.dumps({"arrays": toc, "meta": meta or {}}, separators=(",", ":")).encode("utf-8")
    head = MAGIC + struct.pack("<II", VERSION, len(js)) + js
    head += b" " * (_align(len(head)) - len(head))
    raw = head + b"".join(blobs)
    Path(path).write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    return {"raw_bytes": len(raw), "gz_bytes": Path(path).stat().st_size, "q16d_max_err": report}


def read(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    raw = gzip.decompress(Path(path).read_bytes())
    assert raw[:4] == MAGIC, "not an RSSB file"
    version, jl = struct.unpack_from("<II", raw, 4)
    assert version == VERSION
    doc = json.loads(raw[12 : 12 + jl].decode("utf-8"))
    base = _align(12 + jl)
    out = {}
    for name, e in doc["arrays"].items():
        blob = raw[base + e["offset"] : base + e["offset"] + e["bytes"]]
        if e["enc"] == "q16d":
            out[name] = q16d_decode(blob, e["shape"], e["lo"], e["hi"])
        else:
            out[name] = np.frombuffer(blob, dtype=DTYPES[e["dtype"]]).reshape(e["shape"]).copy()
    return out, doc["meta"]
