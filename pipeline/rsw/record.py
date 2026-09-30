"""Per-control-step recording, written as fixed-size npz chunks.

Each chunk is written to a temporary name and renamed, so a killed process
leaves only whole chunks behind; ``meta.json`` is written last and marks the
take complete. Uncompressed: the simulation is the bottleneck, not the disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


class Recorder:
    def __init__(self, out: Path, chunk: int = 600) -> None:
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.chunk = int(chunk)
        self.rows: dict[str, list[np.ndarray]] = {}
        self.n = 0
        self.written = 0

    def step(self, row: dict[str, np.ndarray | float | int]) -> None:
        for k, v in row.items():
            self.rows.setdefault(k, []).append(np.asarray(v))
        self.n += 1
        if self.n - self.written * self.chunk >= self.chunk:
            self.flush()

    def flush(self) -> None:
        if not self.rows or not next(iter(self.rows.values())):
            return
        arrays = {k: np.stack(v) for k, v in self.rows.items()}
        tmp = self.out / f".chunk_{self.written:04d}.tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, self.out / f"chunk_{self.written:04d}.npz")
        self.rows = {}
        self.written += 1

    def close(self, meta: dict) -> None:
        self.flush()
        meta = dict(meta, steps=self.n, chunks=self.written)
        tmp = self.out / ".meta.tmp.json"
        tmp.write_text(json.dumps(meta, indent=1, default=_json_default))
        os.replace(tmp, self.out / "meta.json")


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def load(take: Path) -> dict[str, np.ndarray]:
    """All chunks of a take, concatenated."""
    files = sorted(Path(take).glob("chunk_*.npz"))
    parts: dict[str, list[np.ndarray]] = {}
    for f in files:
        with np.load(f) as z:
            for k in z.files:
                parts.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v) for k, v in parts.items()}
