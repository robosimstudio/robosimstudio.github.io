"""Which cards a job may use, and proof that it used only that one.

GPUs 2, 3 and 4 run the live phone demos and are never touched. A card with a
foreign tenant is skipped: a contended card does not error, it looks hung.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

ALLOWED = (0, 1, 5, 6, 7)
DENIED = (2, 3, 4)


@dataclass
class Card:
    index: int
    uuid: str
    used_mib: int
    util: int
    pids: tuple[int, ...]


def _smi(query: str, kind: str = "gpu") -> list[list[str]]:
    flag = "--query-gpu" if kind == "gpu" else "--query-compute-apps"
    out = subprocess.run(
        ["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def cards() -> list[Card]:
    apps: dict[str, list[int]] = {}
    for uuid, pid in _smi("gpu_uuid,pid", "apps"):
        apps.setdefault(uuid, []).append(int(pid))
    out = []
    for idx, uuid, used, util in _smi("index,uuid,memory.used,utilization.gpu"):
        out.append(Card(int(idx), uuid, int(used), int(util), tuple(apps.get(uuid, ()))))
    return out


def free(exclude_pids: set[int] | None = None) -> list[int]:
    """Allowed cards nobody else is on."""
    mine = exclude_pids or set()
    ok = []
    for c in cards():
        if c.index not in ALLOWED:
            continue
        foreign = [p for p in c.pids if p not in mine]
        if not foreign and c.used_mib < 1024:
            ok.append(c.index)
    return ok


def check(index: int) -> None:
    if index in DENIED or index not in ALLOWED:
        raise SystemExit(f"GPU {index} is not ours to use (allowed {ALLOWED}; {DENIED} run the live phone demos)")


def placement(pid: int | None = None) -> list[int]:
    """The card indices a process holds contexts on."""
    pid = pid or os.getpid()
    uuid_to_index = {c.uuid: c.index for c in cards()}
    return sorted({uuid_to_index[u] for u, p in _smi("gpu_uuid,pid", "apps") if int(p) == pid and u in uuid_to_index})


def usage(pid: int | None = None) -> dict[int, int]:
    """MiB a process holds on each card."""
    pid = pid or os.getpid()
    uuid_to_index = {c.uuid: c.index for c in cards()}
    out: dict[int, int] = {}
    for u, p, mem in _smi("gpu_uuid,pid,used_memory", "apps"):
        if int(p) == pid and u in uuid_to_index:
            out[uuid_to_index[u]] = int(mem)
    return out


def assert_only_on(index: int, pid: int | None = None, idle_mib: int = 1024) -> None:
    """The work is on ``index``. OVRTX opens an idle ~570 MiB context on EVERY card
    (measured 2026-09-30; ``active_cuda_gpus`` only picks where it computes and
    CUDA_VISIBLE_DEVICES blinds it), so other cards may hold up to ``idle_mib``."""
    use = usage(pid)
    heavy = sorted(i for i, m in use.items() if m > idle_mib)
    if heavy and heavy != [index]:
        raise SystemExit(f"process {pid or os.getpid()} holds heavy memory on GPUs {heavy} ({use}), expected only [{index}]")
