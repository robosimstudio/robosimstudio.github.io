"""Provenance: which code produced an artifact.

The sources are live (RoboSimStudio's tree is uncommitted, phone_demo is edited
in place), so every artifact records the repo HEAD, a hash of its dirty state
and a hash of every file the process imported from the workspace. Read-only git
commands only.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from . import paths


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60).stdout
    except Exception as e:  # noqa: BLE001
        return f"<git failed: {e}>"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(p: Path) -> str:
    return sha(p.read_bytes())


def imported_files() -> dict[str, str]:
    """sha256 of every workspace file this process imported (repo + sibling demos)."""
    roots = [paths.RSS, paths.PHONE, paths.CABLE, paths.PLATES, paths.SLIDES, paths.REPO]
    roots = [r.resolve() for r in roots if r.exists()]
    out: dict[str, str] = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        if ".venv" in p.parts or "site-packages" in p.parts:
            continue
        if any(str(p).startswith(str(r)) for r in roots) and p.suffix == ".py" and p.exists():
            out[str(p)] = file_sha(p)
    return dict(sorted(out.items()))


def record() -> dict:
    head = _git(paths.RSS, "rev-parse", "HEAD").strip()
    status = _git(paths.RSS, "status", "--porcelain")
    diff = _git(paths.RSS, "diff")
    files = imported_files()
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rss_head": head,
        "rss_dirty_sha": sha((status + diff).encode()),
        "files": files,
        "code_sha": sha(json.dumps(files, sort_keys=True).encode()),
    }
