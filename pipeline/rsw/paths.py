"""Where everything lives. One place, so a moved checkout is one edit."""

from __future__ import annotations

import os
import sys
from pathlib import Path

WS = Path(os.environ.get("RSW_WS", Path.home() / "rss_ws"))
RSS = Path(os.environ.get("RSW_RSS", WS / "RoboSimStudio"))
"""The RoboSimStudio checkout the pipeline imports. Never edited from here."""
PHONE = WS / "phone_demo"
CABLE = WS / "cable_wiggle"
PLATES = WS / "plate_rack"
SLIDES = WS / "slides"
GALLERY = WS / "gallery"
DATA = Path(os.environ.get("RSW_DATA", WS / "rsw_data"))
"""Built data. Outside every repo: takes and frames are tens of GB."""
REPO = Path(__file__).resolve().parents[2]
SITE = REPO / "site"
PLAYWRIGHT_PY = WS / "trailer" / ".venv-tools" / "bin" / "python"


def demo_dir(demo: str, *parts: str) -> Path:
    p = DATA.joinpath(demo, *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def use_rss(*extra: Path) -> None:
    """Put the RoboSimStudio checkout (and any sibling tree) on sys.path, first."""
    for p in (RSS, RSS / "rgbs" / "scripts", *extra):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
