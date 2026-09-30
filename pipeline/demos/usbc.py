"""USB-C: the YAM pair inserts a USB-C plug into its receptacle (``usbc_insert/port``; right arm the
work arm, left the support). A track
demo from the gallery's success (``usbc_insert__port__MOR__a0``: reward 1.0, 1215 steps, in the 09-05
tree: ``pipeline/legacy.sh``). The hero is the task's own ``overview`` camera, as the gallery chose."""

from __future__ import annotations

from rsw import paths

from ._bench import BenchTrack


class Spec(BenchTrack):
    id = "usbc"
    RUN = paths.GALLERY / "runs" / "usbc_insert__port__MOR__a0"
    LEGACY = True
    HERO_CAMERA = "overview"

    def static(self) -> dict:
        return {}
