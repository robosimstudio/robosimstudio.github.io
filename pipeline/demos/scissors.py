"""Scissors on the hooks: the YAM pair hangs both pairs of scissors on the two hooks
(``tool_hanging/scissors_level2``). A track demo from the gallery's success (``tool_hanging__scissors_level2__GT__a0``:
the suite's ground-truth graph, reward 1.0, 2865 steps, in the 09-05 tree: ``pipeline/legacy.sh``).
The hero is the task's own ``side`` camera, the one the gallery chose (its ``overhead`` makes hung
tools specks)."""

from __future__ import annotations

from rsw import paths

from ._bench import BenchTrack


class Spec(BenchTrack):
    id = "scissors"
    RUN = paths.GALLERY / "runs" / "tool_hanging__scissors_level2__GT__a0"
    LEGACY = True
    HERO_CAMERA = "side"

    def static(self) -> dict:
        return {}
