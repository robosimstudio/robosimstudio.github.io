"""Syringe: the Panda picks the syringe up off the workbench and drops it through the hole in the
yellow sharps box (``tool_sort/syringe``).
A track demo from the gallery's success (``tool_sort__syringe__R__a0``: the shipped routine, reward
1.0, 1957 steps, in the 09-05 tree: ``pipeline/legacy.sh``). The hero is the task's own
``agentview`` camera, as the gallery chose."""

from __future__ import annotations

from rsw import paths

from ._bench import BenchTrack


class Spec(BenchTrack):
    id = "syringe"
    RUN = paths.GALLERY / "runs" / "tool_sort__syringe__R__a0"
    LEGACY = True
    HERO_CAMERA = "agentview"

    def static(self) -> dict:
        return {}
