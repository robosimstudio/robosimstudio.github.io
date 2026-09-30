"""The conduit take's solution: ``rgbs/thread``'s own routine, as fixed on a side branch.

At the checkout's HEAD the tunnel routine's pinch lets the cable slide through (09-30: never
latched, 7483 steps, reward 0). The fix is RoboSimStudio ``a9c6e5a6`` ("the tunnel's pinch holds
the rod", 09-16), on ``origin/rgbs/tunnel-pinch-and-seed`` only. Its parent ``6d956993`` is an
ancestor of HEAD, and the working tree's ``routine_tunnel.py`` and ``tunnel/`` are byte-identical
to that parent, so the fixed file drops in unchanged.

Nothing in the checkout is edited: the fixed source is read with ``git show`` and executed as a
module inside the env's *own* copy of the suite package (the env loads task modules as private
``rstask.<hash>`` packages), so its relative imports (``.tunnel.box`` ...) resolve to the very
modules the scene was built from.

    cd ~/rss_ws/RoboSimStudio && PYTHONPATH=~/rss_ws/robosimstudio-website/pipeline \\
      CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/bench_run.py --task rgbs/thread \\
      --solve rsw.tunnel_fix:SolutionTunnelFixed --episodes 0:1 --record --out ... --name ...
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from typing import Any

FIX = "a9c6e5a6"
FILE = "rgbs/tasks/rgbs/routine_tunnel.py"
RSS = Path.home() / "rss_ws" / "RoboSimStudio"


def fixed_routine_class(env):
    """``RoutineTunnel`` from ``FIX``, loaded beside the env's own copy of the routine."""
    shipped = env.task.routine
    pkg = shipped.__module__.rpartition(".")[0]
    name = f"{pkg}.routine_tunnel_{FIX}"
    mod = sys.modules.get(name)
    if mod is None:
        src = subprocess.check_output(["git", "-C", str(RSS), "show", f"{FIX}:{FILE}"], text=True)
        mod = types.ModuleType(name)
        mod.__package__ = pkg
        mod.__file__ = f"<{FIX}>/{FILE}"
        sys.modules[name] = mod
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    cls = mod.RoutineTunnel
    print(f"[tunnel_fix] {shipped.__module__}.{shipped.__qualname__} -> {name}.RoutineTunnel", flush=True)
    return cls


def _solution_base():
    from robosimstudio_bench.solutions.base import SolutionBase
    from robosimstudio_bench.solutions.routines import SolutionRoutine

    return SolutionBase, SolutionRoutine


_Base, _Routine = _solution_base()


class SolutionTunnelFixed(_Routine):
    """``SolutionRoutine`` driving the fixed routine instead of the shipped one."""

    def __init__(self, env, **kwargs: Any) -> None:
        _Base.__init__(self, env, **kwargs)
        self.routine = fixed_routine_class(env)(env, **kwargs)
