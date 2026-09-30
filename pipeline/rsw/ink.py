"""The wipe's removal field, mirrored on the host one control step at a time.

``EntityPrepTaint._wipe_world`` (rgbs/tasks/rgbs/prep/taint.py) leaves each step's
live contact samples in ``_ut_buf[:n]`` / ``_rate_buf[:n]`` (``n`` in the device
array ``_counts``) and launches ``_wipe_kernel``: every cell takes the rate of the
*nearest* sample within ``reach`` (first minimum wins, strict ``<``), distance on
the tube's surface with the angle wrapped. The buffers are only fresh while
``_touching[0]`` is true.

What is recorded is that per-cell rate. Removal never depends on the current ink
except through the clamp at zero, so the browser can integrate the rates of the
frames *it* plays and get the visitor's own stripe.
"""

from __future__ import annotations

import numpy as np


class InkMirror:
    def __init__(self, taint, dt: float) -> None:
        self.t = taint
        self.dt = float(dt)
        self.cells = np.asarray(taint._cell_ut_np, dtype=np.float64)
        law = np.asarray(taint._law.numpy(), dtype=np.float64)
        self.reach2, self.radius = float(law[0]), float(law[1])
        self.n_cells = len(self.cells)
        self.I = np.ones(self.n_cells)
        self.reset()

    def reset(self) -> None:
        rest = np.asarray(self.t._rest_intensity, dtype=np.float64)[: self.n_cells]
        self.I = rest.copy()

    def rates(self) -> np.ndarray:
        """This step's per-cell removal rate [1/s] (0 where nothing reaches)."""
        out = np.zeros(self.n_cells, dtype=np.float32)
        if not bool(self.t._touching[0]):
            return out
        n = int(self.t._counts.numpy()[0])
        if n <= 0:
            return out
        ut = np.asarray(self.t._ut_buf[:n], dtype=np.float64)
        r = np.asarray(self.t._rate_buf[:n], dtype=np.float64)
        du = ut[None, :, 0] - self.cells[:, None, 0]
        da = ut[None, :, 1] - self.cells[:, None, 1]
        da = (da + np.pi) % (2.0 * np.pi) - np.pi
        d2 = du * du + (da * self.radius) ** 2
        d2 = np.where(d2 <= self.reach2, d2, np.inf)
        best = np.argmin(d2, axis=1)
        reach = np.isfinite(d2[np.arange(self.n_cells), best])
        out[reach] = r[best[reach]]
        return out

    def integrate(self, rate: np.ndarray) -> None:
        live = self.I > 0.0
        self.I[live] = np.maximum(0.0, self.I[live] - rate[live] * self.dt)

    def field(self) -> np.ndarray:
        return np.asarray(self.t._intensity.numpy(), dtype=np.float64)[: self.n_cells]

    def parity(self) -> float:
        return float(np.max(np.abs(self.field() - self.I)))
