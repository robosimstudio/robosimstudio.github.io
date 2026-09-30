"""Virtual visitors: cursor programs that look like a person dragging a ball.

A program is a cursor path in a demo's 2-D drag box, sampled at the control
rate. It is made of primitives a real visitor produces on a touch screen:

* ``reach``  -- a minimum-jerk move to a target at a slow or medium speed;
* ``flick``  -- a fast move (the cursor far outruns the hand, which chases it);
* ``hold``   -- stay put; ``long_hold`` stays long enough for the material to settle;
* ``osc``    -- scrub back and forth along a direction;
* ``coast``  -- a release with velocity: the cursor keeps going and decays
  (``v * exp(-t/tau)``), the same law the browser applies when a finger lets go.

Targets are drawn so every part of the box and every direction gets visited:
a scrambled Sobol sequence over the box, remapped through an optional density
(the wipe wants most of its time in the contact band). Deterministic per seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Box:
    a: tuple[float, float]
    b: tuple[float, float]

    def clip(self, p: np.ndarray) -> np.ndarray:
        return np.array([np.clip(p[0], *self.a), np.clip(p[1], *self.b)])

    @property
    def size(self) -> np.ndarray:
        return np.array([self.a[1] - self.a[0], self.b[1] - self.b[0]])


@dataclass
class GestureConfig:
    box: Box
    hz: float = 60.0
    weights: dict[str, float] = field(
        default_factory=lambda: {"reach": 0.34, "flick": 0.12, "hold": 0.16, "long_hold": 0.12, "osc": 0.16, "coast": 0.10}
    )
    v_reach: tuple[float, float] = (0.02, 0.25)
    """Cursor speed of a reach [m/s]; a person drags slowly to moderately."""
    v_flick: tuple[float, float] = (0.5, 2.0)
    hold: tuple[float, float] = (0.3, 2.0)
    long_hold: tuple[float, float] = (4.0, 10.0)
    osc_amp: tuple[float, float] = (0.005, 0.04)
    osc_hz: tuple[float, float] = (0.5, 3.0)
    osc_s: tuple[float, float] = (1.0, 4.0)
    coast_tau: float = 0.15
    b_density: str = "uniform"
    """How targets spread over the b axis: ``uniform``, or ``contact`` (the wipe:
    60 % of targets in the lowest ``contact_band`` of b, the rest biased low)."""
    contact_band: tuple[float, float] | None = None


def min_jerk(n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, max(2, n))
    return 10 * t**3 - 15 * t**4 + 6 * t**5


class Program:
    """Deterministic cursor stream: ``next_seconds(s)`` appends and returns samples."""

    def __init__(self, cfg: GestureConfig, seed: int, start: np.ndarray | None = None) -> None:
        from scipy.stats import qmc

        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.sobol = qmc.Sobol(d=2, scramble=True, seed=seed)
        self.cur = np.asarray(start if start is not None else self._target(), dtype=np.float64)
        self.kinds = list(cfg.weights)
        w = np.array([cfg.weights[k] for k in self.kinds], dtype=np.float64)
        self.p = w / w.sum()

    # ------------------------------------------------------------- targets
    def _target(self) -> np.ndarray:
        cfg = self.cfg
        u = self.sobol.random(1)[0]
        a = cfg.box.a[0] + u[0] * (cfg.box.a[1] - cfg.box.a[0])
        if cfg.b_density == "contact" and cfg.contact_band is not None:
            lo, hi = cfg.contact_band
            if self.rng.random() < 0.6:
                b = lo + u[1] * (hi - lo)
            else:
                b = hi + (u[1] ** 2) * (cfg.box.b[1] - hi)
        else:
            b = cfg.box.b[0] + u[1] * (cfg.box.b[1] - cfg.box.b[0])
        return cfg.box.clip(np.array([a, b]))

    # ---------------------------------------------------------- primitives
    def _move(self, target: np.ndarray, speed: float) -> np.ndarray:
        d = float(np.linalg.norm(target - self.cur))
        n = max(2, int(round(d / max(speed, 1e-4) * self.cfg.hz)))
        s = min_jerk(n)[:, None]
        path = self.cur[None, :] + s * (target - self.cur)[None, :]
        self.cur = target.copy()
        return path[1:]

    def _hold(self, seconds: float) -> np.ndarray:
        n = max(1, int(round(seconds * self.cfg.hz)))
        return np.repeat(self.cur[None, :], n, axis=0)

    def _osc(self) -> np.ndarray:
        cfg, rng = self.cfg, self.rng
        amp = rng.uniform(*cfg.osc_amp)
        hz = rng.uniform(*cfg.osc_hz)
        secs = rng.uniform(*cfg.osc_s)
        ang = rng.uniform(0, 2 * np.pi)
        d = np.array([np.cos(ang), np.sin(ang)])
        t = np.arange(1, int(secs * cfg.hz) + 1) / cfg.hz
        env = np.minimum(1.0, np.minimum(t, secs - t) / 0.25).clip(0, 1)  # ease in and out
        path = self.cur[None, :] + (amp * env * np.sin(2 * np.pi * hz * t))[:, None] * d[None, :]
        path = np.array([self.cfg.box.clip(p) for p in path])
        self.cur = path[-1].copy()
        return path

    def _coast(self) -> np.ndarray:
        """A flick that is let go: fast move, then coast with v*exp(-t/tau), clipped to the box."""
        cfg, rng = self.cfg, self.rng
        head = self._move(self._target(), rng.uniform(*cfg.v_flick))
        v = (head[-1] - head[-2]) * cfg.hz if len(head) > 1 else np.zeros(2)
        out = [head]
        p = self.cur.copy()
        t = 0.0
        while np.linalg.norm(v) * np.exp(-t / cfg.coast_tau) > 0.01 and t < 1.0:
            t += 1.0 / cfg.hz
            p = cfg.box.clip(p + v * np.exp(-t / cfg.coast_tau) / cfg.hz)
            out.append(p[None, :])
        self.cur = p
        return np.concatenate(out, axis=0)

    def step(self) -> tuple[str, np.ndarray]:
        cfg, rng = self.cfg, self.rng
        kind = self.kinds[int(rng.choice(len(self.kinds), p=self.p))]
        if kind == "reach":
            return kind, self._move(self._target(), rng.uniform(*cfg.v_reach))
        if kind == "flick":
            return kind, self._move(self._target(), rng.uniform(*cfg.v_flick))
        if kind == "hold":
            return kind, self._hold(rng.uniform(*cfg.hold))
        if kind == "long_hold":
            return kind, self._hold(rng.uniform(*cfg.long_hold))
        if kind == "osc":
            return kind, self._osc()
        return kind, self._coast()

    def seconds(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        """At least ``s`` seconds of cursor, as (N,2) positions and (N,) primitive ids."""
        n_target = int(round(s * self.cfg.hz))
        chunks, ids = [], []
        n = 0
        while n < n_target:
            kind, path = self.step()
            chunks.append(path)
            ids.append(np.full(len(path), self.kinds.index(kind), dtype=np.uint8))
            n += len(path)
        return np.concatenate(chunks)[:n_target], np.concatenate(ids)[:n_target]
