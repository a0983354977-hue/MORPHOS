"""Lorenz-63 chaotic data source (external environment adapter).

Integrates the classic Lorenz system with fixed-step RK4 and emits
samples on its own fine time grid.  The core never sees this object
directly - only cleaned samples through the ingest layer.
"""

from __future__ import annotations

import numpy as np


class LorenzSource:
    def __init__(self, sigma: float = 10.0, rho: float = 28.0,
                 beta: float = 8.0 / 3.0, state0=(1.0, 1.0, 1.0),
                 dt: float = 0.001):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.state = np.asarray(state0, dtype=float)
        self.t = 0.0
        self._pending: list[tuple[float, np.ndarray]] = []

    def _rhs(self, s: np.ndarray) -> np.ndarray:
        x, y, z = s
        return np.array([
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z,
        ])

    def _rk4(self, s: np.ndarray, h: float) -> np.ndarray:
        k1 = self._rhs(s)
        k2 = self._rhs(s + 0.5 * h * k1)
        k3 = self._rhs(s + 0.5 * h * k2)
        k4 = self._rhs(s + h * k3)
        return s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def advance_to(self, t_target: float) -> None:
        """Integrate until source time >= t_target, queueing samples."""
        while self.t < t_target - 1e-12:
            self.state = self._rk4(self.state, self.dt)
            self.t += self.dt
            self._pending.append((self.t, self.state.copy()))

    def read_pending(self) -> list[tuple[float, np.ndarray]]:
        out = self._pending
        self._pending = []
        return out

    @property
    def dim(self) -> int:
        return 3
