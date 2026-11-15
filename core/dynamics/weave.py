"""Multi-source dynamical weave (T2.3).

Wraps the chunked Nengo engine for heterogeneous inputs.  Temporal
coupling between sources emerges from the shared recurrent transform:
every dimension of the ensemble state feeds back into every other,
so the network can learn cross-source dynamical relationships while
the orthogonal projector keeps the input code separable.

Per-source residuals are computed by downstream joint predictors that
slice the shared decoded feature vector - no extra neural hardware is
needed per source.
"""

from __future__ import annotations

import numpy as np

from core.dynamics.engine import NengoDynamics


class WeaveEngine:
    """Thin orchestration layer over NengoDynamics for multi-source D."""

    def __init__(self, feature_dim: int, n_neurons: int = 700,
                 dt: float = 0.005, recurrent_gain: float = 0.5,
                 seed: int = 0):
        self.dt = dt
        self.engine = NengoDynamics(
            feature_dim=feature_dim, n_neurons=n_neurons, dt=dt,
            recurrent_gain=recurrent_gain, seed=seed)
        self.lag: int | None = None

    def run_chunk(self, feats: np.ndarray) -> dict:
        return self.engine.run_chunk(feats)

    def calibrate(self, feats: np.ndarray, decoded: np.ndarray) -> int:
        self.lag = self.engine.estimate_lag(feats, decoded)
        return self.lag

    def reset(self, warm_feats: np.ndarray | None = None) -> None:
        self.engine.reset()
        if warm_feats is not None and len(warm_feats):
            self.engine.run_chunk(np.asarray(warm_feats, dtype=float))

    @property
    def steps_run(self) -> int:
        return self.engine.steps_run
