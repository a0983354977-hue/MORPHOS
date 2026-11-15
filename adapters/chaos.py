"""Chaos injector adapter.

Wraps any sample source and corrupts its output stream the way the
real world does: dropped samples, gaussian noise, NaN/Inf injection,
extreme spikes, and multi-second dropouts.  Everything downstream
(ingest/aligner/encoder/dynamics) must survive this untouched.
"""

from __future__ import annotations

import numpy as np


class ChaosInjector:
    def __init__(self, source, seed: int = 0,
                 noise_sigma: float = 0.0, noise_prob: float = 0.0,
                 drop_prob: float = 0.0, nan_prob: float = 0.0,
                 spike_prob: float = 0.0, spike_scale: float = 1e4):
        self.source = source
        self.rng = np.random.default_rng(seed)
        self.noise_sigma = noise_sigma
        self.noise_prob = noise_prob
        self.drop_prob = drop_prob
        self.nan_prob = nan_prob
        self.spike_prob = spike_prob
        self.spike_scale = spike_scale
        self._dropout_start = -1.0
        self._dropout_until = -1.0
        self.stats = {"dropped": 0, "noisy": 0, "nan": 0, "spike": 0,
                      "dropout": 0}

    def set_dropout(self, t_start: float, duration: float) -> None:
        self._dropout_start = t_start
        self._dropout_until = t_start + duration

    @property
    def dim(self) -> int:
        return self.source.dim

    def advance_to(self, t_target: float) -> None:
        self.source.advance_to(t_target)

    def read_pending(self):
        """Return list of (t, vec_or_None).  None = sample destroyed."""
        out = []
        for t, vec in self.source.read_pending():
            out.append((t, self.corrupt(t, vec)))
        return out

    def corrupt(self, t: float, vec: np.ndarray):
        """Apply the configured corruption to one externally supplied sample."""
        return self._corrupt(t, vec)

    def _corrupt(self, t: float, vec: np.ndarray):
        if self._dropout_start <= t <= self._dropout_until:
            self.stats["dropout"] += 1
            return None
        r = self.rng.random()
        if r < self.drop_prob:
            self.stats["dropped"] += 1
            return None
        v = vec.copy()
        r = self.rng.random()
        if r < self.nan_prob:
            self.stats["nan"] += 1
            v[self.rng.integers(0, v.size)] = np.nan
            return v
        if r < self.nan_prob + self.spike_prob:
            self.stats["spike"] += 1
            v[self.rng.integers(0, v.size)] *= self.spike_scale
            return v
        if self.noise_prob > 0 and self.rng.random() < self.noise_prob:
            self.stats["noisy"] += 1
            v = v + self.rng.normal(0.0, self.noise_sigma, size=v.shape)
        return v
