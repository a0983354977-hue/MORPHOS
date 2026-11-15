"""In-memory trajectory history & snapshot payload assembly.

Keeps a bounded window of recent per-step records for warm-start and
periodic state snapshots (crash recovery).
"""

from __future__ import annotations

import pickle
from collections import deque

import numpy as np


class TrajectoryBuffer:
    """Bounded deque of recent engine-input feature vectors (warm start)
    plus scalar history used for reporting."""

    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self._feats: deque[np.ndarray] = deque(maxlen=capacity)
        self.residuals: deque[float] = deque(maxlen=capacity)
        self.confs: deque[float] = deque(maxlen=capacity)
        self.rates: deque[float] = deque(maxlen=capacity)

    def push_feat(self, feat: np.ndarray) -> None:
        self._feats.append(np.asarray(feat, dtype=float).copy())

    def extend_feats(self, feats: np.ndarray) -> None:
        for row in np.asarray(feats, dtype=float):
            self._feats.append(row.copy())

    def push_stats(self, conf: float, residual: float, rate: float) -> None:
        self.confs.append(float(conf))
        self.residuals.append(float(residual))
        self.rates.append(float(rate))

    def push_step(self, feat: np.ndarray, conf: float, residual: float,
                  rate: float) -> None:
        self.push_feat(feat)
        self.push_stats(conf, residual, rate)

    def recent_feats(self, n: int) -> np.ndarray:
        items = list(self._feats)[-n:]
        if not items:
            return np.zeros((0,))
        return np.vstack(items)

    def __len__(self) -> int:
        return len(self._feats)

    def resid_ema(self, window: int = 100) -> float:
        vals = [r for r in list(self.residuals)[-window:] if np.isfinite(r)]
        return float(np.mean(vals)) if vals else float("nan")


def make_snapshot(step: int, sim_t: float, encoder, predictor,
                  observer, history: TrajectoryBuffer,
                  extras: dict | None = None,
                  warm_feats: np.ndarray | None = None) -> bytes:
    """Serialize everything needed to resume after a crash."""
    if warm_feats is None:
        warm_feats = history.recent_feats(history.capacity)
    payload = {
        "step": int(step),
        "sim_t": float(sim_t),
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "observer": observer.state_dict(),
        "warm_feats": np.asarray(warm_feats, dtype=float).tolist()
        if len(warm_feats) else [],
        "extras": extras or {},
    }
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def load_snapshot(blob: bytes) -> dict:
    return pickle.loads(blob)
