"""Feature encoding & orthogonal projection layer.

Cleansed raw state -> z-score normalised state z in [-1, 1]
-> fixed random orthogonal projection P (proj_dim x state_dim)
-> feature vector [P z, tanh(P z)] fed to the neural ensemble.
"""

from __future__ import annotations

import numpy as np


class FeatureEncoder:
    def __init__(self, state_dim: int = 3, proj_dim: int = 8,
                 z_clip: float = 4.0, var_eps: float = 1e-6,
                 ema_alpha: float = 0.02, seed: int = 0):
        self.state_dim = state_dim
        self.proj_dim = proj_dim
        self.z_clip = z_clip
        self.var_eps = var_eps
        self.ema_alpha = ema_alpha
        self.feature_dim = 2 * proj_dim

        rng = np.random.default_rng(seed)
        # fixed random projection with orthonormal columns (proj_dim x state_dim)
        q, _ = np.linalg.qr(rng.standard_normal((proj_dim, state_dim)))
        self.projection = q

        self.mean = np.zeros(state_dim)
        self.var = np.ones(state_dim)
        self.count = 0

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + self.var_eps)

    def observe(self, raw_vec: np.ndarray, weight: float = 1.0) -> None:
        """Update running z-score stats (confidence-weighted EMA)."""
        x = np.asarray(raw_vec, dtype=float)
        if not np.all(np.isfinite(x)) or weight <= 0.0:
            return
        w = float(np.clip(weight, 0.0, 1.0))
        if self.count == 0:
            self.mean = x.copy()
            self.var = np.full(self.state_dim, 1e-3)
        else:
            a = self.ema_alpha * w
            delta = x - self.mean
            self.mean += a * delta
            self.var += a * (delta * delta - self.var)
            self.var = np.maximum(self.var, 1e-8)
        self.count += 1

    def encode_z(self, raw_vec: np.ndarray) -> np.ndarray:
        x = np.asarray(raw_vec, dtype=float)
        z = (x - self.mean) / self.std
        return np.clip(z, -self.z_clip, self.z_clip) / self.z_clip

    def encode(self, raw_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (z_state, feature_vector)."""
        z = self.encode_z(raw_vec)
        p = self.projection @ z
        feats = np.concatenate([p, np.tanh(p)])
        return z, feats

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
            "z_clip": self.z_clip,
            "ema_alpha": self.ema_alpha,
        }

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.asarray(d["mean"], dtype=float).copy()
        self.var = np.asarray(d["var"], dtype=float).copy()
        self.count = int(d.get("count", 0))
        self.z_clip = float(d.get("z_clip", self.z_clip))
        self.ema_alpha = float(d.get("ema_alpha", self.ema_alpha))
