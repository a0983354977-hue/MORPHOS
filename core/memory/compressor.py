"""Importance-sampled memory compression (T4.7).

Long runs (100k+ steps) otherwise let the rolling buffer, associative
memory and trajectory logs grow without bound.  Most of that history is
redundant: steady-state stretches add nothing that the last few samples
do not already describe.

This compressor keeps a *fixed-size* summary built from two reserves:

  * a short recent ring (dense coverage of the current context), and
  * a bounded top-K reservoir ranked by importance, where importance
    combines residual magnitude (rare, informative events) with a
    turning-point score (1 - cos between successive state vectors), so
    regime changes survive compression.

Memory is therefore constant regardless of run length, while the
informative tail of the history is preserved exactly.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from collections import deque
from typing import Optional

import numpy as np


@dataclass
class CompressionConfig:
    """Configuration for importance-sampled compression."""
    max_entries: int = 2048        # top-K reservoir size (bounded)
    recent_reserve: int = 256      # dense recent ring (bounded)
    residual_weight: float = 0.7   # importance mix
    turning_weight: float = 0.3
    residual_ema: float = 0.05     # scale tracker for residual importance
    min_importance: float = 0.0    # optional floor


class ImportanceCompressor:
    """Fixed-footprint memory summariser."""

    def __init__(self, feature_dim: int,
                 config: Optional[CompressionConfig] = None):
        self.feature_dim = feature_dim
        self.cfg = config or CompressionConfig()

        self._recent: deque = deque(maxlen=self.cfg.recent_reserve)
        # Min-heap of (importance, step) plus a dict of stored entries.
        self._heap: list[tuple[float, int]] = []
        self._entries: dict[int, tuple[np.ndarray, float, float]] = {}
        self._resid_scale = 1.0
        self._prev: Optional[np.ndarray] = None
        self._total = 0
        self._kept = 0

    # ------------------------------------------------------------------
    def _importance(self, feat: np.ndarray, residual: float) -> float:
        r = abs(float(residual)) if np.isfinite(residual) else 0.0
        self._resid_scale = ((1 - self.cfg.residual_ema) * self._resid_scale +
                             self.cfg.residual_ema * max(r, 1e-9))
        r_score = r / max(self._resid_scale, 1e-9)

        if self._prev is None:
            turn = 0.0
        else:
            a = feat / (np.linalg.norm(feat) + 1e-12)
            b = self._prev / (np.linalg.norm(self._prev) + 1e-12)
            turn = float(max(0.0, 1.0 - float(a @ b)))
        return (self.cfg.residual_weight * r_score +
                self.cfg.turning_weight * turn)

    def push(self, step: int, feat: np.ndarray, residual: float) -> bool:
        """Feed one step; returns True if it entered the top-K reservoir."""
        f = np.asarray(feat, dtype=float).ravel()
        if f.shape[0] != self.feature_dim:
            raise ValueError(
                f"expected {self.feature_dim}-D feature, got {f.shape[0]}")

        imp = self._importance(f, residual)
        self._prev = f.copy()
        self._total += 1
        self._recent.append((int(step), f.copy(), float(residual), imp))

        if imp < self.cfg.min_importance:
            return False

        if len(self._heap) < self.cfg.max_entries:
            heapq.heappush(self._heap, (imp, int(step)))
            self._entries[int(step)] = (f.copy(), float(residual), imp)
            self._kept += 1
            return True

        worst_imp, worst_step = self._heap[0]
        if imp <= worst_imp:
            return False
        heapq.heapreplace(self._heap, (imp, int(step)))
        self._entries.pop(worst_step, None)
        self._entries[int(step)] = (f.copy(), float(residual), imp)
        self._kept += 1
        return True

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Return the compressed history (bounded by construction)."""
        merged: dict[int, tuple[np.ndarray, float, float]] = {}
        for step, f, r, imp in self._recent:
            merged[step] = (f, r, imp)
        merged.update(self._entries)          # top-K wins on overlap
        steps = np.array(sorted(merged), dtype=int)
        feats = np.vstack([merged[s][0] for s in steps]) if len(steps) else \
            np.zeros((0, self.feature_dim))
        residuals = np.array([merged[s][1] for s in steps], dtype=float)
        importances = np.array([merged[s][2] for s in steps], dtype=float)
        return {"steps": steps, "features": feats, "residuals": residuals,
                "importances": importances}

    def compress(self, features: np.ndarray, residuals: np.ndarray,
                 steps: Optional[np.ndarray] = None) -> dict:
        """Compress an existing trajectory in one call."""
        features = np.asarray(features, dtype=float)
        residuals = np.asarray(residuals, dtype=float).ravel()
        if steps is None:
            steps = np.arange(len(features))
        for i in range(len(features)):
            self.push(int(steps[i]), features[i], residuals[i])
        return self.snapshot()

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def total_seen(self) -> int:
        return self._total

    def footprint_entries(self) -> int:
        """Bound on stored entries - independent of run length."""
        return min(self.cfg.max_entries, self.size) + self.cfg.recent_reserve

    def stats(self) -> dict:
        return {
            "total_seen": self._total,
            "reservoir": self.size,
            "recent": len(self._recent),
            "max_entries": self.cfg.max_entries,
            "kept_fraction": (self._kept / self._total) if self._total else 0.0,
        }

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        snap = self.snapshot()
        return {
            "cfg": dict(self.cfg.__dict__),
            "feature_dim": self.feature_dim,
            "snapshot": snap,
            "resid_scale": self._resid_scale,
            "prev": None if self._prev is None else self._prev.copy(),
            "total": self._total,
        }

    def load_state_dict(self, d: dict) -> None:
        self.cfg = CompressionConfig(**d["cfg"])
        self.feature_dim = int(d["feature_dim"])
        self._recent = deque(maxlen=self.cfg.recent_reserve)
        self._heap = []
        self._entries = {}
        snap = d["snapshot"]
        for step, f, r, imp in zip(snap["steps"], snap["features"],
                                   snap["residuals"], snap["importances"]):
            heapq.heappush(self._heap, (float(imp), int(step)))
            self._entries[int(step)] = (np.asarray(f, dtype=float),
                                        float(r), float(imp))
        self._resid_scale = float(d["resid_scale"])
        self._prev = None if d["prev"] is None else np.asarray(d["prev"], dtype=float)
        self._total = int(d["total"])


def make_compressor(feature_dim: int,
                    max_entries: int = 2048,
                    recent_reserve: int = 256,
                    **kwargs) -> ImportanceCompressor:
    """Factory for the standard importance compressor."""
    return ImportanceCompressor(feature_dim, CompressionConfig(
        max_entries=max_entries, recent_reserve=recent_reserve, **kwargs))
