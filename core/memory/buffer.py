"""Rolling Temporal Buffer with EMA-based residual tracking (T3.1).

Efficient sliding-window history that preserves multi-scale temporal context
without unbounded memory growth. Combines:
  - Fixed-capacity circular buffer for recent feature vectors
  - Exponential Moving Average (EMA) of residuals at multiple time scales
  - Optional compressed history via downsampling for long-range patterns

Design goals:
  - O(1) push / O(1) recent access
  - Constant memory regardless of run length
  - Multi-scale EMA captures both fast transients and slow drifts
  - Compatible with existing TrajectoryBuffer API for warm-start
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from core.memory.thread_safe_buffer import ThreadSafeBuffer, synchronized


@dataclass
class TemporalConfig:
    """Configuration for the rolling temporal buffer."""
    capacity: int = 2048              # max feature vectors in circular buffer
    ema_scales: tuple[int, ...] = (10, 50, 200, 1000)  # EMA window sizes
    compress_after: int = 4096        # steps before downsampling starts
    compress_ratio: int = 4           # keep 1 sample per N when compressing
    max_compressed: int = 1024        # max compressed history entries


class RollingTemporalBuffer(ThreadSafeBuffer):
    """Sliding-window temporal buffer with multi-scale EMA tracking.
    
    Maintains:
      - Circular buffer of recent feature vectors (for warm-start replay)
      - Multi-scale EMA of residuals (for anomaly detection & trend analysis)
      - Optional compressed long-term history (for pattern matching)
    """
    
    def __init__(self, feature_dim: int, config: Optional[TemporalConfig] = None):
        super().__init__()
        self.feature_dim = feature_dim
        self.cfg = config or TemporalConfig()
        
        # --- Primary circular buffer (recent features for warm-start) ---
        self._feats = np.zeros((self.cfg.capacity, feature_dim), dtype=float)
        self._head = 0
        self._size = 0
        
        # --- Multi-scale EMA of residuals ---
        self._ema_values = {scale: 0.0 for scale in self.cfg.ema_scales}
        self._ema_initialized = {scale: False for scale in self.cfg.ema_scales}
        self._ema_alpha = {scale: 2.0 / (scale + 1) for scale in self.cfg.ema_scales}
        self._ema_count = {scale: 0 for scale in self.cfg.ema_scales}
        
        # --- Compressed long-term history (downsampled) ---
        self._compressed_feats: list[np.ndarray] = []
        self._compressed_residuals: list[float] = []
        self._compressed_steps: list[int] = []
        self._total_steps = 0
        
        # --- Scalar histories (bounded, for reporting) ---
        max_scalar = max(self.cfg.capacity, max(self.cfg.ema_scales))
        self.residuals = deque(maxlen=max_scalar)
        self.confs = deque(maxlen=max_scalar)
        self.rates = deque(maxlen=max_scalar)
        
    # ------------------------------------------------------------------
    # Feature vector management
    # ------------------------------------------------------------------
    @synchronized
    def push_feat(self, feat: np.ndarray) -> None:
        """Add feature vector to circular buffer."""
        feat = np.asarray(feat, dtype=float).ravel()
        assert feat.shape[0] == self.feature_dim, f"Expected {self.feature_dim}, got {feat.shape[0]}"
        
        self._feats[self._head] = feat
        self._head = (self._head + 1) % self.cfg.capacity
        self._size = min(self._size + 1, self.cfg.capacity)
        self._total_steps += 1
        
        # Periodic compression of older history
        if (self._total_steps > self.cfg.compress_after and 
            self._total_steps % self.cfg.compress_ratio == 0):
            self._compress_oldest()
    
    def _compress_oldest(self) -> None:
        """Downsample oldest buffer entries into compressed history."""
        if self._size < self.cfg.capacity:
            return  # buffer not full yet
            
        # Take the entry that's about to be overwritten
        oldest_idx = self._head  # this is where next write goes, so oldest is here
        compressed_feat = self._feats[oldest_idx].copy()
        
        self._compressed_feats.append(compressed_feat)
        # We'll pair with residual when push_stats is called
        self._compressed_steps.append(self._total_steps - self.cfg.capacity)
        
        # Trim compressed history if needed
        if len(self._compressed_feats) > self.cfg.max_compressed:
            # Remove oldest compressed entries (keep most recent)
            excess = len(self._compressed_feats) - self.cfg.max_compressed
            del self._compressed_feats[:excess]
            del self._compressed_residuals[:excess]
            del self._compressed_steps[:excess]
    
    @synchronized
    def recent_feats(self, n: int) -> np.ndarray:
        """Get last n feature vectors in chronological order (oldest first)."""
        n = min(n, self._size)
        if n == 0:
            return np.zeros((0, self.feature_dim))
        
        if n == self._size and self._size == self.cfg.capacity:
            # Full buffer, return in order
            return np.vstack([self._feats[self._head:], self._feats[:self._head]])
        elif n <= self._head:
            return self._feats[self._head - n:self._head]
        else:
            # Wrap around
            part1 = self._feats[self._head - n:self._head] if self._head > 0 else np.zeros((0, self.feature_dim))
            part2 = self._feats[self.cfg.capacity - (n - len(part1)):]
            return np.vstack([part2, part1])
    
    @synchronized
    def get_warm_feats(self, n: int) -> np.ndarray:
        """Get recent features for warm-start (alias for recent_feats)."""
        return self.recent_feats(n)
    
    @property
    @synchronized
    def capacity(self) -> int:
        return self.cfg.capacity
    
    @synchronized
    def __len__(self) -> int:
        return self._size
    
    # ------------------------------------------------------------------
    # Multi-scale EMA residual tracking
    # ------------------------------------------------------------------
    @synchronized
    def push_residual(self, residual: float) -> None:
        """Update all EMA scales with new residual value."""
        if not np.isfinite(residual):
            return
            
        for scale in self.cfg.ema_scales:
            alpha = self._ema_alpha[scale]
            if not self._ema_initialized[scale]:
                self._ema_values[scale] = residual
                self._ema_initialized[scale] = True
                self._ema_count[scale] = 1
            else:
                # Bias-corrected EMA
                self._ema_count[scale] += 1
                self._ema_values[scale] = (alpha * residual + 
                                           (1 - alpha) * self._ema_values[scale])
    
    @synchronized
    def get_ema(self, scale: int) -> float:
        """Get EMA value for a specific scale. Returns NaN if not initialized."""
        if scale not in self._ema_values or not self._ema_initialized[scale]:
            return float("nan")
        return self._ema_values[scale]
    
    @synchronized
    def get_all_ema(self) -> dict[int, float]:
        """Get all EMA values as a dict."""
        return {scale: self.get_ema(scale) for scale in self.cfg.ema_scales}
    
    @synchronized
    def resid_ema(self, window: int = 100) -> float:
        """Legacy compatibility: EMA over recent residuals (approx)."""
        # Find closest configured scale
        best_scale = min(self.cfg.ema_scales, key=lambda s: abs(s - window))
        return self.get_ema(best_scale)
    
    # ------------------------------------------------------------------
    # Scalar statistics (confidence, residual, rate)
    # ------------------------------------------------------------------
    @synchronized
    def push_stats(self, conf: float, residual: float, rate: float) -> None:
        """Push scalar statistics (bounded deques)."""
        self.confs.append(float(conf))
        self.residuals.append(float(residual))
        self.rates.append(float(rate))
        
        # Also update EMA
        self.push_residual(residual)
        
        # Pair with compressed features if available
        if (self._compressed_feats and 
            len(self._compressed_residuals) < len(self._compressed_feats)):
            self._compressed_residuals.append(float(residual))
    
    @synchronized
    def push_step(self, feat: np.ndarray, conf: float, residual: float,
                  rate: float) -> None:
        """Convenience: push feature and stats together."""
        self.push_feat(feat)
        self.push_stats(conf, residual, rate)
    
    # ------------------------------------------------------------------
    # Compressed history access (for associative retrieval)
    # ------------------------------------------------------------------
    @synchronized
    def get_compressed_history(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (features, residuals, steps) for compressed long-term history."""
        if not self._compressed_feats:
            return (np.zeros((0, self.feature_dim)), 
                    np.zeros(0), np.zeros(0, dtype=int))
        return (np.vstack(self._compressed_feats),
                np.asarray(self._compressed_residuals, dtype=float),
                np.asarray(self._compressed_steps, dtype=int))
    
    @synchronized
    def get_recent_history(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return recent (features, residuals, steps) from main buffer."""
        n = min(n, self._size, len(self.residuals))
        if n == 0:
            return (np.zeros((0, self.feature_dim)), 
                    np.zeros(0), np.zeros(0, dtype=int))
        
        feats = self.recent_feats(n)
        # Residuals are in chronological order in deque
        residuals = np.asarray(list(self.residuals)[-n:], dtype=float)
        steps = np.arange(self._total_steps - n, self._total_steps, dtype=int)
        return feats, residuals, steps
    
    # ------------------------------------------------------------------
    # Serialization (for snapshots)
    # ------------------------------------------------------------------
    @synchronized
    def state_dict(self) -> dict:
        """Serialize buffer state for checkpointing."""
        return {
            "feature_dim": self.feature_dim,
            "cfg": {
                "capacity": self.cfg.capacity,
                "ema_scales": self.cfg.ema_scales,
                "compress_after": self.cfg.compress_after,
                "compress_ratio": self.cfg.compress_ratio,
                "max_compressed": self.cfg.max_compressed,
            },
            "feats": self._feats.copy(),
            "head": self._head,
            "size": self._size,
            "total_steps": self._total_steps,
            "ema_values": self._ema_values.copy(),
            "ema_initialized": self._ema_initialized.copy(),
            "ema_count": self._ema_count.copy(),
            "compressed_feats": [f.copy() for f in self._compressed_feats],
            "compressed_residuals": self._compressed_residuals.copy(),
            "compressed_steps": self._compressed_steps.copy(),
            "residuals": list(self.residuals),
            "confs": list(self.confs),
            "rates": list(self.rates),
        }
    
    @synchronized
    def load_state_dict(self, d: dict) -> None:
        """Restore buffer state from checkpoint."""
        self.feature_dim = d["feature_dim"]
        cfg = d["cfg"]
        self.cfg = TemporalConfig(
            capacity=cfg["capacity"],
            ema_scales=tuple(cfg["ema_scales"]),
            compress_after=cfg["compress_after"],
            compress_ratio=cfg["compress_ratio"],
            max_compressed=cfg["max_compressed"],
        )
        
        self._feats = np.asarray(d["feats"], dtype=float)
        self._head = int(d["head"])
        self._size = int(d["size"])
        self._total_steps = int(d["total_steps"])
        self._ema_values = {int(k): float(v) for k, v in d["ema_values"].items()}
        self._ema_initialized = {int(k): bool(v) for k, v in d["ema_initialized"].items()}
        self._ema_count = {int(k): int(v) for k, v in d["ema_count"].items()}
        self._compressed_feats = [np.asarray(f, dtype=float) for f in d["compressed_feats"]]
        self._compressed_residuals = [float(x) for x in d["compressed_residuals"]]
        self._compressed_steps = [int(x) for x in d["compressed_steps"]]
        
        max_scalar = max(self.cfg.capacity, max(self.cfg.ema_scales))
        self.residuals = deque(d["residuals"], maxlen=max_scalar)
        self.confs = deque(d["confs"], maxlen=max_scalar)
        self.rates = deque(d["rates"], maxlen=max_scalar)
        
        # Recompute alphas
        self._ema_alpha = {scale: 2.0 / (scale + 1) for scale in self.cfg.ema_scales}


def make_temporal_buffer(feature_dim: int, 
                         capacity: int = 2048,
                         ema_scales: tuple[int, ...] = (10, 50, 200, 1000)) -> RollingTemporalBuffer:
    """Factory function for backward-compatible buffer creation."""
    config = TemporalConfig(capacity=capacity, ema_scales=ema_scales)
    return RollingTemporalBuffer(feature_dim, config)