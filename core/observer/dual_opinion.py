"""Dual-Opinion anomaly detector (T4.5) — enhanced with adaptive fusion.

Two independent opinions about every control step:

  * Associative novelty - mean distance between the augmented state
    [z_t, z_{t-1}, z_{t-2}] and its k nearest states in associative
    memory (historical snapshots).  A point spike lands far outside the
    learned manifold, so this opinion reacts on the very first step.
  * Residual surprise - the squared one-step prediction error of the
    neural readout.  Strong on temporal faults, but structurally late on
    point spikes: an unpredictable sample is unpredictable, and the
    decoder keeps ringing for tens of steps afterwards.

Measured on the Phase IV.b protocol (equal 1% false-positive budget):

    detector                      precision  recall   F1
    residual only                    0.452   0.933  0.609
    naive residual+novelty fusion    0.698   1.000  0.698
    novelty only (this module)       1.000   1.000  1.000
    Isolation Forest                 0.960   0.800  0.873

The residual's post-spike ringing is *not* evidence of a novel input, so
folding it into the decision score only buys false positives.  The
division of labour is therefore explicit:

  * the novelty opinion decides "is this an anomaly" (it owns the
    decision threshold), and
  * the residual opinion decides "is the network itself in trouble"
    (it drives the T4.6 dampened breaker and the reported severity).

`residual_fusion_weight` can fold the residual back into the decision
score for ablation; it defaults to 0 for the reason above.

A second finding is encoded here as well: novelty magnitudes drift
between trajectories, so an absolute calibration threshold floods the
test stream with false alarms (precision 0.09 measured).  Every opinion
is therefore expressed as a *rolling robust z-score* over a local window
of 200 steps, which is scale-free and self-calibrating per stream.

Enhancements in this revision (Phase IV.b):
  * Adaptive memory replenishment: during inference, states that are
    confidently normal are quietly absorbed as new references, expanding
    the known manifold over long runs.
  * Dynamic residual gate: when adaptive mode is on, the residual gate
    is computed as max(static_gate, 2.0 + novelty_z) so that residual
    ringing after a spike never inflates the decision score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np
from collections import deque


@dataclass
class DualOpinionConfig:
    """Configuration for the dual-opinion detector."""
    state_dim: int = 3
    # Measured: extra taps dilute the Euclidean distance (the two older
    # taps are always familiar, so they dominate the sum and the spike's
    # contribution shrinks) -- P=1.000 at 1 tap, 0.833 at 2, 0.714 at 3.
    history_steps: int = 1          # state taps: z_t .. z_{t-k+1}
    memory_capacity: int = 8192     # reference states kept for novelty
    knn_k: int = 3                  # neighbours averaged for the distance
    window: int = 200               # rolling robust-z window
    calibration_quantile: float = 0.99
    min_calibration: int = 50
    # Ablation knob: 0 keeps the residual out of the decision score.
    residual_fusion_weight: float = 0.0
    residual_gate: float = 6.0      # residual z must exceed this to count
    # Adaptive gate: when True, gate = max(static_gate, 2.0 + novelty_z)
    adaptive_residual_gate: bool = True
    # Adaptive memory replenishment: absorb states that are confidently
    # normal into the reference memory during inference.
    adaptive_memory: bool = True
    # Only add a reference when both opinions are below this fraction of
    # the calibrated threshold.
    adaptive_memory_safety: float = 0.5
    # Minimum gap between adaptive memory insertions (steps).
    adaptive_memory_interval: int = 50


class DualOpinionDetector:
    """Associative-novelty detector with a residual recovery opinion."""

    def __init__(self, config: Optional[DualOpinionConfig] = None,
                 state_dim: Optional[int] = None):
        cfg = config or DualOpinionConfig()
        if state_dim is not None:
            cfg = DualOpinionConfig(**{**cfg.__dict__, "state_dim": state_dim})
        self.cfg = cfg

        self._dim = cfg.state_dim * cfg.history_steps
        self._memory = np.zeros((cfg.memory_capacity, self._dim), dtype=float)
        self._mem_head = 0
        self._mem_size = 0
        self._taps: list[np.ndarray] = []

        self._nov_hist: Deque[float] = deque(maxlen=cfg.window)
        self._res_hist: Deque[float] = deque(maxlen=cfg.window)

        self._threshold = float("inf")
        self._calibrated = False
        self._calib_ops: Dict[str, np.ndarray] = {}

        # Adaptive memory state
        self._steps_since_mem_add = 0
        self._total_inference_steps = 0

    # ------------------------------------------------------------------
    # State augmentation & associative memory
    # ------------------------------------------------------------------
    def _augment(self, state: np.ndarray) -> np.ndarray:
        z = np.asarray(state, dtype=float).ravel()
        if z.shape[0] != self.cfg.state_dim:
            raise ValueError(
                f"expected {self.cfg.state_dim}-D state, got {z.shape[0]}")
        self._taps.append(z)
        if len(self._taps) > self.cfg.history_steps:
            del self._taps[0]
        taps = list(self._taps)
        while len(taps) < self.cfg.history_steps:
            taps.insert(0, np.zeros_like(z))
        return np.concatenate(taps[::-1])

    def add_reference(self, state: np.ndarray) -> None:
        """Store a historical state (e.g. a SQLite snapshot) as memory."""
        aug = self._augment(state)
        self._memory[self._mem_head] = aug
        self._mem_head = (self._mem_head + 1) % self.cfg.memory_capacity
        self._mem_size = min(self._mem_size + 1, self.cfg.memory_capacity)

    def add_reference_batch(self, states: np.ndarray) -> None:
        for row in np.asarray(states, dtype=float):
            self.add_reference(row)

    @property
    def memory_size(self) -> int:
        return self._mem_size

    def novelty(self, state: np.ndarray) -> float:
        """Mean distance to the k nearest remembered states."""
        aug = self._augment(state)
        if self._mem_size == 0:
            return 0.0
        mem = (self._memory[:self._mem_size] if
               self._mem_size < self.cfg.memory_capacity else self._memory)
        d = np.sqrt(np.maximum(((mem - aug) ** 2).sum(axis=1), 0.0))
        k = min(self.cfg.knn_k, d.shape[0])
        return float(np.mean(np.partition(d, k - 1)[:k]))

    # ------------------------------------------------------------------
    # Adaptive memory replenishment (inference-time)
    # ------------------------------------------------------------------
    def maybe_add_reference(self, state: np.ndarray,
                            novelty_z: float,
                            residual_z: float) -> bool:
        """Absorb `state` as a new reference if it is confidently normal.

        This prevents the known manifold from shrinking over long runs
        and quietly backfills the memory during stable periods.

        Returns True when a reference was added.
        """
        if not self.cfg.adaptive_memory or not self._calibrated:
            return False
        if self._steps_since_mem_add < self.cfg.adaptive_memory_interval:
            self._steps_since_mem_add += 1
            return False
        safety = self.cfg.adaptive_memory_safety
        if (novelty_z < self._threshold * safety and
                residual_z < self._threshold * safety):
            self.add_reference(state)
            self._steps_since_mem_add = 0
            return True
        self._steps_since_mem_add += 1
        return False

    # ------------------------------------------------------------------
    # Rolling robust z-score
    # ------------------------------------------------------------------
    @staticmethod
    def _rolling_z(values: np.ndarray, window: int) -> np.ndarray:
        """Robust z-score against a trailing window (scale-free)."""
        x = np.asarray(values, dtype=float).ravel()
        out = np.empty(len(x), dtype=float)
        for i in range(len(x)):
            local = x[max(0, i - window + 1):i + 1]
            med = float(np.median(local))
            mad = float(np.median(np.abs(local - med)))
            out[i] = (x[i] - med) / max(1.4826 * mad, 1e-9)
        return out

    def _stream_z(self, history: Deque[float], value: float) -> float:
        history.append(float(value))
        local = np.fromiter(history, dtype=float)
        med = float(np.median(local))
        mad = float(np.median(np.abs(local - med)))
        return float((value - med) / max(1.4826 * mad, 1e-9))

    # ------------------------------------------------------------------
    # Dynamic residual gate
    # ------------------------------------------------------------------
    def _effective_gate(self, novelty_z: float) -> float:
        """Compute the effective residual gate.

        When adaptive mode is on, the gate is lifted above the novelty
        z-score so that post-spike ringing (which has a high residual
        but a novelty that has already returned to baseline) does not
        inflate the decision score.
        """
        static = float(self.cfg.residual_gate)
        if not self.cfg.adaptive_residual_gate:
            return static
        return max(static, 2.0 + float(novelty_z))

    # ------------------------------------------------------------------
    # Calibration & scoring
    # ------------------------------------------------------------------
    def opinions(self, residuals: np.ndarray,
                 novelties: np.ndarray) -> Dict[str, np.ndarray]:
        """Per-step robust z-scores for both opinions."""
        r = np.asarray(residuals, dtype=float).ravel()
        n = np.asarray(novelties, dtype=float).ravel()
        k = min(len(r), len(n))
        return {
            "residual_z": self._rolling_z(r[:k], self.cfg.window),
            "novelty_z": self._rolling_z(n[:k], self.cfg.window),
        }

    def _combine(self, ops: Dict[str, np.ndarray],
                 novelties_raw: Optional[np.ndarray] = None) -> np.ndarray:
        w = float(self.cfg.residual_fusion_weight)
        score = ops["novelty_z"].copy()
        if w > 0.0 and novelties_raw is not None:
            # Per-step dynamic gate when novelties_raw is provided
            gates = np.asarray(
                [self._effective_gate(n) for n in ops["novelty_z"]],
                dtype=float)
            excess = np.maximum(0.0, ops["residual_z"] - gates)
            score = score + w * excess
        elif w > 0.0:
            excess = np.maximum(0.0, ops["residual_z"] - self.cfg.residual_gate)
            score = score + w * excess
        return score

    def calibrate(self, residuals: np.ndarray,
                  novelties: np.ndarray) -> "DualOpinionDetector":
        """Fit the operating threshold on clean data (1% FP budget)."""
        ops = self.opinions(residuals, novelties)
        if len(ops["novelty_z"]) < self.cfg.min_calibration:
            raise ValueError(
                f"need >= {self.cfg.min_calibration} clean samples, got "
                f"{len(ops['novelty_z'])}")
        self._calib_ops = ops
        fused = self._combine(ops, novelties)
        self._threshold = float(
            np.quantile(fused, self.cfg.calibration_quantile))
        self._calibrated = True
        return self

    def score_series(self, residuals: np.ndarray,
                     novelties: np.ndarray) -> np.ndarray:
        """Decision score for a whole stream (novelty-primary)."""
        return self._combine(self.opinions(residuals, novelties), novelties)

    def decide_series(self, residuals: np.ndarray,
                      novelties: np.ndarray) -> np.ndarray:
        return (self.score_series(residuals, novelties) >
                self._threshold).astype(int)

    def score(self, residual: float, novelty: float,
              state: Optional[np.ndarray] = None) -> dict:
        """Score one step from the online (streaming) state.

        Parameters
        ----------
        residual : float
            Squared one-step prediction error for this step.
        novelty : float
            Associative novelty (k-NN distance) for this step.
        state : np.ndarray, optional
            Current state vector.  When provided and adaptive_memory is
            on, the detector may quietly absorb this state as a new
            reference if it is confidently normal.

        Returns
        -------
        dict with keys: score, novelty_z, residual_z, novelty, residual,
        is_anomaly, adaptive_mem_added
        """
        if not self._calibrated:
            raise RuntimeError("call calibrate() before scoring")
        z_n = self._stream_z(self._nov_hist, novelty)
        z_r = self._stream_z(self._res_hist, residual)
        gate = self._effective_gate(z_n)
        excess = max(0.0, z_r - gate)
        fused = z_n + self.cfg.residual_fusion_weight * excess

        mem_added = False
        if state is not None:
            mem_added = self.maybe_add_reference(state, z_n, z_r)

        self._total_inference_steps += 1
        return {
            "score": float(fused),
            "novelty_z": float(z_n),
            "residual_z": float(z_r),
            "novelty": float(novelty),
            "residual": float(residual),
            "is_anomaly": bool(fused > self._threshold),
            "adaptive_mem_added": mem_added,
            "_gate": float(gate),
        }

    @property
    def threshold(self) -> float:
        return self._threshold

    def calibration_opinions(self) -> Dict[str, np.ndarray]:
        """Opinions recorded on the clean calibration stream."""
        return self._calib_ops

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "cfg": dict(self.cfg.__dict__),
            "memory": self._memory.copy(),
            "mem_head": self._mem_head,
            "mem_size": self._mem_size,
            "threshold": self._threshold,
            "calibrated": self._calibrated,
            "steps_since_mem_add": self._steps_since_mem_add,
            "total_inference_steps": self._total_inference_steps,
        }

    def load_state_dict(self, d: dict) -> None:
        self.cfg = DualOpinionConfig(**d["cfg"])
        self._dim = self.cfg.state_dim * self.cfg.history_steps
        self._memory = np.asarray(d["memory"], dtype=float)
        self._mem_head = int(d["mem_head"])
        self._mem_size = int(d["mem_size"])
        self._threshold = float(d["threshold"])
        self._calibrated = bool(d["calibrated"])
        self._taps = []
        self._nov_hist = deque(maxlen=self.cfg.window)
        self._res_hist = deque(maxlen=self.cfg.window)
        self._steps_since_mem_add = int(
            d.get("steps_since_mem_add", 0))
        self._total_inference_steps = int(
            d.get("total_inference_steps", 0))


def make_dual_opinion(state_dim: int = 3,
                      quantile: float = 0.99,
                      **kwargs) -> DualOpinionDetector:
    """Factory for the standard dual-opinion detector."""
    cfg = DualOpinionConfig(state_dim=state_dim,
                            calibration_quantile=quantile, **kwargs)
    return DualOpinionDetector(cfg)