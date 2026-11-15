"""Real-time health monitor & circuit breaker.

Watches per-chunk neural health metrics (mean/max firing rate, spike
saturation, decoded-output NaN, prediction residual EMA, data confidence)
and trips a breaker before the network can run away.
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np


class CircuitBreaker:
    """States: OK -> WARN -> TRIP (cooldown) -> OK."""

    def __init__(self, rate_warn: float = 250.0, rate_trip: float = 600.0,
                 spike_frac_trip: float = 0.95, resid_ema_warn_ratio: float = 4.0,
                 cooldown_chunks: int = 3, resid_window: int = 50):
        self.rate_warn = rate_warn
        self.rate_trip = rate_trip
        self.spike_frac_trip = spike_frac_trip
        self.resid_ema_warn_ratio = resid_ema_warn_ratio
        self.cooldown_chunks = cooldown_chunks
        self.state = "OK"
        self._cooldown = 0
        self._resid_hist = deque(maxlen=resid_window)
        self._resid_baseline: float | None = None
        self.trip_count = 0
        self.warn_count = 0

    # -- metrics ---------------------------------------------------------
    def observe_residual(self, e: float) -> None:
        if math.isfinite(e):
            self._resid_hist.append(e)

    @property
    def resid_ema(self) -> float:
        if not self._resid_hist:
            return float("nan")
        return float(np.mean(list(self._resid_hist)))

    def _resid_anomalous(self) -> bool:
        if len(self._resid_hist) < 10 or self._resid_baseline is None:
            return False
        cur = float(np.mean(list(self._resid_hist)[-10:]))
        return cur > self.resid_ema_warn_ratio * max(self._resid_baseline, 1e-9)

    def calibrate_residual_baseline(self) -> None:
        """Call after warm-up so the EMA baseline reflects healthy state."""
        if len(self._resid_hist) >= 10:
            self._resid_baseline = self.resid_ema

    # -- evaluation ------------------------------------------------------
    def evaluate(self, stats: dict) -> str | None:
        """stats keys: mean_rate, max_rate, spike_frac, nan_decoded,
        conf_mean, chunk_index.  Returns 'TRIP'/'WARN' event or None."""
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        reasons = []
        if stats.get("nan_decoded", 0) > 0:
            reasons.append("nan_decoded")
        if stats.get("mean_rate", 0.0) > self.rate_trip or \
           stats.get("max_rate", 0.0) > self.rate_trip * 1.5 or \
           stats.get("spike_frac", 0.0) > self.spike_frac_trip:
            reasons.append("firing_rate_runaway")
        if stats.get("mean_rate", 0.0) > self.rate_warn or \
           self._resid_anomalous():
            self.state = "WARN"
            self.warn_count += 1
            return "WARN:" + "+".join(reasons) if reasons else "WARN:residual_or_rate"

        if reasons:
            self.state = "TRIP"
            self.trip_count += 1
            self._cooldown = self.cooldown_chunks
            return "TRIP:" + "+".join(reasons)

        self.state = "OK"
        return None

    def state_dict(self) -> dict:
        return {"state": self.state, "trip_count": self.trip_count,
                "warn_count": self.warn_count,
                "cooldown": self._cooldown,
                "resid_hist": list(self._resid_hist),
                "resid_baseline": self._resid_baseline}

    def load_state_dict(self, d: dict) -> None:
        self.state = d.get("state", "OK")
        self.trip_count = int(d.get("trip_count", 0))
        self.warn_count = int(d.get("warn_count", 0))
        self._cooldown = int(d.get("cooldown", 0))
        self._resid_hist = deque(d.get("resid_hist", []),
                                 maxlen=self._resid_hist.maxlen)
        self._resid_baseline = d.get("resid_baseline")
