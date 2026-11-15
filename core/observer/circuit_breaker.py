"""Dampened circuit breaker & self-healing recovery (T4.6).

The Phase I breaker reacted to a runaway by rebuilding the Nengo
simulator and replaying a warm-up window.  That is a hard discontinuity:
the decoded state jumps, the readout is momentarily invalid, and the
residual spikes again (which in turn raises follow-up false alarms).

This module keeps the same health decision logic but replaces the rebuild
with a *dampened clamp* on the network's input drive:

  1. on TRIP the drive is multiplied by `initial_damping` and the factor
     relaxes back to 1.0 geometrically over `ramp_steps`;
  2. a cooldown prevents a cascade of trips while the network settles;
  3. the simulator object is never rebuilt, so no membrane state is lost
     and the residual decays smoothly instead of jumping.

Two alternatives were measured and rejected on the Phase IV.b protocol:
blending the decoded state toward a pre-trip reference raised the clean
residual median 29x (precision 0.43 -> 0.14), and decaying the frozen
readout's weights on every trip permanently degraded it.  Both perturb
the readout with inputs it was never trained on; damping the drive does
not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.observer.observer import CircuitBreaker


@dataclass
class DampenConfig:
    """Configuration for dampened recovery."""
    ramp_steps: int = 10            # length of the recovery ramp
    initial_damping: float = 0.4    # input drive multiplier at TRIP
    decay: float = 0.6              # per-step decay of the damping factor
    weight_decay: float = 0.0       # readout shrink at TRIP (0 = leave it)
    memory_boost: float = 1.0       # optional memory-reliance multiplier
    min_damping: float = 0.02       # below this the ramp is finished
    cooldown_steps: int = 15        # minimum gap between trips


class DampenedCircuitBreaker:
    """TRIP detection (delegated) + smooth dampened recovery."""

    def __init__(self, dampen: Optional[DampenConfig] = None,
                 health: Optional[CircuitBreaker] = None):
        self.cfg = dampen or DampenConfig()
        self.health = health or CircuitBreaker()
        self._step = 0
        self._alpha = 0.0
        self._ref: Optional[np.ndarray] = None
        self._active = False
        self.trip_count = 0
        self._cooldown = 0

    # -- health monitoring (unchanged semantics) ------------------------
    def observe_residual(self, e: float) -> None:
        self.health.observe_residual(e)

    def evaluate(self, stats: dict) -> Optional[str]:
        return self.health.evaluate(stats)

    # -- dampening ------------------------------------------------------
    @property
    def active(self) -> bool:
        return self._active

    @property
    def alpha(self) -> float:
        return self._alpha

    def trip(self, step: int, reference: Optional[np.ndarray] = None) -> None:
        """Start a dampened recovery from `reference` (pre-trip state)."""
        self._step = int(step)
        self._alpha = float(self.cfg.initial_damping)
        self._ref = (None if reference is None
                     else np.asarray(reference, dtype=float).copy())
        self._active = True
        self.trip_count += 1
        self._cooldown = int(self.cfg.ramp_steps + self.cfg.cooldown_steps)

    def advance(self, steps: int = 1) -> None:
        """Advance the ramp by `steps` control steps."""
        for _ in range(int(steps)):
            if self._cooldown > 0:
                self._cooldown -= 1
            if not self._active:
                continue
            self._alpha *= self.cfg.decay
            self._step += 1
            if self._alpha < self.cfg.min_damping:
                self._alpha = 0.0
                self._active = False

    def can_trip(self) -> bool:
        """True when a new TRIP is allowed (ramp and cooldown elapsed)."""
        return not self._active and self._cooldown <= 0

    def input_gain(self) -> float:
        """Multiplier applied to the network's input drive during recovery.

        Damping the drive (rather than rewriting the decoded state or the
        frozen readout) lets the membrane potential decay naturally back to
        baseline: the reservoir is still driven by the real signal, just
        softer, so nothing the readout sees is out of distribution.
        """
        if not self._active or self.cfg.initial_damping <= 0:
            return 1.0
        frac = float(np.clip(self._alpha / self.cfg.initial_damping, 0.0, 1.0))
        return float(1.0 - (1.0 - self.cfg.initial_damping) * frac)

    def blend(self, decoded: np.ndarray,
              reference: Optional[np.ndarray] = None) -> np.ndarray:
        """Blend decoded state toward the reference by the current alpha."""
        d = np.asarray(decoded, dtype=float)
        if not self._active or self._alpha <= 0.0:
            return d
        ref = reference if reference is not None else self._ref
        if ref is None:
            return d * (1.0 - self._alpha)
        ref = np.asarray(ref, dtype=float)
        if ref.shape != d.shape:
            ref = np.broadcast_to(ref.reshape(-1), d.shape[-1] if d.ndim > 1
                                  else d.shape)
        return (1.0 - self._alpha) * d + self._alpha * ref

    def memory_gain(self) -> float:
        """Multiplier on memory reliance; 1.0 when healthy."""
        if not self._active or self.cfg.initial_damping <= 0:
            return 1.0
        frac = self._alpha / self.cfg.initial_damping
        return float(1.0 + (self.cfg.memory_boost - 1.0) * frac)

    def weight_scale(self) -> float:
        """One-shot readout shrink to apply at TRIP."""
        return float(1.0 - self.cfg.weight_decay)

    def steps_remaining(self) -> int:
        if not self._active:
            return 0
        a = max(self._alpha, 1e-12)
        return max(0, int(math.ceil(
            math.log(self.cfg.min_damping / a) /
            math.log(self.cfg.decay))))

    def reset(self) -> None:
        self._alpha = 0.0
        self._active = False
        self._ref = None
        self._cooldown = 0

    # -- serialization ---------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "cfg": dict(self.cfg.__dict__),
            "step": self._step,
            "alpha": self._alpha,
            "ref": None if self._ref is None else self._ref.copy(),
            "active": self._active,
            "trip_count": self.trip_count,
            "health": self.health.state_dict(),
        }

    def load_state_dict(self, d: dict) -> None:
        self.cfg = DampenConfig(**d["cfg"])
        self._step = int(d["step"])
        self._alpha = float(d["alpha"])
        self._ref = None if d["ref"] is None else np.asarray(d["ref"], dtype=float)
        self._active = bool(d["active"])
        self.trip_count = int(d["trip_count"])
        self.health.load_state_dict(d["health"])


def make_dampened_breaker(ramp_steps: int = 10,
                          initial_damping: float = 0.4,
                          decay: float = 0.6,
                          weight_decay: float = 0.0,
                          memory_boost: float = 1.0,
                          health: Optional[CircuitBreaker] = None
                          ) -> DampenedCircuitBreaker:
    """Factory for the standard dampened breaker."""
    return DampenedCircuitBreaker(DampenConfig(
        ramp_steps=ramp_steps, initial_damping=initial_damping, decay=decay,
        weight_decay=weight_decay, memory_boost=memory_boost),
        health=health)
