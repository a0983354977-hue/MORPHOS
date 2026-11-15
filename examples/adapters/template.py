"""Template: a minimal MORPHOS adapter (copy this file to start).

Everything a new data source needs is in this file. It is intentionally short:
the platform handles sanitising, alignment, encoding, dynamics and anomaly
detection, so an adapter only has to *produce samples*.

Run the conformance check against it:

    python -m adapters.sdk validate examples/adapters/template.py:MySensor

Then point the control loop at it (see docs/SDK.md for the wiring recipe).

------------------------------------------------------------------------
The contract
------------------------------------------------------------------------
  dim           int   Number of values per sample. Constant for the lifetime.
  advance_to(t)       Make all samples with timestamp <= t available.
  read_pending()      Return (timestamp, vector) pairs since last call.

Rules the core relies on:
  * advance_to() does not block (target: well under one control step, 5 ms).
  * timestamps are non-decreasing across calls.
  * each vector is exactly `dim` floats; NaN/Inf is allowed (sanitised later).
------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from adapters.sdk import MorphosSource


class MySensor(MorphosSource):
    # ---- 1. describe yourself -------------------------------------
    name = "my-sensor"
    dim = 3                 # <-- change to your sensor's channel count
    sample_period = 0.001   # nominal seconds between hardware samples

    def __init__(self, dt: float = 0.001):
        # `super().__init__()` validates `dim` and sets up self.t; call it
        # *after* setting self.dim but *before* touching base state.
        self.dt = dt
        self._pending: list[tuple[float, np.ndarray]] = []
        super().__init__()

        # ---- 2. open your hardware / file / socket here -----------
        # e.g. self._handle = serial.Serial("COM3", 115200)
        #      self._fp = open("capture.bin", "rb")
        self._ticks = 0

    # ---- 3. produce samples --------------------------------------
    def advance_to(self, t_target: float) -> None:
        """Advance until self.t >= t_target, queueing samples as you go.

        Two patterns work well:

        *Self-driven (this example)* - integrate/generate internally until the
        requested horizon. Good for simulators and synthetic sources.

        *Device-driven* - drain whatever the device already delivered, then set
        self.t = t_target. Never sleep waiting for hardware; the aligner fills
        gaps and lowers confidence instead.
        """
        while self.t < t_target - 1e-12:
            self.t += self.dt
            self._ticks += 1

            # --- replace this with a real read ---------------------
            phase = self._ticks * self.dt
            vec = np.array([
                np.sin(2 * np.pi * 0.8 * phase),
                np.cos(2 * np.pi * 0.5 * phase),
                np.sin(2 * np.pi * 1.9 * phase) * 0.35,
            ])
            # If your device returns nothing this instant, simply don't
            # append: a gap is legal and is handled downstream.
            self._pending.append((self.t, vec))

    def read_pending(self) -> list[tuple[float, np.ndarray]]:
        out = self._pending
        self._pending = []
        return out

    # ---- 4. optional: diagnostics + cleanup ----------------------
    @property
    def stats(self) -> dict:
        d = super().stats
        d.update({"ticks": self._ticks, "dt": self.dt})
        return d

    def close(self) -> None:
        # Close hardware handles / files here.
        self._pending.clear()
