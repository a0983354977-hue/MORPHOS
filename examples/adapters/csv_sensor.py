"""Worked example: a file-backed CSV sensor adapter.

More realistic than the template: it reads a recorded capture from disk,
handles a missing/garbled header, tolerates blank rows, and re-times samples
onto the control grid. This is the shape most real integrations take (log
shippers, recorded telemetry, exported CSVs from another tool).

Validate it:

    python -m adapters.sdk validate examples/adapters/csv_sensor.py:CSVSensor

Run it standalone to see the output the core would receive:

    python examples/adapters/csv_sensor.py --demo
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from adapters.sdk import AdapterContractError, MorphosSource


def _write_demo_csv(path: str, rows: int = 500) -> None:
    """Create a small capture so the example runs with no external data."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "temp", "pressure", "vibration"])
        rng = np.random.default_rng(7)
        t = 0.0
        for i in range(rows):
            t += 0.001
            # occasional dropped row simulates a real capture gap
            if i % 97 == 0:
                continue
            w.writerow([
                f"{t:.4f}",
                f"{20.0 + 3.0 * np.sin(t * 2.1) + rng.normal(0, 0.02):.5f}",
                f"{101.3 + 0.4 * np.sin(t * 0.7) + rng.normal(0, 0.01):.5f}",
                f"{0.5 * np.sin(t * 11.0) + rng.normal(0, 0.05):.5f}",
            ])


class CSVSensor(MorphosSource):
    """Replay a CSV capture of numeric channels, re-timed onto a fixed grid."""

    name = "csv-sensor"
    sample_period = 0.001

    def __init__(self, path: str | None = None, value_cols: int = 3,
                 loop: bool = True):
        if path is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "demo_capture.csv")
            if not os.path.exists(path):
                _write_demo_csv(path)
        self.path = path
        self.loop = loop
        self._pending: list[tuple[float, np.ndarray]] = []
        self.rows: list[tuple[float, np.ndarray]] = []
        self.skipped = 0
        self._load(value_cols)
        self.dim = value_cols
        super().__init__()
        self._idx = 0
        self._span = self.rows[-1][0] - self.rows[0][0]
        self._origin = self.rows[0][0]
        self._produced = 0

    def _load(self, value_cols: int) -> None:
        if not os.path.exists(self.path):
            raise AdapterContractError(f"capture not found: {self.path}")
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if not row or all(not c.strip() for c in row):
                    self.skipped += 1
                    continue
                try:
                    vals = [float(c) for c in row]
                except ValueError:
                    # header or any non-numeric row
                    self.skipped += 1
                    continue
                if len(vals) < value_cols + 1:
                    self.skipped += 1
                    continue
                self.rows.append((vals[0], np.asarray(vals[1:1 + value_cols])))
        if not self.rows:
            raise AdapterContractError(f"no numeric rows in {self.path}")
        self.rows.sort(key=lambda r: r[0])
        if self.rows[-1][0] <= self.rows[0][0]:
            raise AdapterContractError("capture has no time span")

    # -- interface -----------------------------------------------------
    def advance_to(self, t_target: float) -> None:
        """Map elapsed control time onto the recording's own time base."""
        dt = self.sample_period
        while self.t < t_target - 1e-12:
            self.t += dt
            elapsed = self.t
            src_t = self._origin + (elapsed % self._span if self.loop
                                    else elapsed)
            vec = self._sample_at(src_t)
            if vec is not None:
                self._pending.append((self.t, vec))
                self._produced += 1
            if not self.loop and elapsed > self._span:
                break

    def _sample_at(self, src_t: float) -> np.ndarray | None:
        """Nearest-sample lookup; returns None when the capture ended."""
        while self._idx + 1 < len(self.rows) and self.rows[self._idx + 1][0] <= src_t:
            self._idx += 1
        if self._idx >= len(self.rows):
            if not self.loop:
                return None
            self._idx = 0
        return self.rows[self._idx][1].copy()

    def read_pending(self) -> list[tuple[float, np.ndarray]]:
        out, self._pending = self._pending, []
        return out

    @property
    def stats(self) -> dict:
        d = super().stats
        d.update({"path": os.path.basename(self.path),
                  "rows": len(self.rows), "skipped": self.skipped,
                  "produced": self._produced})
        return d


def main() -> int:
    ap = argparse.ArgumentParser(description="CSV adapter demo")
    ap.add_argument("--demo", action="store_true",
                    help="print the first samples the core would receive")
    ap.add_argument("--csv", default=None, help="capture to read")
    args = ap.parse_args()

    src = CSVSensor(args.csv)
    print(f"loaded {len(src.rows)} rows from {src.path} "
          f"(skipped {src.skipped} non-numeric/blank)")
    if args.demo:
        src.advance_to(0.010)
        for ts, vec in src.read_pending()[:10]:
            print(f"  t={ts:.4f}  " +
                  "  ".join(f"{v:8.4f}" for v in vec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
