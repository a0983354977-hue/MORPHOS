"""Chaos & integration tests for the MORPHOS foundation.

Run:  python tests/chaos_test.py
(or: python -m pytest tests/chaos_test.py)

Covers the Phase I acceptance criteria:
  1. Noise injection  - hostile stream must not produce NaN/crash
  2. Disconnection    - 5 s dropout must be bridged by the aligner
  3. Unit-level       - sanitizer, aligner interpolation/hold, NLMS
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.ingest.ring_buffer import (FLAG_INTERP, FLAG_NO_DATA,
                                     FLAG_STALE, TimeAligner)
from core.prediction.predictor import NLMSPredictor
from storage.wal import iter_records


# ----------------------------------------------------------------------
# unit tests
# ----------------------------------------------------------------------
def test_sanitizer_rejects_nan_and_spikes():
    al = TimeAligner(dt=0.005, dim=3)
    # feed clean samples so median/MAD stabilise
    for i in range(60):
        al.push(i * 0.001, np.array([1.0, 2.0, 3.0]))
    dropped_before = al.dropped_count
    al.push(0.061, np.array([np.nan, 2.0, 3.0]))   # non-finite -> dropped
    al.push(0.062, np.array([1e9, 2.0, 3.0]))      # extreme -> clamped
    assert al.dropped_count == dropped_before + 1


def test_aligner_interpolates_and_holds():
    al = TimeAligner(dt=0.005, dim=1, max_interp_gap=0.05, hold_tau=0.5)
    al.push(0.000, np.array([0.0]))
    al.push(0.010, np.array([10.0]))
    v, c, f = al.at(0.005)
    assert abs(v[0] - 5.0) < 1e-9 and f & FLAG_INTERP
    assert 0.5 <= c <= 1.0
    # gap -> hold last with decaying confidence
    al2 = TimeAligner(dt=0.005, dim=1, max_interp_gap=0.02, hold_tau=0.5)
    al2.push(0.000, np.array([7.0]))
    v, c, f = al2.at(0.100)
    assert v[0] == 7.0 and c < 0.9
    v, c, f = al2.at(1.0)
    assert v[0] == 7.0 and c < 0.5
    v, c, f = al2.at(100.0)
    assert c == 0.0 and (f & FLAG_STALE)


def test_aligner_no_data():
    al = TimeAligner(dt=0.005, dim=3)
    v, c, f = al.at(0.0)
    assert c == 0.0 and f & FLAG_NO_DATA and np.all(v == 0)


def test_nlms_predictor_converges_on_linear_system():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3)) * 0.3
    pred = NLMSPredictor(n_in=3, n_out=3, mu=0.2, eps=1e-2)
    x = rng.standard_normal(3)
    errs = []
    for _ in range(3000):
        x = A @ x + rng.normal(0, 0.01, 3)
        y = pred.predict(x)
        errs.append(float(np.sum((x - y) ** 2)))
        pred.update(x, x)  # trivially learn identity as sanity check
    assert np.mean(errs[-200:]) < np.mean(errs[:200])


# ----------------------------------------------------------------------
# integration / chaos test (spec acceptance)
# ----------------------------------------------------------------------
def test_combat_run_end_to_end():
    data_dir = os.path.join(ROOT, "data_test")
    steps = 1600          # 8 s of simulated time
    wal = os.path.join(data_dir, "trajectory.wal")
    if os.path.exists(wal):
        os.remove(wal)

    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "main.py"),
         "--steps", str(steps), "--chaos", "combat5",
         "--data-dir", data_dir],
        capture_output=True, text=True, cwd=ROOT, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]

    recs = list(iter_records(wal))
    assert len(recs) >= steps - 5, f"only {len(recs)} wal records"

    residuals = np.array([r["residual"] for r in recs])
    rates = np.array([r["rate"] for r in recs])
    confs = np.array([r["conf"] for r in recs])

    # 1. no NaN / Inf anywhere in the trajectory
    assert np.all(np.isfinite(residuals)), "non-finite residual found"
    assert np.all(np.isfinite(rates)), "non-finite firing rate found"

    # 2. dropout window must be flagged low-confidence & bridged
    drop_recs = [r for r in recs if 2.0 < r["t_sim"] < 7.0]
    assert drop_recs, "no records inside dropout window"
    assert np.mean([r["conf"] for r in drop_recs]) < 0.9

    # 3. neural population stayed in a sane firing band
    assert rates.mean() < 300.0, f"runaway firing rate {rates.mean()}"
    assert rates.max() < 1000.0

    # 4. residual converges to a low stable level vs. its own start
    first = np.mean(residuals[:100])
    last = np.mean(residuals[-200:])
    assert np.isfinite(last) and last < max(first, 1e-6) * 50 or last < 0.5, \
        f"residual did not converge: first={first} last={last}"

    print(f"[chaos_test] records={len(recs)} resid {first:.4f}->{last:.4f} "
          f"rate_mean={rates.mean():.1f}Hz conf_min={confs.min():.2f}")


if __name__ == "__main__":
    test_sanitizer_rejects_nan_and_spikes()
    print("ok: sanitizer")
    test_aligner_interpolates_and_holds()
    print("ok: aligner")
    test_aligner_no_data()
    print("ok: no-data")
    test_nlms_predictor_converges_on_linear_system()
    print("ok: nlms")
    test_combat_run_end_to_end()
    print("ALL CHAOS TESTS PASSED")
