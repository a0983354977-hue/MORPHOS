"""Phase IV acceptance tests (T4.1 - T4.4).

Covers the benchmark plumbing end to end:
  * traditional baselines produce sane one-step metrics
  * ARIMA one-step filtering works and is finite
  * Isolation Forest separates injected anomalies
  * precision / recall helpers are correct
  * the multi-timescale stream and its anomaly labels are aligned
  * MORPHOS ablation variants wire memory / multi-scale correctly
  * the runner's acceptance logic flips on the right comparisons

Run:  python tests/test_phase4.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.benchmarks.baselines import (
    ARBaseline, ARIMABaseline, IsolationForestBaseline,
    HAS_SKLEARN, HAS_STATSMODELS,
    best_f1_over_thresholds, calibrate_threshold, precision_recall,
    residual_scores, threshold_labels,
)
from experiments.benchmarks.runner import (
    MEM_CHANNELS, Z_LAGS, MORPHOSBenchmark, BenchmarkConfig,
    build_acceptance, build_stream, make_anomaly_schedule, mixed_series,
    normalise,
)


def _ar1(n: int, phi: float = 0.85, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, 0.3)
    return x[:, None]


def test_precision_recall_helpers():
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    p = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    m = precision_recall(y, p)
    assert m["tp"] == 3 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 3
    assert abs(m["precision"] - 0.75) < 1e-12
    assert abs(m["recall"] - 0.75) < 1e-12
    print(f"  precision={m['precision']:.3f} recall={m['recall']:.3f}")
    return True


def test_threshold_budget():
    rng = np.random.default_rng(0)
    clean = rng.gamma(2.0, 1.0, 5000)
    thr = calibrate_threshold(clean, 0.9)
    labels = threshold_labels(clean, thr)
    fp_rate = labels.mean()
    assert 0.08 <= fp_rate <= 0.12, f"FP budget off: {fp_rate}"
    print(f"  threshold={thr:.4f} clean FP rate={fp_rate:.3f}")
    return True


def test_best_f1_bound():
    y = np.array([0] * 90 + [1] * 10)
    scores = np.concatenate([np.zeros(90), np.ones(10) * 5.0])
    best = best_f1_over_thresholds(y, scores)
    assert best["f1"] > 0.99, f"separable scores should hit F1=1: {best}"
    print(f"  best F1={best['f1']:.3f}")
    return True


def test_ar_one_step_beats_naive():
    series = _ar1(1500)
    train, test = series[:1000], series[1000:]
    ar = ARBaseline(order=5, name="AR5").fit(train)
    res = ar.predict(test)
    naive = np.mean(np.abs(np.diff(test, axis=0)))
    ar_mae = float(np.mean(res.mae))
    assert np.all(np.isfinite(res.mae))
    assert ar_mae < naive, f"AR({ar_mae:.3f}) should beat naive diff({naive:.3f})"
    print(f"  AR5 MAE={ar_mae:.4f} (naive-diff {naive:.4f})")
    return True


def test_arima_one_step_filtering():
    if not HAS_STATSMODELS:
        print("  [SKIP] statsmodels unavailable")
        return True
    series = _ar1(900, phi=0.7, seed=3) + 0.5
    train, test = series[:600], series[600:]
    model = ARIMABaseline(order=(2, 0, 0), name="ARIMA").fit(train)
    res = model.predict(test)
    assert np.all(np.isfinite(res.predictions)), "ARIMA produced non-finite output"
    assert np.all(np.isfinite(res.mae))
    # one-step filtering must beat a constant (train-mean) predictor
    mean_pred = np.abs(test - train.mean(axis=0)).mean()
    assert float(np.mean(res.mae)) < mean_pred, \
        f"ARIMA {np.mean(res.mae):.4f} vs mean-predictor {mean_pred:.4f}"
    print(f"  ARIMA MAE={float(np.mean(res.mae)):.4f} "
          f"(mean predictor {mean_pred:.4f})")
    return True


def test_isolation_forest_separation():
    if not HAS_SKLEARN:
        print("  [SKIP] sklearn unavailable")
        return True
    rng = np.random.default_rng(7)
    clean = rng.normal(0, 0.2, (600, 4))
    anomalous = rng.normal(6.0, 0.2, (40, 4))
    model = IsolationForestBaseline(contamination=0.06, name="IForest")
    model.fit(clean)
    res = model.predict(np.vstack([clean, anomalous]))
    labels = np.concatenate([np.zeros(len(clean), int),
                             np.ones(len(anomalous), int)])
    scores = np.nan_to_num(res.anomaly_scores)
    thr = calibrate_threshold(scores[:len(clean)], 0.9)
    m = precision_recall(labels, threshold_labels(scores, thr))
    assert m["recall"] > 0.9, f"IForest recall too low: {m}"
    print(f"  IForest P={m['precision']:.3f} R={m['recall']:.3f}")
    return True


def test_stream_alignment():
    config = BenchmarkConfig(train_steps=80, test_steps=80, calib_steps=60,
                             anomaly_steps=300)
    schedule = make_anomaly_schedule(300, seed=5, n_runs=4, min_len=4,
                                     max_len=6, min_gap=40)
    assert len(schedule) >= 2, "anomaly schedule empty"
    stream = build_stream(config, 300, seed=11, anomalies=schedule)
    assert stream["lorenz"].shape == (300, 3)
    assert stream["system"].shape == (300, 4)
    expected = sum(min(300, s + l) - s for s, l, _k in schedule)
    assert int(stream["labels"].sum()) == expected, "labels not aligned"
    for s, l, _k in schedule:
        assert stream["labels"][s:s + l].all()
    series, labels = mixed_series(config, 300, seed=11,
                                  anomalies=schedule)
    assert series.shape == (300, 7)
    assert np.array_equal(labels, stream["labels"])
    print(f"  {len(schedule)} injections, {int(labels.sum())} labelled steps")
    return True


def test_normalise_range():
    rng = np.random.default_rng(0)
    raw_tr = rng.normal(10, 3, (200, 7))
    raw_te = rng.normal(10, 3, (100, 7))
    z_tr, z_te = normalise(raw_tr, raw_tr, raw_te)
    assert np.max(np.abs(z_tr)) <= 1.0 + 1e-12
    assert np.max(np.abs(z_te)) <= 1.0 + 1e-12
    assert abs(float(z_tr.mean())) < 0.05
    print(f"  z range [{z_tr.min():.2f}, {z_tr.max():.2f}]")
    return True


def test_ablation_wiring():
    config = BenchmarkConfig(train_steps=40, test_steps=40,
                             n_neurons_per_scale=(40, 30, 20))
    full = MORPHOSBenchmark(config, ablation={})
    single = MORPHOSBenchmark(config, ablation={"no_multiscale": True})
    no_mem = MORPHOSBenchmark(config, ablation={"no_memory": True})

    assert full.use_multiscale and not single.use_multiscale
    # readout width follows the number of scales it actually consumes
    width = Z_LAGS * 7 + MEM_CHANNELS
    assert full.predictor.n_in == 2 * 3 * config.feature_dim + width
    assert single.predictor.n_in == 2 * config.feature_dim + width
    assert no_mem.predictor.n_in == full.predictor.n_in
    assert full.predictor.n_out == 7
    # memory channel is live for the full model, zeroed when ablated
    assert np.allclose(no_mem._memory_vector(), 0.0)
    full.temporal_buffer.push_step(np.ones(config.feature_dim), 1.0, 0.5, 20.0)
    assert np.any(full._memory_vector() != 0.0)
    print("  ablation wiring consistent, memory channel live/zeroed")
    return True


def test_small_pipeline_run():
    config = BenchmarkConfig(train_steps=60, test_steps=60,
                             n_neurons_per_scale=(40, 30, 20))
    model = MORPHOSBenchmark(config, ablation={})
    train = model.run_stream(60, domain="mixed", seed=1, learn=True)
    test = model.run_stream(60, domain="mixed", seed=2, learn=False)
    assert test.residuals.shape[1] == 7
    assert np.all(np.isfinite(test.scores)), "non-finite anomaly scores"
    assert len(test.target_steps) == len(test.scores) == len(test.labels)
    assert test.metrics("lorenz")["mae"] > 0.0
    print(f"  {len(test.scores)} predictions, "
          f"mixed MAE={test.metrics('mixed')['mae']:.4f}")
    return True


def test_acceptance_logic():
    def ablation(full_mae, single_mae):
        return {
            "full": {"mixed": {"mae": full_mae}, "lorenz": {"mae": 0.1}},
            "no_multiscale": {"mixed": {"mae": single_mae},
                              "lorenz": {"mae": 0.2}},
        }

    good_baselines = {"AR5": {"mae": [0.4] * 3 + [0.4] * 4},
                      "ARIMA_5_0_0": {"mae": [0.5] * 3 + [0.5] * 4}}
    # build_acceptance() expects anomaly[task][detector] structure
    good_anomaly = {
        "point_spike": {
            "MORPHOS_dual": {"f1": 0.8, "precision": 0.9},
            "IsolationForest": {"f1": 0.5, "precision": 0.4},
            "ARIMA_5_0_0": {"f1": 0.4},
        }
    }
    config = BenchmarkConfig()

    verdict = build_acceptance(config, ablation(0.1, 0.2), good_baselines,
                               good_anomaly)
    assert verdict["multiscale_beats_single_scale"]["pass"]
    assert verdict["forecast_beats_classical"]["pass"]
    assert verdict["anomaly_beats_classical"]["pass"]
    assert verdict["multiscale_beats_single_scale"]["improvement_pct"] > 0

    bad_anomaly = {
        "point_spike": {
            "MORPHOS_dual": {"f1": 0.2, "precision": 0.1},
            "IsolationForest": {"f1": 0.5, "precision": 0.4},
            "ARIMA_5_0_0": {"f1": 0.4},
        }
    }
    bad = build_acceptance(config, ablation(0.3, 0.2), good_baselines,
                           bad_anomaly)
    assert not bad["multiscale_beats_single_scale"]["pass"]
    assert not bad["anomaly_beats_classical"]["pass"]
    print("  acceptance logic flips correctly")
    return True


def run_all_tests():
    print("=" * 60)
    print("PHASE IV ACCEPTANCE TESTS")
    print("Cross-Domain Intelligence & Benchmarking")
    print("=" * 60)

    tests = [
        ("Precision/Recall Helpers", test_precision_recall_helpers),
        ("Threshold FP Budget", test_threshold_budget),
        ("Best-F1 Bound", test_best_f1_bound),
        ("AR One-Step", test_ar_one_step_beats_naive),
        ("ARIMA One-Step", test_arima_one_step_filtering),
        ("Isolation Forest", test_isolation_forest_separation),
        ("Stream Alignment", test_stream_alignment),
        ("Normalisation Range", test_normalise_range),
        ("Ablation Wiring", test_ablation_wiring),
        ("Small Pipeline Run", test_small_pipeline_run),
        ("Acceptance Logic", test_acceptance_logic),
    ]

    passed = failed = 0
    for name, fn in tests:
        print(f"\n[TEST] {name}")
        try:
            fn()
            passed += 1
            print("  [OK] PASSED")
        except Exception as exc:  # noqa: BLE001 - report every failure
            failed += 1
            print(f"  [FAIL] {exc}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
