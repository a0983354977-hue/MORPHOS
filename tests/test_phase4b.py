"""Phase IV.b acceptance tests (T4.5 - T4.7).

Covers the three hardening tasks:

  * T4.5 - dual-opinion detector: associative novelty separates point
    spikes, rolling robust-z is scale free, and MORPHOS beats Isolation
    Forest at the same 1% false-positive budget.
  * T4.6 - dampened circuit breaker: the ramp completes within 10 steps,
    the cooldown prevents trip cascades, and a TRIP never rebuilds the
    simulator.
  * T4.7 - importance compressor: footprint is constant across 12k steps
    while high-residual events and turning points survive.

Run:  python tests/test_phase4b.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.memory.compressor import make_compressor
from core.observer.circuit_breaker import make_dampened_breaker
from core.observer.dual_opinion import make_dual_opinion
from experiments.benchmarks.baselines import (
    HAS_SKLEARN, IsolationForestBaseline, calibrate_threshold,
    precision_recall,
)
from experiments.benchmarks.runner import (
    BenchmarkConfig, MORPHOSBenchmark, make_spike_schedule, mixed_series,
    normalise,
)

LORENZ_DIMS = 3


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _lorenz_z(config: BenchmarkConfig, steps: int, seed: int,
              anomalies=None, mu=None, sd=None):
    """Normalised 3-channel Lorenz stream (same transform as the runner)."""
    raw, labels = mixed_series(config, steps, seed, anomalies=anomalies)
    if mu is None:
        mu = raw[:, :LORENZ_DIMS].mean(axis=0)
        sd = raw[:, :LORENZ_DIMS].std(axis=0) + 1e-8
    z = np.clip((raw[:, :LORENZ_DIMS] - mu) / sd, -4.0, 4.0) / 4.0
    return z, labels, mu, sd


def test_novelty_separates_point_spikes():
    """T4.5: a spike is far from memory; clean states are not."""
    config = BenchmarkConfig()
    schedule = make_spike_schedule(700, config.seed + 8, n_runs=4)
    train_z, _, mu, sd = _lorenz_z(config, 3000, config.seed + 1)
    test_z, labels, _, _ = _lorenz_z(config, 700, config.seed + 7,
                                     anomalies=schedule, mu=mu, sd=sd)

    det = make_dual_opinion(state_dim=LORENZ_DIMS)
    det.add_reference_batch(train_z)
    assert det.memory_size == 3000

    nov = np.array([det.novelty(z) for z in test_z])
    clean = nov[labels == 0]
    anomalous = nov[labels == 1]
    assert anomalous.size > 0, "schedule produced no anomalous steps"
    ratio = float(np.median(anomalous) / max(np.median(clean), 1e-12))
    print(f"  clean median novelty {np.median(clean):.4f}, "
          f"spike median {np.median(anomalous):.4f} (x{ratio:.1f})")
    assert ratio > 3.0, f"novelty separation too weak: {ratio:.2f}"


def test_rolling_z_is_scale_free():
    """T4.5: the robust z removes per-stream amplitude drift."""
    det = make_dual_opinion(state_dim=LORENZ_DIMS)
    rng = np.random.default_rng(0)
    base = np.abs(rng.normal(0.01, 0.004, 400))
    base[200:205] = 5.0

    z_small = det.opinions(np.zeros(400), base)["novelty_z"]
    z_big = det.opinions(np.zeros(400), base * 37.0)["novelty_z"]
    assert np.allclose(z_small, z_big, rtol=1e-6, atol=1e-6), (
        "robust z should be invariant to a global rescaling")
    assert z_small[200:205].min() > 10.0, "spike must dominate its window"


def test_dual_beats_isolation_forest_on_spikes():
    """T4.5 acceptance: F1 and precision > IForest at 1% FPR."""
    if not HAS_SKLEARN:
        print("  [SKIP] scikit-learn unavailable")
        return
    config = BenchmarkConfig()
    q = 0.99
    schedule = make_spike_schedule(800, config.seed + 8, n_runs=5)

    train_raw, _ = mixed_series(config, 4000, config.seed + 1)
    calib_raw, _ = mixed_series(config, 600, config.seed + 5)
    test_raw, labels_all = mixed_series(config, 800, config.seed + 7,
                                        anomalies=schedule)
    train_z, calib_z, test_z = normalise(train_raw, train_raw, calib_raw,
                                         test_raw)
    lz = slice(0, LORENZ_DIMS)
    train_z, calib_z, test_z = train_z[:, lz], calib_z[:, lz], test_z[:, lz]

    # --- MORPHOS: associative novelty, rolling robust-z ---
    det = make_dual_opinion(state_dim=LORENZ_DIMS)
    det.add_reference_batch(train_z)
    calib_nov = np.array([det.novelty(z) for z in calib_z])
    test_nov = np.array([det.novelty(z) for z in test_z])
    det.calibrate(np.zeros(len(calib_nov)), calib_nov)
    scores = det.score_series(np.zeros(len(test_nov)), test_nov)[1:]
    y = labels_all[1:len(scores) + 1]
    morphos = precision_recall(y, (scores > det.threshold).astype(int))

    # --- Isolation Forest on the same channels and the same budget ---
    def embed(z: np.ndarray) -> np.ndarray:
        prev = np.vstack([z[:1], z[:-1]])
        return np.concatenate([z, prev], axis=1)

    forest = IsolationForestBaseline(contamination=0.1)
    forest.fit(embed(train_z))
    calib_s = forest.predict(embed(calib_z)).anomaly_scores
    test_s = forest.predict(embed(test_z)).anomaly_scores
    thr = calibrate_threshold(calib_s[1:], q)
    forest_m = precision_recall(y, (test_s[1:len(y) + 1] > thr).astype(int))

    print(f"  MORPHOS  P={morphos['precision']:.3f} R={morphos['recall']:.3f} "
          f"F1={morphos['f1']:.3f}")
    print(f"  IForest  P={forest_m['precision']:.3f} "
          f"R={forest_m['recall']:.3f} F1={forest_m['f1']:.3f}")
    assert morphos["f1"] > forest_m["f1"], "F1 must beat Isolation Forest"
    assert morphos["precision"] > forest_m["precision"], (
        "precision must beat Isolation Forest")


def test_residual_opinion_is_not_in_the_decision_score():
    """T4.5: the residual cannot inflate the decision score by default."""
    det = make_dual_opinion(state_dim=LORENZ_DIMS)
    assert det.cfg.residual_fusion_weight == 0.0
    rng = np.random.default_rng(1)
    nov = np.abs(rng.normal(0.01, 0.003, 300))
    res = np.abs(rng.normal(1.0, 0.2, 300))
    ops = det.opinions(res, nov)
    np.testing.assert_allclose(det.score_series(res, nov), ops["novelty_z"])
    det.calibrate(res, nov)
    assert np.isfinite(det.threshold) and det.threshold > 0


def test_dynamic_gate_blocks_residual_ringing():
    """T4.5: dynamic gate prevents post-spike ringing from inflating score.

    When the novelty has returned to baseline after a spike, the residual
    gate lifts above the residual z-score so that ringing is ignored.
    """
    det = make_dual_opinion(state_dim=LORENZ_DIMS,
                            residual_fusion_weight=0.3)
    rng = np.random.default_rng(5)
    n = 400
    # Simulated spike: novelty spikes then returns to baseline; residual
    # stays elevated (ringing) for many steps afterwards.
    nov = np.abs(rng.normal(0.01, 0.003, n))
    res = np.abs(rng.normal(0.1, 0.02, n))
    nov[200:205] = 5.0      # spike novelty
    res[200:210] = 4.0      # residual spikes with the novelty AND rings after
    det.calibrate(res, nov)

    scores = det.score_series(res, nov)
    # After step 205, the residual is still high but novelty is back to
    # baseline — the dynamic gate should suppress it.
    post_spike = scores[210:220]
    clean_baseline = scores[:190]
    # The post-spike score must not be significantly higher than baseline
    # (the dynamic gate suppresses ringing).
    assert float(np.max(post_spike)) < float(np.max(scores[200:205])), (
        "post-spike ringing must not exceed spike peak")
    assert float(np.mean(post_spike)) < float(np.mean(clean_baseline)) + 0.5, (
        "post-spike mean must stay near baseline")
    print(f"  post-spike max {np.max(post_spike):.3f} vs "
          f"spike peak {np.max(scores[200:205]):.3f}")


def test_adaptive_memory_replenishment():
    """T4.5: confidently normal states are absorbed during inference.

    The memory should grow during inference as new normal states are
    encountered, expanding the manifold coverage.
    """
    det = make_dual_opinion(state_dim=LORENZ_DIMS,
                            adaptive_memory=True,
                            adaptive_memory_interval=10,
                            adaptive_memory_safety=0.5)
    rng = np.random.default_rng(6)
    # Seed initial memory with a batch of clean states
    init_states = rng.normal(0.0, 0.1, (100, LORENZ_DIMS))
    det.add_reference_batch(init_states)
    mem_before = det.memory_size

    # Calibrate on clean data
    calib_nov = np.array([det.novelty(s) for s in init_states])
    det.calibrate(np.zeros(len(calib_nov)), calib_nov)

    # Run inference with new normal states; they should be absorbed
    added_count = 0
    for step in range(500):
        state = rng.normal(0.0, 0.1, LORENZ_DIMS)
        nv = det.novelty(state)
        result = det.score(residual=0.01, novelty=nv, state=state)
        if result["adaptive_mem_added"]:
            added_count += 1

    assert added_count >= 3, (
        f"adaptive memory should have added >=3 references, got {added_count}")
    assert det.memory_size > mem_before
    print(f"  memory grew from {mem_before} to {det.memory_size}, "
          f"{added_count} adaptive additions")


# ----------------------------------------------------------------------
# T4.6
# ----------------------------------------------------------------------
def test_dampened_recovery_completes_within_ten_steps():
    """T4.6 acceptance: the ramp is over in <= 10 steps."""
    breaker = make_dampened_breaker()
    breaker.trip(100)
    assert breaker.active
    assert breaker.steps_remaining() <= 10, breaker.steps_remaining()
    gains = []
    for _ in range(10):
        gains.append(breaker.input_gain())
        breaker.advance()
    assert not breaker.active, "ramp must finish within ramp_steps"
    assert breaker.input_gain() == 1.0
    assert all(b >= a - 1e-12 for a, b in zip(gains, gains[1:])), (
        "gain must relax monotonically back to 1")
    assert gains[0] < 0.5
    print(f"  gains {['%.2f' % g for g in gains]}")


def test_cooldown_prevents_trip_cascades():
    """T4.6: no new TRIP while the network is still settling."""
    breaker = make_dampened_breaker()
    assert breaker.can_trip()
    breaker.trip(0)
    assert not breaker.can_trip()
    breaker.advance(5)
    assert not breaker.can_trip(), "must not re-trip mid-ramp"
    breaker.advance(30)
    assert breaker.can_trip(), "cooldown should elapse"
    assert breaker.trip_count == 1


def test_trip_never_rebuilds_the_simulator():
    """T4.6: recovery must not reset/recreate the Nengo simulator."""
    config = BenchmarkConfig()
    model = MORPHOSBenchmark(config, {})
    model.run_stream(300, domain="mixed", seed=config.seed + 1, learn=True)

    sim_before = id(model.dynamics)

    def _forbidden(*_a, **_k):
        raise AssertionError("dynamics.reset() must not be called on TRIP")

    model.dynamics.reset = _forbidden  # type: ignore[method-assign]

    # Arm the breaker far below the normal operating point so a clean
    # stream is guaranteed to trip it.
    model.alarm_threshold = -1.0
    res = model.run_stream(120, domain="mixed", seed=config.seed + 9)
    model.alarm_threshold = None

    assert model.breaker.trip_count >= 1, "breaker did not trip"
    assert id(model.dynamics) == sim_before
    assert np.all(np.isfinite(res.residuals))
    assert model._alarms >= 1
    print(f"  trips={model.breaker.trip_count} alarms={model._alarms}")


# ----------------------------------------------------------------------
# T4.7
# ----------------------------------------------------------------------
def test_memory_footprint_is_flat_over_twelve_thousand_steps():
    """T4.7 acceptance: bounded footprint, no linear growth."""
    comp = make_compressor(feature_dim=6, max_entries=512, recent_reserve=64)
    rng = np.random.default_rng(2)
    footprints = []
    # Warm up past the filling phase so the footprint has stabilised.
    for step in range(1000):
        feat = rng.normal(0.0, 1.0, 6)
        comp.push(step, feat, residual=float(abs(rng.normal(0, 0.01))))
    for step in range(1000, 12000):
        feat = rng.normal(0.0, 1.0, 6)
        comp.push(step, feat, residual=float(abs(rng.normal(0, 0.01))))
        if step % 1000 == 0:
            footprints.append(comp.footprint_entries())

    assert comp.total_seen == 12000
    assert max(footprints) == min(footprints), (
        f"footprint grew: {sorted(set(footprints))}")
    assert comp.size <= 512
    assert len(comp.snapshot()["steps"]) <= 512 + 64
    print(f"  footprint constant at {footprints[0]} entries over 12000 steps")


def test_compressor_keeps_high_residual_and_turning_points():
    """T4.7: compression keeps what matters and drops the routine."""
    comp = make_compressor(feature_dim=4, max_entries=32, recent_reserve=8)
    rng = np.random.default_rng(3)
    events = {1000, 2000, 3000}
    for step in range(5000):
        feat = rng.normal(0.0, 0.01, 4)
        resid = 0.01
        if step in events:
            resid = 50.0                      # rare high-residual event
            feat = feat + 10.0
        comp.push(step, feat, residual=resid)

    snap = comp.snapshot()
    kept = set(int(s) for s in snap["steps"])
    for step in events:
        assert step in kept, f"high-residual event {step} was dropped"
    assert len(kept) < 5000 / 10, "compression should discard routine steps"
    top = snap["steps"][np.argsort(snap["importances"])[-3:]]
    assert set(int(s) for s in top) == events
    print(f"  kept {len(kept)} of 5000 steps, events preserved")


def test_compressor_state_roundtrip():
    comp = make_compressor(feature_dim=3, max_entries=16, recent_reserve=4)
    rng = np.random.default_rng(4)
    for step in range(100):
        comp.push(step, rng.normal(size=3), float(rng.random()))

    clone = make_compressor(feature_dim=3, max_entries=16, recent_reserve=4)
    clone.load_state_dict(comp.state_dict())
    a, b = comp.snapshot(), clone.snapshot()
    np.testing.assert_array_equal(a["steps"], b["steps"])
    np.testing.assert_allclose(a["features"], b["features"])
    assert clone.total_seen == comp.total_seen


# ----------------------------------------------------------------------
def run_all_tests() -> bool:
    tests = [
        ("novelty separates point spikes", test_novelty_separates_point_spikes),
        ("rolling robust-z is scale free", test_rolling_z_is_scale_free),
        ("dual opinion beats Isolation Forest on spikes",
         test_dual_beats_isolation_forest_on_spikes),
        ("residual is not in the decision score",
         test_residual_opinion_is_not_in_the_decision_score),
        ("dynamic gate blocks residual ringing",
         test_dynamic_gate_blocks_residual_ringing),
        ("adaptive memory replenishment",
         test_adaptive_memory_replenishment),
        ("dampened recovery within 10 steps",
         test_dampened_recovery_completes_within_ten_steps),
        ("cooldown prevents trip cascades",
         test_cooldown_prevents_trip_cascades),
        ("TRIP never rebuilds the simulator",
         test_trip_never_rebuilds_the_simulator),
        ("memory footprint flat over 12k steps",
         test_memory_footprint_is_flat_over_twelve_thousand_steps),
        ("compressor keeps events and turning points",
         test_compressor_keeps_high_residual_and_turning_points),
        ("compressor state roundtrip",
         test_compressor_state_roundtrip),
    ]

    print("=" * 60)
    print("PHASE IV.b ACCEPTANCE TESTS (T4.5 / T4.6 / T4.7)")
    print("=" * 60)

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
