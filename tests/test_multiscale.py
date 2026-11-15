"""Unit tests for Multi-Scale Temporal Dynamics (T4.1)."""

import sys
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.dynamics.multiscale import (
    MultiScaleDynamics,
    ScaleConfig,
    MultiScaleConfig,
    GatedConnection,
    create_default_multiscale,
    make_multiscale_dynamics,
)


def test_gated_connection_basic():
    """Test basic gated connection functionality."""
    print("\n[TEST] GatedConnection basic functionality")
    
    gate = GatedConnection(
        input_dim=8,
        output_dim=8,
        gain=1.0,  # Higher gain for stronger signal
        delay_steps=0,  # No delay for this test
        gate_type="linear",  # Use linear for predictable output
        seed=42
    )
    
    # Test with zeros - first call should see zeros from buffer
    x = np.zeros(8)
    out = gate(x)
    assert out.shape == (8,), f"Output shape wrong: {out.shape}"
    assert np.allclose(out, 0), "Zero input should give zero output"
    
    # Test with non-zero - with linear gate, output should be W @ x
    x = np.ones(8) * 0.5
    out = gate(x)
    assert out.shape == (8,), f"Output shape wrong: {out.shape}"
    # With linear gate and non-zero input, output should be non-zero
    assert np.any(np.abs(out) > 1e-10), "Non-zero input should give non-zero output with linear gate"
    
    # Test delay buffer behavior
    gate2 = GatedConnection(input_dim=4, output_dim=4, delay_steps=2, gate_type="linear", seed=42)
    x1 = np.array([1., 0., 0., 0.])
    x2 = np.array([0., 1., 0., 0.])
    x3 = np.array([0., 0., 1., 0.])
    
    out1 = gate2(x1)  # Should use zeros from buffer (delay=2)
    out2 = gate2(x2)  # Should use zeros (first delay slot)
    out3 = gate2(x3)  # Should use x1 (after 2 delays)
    
    # First two outputs should be near zero (delay buffer initialized to zeros)
    assert np.allclose(out1, 0, atol=1e-6), "First step should see zeros"
    assert np.allclose(out2, 0, atol=1e-6), "Second step should see zeros"
    # Third should use x1
    assert np.any(np.abs(out3) > 1e-10), "Third step should see x1"
    
    print("  [OK] GatedConnection basic PASSED")
    return True


def test_gated_connection_gate_types():
    """Test different gate activation types."""
    print("\n[TEST] GatedConnection gate types")
    
    for gate_type in ["sigmoid", "tanh", "linear", "hard"]:
        gate = GatedConnection(
            input_dim=4,
            output_dim=4,
            gain=1.0,
            delay_steps=0,
            gate_type=gate_type,
            seed=42
        )
        x = np.array([1., -1., 2., -2.])
        out = gate(x)
        assert out.shape == (4,), f"Gate {gate_type}: wrong shape"
        assert np.all(np.isfinite(out)), f"Gate {gate_type}: non-finite output"
    
    print("  [OK] GatedConnection gate types PASSED")
    return True


def test_multiscale_creation():
    """Test MultiScaleDynamics creation with default config."""
    print("\n[TEST] MultiScaleDynamics creation")
    
    msd = create_default_multiscale(feature_dim=16, n_scales=3, seed=42)
    
    assert msd.n_scales == 3, f"Expected 3 scales, got {msd.n_scales}"
    assert msd.scale_names == ["fast", "medium", "slow"], f"Wrong scale names: {msd.scale_names}"
    assert len(msd.scales) == 3, f"Expected 3 scale dynamics, got {len(msd.scales)}"
    
    # Check time constants
    assert msd.scales_config["fast"].tau == 0.005
    assert msd.scales_config["medium"].tau == 0.05
    assert msd.scales_config["slow"].tau == 0.5
    
    # Check cross-scale gates exist
    assert ("fast", "medium") in msd.gates
    assert ("medium", "fast") in msd.gates
    assert ("medium", "slow") in msd.gates
    assert ("slow", "medium") in msd.gates
    
    print("  [OK] MultiScaleDynamics creation PASSED")
    return True


def test_multiscale_run_chunk():
    """Test running a chunk through multi-scale dynamics."""
    print("\n[TEST] MultiScaleDynamics run_chunk")
    
    msd = create_default_multiscale(feature_dim=16, n_scales=3, seed=42)
    
    # Run with broadcast features
    n_steps = 10
    features = np.random.default_rng(42).normal(0, 0.5, (n_steps, 16))
    
    results = msd.run_chunk(features)
    
    # Check all scales ran
    assert set(results.keys()) == {"fast", "medium", "slow"}
    
    for name in ["fast", "medium", "slow"]:
        assert "decoded" in results[name]
        assert results[name]["decoded"].shape == (n_steps, 16)
        assert "rate_mean" in results[name]
        assert results[name]["rate_mean"].shape == (n_steps,)
        assert np.all(np.isfinite(results[name]["decoded"]))
    
    # Check step counters
    assert msd.steps_run["fast"] == n_steps
    assert msd.steps_run["medium"] == n_steps
    assert msd.steps_run["slow"] == n_steps
    assert msd.global_step == n_steps
    
    print("  [OK] MultiScaleDynamics run_chunk PASSED")
    return True


def test_multiscale_per_scale_features():
    """Test running with per-scale features."""
    print("\n[TEST] MultiScaleDynamics per-scale features")
    
    msd = create_default_multiscale(feature_dim=8, n_scales=2, seed=42)
    
    n_steps = 5
    features = {
        "fast": np.random.default_rng(1).normal(0, 0.5, (n_steps, 8)),
        "medium": np.random.default_rng(2).normal(0, 0.5, (n_steps, 8)),
    }
    
    results = msd.run_chunk(features)
    
    assert set(results.keys()) == {"fast", "medium"}
    for name in ["fast", "medium"]:
        assert results[name]["decoded"].shape == (n_steps, 8)
    
    print("  [OK] MultiScaleDynamics per-scale features PASSED")
    return True


def test_multiscale_aggregation():
    """Test output aggregation strategies."""
    print("\n[TEST] MultiScaleDynamics aggregation")
    
    # Test weighted_sum (default)
    msd = create_default_multiscale(feature_dim=8, n_scales=2, seed=42)
    n_steps = 5
    features = np.random.default_rng(42).normal(0, 0.5, (n_steps, 8))
    
    results = msd.run_chunk(features)
    agg = msd.get_aggregated_output(results)
    
    assert agg.shape == (n_steps, 8), f"Weighted sum shape: {agg.shape}"
    assert np.all(np.isfinite(agg))
    
    # Test concat
    msd2 = create_default_multiscale(feature_dim=8, n_scales=2, seed=42)
    msd2.config.aggregation = "concat"
    results2 = msd2.run_chunk(features)
    agg2 = msd2.get_aggregated_output(results2)
    
    assert agg2.shape == (n_steps, 16), f"Concat shape: {agg2.shape}"
    
    # Test run_chunk_aggregated
    msd3 = create_default_multiscale(feature_dim=8, n_scales=2, seed=42)
    out = msd3.run_chunk_aggregated(features)
    
    assert "aggregated" in out
    assert "per_scale" in out
    assert out["aggregated"].shape == (n_steps, 8)
    assert "rate_mean" in out
    assert "nan_decoded" in out
    
    print("  [OK] MultiScaleDynamics aggregation PASSED")
    return True


def test_multiscale_reset():
    """Test reset functionality."""
    print("\n[TEST] MultiScaleDynamics reset")
    
    msd = create_default_multiscale(feature_dim=8, n_scales=2, seed=42)
    
    # Run some steps
    features = np.random.default_rng(42).normal(0, 0.5, (10, 8))
    msd.run_chunk(features)
    
    assert msd.global_step == 10
    assert msd.steps_run["fast"] == 10
    
    # Reset without warm start - this rebuilds simulators
    # Note: NengoDynamics._steps_run is NOT reset by _build(), it accumulates
    msd.reset()
    
    assert msd.global_step == 0
    # steps_run accumulates across resets (NengoDynamics behavior)
    assert msd.steps_run["fast"] == 10
    assert msd.steps_run["medium"] == 10
    
    # Reset with warm start (broadcast)
    warm = np.random.default_rng(1).normal(0, 0.5, (5, 8))
    msd.reset(warm_feats=warm)
    
    assert msd.global_step == 0
    # steps_run increases by warm start steps
    assert msd.steps_run["fast"] == 15  # 10 + 5
    assert msd.steps_run["medium"] == 15  # 10 + 5
    
    # Reset with per-scale warm start
    warm_dict = {
        "fast": np.random.default_rng(2).normal(0, 0.5, (3, 8)),
        "medium": np.random.default_rng(3).normal(0, 0.5, (3, 8)),
    }
    msd.reset(warm_feats=warm_dict)
    
    assert msd.steps_run["fast"] == 18  # 15 + 3
    assert msd.steps_run["medium"] == 18  # 15 + 3
    
    print("  [OK] MultiScaleDynamics reset PASSED")
    return True


def test_multiscale_state_dict():
    """Test state serialization."""
    print("\n[TEST] MultiScaleDynamics state_dict")
    
    msd = create_default_multiscale(feature_dim=8, n_scales=2, seed=42)
    
    # Run a bit
    features = np.random.default_rng(42).normal(0, 0.5, (5, 8))
    msd.run_chunk(features)
    
    state = msd.state_dict()
    
    assert "config" in state
    assert "global_step" in state
    assert "steps_run" in state
    assert "scale_states" in state
    assert "gate_states" in state
    assert "cross_inputs" in state
    
    assert state["global_step"] == 5
    assert state["steps_run"]["fast"] == 5
    assert len(state["gate_states"]) == 2  # fast->medium, medium->fast
    
    # Verify gate state structure
    for gate_key, gate_state in state["gate_states"].items():
        assert "W" in gate_state
        assert "gate_bias" in gate_state
        assert "gate_scale" in gate_state
    
    print("  [OK] MultiScaleDynamics state_dict PASSED")
    return True


def test_custom_multiscale_config():
    """Test creating multi-scale with custom config."""
    print("\n[TEST] Custom MultiScaleConfig")
    
    config = MultiScaleConfig(
        scales=[
            ScaleConfig(name="ultra_fast", tau=0.001, n_neurons=100, feature_dim=16, seed=0),
            ScaleConfig(name="fast", tau=0.01, n_neurons=200, feature_dim=16, seed=1),
            ScaleConfig(name="slow", tau=0.1, n_neurons=150, feature_dim=16, seed=2),
        ],
        cross_scale_gain=0.15,
        aggregation="concat",
        seed=42,
    )
    
    msd = MultiScaleDynamics(config)
    
    assert msd.scale_names == ["ultra_fast", "fast", "slow"]
    assert msd.config.aggregation == "concat"
    assert msd.config.cross_scale_gain == 0.15
    assert msd.n_scales == 3
    
    # Check time constants
    assert msd.scales_config["ultra_fast"].tau == 0.001
    assert msd.scales_config["fast"].tau == 0.01
    assert msd.scales_config["slow"].tau == 0.1
    
    # Test run
    features = np.random.default_rng(42).normal(0, 0.5, (5, 16))
    results = msd.run_chunk(features)
    
    assert set(results.keys()) == {"ultra_fast", "fast", "slow"}
    agg = msd.get_aggregated_output(results)
    assert agg.shape == (5, 48)  # concat: 3 * 16
    
    print("  [OK] Custom MultiScaleConfig PASSED")
    return True


def test_fast_vs_slow_dynamics():
    """Test that different scales exhibit different temporal behaviors."""
    print("\n[TEST] Fast vs Slow dynamics differentiation")
    
    # Use a smaller tau_factor so slow scale isn't too slow to show dynamics
    msd = create_default_multiscale(feature_dim=8, n_scales=3, tau_base=0.005, tau_factor=5.0, seed=42)
    
    # Run a longer sequence with oscillating input
    n_steps = 400  # More steps for slow scale to show dynamics
    t = np.arange(n_steps) * 0.005  # fast scale dt
    # Input with both fast and slow components
    fast_signal = np.sin(2 * np.pi * 10 * t)  # ~10 Hz
    slow_signal = np.sin(2 * np.pi * 0.5 * t)  # ~0.5 Hz
    input_signal = fast_signal + 0.5 * slow_signal
    
    features = np.tile(input_signal[:, np.newaxis], (1, 8)) * 0.5
    features += np.random.default_rng(42).normal(0, 0.1, (n_steps, 8))
    
    results = msd.run_chunk(features)
    
    # Both should produce valid outputs
    for name in ["fast", "slow"]:
        decoded = results[name]["decoded"]
        assert decoded.shape == (n_steps, 8)
        assert np.all(np.isfinite(decoded))
        # Should have non-zero variance
        std = np.std(decoded)
        print(f"  {name} scale std: {std:.6f}")
        assert std > 0.005, f"{name} scale has near-zero variance: std={std}"
    
    # Check that they're different
    fast_decoded = results["fast"]["decoded"][:, 0]
    slow_decoded = results["slow"]["decoded"][:, 0]
    diff = np.mean(np.abs(fast_decoded - slow_decoded))
    assert diff > 0.01, f"Scales too similar: diff={diff}"
    
    print(f"  Fast-slow difference: {diff:.4f}")
    print(f"  Fast std: {np.std(fast_decoded):.4f}, Slow std: {np.std(slow_decoded):.4f}")
    print("  [OK] Fast vs Slow dynamics differentiation PASSED")
    return True


def run_all_tests():
    """Run all multi-scale dynamics tests."""
    print("=" * 60)
    print("MULTI-SCALE TEMPORAL DYNAMICS TESTS (T4.1)")
    print("=" * 60)
    
    tests = [
        ("GatedConnection Basic", test_gated_connection_basic),
        ("GatedConnection Gate Types", test_gated_connection_gate_types),
        ("MultiScaleDynamics Creation", test_multiscale_creation),
        ("MultiScaleDynamics Run Chunk", test_multiscale_run_chunk),
        ("MultiScaleDynamics Per-Scale Features", test_multiscale_per_scale_features),
        ("MultiScaleDynamics Aggregation", test_multiscale_aggregation),
        ("MultiScaleDynamics Reset", test_multiscale_reset),
        ("MultiScaleDynamics State Dict", test_multiscale_state_dict),
        ("Custom MultiScaleConfig", test_custom_multiscale_config),
        ("Fast vs Slow Dynamics", test_fast_vs_slow_dynamics),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n[OK] ALL MULTI-SCALE DYNAMICS TESTS PASSED")
        return True
    else:
        print(f"\n[FAIL] {failed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)