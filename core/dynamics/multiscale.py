"""Multi-Scale Temporal Dynamics Core (T4.1).

Implements hierarchical time constants (Multi-Layer Time Constants, τ) with
gated cross-scale connections. Fast layers process high-frequency impulses
(ms-level, e.g., mathematical chaos), slow layers process slow evolution
(seconds to minutes, e.g., system load). Gated connections prevent phase
lag and gradient oscillation between scales.

Architecture:
- MultiScaleDynamics: Main class orchestrating multiple NengoDynamics layers
- Each layer has its own time constant τ (simulation dt scaling)
- Cross-scale gates: learnable/parameterized connections with optional delays
- Input routing: frequency-aware or explicit per-source routing
- Output aggregation: weighted combination with learnable or fixed weights
"""

from __future__ import annotations

import numpy as np
import nengo
from dataclasses import dataclass, field
from typing import Optional, Callable, Sequence
from collections import deque

from core.dynamics.engine import NengoDynamics


@dataclass
class ScaleConfig:
    """Configuration for a single temporal scale layer."""
    name: str
    tau: float                    # Time constant in seconds (e.g., 0.005, 0.05, 0.5)
    n_neurons: int = 300
    feature_dim: int = 32
    recurrent_gain: float = 0.5
    seed: int = 0
    synapse: Optional[float] = None  # Defaults to tau if None
    # Simulation step.  Every scale advances on the same control grid;
    # tau (not dt) carries the time-scale separation.  Advancing slow
    # layers with dt=tau made a control chunk span 25-100x more
    # simulated time in the slow layers, which destroyed cross-scale
    # phase alignment instead of creating a hierarchy.
    dt: float = 0.005
    # Input projection from previous scale (if not bottom layer)
    input_gain: float = 0.3
    # Output gate to next scale (if not top layer)
    output_gain: float = 0.3
    # Gate dynamics
    gate_tau: float = 0.01        # Gate time constant
    gate_type: str = "sigmoid"    # "sigmoid", "tanh", "linear", "hard"
    

@dataclass
class MultiScaleConfig:
    """Configuration for the full multi-scale system."""
    scales: list[ScaleConfig]
    # Cross-scale gating
    cross_scale_gain: float = 0.2
    # How to combine outputs from all scales
    aggregation: str = "weighted_sum"  # "weighted_sum", "concat", "attention"
    # Optional learnable aggregation weights (one per scale)
    agg_weights: Optional[list[float]] = None
    # Global seed
    seed: int = 0


class GatedConnection:
    """Parameterized gated connection between scales with optional delay.
    
    Implements: y(t) = gate(x(t - delay)) * W * x(t - delay)
    where gate is a learnable/parameterized nonlinearity.
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 gain: float = 0.3,
                 delay_steps: int = 0,
                 gate_type: str = "sigmoid",
                 gate_tau: float = 0.01,
                 seed: int = 0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gain = gain
        self.delay_steps = delay_steps
        self.gate_type = gate_type
        self.gate_tau = gate_tau
        
        rng = np.random.default_rng(seed)
        # Orthogonal projection matrix
        q, _ = np.linalg.qr(rng.standard_normal((output_dim, input_dim)))
        self.W = gain * q
        
        # Gate parameters
        self.gate_bias = np.zeros(output_dim)
        self.gate_scale = np.ones(output_dim)
        
        # Delay buffer
        self.delay_buffer: deque[np.ndarray] = deque(maxlen=max(1, delay_steps))
        for _ in range(delay_steps):
            self.delay_buffer.append(np.zeros(input_dim))
    
    def set_gate_params(self, bias: np.ndarray | None = None, 
                        scale: np.ndarray | None = None) -> None:
        """Set gate bias and scale parameters."""
        if bias is not None:
            self.gate_bias = np.asarray(bias, dtype=float)
        if scale is not None:
            self.gate_scale = np.asarray(scale, dtype=float)
    
    def compute_gate(self, x: np.ndarray) -> np.ndarray:
        """Compute gate activation for input x."""
        # x shape: (input_dim,) or (batch, input_dim)
        pre_gate = (self.W @ x.T).T if x.ndim == 2 else self.W @ x
        pre_gate = pre_gate * self.gate_scale + self.gate_bias
        
        if self.gate_type == "sigmoid":
            return 1.0 / (1.0 + np.exp(-pre_gate))
        elif self.gate_type == "tanh":
            return np.tanh(pre_gate)
        elif self.gate_type == "linear":
            return pre_gate
        elif self.gate_type == "hard":
            return (pre_gate > 0).astype(float)
        else:
            raise ValueError(f"Unknown gate_type: {self.gate_type}")
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply gated connection with delay.
        
        If delay_steps=0, uses current input directly (no delay).
        If delay_steps>0, uses input from delay_steps steps ago.
        """
        if self.delay_steps == 0:
            # No delay - use current input directly
            delayed = x
        else:
            # Get delayed input (from previous step(s))
            delayed = self.delay_buffer[0]
        
        # Compute gated output
        gate = self.compute_gate(delayed)
        output = (self.W @ delayed) * gate
        
        # Push current input to delay buffer for next call
        if self.delay_steps > 0:
            self.delay_buffer.append(x)
        
        return output
    
    def reset(self) -> None:
        """Clear delay buffer."""
        for i in range(len(self.delay_buffer)):
            self.delay_buffer[i] = np.zeros(self.input_dim)


class MultiScaleDynamics:
    """Multi-scale temporal dynamics with hierarchical time constants.
    
    Each scale runs a NengoDynamics ensemble with its own time constant τ.
    Scales are connected via gated connections to allow information flow
    without phase lag or gradient oscillation.
    
    Usage:
        config = MultiScaleConfig(scales=[
            ScaleConfig(name="fast", tau=0.005, n_neurons=300, feature_dim=32),
            ScaleConfig(name="medium", tau=0.05, n_neurons=200, feature_dim=32),
            ScaleConfig(name="slow", tau=0.5, n_neurons=100, feature_dim=32),
        ])
        msd = MultiScaleDynamics(config)
        result = msd.run_chunk(features_dict)  # features per scale or shared
    """
    
    def __init__(self, config: MultiScaleConfig):
        self.config = config
        self.scales_config = {s.name: s for s in config.scales}
        self.scale_names = [s.name for s in config.scales]
        self.n_scales = len(config.scales)
        
        # Build each scale's NengoDynamics
        self.scales: dict[str, NengoDynamics] = {}
        self.scale_probes: dict[str, tuple] = {}
        
        for i, sc in enumerate(config.scales):
            synapse = sc.synapse if sc.synapse is not None else sc.tau

            dyn = NengoDynamics(
                feature_dim=sc.feature_dim,
                n_neurons=sc.n_neurons,
                dt=sc.dt,
                recurrent_gain=sc.recurrent_gain,
                seed=config.seed + i * 1000,
                synapse=synapse,
                recurrent_synapse=sc.tau,
            )
            self.scales[sc.name] = dyn
        
        # Build cross-scale gated connections
        self.gates: dict[tuple[str, str], GatedConnection] = {}
        for i in range(self.n_scales - 1):
            lower = config.scales[i]
            upper = config.scales[i + 1]
            # Lower -> Upper (fast to slow)
            gate = GatedConnection(
                input_dim=lower.feature_dim,
                output_dim=upper.feature_dim,
                gain=config.cross_scale_gain,
                delay_steps=1,  # One step delay to prevent direct feedback
                gate_type=upper.gate_type,
                gate_tau=upper.gate_tau,
                seed=config.seed + i * 100 + 500
            )
            self.gates[(lower.name, upper.name)] = gate
            
            # Upper -> Lower (slow to fast) - top-down modulation
            gate_td = GatedConnection(
                input_dim=upper.feature_dim,
                output_dim=lower.feature_dim,
                gain=config.cross_scale_gain * 0.5,  # Weaker top-down
                delay_steps=1,
                gate_type=lower.gate_type,
                gate_tau=lower.gate_tau,
                seed=config.seed + i * 100 + 600
            )
            self.gates[(upper.name, lower.name)] = gate_td
        
        # Aggregation weights
        if config.agg_weights is not None:
            self.agg_weights = np.array(config.agg_weights, dtype=float)
        else:
            # Default: equal weights, normalized
            self.agg_weights = np.ones(self.n_scales) / self.n_scales
        
        # Track steps per scale (since they run at different dt)
        self.steps_run = {name: 0 for name in self.scale_names}
        self.global_step = 0
        
        # Cache for cross-scale inputs
        self._cross_inputs: dict[str, np.ndarray] = {
            name: np.zeros(self.scales_config[name].feature_dim) 
            for name in self.scale_names
        }
    
    def run_chunk(self, 
                  features: dict[str, np.ndarray] | np.ndarray,
                  n_steps: Optional[int] = None) -> dict[str, dict]:
        """Advance all scales over a chunk of steps.
        
        Args:
            features: Either:
                - dict mapping scale_name -> (n_steps, feature_dim) array
                - Single (n_steps, feature_dim) array broadcast to all scales
            n_steps: Number of steps (inferred from features if not given)
        
        Returns:
            Dict mapping scale_name -> NengoDynamics.run_chunk output dict
        """
        # Normalize input
        if isinstance(features, np.ndarray):
            # Broadcast to all scales
            feats_dict = {name: features for name in self.scale_names}
            if n_steps is None:
                n_steps = features.shape[0]
        else:
            feats_dict = features
            if n_steps is None:
                n_steps = next(iter(features.values())).shape[0]
        
        # Run each scale
        results = {}
        
        # First pass: compute cross-scale inputs from previous step's outputs
        # (using cached decoded outputs from last run)
        cross_inputs = {name: np.zeros(sc.feature_dim) 
                       for name, sc in self.scales_config.items()}
        
        for i in range(self.n_scales - 1):
            lower = self.config.scales[i]
            upper = self.config.scales[i + 1]
            
            # Lower -> Upper
            if lower.name in self._cross_inputs and upper.name in self.scales:
                gate_key = (lower.name, upper.name)
                if gate_key in self.gates:
                    cross_inputs[upper.name] += self.gates[gate_key](
                        self._cross_inputs[lower.name])
            
            # Upper -> Lower (top-down)
            gate_key = (upper.name, lower.name)
            if gate_key in self.gates:
                cross_inputs[lower.name] += self.gates[gate_key](
                    self._cross_inputs[upper.name])
        
        # Second pass: run each scale with cross-scale inputs added
        for name in self.scale_names:
            sc = self.scales_config[name]
            dyn = self.scales[name]
            
            # Get input features for this scale
            scale_feats = feats_dict[name]
            if scale_feats.shape[0] != n_steps:
                raise ValueError(f"Scale {name}: expected {n_steps} steps, "
                               f"got {scale_feats.shape[0]}")
            
            # Add cross-scale input (broadcast across steps)
            if np.any(cross_inputs[name] != 0):
                scale_feats = scale_feats + cross_inputs[name].reshape(1, -1)
            
            # Run this scale
            result = dyn.run_chunk(scale_feats)
            results[name] = result
            
            # Cache decoded output for next cross-scale computation
            self._cross_inputs[name] = result["decoded"][-1]  # Last step
            
            self.steps_run[name] += n_steps
        
        self.global_step += n_steps
        return results
    
    def get_aggregated_output(self, results: dict[str, dict]) -> np.ndarray:
        """Combine multi-scale outputs according to aggregation strategy.
        
        Returns:
            Aggregated output of shape (n_steps, feature_dim)
        """
        n_steps = results[self.scale_names[0]]["decoded"].shape[0]
        feature_dim = self.config.scales[0].feature_dim
        
        if self.config.aggregation == "weighted_sum":
            agg = np.zeros((n_steps, feature_dim))
            for i, name in enumerate(self.scale_names):
                agg += self.agg_weights[i] * results[name]["decoded"]
            return agg
        
        elif self.config.aggregation == "concat":
            # Concatenate all scale outputs
            parts = [results[name]["decoded"] for name in self.scale_names]
            return np.concatenate(parts, axis=1)
        
        elif self.config.aggregation == "attention":
            # Simple attention: weight by inverse residual/rate
            # For now, fall back to weighted sum
            return self.get_aggregated_output(results)
        
        else:
            raise ValueError(f"Unknown aggregation: {self.config.aggregation}")
    
    def run_chunk_aggregated(self, 
                             features: dict[str, np.ndarray] | np.ndarray,
                             n_steps: Optional[int] = None) -> dict:
        """Run all scales and return aggregated output + per-scale details."""
        results = self.run_chunk(features, n_steps)
        aggregated = self.get_aggregated_output(results)
        
        # Compute aggregate stats
        all_rates = np.concatenate([r["rate_mean"] for r in results.values()])
        all_spikes = np.concatenate([r["spike_frac"] for r in results.values()])
        total_nan = sum(r["nan_decoded"] for r in results.values())
        
        return {
            "aggregated": aggregated,
            "per_scale": results,
            "rate_mean": float(np.mean(all_rates)),
            "rate_max": float(np.max([r["rate_max"].max() for r in results.values()])),
            "spike_frac": float(np.mean(all_spikes)),
            "nan_decoded": total_nan,
            "steps_run": self.global_step,
        }
    
    def estimate_lag(self, features: dict[str, np.ndarray] | np.ndarray,
                     decoded: np.ndarray, max_lag: int = 4) -> int:
        """Estimate lag between input and aggregated decoded output."""
        if isinstance(features, dict):
            # Use first scale's features as reference
            ref_feats = features[self.scale_names[0]]
        else:
            ref_feats = features
        
        # Use the first scale's estimator
        return self.scales[self.scale_names[0]].estimate_lag(
            ref_feats, decoded, max_lag)
    
    def reset(self, warm_feats: dict[str, np.ndarray] | np.ndarray | None = None) -> None:
        """Reset all scales, optionally warm-starting."""
        if warm_feats is None:
            for dyn in self.scales.values():
                dyn.reset()
        elif isinstance(warm_feats, dict):
            for name, dyn in self.scales.items():
                dyn.reset(warm_feats.get(name))
        else:
            # Broadcast
            for dyn in self.scales.values():
                dyn.reset(warm_feats)
        
        # Reset gates
        for gate in self.gates.values():
            gate.reset()
        
        # Sync steps_run from scales (NengoDynamics accumulates steps_run across resets)
        self.steps_run = {name: dyn.steps_run for name, dyn in self.scales.items()}
        
        self.global_step = 0
        self._cross_inputs = {name: np.zeros(sc.feature_dim) 
                             for name, sc in self.scales_config.items()}
    
    def set_aggregation_weights(self, weights: Sequence[float]) -> None:
        """Set custom aggregation weights (normalized)."""
        w = np.array(weights, dtype=float)
        if w.shape != (self.n_scales,):
            raise ValueError(f"Expected {self.n_scales} weights, got {len(w)}")
        self.agg_weights = w / (w.sum() + 1e-12)
    
    def get_scale_state(self, name: str) -> dict:
        """Get state dict for a specific scale."""
        dyn = self.scales[name]
        return {
            "name": name,
            "steps_run": dyn.steps_run,
            "dt": dyn.dt,
            "feature_dim": dyn.feature_dim,
        }
    
    def state_dict(self) -> dict:
        """Serialize full multi-scale state."""
        return {
            "config": {
                "scales": [
                    {
                        "name": sc.name,
                        "tau": sc.tau,
                        "n_neurons": sc.n_neurons,
                        "feature_dim": sc.feature_dim,
                        "recurrent_gain": sc.recurrent_gain,
                        "seed": sc.seed,
                        "synapse": sc.synapse,
                        "dt": sc.dt,
                        "input_gain": sc.input_gain,
                        "output_gain": sc.output_gain,
                        "gate_tau": sc.gate_tau,
                        "gate_type": sc.gate_type,
                    }
                    for sc in self.config.scales
                ],
                "cross_scale_gain": self.config.cross_scale_gain,
                "aggregation": self.config.aggregation,
                "agg_weights": self.agg_weights.tolist(),
                "seed": self.config.seed,
            },
            "global_step": self.global_step,
            "steps_run": self.steps_run,
            "scale_states": {name: self.get_scale_state(name) 
                           for name in self.scale_names},
            "gate_states": {
                f"{k[0]}->{k[1]}": {
                    "W": v.W.tolist(),
                    "gate_bias": v.gate_bias.tolist(),
                    "gate_scale": v.gate_scale.tolist(),
                }
                for k, v in self.gates.items()
            },
            "cross_inputs": {k: v.tolist() for k, v in self._cross_inputs.items()},
        }
    
    def load_state_dict(self, d: dict) -> None:
        """Load multi-scale state (partial - scales need rebuild)."""
        self.global_step = d.get("global_step", 0)
        self.steps_run = d.get("steps_run", self.steps_run)
        self.agg_weights = np.array(d["config"]["agg_weights"])
        
        # Note: Full restore requires rebuilding Nengo simulators
        # This is a metadata-only restore for configuration
        if "gate_states" in d:
            for key, state in d["gate_states"].items():
                src, dst = key.split("->")
                if (src, dst) in self.gates:
                    gate = self.gates[(src, dst)]
                    gate.W = np.array(state["W"])
                    gate.gate_bias = np.array(state["gate_bias"])
                    gate.gate_scale = np.array(state["gate_scale"])
        
        if "cross_inputs" in d:
            for k, v in d["cross_inputs"].items():
                self._cross_inputs[k] = np.array(v)


def create_default_multiscale(feature_dim: int = 32,
                               n_scales: int = 3,
                               tau_base: float = 0.005,
                               tau_factor: float = 10.0,
                               neurons_per_scale: Sequence[int] = (300, 200, 100),
                               seed: int = 0) -> MultiScaleDynamics:
    """Factory for a standard multi-scale configuration.
    
    Args:
        feature_dim: Feature dimension for each scale
        n_scales: Number of scales (typically 3: fast, medium, slow)
        tau_base: Base time constant for fastest scale (seconds)
        tau_factor: Multiplicative factor between adjacent scales
        neurons_per_scale: Neuron counts per scale (len >= n_scales)
        seed: Random seed
    
    Returns:
        Configured MultiScaleDynamics instance
    """
    scales = []
    for i in range(n_scales):
        tau = tau_base * (tau_factor ** i)
        name = ["fast", "medium", "slow", "slower", "slowest"][i] if i < 5 else f"scale_{i}"
        scales.append(ScaleConfig(
            name=name,
            tau=tau,
            n_neurons=neurons_per_scale[i] if i < len(neurons_per_scale) else 100,
            feature_dim=feature_dim,
            recurrent_gain=0.5,
            seed=seed + i,
            gate_type="sigmoid",
            gate_tau=tau * 2,  # Gate slower than layer
        ))
    
    config = MultiScaleConfig(
        scales=scales,
        cross_scale_gain=0.2,
        aggregation="weighted_sum",
        agg_weights=None,  # Equal weights
        seed=seed,
    )
    
    return MultiScaleDynamics(config)


# Convenience function for Phase IV integration
def make_multiscale_dynamics(feature_dim: int = 32, 
                             config: Optional[MultiScaleConfig] = None,
                             **kwargs) -> MultiScaleDynamics:
    """Create multi-scale dynamics with sensible defaults."""
    if config is None:
        config = MultiScaleConfig(
            scales=[
                ScaleConfig(name="fast", tau=0.005, n_neurons=300, feature_dim=feature_dim, seed=0),
                ScaleConfig(name="medium", tau=0.05, n_neurons=200, feature_dim=feature_dim, seed=1),
                ScaleConfig(name="slow", tau=0.5, n_neurons=100, feature_dim=feature_dim, seed=2),
            ],
            cross_scale_gain=0.2,
            aggregation="weighted_sum",
            seed=0,
        )
    return MultiScaleDynamics(config)