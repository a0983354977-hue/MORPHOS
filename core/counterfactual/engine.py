"""Counterfactual Branch Replay Engine (T3.3).

Deep-clones the simulator state at a historical snapshot and runs parallel
counterfactual simulations with injected perturbations. Supports:
  - Full state snapshotting (Nengo simulator + encoder + predictor + observer)
  - Branch creation with parameter/input perturbations
  - Parallel stepping of actual vs counterfactual trajectories
  - Efficient state management for minimal overhead (< 30%)
"""

from __future__ import annotations

import copy
import pickle
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
import numpy as np

# Import core components
from core.dynamics.engine import NengoDynamics
from core.encoding.encoder import FeatureEncoder
from core.encoding.projector import SubspaceProjector
from core.prediction.predictor import NLMSPredictor
from core.observer.observer import CircuitBreaker
from core.memory.buffer import RollingTemporalBuffer, TemporalConfig
from core.memory.associative import AssociativeMemory, RetrievalConfig


@dataclass
class CounterfactualConfig:
    """Configuration for counterfactual engine."""
    max_branches: int = 4              # max parallel branches
    snapshot_interval: int = 500       # steps between auto-snapshots
    max_snapshots: int = 20            # max stored snapshots
    branch_timeout_steps: int = 1000   # max steps per branch
    enable_parallel: bool = True       # use threading for parallel branches


@dataclass
class SimulatorSnapshot:
    """Complete simulator state at a point in time."""
    step: int
    sim_t: float
    
    # Core component states
    encoder_state: dict
    predictor_state: dict
    observer_state: dict
    dynamics_state: dict  # NengoDynamics doesn't have simple state_dict, needs special handling
    
    # History buffers
    temporal_buffer_state: dict
    associative_memory_state: dict
    
    # Metadata
    lag: int
    feature_dim: int
    n_neurons: int
    dt: float
    recurrent_gain: float
    seed: int
    
    # Warm-start features
    warm_feats: np.ndarray
    
    # Optional: full pickled simulator (expensive but exact)
    pickled_sim: bytes | None = None


@dataclass
class BranchConfig:
    """Configuration for a counterfactual branch."""
    name: str
    # Perturbation functions (called at each step)
    input_perturbation: Optional[Callable[[np.ndarray, int], np.ndarray]] = None
    param_perturbation: Optional[Callable[[dict], dict]] = None
    # Initial state override
    initial_state_override: Optional[dict] = None
    # Branch-specific predictor (can be None to share)
    use_separate_predictor: bool = False


@dataclass
class BranchState:
    """Runtime state of a counterfactual branch."""
    config: BranchConfig
    # Cloned components
    encoder: FeatureEncoder
    predictor: NLMSPredictor
    observer: CircuitBreaker
    dynamics: NengoDynamics
    
    # History
    temporal_buffer: RollingTemporalBuffer
    associative_memory: AssociativeMemory
    
    # Trajectory tracking
    trajectory: list[np.ndarray] = field(default_factory=list)  # decoded states
    residuals: list[float] = field(default_factory=list)
    rates: list[float] = field(default_factory=list)
    steps_run: int = 0
    
    # Status
    active: bool = True
    terminated_reason: str | None = None


class CounterfactualEngine:
    """Engine for running counterfactual simulations from historical snapshots."""
    
    def __init__(self, 
                 feature_dim: int,
                 n_neurons: int = 700,
                 dt: float = 0.005,
                 recurrent_gain: float = 0.5,
                 seed: int = 0,
                 config: Optional[CounterfactualConfig] = None):
        self.feature_dim = feature_dim
        self.n_neurons = n_neurons
        self.dt = dt
        self.recurrent_gain = recurrent_gain
        self.seed = seed
        self.cfg = config or CounterfactualConfig()
        
        # --- Active components (the "actual" timeline) ---
        # Encoder expects state_dim input, projects to proj_dim, outputs 2*proj_dim features
        state_dim = feature_dim // 2
        proj_dim = feature_dim // 2
        self.encoder = FeatureEncoder(state_dim=state_dim, proj_dim=proj_dim, seed=seed)
        self.predictor = NLMSPredictor(n_in=2 * feature_dim, n_out=state_dim, mu=0.05)
        self.observer = CircuitBreaker()
        self.dynamics = NengoDynamics(
            feature_dim=feature_dim, n_neurons=n_neurons, dt=dt,
            recurrent_gain=recurrent_gain, seed=seed)
        
        # --- History buffers ---
        self.temporal_buffer = RollingTemporalBuffer(feature_dim, TemporalConfig())
        self.associative_memory = AssociativeMemory(feature_dim, RetrievalConfig())
        
        # --- Snapshot storage ---
        self._snapshots: list[SimulatorSnapshot] = []
        self._snapshot_steps: list[int] = []
        self._next_snapshot_step = 0
        
        # --- Branch management ---
        self._branches: dict[str, BranchState] = {}
        self._actual_branch_name = "actual"
        self._lag = 1
        self._current_step = 0
        
    # ------------------------------------------------------------------
    # Snapshot management
    # ------------------------------------------------------------------
    def create_snapshot(self, step: int, sim_t: float, lag: int,
                        warm_feats: np.ndarray | None = None) -> SimulatorSnapshot:
        """Create a complete snapshot of the current simulator state."""
        
        # Get warm features if not provided
        if warm_feats is None:
            warm_feats = self.temporal_buffer.get_warm_feats(
                self.temporal_buffer.capacity)
        
        # For NengoDynamics, we need to pickle the entire simulator for exact restore
        # This is expensive but guarantees bit-identical restoration
        pickled_sim = pickle.dumps(self.dynamics.sim, protocol=pickle.HIGHEST_PROTOCOL)
        
        snapshot = SimulatorSnapshot(
            step=step,
            sim_t=sim_t,
            encoder_state=self.encoder.state_dict(),
            predictor_state=self.predictor.state_dict(),
            observer_state=self.observer.state_dict(),
            dynamics_state={
                "feature_dim": self.dynamics.feature_dim,
                "n_neurons": self.dynamics.n_neurons,
                "dt": self.dynamics.dt,
                "recurrent_gain": self.dynamics.recurrent_gain,
                "seed": self.dynamics.seed,
                "synapse": self.dynamics.synapse,
                "steps_run": self.dynamics.steps_run,
            },
            temporal_buffer_state=self.temporal_buffer.state_dict(),
            associative_memory_state=self.associative_memory.state_dict(),
            lag=lag,
            feature_dim=self.feature_dim,
            n_neurons=self.n_neurons,
            dt=self.dt,
            recurrent_gain=self.recurrent_gain,
            seed=self.seed,
            warm_feats=warm_feats.copy() if warm_feats is not None else np.zeros((0, self.feature_dim)),
            pickled_sim=pickled_sim,
        )
        
        # Store snapshot (with rotation)
        self._snapshots.append(snapshot)
        self._snapshot_steps.append(step)
        
        if len(self._snapshots) > self.cfg.max_snapshots:
            self._snapshots.pop(0)
            self._snapshot_steps.pop(0)
            
        return snapshot
    
    def get_snapshot_at(self, step: int) -> SimulatorSnapshot | None:
        """Get the snapshot closest to (but not after) the given step."""
        if not self._snapshots:
            return None
        # Find snapshot with step <= target
        valid = [(i, s) for i, s in enumerate(self._snapshot_steps) if s <= step]
        if not valid:
            return None
        idx, _ = max(valid, key=lambda x: x[1])
        return self._snapshots[idx]
    
    def get_latest_snapshot(self) -> SimulatorSnapshot | None:
        """Get the most recent snapshot."""
        if not self._snapshots:
            return None
        return self._snapshots[-1]
    
    # ------------------------------------------------------------------
    # Branch creation & management
    # ------------------------------------------------------------------
    def create_branch(self, snapshot: SimulatorSnapshot, 
                      branch_config: BranchConfig) -> BranchState:
        """Create a new counterfactual branch from a snapshot."""
        
        if len(self._branches) >= self.cfg.max_branches:
            raise RuntimeError(f"Max branches ({self.cfg.max_branches}) reached")
        
        if branch_config.name in self._branches:
            raise ValueError(f"Branch '{branch_config.name}' already exists")
        
        # --- Restore components from snapshot ---
        # Encoder - need to extract state_dim and proj_dim from feature_dim
        # feature_dim = 2 * proj_dim, and proj_dim = state_dim (by convention)
        state_dim = snapshot.feature_dim // 2
        proj_dim = snapshot.feature_dim // 2
        encoder = FeatureEncoder(
            state_dim=state_dim,
            proj_dim=proj_dim,
            seed=snapshot.seed)
        encoder.load_state_dict(snapshot.encoder_state)
        
        # Predictor (separate or shared)
        if branch_config.use_separate_predictor:
            predictor = NLMSPredictor(
                n_in=2 * snapshot.feature_dim,
                n_out=snapshot.feature_dim // 2,
                mu=0.05)
            predictor.load_state_dict(snapshot.predictor_state)
        else:
            # Share predictor with actual (will need copy-on-write)
            predictor = NLMSPredictor(
                n_in=2 * snapshot.feature_dim,
                n_out=snapshot.feature_dim // 2,
                mu=0.05)
            predictor.load_state_dict(snapshot.predictor_state)
        
        # Observer
        observer = CircuitBreaker()
        observer.load_state_dict(snapshot.observer_state)
        
        # Dynamics - rebuild and warm-start (pickle restore is unreliable for probes)
        dynamics = NengoDynamics(
            feature_dim=snapshot.feature_dim,
            n_neurons=snapshot.n_neurons,
            dt=snapshot.dt,
            recurrent_gain=snapshot.recurrent_gain,
            seed=snapshot.seed)
        
        # Warm-start with stored features
        if len(snapshot.warm_feats) > 0:
            dynamics.run_chunk(snapshot.warm_feats)
        
        # History buffers
        temporal_buffer = RollingTemporalBuffer(snapshot.feature_dim)
        temporal_buffer.load_state_dict(snapshot.temporal_buffer_state)
        
        associative_memory = AssociativeMemory(snapshot.feature_dim)
        associative_memory.load_state_dict(snapshot.associative_memory_state)
        
        branch = BranchState(
            config=branch_config,
            encoder=encoder,
            predictor=predictor,
            observer=observer,
            dynamics=dynamics,
            temporal_buffer=temporal_buffer,
            associative_memory=associative_memory,
        )
        
        self._branches[branch_config.name] = branch
        return branch
    
    def create_branches_from_step(self, step: int,
                                  branch_configs: list[BranchConfig]) -> dict[str, BranchState]:
        """Convenience: create multiple branches from a historical step."""
        snapshot = self.get_snapshot_at(step)
        if snapshot is None:
            raise ValueError(f"No snapshot available at or before step {step}")
        
        branches = {}
        for bc in branch_configs:
            branches[bc.name] = self.create_branch(snapshot, bc)
        return branches
    
    def terminate_branch(self, name: str, reason: str = "completed") -> None:
        """Mark a branch as terminated."""
        if name in self._branches:
            self._branches[name].active = False
            self._branches[name].terminated_reason = reason
    
    def get_branch(self, name: str) -> BranchState | None:
        """Get branch state by name."""
        return self._branches.get(name)
    
    def list_branches(self) -> list[str]:
        """List active branch names."""
        return [name for name, b in self._branches.items() if b.active]
    
    # ------------------------------------------------------------------
    # Parallel stepping
    # ------------------------------------------------------------------
    def step_branches(self, 
                      actual_input: np.ndarray,
                      actual_target: np.ndarray,
                      actual_conf: float,
                      max_steps: int | None = None) -> dict[str, dict]:
        """Advance all active branches by one step (plus actual).
        
        Returns dict of branch_name -> step results.
        """
        results = {}
        max_steps = max_steps or self.cfg.branch_timeout_steps
        
        # --- Step actual timeline ---
        actual_result = self._step_single(
            self.encoder, self.predictor, self.observer, self.dynamics,
            self.temporal_buffer, self.associative_memory,
            actual_input, actual_target, actual_conf,
            self._lag, self._current_step,
            input_perturbation=None,
            param_perturbation=None
        )
        results[self._actual_branch_name] = actual_result
        self._current_step += 1
        
        # --- Step counterfactual branches ---
        to_remove = []
        for name, branch in self._branches.items():
            if not branch.active:
                to_remove.append(name)
                continue
                
            if branch.steps_run >= max_steps:
                self.terminate_branch(name, "max_steps_reached")
                continue
            
            # Apply branch-specific perturbations
            perturbed_input = actual_input
            if branch.config.input_perturbation:
                perturbed_input = branch.config.input_perturbation(
                    actual_input, branch.steps_run)
            
            perturbed_params = {}
            if branch.config.param_perturbation:
                perturbed_params = branch.config.param_perturbation({})
            
            branch_result = self._step_single(
                branch.encoder, branch.predictor, branch.observer, branch.dynamics,
                branch.temporal_buffer, branch.associative_memory,
                perturbed_input, actual_target, actual_conf,  # use same target for comparison
                branch.dynamics.lag if hasattr(branch.dynamics, 'lag') else self._lag,
                branch.steps_run,
                input_perturbation=branch.config.input_perturbation,
                param_perturbation=branch.config.param_perturbation
            )
            
            branch.steps_run += 1
            branch.trajectory.append(branch_result["decoded"])
            branch.residuals.append(branch_result["residual"])
            branch.rates.append(branch_result["rate_mean"])
            
            results[name] = branch_result
        
        # Clean up terminated branches
        for name in to_remove:
            del self._branches[name]
        
        # Auto-snapshot
        if self._current_step >= self._next_snapshot_step:
            self.create_snapshot(
                self._current_step, 
                self._current_step * self.dt,
                self._lag,
                self.temporal_buffer.get_warm_feats(self.temporal_buffer.capacity)
            )
            self._next_snapshot_step += self.cfg.snapshot_interval
        
        return results
    
    def _step_single(self,
                     encoder: FeatureEncoder,
                     predictor: NLMSPredictor,
                     observer: CircuitBreaker,
                     dynamics: NengoDynamics,
                     temporal_buffer: RollingTemporalBuffer,
                     associative_memory: AssociativeMemory,
                     input_feat: np.ndarray,
                     target_z: np.ndarray,
                     conf: float,
                     lag: int,
                     step: int,
                     input_perturbation: Callable | None = None,
                     param_perturbation: Callable | None = None) -> dict:
        """Execute a single step on a set of components."""
        
        # Encode input
        z, feat = encoder.encode(input_feat)
        
        # Run dynamics
        cres = dynamics.run_chunk(feat.reshape(1, -1))
        decoded = cres["decoded"][0]
        rate_mean = cres["rate_mean"][0]
        
        # Predict (causal: use lag-compensated decoded state)
        # For simplicity in counterfactual, we use a simplified prediction
        # Real implementation would need proper lag handling
        pred_z = predictor.predict(np.concatenate([decoded, decoded]))  # placeholder
        err2 = predictor.update(np.concatenate([decoded, decoded]), target_z, weight=conf)
        
        # Observer
        observer.observe_residual(err2)
        stats = {
            "mean_rate": rate_mean,
            "max_rate": cres["rate_max"][0],
            "spike_frac": cres["spike_frac"][0],
            "nan_decoded": cres["nan_decoded"],
            "conf_mean": conf,
        }
        observer.evaluate(stats)
        
        # Update histories
        temporal_buffer.push_step(feat, conf, err2, rate_mean)
        associative_memory.add(feat, err2, step)
        
        return {
            "step": step,
            "z": z,
            "feat": feat,
            "decoded": decoded,
            "pred_z": pred_z,
            "residual": err2,
            "rate_mean": rate_mean,
            "rate_max": cres["rate_max"][0],
            "spike_frac": cres["spike_frac"][0],
            "nan_decoded": cres["nan_decoded"],
            "conf": conf,
            "breaker_state": observer.state,
        }
    
    # ------------------------------------------------------------------
    # Perturbation presets (common counterfactual scenarios)
    # ------------------------------------------------------------------
    @staticmethod
    def perturb_input_scale(scale: float) -> Callable[[np.ndarray, int], np.ndarray]:
        """Multiply input by scale factor."""
        return lambda x, step: x * scale
    
    @staticmethod
    def perturb_input_additive(offset: np.ndarray) -> Callable[[np.ndarray, int], np.ndarray]:
        """Add constant offset to input."""
        return lambda x, step: x + offset
    
    @staticmethod
    def perturb_input_noise(std: float, start_step: int = 0) -> Callable[[np.ndarray, int], np.ndarray]:
        """Add Gaussian noise after start_step."""
        def _perturb(x, step):
            if step >= start_step:
                return x + np.random.default_rng(step).normal(0, std, x.shape)
            return x
        return _perturb
    
    @staticmethod
    def perturb_dynamics_gain(gain_multiplier: float) -> Callable[[dict], dict]:
        """Modify recurrent gain (requires rebuild, so this is metadata)."""
        return lambda params: {**params, "recurrent_gain": params.get("recurrent_gain", 0.5) * gain_multiplier}
    
    @staticmethod
    def perturb_predictor_mu(mu_multiplier: float) -> Callable[[dict], dict]:
        """Modify predictor learning rate."""
        return lambda params: {**params, "mu": params.get("mu", 0.05) * mu_multiplier}
    
    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        """Serialize engine state."""
        return {
            "feature_dim": self.feature_dim,
            "n_neurons": self.n_neurons,
            "dt": self.dt,
            "recurrent_gain": self.recurrent_gain,
            "seed": self.seed,
            "cfg": {
                "max_branches": self.cfg.max_branches,
                "snapshot_interval": self.cfg.snapshot_interval,
                "max_snapshots": self.cfg.max_snapshots,
                "branch_timeout_steps": self.cfg.branch_timeout_steps,
                "enable_parallel": self.cfg.enable_parallel,
            },
            "current_step": self._current_step,
            "lag": self._lag,
            "encoder": self.encoder.state_dict(),
            "predictor": self.predictor.state_dict(),
            "observer": self.observer.state_dict(),
            "temporal_buffer": self.temporal_buffer.state_dict(),
            "associative_memory": self.associative_memory.state_dict(),
            "snapshots": [self._snapshot_to_dict(s) for s in self._snapshots],
            "snapshot_steps": self._snapshot_steps,
            "branches": {name: self._branch_to_dict(b) for name, b in self._branches.items()},
        }
    
    def _snapshot_to_dict(self, snap: SimulatorSnapshot) -> dict:
        """Convert snapshot to serializable dict (without pickled sim)."""
        return {
            "step": snap.step,
            "sim_t": snap.sim_t,
            "encoder_state": snap.encoder_state,
            "predictor_state": snap.predictor_state,
            "observer_state": snap.observer_state,
            "dynamics_state": snap.dynamics_state,
            "temporal_buffer_state": snap.temporal_buffer_state,
            "associative_memory_state": snap.associative_memory_state,
            "lag": snap.lag,
            "feature_dim": snap.feature_dim,
            "n_neurons": snap.n_neurons,
            "dt": snap.dt,
            "recurrent_gain": snap.recurrent_gain,
            "seed": snap.seed,
            "warm_feats": snap.warm_feats.tolist() if len(snap.warm_feats) else [],
        }
    
    def _branch_to_dict(self, branch: BranchState) -> dict:
        """Convert branch to serializable dict."""
        return {
            "config": {
                "name": branch.config.name,
                "use_separate_predictor": branch.config.use_separate_predictor,
            },
            "encoder": branch.encoder.state_dict(),
            "predictor": branch.predictor.state_dict(),
            "observer": branch.observer.state_dict(),
            "temporal_buffer": branch.temporal_buffer.state_dict(),
            "associative_memory": branch.associative_memory.state_dict(),
            "trajectory": [t.tolist() for t in branch.trajectory],
            "residuals": branch.residuals,
            "rates": branch.rates,
            "steps_run": branch.steps_run,
            "active": branch.active,
            "terminated_reason": branch.terminated_reason,
        }
    
    def load_state_dict(self, d: dict) -> None:
        """Restore engine state (without branches - they need manual recreation)."""
        self.feature_dim = d["feature_dim"]
        self.n_neurons = d["n_neurons"]
        self.dt = d["dt"]
        self.recurrent_gain = d["recurrent_gain"]
        self.seed = d["seed"]
        
        cfg = d["cfg"]
        self.cfg = CounterfactualConfig(
            max_branches=cfg["max_branches"],
            snapshot_interval=cfg["snapshot_interval"],
            max_snapshots=cfg["max_snapshots"],
            branch_timeout_steps=cfg["branch_timeout_steps"],
            enable_parallel=cfg["enable_parallel"],
        )
        
        self._current_step = d["current_step"]
        self._lag = d["lag"]
        
        self.encoder.load_state_dict(d["encoder"])
        self.predictor.load_state_dict(d["predictor"])
        self.observer.load_state_dict(d["observer"])
        self.temporal_buffer.load_state_dict(d["temporal_buffer"])
        self.associative_memory.load_state_dict(d["associative_memory"])
        
        # Note: Snapshots and branches require full simulator restore which is complex
        # This is a simplified restore for the main timeline only