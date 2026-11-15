"""Trajectory Divergence Observer & Scenario Comparator (T3.4).

Analyzes parallel trajectories from the counterfactual engine to compute:
  - Euclidean / Mahalanobis divergence between actual and counterfactual
  - Residual diffusion rate and recovery time
  - Resilience index (how quickly system returns to baseline)
  - Scenario comparison reports for decision support
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import json
import time


@dataclass
class DivergenceConfig:
    """Configuration for divergence analysis."""
    window_sizes: tuple[int, ...] = (10, 50, 200, 500)  # analysis windows
    recovery_threshold: float = 1.5   # multiple of baseline divergence
    min_recovery_steps: int = 10      # minimum steps to confirm recovery
    divergence_metric: str = "euclidean"  # "euclidean", "mahalanobis", "cosine"
    baseline_steps: int = 100         # steps to establish baseline


@dataclass
class DivergenceMetrics:
    """Metrics for a single time step."""
    step: int
    euclidean_dist: float
    mahalanobis_dist: float
    cosine_dist: float
    residual_ratio: float  # counterfactual_residual / actual_residual
    rate_ratio: float      # counterfactual_rate / actual_rate
    
    # Rolling statistics
    euclidean_ema_10: float = 0.0
    euclidean_ema_50: float = 0.0
    euclidean_ema_200: float = 0.0
    euclidean_ema_500: float = 0.0
    
    # Recovery tracking
    is_diverged: bool = False
    is_recovering: bool = False
    is_recovered: bool = False
    steps_since_divergence: int = 0
    steps_since_recovery_start: int = 0


@dataclass
class ScenarioReport:
    """Complete comparison report for a counterfactual scenario."""
    scenario_name: str
    branch_name: str
    snapshot_step: int
    start_step: int
    end_step: int
    total_steps: int
    
    # Divergence summary
    max_euclidean_divergence: float
    mean_euclidean_divergence: float
    final_euclidean_divergence: float
    
    # Recovery analysis
    diverged: bool
    divergence_step: int | None
    recovery_step: int | None
    recovery_time: int | None  # steps from divergence to recovery
    resilience_index: float    # 0-1, higher = more resilient
    
    # Residual analysis
    max_residual_ratio: float
    mean_residual_ratio: float
    residual_explosion: bool   # residual grew unbounded
    
    # Rate analysis
    max_rate_ratio: float
    mean_rate_ratio: float
    rate_runaway: bool
    
    # Trajectory data (for plotting)
    steps: list[int] = field(default_factory=list)
    euclidean_divergence: list[float] = field(default_factory=list)
    residual_ratios: list[float] = field(default_factory=list)
    rate_ratios: list[float] = field(default_factory=list)
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    config: dict = field(default_factory=dict)


class DivergenceObserver:
    """Real-time divergence tracking between actual and counterfactual trajectories."""
    
    def __init__(self, feature_dim: int, config: Optional[DivergenceConfig] = None):
        self.feature_dim = feature_dim
        self.cfg = config or DivergenceConfig()
        
        # --- Baseline statistics (established during initial run) ---
        self._baseline_established = False
        self._baseline_euclidean: float = 0.0
        self._baseline_residual: float = 0.0
        self._baseline_rate: float = 0.0
        self._baseline_cov: np.ndarray | None = None  # for Mahalanobis
        
        # --- Current tracking ---
        self._actual_trajectory: list[np.ndarray] = []
        self._counterfactual_trajectories: dict[str, list[np.ndarray]] = {}
        self._actual_residuals: list[float] = []
        self._counterfactual_residuals: dict[str, list[float]] = {}
        self._actual_rates: list[float] = []
        self._counterfactual_rates: dict[str, list[float]] = {}
        
        # --- Divergence metrics per branch ---
        self._branch_metrics: dict[str, list[DivergenceMetrics]] = {}
        
        # --- EMA state for each branch and window ---
        self._ema_state: dict[str, dict[int, float]] = {}  # branch -> {window: value}
        self._ema_initialized: dict[str, dict[int, bool]] = {}
        
        # --- Recovery tracking ---
        self._divergence_detected: dict[str, bool] = {}
        self._recovery_started: dict[str, bool] = {}
        self._recovery_confirmed: dict[str, bool] = {}
        self._divergence_step: dict[str, int | None] = {}
        self._recovery_step: dict[str, int | None] = {}
        
    # ------------------------------------------------------------------
    # Baseline establishment
    # ------------------------------------------------------------------
    def establish_baseline(self, 
                          actual_trajectory: list[np.ndarray],
                          actual_residuals: list[float],
                          actual_rates: list[float]) -> None:
        """Compute baseline statistics from clean run."""
        if len(actual_trajectory) < self.cfg.baseline_steps:
            return
            
        # Use last baseline_steps for baseline
        traj = np.array(actual_trajectory[-self.cfg.baseline_steps:])
        residuals = np.array(actual_residuals[-self.cfg.baseline_steps:])
        rates = np.array(actual_rates[-self.cfg.baseline_steps:])
        
        # Self-divergence (should be near zero)
        diffs = np.diff(traj, axis=0)
        self._baseline_euclidean = float(np.mean(np.linalg.norm(diffs, axis=1)))
        
        self._baseline_residual = float(np.mean(residuals))
        self._baseline_rate = float(np.mean(rates))
        
        # Covariance for Mahalanobis
        if len(traj) > self.feature_dim:
            self._baseline_cov = np.cov(traj.T)
            # Regularize
            self._baseline_cov += np.eye(self.feature_dim) * 1e-6
        else:
            self._baseline_cov = np.eye(self.feature_dim)
            
        self._baseline_established = True
    
    # ------------------------------------------------------------------
    # Real-time tracking
    # ------------------------------------------------------------------
    def observe_step(self,
                     step: int,
                     actual_decoded: np.ndarray,
                     actual_residual: float,
                     actual_rate: float,
                     counterfactual_data: dict[str, dict]) -> dict[str, DivergenceMetrics]:
        """Process one step of actual + counterfactual data.
        
        Args:
            step: Global step number
            actual_decoded: Actual trajectory decoded state
            actual_residual: Actual prediction residual
            actual_rate: Actual firing rate
            counterfactual_data: Dict of branch_name -> {
                "decoded": np.ndarray,
                "residual": float,
                "rate": float
            }
            
        Returns:
            Dict of branch_name -> DivergenceMetrics for this step
        """
        # Store actual
        self._actual_trajectory.append(actual_decoded.copy())
        self._actual_residuals.append(actual_residual)
        self._actual_rates.append(actual_rate)
        
        # Trim history to save memory (keep enough for largest window)
        max_keep = max(self.cfg.window_sizes) * 2
        if len(self._actual_trajectory) > max_keep:
            self._actual_trajectory = self._actual_trajectory[-max_keep:]
            self._actual_residuals = self._actual_residuals[-max_keep:]
            self._actual_rates = self._actual_rates[-max_keep:]
        
        results = {}
        
        for branch_name, cf_data in counterfactual_data.items():
            cf_decoded = cf_data["decoded"]
            cf_residual = cf_data["residual"]
            cf_rate = cf_data["rate"]
            
            # Initialize branch tracking
            if branch_name not in self._counterfactual_trajectories:
                self._counterfactual_trajectories[branch_name] = []
                self._counterfactual_residuals[branch_name] = []
                self._counterfactual_rates[branch_name] = []
                self._branch_metrics[branch_name] = []
                self._ema_state[branch_name] = {w: 0.0 for w in self.cfg.window_sizes}
                self._ema_initialized[branch_name] = {w: False for w in self.cfg.window_sizes}
                self._divergence_detected[branch_name] = False
                self._recovery_started[branch_name] = False
                self._recovery_confirmed[branch_name] = False
                self._divergence_step[branch_name] = None
                self._recovery_step[branch_name] = None
            
            # Store counterfactual
            self._counterfactual_trajectories[branch_name].append(cf_decoded.copy())
            self._counterfactual_residuals[branch_name].append(cf_residual)
            self._counterfactual_rates[branch_name].append(cf_rate)
            
            # Trim
            if len(self._counterfactual_trajectories[branch_name]) > max_keep:
                self._counterfactual_trajectories[branch_name] = \
                    self._counterfactual_trajectories[branch_name][-max_keep:]
                self._counterfactual_residuals[branch_name] = \
                    self._counterfactual_residuals[branch_name][-max_keep:]
                self._counterfactual_rates[branch_name] = \
                    self._counterfactual_rates[branch_name][-max_keep:]
            
            # Compute metrics
            metrics = self._compute_metrics(
                branch_name, step, actual_decoded, actual_residual, actual_rate,
                cf_decoded, cf_residual, cf_rate)
            
            self._branch_metrics[branch_name].append(metrics)
            results[branch_name] = metrics
            
        return results
    
    def _compute_metrics(self,
                        branch_name: str,
                        step: int,
                        actual_decoded: np.ndarray,
                        actual_residual: float,
                        actual_rate: float,
                        cf_decoded: np.ndarray,
                        cf_residual: float,
                        cf_rate: float) -> DivergenceMetrics:
        """Compute divergence metrics for one branch at one step."""
        
        # --- Distance metrics ---
        diff = cf_decoded - actual_decoded
        euclidean = float(np.linalg.norm(diff))
        
        # Cosine distance
        actual_norm = np.linalg.norm(actual_decoded)
        cf_norm = np.linalg.norm(cf_decoded)
        if actual_norm > 1e-12 and cf_norm > 1e-12:
            cosine_sim = float(np.dot(actual_decoded, cf_decoded) / (actual_norm * cf_norm))
            cosine_dist = 1.0 - cosine_sim
        else:
            cosine_dist = 1.0
        
        # Mahalanobis distance
        if self._baseline_cov is not None:
            try:
                inv_cov = np.linalg.inv(self._baseline_cov)
                mahal = float(np.sqrt(diff @ inv_cov @ diff))
            except np.linalg.LinAlgError:
                mahal = euclidean
        else:
            mahal = euclidean
        
        # Residual and rate ratios
        residual_ratio = cf_residual / max(actual_residual, 1e-12)
        rate_ratio = cf_rate / max(actual_rate, 1e-12)
        
        # --- EMA tracking ---
        ema_vals = {}
        for window in self.cfg.window_sizes:
            alpha = 2.0 / (window + 1)
            if not self._ema_initialized[branch_name][window]:
                self._ema_state[branch_name][window] = euclidean
                self._ema_initialized[branch_name][window] = True
            else:
                self._ema_state[branch_name][window] = (
                    alpha * euclidean + (1 - alpha) * self._ema_state[branch_name][window])
            ema_vals[window] = self._ema_state[branch_name][window]
        
        # --- Recovery tracking ---
        is_diverged = self._divergence_detected[branch_name]
        is_recovering = self._recovery_started[branch_name]
        is_recovered = self._recovery_confirmed[branch_name]
        steps_since_div = 0
        steps_since_rec = 0
        
        if self._baseline_established and self._baseline_euclidean > 0:
            threshold = self.cfg.recovery_threshold * self._baseline_euclidean
            
            # Detect divergence
            if not is_diverged and euclidean > threshold:
                self._divergence_detected[branch_name] = True
                self._divergence_step[branch_name] = step
                is_diverged = True
            
            # Track recovery
            if is_diverged and not is_recovered:
                steps_since_div = step - (self._divergence_step[branch_name] or step)
                
                # Recovery: divergence falls below threshold for min_recovery_steps
                if euclidean < threshold:
                    if not is_recovering:
                        self._recovery_started[branch_name] = True
                        is_recovering = True
                    steps_since_rec = step - self._find_recovery_start(branch_name)
                    
                    if steps_since_rec >= self.cfg.min_recovery_steps:
                        self._recovery_confirmed[branch_name] = True
                        self._recovery_step[branch_name] = step
                        is_recovered = True
                else:
                    # Divergence re-occurred, reset recovery
                    self._recovery_started[branch_name] = False
                    is_recovering = False
                    steps_since_rec = 0
        
        return DivergenceMetrics(
            step=step,
            euclidean_dist=euclidean,
            mahalanobis_dist=mahal,
            cosine_dist=cosine_dist,
            residual_ratio=residual_ratio,
            rate_ratio=rate_ratio,
            euclidean_ema_10=ema_vals.get(10, 0),
            euclidean_ema_50=ema_vals.get(50, 0),
            euclidean_ema_200=ema_vals.get(200, 0),
            euclidean_ema_500=ema_vals.get(500, 0),
            is_diverged=is_diverged,
            is_recovering=is_recovering,
            is_recovered=is_recovered,
            steps_since_divergence=steps_since_div,
            steps_since_recovery_start=steps_since_rec,
        )
    
    def _find_recovery_start(self, branch_name: str) -> int:
        """Find the step where recovery started (first step below threshold after divergence)."""
        div_step = self._divergence_step[branch_name]
        if div_step is None:
            return 0
        metrics = self._branch_metrics[branch_name]
        for m in metrics:
            if m.step > div_step and m.euclidean_dist < self.cfg.recovery_threshold * self._baseline_euclidean:
                return m.step
        return div_step
    
    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------
    def generate_report(self, 
                       branch_name: str,
                       scenario_name: str,
                       snapshot_step: int) -> ScenarioReport | None:
        """Generate a complete scenario comparison report."""
        if branch_name not in self._branch_metrics:
            return None
            
        metrics = self._branch_metrics[branch_name]
        if not metrics:
            return None
        
        steps = [m.step for m in metrics]
        euclidean = [m.euclidean_dist for m in metrics]
        residual_ratios = [m.residual_ratio for m in metrics]
        rate_ratios = [m.rate_ratio for m in metrics]
        
        # Divergence stats
        max_eucl = max(euclidean)
        mean_eucl = float(np.mean(euclidean))
        final_eucl = euclidean[-1]
        
        # Recovery analysis
        diverged = self._divergence_detected[branch_name]
        div_step = self._divergence_step[branch_name]
        rec_step = self._recovery_step[branch_name]
        rec_time = None
        resilience = 1.0
        
        if diverged and div_step is not None:
            if rec_step is not None:
                rec_time = rec_step - div_step
                # Resilience: how close final divergence is to baseline
                if self._baseline_euclidean > 0:
                    resilience = max(0.0, 1.0 - final_eucl / (self.cfg.recovery_threshold * self._baseline_euclidean))
            else:
                # Never recovered
                resilience = 0.0
        
        # Residual analysis
        max_res_ratio = max(residual_ratios)
        mean_res_ratio = float(np.mean(residual_ratios))
        residual_explosion = max_res_ratio > 100.0  # arbitrary threshold
        
        # Rate analysis
        max_rate_ratio = max(rate_ratios)
        mean_rate_ratio = float(np.mean(rate_ratios))
        rate_runaway = max_rate_ratio > 10.0
        
        return ScenarioReport(
            scenario_name=scenario_name,
            branch_name=branch_name,
            snapshot_step=snapshot_step,
            start_step=steps[0],
            end_step=steps[-1],
            total_steps=len(steps),
            max_euclidean_divergence=max_eucl,
            mean_euclidean_divergence=mean_eucl,
            final_euclidean_divergence=final_eucl,
            diverged=diverged,
            divergence_step=div_step,
            recovery_step=rec_step,
            recovery_time=rec_time,
            resilience_index=resilience,
            max_residual_ratio=max_res_ratio,
            mean_residual_ratio=mean_res_ratio,
            residual_explosion=residual_explosion,
            max_rate_ratio=max_rate_ratio,
            mean_rate_ratio=mean_rate_ratio,
            rate_runaway=rate_runaway,
            steps=steps,
            euclidean_divergence=euclidean,
            residual_ratios=residual_ratios,
            rate_ratios=rate_ratios,
            config={
                "recovery_threshold": self.cfg.recovery_threshold,
                "min_recovery_steps": self.cfg.min_recovery_steps,
                "divergence_metric": self.cfg.divergence_metric,
                "baseline_euclidean": self._baseline_euclidean,
            }
        )
    
    def generate_comparison_report(self,
                                   scenario_name: str,
                                   snapshot_step: int) -> dict:
        """Generate comparison across all branches."""
        branches = list(self._branch_metrics.keys())
        if not branches:
            return {"error": "no branches tracked"}
        
        reports = {}
        for branch in branches:
            reports[branch] = self.generate_report(branch, scenario_name, snapshot_step)
        
        # Summary
        summary = {
            "scenario": scenario_name,
            "snapshot_step": snapshot_step,
            "num_branches": len(branches),
            "branches": {},
            "most_divergent": None,
            "most_resilient": None,
            "any_recovered": False,
            "any_residual_explosion": False,
            "any_rate_runaway": False,
        }
        
        max_div = -1
        max_resil = -1
        
        for branch, report in reports.items():
            if report is None:
                continue
            summary["branches"][branch] = {
                "max_divergence": report.max_euclidean_divergence,
                "mean_divergence": report.mean_euclidean_divergence,
                "diverged": report.diverged,
                "recovered": report.is_recovered if hasattr(report, 'is_recovered') else (report.recovery_step is not None),
                "recovery_time": report.recovery_time,
                "resilience_index": report.resilience_index,
                "residual_explosion": report.residual_explosion,
                "rate_runaway": report.rate_runaway,
            }
            
            if report.max_euclidean_divergence > max_div:
                max_div = report.max_euclidean_divergence
                summary["most_divergent"] = branch
            
            if report.resilience_index > max_resil:
                max_resil = report.resilience_index
                summary["most_resilient"] = branch
            
            if report.recovery_step is not None:
                summary["any_recovered"] = True
            if report.residual_explosion:
                summary["any_residual_explosion"] = True
            if report.rate_runaway:
                summary["any_rate_runaway"] = True
        
        return summary
    
    # ------------------------------------------------------------------
    # Export / serialization
    # ------------------------------------------------------------------
    def export_metrics_json(self, branch_name: str, filepath: str) -> None:
        """Export branch metrics to JSON for plotting."""
        if branch_name not in self._branch_metrics:
            return
            
        metrics = self._branch_metrics[branch_name]
        data = {
            "branch": branch_name,
            "metrics": [
                {
                    "step": m.step,
                    "euclidean": m.euclidean_dist,
                    "mahalanobis": m.mahalanobis_dist,
                    "cosine": m.cosine_dist,
                    "residual_ratio": m.residual_ratio,
                    "rate_ratio": m.rate_ratio,
                    "ema_10": m.euclidean_ema_10,
                    "ema_50": m.euclidean_ema_50,
                    "ema_200": m.euclidean_ema_200,
                    "ema_500": m.euclidean_ema_500,
                    "diverged": m.is_diverged,
                    "recovering": m.is_recovering,
                    "recovered": m.is_recovered,
                }
                for m in metrics
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_latest_metrics(self, branch_name: str) -> DivergenceMetrics | None:
        """Get most recent metrics for a branch."""
        if branch_name not in self._branch_metrics:
            return None
        metrics = self._branch_metrics[branch_name]
        return metrics[-1] if metrics else None
    
    def state_dict(self) -> dict:
        """Serialize observer state."""
        return {
            "feature_dim": self.feature_dim,
            "cfg": {
                "window_sizes": self.cfg.window_sizes,
                "recovery_threshold": self.cfg.recovery_threshold,
                "min_recovery_steps": self.cfg.min_recovery_steps,
                "divergence_metric": self.cfg.divergence_metric,
                "baseline_steps": self.cfg.baseline_steps,
            },
            "baseline_established": self._baseline_established,
            "baseline_euclidean": self._baseline_euclidean,
            "baseline_residual": self._baseline_residual,
            "baseline_rate": self._baseline_rate,
            "baseline_cov": self._baseline_cov.tolist() if self._baseline_cov is not None else None,
            "branch_metrics": {
                name: [
                    {
                        "step": m.step,
                        "euclidean_dist": m.euclidean_dist,
                        "mahalanobis_dist": m.mahalanobis_dist,
                        "cosine_dist": m.cosine_dist,
                        "residual_ratio": m.residual_ratio,
                        "rate_ratio": m.rate_ratio,
                        "euclidean_ema_10": m.euclidean_ema_10,
                        "euclidean_ema_50": m.euclidean_ema_50,
                        "euclidean_ema_200": m.euclidean_ema_200,
                        "euclidean_ema_500": m.euclidean_ema_500,
                        "is_diverged": m.is_diverged,
                        "is_recovering": m.is_recovering,
                        "is_recovered": m.is_recovered,
                        "steps_since_divergence": m.steps_since_divergence,
                        "steps_since_recovery_start": m.steps_since_recovery_start,
                    }
                    for m in metrics
                ]
                for name, metrics in self._branch_metrics.items()
            },
            "divergence_detected": self._divergence_detected,
            "recovery_started": self._recovery_started,
            "recovery_confirmed": self._recovery_confirmed,
            "divergence_step": self._divergence_step,
            "recovery_step": self._recovery_step,
        }
    
    def load_state_dict(self, d: dict) -> None:
        """Restore observer state."""
        self.feature_dim = d["feature_dim"]
        cfg = d["cfg"]
        self.cfg = DivergenceConfig(
            window_sizes=tuple(cfg["window_sizes"]),
            recovery_threshold=cfg["recovery_threshold"],
            min_recovery_steps=cfg["min_recovery_steps"],
            divergence_metric=cfg["divergence_metric"],
            baseline_steps=cfg["baseline_steps"],
        )
        
        self._baseline_established = d["baseline_established"]
        self._baseline_euclidean = d["baseline_euclidean"]
        self._baseline_residual = d["baseline_residual"]
        self._baseline_rate = d["baseline_rate"]
        self._baseline_cov = np.asarray(d["baseline_cov"]) if d["baseline_cov"] else None
        
        # Restore metrics
        self._branch_metrics = {}
        for name, metrics_list in d["branch_metrics"].items():
            self._branch_metrics[name] = [
                DivergenceMetrics(**m) for m in metrics_list
            ]
        
        self._divergence_detected = d["divergence_detected"]
        self._recovery_started = d["recovery_started"]
        self._recovery_confirmed = d["recovery_confirmed"]
        self._divergence_step = d["divergence_step"]
        self._recovery_step = d["recovery_step"]
        
        # Rebuild EMA state
        self._ema_state = {}
        self._ema_initialized = {}
        for name in self._branch_metrics:
            self._ema_state[name] = {w: 0.0 for w in self.cfg.window_sizes}
            self._ema_initialized[name] = {w: False for w in self.cfg.window_sizes}
            if self._branch_metrics[name]:
                last = self._branch_metrics[name][-1]
                self._ema_state[name][10] = last.euclidean_ema_10
                self._ema_state[name][50] = last.euclidean_ema_50
                self._ema_state[name][200] = last.euclidean_ema_200
                self._ema_state[name][500] = last.euclidean_ema_500
                self._ema_initialized[name] = {w: True for w in self.cfg.window_sizes}


def make_divergence_observer(feature_dim: int,
                            recovery_threshold: float = 1.5,
                            min_recovery_steps: int = 10) -> DivergenceObserver:
    """Factory for backward-compatible creation."""
    config = DivergenceConfig(
        recovery_threshold=recovery_threshold,
        min_recovery_steps=min_recovery_steps,
    )
    return DivergenceObserver(feature_dim, config)