"""Experiment 003 - Cross-Domain Dynamic Feature Transfer (T4.2).

Tests zero-shot transfer of neural dynamics trained on one domain (Lorenz chaos)
to a different domain (system metrics: CPU/RAM/Disk/Net).

Hypothesis: The dynamic fingerprints learned on chaotic mathematical systems
exhibit universal properties that transfer to heterogeneous real-world signals
without retraining.

Arms:
1. source_only: Train on Lorenz, test on Lorenz (baseline)
2. zero_shot_system: Train on Lorenz, test on System (no retraining)
3. finetune_system: Train on Lorenz, fine-tune on System (upper bound)
4. system_only: Train on System, test on System (lower bound for reference)

Metrics: MAE, RMSE, anomaly detection Precision/Recall, cross-domain similarity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from adapters.chaos import ChaosInjector
from adapters.lorenz import LorenzSource
from adapters.system import SystemSource
from core.dynamics.multiscale import (
    MultiScaleDynamics, ScaleConfig, MultiScaleConfig, create_default_multiscale
)
from core.encoding.encoder import FeatureEncoder
from core.encoding.projector import SubspaceProjector
from core.ingest.ring_buffer import TimeAligner
from core.memory.buffer import RollingTemporalBuffer, TemporalConfig, make_temporal_buffer
from core.memory.associative import AssociativeMemory, RetrievalConfig, make_associative_memory
from core.observer.observer import CircuitBreaker
from core.prediction.predictor import NLMSPredictor

DT = 0.005
SRC_DT = 0.001
LOOKAHEAD = 0.002
CHUNK = 200
NEURONS_FAST = 300
NEURONS_MED = 200
NEURONS_SLOW = 100
PROJ_DIM = 16
FEATURE_DIM = 32  # 2 * PROJ_DIM


@dataclass
class ExperimentConfig:
    """Configuration for the cross-domain transfer experiment."""
    train_steps: int = 1000
    test_steps: int = 500
    n_neurons_per_scale: tuple = (NEURONS_FAST, NEURONS_MED, NEURONS_SLOW)
    feature_dim: int = FEATURE_DIM
    proj_dim: int = PROJ_DIM
    seed: int = 42


class DomainAdapter:
    """Adapts a trained multi-scale dynamics + predictor to a new domain."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.feature_dim = config.feature_dim
        self.proj_dim = config.proj_dim
        
        # Multi-scale dynamics (shared across domains)
        self.dynamics = create_default_multiscale(
            feature_dim=config.feature_dim,
            n_scales=3,
            tau_base=0.005,
            tau_factor=5.0,
            neurons_per_scale=config.n_neurons_per_scale,
            seed=config.seed
        )
        
        # Encoders: Lorenz state_dim=3 -> proj_dim=8 -> features=16
        # System state_dim=4 -> proj_dim=8 -> features=16
        self.lorenz_encoder = FeatureEncoder(state_dim=3, proj_dim=8, seed=config.seed)
        self.system_encoder = FeatureEncoder(state_dim=4, proj_dim=8, seed=config.seed + 1)
        
        # Projector: maps (lorenz: 3, system: 4) -> proj_dim=16 -> features=32
        # Uses the z-scored state dimensions (3 and 4), not proj_dim (8)
        self.projector = SubspaceProjector(
            {"lorenz": 3, "system": 4},
            proj_dim=config.proj_dim,
            mode="orthogonal",
            seed=config.seed
        )
        
        # Predictor: input = 64 features (2 * feature_dim), output = combined z (3+4=7)
        self.predictor = NLMSPredictor(
            n_in=2 * config.feature_dim,
            n_out=7,  # 3 (lorenz) + 4 (system)
            mu=0.05
        )
        
        # Memory buffers
        from core.memory.buffer import make_temporal_buffer
        from core.memory.associative import make_associative_memory
        self.temporal_buffer = make_temporal_buffer(config.feature_dim, capacity=2048)
        self.associative_memory = make_associative_memory(config.feature_dim, max_history=4096)
        
        # Training state
        self.is_trained = False
        self.train_source = None
        self._lag = None
        self._step = 0
        self._decoded_history = []
        self._z_history = {}
        
    def train_on_lorenz(self, steps: int) -> dict:
        """Train the full pipeline on Lorenz chaos."""
        print(f"[exp003] Training on Lorenz for {steps} steps...")
        
        lorenz = ChaosInjector(LorenzSource(dt=SRC_DT), seed=1,
                               noise_sigma=0.3, noise_prob=0.05,
                               drop_prob=0.01, nan_prob=0.005,
                               spike_prob=0.005, spike_scale=1e4)
        system = SystemSource()  # Dummy system source
        
        al_l = TimeAligner(dt=DT, dim=3)
        al_s = TimeAligner(dt=DT, dim=4)
        
        warm_feats = []
        t_wall = time.perf_counter()
        
        for S in range(0, steps, CHUNK):
            n = min(CHUNK, steps - S)
            
            zs_l_list, feats_list = [], []
            for k in range(n):
                t = (S + k) * DT
                lorenz.advance_to(t + LOOKAHEAD)
                for ts, vec in lorenz.read_pending():
                    if vec is not None:
                        al_l.push(ts, vec)
                raw_l, conf_l, _ = al_l.at(t)
                self.lorenz_encoder.observe(raw_l, weight=conf_l)
                z_l = self.lorenz_encoder.encode_z(raw_l)
                zs_l_list.append(z_l)
                
                # Dummy system (zeros)
                al_s.push(t, np.zeros(4))
                raw_s, conf_s, _ = al_s.at(t)
                self.system_encoder.observe(raw_s, weight=conf_s)
                z_s = self.system_encoder.encode_z(raw_s)
                
                # Project
                feat = self.projector.features({"lorenz": z_l, "system": z_s})
                feats_list.append(feat)
            
            feats = np.asarray(feats_list)
            cres = self.dynamics.run_chunk_aggregated(feats)
            decoded = cres["aggregated"]
            
            if self._lag is None:
                self._lag = self.dynamics.estimate_lag(feats, decoded)
            
            # Causal prediction
            zero_rep = np.zeros(self.feature_dim)
            for k in range(n):
                s = S + k
                idx = k - (S - self._step) + self._lag if S >= self._step else k + self._lag
                if idx < 0 or idx >= len(self._decoded_history):
                    continue
                dec = self._decoded_history[idx]
                dec_prev = self._decoded_history[idx - 1] if idx - 1 >= 0 else zero_rep
                x = np.concatenate([dec, dec_prev])
                target = np.concatenate([zs_l_list[k], np.zeros(4)])
                conf = 0.9
                pred_vec = self.predictor.predict(x)
                self.predictor.update(x, target, weight=conf)
                self._z_history[s] = target
            
            self._decoded_history.extend(decoded)
            self._step += n
            warm_feats.extend(feats)
            
            if S % 500 == 0:
                print(f"  Step {S}/{steps}")
        
        self.is_trained = True
        self.train_source = "lorenz"
        
        wall = time.perf_counter() - t_wall
        print(f"[exp003] Training complete: {wall:.1f}s, lag={self._lag}")
        
        return {"wall_s": wall, "lag": self._lag, "steps": steps}
    
    def test_on_system(self, steps: int, finetune: bool = False) -> dict:
        """Test (or finetune) on system metrics."""
        mode = "finetune" if finetune else "test"
        print(f"[exp003] {mode.capitalize()} on System for {steps} steps...")
        
        lorenz = ChaosInjector(LorenzSource(dt=SRC_DT), seed=2,
                               noise_sigma=0.3, noise_prob=0.05,
                               drop_prob=0.01, nan_prob=0.005,
                               spike_prob=0.005, spike_scale=1e4)
        system = SystemSource()
        
        al_l = TimeAligner(dt=DT, dim=3)
        al_s = TimeAligner(dt=DT, dim=4)
        
        residuals_lorenz = []
        residuals_system = []
        random_lorenz = []
        random_system = []
        rng = np.random.default_rng(1234)
        predictions = []
        targets = []
        
        t_wall = time.perf_counter()
        
        for S in range(0, steps, CHUNK):
            n = min(CHUNK, steps - S)
            
            # Sample system once per chunk (psutil is slow)
            system.t = (S + n / 2) * DT
            sys_vec = system.sample()
            
            zs_l_list, zs_s_list, feats_list = [], [], []
            for k in range(n):
                t = (S + k) * DT
                
                # Lorenz
                lorenz.advance_to(t + LOOKAHEAD)
                for ts, vec in lorenz.read_pending():
                    if vec is not None:
                        al_l.push(ts, vec)
                raw_l, conf_l, _ = al_l.at(t)
                self.lorenz_encoder.observe(raw_l, weight=conf_l)
                z_l = self.lorenz_encoder.encode_z(raw_l)
                
                # System
                al_s.push(t, sys_vec)
                raw_s, conf_s, _ = al_s.at(t)
                self.system_encoder.observe(raw_s, weight=conf_s)
                z_s = self.system_encoder.encode_z(raw_s)
                
                zs_l_list.append(z_l)
                zs_s_list.append(z_s)
                feat = self.projector.features({"lorenz": z_l, "system": z_s})
                feats_list.append(feat)
            
            feats = np.asarray(feats_list)
            cres = self.dynamics.run_chunk_aggregated(feats)
            decoded = cres["aggregated"]
            
            zero_rep = np.zeros(self.feature_dim)
            for k in range(n):
                s = S + k
                idx = k - (S - self._step) + self._lag if S >= self._step else k + self._lag
                if idx < 0 or idx >= len(self._decoded_history):
                    continue
                dec = self._decoded_history[idx]
                dec_prev = self._decoded_history[idx - 1] if idx - 1 >= 0 else zero_rep
                x = np.concatenate([dec, dec_prev])
                target = np.concatenate([zs_l_list[k], zs_s_list[k]])
                
                pred_vec = self.predictor.predict(x)
                if finetune:
                    self.predictor.update(x, target, weight=min(conf_l, conf_s))
                
                e_l = float(np.sum((target[:3] - pred_vec[:3]) ** 2))
                e_s = float(np.sum((target[3:] - pred_vec[3:]) ** 2))
                # Zero-information control: guess from N(0, 1) in z-space.
                guess_l = rng.normal(0.0, 1.0, size=3)
                guess_s = rng.normal(0.0, 1.0, size=4)
                random_lorenz.append(float(np.sum((target[:3] - guess_l) ** 2)))
                random_system.append(float(np.sum((target[3:] - guess_s) ** 2)))
                residuals_lorenz.append(e_l)
                residuals_system.append(e_s)
                predictions.append(pred_vec.copy())
                targets.append(target.copy())
            
            self._decoded_history.extend(decoded)
            self._step += n
            
            if S % 200 == 0:
                print(f"  Step {S}/{steps}")
        
        wall = time.perf_counter() - t_wall
        
        return {
            "wall_s": wall,
            "steps": steps,
            "residuals_lorenz": residuals_lorenz,
            "residuals_system": residuals_system,
            "mean_e_lorenz": float(np.mean(residuals_lorenz)),
            "mean_e_system": float(np.mean(residuals_system)),
            "mae_lorenz": float(np.mean(np.sqrt(residuals_lorenz))),
            "mae_system": float(np.mean(np.sqrt(residuals_system))),
            "rmse_lorenz": float(np.sqrt(np.mean(residuals_lorenz))),
            "rmse_system": float(np.sqrt(np.mean(residuals_system))),
            "random_mae_lorenz": float(np.mean(np.sqrt(random_lorenz))),
            "random_mae_system": float(np.mean(np.sqrt(random_system))),
            "random_rmse_system": float(np.sqrt(np.mean(random_system))),
        }
    
    def train_on_system(self, steps: int) -> dict:
        """Train from scratch on system metrics (lower bound)."""
        print(f"[exp003] Training on System from scratch for {steps} steps...")
        
        # Reset dynamics and predictor
        self.dynamics.reset()
        self.predictor = NLMSPredictor(
            n_in=2 * self.feature_dim,
            n_out=7,  # 3 (lorenz) + 4 (system)
            mu=0.05
        )
        self._decoded_history = []
        self._z_history = {}
        self._step = 0
        self._lag = None
        
        lorenz = ChaosInjector(LorenzSource(dt=SRC_DT), seed=3,
                               noise_sigma=0.3, noise_prob=0.05,
                               drop_prob=0.01, nan_prob=0.005,
                               spike_prob=0.005, spike_scale=1e4)
        system = SystemSource()
        
        al_l = TimeAligner(dt=DT, dim=3)
        al_s = TimeAligner(dt=DT, dim=4)
        
        warm_feats = []
        t_wall = time.perf_counter()
        
        for S in range(0, steps, CHUNK):
            n = min(CHUNK, steps - S)
            
            system.t = (S + n / 2) * DT
            sys_vec = system.sample()
            
            zs_l_list, zs_s_list, feats_list = [], [], []
            for k in range(n):
                t = (S + k) * DT
                
                lorenz.advance_to(t + LOOKAHEAD)
                for ts, vec in lorenz.read_pending():
                    if vec is not None:
                        al_l.push(ts, vec)
                raw_l, conf_l, _ = al_l.at(t)
                self.lorenz_encoder.observe(raw_l, weight=conf_l)
                z_l = self.lorenz_encoder.encode_z(raw_l)
                
                al_s.push(t, sys_vec)
                raw_s, conf_s, _ = al_s.at(t)
                self.system_encoder.observe(raw_s, weight=conf_s)
                z_s = self.system_encoder.encode_z(raw_s)
                
                zs_l_list.append(z_l)
                zs_s_list.append(z_s)
                feat = self.projector.features({"lorenz": z_l, "system": z_s})
                feats_list.append(feat)
            
            feats = np.asarray(feats_list)
            cres = self.dynamics.run_chunk_aggregated(feats)
            decoded = cres["aggregated"]
            
            if self._lag is None:
                self._lag = self.dynamics.estimate_lag(feats, decoded)
            
            zero_rep = np.zeros(self.feature_dim)
            for k in range(n):
                s = S + k
                idx = k - (S - self._step) + self._lag if S >= self._step else k + self._lag
                if idx < 0 or idx >= len(self._decoded_history):
                    continue
                dec = self._decoded_history[idx]
                dec_prev = self._decoded_history[idx - 1] if idx - 1 >= 0 else zero_rep
                x = np.concatenate([dec, dec_prev])
                target = np.concatenate([zs_l_list[k], zs_s_list[k]])
                conf = 0.9
                pred_vec = self.predictor.predict(x)
                self.predictor.update(x, target, weight=conf)
            
            self._decoded_history.extend(decoded)
            self._step += n
            warm_feats.extend(feats)
            
            if S % 500 == 0:
                print(f"  Step {S}/{steps}")
        
        self.is_trained = True
        self.train_source = "system"
        
        wall = time.perf_counter() - t_wall
        print(f"[exp003] System training complete: {wall:.1f}s")
        
        return {"wall_s": wall, "lag": self._lag, "steps": steps}


def compute_dynamic_fingerprint(dynamics: MultiScaleDynamics, 
                                 features: np.ndarray,
                                 n_samples: int = 100) -> dict:
    """Compute dynamic fingerprint of the neural dynamics response."""
    # Run a sample and analyze the multi-scale responses
    results = dynamics.run_chunk_aggregated(features[:n_samples])
    
    fingerprint = {}
    for name in dynamics.scale_names:
        decoded = results["per_scale"][name]["decoded"]
        rates = results["per_scale"][name]["rate_mean"]
        
        # Compute spectral features
        fft_mag = np.abs(np.fft.rfft(decoded, axis=0))
        spectral_centroid = np.sum(fft_mag * np.arange(fft_mag.shape[0])[:, None], axis=0) / (np.sum(fft_mag, axis=0) + 1e-9)
        
        fingerprint[name] = {
            "mean_rate": float(np.mean(rates)),
            "rate_std": float(np.std(rates)),
            "decoded_std": float(np.std(decoded)),
            "spectral_centroid_mean": float(np.mean(spectral_centroid)),
            "spectral_centroid_std": float(np.std(spectral_centroid)),
        }
    
    return fingerprint


def compute_cross_domain_similarity(fp1: dict, fp2: dict) -> dict:
    """Compute similarity between two dynamic fingerprints."""
    similarities = {}
    for name in fp1:
        if name in fp2:
            v1 = np.array([fp1[name][k] for k in ["mean_rate", "rate_std", "decoded_std", 
                                                    "spectral_centroid_mean", "spectral_centroid_std"]])
            v2 = np.array([fp2[name][k] for k in ["mean_rate", "rate_std", "decoded_std",
                                                    "spectral_centroid_mean", "spectral_centroid_std"]])
            # Cosine similarity
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            # Euclidean distance
            euclid = np.linalg.norm(v1 - v2)
            similarities[name] = {
                "cosine_similarity": float(cos_sim),
                "euclidean_distance": float(euclid),
            }
    return similarities


def run_experiment_003(config: ExperimentConfig, out_dir: str) -> dict:
    """Run the full cross-domain transfer experiment."""
    
    results = {
        "config": asdict(config),
        "arms": {},
        "fingerprints": {},
        "similarities": {},
    }
    
    # Arm 1: Train on Lorenz, test on Lorenz (baseline)
    print("\n" + "="*60)
    print("ARM 1: Train Lorenz -> Test Lorenz (baseline)")
    print("="*60)
    
    adapter1 = DomainAdapter(config)
    train_meta = adapter1.train_on_lorenz(config.train_steps)
    
    # Test on Lorenz (generate Lorenz test data)
    # For simplicity, reuse the same adapter but with fresh Lorenz source
    lorenz_test = ChaosInjector(LorenzSource(dt=SRC_DT), seed=10,
                                noise_sigma=0.3, noise_prob=0.05,
                                drop_prob=0.01, nan_prob=0.005,
                                spike_prob=0.005, spike_scale=1e4)
    
    # Quick Lorenz test
    test_meta = test_on_lorenz(adapter1, config.test_steps)
    results["arms"]["lorenz_to_lorenz"] = {**train_meta, **test_meta}
    
    # Capture fingerprint
    sample_features = generate_lorenz_features(adapter1, config.test_steps // 2)
    results["fingerprints"]["lorenz_trained"] = compute_dynamic_fingerprint(
        adapter1.dynamics, sample_features)
    
    # Arm 2: Train on Lorenz, zero-shot test on System
    print("\n" + "="*60)
    print("ARM 2: Train Lorenz -> Zero-shot System")
    print("="*60)
    
    adapter2 = DomainAdapter(config)
    train_meta2 = adapter2.train_on_lorenz(config.train_steps)
    test_meta2 = adapter2.test_on_system(config.test_steps, finetune=False)
    results["arms"]["lorenz_to_system_zeroshot"] = {**train_meta2, **test_meta2}
    
    # Capture fingerprint
    sample_features2 = generate_system_features(adapter2, config.test_steps // 2)
    results["fingerprints"]["lorenz_trained_on_system"] = compute_dynamic_fingerprint(
        adapter2.dynamics, sample_features2)
    
    # Arm 3: Train on Lorenz, finetune on System
    print("\n" + "="*60)
    print("ARM 3: Train Lorenz -> Finetune System")
    print("="*60)
    
    adapter3 = DomainAdapter(config)
    train_meta3 = adapter3.train_on_lorenz(config.train_steps)
    test_meta3 = adapter3.test_on_system(config.test_steps, finetune=True)
    results["arms"]["lorenz_to_system_finetune"] = {**train_meta3, **test_meta3}
    
    # Arm 4: Train on System from scratch
    print("\n" + "="*60)
    print("ARM 4: Train System -> Test System (lower bound)")
    print("="*60)
    
    adapter4 = DomainAdapter(config)
    train_meta4 = adapter4.train_on_system(config.train_steps)
    test_meta4 = adapter4.test_on_system(config.test_steps, finetune=False)
    results["arms"]["system_to_system"] = {**train_meta4, **test_meta4}
    
    results["fingerprints"]["system_trained"] = compute_dynamic_fingerprint(
        adapter4.dynamics, generate_system_features(adapter4, config.test_steps // 2))
    
    # Compute cross-domain similarities
    print("\n" + "="*60)
    print("COMPUTING CROSS-DOMAIN SIMILARITIES")
    print("="*60)
    
    # Lorenz-trained dynamics on Lorenz vs System
    results["similarities"]["lorenz_dynamics_lorenz_vs_system"] = compute_cross_domain_similarity(
        results["fingerprints"]["lorenz_trained"],
        results["fingerprints"]["lorenz_trained_on_system"]
    )
    
    # Lorenz-trained vs System-trained on System
    results["similarities"]["lorenz_vs_system_trained_on_system"] = compute_cross_domain_similarity(
        results["fingerprints"]["lorenz_trained_on_system"],
        results["fingerprints"]["system_trained"]
    )
    
    # Summary metrics
    results["summary"] = {
        "zeroshot_transfer_ratio_lorenz": 
            results["arms"]["lorenz_to_system_zeroshot"]["mae_lorenz"] /
            max(results["arms"]["lorenz_to_lorenz"]["mae_lorenz"], 1e-9),
        "zeroshot_transfer_ratio_system":
            results["arms"]["lorenz_to_system_zeroshot"]["mae_system"] /
            max(results["arms"]["system_to_system"]["mae_system"], 1e-9),
        "finetune_improvement_lorenz":
            results["arms"]["lorenz_to_system_finetune"]["mae_lorenz"] /
            max(results["arms"]["lorenz_to_system_zeroshot"]["mae_lorenz"], 1e-9),
        "finetune_improvement_system":
            results["arms"]["lorenz_to_system_finetune"]["mae_system"] /
            max(results["arms"]["lorenz_to_system_zeroshot"]["mae_system"], 1e-9),
    }

    # Acceptance: zero-shot transfer must beat a zero-information guess.
    zs = results["arms"]["lorenz_to_system_zeroshot"]
    in_domain = results["arms"]["system_to_system"]
    results["verdict"] = {
        "zeroshot_system_mae": zs["mae_system"],
        "random_guess_system_mae": zs.get("random_mae_system"),
        "beats_random_system": bool(
            zs["mae_system"] < zs.get("random_mae_system", float("inf"))),
        "zeroshot_lorenz_mae": zs["mae_lorenz"],
        "random_guess_lorenz_mae": zs.get("random_mae_lorenz"),
        "beats_random_lorenz": bool(
            zs["mae_lorenz"] < zs.get("random_mae_lorenz", float("inf"))),
        "in_domain_system_mae": in_domain["mae_system"],
        "zeroshot_vs_in_domain_ratio": zs["mae_system"] /
            max(in_domain["mae_system"], 1e-9),
    }
    
    # Save results
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save residuals CSV
    with open(os.path.join(out_dir, "residuals.csv"), "w") as f:
        f.write("arm,step,e_lorenz,e_system\n")
        for arm_name, arm_data in results["arms"].items():
            for i, (e_l, e_s) in enumerate(zip(
                arm_data.get("residuals_lorenz", []),
                arm_data.get("residuals_system", []))):
                f.write(f"{arm_name},{i},{e_l:.6g},{e_s:.6g}\n")
    
    print("\n" + "="*60)
    print("EXPERIMENT 003 COMPLETE")
    print("="*60)
    print(json.dumps(results["summary"], indent=2))
    print(json.dumps(results["verdict"], indent=2))
    print(f"\nSaved to {out_dir}")
    
    return results


def test_on_lorenz(adapter: DomainAdapter, steps: int) -> dict:
    """Quick test on Lorenz."""
    lorenz = ChaosInjector(LorenzSource(dt=SRC_DT), seed=10,
                           noise_sigma=0.3, noise_prob=0.05,
                           drop_prob=0.01, nan_prob=0.005,
                           spike_prob=0.005, spike_scale=1e4)
    
    al_l = TimeAligner(dt=DT, dim=3)
    al_s = TimeAligner(dt=DT, dim=4)
    
    residuals_lorenz = []
    residuals_system = []
    random_lorenz = []
    rng = np.random.default_rng(1234)
    
    for S in range(0, steps, CHUNK):
        n = min(CHUNK, steps - S)
        
        zs_l_list, feats_list = [], []
        for k in range(n):
            t = (S + k) * DT
            lorenz.advance_to(t + LOOKAHEAD)
            for ts, vec in lorenz.read_pending():
                if vec is not None:
                    al_l.push(ts, vec)
            raw_l, conf_l, _ = al_l.at(t)
            adapter.lorenz_encoder.observe(raw_l, weight=conf_l)
            z_l = adapter.lorenz_encoder.encode_z(raw_l)
            zs_l_list.append(z_l)
            
            al_s.push(t, np.zeros(4))
            raw_s, conf_s, _ = al_s.at(t)
            adapter.system_encoder.observe(raw_s, weight=conf_s)
            z_s = adapter.system_encoder.encode_z(raw_s)
            
            feat = adapter.projector.features({"lorenz": z_l, "system": z_s})
            feats_list.append(feat)
        
        feats = np.asarray(feats_list)
        cres = adapter.dynamics.run_chunk_aggregated(feats)
        decoded = cres["aggregated"]
        
        zero_rep = np.zeros(adapter.feature_dim)
        for k in range(n):
            s = S + k
            idx = k - (S - adapter._step) + adapter._lag if S >= adapter._step else k + adapter._lag
            if idx < 0 or idx >= len(adapter._decoded_history):
                continue
            dec = adapter._decoded_history[idx]
            dec_prev = adapter._decoded_history[idx - 1] if idx - 1 >= 0 else zero_rep
            x = np.concatenate([dec, dec_prev])
            target = np.concatenate([zs_l_list[k], np.zeros(4)])
            pred_vec = adapter.predictor.predict(x)
            e_l = float(np.sum((target[:3] - pred_vec[:3]) ** 2))
            e_s = float(np.sum((target[3:] - pred_vec[3:]) ** 2))
            guess_l = rng.normal(0.0, 1.0, size=3)
            random_lorenz.append(float(np.sum((target[:3] - guess_l) ** 2)))
            residuals_lorenz.append(e_l)
            residuals_system.append(e_s)
        
        adapter._decoded_history.extend(decoded)
        adapter._step += n
    
    return {
        "test_steps": steps,
        "residuals_lorenz": residuals_lorenz,
        "residuals_system": residuals_system,
        "mean_e_lorenz": float(np.mean(residuals_lorenz)),
        "mean_e_system": float(np.mean(residuals_system)),
        "mae_lorenz": float(np.mean(np.sqrt(residuals_lorenz))),
        "mae_system": float(np.mean(np.sqrt(residuals_system))),
        "rmse_lorenz": float(np.sqrt(np.mean(residuals_lorenz))),
        "rmse_system": float(np.sqrt(np.mean(residuals_system))),
        "random_mae_lorenz": float(np.mean(np.sqrt(random_lorenz))),
    }


def generate_lorenz_features(adapter: DomainAdapter, steps: int) -> np.ndarray:
    """Generate features from Lorenz for fingerprinting."""
    lorenz = ChaosInjector(LorenzSource(dt=SRC_DT), seed=100,
                           noise_sigma=0.3, noise_prob=0.05)
    al_l = TimeAligner(dt=DT, dim=3)
    al_s = TimeAligner(dt=DT, dim=4)
    
    feats_list = []
    for S in range(0, steps, CHUNK):
        n = min(CHUNK, steps - S)
        for k in range(n):
            t = (S + k) * DT
            lorenz.advance_to(t + LOOKAHEAD)
            for ts, vec in lorenz.read_pending():
                if vec is not None:
                    al_l.push(ts, vec)
            raw_l, conf_l, _ = al_l.at(t)
            z_l = adapter.lorenz_encoder.encode_z(raw_l)
            al_s.push(t, np.zeros(4))
            raw_s, conf_s, _ = al_s.at(t)
            z_s = adapter.system_encoder.encode_z(raw_s)
            feat = adapter.projector.features({"lorenz": z_l, "system": z_s})
            feats_list.append(feat)
    
    return np.asarray(feats_list)


def generate_system_features(adapter: DomainAdapter, steps: int) -> np.ndarray:
    """Generate features from System for fingerprinting."""
    lorenz = ChaosInjector(LorenzSource(dt=SRC_DT), seed=100,
                           noise_sigma=0.3, noise_prob=0.05)
    system = SystemSource()
    al_l = TimeAligner(dt=DT, dim=3)
    al_s = TimeAligner(dt=DT, dim=4)
    
    feats_list = []
    for S in range(0, steps, CHUNK):
        n = min(CHUNK, steps - S)
        system.t = (S + n / 2) * DT
        sys_vec = system.sample()
        for k in range(n):
            t = (S + k) * DT
            lorenz.advance_to(t + LOOKAHEAD)
            for ts, vec in lorenz.read_pending():
                if vec is not None:
                    al_l.push(ts, vec)
            raw_l, conf_l, _ = al_l.at(t)
            z_l = adapter.lorenz_encoder.encode_z(raw_l)
            al_s.push(t, sys_vec)
            raw_s, conf_s, _ = al_s.at(t)
            z_s = adapter.system_encoder.encode_z(raw_s)
            feat = adapter.projector.features({"lorenz": z_l, "system": z_s})
            feats_list.append(feat)
    
    return np.asarray(feats_list)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-steps", type=int, default=1000)
    ap.add_argument("--test-steps", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "experiments", "experiment_003", "results"))
    args = ap.parse_args()
    
    config = ExperimentConfig(
        train_steps=args.train_steps,
        test_steps=args.test_steps,
    )
    
    run_experiment_003(config, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())