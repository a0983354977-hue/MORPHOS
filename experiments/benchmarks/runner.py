"""Automated Benchmark Runner for Phase IV (T4.4).

Produces the Phase IV comparison matrix:

  * MORPHOS ablation study - full system vs. no multi-scale dynamics,
    no orthogonal projection, no temporal memory, and minimal.
  * Traditional baselines - Naive, EMA, AR, ARIMA, Kalman on the exact
    same normalised stream that MORPHOS predicts.
  * Anomaly detection - precision / recall / F1 for MORPHOS vs.
    Isolation Forest and ARIMA-style residual detectors, all calibrated
    to the same false-positive budget on clean data.
  * Acceptance verdicts written to JSON and Markdown.

Everything is deterministic: the multi-timescale stream is synthetic
(mixed fast chaos + slow resource evolution) so reports are reproducible.

Run:  python experiments/benchmarks/runner.py --train-steps 800 --test-steps 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from adapters.lorenz import LorenzSource
from core.dynamics.engine import NengoDynamics
from core.dynamics.multiscale import create_default_multiscale
from core.encoding.projector import SubspaceProjector
from core.ingest.ring_buffer import TimeAligner
from core.memory.associative import make_associative_memory
from core.memory.buffer import make_temporal_buffer
from core.memory.compressor import make_compressor
from core.observer.circuit_breaker import make_dampened_breaker
from core.observer.dual_opinion import make_dual_opinion
from core.prediction.predictor import NLMSPredictor
from experiments.benchmarks.baselines import (
    HAS_SKLEARN, HAS_STATSMODELS,
    ARBaseline, ARIMABaseline, IsolationForestBaseline,
    best_f1_over_thresholds, calibrate_threshold, get_all_baselines,
    precision_recall, residual_scores, run_baseline_comparison,
    threshold_labels,
)

DT = 0.005              # control step (s)
SRC_DT = 0.001          # source sample step (s)
LOOKAHEAD = 0.002       # source horizon pushed before align()
CHUNK = 50              # steps per neural simulation chunk
PROJ_DIM = 16           # projector output dim (feature_dim = 2 * proj_dim)
FEATURE_DIM = 32
MEM_CHANNELS = 4        # temporal-buffer EMA scales fed to the predictor
Z_LAGS = 5              # encoded-state taps (matches the AR/ARIMA order)
HIST_LIMIT = 8192       # bound decoded-history memory
LORENZ_DIMS = 3
SYSTEM_DIMS = 4
TARGET_DIMS = LORENZ_DIMS + SYSTEM_DIMS


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    train_steps: int = 6000
    test_steps: int = 500
    calib_steps: int = 600
    anomaly_steps: int = 900
    anomaly_quantile: float = 0.99
    seed: int = 42
    # Ablation flags
    enable_multiscale: bool = True
    enable_orthogonal_projection: bool = True
    enable_memory: bool = True
    # MORPHOS architecture
    n_neurons_per_scale: tuple = (300, 200, 100)
    feature_dim: int = FEATURE_DIM
    proj_dim: int = PROJ_DIM


# ----------------------------------------------------------------------
# Deterministic multi-timescale data
# ----------------------------------------------------------------------
class SyntheticSystemSource:
    """Resource-like signal mixing fast bursts with slow load evolution.

    Periods span 0.4 s (bursts) to 20 s (load trend), so the stream
    exercises both ends of the multi-scale time-constant hierarchy while
    staying fully reproducible (no wall-clock or psutil dependence).
    """

    dim = SYSTEM_DIMS

    def __init__(self, dt: float = DT, seed: int = 0):
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.i = 0

    def sample(self) -> np.ndarray:
        t = self.i * self.dt
        self.i += 1
        slow = 0.12 * np.sin(2 * np.pi * t / 20.0) + \
            0.06 * np.sin(2 * np.pi * t / 5.0)
        fast = 0.05 * np.sin(2 * np.pi * t / 0.4) + \
            self.rng.normal(0.0, 0.01)
        cpu = np.clip(0.45 + slow + fast, 0.0, 1.0)
        mem = np.clip(0.60 + 0.5 * slow + 0.02 * np.sin(2 * np.pi * t / 12.0),
                      0.0, 1.0)
        disk = max(0.0, 0.20 + 0.15 * np.sin(2 * np.pi * t / 3.0) +
                   0.05 * self.rng.standard_normal())
        net = max(0.0, 0.25 + 0.10 * np.sin(2 * np.pi * t / 7.0) +
                  0.05 * self.rng.standard_normal())
        return np.array([cpu, mem, disk, net], dtype=float)


def make_anomaly_schedule(steps: int, seed: int, n_runs: int = 6,
                          min_len: int = 4, max_len: int = 8,
                          min_gap: int = 100) -> List[tuple]:
    """Deterministic schedule of labelled anomaly injections.

    Cycles through three fault modes so a detector cannot win by only
    modelling one kind of outlier:
      spike - extreme magnitude excursion (sensor fault)
      shift - sustained level offset (regime change)
      var   - variance inflation around the local level
    """
    rng = np.random.default_rng(seed)
    margin = 40
    kinds = ("spike", "shift", "var")
    schedule: List[tuple] = []
    pos = margin
    # Faults are separated by min_gap clean steps so the monitored system
    # returns to baseline between events - otherwise a persistent-state
    # model is penalised for ringing after an earlier fault.
    while len(schedule) < n_runs and pos + max_len < steps - margin:
        length = int(rng.integers(min_len, max_len + 1))
        schedule.append((int(pos), length, kinds[len(schedule) % len(kinds)]))
        pos += length + int(min_gap)
    return schedule


def build_stream(config: BenchmarkConfig, steps: int, seed: int,
                 anomalies: Optional[List[tuple]] = None) -> dict:
    """Aligned fast (Lorenz) + slow (resource) stream with labels."""
    # Distinct initial condition per seed plus a burn-in: every benchmark
    # stream must be an independent trajectory, otherwise train and test
    # are the same Lorenz prefix and offline baselines memorise the test
    # set (which silently produced absurd 0.003 MAE numbers before).
    rng = np.random.default_rng(seed)
    state0 = np.array([1.0, 1.0, 1.0]) + rng.uniform(-6.0, 6.0, 3)
    lorenz = LorenzSource(dt=SRC_DT, state0=state0)
    burn_in = 30.0
    lorenz.advance_to(burn_in)   # decay onto the attractor
    lorenz.read_pending()        # discard burn-in samples

    system = SyntheticSystemSource(dt=DT, seed=seed + 991)
    aligner = TimeAligner(dt=DT, dim=LORENZ_DIMS)

    raw_l = np.zeros((steps, LORENZ_DIMS))
    raw_s = np.zeros((steps, SYSTEM_DIMS))

    for S in range(0, steps, CHUNK):
        n = min(CHUNK, steps - S)
        for k in range(n):
            # The control grid continues from the source clock after the
            # burn-in; starting again at t=0 would leave the source ahead
            # of the grid and silently starve the aligner.
            t = burn_in + (S + k) * DT
            lorenz.advance_to(t + LOOKAHEAD)
            for ts, vec in lorenz.read_pending():
                if vec is not None:
                    aligner.push(ts, vec)
            raw_l[S + k], _conf, _flags = aligner.at(t)
            raw_s[S + k] = system.sample()

    labels = np.zeros(steps, dtype=int)
    if anomalies:
        for start, length, kind in anomalies:
            if start >= steps:
                continue
            end = min(steps, start + length)
            seg = raw_l[start:end]
            if kind == "spike":
                raw_l[start:end] = seg * 8.0
            elif kind == "shift":
                raw_l[start:end] = seg + 25.0
            elif kind == "var":
                base = (raw_l[max(0, start - 50):start].mean(axis=0)
                        if start > 0 else seg.mean(axis=0))
                raw_l[start:end] = base + 3.0 * (seg - base)
            else:
                raise ValueError(f"unknown anomaly kind: {kind}")
            labels[start:end] = 1

    return {"lorenz": raw_l, "system": raw_s, "labels": labels}


def mixed_series(config: BenchmarkConfig, steps: int, seed: int,
                 anomalies: Optional[List[tuple]] = None):
    """Return (7-D raw series, per-step anomaly labels)."""
    s = build_stream(config, steps, seed, anomalies=anomalies)
    return (np.concatenate([s["lorenz"], s["system"]], axis=1), s["labels"])


def normalise(train_raw: np.ndarray, *others: np.ndarray):
    """Z-score with train statistics, clipped to the encoder's [-1, 1]."""
    mu = train_raw.mean(axis=0)
    sd = train_raw.std(axis=0) + 1e-8

    def _apply(x: np.ndarray) -> np.ndarray:
        return np.clip((x - mu) / sd, -4.0, 4.0) / 4.0

    return tuple(_apply(x) for x in others)


# ----------------------------------------------------------------------
# MORPHOS under benchmark
# ----------------------------------------------------------------------
@dataclass
class StreamResult:
    domain: str
    residuals: np.ndarray        # (n, 7) target - prediction, z-space
    scores: np.ndarray           # (n,) squared error magnitude
    novelty: np.ndarray          # (n,) associative novelty of the monitored state
    target_steps: np.ndarray     # (n,) absolute step of the target
    labels: np.ndarray           # (n,) ground-truth anomaly labels
    wall_s: float

    def pick(self, dims: str) -> np.ndarray:
        if dims == "lorenz":
            return self.residuals[:, :LORENZ_DIMS]
        if dims == "system":
            return self.residuals[:, LORENZ_DIMS:]
        if dims == "mixed":
            return self.residuals
        raise ValueError(f"unknown dims: {dims}")

    def metrics(self, dims: str = "mixed") -> dict:
        r = self.pick(dims)
        if r.size == 0:
            return {"mae": float("nan"), "rmse": float("nan"),
                    "mse": float("nan"), "n_steps": 0}
        mse = float(np.mean(r ** 2))
        return {
            "mae": float(np.mean(np.abs(r))),
            "rmse": float(np.sqrt(mse)),
            "mse": mse,
            "per_dim_mae": np.mean(np.abs(r), axis=0).tolist(),
            "n_steps": int(r.shape[0]),
        }


class MORPHOSBenchmark:
    """Full MORPHOS pipeline (encode -> multi-scale dynamics -> memory -> predict)."""

    def __init__(self, config: BenchmarkConfig,
                 ablation: Optional[Dict] = None):
        self.config = config
        self.ablation = ablation or {}

        self.use_multiscale = (config.enable_multiscale and
                               not self.ablation.get("no_multiscale", False))
        self.use_orthogonal = (config.enable_orthogonal_projection and
                               not self.ablation.get("no_orthogonal", False))
        self.use_memory = (config.enable_memory and
                           not self.ablation.get("no_memory", False))

        if self.use_multiscale:
            self.dynamics = create_default_multiscale(
                feature_dim=config.feature_dim,
                n_scales=3,
                tau_base=0.005,
                tau_factor=5.0,
                neurons_per_scale=config.n_neurons_per_scale,
                seed=config.seed,
            )
            self.n_scales = 3
        else:
            # Single time constant: one ensemble, no hierarchy.
            self.dynamics = NengoDynamics(
                feature_dim=config.feature_dim,
                n_neurons=int(sum(config.n_neurons_per_scale)),
                dt=DT,
                recurrent_gain=0.5,
                seed=config.seed,
            )
            self.n_scales = 1

        self.projector = SubspaceProjector(
            {"lorenz": LORENZ_DIMS, "system": SYSTEM_DIMS},
            proj_dim=config.proj_dim,
            mode="orthogonal" if self.use_orthogonal else "interference",
            seed=config.seed,
        )
        # Benchmark normalisation: one fixed z-scoring shared with the
        # classical baselines.  Using the adaptive online encoder here
        # would put MORPHOS residuals in a different (and differently
        # scaled) space than the baselines, making MAE incomparable.
        self._z_mu: Optional[np.ndarray] = None
        self._z_sd: Optional[np.ndarray] = None
        # Reservoir-computing readout: every scale's decoded state at t and
        # t-1, the current/previous encoded state, and the multi-scale
        # residual memory.  Feeding the encoded state t (and not only the
        # compressed neural state) is what a standard readout does; without
        # it the predictor is far weaker than a plain linear AR model and
        # the comparison would be meaningless.
        self.n_in = (2 * self.n_scales * config.feature_dim +
                     Z_LAGS * TARGET_DIMS + MEM_CHANNELS)
        self.predictor = NLMSPredictor(n_in=self.n_in, n_out=TARGET_DIMS,
                                       mu=0.08)

        self.temporal_buffer = make_temporal_buffer(
            config.feature_dim, capacity=2048)
        self.associative_memory = make_associative_memory(
            config.feature_dim, max_history=4096)
        # T4.7: importance-sampled memory compressor for long-run stability
        self.compressor = make_compressor(
            config.feature_dim, max_entries=2048, recent_reserve=256)

        self._hist: List[np.ndarray] = []
        self._hist_base = 0
        self._z: Dict[int, np.ndarray] = {}
        self._pred_ptr = 0
        self._lag: Optional[int] = None
        self._last_rate = 0.0
        self._total_steps = 0
        # Circuit breaker: when an alarm fires during evaluation the
        # reservoir is reset and warm-started, so a fault cannot leave the
        # network ringing and generate a burst of follow-up false alarms.
        self.alarm_threshold: Optional[float] = None
        self._recent_feats: List[np.ndarray] = []
        self._alarms = 0
        # T4.5 dual opinion + T4.6 dampened recovery
        self.dual = make_dual_opinion(state_dim=LORENZ_DIMS,
                                      quantile=config.anomaly_quantile)
        self.breaker = make_dampened_breaker()
        self._nov: Dict[int, float] = {}
        self._decoded_ref: Optional[np.ndarray] = None

        self.fit_time_ = 0.0
        self.predict_time_ = 0.0

    # -- bounded decoded history ---------------------------------------
    def _append_history(self, decoded: np.ndarray) -> None:
        self._hist.extend(np.asarray(decoded, dtype=float))
        if len(self._hist) > HIST_LIMIT:
            want = len(self._hist) - HIST_LIMIT // 2
            safe = max(0, (self._pred_ptr - 2) - self._hist_base)
            drop = min(want, safe)
            if drop > 0:
                del self._hist[:drop]
                self._hist_base += drop

    def _hist_get(self, g: int) -> Optional[np.ndarray]:
        i = g - self._hist_base
        if 0 <= i < len(self._hist):
            return self._hist[i]
        return None

    def _prune_z(self) -> None:
        cutoff = self._pred_ptr - 4
        for key in [k for k in self._z if k < cutoff]:
            del self._z[key]
        for key in [k for k in self._nov if k < cutoff]:
            del self._nov[key]

    def _memory_vector(self) -> np.ndarray:
        if not self.use_memory:
            return np.zeros(MEM_CHANNELS)
        emas = self.temporal_buffer.get_all_ema()
        v = np.asarray([emas[s] for s in self.temporal_buffer.cfg.ema_scales],
                       dtype=float)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        # T4.6: while recovering, lean harder on known-good memory.
        return np.tanh(10.0 * v) * self.breaker.memory_gain()

    # -- rolling evaluation --------------------------------------------
    def run_stream(self, steps: int, *, domain: str = "mixed", seed: int = 3,
                   learn: bool = False, finetune: bool = False,
                   anomalies: Optional[List[tuple]] = None) -> StreamResult:
        """Advance the pipeline over a stream and collect one-step errors.

        learn=True trains the NLMS predictor; otherwise predictions are
        evaluated out-of-sample (finetune=True trains during evaluation).
        """
        stream = build_stream(self.config, steps, seed, anomalies=anomalies)
        raw_l, raw_s = stream["lorenz"], stream["system"]
        labels_all = stream["labels"]

        if learn or self._z_mu is None:
            all_raw = np.concatenate([raw_l, raw_s], axis=1)
            self._z_mu = all_raw.mean(axis=0)
            self._z_sd = all_raw.std(axis=0) + 1e-8
        mu_l, sd_l = self._z_mu[:LORENZ_DIMS], self._z_sd[:LORENZ_DIMS]
        mu_s, sd_s = self._z_mu[LORENZ_DIMS:], self._z_sd[LORENZ_DIMS:]

        resid: List[np.ndarray] = []
        scores: List[float] = []
        novelty: List[float] = []
        tsteps: List[int] = []
        labels: List[int] = []
        train_X: List[np.ndarray] = []
        train_Y: List[np.ndarray] = []
        start_wall = time.perf_counter()
        # Streams are consumed back-to-back, so absolute step numbers keep
        # history and targets consistent across train / eval phases.
        base = self._total_steps

        for S in range(0, steps, CHUNK):
            n = min(CHUNK, steps - S)

            feats = np.empty((n, 2 * self.config.proj_dim))
            zl = np.empty((n, LORENZ_DIMS))
            zs = np.empty((n, SYSTEM_DIMS))
            zl[:] = np.clip((raw_l[S:S + n] - mu_l) / sd_l, -4.0, 4.0) / 4.0
            zs[:] = np.clip((raw_s[S:S + n] - mu_s) / sd_s, -4.0, 4.0) / 4.0
            for k in range(n):
                feats[k] = self.projector.features({"lorenz": zl[k],
                                                    "system": zs[k]})

            # T4.5: training states seed associative memory; evaluation
            # states produce the novelty opinion.
            for k in range(n):
                if learn:
                    self.dual.add_reference(zl[k])
                else:
                    self._nov[base + S + k] = self.dual.novelty(zl[k])

            # T4.6: during recovery the input drive is damped, so the
            # membrane potential decays smoothly instead of the whole
            # simulator being rebuilt.
            if self.breaker.active:
                feats = feats * self.breaker.input_gain()

            if self.use_multiscale:
                cres = self.dynamics.run_chunk(feats)
                scale_names = self.dynamics.scale_names
                decoded = np.concatenate(
                    [cres[name]["decoded"] for name in scale_names], axis=1)
                rates = np.mean([cres[name]["rate_mean"]
                                 for name in scale_names], axis=0)
            else:
                cres = self.dynamics.run_chunk(feats)
                decoded = cres["decoded"]
                rates = np.asarray(cres["rate_mean"], dtype=float)

            decoded = np.asarray(decoded, dtype=float).copy()
            self.breaker.advance(n)

            if self._lag is None:
                self._lag = self.dynamics.estimate_lag(
                    feats, decoded[:, :self.config.feature_dim])

            self._last_rate = float(np.mean(rates)) if rates.size else 0.0

            self._append_history(decoded)
            self._recent_feats.extend(np.asarray(feats, dtype=float))
            if len(self._recent_feats) > 100:
                del self._recent_feats[:-100]
            for k in range(n):
                self._z[base + S + k] = np.concatenate([zl[k], zs[k]])

            # Only drain predictions whose decoded representation and
            # target both exist; the tail is deferred to the next chunk.
            # The readout uses the state at t to predict t+1: strictly
            # causal, no within-chunk lookahead.  (Indexing decoded at
            # t+lag leaked the target's own step into the input.)
            last_s = base + (S + n - 2)
            while self._pred_ptr <= last_s:
                s = self._pred_ptr
                rep = self._hist_get(s)
                if rep is None:
                    break
                prev = self._hist_get(s - 1)
                if prev is None:
                    prev = np.zeros(self.n_scales * self.config.feature_dim)
                target = self._z.get(s + 1)
                if target is None:
                    break
                if self._z.get(s) is None:
                    break
                zeros = np.zeros(TARGET_DIMS)
                z_taps = [self._z.get(s - i, zeros) for i in range(Z_LAGS)]

                x = np.concatenate([rep, prev, *z_taps,
                                    self._memory_vector()])
                pred = self.predictor.predict(x)
                err = target - pred
                sc = float(err @ err)

                if learn or finetune:
                    self.predictor.update(x, target, weight=0.95)

                if learn:
                    train_X.append(x)
                    train_Y.append(target)

                if self.use_memory:
                    # Memory tracks the fast scale's representation only;
                    # it indexes feature_dim vectors, not the readout stack.
                    mem_feat = rep[:self.config.feature_dim]
                    self.temporal_buffer.push_step(mem_feat, 0.95, sc,
                                                   self._last_rate)
                    self.associative_memory.add(mem_feat, sc, s + 1)

                if (not learn and not finetune and
                        self.alarm_threshold is not None and
                        sc > self.alarm_threshold and
                        self.breaker.can_trip()):
                    self.breaker.trip(s)
                    self._alarms += 1

                resid.append(err)
                scores.append(sc)
                novelty.append(float(self._nov.get(s, 0.0)))
                local = s + 1 - base
                tsteps.append(local)
                labels.append(int(labels_all[local]))
                self._pred_ptr += 1

            self._prune_z()

        self._total_steps = base + steps

        if learn and train_X:
            # Offline ridge fit, matching how the classical baselines are
            # trained on the full training segment.
            self.predictor.fit_batch(np.asarray(train_X),
                                     np.asarray(train_Y))
        wall = time.perf_counter() - start_wall
        if learn:
            self.fit_time_ += wall
        else:
            self.predict_time_ += wall

        if resid:
            R = np.asarray(resid)
            S_ = np.asarray(scores, dtype=float)
            T = np.asarray(tsteps, dtype=int)
        else:
            R = np.zeros((0, TARGET_DIMS))
            S_ = np.zeros(0)
            T = np.zeros(0, dtype=int)
        return StreamResult(
            domain=domain,
            residuals=R,
            scores=S_,
            novelty=np.asarray(novelty, dtype=float),
            target_steps=T,
            labels=np.asarray(labels, dtype=int),
            wall_s=wall,
        )


# ----------------------------------------------------------------------
# Benchmark phases
# ----------------------------------------------------------------------
ABLATIONS = [
    {"name": "full", "no_multiscale": False, "no_orthogonal": False,
     "no_memory": False},
    {"name": "no_multiscale", "no_multiscale": True, "no_orthogonal": False,
     "no_memory": False},
    {"name": "no_orthogonal", "no_multiscale": False, "no_orthogonal": True,
     "no_memory": False},
    {"name": "no_memory", "no_multiscale": False, "no_orthogonal": False,
     "no_memory": True},
    {"name": "minimal", "no_multiscale": True, "no_orthogonal": True,
     "no_memory": True},
]


def run_morphos_ablation_study(config: BenchmarkConfig) -> tuple:
    """Train each ablation variant and evaluate on the mixed stream."""
    results: Dict[str, dict] = {}
    models: Dict[str, MORPHOSBenchmark] = {}

    for abl in ABLATIONS:
        name = abl["name"]
        print(f"\n[ablation] {name}")
        model = MORPHOSBenchmark(config, ablation=abl)

        trained = model.run_stream(config.train_steps, domain="mixed",
                                   seed=config.seed + 1, learn=True)
        tested = model.run_stream(config.test_steps, domain="mixed",
                                  seed=config.seed + 2, learn=False)

        entry = {
            "lorenz": tested.metrics("lorenz"),
            "system": tested.metrics("system"),
            "mixed": tested.metrics("mixed"),
            "train_time_s": trained.wall_s,
            "eval_time_s": tested.wall_s,
            "lag": model._lag,
        }
        results[name] = entry
        models[name] = model

        print(f"  MAE lorenz={entry['lorenz']['mae']:.4f} "
              f"system={entry['system']['mae']:.4f} "
              f"mixed={entry['mixed']['mae']:.4f} "
              f"({entry['train_time_s']:.1f}s train)")

    return results, models


def run_baseline_forecast(config: BenchmarkConfig) -> Dict[str, dict]:
    """Traditional forecasters on the exact stream MORPHOS predicts."""
    train_raw, _ = mixed_series(config, config.train_steps, config.seed + 1)
    test_raw, _ = mixed_series(config, config.test_steps, config.seed + 2)
    train_z, test_z = normalise(train_raw, train_raw, test_raw)

    forecasters = [b for b in get_all_baselines()
                   if not b.name.startswith("IForest")]
    print(f"\n[baselines] {len(forecasters)} forecasters on 7-D z-scored stream")
    return run_baseline_comparison(train_z, test_z, forecasters)


def _thresholded(scores_calib: np.ndarray, scores_test: np.ndarray,
                 labels_test: np.ndarray, quantile: float) -> dict:
    thr = calibrate_threshold(scores_calib, quantile)
    m = precision_recall(labels_test, threshold_labels(scores_test, thr))
    m["threshold"] = thr
    m["best_f1"] = best_f1_over_thresholds(labels_test, scores_test)
    return m


def make_spike_schedule(steps: int, seed: int, n_runs: int = 6,
                        length: int = 5, gap: int = 120) -> List[tuple]:
    """Point-spike-only fault schedule (the T4.5 acceptance task)."""
    schedule: List[tuple] = []
    pos = 50
    while len(schedule) < n_runs and pos + length < steps - 40:
        schedule.append((int(pos), int(length), "spike"))
        pos += int(length) + int(gap)
    return schedule


def _monitored_scores(result: StreamResult) -> np.ndarray:
    """Squared one-step error on the monitored channels only."""
    return np.sum(result.residuals[:, :LORENZ_DIMS] ** 2, axis=1)


def _run_anomaly_task(config: BenchmarkConfig, model: MORPHOSBenchmark,
                      schedule: List[tuple], q: float,
                      labels_all: np.ndarray) -> Dict[str, dict]:
    """Evaluate every detector on one labelled stream.

    All detectors see the same 3 monitored channels and are scored on the
    same target indices, all calibrated to the same clean FP budget.
    """
    calib = model.run_stream(config.calib_steps, domain="mixed",
                             seed=config.seed + 5, learn=False)
    calib_err = _monitored_scores(calib)

    # T4.5: calibrate the dual-opinion detector on clean data, then arm
    # the T4.6 breaker at the same operating point.
    model.dual.calibrate(calib_err, calib.novelty)
    model.alarm_threshold = calibrate_threshold(calib_err, q)
    model._alarms = 0
    test = model.run_stream(config.anomaly_steps, domain="mixed",
                            seed=config.seed + 7, anomalies=schedule)
    model.alarm_threshold = None
    test_err = _monitored_scores(test)
    y = test.labels

    results: Dict[str, dict] = {}
    # Convention: MORPHOS declares an anomaly on the sample it has just
    # observed (state s), exactly like Isolation Forest scores z_t and is
    # judged against label t.  The recorded novelty is state s, so its
    # label is the label of step s, not of the target s+1.
    y_state = labels_all[test.target_steps - 1]
    dual_scores = model.dual.score_series(test_err, test.novelty)
    dual_pred = (dual_scores > model.dual.threshold).astype(int)
    dual = precision_recall(y_state, dual_pred)
    dual["threshold"] = float(model.dual.threshold)
    dual["best_f1"] = best_f1_over_thresholds(y_state, dual_scores)
    dual["breaker_trips"] = int(model._alarms)
    results["MORPHOS_dual"] = dual

    # Ablation: the residual opinion scored on its own, same clean budget.
    ops_test = model.dual.opinions(test_err, test.novelty)
    ops_cal = model.dual.calibration_opinions()
    thr_r = float(np.quantile(ops_cal["residual_z"], q))
    res_ab = precision_recall(y, (ops_test["residual_z"] > thr_r).astype(int))
    res_ab["threshold"] = thr_r
    res_ab["best_f1"] = best_f1_over_thresholds(y, ops_test["residual_z"])
    results["MORPHOS_residual"] = res_ab

    train_raw, _ = mixed_series(config, config.train_steps, config.seed + 1)
    calib_raw, _ = mixed_series(config, config.calib_steps, config.seed + 5)
    test_raw, _ = mixed_series(config, config.anomaly_steps,
                               config.seed + 7, anomalies=schedule)
    train_z, calib_z, test_z = normalise(train_raw, train_raw, calib_raw,
                                         test_raw)
    lz = slice(0, LORENZ_DIMS)
    train_z, calib_z, test_z = train_z[:, lz], calib_z[:, lz], test_z[:, lz]

    ci = calib.target_steps
    ti = test.target_steps

    if HAS_SKLEARN:
        def embed(z: np.ndarray) -> np.ndarray:
            prev = np.vstack([z[:1], z[:-1]])
            return np.concatenate([z, prev], axis=1)

        iforest = IsolationForestBaseline(contamination=0.1,
                                          name="IsolationForest")
        iforest.fit(embed(train_z))
        calib_s = iforest.predict(embed(calib_z)).anomaly_scores
        test_s = iforest.predict(embed(test_z)).anomaly_scores
        results["IsolationForest"] = _thresholded(
            calib_s[ci], test_s[ti], labels_all[ti], q)

    residual_models = [ARBaseline(order=5, name="AR5")]
    if HAS_STATSMODELS:
        residual_models.append(ARIMABaseline(order=(5, 0, 0),
                                             name="ARIMA_5_0_0"))
    for det in residual_models:
        det.fit(train_z)
        calib_s = residual_scores(det.predict(calib_z))
        test_s = residual_scores(det.predict(test_z))
        results[det.name] = _thresholded(calib_s[ci], test_s[ti],
                                         labels_all[ti], q)
    return results


def run_anomaly_benchmark(config: BenchmarkConfig,
                          model: MORPHOSBenchmark) -> Dict[str, dict]:
    """Anomaly detection on the mixed task and the point-spike task."""
    q = config.anomaly_quantile
    mixed = make_anomaly_schedule(config.anomaly_steps, seed=config.seed + 6)
    spikes = make_spike_schedule(config.anomaly_steps, seed=config.seed + 8)

    tasks = {"mixed": mixed, "point_spike": spikes}
    results: Dict[str, dict] = {}
    for task, schedule in tasks.items():
        n_anom = sum(length for _s, length, _k in schedule)
        print(f"\n[anomaly:{task}] {len(schedule)} injections, "
              f"{n_anom} anomalous steps")
        _, labels_all = mixed_series(config, config.anomaly_steps,
                                     config.seed + 7, anomalies=schedule)
        rows = _run_anomaly_task(config, model, schedule, q, labels_all)
        for name, m in rows.items():
            print(f"  {name}: P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"F1={m['f1']:.3f} (thr={m['threshold']:.4f})")
        results[task] = rows
    return results


# ----------------------------------------------------------------------
# Acceptance + reports
# ----------------------------------------------------------------------
def build_acceptance(config: BenchmarkConfig, ablation: Dict[str, dict],
                     baselines: Dict[str, dict],
                     anomaly: Dict[str, dict]) -> Dict[str, dict]:
    full = ablation.get("full", {})
    single = ablation.get("no_multiscale", {})

    ms_ok = (full.get("mixed", {}).get("mae", float("inf")) <
             single.get("mixed", {}).get("mae", float("inf")))
    ms_improvement = 0.0
    if single.get("mixed", {}).get("mae"):
        ms_improvement = 100.0 * (
            1.0 - full["mixed"]["mae"] / single["mixed"]["mae"])

    def _base(name: str, dims: str):
        res = baselines.get(name, {})
        per_dim = res.get("mae")
        if per_dim is None:
            return float("inf")
        dim_slice = (slice(0, LORENZ_DIMS) if dims == "lorenz"
                     else slice(LORENZ_DIMS, TARGET_DIMS))
        return float(np.mean(np.asarray(per_dim)[dim_slice]))

    arima_mae = _base("ARIMA_5_0_0", "lorenz")
    ar5_mae = _base("AR5", "lorenz")
    best_classical = min(arima_mae, ar5_mae)
    morphos_lorenz = full.get("lorenz", {}).get("mae", float("inf"))
    forecast_ok = morphos_lorenz < best_classical if np.isfinite(
        best_classical) else False

    # T4.5 acceptance is the point-spike task: F1 *and* precision must
    # beat Isolation Forest at the same FP budget.
    spike = anomaly.get("point_spike", {})
    m = spike.get("MORPHOS_dual", {})
    iforest = spike.get("IsolationForest", {})
    arima = spike.get("ARIMA_5_0_0", {})
    anomaly_ok = bool(m) and bool(iforest) and (
        m.get("f1", 0.0) > iforest.get("f1", 0.0) and
        m.get("precision", 0.0) > iforest.get("precision", 0.0)
    )

    return {
        "multiscale_beats_single_scale": {
            "description": "multi-scale residual MAE < single-scale on the "
                           "mixed fast/slow stream",
            "full_mae": full.get("mixed", {}).get("mae"),
            "single_scale_mae": single.get("mixed", {}).get("mae"),
            "improvement_pct": float(ms_improvement),
            "pass": bool(ms_ok),
        },
        "forecast_beats_classical": {
            "description": "MORPHOS Lorenz-dim MAE < best classical "
                           "forecaster (AR/ARIMA)",
            "morphos_mae": morphos_lorenz,
            "best_classical_mae": best_classical,
            "pass": bool(forecast_ok),
        },
        "anomaly_beats_classical": {
            "description": "point-spike task: MORPHOS dual-opinion F1 and "
                           "precision > Isolation Forest at 1% FPR",
            "morphos_f1": m.get("f1"),
            "morphos_precision": m.get("precision"),
            "isolation_forest_f1": iforest.get("f1"),
            "isolation_forest_precision": iforest.get("precision"),
            "arima_f1": arima.get("f1"),
            "pass": bool(anomaly_ok),
        },
    }


def generate_markdown_report(results: Dict, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Phase IV Benchmark Report\n\n")
        f.write(f"**Generated:** {results.get('timestamp', 'N/A')}  \n")
        f.write(f"**Runtime:** {results.get('runtime_s', 0):.1f}s\n\n")

        f.write("## Configuration\n\n")
        for k, v in results.get("config", {}).items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n")

        f.write("## 1. MORPHOS Ablation Study\n\n")
        f.write("| Configuration | Lorenz MAE | System MAE | Mixed MAE | "
                "Mixed RMSE |\n")
        f.write("|---|---|---|---|---|\n")
        for name, res in results.get("morphos_ablation", {}).items():
            f.write(f"| {name} | {res['lorenz']['mae']:.4f} | "
                    f"{res['system']['mae']:.4f} | "
                    f"{res['mixed']['mae']:.4f} | "
                    f"{res['mixed']['rmse']:.4f} |\n")
        f.write("\n")

        f.write("## 2. Traditional Baselines (7-D z-scored mixed stream)\n\n")
        f.write("| Model | MAE | RMSE | Fit (s) | Predict (s) |\n")
        f.write("|---|---|---|---|---|\n")
        for name, res in results.get("baselines", {}).items():
            if "error" in res:
                f.write(f"| {name} | ERROR: {res['error']} | - | - | - |\n")
            else:
                f.write(f"| {name} | {res['mean_mae']:.4f} | "
                        f"{res['mean_rmse']:.4f} | {res['fit_time']:.2f} | "
                        f"{res['predict_time']:.4f} |\n")
        f.write("\n")

        f.write("## 3. Anomaly Detection (equal false-positive budget)\n\n")
        f.write("All detectors monitor the same 3-channel Lorenz subsystem, "
                "calibrated to the same false-positive budget on clean "
                "data.\n\n")
        for task, rows in results.get("anomaly", {}).items():
            f.write(f"### Task: {task}\n\n")
            f.write("| Detector | Precision | Recall | F1 | Best F1 |\n")
            f.write("|---|---|---|---|---|\n")
            for name, m in rows.items():
                f.write(f"| {name} | {m['precision']:.3f} | "
                        f"{m['recall']:.3f} | {m['f1']:.3f} | "
                        f"{m['best_f1']['f1']:.3f} |\n")
            f.write("\n")

        f.write("## 4. Acceptance Verdicts\n\n")
        f.write("| Criterion | Result | Detail |\n|---|---|---|\n")
        for name, a in results.get("acceptance", {}).items():
            status = "PASS" if a.get("pass") else "FAIL"
            if name == "multiscale_beats_single_scale":
                detail = (f"full {a['full_mae']:.4f} vs single "
                          f"{a['single_scale_mae']:.4f} "
                          f"({a['improvement_pct']:+.1f}%)")
            elif name == "forecast_beats_classical":
                detail = (f"MAE {a['morphos_mae']:.4f} vs "
                          f"{a['best_classical_mae']:.4f}")
            else:
                detail = (f"F1 {a['morphos_f1']:.3f} vs IForest "
                          f"{a['isolation_forest_f1']:.3f} / ARIMA "
                          f"{a['arima_f1']:.3f}")
            f.write(f"| {name} | {status} | {detail} |\n")
        f.write("\n---\n*Phase IV Benchmark Runner*\n")


def run_full_benchmark(config: BenchmarkConfig, out_dir: str) -> Dict:
    t0 = time.perf_counter()
    print("=" * 60)
    print("PHASE IV BENCHMARK SUITE")
    print("=" * 60)

    ablation, models = run_morphos_ablation_study(config)
    baselines = run_baseline_forecast(config)
    anomaly = run_anomaly_benchmark(config, models["full"])
    acceptance = build_acceptance(config, ablation, baselines, anomaly)

    all_results = {
        "config": asdict(config),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_s": time.perf_counter() - t0,
        "morphos_ablation": ablation,
        "baselines": baselines,
        "anomaly": anomaly,
        "acceptance": acceptance,
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "benchmark_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    md_path = os.path.join(out_dir, "benchmark_report.md")
    generate_markdown_report(all_results, md_path)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    for name, a in acceptance.items():
        print(f"  [{'PASS' if a['pass'] else 'FAIL'}] {name}")
    print(f"JSON: {json_path}")
    print(f"MD  : {md_path}")
    return all_results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-steps", type=int, default=6000)
    ap.add_argument("--test-steps", type=int, default=500)
    ap.add_argument("--calib-steps", type=int, default=600)
    ap.add_argument("--anomaly-steps", type=int, default=900)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "experiments", "benchmarks", "results"))
    args = ap.parse_args()

    config = BenchmarkConfig(
        train_steps=args.train_steps,
        test_steps=args.test_steps,
        calib_steps=args.calib_steps,
        anomaly_steps=args.anomaly_steps,
    )
    run_full_benchmark(config, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
