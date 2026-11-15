"""MORPHOS streaming runtime owned by the IPC daemon process."""

from __future__ import annotations

import logging
import os
from collections import deque

import numpy as np

from adapters.chaos import ChaosInjector
from adapters.lorenz import LorenzSource
from core.counterfactual.engine import (
    BranchConfig,
    CounterfactualConfig,
    CounterfactualEngine,
)
from core.dynamics.engine import NengoDynamics
from core.encoding.encoder import FeatureEncoder
from core.gateway.schemas import GatewayRequest, GatewayResponse
from core.ingest.ring_buffer import TimeAligner
from core.memory.buffer import RollingTemporalBuffer, TemporalConfig
from core.memory.compressor import make_compressor
from core.observer.circuit_breaker import make_dampened_breaker
from core.observer.dual_opinion import make_dual_opinion
from core.observer.observer import CircuitBreaker
from core.prediction.predictor import NLMSPredictor
from core.memory.thread_safe_buffer import ThreadSafeBuffer, synchronized
from storage.sqlite_store import Store
from storage.wal import WALWriter

LOGGER = logging.getLogger(__name__)

DT = 0.005
LOW_CONF = 0.2
WAL_FLAG_LOWS = 1 << 6
WAL_FLAG_BREAKER = 1 << 7
WAL_FLAG_ANOMALY = 1 << 8
DUAL_CALIBRATION_STEPS = 600


class MORPHOSRuntime(ThreadSafeBuffer):
    """Run the neural, prediction, anomaly, and WAL pipeline in one process."""

    def __init__(self, data_dir: str = "data_gateway", neurons: int = 400,
                 dt: float = DT, dim: int = 3):
        if dim != 3:
            raise ValueError("the MORPHOS WAL/core contract requires 3 dimensions")
        if neurons < 1:
            raise ValueError("neurons must be positive")
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("dt must be finite and positive")
        super().__init__()
        self.data_dir = data_dir
        self.dt = float(dt)
        self.aligner = TimeAligner(dt=self.dt, dim=dim)
        self.encoder = FeatureEncoder(state_dim=dim, proj_dim=8, seed=0)
        self.engine = NengoDynamics(
            feature_dim=self.encoder.feature_dim, n_neurons=neurons,
            dt=self.dt, recurrent_gain=0.5, seed=0)
        self.predictor = NLMSPredictor(
            n_in=2 * self.encoder.feature_dim, n_out=dim, mu=0.05)
        self.health = CircuitBreaker()
        self.breaker = make_dampened_breaker(health=self.health)
        self.dual = make_dual_opinion(state_dim=dim)
        self.compressor = make_compressor(
            self.encoder.feature_dim, max_entries=2048)
        self.temporal = RollingTemporalBuffer(
            self.encoder.feature_dim, TemporalConfig())
        self._decoded: deque[np.ndarray] = deque(maxlen=2)
        self._calib_residuals: list[float] = []
        self._calib_novelties: list[float] = []
        self._calibrated = False
        self._step = 0
        self._accepted = 0
        self._closed = False
        self._mode = "live"
        self._chaos_dropped = 0
        self._chaos = ChaosInjector(
            LorenzSource(dt=self.dt), seed=1, noise_sigma=0.5,
            noise_prob=0.10, drop_prob=0.02, nan_prob=0.01,
            spike_prob=0.01, spike_scale=1e4)
        self._state_history: deque[tuple[int, np.ndarray, float]] = deque(
            maxlen=1024)
        self._counterfactual: CounterfactualEngine | None = None
        self._branch_serial = 0
        self._latest: dict | None = None

        os.makedirs(data_dir, exist_ok=True)
        self.store = Store(os.path.join(data_dir, "morphos.db"),
                           check_same_thread=False)
        self.wal = WALWriter(os.path.join(data_dir, "trajectory.wal"))
        self._step = self.wal.total_records
        self.store.begin_run(note="IPC gateway MORPHOS runtime")
        self.store.log_event(
            "INFO", "IPC gateway core started",
            data={"dt": self.dt, "neurons": neurons, "dimensions": dim})

    @synchronized
    def handle(self, request: GatewayRequest) -> GatewayResponse:
        if self._closed:
            return GatewayResponse.failure(
                "core_closed", "MORPHOS runtime is shutting down",
                request.request_id)

        if request.op == "ingest":
            if request.timestamp is None or request.values is None:
                return GatewayResponse.failure(
                    "invalid_request", "ingest requires timestamp and values",
                    request.request_id)
            if len(request.values) != self.aligner.dim:
                return GatewayResponse.failure(
                    "dimension_mismatch",
                    f"expected {self.aligner.dim} values",
                    request.request_id)
            result = self._ingest(request.timestamp, request.values)
            return GatewayResponse.success(result, request.request_id)

        if request.op == "query":
            if request.timestamp is None:
                return GatewayResponse.failure(
                    "invalid_request", "query requires timestamp",
                    request.request_id)
            values, confidence, flags = self.aligner.at(request.timestamp)
            return GatewayResponse.success({
                "timestamp": request.timestamp,
                "values": values.tolist(),
                "confidence": confidence,
                "flags": int(flags),
            }, request.request_id)

        if request.op == "reset":
            self._reset()
            return GatewayResponse.success({
                "reset": True,
                "step": self._step,
                "wal_records": self.wal.total_records,
            }, request.request_id)

        if request.op == "control":
            return self._control(request)

        return GatewayResponse.success(self._status(), request.request_id)

    def _ingest(self, timestamp: float, values: tuple[float, ...]) -> dict:
        raw_values = np.asarray(values, dtype=float)
        if self._mode == "combat5":
            raw_values = self._chaos.corrupt(timestamp, raw_values)
            if raw_values is None:
                self._chaos_dropped += 1
                self._accepted += 1
                return {
                    "accepted": True,
                    "dropped": True,
                    "step": self._step,
                    "samples_received": self._accepted,
                    "wal_records": self.wal.total_records,
                }

        self.aligner.push(timestamp, raw_values)
        raw, confidence, flags = self.aligner.at(timestamp)
        self.encoder.observe(raw, weight=confidence)
        z_state, feats = self.encoder.encode(raw)

        novelty = self.dual.novelty(z_state)
        if not self._calibrated:
            self.dual.add_reference(z_state)

        driven = feats * self.breaker.input_gain()
        neural = self.engine.run_chunk(driven.reshape(1, -1))
        decoded = neural["decoded"][0].copy()
        rate = float(neural["rate_mean"][0])

        if self._decoded:
            previous = self._decoded[-1]
            older = (self._decoded[-2] if len(self._decoded) > 1
                     else np.zeros(self.encoder.feature_dim))
            model_input = np.concatenate([previous, older])
            prediction = self.predictor.predict(model_input)
            residual = self.predictor.update(
                model_input, z_state, weight=confidence)
        else:
            prediction = np.zeros(3, dtype=float)
            residual = float(z_state @ z_state)
        self._decoded.append(decoded)

        self.breaker.observe_residual(residual)
        is_anomaly = False
        opinion = None
        if self._calibrated:
            opinion = self.dual.score(
                residual=residual, novelty=novelty, state=z_state)
            is_anomaly = bool(opinion["is_anomaly"])
        else:
            self._calib_residuals.append(residual)
            self._calib_novelties.append(novelty)
            if len(self._calib_residuals) >= DUAL_CALIBRATION_STEPS:
                try:
                    self.dual.calibrate(
                        np.asarray(self._calib_residuals, dtype=float),
                        np.asarray(self._calib_novelties, dtype=float))
                    self._calibrated = True
                    self.store.log_event(
                        "INFO", "dual-opinion calibrated",
                        sim_t=self._step * self.dt,
                        data={"threshold": self.dual.threshold})
                except ValueError as exc:
                    LOGGER.warning("dual-opinion calibration deferred: %s", exc)
        if is_anomaly and opinion is not None:
            self.store.log_event(
                "WARN", "Dual-Opinion anomaly detected",
                sim_t=self._step * self.dt,
                data={"score": opinion["score"],
                      "threshold": self.dual.threshold,
                      "residual": residual, "novelty": novelty})

        stats = {
            "mean_rate": rate,
            "max_rate": float(neural["rate_max"][0]),
            "spike_frac": float(neural["spike_frac"][0]),
            "nan_decoded": int(neural["nan_decoded"]),
            "conf_mean": confidence,
        }
        event = self.breaker.evaluate(stats)
        if event:
            level = "ERROR" if event.startswith("TRIP") else "WARN"
            self.store.log_event(level, event, sim_t=self._step * self.dt,
                                 data=stats)
            if event.startswith("TRIP"):
                self.breaker.trip(self._step, reference=z_state)

        feature_to_keep = (decoded if decoded.size >= self.encoder.feature_dim
                           else feats)
        self.compressor.push(self._step, feature_to_keep, residual)
        self.temporal.push_step(feats, confidence, residual, rate)
        self._state_history.append((self._step + 1, z_state.copy(), confidence))
        if self._counterfactual is not None:
            self._counterfactual.step_branches(
                z_state, z_state, confidence)

        wal_flags = int(flags)
        if confidence < LOW_CONF:
            wal_flags |= WAL_FLAG_LOWS
        if self.breaker.active:
            wal_flags |= WAL_FLAG_BREAKER
        if is_anomaly:
            wal_flags |= WAL_FLAG_ANOMALY
        self.wal.append(
            (self._step + 1) * self.dt, confidence, z_state, prediction,
            residual, rate, wal_flags)
        self._step += 1
        self._accepted += 1
        self.breaker.advance()
        self._latest = {
            "step": self._step,
            "timestamp": self._step * self.dt,
            "actual": z_state.tolist(),
            "predicted": prediction.tolist(),
            "residual": residual,
            "residual_ema": self.temporal.resid_ema(200),
            "rate": rate,
            "confidence": confidence,
            "flags": wal_flags,
            "anomaly": is_anomaly,
            "anomaly_score": (opinion["score"] if opinion else None),
            "anomaly_threshold": (self.dual.threshold
                                  if self._calibrated else None),
            "breaker_state": self.health.state,
        }

        return {
            "accepted": True,
            "step": self._step,
            "samples_received": self._accepted,
            "wal_records": self.wal.total_records,
            "confidence": confidence,
            "residual": residual,
            "anomaly": is_anomaly,
            "dual_calibrated": self._calibrated,
            "mode": self._mode,
        }

    def _control(self, request: GatewayRequest) -> GatewayResponse:
        if request.action == "set_mode":
            self._mode = request.mode or "live"
            self.store.log_event("INFO", "runtime mode changed",
                                 sim_t=self._step * self.dt,
                                 data={"mode": self._mode})
            result = {"mode": self._mode}
        elif request.action == "reset_breaker":
            self.health = CircuitBreaker()
            self.breaker = make_dampened_breaker(health=self.health)
            self.store.log_event("INFO", "circuit breaker reset",
                                 sim_t=self._step * self.dt)
            result = {"breaker_state": self.health.state,
                      "breaker_trips": self.breaker.trip_count}
        elif request.action == "branch":
            try:
                result = self._create_branch(
                    request.branch_mode or "", request.step)
            except (ValueError, RuntimeError) as exc:
                return GatewayResponse.failure(
                    "branch_unavailable", str(exc), request.request_id)
        else:
            return GatewayResponse.failure(
                "invalid_control", "unsupported control action",
                request.request_id)
        return GatewayResponse.success(result, request.request_id)

    def _create_branch(self, mode: str, step: int | None) -> dict:
        if step is None:
            raise ValueError("branch step is required")
        if mode not in {"gain", "noise", "step", "dropout"}:
            raise ValueError("unsupported counterfactual mode")
        history = [item for item in self._state_history if item[0] <= step]
        if not history or history[-1][0] != step:
            raise ValueError("requested step is outside retained core history")
        if self._counterfactual is not None and \
                self._counterfactual.list_branches():
            raise RuntimeError("an active counterfactual branch already exists")

        counterfactual = CounterfactualEngine(
            feature_dim=6, n_neurons=self.engine.n_neurons, dt=self.dt,
            config=CounterfactualConfig(
                max_branches=4, snapshot_interval=1 << 30,
                max_snapshots=2, branch_timeout_steps=1000))
        for _history_step, state, confidence in history:
            counterfactual.step_branches(state, state, confidence)
        snapshot = counterfactual.create_snapshot(
            step, step * self.dt, lag=1,
            warm_feats=counterfactual.temporal_buffer.get_warm_feats(
                counterfactual.temporal_buffer.capacity))

        self._branch_serial += 1
        name = f"branch-{self._branch_serial}"
        perturbations = {
            "gain": CounterfactualEngine.perturb_input_scale(1.6),
            "noise": CounterfactualEngine.perturb_input_noise(0.35),
            "step": CounterfactualEngine.perturb_input_additive(
                np.full(3, 0.8)),
            "dropout": lambda _state, _branch_step: np.zeros(3),
        }
        counterfactual.create_branch(snapshot, BranchConfig(
            name=name, input_perturbation=perturbations[mode],
            use_separate_predictor=True))
        self._counterfactual = counterfactual
        self.store.log_event(
            "INFO", "counterfactual branch created",
            sim_t=step * self.dt, data={"branch": name, "mode": mode})
        return {"branch": name, "mode": mode, "origin_step": step}

    def _reset(self) -> None:
        self.aligner.reset()
        self.encoder = FeatureEncoder(state_dim=3, proj_dim=8, seed=0)
        self.engine.reset()
        self.predictor = NLMSPredictor(
            n_in=2 * self.encoder.feature_dim, n_out=3, mu=0.05)
        self.health = CircuitBreaker()
        self.breaker = make_dampened_breaker(health=self.health)
        self.dual = make_dual_opinion(state_dim=3)
        self.compressor = make_compressor(
            self.encoder.feature_dim, max_entries=2048)
        self.temporal = RollingTemporalBuffer(
            self.encoder.feature_dim, TemporalConfig())
        self._decoded.clear()
        self._calib_residuals.clear()
        self._calib_novelties.clear()
        self._calibrated = False
        self._accepted = 0
        self._mode = "live"
        self._chaos_dropped = 0
        self._state_history.clear()
        self._counterfactual = None
        self._latest = None
        self.store.log_event("INFO", "IPC runtime reset",
                             sim_t=self._step * self.dt)

    def _status(self) -> dict:
        return {
            "dimension": self.aligner.dim,
            "samples_received": self._accepted,
            "steps_processed": self._step,
            "wal_records": self.wal.total_records,
            "dropped_samples": self.aligner.dropped_count,
            "dual_calibrated": self._calibrated,
            "dual_threshold": (self.dual.threshold
                               if self._calibrated else None),
            "neural_steps": self.engine.steps_run,
            "breaker_state": self.health.state,
            "breaker_trips": self.breaker.trip_count,
            "temporal_samples": len(self.temporal),
            "mode": self._mode,
            "chaos_dropped": self._chaos_dropped,
            "latest": self._latest,
            "branches": self._branch_status(),
        }

    def _branch_status(self) -> list[dict]:
        if self._counterfactual is None:
            return []
        branches = []
        for name in self._counterfactual.list_branches():
            branch = self._counterfactual.get_branch(name)
            if branch is None:
                continue
            latest = branch.trajectory[-1].tolist() if branch.trajectory else None
            branches.append({
                "name": name,
                "steps": branch.steps_run,
                "latest_decoded": latest,
                "latest_residual": (float(branch.residuals[-1])
                                    if branch.residuals else None),
                "latest_rate": (float(branch.rates[-1])
                                if branch.rates else None),
            })
        return branches

    @synchronized
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.store.log_event(
            "INFO", "IPC gateway core stopped",
            sim_t=self._step * self.dt,
            data={"steps": self._step,
                  "wal_records": self.wal.total_records})
        self.store.end_run(self._step)
        self.wal.close()
        self.store.close()
