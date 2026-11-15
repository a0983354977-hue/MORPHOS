"""PROJECT MORPHOS - Phase I main loop (upgraded with Phase IV.b).

Pipeline per control step (dt = 5 ms):
    adapters (Lorenz + chaos) -> ingest/aligner -> encoder
    -> nengo dynamics (chunked) -> NLMS predictor
    -> Phase IV.b: dual-opinion detector + dampened breaker + compressor
    -> WAL binary log + SQLite metadata/snapshots

Phase IV.b upgrades integrated:
  * DualOpinionDetector (T4.5): associative novelty + residual opinion,
    adaptive memory replenishment during long runs.
  * DampenedCircuitBreaker (T4.6): smooth dampened recovery instead of
    hard simulator rebuild on TRIP.
  * ImportanceCompressor (T4.7): bounded-footprint trajectory summary
    for infinite-horizon runs.

Run:  python main.py [--steps N] [--chaos none|combat] [--resume]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adapters.chaos import ChaosInjector
from adapters.lorenz import LorenzSource
from core.dynamics.engine import NengoDynamics
from core.encoding.encoder import FeatureEncoder
from core.ingest.ring_buffer import TimeAligner
from core.memory.history import (TrajectoryBuffer, load_snapshot,
                                 make_snapshot)
from core.memory.compressor import make_compressor
from core.observer.circuit_breaker import make_dampened_breaker
from core.observer.dual_opinion import make_dual_opinion
from core.observer.observer import CircuitBreaker
from core.prediction.predictor import NLMSPredictor
from storage.sqlite_store import Store
from storage.wal import WALWriter

logging.getLogger("nengo").setLevel(logging.CRITICAL)

DT = 0.005               # control step (s)
SRC_DT = 0.001           # source sample step (s)
LOOKAHEAD = 0.002        # extra source horizon pushed before align()
CHUNK = 200              # neural sim steps per chunk
WARM_K = 100             # steps replayed after a breaker reset
SNAP_EVERY = 5           # snapshot every N chunks
LOW_CONF = 0.2
WAL_FLAG_LOWS = 1 << 6
WAL_FLAG_BREAKER = 1 << 7
# Phase IV.b defaults
DUAL_CALIB_STEPS = 600   # steps of clean data used for dual-opinion calibration
COMPRESSOR_MAX = 2048    # max top-K entries in importance compressor


def build_components(args):
    if getattr(args, "source", "lorenz") == "rust":
        # T5.2: same protocol, but samples are decoded by the Rust engine out
        # of a binary trace instead of being integrated in Python.
        from adapters.rust_trace_source import RustTraceSource
        src = RustTraceSource(path=args.trace,
                              tick_rate=1.0 / SRC_DT,
                              chaos=args.chaos)
    else:
        src = LorenzSource(dt=SRC_DT)
        if args.chaos in ("combat", "combat5"):
            src = ChaosInjector(
                src, seed=1, noise_sigma=0.5, noise_prob=0.10,
                drop_prob=0.02, nan_prob=0.01, spike_prob=0.01,
                spike_scale=1e4)
            if args.chaos == "combat":
                src.set_dropout(t_start=2.5, duration=1.0)
            else:
                src.set_dropout(t_start=2.0, duration=5.0)
    encoder = FeatureEncoder(state_dim=3, proj_dim=8, seed=0)
    engine = NengoDynamics(feature_dim=encoder.feature_dim,
                           n_neurons=args.neurons, dt=DT,
                           recurrent_gain=args.recurrent_gain, seed=0)
    predictor = NLMSPredictor(n_in=2 * encoder.feature_dim, n_out=3,
                              mu=args.mu)
    return src, encoder, engine, predictor


def ingest_chunk(src, aligner, encoder, step0: int, n: int):
    """Advance source, push samples, align & encode n control steps."""
    zs, feats, confs, flags = [], [], [], []
    for k in range(n):
        t = (step0 + k) * DT
        src.advance_to(t + LOOKAHEAD)
        for ts, vec in src.read_pending():
            if vec is not None:
                aligner.push(ts, vec)
        raw, conf, fl = aligner.at(t)
        encoder.observe(raw, weight=conf)
        z, feat = encoder.encode(raw)
        zs.append(z)
        feats.append(feat)
        confs.append(conf)
        flags.append(fl)
    return (np.asarray(zs), np.asarray(feats),
            np.asarray(confs), np.asarray(flags, dtype=np.int64))


def make_snapshot_bytes(step, sim_t, encoder, predictor, observer,
                        history, lag, dual, breaker, compressor,
                        warm_feats=None):
    """Build a snapshot blob that includes Phase IV.b module state."""
    extras = {
        "lag": lag,
        "dual": dual.state_dict(),
        "breaker": breaker.state_dict(),
        "compressor": compressor.state_dict(),
    }
    return make_snapshot(step, sim_t, encoder, predictor, observer,
                         history, extras=extras,
                         warm_feats=warm_feats)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--neurons", type=int, default=400)
    ap.add_argument("--recurrent-gain", type=float, default=0.5)
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--chaos", choices=["none", "combat", "combat5"],
                    default="none")
    ap.add_argument("--source", choices=["lorenz", "rust"], default="lorenz",
                    help="state source: python Lorenz integrator or the "
                         "Rust binary trace engine (T5.2)")
    ap.add_argument("--trace", default=None,
                    help="trace file for --source rust (generated if absent)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--feed", default=None,
                    help="POST live frames to a MORPHOS Studio hub, e.g. "
                         "http://localhost:8787 (T5.3)")
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    wal_path = os.path.join(args.data_dir, "trajectory.wal")
    db_path = os.path.join(args.data_dir, "morphos.db")

    src, encoder, engine, predictor = build_components(args)
    aligner = TimeAligner(dt=DT, dim=3)
    # Phase I observer (health metrics only; tripping is delegated)
    health_observer = CircuitBreaker()
    # Phase IV.b: dampened breaker wraps health observer
    breaker = make_dampened_breaker(health=health_observer)
    # Phase IV.b: dual-opinion anomaly detector
    dual = make_dual_opinion(state_dim=3)
    # Phase IV.b: importance-sampled memory compressor
    compressor = make_compressor(encoder.feature_dim,
                                 max_entries=COMPRESSOR_MAX)
    history = TrajectoryBuffer(capacity=1024)
    store = Store(db_path)
    wal = WALWriter(wal_path)
    feed = StudioFeed(args.feed) if args.feed else None

    start_step = 0
    lag = None
    feat_hist: deque[np.ndarray] = deque(maxlen=WARM_K)
    dual_calibration_done = False
    calib_resid: list[float] = []
    calib_nov: list[float] = []
    if args.resume:
        snap = store.load_snapshot("latest")
        if snap is not None:
            payload = load_snapshot(snap["blob"])
            encoder.load_state_dict(payload["encoder"])
            predictor.load_state_dict(payload["predictor"])
            health_observer.load_state_dict(payload["observer"])
            start_step = payload["step"]
            lag = int(payload.get("extras", {}).get("lag", 1))
            # Restore Phase IV.b state
            extras = payload.get("extras", {})
            if "dual" in extras:
                dual.load_state_dict(extras["dual"])
                dual_calibration_done = dual._calibrated
            if "breaker" in extras:
                breaker.load_state_dict(extras["breaker"])
            if "compressor" in extras:
                compressor.load_state_dict(extras["compressor"])
            warm = np.asarray(payload.get("warm_feats", []), dtype=float)
            if warm.size:
                warm = warm.reshape(-1, encoder.feature_dim)
                engine.run_chunk(warm)
            store.log_event("INFO", f"resumed from step {start_step}",
                            sim_t=start_step * DT,
                            data={"snapshot_step": snap["step"]})
            print(f"[resume] loaded snapshot at step {start_step}")
        else:
            print("[resume] no snapshot found, starting fresh")

    run_id = store.begin_run(note=f"steps={args.steps} chaos={args.chaos}")
    store.log_event("INFO", "run started",
                    data={"steps": args.steps, "chaos": args.chaos,
                          "neurons": args.neurons, "dt": DT})

    # global histories keyed by global step (resume-safe)
    all_z: dict[int, np.ndarray] = {}
    all_conf: dict[int, float] = {}
    all_flags: dict[int, int] = {}
    all_dec: list[np.ndarray] = []
    all_rate: list[float] = []
    dec_base = start_step
    pred_ptr = start_step
    breaker_recent = 0

    t_wall0 = time.perf_counter()
    total_wall = 0.0
    step_latency = []
    nan_decoded_total = 0
    adaptive_mem_total = 0

    for chunk_i, S in enumerate(range(start_step, args.steps, args.chunk)):
        n = min(args.chunk, args.steps - S)
        t_chunk = time.perf_counter()

        zs, feats, confs, flags = ingest_chunk(src, aligner, encoder,
                                               S, n)
        for k in range(n):
            all_z[S + k] = zs[k]
            all_conf[S + k] = float(confs[k])
            all_flags[S + k] = int(flags[k])

        # ---- Phase IV.b: seed dual-opinion memory with first N states ----
        if not dual_calibration_done:
            for k in range(n):
                dual.add_reference(zs[k])

        # ---- Phase IV.b: dampened input drive during recovery ------------
        if breaker.active:
            feats = feats * breaker.input_gain()

        cres = engine.run_chunk(feats)
        if lag is None or chunk_i == 0:
            lag = engine.estimate_lag(feats, cres["decoded"])
            store.log_event("INFO", f"estimated input->decoded lag={lag}",
                            sim_t=S * DT)

        # ---- circuit breaker (Phase I health + Phase IV.b dampened) ------
        stats = {
            "mean_rate": float(np.mean(cres["rate_mean"])),
            "max_rate": float(np.max(cres["rate_max"])),
            "spike_frac": float(np.mean(cres["spike_frac"])),
            "nan_decoded": cres["nan_decoded"],
            "conf_mean": float(np.mean(confs)),
        }
        event = breaker.evaluate(stats)
        tripped = False
        if event and event.startswith("TRIP"):
            tripped = True
            breaker_recent = WARM_K + n
            store.log_event("ERROR", f"dampened breaker TRIP: {event}",
                            sim_t=S * DT, data=stats)
            print(f"[breaker] TRIP at step {S}: {event} -> dampened recovery")
            ref = (all_z.get(S, np.zeros(3, dtype=float))
                   if S in all_z else None)
            breaker.trip(S, reference=ref)
        elif event:
            store.log_event("WARN", event, sim_t=S * DT, data=stats)

        all_dec.extend(list(cres["decoded"]))
        all_rate.extend(cres["rate_mean"].tolist())
        nan_decoded_total += cres["nan_decoded"]
        feat_hist.extend(feats)

        # ---- causal prediction / residual loop ------------------------
        zero_rep = np.zeros(encoder.feature_dim)
        while pred_ptr < args.steps:
            s = pred_ptr
            if s + 1 not in all_z:
                break
            k = s - dec_base + lag
            if k < 0 or k >= len(all_dec):
                break
            rep_s = all_dec[k]
            rep_prev = all_dec[k - 1] if k - 1 >= 0 else zero_rep
            x = np.concatenate([rep_s, rep_prev])
            target = all_z[s + 1]
            conf = all_conf[s]
            pred_z = predictor.predict(x)
            err2 = predictor.update(x, target, weight=conf)

            # ---- Phase IV.b: dual-opinion scoring -----------------------
            breaker.observe_residual(err2)
            z_state = all_z.get(s, np.zeros(3))
            nov = dual.novelty(z_state)
            if dual_calibration_done:
                result = dual.score(
                    residual=err2, novelty=nov, state=z_state)
                is_anomaly = result["is_anomaly"]
                if result["adaptive_mem_added"]:
                    adaptive_mem_total += 1
            else:
                is_anomaly = False
                if s >= start_step + 50:
                    calib_resid.append(err2)
                    calib_nov.append(nov)

            # ---- Phase IV.b: importance compressor ----------------------
            mem_feat = rep_s[:encoder.feature_dim] if len(rep_s) >= \
                encoder.feature_dim else rep_s
            compressor.push(s, mem_feat, residual=err2)

            rate = all_rate[k] if k < len(all_rate) else 0.0
            flags_w = int(all_flags[s])
            if conf < LOW_CONF:
                flags_w |= WAL_FLAG_LOWS
            if breaker_recent > 0 or breaker.active:
                flags_w |= WAL_FLAG_BREAKER
                breaker_recent -= 1
            wal.append((s + 1) * DT, conf, target, pred_z, err2, rate,
                       flags_w)
            if feed is not None:
                feed.push({
                    "step": int(s + 1),
                    "t": round((s + 1) * DT, 6),
                    "conf": round(float(conf), 4),
                    "a": [round(float(v), 5) for v in target],
                    "p": [round(float(v), 5) for v in pred_z],
                    "r": round(float(err2), 6) if np.isfinite(err2) else None,
                    "rate": round(float(rate), 3),
                    "flags": int(flags_w),
                    "low_conf": bool(conf < LOW_CONF),
                    "breaker": bool(breaker_recent > 0 or breaker.active),
                    "anom": ("anomaly" if is_anomaly else None),
                })
            history.push_stats(conf, err2, rate)
            pred_ptr += 1

        if chunk_i == 2:
            health_observer.calibrate_residual_baseline()

        # ---- Phase IV.b: dual-opinion calibration -----------------------
        if (not dual_calibration_done and
                len(calib_resid) >= DUAL_CALIB_STEPS):
            try:
                dual.calibrate(
                    np.asarray(calib_resid, dtype=float),
                    np.asarray(calib_nov, dtype=float))
                dual_calibration_done = True
                store.log_event("INFO",
                                f"dual-opinion calibrated, threshold="
                                f"{dual.threshold:.4f}",
                                sim_t=S * DT)
                print(f"[dual] calibrated at step {S}, "
                      f"threshold={dual.threshold:.4f}")
            except ValueError as exc:
                print(f"[dual] calibration skipped: {exc}")

        # ---- periodic snapshot ----------------------------------------
        if (chunk_i + 1) % SNAP_EVERY == 0 or S + n >= args.steps:
            warm_snap = (np.asarray(list(feat_hist), dtype=float)
                         if feat_hist else None)
            blob = make_snapshot_bytes(
                S + n, (S + n) * DT, encoder, predictor, health_observer,
                history, lag, dual, breaker, compressor,
                warm_feats=warm_snap)
            store.save_snapshot(blob, sim_t=(S + n) * DT, step=S + n,
                                key="latest")

        dt_chunk = time.perf_counter() - t_chunk
        total_wall += dt_chunk
        step_latency.extend([dt_chunk / n] * n)
        if (chunk_i + 1) % 5 == 0 or S + n >= args.steps:
            ema = history.resid_ema(200)
            status = "TRIP" if breaker.active else \
                     health_observer.state
            print(f"[step {S + n:5d}/{args.steps}] "
                  f"resid_ema={ema:.4f} "
                  f"rate={stats['mean_rate']:.0f}Hz "
                  f"conf={stats['conf_mean']:.2f} "
                  f"state={status} "
                  f"trips={health_observer.trip_count} "
                  f"mem_add={adaptive_mem_total} "
                  f"chunk={dt_chunk * 1000 / n:.1f}ms/step")

    if lag is None:
        lag = 1
    if start_step >= args.steps:
        print(f"[resume] snapshot step {start_step} >= --steps "
              f"{args.steps}; nothing to do")
    # ---- final summary -------------------------------------------------
    lat = np.asarray(step_latency) * 1000.0
    comp_stats = compressor.stats()
    summary = {
        "steps": args.steps,
        "sim_seconds": args.steps * DT,
        "mean_step_ms": float(lat.mean()) if lat.size else float("nan"),
        "p95_step_ms": float(np.percentile(lat, 95)) if lat.size
        else float("nan"),
        "max_step_ms": float(lat.max()) if lat.size else float("nan"),
        "resid_ema_last200": history.resid_ema(200),
        "resid_ema_first200": _first_resid_ema(wal_path, 200),
        "mean_rate_hz": float(np.mean(list(history.rates)[-500:]))
        if history.rates else float("nan"),
        "max_rate_hz": float(np.max(list(history.rates)[-500:]))
        if history.rates else float("nan"),
        "nan_decoded_total": int(nan_decoded_total),
        "breaker_trips": health_observer.trip_count,
        "breaker_warns": health_observer.warn_count,
        "dampened_recoveries": breaker.trip_count,
        "wal_records": wal.records,
        "aligner_dropped": aligner.dropped_count,
        "chaos_stats": getattr(src, "stats", None),
        # Phase IV.b stats
        "dual_calibrated": dual_calibration_done,
        "dual_threshold": float(dual.threshold) if dual_calibration_done
        else float("nan"),
        "compressor_total_seen": comp_stats["total_seen"],
        "compressor_reservoir": comp_stats["reservoir"],
        "compressor_kept_fraction":
            float(comp_stats["kept_fraction"]),
        "adaptive_mem_additions": adaptive_mem_total,
    }
    if feed is not None:
        feed.close()
        summary["studio_frames_sent"] = feed.sent
        summary["studio_frames_failed"] = feed.failed
    store.log_event("INFO", "run finished", data=summary)
    store.end_run(args.steps)
    store.set_meta("last_summary", str(summary))
    wal.close()
    store.close()

    print("\n===== MORPHOS Phase IV.b summary =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  wal: {wal_path}")
    print(f"  db : {db_path}")
    return 0


class StudioFeed:
    """Optional live feed to MORPHOS Studio (T5.3).

    Batches frames and POSTs them to the hub's /api/ingest endpoint from a
    background thread. Every send is best-effort: a missing or slow studio
    must never perturb the control loop, so failures are counted and swallowed.
    """

    def __init__(self, url: str, batch: int = 20):
        import threading
        import queue
        self.url = url.rstrip("/") + "/api/ingest"
        self.batch = batch
        self.sent = 0
        self.failed = 0
        self._q: "queue.Queue[dict]" = queue.Queue(maxsize=4096)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True,
                                   name="morphos-studio-feed")
        self._t.start()

    def push(self, frame: dict) -> None:
        try:
            self._q.put_nowait(frame)
        except Exception:
            self.failed += 1

    def _post(self, frames: list[dict]) -> None:
        import json
        import urllib.request
        body = json.dumps({"frames": frames}).encode()
        req = urllib.request.Request(
            self.url, data=body,
            headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                resp.read()
            self.sent += len(frames)
        except Exception:
            self.failed += len(frames)

    def _run(self) -> None:
        batch: list[dict] = []
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.25)
                batch.append(item)
            except Exception:
                pass
            if len(batch) >= self.batch or (batch and self._q.empty()
                                            and self._q.qsize() == 0):
                # flush on size, or when the queue momentarily drains
                if batch:
                    self._post(batch)
                    batch = []
                if self._q.empty():
                    self._stop.wait(0.15)
        if batch:
            self._post(batch)

    def close(self) -> None:
        self._stop.set()
        self._t.join(timeout=2.0)


def _first_resid_ema(wal_path: str, window: int) -> float:
    try:
        from storage.wal import iter_records
        vals = []
        for i, rec in enumerate(iter_records(wal_path)):
            if i >= window:
                break
            if np.isfinite(rec["residual"]):
                vals.append(rec["residual"])
        return float(np.mean(vals)) if vals else float("nan")
    except Exception:
        return float("nan")


if __name__ == "__main__":
    raise SystemExit(main())