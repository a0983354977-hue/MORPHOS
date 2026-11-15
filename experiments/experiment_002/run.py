"""Experiment 002 - Heterogeneous DSW ablation study (T2.4).

Feeds two physically different streams into one neural dynamical core:
  - Lorenz-63 chaotic mathematics source
  - live OS resource source (psutil: cpu/mem/disk/net)

Ablation arms:
  orthogonal_clean      - orthogonal projector, no anomaly (baseline)
  orthogonal_anomaly    - orthogonal projector + system-source anomaly
  interference_anomaly  - interference projector + same anomaly

Hypothesis: with orthogonal subspace projection, a violent mutation in
the system-source stream must NOT degrade the Lorenz residual (and vice
versa).  With the ablated dense projection, cross-source interference
should inflate the unaffected source's residual.

Run:  python experiments/experiment_002/run.py [--steps 1200]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from adapters.chaos import ChaosInjector
from adapters.lorenz import LorenzSource
from adapters.system import SystemSource
from core.dynamics.weave import WeaveEngine
from core.encoding.encoder import FeatureEncoder
from core.encoding.projector import SubspaceProjector
from core.ingest.ring_buffer import TimeAligner
from core.observer.observer import CircuitBreaker
from core.prediction.predictor import NLMSPredictor

DT = 0.005
SRC_DT = 0.001
LOOKAHEAD = 0.002
CHUNK = 200
NEURONS = 700
PROJ_DIM = 16
ANOM_T0, ANOM_T1 = 3.0, 4.0      # 1 s anomaly window (sim time)
ANOM_VEC = np.array([1.0, 1.0, 1.0, 1.0])  # saturated system state


class AnomalousSystemSource(SystemSource):
    """SystemSource with a programmable anomaly window."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.anom = None  # (t0, t1, vec)

    def set_anomaly(self, t0, t1, vec):
        self.anom = (t0, t1, np.asarray(vec, dtype=float))

    def sample(self):
        if self.anom is not None:
            t0, t1, vec = self.anom
            # aligner timestamps track sim time via advance_to; use t
            if t0 <= self.t <= t1:
                return vec.copy()
        return super().sample()


def run_arm(mode: str, anomaly: bool, steps: int,
            out_rows: list[dict]) -> dict:
    # ---- sources -------------------------------------------------------
    lorenz = ChaosInjector(LorenzSource(dt=SRC_DT), seed=1,
                           noise_sigma=0.3, noise_prob=0.05,
                           drop_prob=0.01, nan_prob=0.005,
                           spike_prob=0.005, spike_scale=1e4)
    system = AnomalousSystemSource()
    if anomaly:
        system.set_anomaly(ANOM_T0, ANOM_T1, ANOM_VEC)

    # ---- per-source z-score encoders (projection unused) --------------
    enc_l = FeatureEncoder(state_dim=3, proj_dim=1, seed=0)
    enc_s = FeatureEncoder(state_dim=4, proj_dim=1, seed=0)

    # ---- aligners -------------------------------------------------------
    al_l = TimeAligner(dt=DT, dim=3)
    al_s = TimeAligner(dt=DT, dim=4)

    # ---- orthogonal / interference projector ---------------------------
    projector = SubspaceProjector({"lorenz": 3, "system": 4},
                                  proj_dim=PROJ_DIM, mode=mode, seed=0)

    # ---- neural weave + joint predictor ---------------------------------
    weave = WeaveEngine(feature_dim=projector.feature_dim,
                        n_neurons=NEURONS, dt=DT,
                        recurrent_gain=0.5, seed=0)
    n_out = 3 + 4
    pred = NLMSPredictor(n_in=2 * projector.feature_dim, n_out=n_out,
                         mu=0.05)
    breaker = CircuitBreaker()

    # ---- global causal histories ----------------------------------------
    z_l_map: dict[int, np.ndarray] = {}
    z_s_map: dict[int, np.ndarray] = {}
    conf_l_map: dict[int, float] = {}
    conf_s_map: dict[int, float] = {}
    all_dec: list[np.ndarray] = []
    dec_base = 0
    lag = None
    pred_ptr = 0
    warm_feats: list[np.ndarray] = []

    t_wall = time.perf_counter()
    latencies = []

    for S in range(0, steps, CHUNK):
        n = min(CHUNK, steps - S)
        tc = time.perf_counter()

        zs_l, zs_s, feats_in = [], [], []
        # psutil syscalls cost ~3.5 ms each on Windows - far beyond the
        # 1 ms/step budget.  Sample OS metrics once per chunk (~1 s sim /
        # ~0.2 s wall) and hold the reading across the chunk; the aligner
        # still sees a sample on every grid step.
        system.t = (S + n / 2) * DT
        sys_vec = system.sample()
        for k in range(n):
            t = (S + k) * DT
            # lorenz stream
            lorenz.advance_to(t + LOOKAHEAD)
            for ts, vec in lorenz.read_pending():
                if vec is not None:
                    al_l.push(ts, vec)
            raw_l, conf_l, _ = al_l.at(t)
            enc_l.observe(raw_l, weight=conf_l)
            z_l = enc_l.encode_z(raw_l)
            # system stream (chunk-rate sensor sample, held per step)
            al_s.push(t, sys_vec)
            raw_s, conf_s, _ = al_s.at(t)
            enc_s.observe(raw_s, weight=conf_s)
            z_s = enc_s.encode_z(raw_s)

            zs_l.append(z_l)
            zs_s.append(z_s)
            conf_l_map[S + k] = conf_l
            conf_s_map[S + k] = conf_s
            feats_in.append(projector.features(
                {"lorenz": z_l, "system": z_s}))

        for k in range(n):
            z_l_map[S + k] = zs_l[k]
            z_s_map[S + k] = zs_s[k]

        feats = np.asarray(feats_in)
        cres = weave.run_chunk(feats)
        if lag is None:
            lag = weave.calibrate(feats, cres["decoded"])

        # breaker (safety only - experiments expect stability)
        stats = {"mean_rate": float(np.mean(cres["rate_mean"])),
                 "max_rate": float(np.max(cres["rate_max"])),
                 "spike_frac": float(np.mean(cres["spike_frac"])),
                 "nan_decoded": cres["nan_decoded"],
                 "conf_mean": float(np.mean([conf_l_map[S],
                                             conf_s_map[S]]))}
        if breaker.evaluate(stats) and breaker.state == "TRIP":
            warm = np.asarray(warm_feats[-100:], dtype=float) \
                if warm_feats else None
            kwarm = len(warm) if warm is not None else 0
            weave.reset(warm)
            keep = max(0, (S - kwarm) - dec_base)
            del all_dec[keep:]
            dec_base = S - kwarm
            if warm is not None:
                wres = weave.run_chunk(warm)
                all_dec.extend(list(wres["decoded"]))
            cres = weave.run_chunk(feats)

        all_dec.extend(list(cres["decoded"]))
        warm_feats.extend(feats)
        latencies.append((time.perf_counter() - tc) / n)

        # ---- causal joint prediction ------------------------------------
        zero = np.zeros(projector.feature_dim)
        while pred_ptr + 1 < steps and (pred_ptr + 1) in z_l_map:
            s = pred_ptr
            k = s - dec_base + lag
            if k < 0 or k >= len(all_dec):
                break
            x = np.concatenate([all_dec[k],
                                all_dec[k - 1] if k - 1 >= 0 else zero])
            target = np.concatenate([z_l_map[s + 1], z_s_map[s + 1]])
            conf = min(conf_l_map[s], conf_s_map[s])
            pred_vec = pred.predict(x)
            err2 = pred.update(x, target, weight=conf)
            # per-source squared residuals (pre-update prediction)
            e_l = float(np.sum((target[:3] - pred_vec[:3]) ** 2))
            e_s = float(np.sum((target[3:] - pred_vec[3:]) ** 2))
            out_rows.append({
                "arm": f"{mode}_{'anom' if anomaly else 'clean'}",
                "t": (s + 1) * DT, "s": s,
                "e_lorenz": e_l, "e_system": e_s,
                "conf_l": conf_l_map[s], "conf_s": conf_s_map[s],
                "anom": int(ANOM_T0 <= (s + 1) * DT <= ANOM_T1)
                if anomaly else 0,
            })
            pred_ptr += 1

    wall = time.perf_counter() - t_wall
    return {
        "arm": f"{mode}_{'anom' if anomaly else 'clean'}",
        "mode": mode, "anomaly": anomaly, "steps": steps,
        "wall_s": wall, "mean_step_ms": float(np.mean(latencies) * 1000),
        "lag": lag, "breaker_trips": breaker.trip_count,
    }


def summarise(rows: list[dict], meta: list[dict]) -> dict:
    """Per-arm residual stats + anomaly-window isolation metrics."""
    report = {"arms": {}, "meta": meta}

    # same-window means for every arm (clean arm supplies the reference
    # trajectory; learning-curve drift is cancelled by this comparison)
    win = {}
    for m in meta:
        arm = m["arm"]
        rs = [r for r in rows if r["arm"] == arm]
        t = np.array([r["t"] for r in rs])
        w = (t >= ANOM_T0) & (t <= ANOM_T1)
        win[arm] = {
            "e_lorenz": float(np.mean([r["e_lorenz"] for r, m2 in
                                       zip(rs, w) if m2])),
            "e_system": float(np.mean([r["e_system"] for r, m2 in
                                       zip(rs, w) if m2])),
        }
    clean_arm = next((m["arm"] for m in meta
                      if m["mode"] == "orthogonal" and not m["anomaly"]),
                     None)

    for m in meta:
        arm = m["arm"]
        rs = [r for r in rows if r["arm"] == arm]
        e_l = np.array([r["e_lorenz"] for r in rs])
        e_s = np.array([r["e_system"] for r in rs])
        anom = np.array([r["anom"] for r in rs], dtype=bool)
        anom_l = e_l[anom] if anom.any() else np.array([])
        anom_s = e_s[anom] if anom.any() else np.array([])
        ts = np.array([r["t"] for r in rs])
        pre_l = e_l[ts < ANOM_T0][-200:]
        pre_s = e_s[ts < ANOM_T0][-200:]
        post_l = e_l[ts > ANOM_T1][:200]
        post_s = e_s[ts > ANOM_T1][:200]
        pre_l_m = float(np.mean(pre_l)) if len(pre_l) else float("nan")
        pre_s_m = float(np.mean(pre_s)) if len(pre_s) else float("nan")
        report["arms"][arm] = {
            "e_lorenz_mean": float(np.mean(e_l)),
            "e_lorenz_last200": float(np.mean(e_l[-200:])),
            "e_system_mean": float(np.mean(e_s)),
            "e_system_last200": float(np.mean(e_s[-200:])),
            "pre_anom_e_lorenz": pre_l_m,
            "pre_anom_e_system": pre_s_m,
            "post_anom_e_lorenz": float(np.mean(post_l))
            if len(post_l) else None,
            "post_anom_e_system": float(np.mean(post_s))
            if len(post_s) else None,
            "anom_e_lorenz_mean": float(np.mean(anom_l))
            if len(anom_l) else None,
            "anom_e_system_mean": float(np.mean(anom_s))
            if len(anom_s) else None,
            "anom_lorenz_ratio": float(np.mean(anom_l) / pre_l_m)
            if len(anom_l) and np.isfinite(pre_l_m) and pre_l_m > 0
            else None,
            "anom_system_ratio": float(np.mean(anom_s) / pre_s_m)
            if len(anom_s) and np.isfinite(pre_s_m) and pre_s_m > 0
            else None,
        }
        # excess residual vs the clean arm over the SAME time window:
        # isolates anomaly-driven cross-talk from the learning curve
        if clean_arm and arm != clean_arm:
            ref = win[clean_arm]
            cur = win[arm]
            report["arms"][arm]["excess_e_lorenz"] = \
                cur["e_lorenz"] - ref["e_lorenz"]
            report["arms"][arm]["excess_e_system"] = \
                cur["e_system"] - ref["e_system"]
            report["arms"][arm]["cross_talk_ratio"] = \
                (cur["e_lorenz"] / ref["e_lorenz"]
                 if ref["e_lorenz"] > 0 else None)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "experiments", "experiment_002", "results"))
    args = ap.parse_args()

    rows: list[dict] = []
    meta = []
    arms = [
        ("orthogonal", False),
        ("orthogonal", True),
        ("interference", True),
    ]
    for mode, anomaly in arms:
        print(f"[exp002] running arm mode={mode} anomaly={anomaly} "
              f"steps={args.steps} neurons={NEURONS}")
        m = run_arm(mode, anomaly, args.steps, rows)
        meta.append(m)
        print(f"[exp002]   done wall={m['wall_s']:.1f}s "
              f"step={m['mean_step_ms']:.2f}ms lag={m['lag']} "
              f"trips={m['breaker_trips']}")

    report = summarise(rows, meta)
    os.makedirs(args.out, exist_ok=True)
    report_path = os.path.join(args.out, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # compact residual csv
    csv_path = os.path.join(args.out, "residuals.csv")
    with open(csv_path, "w") as f:
        f.write("arm,t,e_lorenz,e_system,conf_l,conf_s,anom\n")
        for r in rows:
            f.write(f"{r['arm']},{r['t']:.3f},{r['e_lorenz']:.6g},"
                    f"{r['e_system']:.6g},{r['conf_l']:.3f},"
                    f"{r['conf_s']:.3f},{r['anom']}\n")

    print("\n===== Experiment 002 report =====")
    print(json.dumps(report["arms"], indent=2))
    print(f"meta: {json.dumps(meta, indent=2)}")
    print(f"saved: {report_path}\n       {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
