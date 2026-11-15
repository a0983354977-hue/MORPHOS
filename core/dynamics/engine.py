"""Nengo dynamics engine.

Wraps a spiking (LIF) neural ensemble with:
  - a Node input fed chunk-wise (amortises simulator overhead)
  - an optional random-orthogonal recurrent transform (reservoir dynamics)
  - decoded_output + neural-output probes for prediction features & health

The simulator is advanced in chunks of `chunk` control steps; per-step
Python overhead stays ~0.1 ms/step instead of ~100 ms/step.
"""

from __future__ import annotations

import numpy as np
import nengo

from core.memory.thread_safe_buffer import ThreadSafeBuffer, synchronized


class NengoDynamics(ThreadSafeBuffer):
    def __init__(self, feature_dim: int, n_neurons: int = 400,
                 dt: float = 0.005, recurrent_gain: float = 0.5,
                 seed: int = 0, synapse: float | None = None,
                 recurrent_synapse: float | None = None):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_neurons = n_neurons
        self.dt = dt
        self.recurrent_gain = recurrent_gain
        self.seed = seed
        self.synapse = dt if synapse is None else synapse
        # Intrinsic recurrent time constant of the reservoir.  Multi-scale
        # layers set this to their own tau so a slow layer really does
        # integrate over a longer window.
        self.recurrent_synapse = (0.02 if recurrent_synapse is None
                                  else recurrent_synapse)

        self._inputs = np.zeros((1, feature_dim))
        self._t0 = 0.0
        self._dec_offset = 0
        self._rate_offset = 0
        self._steps_run = 0
        self._build()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        super().__init__()

    # ------------------------------------------------------------------
    @synchronized
    def _build(self) -> None:
        rng = np.random.default_rng(self.seed)
        net = nengo.Network(seed=self.seed)
        with net:
            self._input_node = nengo.Node(self._input_fn,
                                          size_out=self.feature_dim)
            self.ens = nengo.Ensemble(
                self.n_neurons, dimensions=self.feature_dim,
                neuron_type=nengo.LIF(), seed=self.seed)
            nengo.Connection(self._input_node, self.ens)
            if self.recurrent_gain > 0.0:
                q, _ = np.linalg.qr(rng.standard_normal(
                    (self.feature_dim, self.feature_dim)))
                transform = self.recurrent_gain * q
                nengo.Connection(self.ens, self.ens, transform=transform,
                                 synapse=self.recurrent_synapse)
            self._probe_dec = nengo.Probe(self.ens, "decoded_output",
                                          synapse=self.synapse)
            self._probe_rate = nengo.Probe(self.ens.neurons, "output")
        self.sim = nengo.Simulator(net, dt=self.dt, seed=self.seed,
                                   progress_bar=False)
        nt = self.ens.neuron_type
        amp = getattr(nt, "amplitude", 1.0)
        self._amplitude = float(np.mean(np.asarray(amp, dtype=float)))
        self._dec_offset = 0
        self._rate_offset = 0

    def _input_fn(self, t: float) -> np.ndarray:
        idx = int(round((t - self._t0) / self.dt))
        idx = min(max(idx, 0), self._inputs.shape[0] - 1)
        return self._inputs[idx]

    # ------------------------------------------------------------------
    @synchronized
    def run_chunk(self, feats: np.ndarray) -> dict:
        """Advance the network over `feats` (n_steps, feature_dim).

        Returns per-step decoded features and firing-rate statistics.
        """
        feats = np.asarray(feats, dtype=float)
        n = feats.shape[0]
        self._inputs = feats
        self._t0 = float(self.sim.time)
        self.sim.run(n * self.dt)

        dec_all = np.asarray(self.sim.data[self._probe_dec])
        rate_all = np.asarray(self.sim.data[self._probe_rate])
        dec = dec_all[self._dec_offset:self._dec_offset + n]
        out = rate_all[self._rate_offset:self._rate_offset + n]
        self._dec_offset += n
        self._rate_offset += n
        self._steps_run += n

        # firing rate (Hz) per step: output = (amplitude/dt)*n_spikes
        rates = out / max(self._amplitude, 1e-9)
        rate_mean = rates.mean(axis=1) if rates.size else np.zeros(n)
        rate_max = rates.max(axis=1) if rates.size else np.zeros(n)
        spike_frac = (out > 0).mean(axis=1) if out.size else np.zeros(n)

        return {
            "decoded": dec,
            "rate_mean": rate_mean,
            "rate_max": rate_max,
            "spike_frac": spike_frac,
            "nan_decoded": int(np.sum(~np.isfinite(dec))),
        }

    def estimate_lag(self, feats: np.ndarray, decoded: np.ndarray,
                     max_lag: int = 4) -> int:
        """Estimate how many steps `decoded` lags `feats` (0..max_lag)."""
        n = min(feats.shape[0], decoded.shape[0])
        if n < 8:
            return 1
        best_lag, best_score = 1, -np.inf
        f = feats[:n]
        d = decoded[:n]
        for lag in range(max_lag + 1):
            if lag == 0:
                a, b = d, f
            else:
                a, b = d[lag:], f[:n - lag]
            if a.shape[0] < 8:
                continue
            # correlate first differences: raw signals are smooth, so
            # plain correlation cannot resolve a 1-2 step lag
            fa = np.diff(a, axis=0)
            fb = np.diff(b, axis=0)
            fa = fa - fa.mean(axis=0, keepdims=True)
            fb = fb - fb.mean(axis=0, keepdims=True)
            num = np.sum(fa * fb)
            den = np.sqrt(np.sum(fa * fa) * np.sum(fb * fb)) + 1e-12
            score = num / den
            if score > best_score:
                best_score, best_lag = score, lag
        return best_lag

    @synchronized
    def reset(self, warm_feats: np.ndarray | None = None) -> None:
        """Rebuild the simulator (circuit-breaker reset).

        Optionally warm-start by replaying recent inputs so membrane
        state is not cold.
        """
        self._build()
        if warm_feats is not None and len(warm_feats) > 0:
            self.run_chunk(np.asarray(warm_feats, dtype=float))

    @property
    @synchronized
    def steps_run(self) -> int:
        return self._steps_run
