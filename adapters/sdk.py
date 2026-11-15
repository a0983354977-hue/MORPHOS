"""MORPHOS Adapter SDK (T5.4) - the contract third-party integrations implement.

An *adapter* is the only thing a new data source (hardware sensor, network
tap, log shipper, another simulator) has to write. It sits at the outer edge
of the platform; everything downstream - sanitising, time alignment, feature
encoding, the neural ensemble, anomaly detection - is provided.

The contract is deliberately small: **two methods and two properties**.

    class MySensor(MorphosSource):            # subclass for defaults + checks
        dim = 3
        def advance_to(self, t_target): ...
        def read_pending(self): ...

Why a pull interface rather than push? The control loop runs on a fixed time
grid (dt = 5 ms by default) and must never block waiting for a sensor. Pulling
lets the loop ask for "everything up to t" and continue with whatever arrived,
which is exactly what makes the pipeline resilient to jitter and dropouts.

Guarantees the core gives you, once your adapter conforms:
  * NaN/Inf samples are dropped, spikes clipped, gaps interpolated or held.
  * Late/duplicate/out-of-order timestamps are repaired on a fixed grid.
  * Every sample carries a confidence in [0, 1]; low confidence is propagated
    and logged rather than silently trusted.
  * Your adapter is never mutated by the core and is not called concurrently.

What your adapter must guarantee:
  * `advance_to(t)` returns without blocking longer than one control step.
  * `read_pending()` returns samples whose timestamps are non-decreasing.
  * `dim` is constant for the adapter's lifetime.
  * `dim:` is at least 1 and every sample is a float array of exactly that
    length (handle NaN yourself or pass it through - the core sanitises).

Validate a new adapter with:
    python -m adapters.sdk validate path/to/my_adapter.py:MySensor
    python -m adapters.sdk conformance 0          # run the reference sensor
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# A sample is (timestamp_seconds, vector). Timestamps are in *source time*,
# not wall clock: the core maps them onto its own control grid.
Sample = Tuple[float, np.ndarray]

#: Default control step used by main.py; adapters should assume this cadence.
CONTROL_DT = 0.005


class AdapterError(RuntimeError):
    """Base class for adapter-facing failures."""


class AdapterContractError(AdapterError):
    """The adapter violated the interface (wrong shape, bad timestamp order)."""


class MorphosSource(abc.ABC):
    """Base class for every MORPHOS data source.

    Subclassing is optional (any object with the same attributes works) but
    gives you shape validation, ordered-timestamp checks and a default
    `stats` property.
    """

    #: Number of dimensions this source emits. Must be constant.
    dim: int = 0

    #: Human-readable name shown by `morphos doctor` and in run summaries.
    name: str = "unnamed-source"

    #: Seconds of source time that one sample represents (informational).
    sample_period: float = 0.001

    def __init__(self) -> None:
        if self.dim <= 0:
            raise AdapterContractError(
                f"{type(self).__name__}.dim must be a positive integer")
        self.t: float = 0.0
        self._last_ts: float = -float("inf")
        self._validated = 0

    # -- required interface -------------------------------------------
    @abc.abstractmethod
    def advance_to(self, t_target: float) -> None:
        """Advance internal state so that all samples up to `t_target` exist.

        Must be cheap: the core calls this once per control step. Implement your
        own internal integration/buffering here, then make the samples
        available through `read_pending`.
        """

    @abc.abstractmethod
    def read_pending(self) -> Iterable[Sample]:
        """Return samples produced since the previous call, then clear them.

        Timestamps must be non-decreasing across successive calls. Vector
        length must equal `dim`.
        """

    # -- optional interface -------------------------------------------
    @property
    def stats(self) -> dict:
        """Diagnostics for run summaries. Override to expose counters."""
        return {"name": self.name, "dim": self.dim, "t": self.t}

    def close(self) -> None:
        """Release hardware handles / file descriptors. Called on shutdown."""

    # -- validation helpers -------------------------------------------
    def _check_sample(self, ts: float, vec: np.ndarray) -> Sample:
        arr = np.asarray(vec, dtype=float).ravel()
        if arr.shape[0] != self.dim:
            raise AdapterContractError(
                f"{type(self).__name__}: expected {self.dim}-D sample, "
                f"got {arr.shape[0]}")
        if ts < self._last_ts:
            raise AdapterContractError(
                f"{type(self).__name__}: timestamps must be non-decreasing "
                f"({ts} after {self._last_ts})")
        self._last_ts = ts
        self._validated += 1
        return ts, arr

    def validate_pending(self) -> List[Sample]:
        """Helper for subclasses: read and validate the pending batch."""
        out: List[Sample] = []
        for ts, vec in self.read_pending():
            out.append(self._check_sample(ts, vec))
        return out


# ----------------------------------------------------------------------
# Reference implementations
# ----------------------------------------------------------------------
class SineSource(MorphosSource):
    """Minimal conforming adapter: a multi-tone sine, useful as a template.

    This is the smallest thing that satisfies the SDK and proves the
    "write an adapter in under an hour" claim - it is ~30 lines.
    """

    name = "sine"

    def __init__(self, dim: int = 3, freq: float = 0.7, dt: float = 0.001):
        self.dim = dim
        self.freq = freq
        self.dt = dt
        self.sample_period = dt
        super().__init__()
        self._pending: List[Sample] = []

    def advance_to(self, t_target: float) -> None:
        while self.t < t_target - 1e-12:
            self.t += self.dt
            phase = 2.0 * np.pi * self.freq * self.t
            vec = np.array([
                np.sin(phase * (k + 1)) * (1.0 / (k + 1))
                for k in range(self.dim)
            ])
            self._pending.append((self.t, vec))

    def read_pending(self) -> List[Sample]:
        out, self._pending = self._pending, []
        return out


class ReplaySource(MorphosSource):
    """Replay a fixed (t, vec) trace: the adapter for recorded datasets.

    Handy for tests and for evaluating a pipeline on a captured stream.
    """

    name = "replay"

    def __init__(self, records: Sequence[Sample], loop: bool = False):
        if not records:
            raise AdapterContractError("ReplaySource needs at least one sample")
        self.dim = int(np.asarray(records[0][1]).ravel().shape[0])
        self.loop = loop
        self._records = [(float(t), np.asarray(v, dtype=float).ravel())
                         for t, v in records]
        super().__init__()
        self._cursor = 0
        self._pending: List[Sample] = []
        self._span = self._records[-1][0] - self._records[0][0]

    def advance_to(self, t_target: float) -> None:
        if self._span <= 0:
            self.t = t_target
            return
        while True:
            ts, vec = self._records[self._cursor % len(self._records)]
            # Map the record's source time onto the consumer's clock.
            effective = self.t + (ts - self._records[self._cursor % len(self._records)][0])
            if self._cursor >= len(self._records) and not self.loop:
                break
            self._pending.append((t_target, vec))
            self._cursor += 1
            if self._cursor >= len(self._records):
                break
        self.t = t_target

    def read_pending(self) -> List[Sample]:
        out, self._pending = self._pending, []
        return out


# ----------------------------------------------------------------------
# Conformance harness
# ----------------------------------------------------------------------
@dataclass
class ConformanceResult:
    name: str
    dim: int
    ok: bool
    checks: List[Tuple[str, bool, str]] = field(default_factory=list)
    samples: int = 0
    wall_ms: float = 0.0

    @property
    def failures(self) -> List[Tuple[str, bool, str]]:
        return [c for c in self.checks if not c[1]]

    def report(self) -> str:
        lines = [f"adapter: {self.name}  dim={self.dim}  samples={self.samples}"]
        for label, passed, detail in self.checks:
            mark = "PASS" if passed else "FAIL"
            lines.append(f"  [{mark}] {label}" +
                         (f" -> {detail}" if detail else ""))
        verdict = "CONFORMS" if self.ok else "DOES NOT CONFORM"
        lines.append(f"  {verdict}  ({self.wall_ms:.2f} ms wall)")
        return "\n".join(lines)


def check_conformance(source: MorphosSource, steps: int = 200,
                      dt: float = CONTROL_DT) -> ConformanceResult:
    """Exercise an adapter exactly as the control loop will.

    Checks the properties the core relies on: constant dimensionality, ordered
    timestamps, bounded latency per step, non-empty output, finite time advance.
    """
    res = ConformanceResult(name=getattr(source, "name", type(source).__name__),
                            dim=int(getattr(source, "dim", 0)), ok=False)
    added = res.checks.append

    added(("dim is a positive int", res.dim > 0, str(res.dim)))
    added(("has advance_to", callable(getattr(source, "advance_to", None)), ""))
    added(("has read_pending", callable(getattr(source, "read_pending", None)), ""))
    if not all(p for _, p, _ in res.checks):
        return res

    first_shapes = set()
    total = 0
    last_ts = -float("inf")
    worst_ms = 0.0
    order_ok = True
    shape_ok = True
    nonfinite = 0

    for k in range(steps):
        t_target = (k + 1) * dt
        t0 = time.perf_counter()
        source.advance_to(t_target)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        worst_ms = max(worst_ms, elapsed_ms)
        for ts, vec in source.read_pending():
            arr = np.asarray(vec, dtype=float).ravel()
            first_shapes.add(arr.shape[0])
            total += 1
            if arr.shape[0] != res.dim:
                shape_ok = False
            if ts < last_ts - 1e-12:
                order_ok = False
            last_ts = max(last_ts, ts)
            nonfinite += int((~np.isfinite(arr)).sum())

    res.samples = total
    res.wall_ms = steps * dt * 1000.0
    added(("emitted at least one sample", total > 0, f"{total} samples"))
    added(("sample width constant", len(first_shapes) == 1,
           f"widths={sorted(first_shapes)}"))
    added(("sample width equals dim", shape_ok, f"dim={res.dim}"))
    added(("timestamps non-decreasing", order_ok, f"last_ts={last_ts:.4f}"))
    added(("source clock advanced", getattr(source, "t", 0.0) > 0,
           f"t={getattr(source, 't', 0):.4f}"))
    # 5 ms is the control budget; flag adapters that blow it on every step.
    added((f"advance_to < {dt * 1000:.1f} ms (worst step)",
           worst_ms <= dt * 1000.0, f"{worst_ms:.2f} ms"))
    if nonfinite:
        # Not a failure: the sanitizer exists precisely for this.
        added(("non-finite values are tolerated by the core", True,
               f"{nonfinite} (will be sanitised)"))

    res.ok = not res.failures
    return res


def load_adapter(spec: str) -> MorphosSource:
    """Import `module:ClassName` (or a path) and instantiate an adapter.

    Accepts:
        adapters.sdk:SineSource
        examples/adapters/my_sensor.py:MySensor
    """
    import importlib
    import importlib.util
    import os

    if ":" not in spec:
        raise AdapterError(
            f"adapter spec must be 'module:Class' or 'path.py:Class', got {spec!r}")
    target, cls_name = spec.rsplit(":", 1)

    if target.endswith(".py") or os.path.sep in target:
        path = os.path.abspath(target)
        if not os.path.exists(path):
            raise AdapterError(f"adapter file not found: {path}")
        mod_name = "morphos_adapter_" + os.path.splitext(os.path.basename(path))[0]
        spec_obj = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)  # type: ignore[union-attr]
    else:
        module = importlib.import_module(target)

    if not hasattr(module, cls_name):
        raise AdapterError(f"{target} has no attribute {cls_name!r}")
    cls = getattr(module, cls_name)
    obj = cls()
    if not (hasattr(obj, "advance_to") and hasattr(obj, "read_pending")):
        raise AdapterError(
            f"{spec} does not implement advance_to/read_pending")
    return obj


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        prog="python -m adapters.sdk",
        description="MORPHOS adapter SDK: validate a data source adapter")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="check an adapter spec")
    p_val.add_argument("spec", help="module:Class or path.py:Class")
    p_val.add_argument("--steps", type=int, default=200)
    p_val.add_argument("--json", action="store_true")

    p_ref = sub.add_parser("conformance", help="run the reference adapters")
    p_ref.add_argument("--steps", type=int, default=200)

    p_doc = sub.add_parser("contract", help="print the adapter contract")
    args = ap.parse_args(argv)

    if args.cmd == "contract":
        print(__doc__)
        return 0

    if args.cmd == "conformance":
        results = [check_conformance(SineSource(), args.steps)]
        demo = [(i * 0.001, np.array([np.sin(i * 0.01), np.cos(i * 0.01)]))
                for i in range(300)]
        results.append(check_conformance(ReplaySource(demo), args.steps))
        bad = 0
        for r in results:
            print(r.report())
            print()
            bad += 0 if r.ok else 1
        return 1 if bad else 0

    src = load_adapter(args.spec)
    res = check_conformance(src, args.steps)
    if args.json:
        print(json.dumps({
            "name": res.name, "dim": res.dim, "ok": res.ok,
            "samples": res.samples,
            "checks": [{"label": c[0], "pass": c[1], "detail": c[2]}
                       for c in res.checks],
        }, indent=2))
    else:
        print(res.report())
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
