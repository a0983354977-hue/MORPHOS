"""Phase V / T5.4 acceptance: Adapter SDK contract + conformance harness.

    python tests/test_phase5_sdk.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from adapters.sdk import (AdapterContractError, MorphosSource,  # noqa: E402
                          ReplaySource, SineSource, check_conformance,
                          load_adapter)


# ----------------------------------------------------------------------
# reference adapters conform
# ----------------------------------------------------------------------
def test_sine_source_conforms():
    res = check_conformance(SineSource(dim=3), steps=200)
    assert res.ok, res.report()
    assert res.samples >= 200
    assert res.dim == 3


def test_replay_source_conforms():
    recs = [(i * 0.001, np.array([np.sin(i * 0.01), np.cos(i * 0.01)]))
            for i in range(400)]
    res = check_conformance(ReplaySource(recs), steps=100)
    assert res.ok, res.report()
    assert res.dim == 2


# ----------------------------------------------------------------------
# shipped examples conform
# ----------------------------------------------------------------------
def test_template_example_conforms():
    src = load_adapter("examples/adapters/template.py:MySensor")
    res = check_conformance(src, steps=200)
    assert res.ok, res.report()


def test_csv_example_conforms():
    src = load_adapter("examples/adapters/csv_sensor.py:CSVSensor")
    res = check_conformance(src, steps=200)
    assert res.ok, res.report()


def test_load_adapter_rejects_bad_spec():
    for spec in ["nosuchmodule", "adapters.sdk:NoSuchClass"]:
        try:
            load_adapter(spec)
        except Exception:
            pass
        else:
            raise AssertionError(f"{spec!r} should have failed")


# ----------------------------------------------------------------------
# the harness must actually catch violations
# ----------------------------------------------------------------------
def _bad_dim():
    class BadDim(MorphosSource):
        name = "bad-dim"
        dim = 3

        def __init__(self):
            super().__init__()
            self.i = 0

        def advance_to(self, t_target):
            self.t = t_target
            self.i += 1

        def read_pending(self):
            # wrong width on every sample
            return [(self.t, np.array([1.0, 2.0]))]
    return BadDim()


def test_detects_wrong_sample_width():
    res = check_conformance(_bad_dim(), steps=20)
    assert not res.ok
    labels = [c[0] for c in res.failures]
    assert any("width" in l for l in labels), labels


def _bad_order():
    class BadOrder(MorphosSource):
        name = "bad-order"
        dim = 2

        def __init__(self):
            super().__init__()
            self.n = 0

        def advance_to(self, t_target):
            self.t = t_target

        def read_pending(self):
            self.n += 1
            # timestamps go backwards
            return [(100.0 - self.n, np.array([0.0, 0.0]))]
    return BadOrder()


def test_detects_decreasing_timestamps():
    res = check_conformance(_bad_order(), steps=20)
    assert not res.ok
    assert any("timestamp" in c[0] for c in res.failures), res.failures


def _bad_slow():
    import time

    class TooSlow(MorphosSource):
        name = "too-slow"
        dim = 2

        def __init__(self):
            super().__init__()

        def advance_to(self, t_target):
            time.sleep(0.008)      # 8 ms > 5 ms control budget
            self.t = t_target

        def read_pending(self):
            return [(self.t, np.array([0.0, 0.0]))]
    return TooSlow()


def test_detects_slow_adapter():
    res = check_conformance(_bad_slow(), steps=5)
    assert not res.ok
    assert any("ms" in c[0] for c in res.failures), res.failures


def test_zero_dim_rejected():
    class ZeroDim(MorphosSource):
        name = "zero"
        dim = 0

        def advance_to(self, t_target):
            pass

        def read_pending(self):
            return []
    try:
        ZeroDim()
    except AdapterContractError:
        pass
    else:
        raise AssertionError("dim=0 must be rejected at construction")


# ----------------------------------------------------------------------
# duck typing: a plain object without the base class still works
# ----------------------------------------------------------------------
def test_duck_typed_source_conforms():
    class DuckSource:
        name = "duck"
        dim = 2

        def __init__(self):
            self.t = 0.0
            self._dt = 0.001
            self._q = []

        def advance_to(self, t_target):
            while self.t < t_target - 1e-12:
                self.t += self._dt
                self._q.append((self.t, np.array([self.t, self.t * 2])))

        def read_pending(self):
            out, self._q = self._q, []
            return out

    duck = DuckSource()
    res = check_conformance(duck, steps=100)
    assert res.ok, res.report()


# ----------------------------------------------------------------------
# values are delivered intact (no silent coercion)
# ----------------------------------------------------------------------
def test_sample_values_roundtrip():
    vals = np.array([[1.5, -2.25, 0.0], [-0.125, 3.75, 42.0]])
    recs = [(i * 0.001, vals[i]) for i in range(len(vals))]
    src = ReplaySource(recs)
    src.advance_to(0.010)
    got = src.read_pending()
    assert len(got) == 2
    for i, (_, vec) in enumerate(got):
        assert np.allclose(vec, vals[i]), (vec, vals[i])


def test_nan_passes_through_for_sanitizer():
    """The adapter may emit NaN; the core's sanitizer is what removes it."""
    class WithNaN(MorphosSource):
        name = "with-nan"
        dim = 2

        def __init__(self):
            super().__init__()
            self.n = 0

        def advance_to(self, t_target):
            self.t = t_target
            self.n += 1

        def read_pending(self):
            v = np.array([np.nan, 1.0]) if self.n % 2 else np.array([1.0, 1.0])
            return [(self.t, v)]

    res = check_conformance(WithNaN(), steps=10)
    # NaN is tolerated, not a conformance failure
    assert res.ok, res.report()
    assert any("non-finite" in c[0] for c in res.checks)


if __name__ == "__main__":
    failures = 0
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
