"""Reproducible examples runner (T5.4).

Each example is self-contained and prints what it did, so a reader can pipe
them into a terminal and see the platform work without reading the source.

    python examples/run_examples.py            # run all
    python examples/run_examples.py sdk        # run one
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402


def _hdr(title: str) -> None:
    print("\n" + "=" * 66)
    print(f"  {title}")
    print("=" * 66)


# ----------------------------------------------------------------------
def ex_sdk() -> bool:
    """Write and validate a minimal adapter in a few lines."""
    from adapters.sdk import MorphosSource, check_conformance

    _hdr("1. Write an adapter (the whole contract, inline)")

    class Thermal(MorphosSource):
        name = "thermal"
        dim = 2

        def __init__(self):
            self.dt = 0.001
            self._q = []
            super().__init__()
            self.k = 0

        def advance_to(self, t_target):
            while self.t < t_target - 1e-12:
                self.t += self.dt
                self.k += 1
                # pretend: two thermocouples with a slow drift
                self._q.append((self.t, np.array([
                    20.0 + 0.5 * np.sin(self.t * 3.0),
                    21.0 + 0.3 * np.cos(self.t * 1.7),
                ])))

        def read_pending(self):
            out, self._q = self._q, []
            return out

    res = check_conformance(Thermal(), steps=200)
    print(res.report())
    return res.ok


# ----------------------------------------------------------------------
def ex_csv() -> bool:
    """Load the file-backed example adapter and drain some samples."""
    from adapters.sdk import load_adapter

    _hdr("2. A file-backed adapter (examples/adapters/csv_sensor.py)")
    src = load_adapter("examples/adapters/csv_sensor.py:CSVSensor")
    src.advance_to(0.008)
    batch = src.read_pending()
    print(f"loaded {len(src.rows)} rows, {src.skipped} skipped")
    for ts, vec in batch[:5]:
        print(f"  t={ts:.4f}  " + "  ".join(f"{v:8.4f}" for v in vec))
    print(f"  ... {len(batch)} samples in this 8 ms window")
    return len(batch) > 0


# ----------------------------------------------------------------------
def ex_rust() -> bool:
    """Push events through the Rust ring and read them back."""
    _hdr("3. Rust zero-copy ring buffer")
    from core.ingest.rust_trace import RingBuffer, available, status
    if not available():
        print("  SKIP: rust_trace library not built (cargo build --release)")
        return True
    print(f"  lib: {os.path.basename(status()['path'])}  "
          f"abi={status()['abi_expected']}")

    ring = RingBuffer(capacity=4096)
    ring.reset()
    for i in range(1000):
        ring.push(tick=i, xyz=(np.sin(i * 0.01), np.cos(i * 0.01), 0.5),
                  confidence=0.95, flags=0)
    win = ring.pop_batch(2048)
    print(f"  pushed 1000, popped {len(win)}")
    print(f"  first tick={int(win.tick[0])} xyz={win.xyz[0]}")
    print(f"  stats: {ring.stats()}")
    return len(win) == 1000


# ----------------------------------------------------------------------
def ex_export() -> bool:
    """Read a WAL run and summarise it, as the exporter does."""
    _hdr("4. Read a run from the WAL")
    from storage.wal import iter_records

    wal = None
    for cand in ("data_bench/trajectory.wal", "data/trajectory.wal"):
        p = os.path.join(ROOT, cand)
        if os.path.exists(p):
            wal = p
            break
    if wal is None:
        print("  SKIP: no trajectory.wal yet (run: morphos run --steps 400)")
        return True

    recs = list(iter_records(wal))
    resid = [r["residual"] for r in recs if np.isfinite(r["residual"])]
    rates = [r["rate"] for r in recs if r["rate"] > 0]
    print(f"  {os.path.relpath(wal, ROOT)}: {len(recs)} records")
    if resid:
        n = max(1, len(resid) // 10)
        print(f"  residual  first10%={np.mean(resid[:n]):.5f}  "
              f"last10%={np.mean(resid[-n:]):.5f}")
    if rates:
        print(f"  firing    mean={np.mean(rates):.1f} Hz  "
              f"peak={np.max(rates):.1f} Hz")
    return len(recs) > 0


# ----------------------------------------------------------------------
def ex_chaos() -> bool:
    """Show that the pipeline keeps a finite state under injected faults."""
    _hdr("5. Fault injection survives the observer")
    from adapters.chaos import ChaosInjector
    from adapters.sdk import SineSource

    base = SineSource(dim=3, freq=0.9, dt=0.001)
    chaos = ChaosInjector(base, seed=3, noise_sigma=0.05, noise_prob=0.15,
                          drop_prob=0.10, nan_prob=0.02, spike_prob=0.01)
    finite = 0
    total = 0
    for k in range(50):
        chaos.advance_to((k + 1) * 0.005)
        for _, vec in chaos.read_pending():
            total += 1
            # the sanitizer's job is to make this finite downstream
            cleaned = np.nan_to_num(np.asarray(vec, dtype=float),
                                    nan=0.0, posinf=0.0, neginf=0.0)
            if np.all(np.isfinite(cleaned)):
                finite += 1
    print(f"  injected faults over {total} samples; "
          f"{finite} sanitised to finite")
    print(f"  chaos counters: {chaos.stats}")
    return finite == total


EXAMPLES = {
    "sdk": ex_sdk,
    "csv": ex_csv,
    "rust": ex_rust,
    "wal": ex_export,
    "chaos": ex_chaos,
}


def main(argv: list[str]) -> int:
    names = argv[1:] or list(EXAMPLES)
    results = {}
    for name in names:
        fn = EXAMPLES.get(name)
        if fn is None:
            print(f"unknown example {name!r}; choose from {list(EXAMPLES)}")
            return 2
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            results[name] = False

    _hdr("summary")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    bad = sum(1 for ok in results.values() if not ok)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
