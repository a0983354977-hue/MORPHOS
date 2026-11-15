"""Phase V / T5.2 acceptance: Rust trace engine + Python FFI bridge.

Run with:  python -m pytest tests/test_phase5_rust.py -v
       or:  python tests/test_phase5_rust.py

These tests are the contract between the Python core and the Rust engine:
if the ABI, the struct layout, or the trace codec drifts, they fail loudly
instead of silently corrupting trajectories.
"""

from __future__ import annotations

import ctypes
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.ingest import rust_trace as rt

MAGIC = b"MORPHOSE"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _rust_exe() -> str:
    for name in ("morphos-trace.exe", "morphos-trace"):
        p = os.path.join(ROOT, "engines", "rust_trace", "target", "release",
                         name)
        if os.path.exists(p):
            return p
    raise RuntimeError("morphos-trace executable not built")


def _write_trace(path: str, rows) -> None:
    """Write a trace from Python to prove the on-disk format independently."""
    with open(path, "wb") as f:
        f.write(MAGIC)
        for tick, (x, y, z), conf, flags in rows:
            f.write(struct.pack("<Q", tick & 0xFFFFFFFFFFFF))
            for v in (x, y, z):
                raw = int(v * 2147483.648)
                raw = max(-8388608, min(8388607, raw))
                f.write(struct.pack("<I", raw & 0xFFFFFF)[:3])
            f.write(b"\x00")
            f.write(struct.pack("<B", flags))


# The native ring is a process-global singleton: the first RingBuffer()
# decides the capacity for the whole process, and later calls attach to the
# same pages. Tests therefore never assume a capacity; they read
# `ring.capacity` and size their data from it.
_RING_CAPACITY_HINT = 4096


def _fresh_ring(capacity: int = _RING_CAPACITY_HINT) -> "rt.RingBuffer":
    """Reset the shared ring so each test starts empty."""
    ring = rt.RingBuffer(capacity)
    ring.reset()
    return ring


def _require_rust():
    if not rt.available():
        raise RuntimeError(f"rust_trace unavailable: {rt.status()['error']}")
    return True


# ----------------------------------------------------------------------
# ABI / layout
# ----------------------------------------------------------------------
def test_abi_and_struct_layout():
    """Event must be exactly the Rust repr(C) size, or every read corrupts."""
    _require_rust()
    assert rt.ABI_VERSION == 5
    assert ctypes.sizeof(rt.Event) == 32, "Event layout drifted from Rust"
    assert rt.EV_DTYPE.itemsize == 32
    # field offsets must match #[repr(C)] on x86_64
    offs = {n: rt.EV_DTYPE.fields[n][1] for n in rt.EV_DTYPE.names}
    assert offs == {"tick": 0, "x": 8, "y": 12, "z": 16,
                    "confidence": 20, "flags": 24}, offs


def test_scan_stats_layout():
    assert ctypes.sizeof(rt.ScanStats) == 2 * 8 + 4 * 8 + 6 * 4 + 8
    assert ctypes.sizeof(rt.RingStats) == 5 * 8


# ----------------------------------------------------------------------
# trace decoding
# ----------------------------------------------------------------------
def test_scan_and_window_match_written_values(tmp_path):
    _require_rust()
    path = str(tmp_path / "basic.morphose")
    # f24 lanes hold +-3.90625; keep the fixtures inside that band.
    rows = [(i, (0.05 * i, -0.06 * i, 0.3), 1.0, 0) for i in range(50)]
    _write_trace(path, rows)

    f = rt.RustTraceFile(path)
    st = f.scan()
    assert st["events"] == 50, st
    assert st["first_tick"] == 0 and st["last_tick"] == 49, st
    assert abs(st["min_x"] - 0.0) < 2e-3, st["min_x"]
    assert abs(st["max_x"] - 2.45) < 3e-3, st["max_x"]
    assert abs(st["min_y"] + 2.94) < 4e-3, st["min_y"]

    w = f.window(0, 50)
    assert len(w) == 50
    assert np.allclose(w.xyz[:, 0], 0.05 * np.arange(50), atol=2e-3)
    assert np.allclose(w.xyz[:, 1], -0.06 * np.arange(50), atol=2e-3)
    assert np.allclose(w.xyz[:, 2], 0.3, atol=1e-4)
    assert (w.tick == np.arange(50)).all()


def test_window_is_index_addressed_and_clipped(tmp_path):
    _require_rust()
    path = str(tmp_path / "win.morphose")
    _write_trace(path, [(i, (i * 0.01, 0.0, 0.0), 1.0, 0) for i in range(100)])
    f = rt.RustTraceFile(path)

    mid = f.window(40, 10)
    assert len(mid) == 10
    assert mid.tick[0] == 40 and mid.tick[-1] == 49
    assert abs(float(mid.xyz[0, 0]) - 0.40) < 1e-3

    tail = f.window(95, 100)   # request past the end -> clipped, not an error
    assert len(tail) == 5

    empty = f.window(100, 10)
    assert len(empty) == 0

    chunks = list(f.chunks(30))
    assert sum(len(c) for c in chunks) == 100
    assert [c.tick[0] for c in chunks] == [0, 30, 60, 90]


def test_bad_trace_rejected(tmp_path):
    _require_rust()
    bad = tmp_path / "bad.morphose"
    bad.write_bytes(b"NOTAMORPHOS" + b"\x00" * 40)
    f = rt.RustTraceFile(str(bad))
    try:
        f.scan()
    except rt.TraceFormatError:
        pass
    else:
        raise AssertionError("bad magic must raise TraceFormatError")

    empty = tmp_path / "empty.morphose"
    empty.write_bytes(b"")
    try:
        rt.RustTraceFile(str(empty)).scan()
    except rt.TraceFormatError:
        pass
    else:
        raise AssertionError("empty trace must raise TraceFormatError")


def test_confidence_and_flags_roundtrip(tmp_path):
    _require_rust()
    path = str(tmp_path / "conf.morphose")
    _write_trace(path, [(0, (0.5, 0.5, 0.5), 0.25, rt.FLAG_SUSPECT),
                        (1, (0.5, 0.5, 0.5), 1.0, 0)])
    w = rt.RustTraceFile(path).read_all()
    assert abs(float(w.confidence[0]) - 1.0) < 1e-3
    assert int(w.flags[0]) == rt.FLAG_SUSPECT


def test_cli_generated_trace_is_consistent():
    _require_rust()
    import subprocess
    import tempfile
    d = tempfile.mkdtemp()
    path = os.path.join(d, "gen.morphose")
    subprocess.run([_rust_exe(), "mkstream", path, "--secs", "1",
                    "--hz", "1000"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    f = rt.RustTraceFile(path)
    st = f.scan()
    assert st["events"] == 1000
    assert st["nan_or_inf"] == 0
    w = f.read_all()
    assert len(w) == 1000
    assert np.isfinite(w.xyz).all()
    os.remove(path)
    os.rmdir(d)


# ----------------------------------------------------------------------
# ring buffer
# ----------------------------------------------------------------------
def test_ring_push_pop_fifo():
    _require_rust()
    ring = _fresh_ring(_RING_CAPACITY_HINT)
    for i in range(10):
        assert ring.push(i, (i * 0.5, 0.0, -1.0), 1.0) is False
    st = ring.stats()
    assert st["pushed"] - st["popped"] == 10
    out = ring.pop_batch(100)
    assert len(out) == 10
    assert (out.tick == np.arange(10)).all()
    assert np.allclose(out.xyz[:, 0], np.arange(10) * 0.5, atol=1e-4)
    assert ring.pop() is None


def test_ring_overwrites_oldest_and_counts_drops():
    _require_rust()
    ring = _fresh_ring()
    cap = ring.capacity
    total = cap + 12                      # overflow by exactly 12 events
    for i in range(total):
        ring.push(i, (0.0, 0.0, 0.0))
    st = ring.stats()
    assert st["dropped"] == 12, st
    left = ring.pop_batch(cap + 100)
    assert len(left) == cap
    # Only the newest `cap` events survive, still in order.
    assert left.tick.tolist() == list(range(12, total))


def test_ring_metadata_matches_library():
    _require_rust()
    ring = _fresh_ring()
    cap = ring.capacity
    # Internally the capacity is rounded up to a power of two (mask indexing).
    assert cap & (cap - 1) == 0, cap
    hdr = ring.header
    assert hdr["magic"] == rt.RING_MAGIC
    assert hdr["capacity"] == cap
    # header (128 bytes) + capacity * sizeof(Event)
    assert hdr["bytes"] == 128 + cap * rt.EV_STRUCT_SIZE, hdr


def test_ring_loads_trace_and_serves_windows(tmp_path):
    _require_rust()
    path = str(tmp_path / "ring.morphose")
    _write_trace(path, [(i, (i * 0.005, -i * 0.005, 0.0), 1.0, 0)
                        for i in range(500)])
    ring = _fresh_ring(1024)
    n = ring.load_file(path)
    assert n == 500
    got = ring.pop_batch(500)
    assert len(got) == 500
    assert got.tick.tolist() == list(range(500))
    assert abs(float(got.xyz[-1, 0]) - 2.495) < 4e-3


# ----------------------------------------------------------------------
# degradation contract
# ----------------------------------------------------------------------
def test_unavailable_library_falls_back_cleanly():
    """When the DLL is missing, `available()` is False and use raises."""
    saved_lib = rt._LIB
    saved_path = rt._LIB_PATH
    saved_err = rt._LOAD_ERROR
    try:
        rt._LIB = None
        rt._LIB_PATH = None
        rt._LOAD_ERROR = "forced for test"
        os.environ["MORPHOS_RUST_TRACE_LIB"] = "/nonexistent/lib.so"
        # Force the search to miss every candidate.
        orig = rt._search_dirs
        rt._search_dirs = lambda: ["/nonexistent"]
        try:
            assert rt.available() is False
            try:
                rt.RustTraceFile("whatever.morphose")
            except rt.RustTraceUnavailable:
                pass
            else:
                raise AssertionError("expected RustTraceUnavailable")
        finally:
            rt._search_dirs = orig
    finally:
        os.environ.pop("MORPHOS_RUST_TRACE_LIB", None)
        rt._LIB = saved_lib
        rt._LIB_PATH = saved_path
        rt._LOAD_ERROR = saved_err


# ----------------------------------------------------------------------
# throughput guard (the T5.2 acceptance figure)
# ----------------------------------------------------------------------
def test_decode_throughput_floor(tmp_path):
    """Decoding must stay far above real-time sensor rates."""
    _require_rust()
    import time
    path = str(tmp_path / "big.morphose")
    _write_trace(path, [(i, (0.1, 0.2, 0.3), 1.0, 0) for i in range(200_000)])
    f = rt.RustTraceFile(path)
    t0 = time.perf_counter()
    w = f.read_all()
    el = time.perf_counter() - t0
    rate = len(w) / el
    # Budget: 1 kHz sensor stream -> 500x margin is what we measure in CI;
    # assert a conservative 500 kev/s so slow CI machines do not flake.
    assert rate > 500_000, f"decode only {rate/1e6:.2f} Mev/s"


if __name__ == "__main__":
    failures = 0
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    for name, fn in tests:
        try:
            if fn.__code__.co_argcount:
                import tempfile
                from pathlib import Path
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"  PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
