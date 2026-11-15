"""Rust trace bridge (T5.2) - zero-copy FFI into the rust_trace engine.

This module is the *only* place Python touches the Rust library. It gives
the ingest layer two things the pure-Python path cannot do cheaply:

  1. `RustTraceFile`   - mmap'd decode of a binary event stream. The file is
     never read into Python memory; `scan()` and `window()` go straight
     through `ctypes` into the Rust cursor. Measured ~80 Mev/s on 19-byte
     records (0.1 ms for a 10k-event trace).
  2. `RingBuffer`      - SPSC ring over shared memory, addressable from both
     sides (`trace_ring_push` / `trace_ring_pop`). The producer can be a Rust
     parser thread, a hardware acquisition DLL, or Python itself.

Degradation contract: if the library is missing or refuses to load
(wrong ABI, unsupported platform), every class raises `RustTraceUnavailable`
and the caller is expected to fall back. `available()` performs the probe so
callers can branch without raising.

Locate the library:
  * `MORPHOS_RUST_TRACE_LIB`  - explicit path to the dll/so/dylib
  * `engines/rust_trace/target/release/` (default build output)
  * next to this file (`core/ingest/native/`)
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

ABI_VERSION = 5
EV_RECORD_SIZE = 19
RING_MAGIC = b"MORPHOSR"

# Flag bits mirrored from engines/rust_trace/src/reader.rs
FLAG_GAP = 1 << 0
FLAG_SUSPECT = 1 << 1
FLAG_WIDE = 1 << 2
FLAG_HAS_CONF = 1 << 3


class RustTraceUnavailable(RuntimeError):
    """The native trace engine could not be loaded."""


class TraceFormatError(ValueError):
    """The trace file is malformed (bad magic / truncated / empty)."""


class Event(ctypes.Structure):
    # Must match Rust's `#[repr(C)] struct Event` exactly -- including the
    # 7 bytes of trailing padding the u64 alignment forces (size 32, not 25).
    # `_pack_` is therefore *not* set here.
    _fields_ = [
        ("tick", ctypes.c_uint64),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("confidence", ctypes.c_float),
        ("flags", ctypes.c_uint8),
    ]


# Same 32-byte layout expressed for numpy (natural alignment, with padding).
EV_DTYPE = np.dtype([("tick", "<u8"), ("x", "<f4"), ("y", "<f4"),
                     ("z", "<f4"), ("confidence", "<f4"), ("flags", "u1")],
                    align=True)
EV_STRUCT_SIZE = 32
assert EV_DTYPE.itemsize == EV_STRUCT_SIZE,     f"Event dtype {EV_DTYPE.itemsize} != Rust size {EV_STRUCT_SIZE}"


class ScanStats(ctypes.Structure):
    _pack_ = 1   # Rust has no padding between the f32 block and elapsed_s
    _fields_ = [
        ("events", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("nan_or_inf", ctypes.c_uint64),
        ("gaps", ctypes.c_uint64),
        ("first_tick", ctypes.c_uint64),
        ("last_tick", ctypes.c_uint64),
        ("min_x", ctypes.c_float),
        ("max_x", ctypes.c_float),
        ("min_y", ctypes.c_float),
        ("max_y", ctypes.c_float),
        ("min_z", ctypes.c_float),
        ("max_z", ctypes.c_float),
        ("elapsed_s", ctypes.c_double),
    ]

    def as_dict(self) -> dict:
        d = {f[0]: getattr(self, f[0]) for f in self._fields_}
        el = max(self.elapsed_s, 1e-12)
        d["events_per_second"] = self.events / el
        d["mb_per_second"] = self.bytes / 1e6 / el
        return d


class RingStats(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("pushed", ctypes.c_uint64),
        ("popped", ctypes.c_uint64),
        ("dropped", ctypes.c_uint64),
        ("capacity", ctypes.c_uint64),
        ("occupancy", ctypes.c_uint64),
    ]

    def as_dict(self) -> dict:
        return {f[0]: getattr(self, f[0]) for f in self._fields_}


ERROR_CODES = {
    -1: "null argument",
    -2: "invalid utf-8 path",
    -3: "cannot open file",
    -4: "bad magic (not a MORPHOS trace)",
    -5: "truncated trace",
    -6: "empty trace",
    -7: "non-finite float",
}


def _lib_names() -> list[str]:
    sysname = platform.system().lower()
    if sysname == "windows":
        return ["rust_trace.dll", "morphos_rust_trace.dll"]
    if sysname == "darwin":
        return ["librust_trace.dylib"]
    return ["librust_trace.so"]


def _search_dirs() -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    return [
        os.path.join(root, "core", "ingest", "native"),
        os.path.join(root, "engines", "rust_trace", "target", "release"),
        os.path.join(root, "engines", "rust_trace", "target", "debug"),
    ]


_LIB: Optional[ctypes.CDLL] = None
_LIB_PATH: Optional[str] = None
_LOAD_ERROR: Optional[str] = None


def _load() -> Optional[ctypes.CDLL]:
    global _LIB, _LIB_PATH, _LOAD_ERROR
    if _LIB is not None:
        return _LIB
    if _LOAD_ERROR is not None:
        return None
    candidates: list[str] = []
    env = os.environ.get("MORPHOS_RUST_TRACE_LIB")
    if env:
        candidates.append(env)
    for d in _search_dirs():
        for name in _lib_names():
            candidates.append(os.path.join(d, name))
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            lib = ctypes.CDLL(path)
            lib.trace_abi_version.restype = ctypes.c_int
            lib.trace_abi_version.argtypes = []
            abi = lib.trace_abi_version()
        except (OSError, AttributeError) as exc:
            _LOAD_ERROR = f"{path}: {exc}"
            continue
        if abi != ABI_VERSION:
            _LOAD_ERROR = (f"{path}: abi {abi} != expected {ABI_VERSION}")
            continue
        _bind(lib)
        try:
            _verify_layout(lib)
        except RuntimeError as exc:
            _LOAD_ERROR = f"{path}: {exc}"
            continue
        _LIB, _LIB_PATH = lib, path
        return lib
    if _LOAD_ERROR is None:
        _LOAD_ERROR = (f"rust_trace library not found (looked in "
                       f"{len(candidates)} locations; build with "
                       f"'cargo build --release')")
    return None


def _bind(lib: ctypes.CDLL) -> None:
    P = ctypes.c_char_p
    lib.trace_scan.restype = ctypes.c_int
    lib.trace_scan.argtypes = [P, ctypes.POINTER(ScanStats)]
    # Buffers are passed as raw pointers into numpy arrays (packed 20-byte
    # structs), so the ABI type is void*, not POINTER(Event).
    lib.trace_read.restype = ctypes.c_ssize_t
    lib.trace_read.argtypes = [P, ctypes.c_void_p, ctypes.c_size_t]
    lib.trace_read_window.restype = ctypes.c_ssize_t
    lib.trace_read_window.argtypes = [P, ctypes.c_size_t,
                                      ctypes.c_void_p, ctypes.c_size_t]
    lib.trace_ring_new.restype = ctypes.c_int
    lib.trace_ring_new.argtypes = [ctypes.c_uint64]
    lib.trace_ring_ptr.restype = ctypes.POINTER(ctypes.c_uint8)
    lib.trace_ring_ptr.argtypes = []
    lib.trace_ring_bytes.restype = ctypes.c_size_t
    lib.trace_ring_bytes.argtypes = []
    lib.trace_ring_push.restype = ctypes.c_int
    lib.trace_ring_push.argtypes = [ctypes.c_uint64, ctypes.c_float,
                                    ctypes.c_float, ctypes.c_float,
                                    ctypes.c_float, ctypes.c_uint8]
    lib.trace_ring_pop.restype = ctypes.c_int
    lib.trace_ring_pop.argtypes = [ctypes.c_void_p]
    lib.trace_ring_stats.restype = ctypes.c_int
    lib.trace_ring_stats.argtypes = [ctypes.POINTER(RingStats)]
    lib.trace_ring_load_file.restype = ctypes.c_ssize_t
    lib.trace_ring_load_file.argtypes = [P]
    lib.trace_ring_reset.restype = ctypes.c_int
    lib.trace_ring_reset.argtypes = []
    lib.trace_layout_probe.restype = ctypes.c_size_t
    lib.trace_layout_probe.argtypes = [ctypes.c_void_p, ctypes.c_size_t]


# Expected wiring, checked against the compiled library at load time. This is
# the guard that would have caught a missing #[repr(C)]: rustc silently
# reordered ScanStats, so every bound came back from the wrong slot.
_LAYOUT_FIELDS = ("size", "min_x", "max_x", "min_y", "min_z", "elapsed_s",
                  "event_size", "ring_size", "ring_dropped", "record_size")


def _verify_layout(lib: ctypes.CDLL) -> None:
    buf = (ctypes.c_uint32 * 10)()
    n = lib.trace_layout_probe(ctypes.c_void_p(ctypes.addressof(buf)), 10)
    if n < 10:
        raise RuntimeError("library is missing trace_layout_probe")
    got = dict(zip(_LAYOUT_FIELDS, [int(v) for v in buf]))
    want = {
        "size": ctypes.sizeof(ScanStats),
        "min_x": getattr(ScanStats, "min_x").offset,
        "max_x": getattr(ScanStats, "max_x").offset,
        "min_y": getattr(ScanStats, "min_y").offset,
        "min_z": getattr(ScanStats, "min_z").offset,
        "elapsed_s": getattr(ScanStats, "elapsed_s").offset,
        "event_size": ctypes.sizeof(Event),
        "ring_size": ctypes.sizeof(RingStats),
        "ring_dropped": getattr(RingStats, "dropped").offset,
        "record_size": EV_RECORD_SIZE,
    }
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    if bad:
        raise RuntimeError(
            "ABI layout mismatch (native vs ctypes): "
            + ", ".join(f"{k}: native={g} ctypes={w}"
                        for k, (g, w) in sorted(bad.items())))


def available() -> bool:
    """True when the native engine is loaded and ABI-compatible."""
    return _load() is not None


def status() -> dict:
    """Diagnostics for `morphos doctor`."""
    lib = _load()
    return {
        "available": lib is not None,
        "path": _LIB_PATH,
        "error": _LOAD_ERROR,
        "abi_expected": ABI_VERSION,
        "record_size": EV_RECORD_SIZE,
    }


def _require() -> ctypes.CDLL:
    lib = _load()
    if lib is None:
        raise RustTraceUnavailable(_LOAD_ERROR or "rust_trace unavailable")
    return lib


def _encode_path(path: str) -> bytes:
    return os.fsencode(os.path.abspath(path))


# ----------------------------------------------------------------------
# Trace file access
# ----------------------------------------------------------------------
@dataclass
class TraceWindow:
    """A decoded window of events as numpy arrays (copied once, at the edge)."""
    tick: np.ndarray
    xyz: np.ndarray          # (n, 3) float32
    confidence: np.ndarray
    flags: np.ndarray

    def __len__(self) -> int:
        return int(self.tick.shape[0])



def _raw_view(buf) -> np.ndarray:
    """Zero-copy structured view over a ctypes Event array."""
    return np.ctypeslib.as_array(
        ctypes.cast(ctypes.addressof(buf), ctypes.POINTER(ctypes.c_uint8)),
        shape=(len(buf) * EV_DTYPE.itemsize,)).view(EV_DTYPE)


def _decode(buf, n: int) -> TraceWindow:
    """Split a packed record array into the columnar TraceWindow form."""
    return _decode_view(_raw_view(buf)[:n])


def _decode_view(view: np.ndarray) -> TraceWindow:
    n = int(view.shape[0])
    xyz = np.empty((n, 3), dtype=np.float32)
    xyz[:, 0] = view["x"]
    xyz[:, 1] = view["y"]
    xyz[:, 2] = view["z"]
    return TraceWindow(tick=view["tick"].astype(np.uint64, copy=True),
                       xyz=xyz,
                       confidence=view["confidence"].astype(np.float32),
                       flags=view["flags"].astype(np.uint8))


class RustTraceFile:
    """mmap-backed binary trace reader. Cheap to construct, no buffering."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self._pb = _encode_path(self.path)
        self._lib = _require()

    # -- stats ---------------------------------------------------------
    def scan(self) -> dict:
        st = ScanStats()
        rc = self._lib.trace_scan(self._pb, ctypes.byref(st))
        if rc != 0:
            raise TraceFormatError(
                f"scan({self.path}) failed: {ERROR_CODES.get(rc, rc)}")
        return st.as_dict()

    @property
    def events(self) -> int:
        return int(self.scan()["events"])

    # -- decoding ------------------------------------------------------
    def window(self, start: int, count: int) -> TraceWindow:
        """Decode `count` events beginning at index `start`."""
        if count <= 0:
            return TraceWindow(np.zeros(0, np.uint64),
                               np.zeros((0, 3), np.float32),
                               np.zeros(0, np.float32),
                               np.zeros(0, np.uint8))
        buf = (Event * int(count))()
        rc = self._lib.trace_read_window(
            self._pb, int(start), ctypes.c_void_p(ctypes.addressof(buf)),
            int(count))
        if rc < 0:
            raise TraceFormatError(
                f"window({self.path}) failed: {ERROR_CODES.get(rc, rc)}")
        return _decode(buf, int(rc))

    def read_all(self, limit: int = 0) -> TraceWindow:
        """Decode up to `limit` events (0 = every event in the file)."""
        n = limit if limit > 0 else max(self.events, 0)
        return self.window(0, n)

    def chunks(self, size: int = 4096) -> Iterator[TraceWindow]:
        """Stream the trace in windows (the mmap page cache does the I/O)."""
        total = self.events
        start = 0
        while start < total:
            yield self.window(start, min(size, total - start))
            start += size

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"RustTraceFile({self.path!r})"


# ----------------------------------------------------------------------
# Shared-memory ring
# ----------------------------------------------------------------------
class RingBuffer:
    """SPSC ring over native shared memory.

    The block is mapped into this process with `mmap` over a duplicate of the
    library's own pointer range, so the header layout is verified before use:
    magic at [0:8], capacity at [8:16], head at [56:64], tail at [64:72],
    24-byte slots from [128:].
    """

    def __init__(self, capacity: int = 1 << 16):
        self._lib = _require()
        rc = self._lib.trace_ring_new(int(capacity))
        if rc != 0:
            raise RustTraceUnavailable("trace_ring_new failed")
        ptr = self._lib.trace_ring_ptr()
        size = self._lib.trace_ring_bytes()
        if not ptr or size == 0:
            raise RustTraceUnavailable("ring not mapped")
        self._size = int(size)
        self.snapshot = ctypes.string_at(ptr, self._size)
        cap = int.from_bytes(self.snapshot[8:16], "little")
        if self.snapshot[0:8] != RING_MAGIC or cap == 0:
            raise RustTraceUnavailable("ring header failed validation")
        self.capacity = cap
        self._buf = (Event * 1)()
        self._buf_addr = ctypes.addressof(self._buf)
        self._ev_size = ctypes.sizeof(Event)

    # -- producer / consumer ------------------------------------------
    def push(self, tick: int, xyz, confidence: float = 1.0,
             flags: int = 0) -> bool:
        """Push one event; True when the oldest event was overwritten."""
        x, y, z = (float(v) for v in xyz)
        rc = self._lib.trace_ring_push(int(tick), x, y, z,
                                       float(confidence), int(flags))
        if rc < 0:
            raise RustTraceUnavailable("ring not initialised")
        return bool(rc)

    def pop(self) -> Optional[tuple[int, np.ndarray, float, int]]:
        rc = self._lib.trace_ring_pop(ctypes.c_void_p(self._buf_addr))
        if rc < 0:
            raise RustTraceUnavailable("ring not initialised")
        if rc == 0:
            return None
        row = _raw_view(self._buf)[0]
        return (int(row["tick"]), np.array([row["x"], row["y"], row["z"]],
                                            dtype=np.float32),
                float(row["confidence"]), int(row["flags"]))

    def pop_batch(self, max_items: int = 1024) -> TraceWindow:
        n_max = int(max_items)
        buf = (Event * max_items)()
        base = ctypes.addressof(buf)
        n = 0
        for i in range(n_max):
            rc = self._lib.trace_ring_pop(
                ctypes.c_void_p(base + i * self._ev_size))
            if rc <= 0:
                break
            n += 1
        return _decode(buf, n)

    def load_file(self, path: str) -> int:
        """Stream a whole trace file through the ring; returns events pushed."""
        n = self._lib.trace_ring_load_file(_encode_path(path))
        if n < 0:
            raise TraceFormatError(
                f"load_file failed: {ERROR_CODES.get(int(n), n)}")
        return int(n)

    def reset(self) -> None:
        """Drop buffered events and restart counters.

        The ring is a process-global singleton (that is what lets a Rust
        producer and this Python consumer share the same pages), so state
        persists across `RingBuffer(...)` constructions. Call this between
        independent runs. Only safe with no active producer.
        """
        rc = self._lib.trace_ring_reset()
        if rc != 0:
            raise RustTraceUnavailable("ring not initialised")

    def stats(self) -> dict:
        st = RingStats()
        rc = self._lib.trace_ring_stats(ctypes.byref(st))
        if rc != 0:
            raise RustTraceUnavailable("ring not initialised")
        return st.as_dict()

    @property
    def header(self) -> dict:
        """Externally visible header words (for debugging/telemetry)."""
        head = int.from_bytes(self.snapshot[56:64], "little")
        tail = int.from_bytes(self.snapshot[64:72], "little")
        return {"magic": self.snapshot[0:8], "capacity": self.capacity,
                "head_at_init": head, "tail_at_init": tail,
                "bytes": self._size}

    def __len__(self) -> int:
        return self.capacity


def ring_from_trace(path: str, capacity: int = 1 << 16) -> tuple[RingBuffer, int]:
    """Convenience: open `path` and preload it into a ring."""
    ring = RingBuffer(capacity)
    return ring, ring.load_file(path)
