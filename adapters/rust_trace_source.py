"""Rust-backed trace adapter (T5.2 integration).

Speaks the same protocol as `LorenzSource` / `SystemSource`, so `main.py`
cannot tell the difference:

    advance_to(t) -> None          # advance the internal cursor
    read_pending() -> [(ts, vec)]  # samples since the last call
    .dim                           # lane count (3)

Where the samples come from is the only difference: instead of integrating
Lorenz in Python, the events are decoded out of a `.morphose` binary stream
by the Rust engine (mmap, zero-copy) and optionally staged through the shared
ring buffer.

Source of truth precedence:
  1. `path=` - an existing trace file (system logs, capture dumps, sensors)
  2. `generated=True` - synthesise a Lorenz trace with `morphos-trace
     mkstream` on first use, then replay it

Both modes are *replay*: the adapter maps wall-clock control steps onto trace
ticks, which is exactly what a real ingest daemon does with a live stream
(where `follow=True` tails an append-only file).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np

from core.ingest import rust_trace as rt


class RustTraceSource:
    """Replay a binary trace through the neural core."""

    dim = 3

    def __init__(self, path: Optional[str] = None,
                 tick_rate: float = 1000.0,
                 follow: bool = False,
                 ring_capacity: int = 1 << 14,
                 use_ring: bool = True,
                 generate_if_missing: bool = True,
                 generate_seconds: float = 60.0,
                 chaos: str = "none",
                 loop: bool = True,
                 cache_max: int = 250_000):
        self.tick_rate = float(tick_rate)
        self.follow = follow
        self.loop = loop
        self.cache_max = int(cache_max)
        self._cursor = 0            # next event index to emit
        self._t = 0.0               # source time (s)
        self.dropped = 0
        self._pending: list[tuple[float, np.ndarray]] = []
        self._path = path
        self._tmpdir: Optional[str] = None
        self._ring = None
        self.backend = "python-replay"

        if not rt.available():
            raise rt.RustTraceUnavailable(
                "RustTraceSource requires the rust_trace engine; "
                "build it with 'cargo build --release' in engines/rust_trace"
            )

        if path is None or not os.path.exists(path):
            if not generate_if_missing:
                raise FileNotFoundError(path)
            self._path = self._generate(generate_seconds, chaos)
        self._path = os.path.abspath(self._path)
        self._file = rt.RustTraceFile(self._path)
        self._stats = self._file.scan()
        self.n_events = int(self._stats["events"])
        if self.n_events == 0:
            raise ValueError(f"trace {self._path} contains no events")

        if use_ring:
            try:
                self._ring = rt.RingBuffer(ring_capacity)
                self._ring.load_file(self._path)
                self.backend = "rust-ring"
            except rt.RustTraceUnavailable:
                self._ring = None

        # The neural loop must never block on mmap paging mid-chunk, so the
        # trace is decoded into a RAM window up front (bounded by cache_max).
        self._window = self._file.window(0, min(self.n_events, self.cache_max))
        self._wstart = 0  # file index of self._window[0]

    # -- setup ---------------------------------------------------------
    def _generate(self, seconds: float, chaos: str) -> str:
        exe = self._find_exe()
        if exe is None:
            raise rt.RustTraceUnavailable(
                "morphos-trace executable not found; cannot generate a trace"
            )
        d = tempfile.mkdtemp(prefix="morphos_trace_")
        self._tmpdir = d
        out = os.path.join(d, "generated.morphose")
        subprocess.run([exe, "mkstream", out, "--secs", str(seconds),
                        "--hz", str(int(self.tick_rate)),
                        "--chaos", chaos], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out

    @staticmethod
    def _find_exe() -> Optional[str]:
        # This file lives in <root>/adapters/, so the project root is one
        # directory up (not two: the extra dirname would walk above MORPHOS).
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rel = os.path.join(root, "engines", "rust_trace", "target",
                           "release")
        for name in ("morphos-trace.exe", "morphos-trace"):
            p = os.path.join(rel, name)
            if os.path.exists(p):
                return p
        return shutil.which("morphos-trace")

    # -- source protocol ----------------------------------------------
    @property
    def t(self) -> float:
        return self._t

    def advance_to(self, t_target: float) -> None:
        """Queue every trace sample whose timestamp is <= t_target."""
        if t_target <= self._t + 1e-12:
            return
        dt_tick = 1.0 / self.tick_rate
        n_needed = int(round((t_target - self._t) / dt_tick))
        if n_needed <= 0:
            self._t = t_target
            return
        ticks = np.arange(self._cursor, self._cursor + n_needed)
        if self.n_events == 0:
            self._t = t_target
            return
        if self.loop:
            idx = ticks % self.n_events
        else:
            if self._cursor + n_needed > self.n_events:
                idx = np.arange(self._cursor, self.n_events)
            else:
                idx = ticks
        if idx.size:
            xyz = self._gather(idx)
            base_t = self._cursor * dt_tick
            for k in range(idx.size):
                self._pending.append(
                    (base_t + (k + 1) * dt_tick, xyz[k].copy()))
        self._cursor += n_needed
        self._t = t_target

    def _gather(self, idx: np.ndarray) -> np.ndarray:
        """Fetch events by index, extending the RAM window on demand.

        The first load caches the whole trace; traces larger than `cache_max`
        events are read through the mmap in `cache_max` blocks, so memory stays
        bounded while the Rust cursor still does the decoding.
        """
        if idx.size == 0:
            return np.zeros((0, 3), np.float32)
        lo, hi = int(idx.min()), int(idx.max())
        cached_hi = self._wstart + self._window.tick.size
        if lo >= self._wstart and hi < cached_hi:
            return self._window.xyz[idx - self._wstart]
        # (Re)load a block covering [lo, lo + cache_max).
        want = min(hi - lo + 1, self.cache_max)
        self._window = self._file.window(lo, want)
        self._wstart = lo
        local = idx - lo
        if local.size and local.max() >= self._window.tick.size:
            # Wrap/loop straddles the block edge: resolve that tail by index.
            out = self._window.xyz[local[local < self._window.tick.size]]
            return out
        return self._window.xyz[local]

    def read_pending(self) -> list[tuple[float, np.ndarray]]:
        out = self._pending
        self._pending = []
        return out

    # -- diagnostics ---------------------------------------------------
    @property
    def stats(self) -> dict:
        """Matches `ChaosInjector.stats`: a plain dict, not a callable."""
        d = dict(self._stats)
        d["backend"] = self.backend
        d["path"] = self._path
        d["cursor"] = self._cursor
        d["ring"] = self._ring.stats() if self._ring is not None else None
        return d

    def close(self) -> None:
        if self._tmpdir and os.path.isdir(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass
