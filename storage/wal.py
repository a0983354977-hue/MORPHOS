"""Append-only binary write-ahead log for high-frequency trajectories.

Layout: 8-byte magic header, then fixed-size records:

    <d d          t_sim, confidence
    <3d           actual z-state (t+1 target)
    <3d           predicted z-state
    <d            squared residual
    <d            mean firing rate (Hz)
    <i            flags bitmask
    <d            wall-clock unix time
"""

from __future__ import annotations

import os
import struct
import time

MAGIC = b"MORPHOSWAL"
RECORD = struct.Struct("<dd3d3dddid")
assert RECORD.size == struct.calcsize("<dd3d3dddid")


class WALWriter:
    def __init__(self, path: str):
        self.path = path
        self._f = None
        self.records = 0
        self._existing_records = 0
        self._open()

    def _open(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)),
                    exist_ok=True)
        new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        if new:
            self._f = open(self.path, "ab", buffering=0)
            self._f.write(MAGIC)
            return

        self._f = open(self.path, "r+b", buffering=0)
        magic = self._f.read(len(MAGIC))
        if magic != MAGIC:
            self._f.close()
            self._f = None
            raise ValueError(f"bad WAL magic in {self.path}")
        file_size = os.fstat(self._f.fileno()).st_size
        record_bytes = file_size - len(MAGIC)
        complete_bytes = record_bytes - (record_bytes % RECORD.size)
        if complete_bytes != record_bytes:
            self._f.truncate(len(MAGIC) + complete_bytes)
        self._existing_records = complete_bytes // RECORD.size
        self._f.seek(0, os.SEEK_END)

    def append(self, t_sim: float, conf: float, actual, pred,
               residual: float, rate: float, flags: int) -> None:
        if self._f is None:
            raise RuntimeError("WAL writer is closed")
        rec = RECORD.pack(
            float(t_sim), float(conf),
            float(actual[0]), float(actual[1]), float(actual[2]),
            float(pred[0]), float(pred[1]), float(pred[2]),
            float(residual), float(rate), int(flags),
            time.time())
        self._f.write(rec)
        self.records += 1

    def flush(self) -> None:
        if self._f is not None:
            self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.flush()
            self._f.close()
            self._f = None

    @property
    def total_records(self) -> int:
        return self._existing_records + self.records

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def iter_records(path: str):
    """Yield dicts for each WAL record (skips bad tail on crash)."""
    with open(path, "rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"bad WAL magic in {path}")
        size = RECORD.size
        while True:
            buf = f.read(size)
            if len(buf) < size:
                return
            vals = RECORD.unpack(buf)
            yield {
                "t_sim": vals[0], "conf": vals[1],
                "actual": np_array3(vals[2:5]),
                "pred": np_array3(vals[5:8]),
                "residual": vals[8], "rate": vals[9],
                "flags": vals[10], "wall": vals[11],
            }


def np_array3(vals) -> "object":
    import numpy as np
    return np.asarray(vals, dtype=float)
