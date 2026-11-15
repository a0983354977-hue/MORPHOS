"""System resource adapter (T2.1).

Samples real OS metrics via psutil: CPU utilisation, memory pressure,
disk I/O throughput and network I/O throughput.  Produces a fixed
4-D schema per sample; irregular sampling is normalised by the core
ingest layer (TimeAligner) exactly like any other external source.

Schema: [cpu_util(0-1), mem_percent(0-1), disk_io_rate, net_io_rate]
Rates are bytes/sec log-compressed with log1p to keep spikes bounded.
"""

from __future__ import annotations

import time

import numpy as np
import psutil


class SystemSource:
    dim = 4

    def __init__(self, disk_mount: str | None = None):
        self._proc = psutil.Process()
        self._last_cpu_times = psutil.cpu_times()
        self._last_cpu_wall = time.perf_counter()
        self._last_disk = psutil.disk_io_counters()
        self._last_net = psutil.net_io_counters()
        self._last_io_wall = time.perf_counter()
        self._disk_mount = disk_mount
        self.t = 0.0
        self._pending: list[tuple[float, np.ndarray]] = []
        # prime cpu_percent baseline
        psutil.cpu_percent(interval=None)

    # -- raw metric helpers ---------------------------------------------
    def _cpu_util(self) -> float:
        ct = psutil.cpu_times()
        wall = time.perf_counter()
        dt = max(wall - self._last_cpu_wall, 1e-6)
        busy = ((ct.user - self._last_cpu_times.user) +
                (ct.system - self._last_cpu_times.system) +
                getattr(ct, "interrupt", 0.0) -
                getattr(self._last_cpu_times, "interrupt", 0.0))
        # cpu_times are per-CPU aggregated seconds; normalise by elapsed
        ncpu = psutil.cpu_count() or 1
        util = busy / (dt * ncpu)
        self._last_cpu_times = ct
        self._last_cpu_wall = wall
        return float(np.clip(util, 0.0, 1.0))

    def _io_rates(self) -> tuple[float, float]:
        now = time.perf_counter()
        dt = max(now - self._last_io_wall, 1e-6)
        d = psutil.disk_io_counters()
        n = psutil.net_io_counters()
        disk_b = ((d.read_bytes - self._last_disk.read_bytes) +
                  (d.write_bytes - self._last_disk.write_bytes)) / dt
        net_b = ((n.bytes_sent - self._last_net.bytes_sent) +
                 (n.bytes_recv - self._last_net.bytes_recv)) / dt
        self._last_disk, self._last_net = d, n
        self._last_io_wall = now
        # log-compress: typical idle ~0, bursts -> large but bounded
        return float(np.log1p(max(disk_b, 0.0)) / 20.0), \
            float(np.log1p(max(net_b, 0.0)) / 20.0)

    def sample(self) -> np.ndarray:
        mem = psutil.virtual_memory().percent / 100.0
        cpu = self._cpu_util()
        disk_r, net_r = self._io_rates()
        return np.array([cpu, mem, disk_r, net_r], dtype=float)

    # -- source interface (mirrors LorenzSource/ChaosInjector) ----------
    def advance_to(self, t_target: float) -> None:
        # system clock is wall-driven; advance_to just stamps pending
        # samples at the requested grid horizon. One sample per call is
        # enough - the aligner handles the rest.
        self.t = t_target
        self._pending.append((t_target, self.sample()))

    def read_pending(self):
        out = self._pending
        self._pending = []
        return out
