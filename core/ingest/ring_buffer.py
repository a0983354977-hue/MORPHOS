"""Resilient ring buffer & time aligner.

Assumes the input stream is hostile: samples may arrive late, duplicated,
out-of-order, contain NaN/Inf, or contain extreme spikes.  The aligner
produces one clean sample per control tick on a fixed time grid, using
linear interpolation when possible and confidence-weighted hold-last
otherwise, so downstream neurons never see step-function shocks.
"""

from __future__ import annotations

import bisect
import math
from collections import deque

import numpy as np
from core.memory.thread_safe_buffer import ThreadSafeBuffer, synchronized

# flag bits recorded per aligned sample
FLAG_NONE = 0
FLAG_CLAMPED = 1 << 0    # extreme value was clipped to safe range
FLAG_SPIKE = 1 << 1      # statistical spike was pulled back toward median
FLAG_INTERP = 1 << 2     # value came from interpolation
FLAG_HOLD = 1 << 3       # value is held from a stale sample
FLAG_STALE = 1 << 4      # hold exceeded max stale time -> confidence ~ 0
FLAG_NO_DATA = 1 << 5    # no sample available at all


class Sanitizer:
    """Per-dimension rolling median/MAD filter + absolute clamp.

    - non-finite values  -> sample marked invalid (returns None)
    - |x| > safe_abs     -> clipped, FLAG_CLAMPED
    - |x - median| > k*MAD -> pulled toward median, FLAG_SPIKE
    """

    def __init__(self, dim: int, window: int = 201, k: float = 8.0,
                 safe_abs: float = 1e6, min_samples: int = 20):
        self.dim = dim
        self.window = window
        self.k = k
        self.safe_abs = safe_abs
        self.min_samples = min_samples
        self._hist = [deque(maxlen=window) for _ in range(dim)]
        # median/MAD are cached and only recomputed every `stat_every`
        # pushes - robust stats do not need per-sample freshness, and
        # np.median on every sample was the ingest hot spot.
        self.stat_every = 16
        self._seen = [0] * dim
        self._stat_cache = [None] * dim  # (med, scale, seen_at_compute)

    def _stats(self, d: int):
        hist = self._hist[d]
        if len(hist) < self.min_samples:
            return None
        cache = self._stat_cache[d]
        if cache is not None and self._seen[d] - cache[2] < self.stat_every:
            return cache[0], cache[1]
        arr = np.asarray(hist)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        scale = 1.4826 * mad
        self._stat_cache[d] = (med, scale, self._seen[d])
        return med, scale

    def push_raw(self, vec: np.ndarray) -> None:
        """Feed a raw (pre-sanitize) finite value into the stats window."""
        for d in range(self.dim):
            v = float(vec[d])
            if math.isfinite(v):
                self._hist[d].append(v)
                self._seen[d] += 1

    def sanitize(self, vec: np.ndarray):
        """Return (clean_vec, flags) or (None, FLAG_NO_DATA) if unusable."""
        v = np.asarray(vec, dtype=float)
        if v.shape[0] != self.dim or not np.all(np.isfinite(v)):
            return None, FLAG_NO_DATA

        out = v.copy()
        flags = FLAG_NONE
        for d in range(self.dim):
            x = float(out[d])
            if abs(x) > self.safe_abs:
                out[d] = math.copysign(self.safe_abs, x)
                flags |= FLAG_CLAMPED
                x = float(out[d])

            stats = self._stats(d)
            if stats is not None:
                med, scale = stats
                if scale > 1e-12:
                    dev = abs(x - med)
                    limit = self.k * scale
                    if dev > limit:
                        out[d] = med + math.copysign(limit, x - med)
                        flags |= FLAG_SPIKE
        return out, flags

    def reset(self) -> None:
        for h in self._hist:
            h.clear()
        self._seen = [0] * self.dim
        self._stat_cache = [None] * self.dim


class RingBuffer(ThreadSafeBuffer):
    """Fixed-capacity append buffer with batch trimming.

    Keeps parallel python lists (fast bisect) and trims in batches so the
    amortised cost stays O(1).
    """

    def __init__(self, capacity: int = 16384, trim_batch: int = 4096):
        super().__init__()
        self.capacity = capacity
        self.trim_batch = trim_batch
        self._t: list[float] = []
        self._v: list[np.ndarray] = []
        self._q: list[int] = []
        self._f: list[int] = []

    @synchronized
    def append(self, t: float, vec: np.ndarray, quality: float,
               flags: int = FLAG_NONE) -> None:
        # out-of-order / duplicate guard: keep buffer sorted by t
        i = bisect.bisect_left(self._t, t)
        if i < len(self._t) and self._t[i] == t:
            # duplicate timestamp -> keep the newer (last) sample
            self._v[i] = vec
            self._q[i] = quality
            self._f[i] = flags
            return
        self._t.insert(i, t)
        self._v.insert(i, vec)
        self._q.insert(i, quality)
        self._f.insert(i, flags)
        if len(self._t) > self.capacity + self.trim_batch:
            del self._t[: self.trim_batch]
            del self._v[: self.trim_batch]
            del self._q[: self.trim_batch]
            del self._f[: self.trim_batch]

    @synchronized
    def __len__(self) -> int:
        return len(self._t)

    @synchronized
    def clear(self) -> None:
        self._t.clear()
        self._v.clear()
        self._q.clear()
        self._f.clear()


class TimeAligner(ThreadSafeBuffer):
    """Aligns an irregular sample stream onto a fixed time grid.

    at(t) semantics (causal: only uses samples already pushed):
      - bracketed by two valid samples within max_interp_gap -> lerp
      - otherwise -> hold last sample, confidence decays with age
      - nothing available -> zeros with confidence 0
    """

    def __init__(self, dt: float, dim: int, max_interp_gap: float = 0.05,
                 hold_tau: float = 0.5, max_stale: float = 30.0,
                 capacity: int = 16384):
        super().__init__()
        self.dt = dt
        self.dim = dim
        self.max_interp_gap = max_interp_gap
        self.hold_tau = hold_tau
        self.max_stale = max_stale
        self.buf = RingBuffer(capacity=capacity)
        self.sanitizer = Sanitizer(dim=dim)
        self._dropped = 0

    @property
    @synchronized
    def dropped_count(self) -> int:
        return self._dropped

    @synchronized
    def push(self, t: float, raw_vec) -> None:
        """Ingest one raw sample. Non-finite samples are counted & skipped."""
        vec = np.asarray(raw_vec, dtype=float).reshape(-1)
        clean, flags = self.sanitizer.sanitize(vec)
        if clean is None:
            self._dropped += 1
            # still feed finite dims into stats where possible
            finite = vec[np.isfinite(vec)]
            if finite.size == self.dim:
                self.sanitizer.push_raw(vec)
            return
        self.sanitizer.push_raw(clean)
        quality = 1.0
        if flags & (FLAG_CLAMPED | FLAG_SPIKE):
            quality = 0.6
        self.buf.append(float(t), clean, quality, flags)

    @synchronized
    def at(self, t: float):
        """Return (vec, confidence, flags) for grid time t."""
        times = self.buf._t
        n = len(times)
        if n == 0:
            return np.zeros(self.dim), 0.0, FLAG_NO_DATA

        i = bisect.bisect_right(times, t)  # first index with ts > t
        prev_i = i - 1 if i > 0 else None
        next_i = i if i < n else None

        prev_ok = prev_i is not None
        next_ok = next_i is not None

        if prev_ok and next_ok:
            tp, tn = times[prev_i], times[next_i]
            gap = tn - tp
            if 0.0 < gap <= self.max_interp_gap and tp <= t <= tn:
                a = (t - tp) / gap
                vec = (1.0 - a) * self.buf._v[prev_i] + a * self.buf._v[next_i]
                q = min(self.buf._q[prev_i], self.buf._q[next_i])
                # confidence highest when t sits on a real sample,
                # lowest at the midpoint between two samples
                d_near = min(t - tp, tn - t) / gap
                conf = q * (1.0 - 0.5 * d_near)
                return vec, float(np.clip(conf, 0.0, 1.0)), \
                    self.buf._f[prev_i] | self.buf._f[next_i] | FLAG_INTERP

        if not prev_ok:
            return np.zeros(self.dim), 0.0, FLAG_NO_DATA

        # hold-last path
        vec = self.buf._v[prev_i]
        age = t - times[prev_i]
        q = self.buf._q[prev_i]
        if age > self.max_stale:
            return vec, 0.0, self.buf._f[prev_i] | FLAG_STALE
        conf = q * math.exp(-max(age, 0.0) / self.hold_tau)
        flags = self.buf._f[prev_i] | FLAG_HOLD
        if age > self.max_interp_gap:
            flags |= FLAG_STALE
        return vec, float(np.clip(conf, 0.0, 1.0)), flags

    @synchronized
    def reset(self) -> None:
        self.buf.clear()
        self.sanitizer.reset()
        self._dropped = 0
