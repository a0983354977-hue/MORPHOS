# MORPHOS Adapter SDK

**Write a data source adapter in under an hour.** This guide is the complete
contract, a working example, and the commands to validate your work.

---

## 1. What an adapter is

MORPHOS is a **neuro-inspired streaming pipeline**. A new data source — a
sensor, a network tap, a log shipper, another simulator — only has to answer
one question: *"what values arrived, and when?"*

Everything else is provided by the platform:

```
   your adapter            the MORPHOS core (provided)
  ┌────────────┐   ┌───────────────────────────────────────────────┐
  │ samples +  │──▶│ sanitize → align → encode → neural dynamics   │
  │ timestamps │   │        → predict → observe → WAL / SQLite     │
  └────────────┘   └───────────────────────────────────────────────┘
                                                       │
                                            Studio (3D) ▼
```

Sanitising (NaN/Inf, spikes), time alignment (jitter, gaps, duplicates),
feature encoding into a neural ensemble, and anomaly detection all happen
downstream. **You do not implement any of that.**

## 2. The contract

Two methods and two properties:

| Member | Kind | Meaning |
|---|---|---|
| `dim` | `int` property/attr | Number of values per sample. Constant for the adapter's life. |
| `advance_to(t_target)` | method | Make every sample with timestamp ≤ `t_target` available. |
| `read_pending()` | method | Return `(timestamp, vector)` pairs produced since the last call, then clear. |
| `stats` | property (optional) | Dict merged into the run summary (diagnostics). |
| `close()` | method (optional) | Release hardware handles / descriptors at shutdown. |

### Rules the core relies on

1. **`advance_to` must not block.** The control loop runs at `dt = 5 ms`. Keep
   each call well under that. Never `sleep()` waiting for hardware — if
   nothing arrived, queue nothing and return.
2. **Timestamps must be non-decreasing** across successive `read_pending()`
   calls. The aligner repairs ordering *within* what it is given, but your
   adapter must not go backwards.
3. **`dim` is constant.** Resize by allocating a new adapter, not by varying
   `dim`. A vector of the wrong width is a hard error.
4. **NaN/Inf is allowed.** Emit it and let the sanitiser handle it — that is
   how the pipeline stays robust to real hardware faults.

### Why pull, not push?

A push interface forces the core to buffer unboundedly or drop data. Pulling
lets the loop ask for *"the next 5 ms"* and continue with whatever exists,
which is exactly what makes the pipeline survive jitter and dropouts. It also
makes your adapter trivially testable: drive it with a clock and read out.

## 3. Minimal adapter (copy this)

```python
from adapters.sdk import MorphosSource
import numpy as np

class MySensor(MorphosSource):
    name = "my-sensor"
    dim = 3
    sample_period = 0.001          # nominal hardware rate

    def __init__(self, dt: float = 0.001):
        self.dt = dt
        self._pending = []
        super().__init__()          # call AFTER setting self.dim
        self._ticks = 0

    def advance_to(self, t_target: float) -> None:
        while self.t < t_target - 1e-12:
            self.t += self.dt
            self._ticks += 1
            # ---- replace with a real device read ----
            phase = self._ticks * self.dt
            self._pending.append((self.t, np.array([
                np.sin(phase), np.cos(phase), 0.5 * np.sin(3 * phase),
            ])))

    def read_pending(self):
        out, self._pending = self._pending, []
        return out
```

That is the whole thing. See `examples/adapters/template.py` for the annotated
version and `examples/adapters/csv_sensor.py` for a realistic file-backed one.

## 4. Validate your adapter

```bash
python -m adapters.sdk validate examples/adapters/template.py:MySensor
```

```
adapter: my-sensor  dim=3  samples=1000
  [PASS] dim is a positive int -> 3
  [PASS] has advance_to
  [PASS] has read_pending
  [PASS] emitted at least one sample -> 1000 samples
  [PASS] sample width constant -> widths=[3]
  [PASS] sample width equals dim -> dim=3
  [PASS] timestamps non-decreasing -> last_ts=1.0000
  [PASS] source clock advanced -> t=1.0000
  [PASS] advance_to < 5.0 ms (worst step) -> 0.04 ms
  CONFORMS  (1000.00 ms wall)
```

The harness drives your adapter for 200 control steps exactly as `main.py`
would and checks every rule above. Use `--json` for machine-readable output
(handy in CI).

Also available:

```bash
python -m adapters.sdk conformance    # run the two reference adapters
python -m adapters.sdk contract       # print the contract docstring
```

## 5. Wiring an adapter into a run

The bundled sources (`lorenz`, `rust`) are selected by flag. For a custom
adapter, construct it and pass it where the run builds its source
(`source = MySensor()` in `main.py`), or instantiate through the loader:

```python
from adapters.sdk import load_adapter, check_conformance

src = load_adapter("examples/adapters/csv_sensor.py:CSVSensor")
assert check_conformance(src).ok          # fail fast before a long run
```

`load_adapter` accepts `module:Class` or `path/to/file.py:Class`.

## 6. Patterns that work

**Device-driven (hardware, sockets).** Drain what the device already gave you,
then set `self.t = t_target`:

```python
def advance_to(self, t_target):
    for raw in self._dev.poll():           # whatever is waiting, maybe nothing
        ts, vec = self._decode(raw)
        self._pending.append((ts, vec))
    self.t = t_target                      # never block for hardware
```

**File / log replay.** Map elapsed control time onto the recording's own time
base and loop or stop at the end — see `CSVSensor`.

**Derived features.** If your sensor gives raw frames and you need features,
compute them in the adapter and expose one dimension per feature. The core's
encoder handles scaling.

## 7. Common mistakes

| Symptom | Cause | Fix |
|---|---|---|
| `expected 3-D sample, got 2` | ragged parse | validate width per row; skip bad rows |
| `timestamps must be non-decreasing` | re-reading a buffer | clear `_pending` in `read_pending` |
| Harness fails the latency check | `sleep` in `advance_to` | never wait; return early instead |
| Trajectory flatlines | adapter returns nothing | check `advance_to` actually advances `self.t` |
| Slow first sample | lazy file open inside `advance_to` | open in `__init__` |

## 8. Performance target

The conformance harness reports your worst `advance_to` in milliseconds and
fails above the 5 ms control budget. For 1 kHz sources aim for < 0.5 ms. If
your adapter needs to parse binary at high rate, consider the Rust trace
engine (`engines/rust_trace/`) and wrap it — `adapters/rust_trace_source.py`
is a worked example of exactly that (10 M events/s through ctypes).

## 9. Reference

| Module | Purpose |
|---|---|
| `adapters/sdk.py` | `MorphosSource`, `check_conformance`, `load_adapter` |
| `adapters/lorenz.py` | Lorenz attractor simulator source |
| `adapters/system.py` | Host CPU/memory/disk/network source |
| `adapters/chaos.py` | Fault-injecting wrapper (drops, noise, spikes) |
| `adapters/rust_trace_source.py` | Zero-copy Rust trace/ring source |
| `examples/adapters/template.py` | Annotated starting point |
| `examples/adapters/csv_sensor.py` | File-backed worked example |
| `tests/test_phase5_sdk.py` | Contract + harness tests (incl. negative cases) |

---

*Part of Phase V (Studio & Ecosystem). See `docs/API.md` for the Python API and
`docs/WHITEPAPER.md` for the architecture.*
