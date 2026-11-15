# MORPHOS API Reference

Two surfaces: the **Python core** (embed the pipeline in your own program) and
the **CLI** (drive it from a shell). The 3D Studio is a thin client over the
HTTP/WebSocket hub documented in §4.

All Python paths assume the repository root is on `sys.path`.

---

## 1. Pipeline overview

```
adapters/*        pull samples        (you own this layer — see docs/SDK.md)
    │
core/ingest       sanitize + align    ring buffer, Rust trace bridge
    │
core/encoding     features → ensemble activity (Nengo)
    │
core/dynamics     recurrent state, multiscale weave
    │
core/prediction   one-step forecast
    │
core/observer     residual EMA, health, dual-opinion anomaly, circuit breaker
    │
core/counterfactual  what-if replay of alternative branches
    │
storage/*         WAL (binary, append-only) + SQLite (queries)
```

---

## 2. CLI (`bin/morphos.exe`)

Built from `cli/morphos.nim`. No runtime dependencies.

| Command | Purpose |
|---|---|
| `morphos init` | Write `morphos.toml`, `morphos.rust.toml`, create `data/ results/ traces/` |
| `morphos doctor` | Probe Python/Nengo, Rust lib + FFI ABI, studio build, config |
| `morphos run` | Run the control loop (the main experiment) |
| `morphos ingest` | Generate / scan / stream a Rust trace file |
| `morphos export` | Convert a WAL run to `json` / `ndjson` / `csv` / `html` |
| `morphos serve` | Start the 3D Studio server |
| `morphos experiment {interop,transfer,anomaly}` | Run the named experiment |
| `morphos version`, `morphos help` | Metadata |

### `morphos run`

```
morphos run [--steps N] [--neurons N] [--chunk N] [--gain F] [--mu F]
            [--source lorenz|rust] [--trace PATH] [--chaos none|combat|...]
            [--data-dir DIR] [--feed URL] [--resume] [--detach]
```

| Flag | Default | Meaning |
|---|---|---|
| `--steps` | 1200 | Control steps to execute |
| `--neurons` | 400 | Ensemble size |
| `--chunk` | 200 | Steps per rebuild batch |
| `--gain` | 0.5 | Recurrent gain |
| `--mu` | 0.05 | Recurrent time constant |
| `--source` | `lorenz` | Sample source |
| `--trace` | — | Trace file for `--source rust` |
| `--chaos` | `none` | Fault injection profile |
| `--data-dir` | `data` | Where WAL/SQLite live |
| `--feed` | — | POST live frames to a Studio hub |
| `--resume` | off | Continue from existing WAL |
| `--detach` | off | Background it, log to `data/run.log` |

### `morphos ingest`

```
morphos ingest --mkstream --out traces/run.trace --events 200000
morphos ingest --trace traces/run.trace              # scan, report stats
morphos ingest --trace traces/run.trace --stream     # replay at a given rate
```

### `morphos export`

```
morphos export --wal data/trajectory.wal --format json --out data/run.frames.json
```

Formats: `json` (object with `frames[]`), `ndjson`, `csv`, `html` (self-contained
canvas viewer). The frame schema is shared with the Studio:

```json
{"step": 42, "t": 0.21, "conf": 1.0,
 "a": [x, y, z], "p": [x, y, z], "r": 0.0013,
 "rate": 72.4, "flags": 0, "low_conf": false, "breaker": false}
```

`a` = actual ensemble output, `p` = one-step prediction, `r` = residual.

---

## 3. Python core

### 3.1 Runs

```python
import sys; sys.path.insert(0, ".")
import main

sys.argv = ["main.py", "--steps", "400", "--data-dir", "data"]
raise SystemExit(main.main())
```

`main.main()` returns a process exit code and writes `data/trajectory.wal` +
`data/morphos.db`.

### 3.2 Reading results

```python
from storage.wal import iter_records

for rec in iter_records("data/trajectory.wal"):
    print(rec["t_sim"], rec["conf"], rec["actual"], rec["residual"])
```

Each record is a dict:

| Key | Type | Meaning |
|---|---|---|
| `t_sim` | float | Simulation time (s) |
| `conf` | float | Confidence in [0, 1] |
| `actual` | float[3] | Ensemble output |
| `pred` | float[3] | One-step prediction |
| `residual` | float | Prediction error |
| `rate` | float | Ensemble firing rate (Hz) |
| `flags` | int | Bitfield: low-confidence / breaker / anomalies |
| `wall` | float | Wall-clock timestamp when written |

```python
from storage.sqlite_store import Store
store = Store("data/morphos.db")
store.log_event("INFO", "custom marker", data={"epoch": 3})
print(store.get_meta("last_run"))
store.save_snapshot(step=200, payload=b"...")   # resume points
store.close()
```

`Store(path)` methods: `begin_run`, `end_run`, `log_event`, `set_meta`,
`get_meta`, `save_snapshot`, `load_snapshot`, `close`.

### 3.3 Adapters

See `docs/SDK.md`. Short version:

```python
from adapters.sdk import MorphosSource, check_conformance, load_adapter

src = load_adapter("examples/adapters/csv_sensor.py:CSVSensor")
assert check_conformance(src).ok
```

### 3.4 Rust trace acceleration

```python
from core.ingest.rust_trace import (RustTraceFile, RingBuffer,
                                    available, status)

assert available(), status()

tf = RustTraceFile("traces/run.trace")
stats = tf.scan()                 # min/max/mean, event count, throughput
win = tf.read_all()               # TraceWindow: columnar numpy arrays
print(len(win), win.xyz[:3])      # fields: tick, xyz (n,3), confidence, flags

ring = RingBuffer(capacity=1 << 16)
ring.push(tick=1, xyz=(0.1, 0.2, 0.3), confidence=0.9, flags=0)
win = ring.pop_batch(1024)        # -> TraceWindow
print(len(win), win.tick[:3], win.xyz[:3])
print(ring.stats())               # head/tail/dropped counters
```

`RingBuffer` is a **process-global singleton** — the first construction fixes the
capacity; call `ring.reset()` between independent runs. `ABI_VERSION` is checked
against the loaded library on import, and the struct layout is verified against
the Rust `offset_of!` guards, so a mismatched build fails loudly instead of
returning garbage.

### 3.5 Observer / anomaly detection

```python
from core.observer.dual_opinion import DualOpinion
```

Wraps two independent detectors and requires agreement, so a single
miscalibrated signal cannot raise an alarm. Calibrated after 600 steps in the
default configuration (threshold ≈ 3.3 × robust residual scale).

### 3.6 Counterfactual replay

```python
from core.counterfactual.engine import CounterfactualEngine
```

Re-runs the dynamics from a chosen step under a modified input, letting you ask
*"what would have happened if…"*. The Studio visualises these as branch lines
in the state space.

---

## 4. Studio HTTP/WebSocket API

Start with `morphos serve` (or `node visualization/server.js`). Default port
**8787**.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Studio shell (`index.html`) |
| `/studio.js`, `/studio.css`, `/vendor/three.module.min.js` | GET | Static assets |
| `/frames` | GET | `{"run": str, "count": int, "frames": [frame...]}` |
| `/api/vault` | GET | Same shape, summary of the loaded vault |
| `/api/ingest` | POST | `{"frames": [...]}` — append to the live vault |

WebSocket (same port, path `/`):

```jsonc
// client → server
{"type": "hello"}
{"type": "want-live"}

// server → client
{"type": "hello", "run": "run.frames.json"}
{"type": "frame", "frame": { /* frame schema from §2 */ }}
```

To stream a **live** run into the studio:

```bash
# terminal 1
morphos serve
# terminal 2
morphos run --steps 800 --feed http://localhost:8787
```

Frames are batched and POSTed from a background thread, so a slow or absent
studio can never perturb the control loop; failures are counted
(`studio_frames_failed`) and swallowed.

---

## 5. Errors

| Exception | Raised when |
|---|---|
| `AdapterContractError` | adapter returns wrong width / non-monotonic timestamps / `dim <= 0` |
| `RustTraceUnavailable` | `rust_trace` library missing or ABI mismatch |
| `TraceFormatError` | trace file has a bad magic, truncated record, or bad length |
| `AdapterError` | `load_adapter` spec is malformed or the class is missing |

All inherit from `Exception` and carry a message that names the offending
value, so they are safe to surface directly to a user.

---

## 6. Measured performance

From the reference machine (Windows, Python 3.12.9, 400-neuron ensemble):

| Metric | Value |
|---|---|
| Control step cost | **0.2 ms** (5 ms budget) |
| Ensemble firing rate | 72.4 Hz mean, 80.5 Hz peak |
| Residual convergence | 0.189 → 0.0014 over 2000 steps (135×) |
| Trace scan (Rust) | 80 M events/s |
| Ring throughput (Rust) | 154 M events/s |
| Python parse rate | 10.3 M events/s |
| CPU at 1 kHz / 10 kHz / 100 kHz | 0.021 % / 0.097 % / 0.44 % |
| Studio render (software GL) | 60 FPS, 799 frames, 2 forks |

---

*See `docs/SDK.md` for adapters and `docs/WHITEPAPER.md` for the design rationale.*
