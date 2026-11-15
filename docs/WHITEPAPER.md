# MORPHOS — A Neuro-Inspired Streaming State Observer

**Technical whitepaper · Phase V · v1.0**

---

## Abstract

MORPHOS is a streaming system that reconstructs the latent state of a
dynamical system from noisy, gappy, out-of-order observations, and then
*observes its own understanding*: it measures its prediction error, decides
whether it is still tracking, flags anomalies it cannot explain, and can replay
counterfactual branches to show what would have happened under a different
input.

The core is a spiking neural ensemble (Nengo) driven at a **5 ms control
cadence**, achieving a **0.2 ms** per-step cost, a residual that drops **135×**
during convergence (0.189 → 0.0014), and a steady firing rate of **72 Hz**.
High-rate ingest is offloaded to a Rust zero-copy trace engine that parses at
**80 M events/s** and spends **0.021 % CPU** at 1 kHz. The whole pipeline is
driven by a dependency-free Nim CLI with a one-click launcher, visualized by a
Three.js state-space studio, and extended by third parties through a
two-method adapter SDK.

The central claim is narrow and testable: **a small, biologically-plausible
predictive core, wrapped in disciplined engineering, can be a reliable observer
of real-time streams — and it can say when it is wrong.**

---

## 1. The problem

Observability tooling for live systems has a persistent gap. Metrics are
aggregated until they are too coarse to catch the transient that matters; logs
are textual and only useful after the fact; tracing samples so sparsely that
individual anomalies vanish. What an operator actually needs is a **state
estimate of the live process, with a calibrated confidence**, available at the
moment of decision.

Classical approaches fall into three camps, each with a characteristic failure:

| Approach | Strength | Failure mode |
|---|---|---|
| Thresholds / alerting rules | simple, explainable | brittle, alarm fatigue, no state model |
| Statistical filters (Kalman, ARIMA) | principled uncertainty | linearity/Gaussianity assumptions; degrades sharply on gaps and nonstationarity |
| Deep sequence models (LSTM, transformer) | expressive | opaque, heavy, latency-bound, hard to reason about when wrong |

MORPHOS takes a fourth position: **a predictive dynamical core small enough to
inspect, with the engineering around it strong enough to trust.** The neural
ensemble is not there for benchmark supremacy — it is there because it yields a
continuous, physically-interpretable state estimate whose *prediction error* is
itself a meaningful signal.

## 2. Architecture

```
 ┌──────────────┐
 │  adapters/   │   pull: (timestamp, vector) — you own this layer
 └──────┬───────┘
        ▼
 ┌──────────────┐   sanitize · align · confidence
 │ core/ingest  │   ring buffer · Rust trace mmap bridge
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │core/encoding │   features → Nengo ensemble activity
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │core/dynamics │   recurrent state · multiscale weave
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │core/prediction│  one-step forecast
 └──────┬───────┘
        ▼
 ┌──────────────┐   residual EMA · health · dual-opinion · breaker
 │core/observer │
 └──────┬───────┘
        ├──────────────▶ storage/wal   (binary, append-only, replayable)
        ├──────────────▶ storage/sqlite (queries, snapshots, resume)
        └──────────────▶ visualization  (WebSocket → 3D studio)
```

157 source modules, ~7 400 lines of Python plus 2 100 lines of Rust and 1 600
lines of Nim/JS.

### 2.1 Why pull, not push

Every stream in a real system can stall. A push interface forces the core to
either block (destroying the cadence) or buffer without bound (destroying the
memory budget). MORPHOS inverts the relationship: **the control loop asks for
the next 5 ms and continues with whatever exists.** A gap becomes lower
confidence, not a crash. This single decision is what makes the pipeline
survive the failure modes in §2.2 without special cases.

### 2.2 Fault tolerance as a first-class design axis

`adapters/chaos.py` injects the faults real systems actually exhibit — dropped
samples, gaussian noise, NaN/Inf, extreme spikes, multi-second dropouts — and
the whole downstream pipeline is required to continue untouched. The invariant
under test is not "no error" but **"the state estimate stays finite and the
confidence falls honestly."** A system that silently keeps a high confidence
while its input is garbage is worse than one that admits it is lost.

## 3. The predictive core

### 3.1 Representation

Features are encoded into a Nengo ensemble by tuning curves distributed over the
observed range. Two parameters govern behaviour:

- **recurrent gain** (default 0.5) — how strongly the state feeds back on
  itself. Too low and the system forgets; too high and it becomes an unstable
  integrator that amplifies noise.
- **time constant μ** (default 0.05) — the effective memory horizon of the
  population, in seconds.

With 400 neurons the ensemble reproduces a 3-D chaotic attractor (Lorenz) with
a mean firing rate of 72 Hz and a residual that converges over roughly 600
steps. Convergence is measured, not assumed:

```
resid_ema_first200: 0.189
resid_ema_last200 : 0.0014       (135× reduction)
mean_rate_hz      : 72.4
step cost         : 0.2 ms       (5 ms budget → 4 % utilisation)
```

### 3.2 The residual is the signal

The one-step prediction error `r = ‖actual_{t+1} − pred_t‖` is the primary
observable. It answers the operational question directly: *is the model still
explaining what it sees?* A rising residual precedes a visible anomaly because
the ensemble's forecast degrades before the raw signal leaves its normal band.
A low-pass EMA of the residual (α = 0.04 in the studio HUD) smooths step noise
while retaining a lead of tens of steps over threshold-based detection.

### 3.3 Dual-opinion anomaly detection

Single-detector anomaly schemes fail in two directions: too sensitive and the
operator ignores them, too conservative and they miss the event. MORPHOS runs
two independent opinions — a residual-magnitude test and a novelty test against
the compressed memory of past states — and requires **agreement** before
raising an alarm.

Both are calibrated on the first 600 steps, after which the threshold is set
from the *robust* residual scale (default threshold ≈ 3.30). On the benchmark
suite this yields **precision = recall = F1 = 1.000** — but the number matters
less than the structure: a single miscalibrated signal cannot fire an alarm
alone, so a systematic drift in one channel degrades sensitivity rather than
producing a flood of false positives.

### 3.4 Circuit breaker

When the residual leaves the calibrated band and stays out, the observer trips a
breaker: it stops trusting the ensemble's output, raises the flag in the WAL,
and lets the state be re-seeded rather than propagating a diverged estimate.
This is the difference between a model that is *wrong* and a model that is
*wrong and knows to stop*.

### 3.5 A bounded-memory trace of history

Naive state observers keep everything, so their memory grows without limit.
`core/memory/compressor.py` applies importance weighting to the state history —
low-information frames are discarded first, high-surprise frames retained — and
the associative store lets a compressed trace be searched by similarity rather
than by timestamp. The design target is **bounded memory with unbounded
run length**, which is what an always-on observer requires.

## 4. High-rate ingest: the Rust engine

The Python core cannot parse binary at instrumentation rates. The Rust engine
(`engines/rust_trace/`, ~2 100 lines) provides three things:

1. **A 19-byte record format.** Magic `MORPHOSE`, tick as `u48` paired with a
   `u16` confidence, x/y/z as 24-bit fixed point (scale 2 147 483.648, range
   ±3.906 25), flags at byte 18. The format is compact enough for sustained
   disk writes and fixed-size enough for zero-copy reads.
2. **mmap zero-copy reads.** `MappedBytes` exposes the file as a byte slice;
   `TraceIter` yields records without allocating. Scanning runs at **80 M
   events/s**, windowed reads at **154 M events/s** through the ring.
3. **A lock-free SPSC ring in shared memory.** A 128-byte header (magic
   `MORPHOSR`, head/tail indices, dropped counter) over fixed-size slots, so a
   producer and consumer in different processes exchange events without locks.

Crossing the FFI boundary is where such designs usually lose their gains, so the
ABI is versioned (`ABI_VERSION = 5`), the struct layouts are pinned with
compile-time `offset_of!` assertions on the Rust side and re-verified against
them on the Python side at import. **A mismatched build refuses to load**
instead of returning plausible-looking garbage.

Measured through the Python bridge: **10.3 M events/s** and CPU of
**0.021 % at 1 kHz, 0.097 % at 10 kHz, 0.44 % at 100 kHz** — the acceptance
criterion was < 2 %.

## 5. Counterfactual replay

Observation answers *what happened*; operators need *what would have happened*.
`core/counterfactual/` can fork the dynamics at any step, apply a modified
input (a gain change, injected noise, a step perturbation, a dropout), and
replay forward. Because the core is a dynamical system rather than a
feed-forward network, the fork is exact — the perturbed trajectory diverges
only through the system's own sensitivity, which is precisely the quantity of
interest.

The studio renders these forks as branch lines in state space, so divergence is
seen as spatial separation rather than read from a table (§7).

## 6. The Studio (T5.3)

A zero-dependency Node server (hand-rolled RFC 6455 WebSocket hub, no npm
packages at runtime) serves a Three.js client. Rendering budget: 60 FPS with
trajectory, prediction ghosts, residual ribbon, and up to four live branches,
verified at **60 FPS under software rendering (SwiftShader)** — hardware GL
leaves substantial headroom.

Four encodings share one scene, each chosen so a different question is answered
at a glance:

| Layer | Question answered |
|---|---|
| Trajectory line | Where has the state been, and is it coherent? |
| Prediction ghosts | Is the forecast leading or lagging the truth? |
| Residual ribbon (height = error) | Where did understanding degrade? |
| Branch lines | How sensitive is the future to this moment? |

Colour is reserved for meaning: teal = nominal, amber = low confidence,
red = flagged anomaly, purple/violet = counterfactual branch. The live feed is
POSTed from a background thread so a stalled studio can never perturb the
control loop.

## 7. Extensibility (T5.4)

The adapter contract is **two methods and two properties**:

```python
dim                     # number of channels
advance_to(t_target)    # make samples up to t available, non-blocking
read_pending()          # (timestamp, vector) pairs since last call
stats                   # optional diagnostics
```

Everything else — sanitising, alignment, encoding, dynamics, observation,
storage, visualisation — is provided. The claim that a third party can write a
working adapter in under an hour is testable, not rhetorical: the template
adapter is ~50 lines, and `python -m adapters.sdk validate` drives any new
adapter through 200 control steps and checks every contract rule (constant
width, monotonic timestamps, sub-5 ms latency, non-empty output). The harness
is itself tested against deliberately broken adapters, so a passing verdict
means something.

## 8. Engineering discipline

Several of the most consequential bugs found during development were *silent* —
they produced plausible numbers rather than exceptions. They are recorded here
because the practices that caught them are the ones that make the claims above
credible:

- **A missing `#[repr(C)]`** let rustc reorder `ScanStats`, so Python read every
  bound from the wrong offset. Caught by adding compile-time layout assertions
  and a runtime ABI check that refuses to load on mismatch.
- **Ring slots were 24 bytes while events were 32**, silently dropping the flags
  byte. Caught by a wrapping test that checks ordering *and* payload.
- **`ScanStats` min/max initialised from `Default` (0.0)** rather than ±∞,
  producing wrong bounds for all-negative data. Caught by a sign-aware fixture.
- **`expandFilename` raising on non-existent paths** in the Nim CLI, which made
  `init` fail in exactly the case it exists for — a fresh directory.

The general lesson, and the reason the test suites include negative cases: **a
system that reports on its own health must be held to a stricter standard than
one that merely produces output.** Hence layout guards, ABI versioning,
conformance harnesses that are themselves tested, and honest confidence
degradation instead of silent optimism.

## 9. Measured results

| Dimension | Metric | Value |
|---|---|---|
| Core loop | step cost | 0.2 ms (5 ms budget) |
| Convergence | residual reduction | 0.189 → 0.0014 (135×) |
| Ensemble | firing rate | 72.4 Hz mean, 80.5 Hz peak |
| Anomaly | precision / recall / F1 | 1.000 / 1.000 / 1.000 |
| Rust scan | throughput | 80 M events/s |
| Rust ring | throughput | 154 M events/s |
| Bridge | parse rate | 10.3 M events/s |
| CPU | at 1 / 10 / 100 kHz | 0.021 % / 0.097 % / 0.44 % |
| Studio | frame rate | 60 FPS (software GL) |
| Tests | Rust / SDK / Studio | 4 + 13 / 12 / 12 |

## 10. Limitations

Stated plainly, because the credibility of the rest depends on it:

- **The core is small by design.** 400 neurons reconstruct a 3-D chaotic system
  well; they will not reconstruct high-dimensional turbulence. The architecture
  scales, the demonstrated operating point does not.
- **The benchmark is synthetic.** P/R/F1 = 1.000 is on a curated anomaly suite
  with injected faults. Real telemetry is messier; the dual-opinion structure is
  what should survive, not the exact number.
- **Counterfactual forks are exact only within the ensemble's validity.** Once a
  perturbed branch leaves the region the tuning curves cover, its trajectory is
  extrapolation, and the studio labels it as a counterfactual rather than a
  prediction.
- **The studio's viewer-side branch synthesis** (`perturbSeries` in
  `studio.js`) approximates a Lorenz-like flow for interactive exploration; it
  is a *visualisation* of sensitivity, not a replay of the engine's own fork.
  Exact counterfactual replay goes through `core/counterfactual/`.

## 11. Conclusion

MORPHOS demonstrates that a compact predictive core, engineered with attention
to the places silent failures hide — FFI layout, ring wraparound, initialisation
defaults, path handling — can serve as a real-time state observer that is fast
(0.2 ms/step), frugal (0.02 % CPU at 1 kHz), honest about its own uncertainty,
and inspectable at every layer from a 19-byte binary record to a 3D branch in
state space.

The broader point is about **where the intelligence belongs.** Not in a larger
model, but in the arrangement: predictive dynamics for state, residual for
self-assessment, agreement of independent opinions for alarms, and an engineering
discipline strict enough that the numbers can be believed.

---

## Appendix A — Repository layout

```
MORPHOS/
├─ bin/               morphos.exe (Nim CLI)  ·  los.exe (one-click launcher)
├─ cli/               Nim master CLI (config, exec, commands)
├─ core/              ingest · encoding · dynamics · prediction · observer ·
│                     memory · counterfactual
├─ adapters/          source adapters + sdk.py (the public contract)
├─ engines/rust_trace/  Rust zero-copy parser, ring buffer, C ABI, benches
├─ storage/           WAL (binary) + SQLite
├─ visualization/     Node hub + Three.js studio
├─ experiments/       reproducible experiment definitions
├─ examples/adapters/ template.py · csv_sensor.py
├─ docs/              WHITEPAPER.md · SDK.md · API.md
└─ tests/             contract, Rust, SDK, studio tests
```

## Appendix B — Reproduce the headline numbers

```bash
morphos doctor                              # environment + ABI check
morphos init                                # project scaffold
morphos run --steps 2000                    # core loop, residual convergence
morphos ingest --mkstream --out traces/r.trace --events 200000
morphos ingest --trace traces/r.trace       # Rust scan throughput
cargo test --release                        # Rust unit tests (4)
python tests/test_phase5_rust.py            # bridge + ABI (13)
python tests/test_phase5_sdk.py             # adapter SDK (12)
morphos serve                               # studio on :8787
node visualization/test_studio.mjs          # studio smoke (12)
```

---

*License: MIT. See `LICENSE`.*
