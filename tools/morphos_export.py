"""Export MORPHOS trajectory data (T5.1 support tool / T5.3 data source).

Reads the binary WAL written by main.py and emits it in a form the studio,
notebooks or spreadsheets can consume:

    json   - array of frames, direct WebSocket/JS payload
    ndjson - one frame per line, streaming friendly
    csv    - flat table, one row per control step
    html   - a self-contained interactive 3D viewer (no server needed)

Usage:
    python tools/morphos_export.py --wal data/trajectory.wal --format json
    python tools/morphos_export.py --format html --out studio.html
    python tools/morphos_export.py --format csv --limit 5000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Iterator, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from storage.wal import iter_records  # noqa: E402

FLAG_LOW_CONF = 1 << 6
FLAG_BREAKER = 1 << 7


def _frame(rec: dict, step: int) -> dict:
    """Convert a WAL record into the studio's frame schema."""
    resid = rec["residual"]
    return {
        "step": step,
        "t": round(rec["t_sim"], 6),
        "conf": round(rec["conf"], 4),
        "a": [round(float(v), 5) for v in rec["actual"]],
        "p": [round(float(v), 5) for v in rec["pred"]],
        "r": round(resid, 6) if math.isfinite(resid) else None,
        "rate": round(rec["rate"], 3),
        "flags": int(rec["flags"]),
        "low_conf": bool(rec["flags"] & FLAG_LOW_CONF),
        "breaker": bool(rec["flags"] & FLAG_BREAKER),
    }


def read_frames(wal_path: str, limit: int = 0) -> Iterator[dict]:
    """Yield studio frames from a WAL file, tolerating a truncated tail."""
    if not os.path.exists(wal_path):
        raise SystemExit(f"WAL not found: {wal_path}")
    for i, rec in enumerate(iter_records(wal_path)):
        if limit and i >= limit:
            break
        yield _frame(rec, i)


def to_json(frames: list[dict]) -> str:
    return json.dumps({"schema": "morphos.frames/1",
                       "count": len(frames),
                       "frames": frames}, separators=(",", ":"))


CSV_HEADER = ("step,t,conf,actual_x,actual_y,actual_z,"
              "pred_x,pred_y,pred_z,residual,rate_hz,flags,low_conf,breaker")


def to_csv(frames: list[dict]) -> str:
    lines = [CSV_HEADER]
    for f in frames:
        a, p = f["a"], f["p"]
        lines.append(",".join(str(v) for v in (
            f["step"], f["t"], f["conf"],
            a[0], a[1], a[2], p[0], p[1], p[2],
            "" if f["r"] is None else f["r"],
            f["rate"], f["flags"], int(f["low_conf"]), int(f["breaker"]),
        )))
    return "\n".join(lines) + "\n"


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>MORPHOS trajectory export</title>
<style>
  html,body{margin:0;height:100%;background:#080a10;color:#c8d2e0;
    font:13px/1.5 ui-monospace,Consolas,monospace;overflow:hidden}
  canvas{display:block}
  #hud{position:fixed;top:12px;left:12px;background:#0009;padding:10px 14px;
    border:1px solid #1e2a3a;border-radius:6px;pointer-events:none}
  #hud b{color:#4fd1c5}
  #bar{position:fixed;bottom:14px;left:12px;right:12px;display:flex;gap:10px;
    align-items:center;background:#0009;padding:8px 12px;border-radius:6px;
    border:1px solid #1e2a3a}
  input[type=range]{flex:1}
</style></head><body>
<canvas id="c"></canvas>
<div id="hud">step <b id="s">0</b> &nbsp; t <b id="t">0.000</b>s
  &nbsp; residual <b id="r">-</b> &nbsp; rate <b id="hz">0</b>Hz</div>
<div id="bar">
  <button id="play">pause</button>
  <input id="scrub" type="range" min="0" max="100" value="100">
  <span id="pos">0 / 0</span>
</div>
<script>
const FRAMES = __FRAMES__;
const c = document.getElementById('c'), ctx = c.getContext('2d');
let W, H, idx = FRAMES.length - 1, playing = true;
function resize() {
  W = c.width = innerWidth * devicePixelRatio;
  H = c.height = innerHeight * devicePixelRatio;
  c.style.width = innerWidth + 'px'; c.style.height = innerHeight + 'px';
  draw();
}
addEventListener('resize', resize);

// Fit the trajectory bounds once so the projection is stable while playing.
let lo = [1e9,1e9,1e9], hi = [-1e9,-1e9,-1e9];
for (const f of FRAMES) for (let k = 0; k < 3; k++) {
  lo[k] = Math.min(lo[k], f.a[k]); hi[k] = Math.max(hi[k], f.a[k]);
}
const span = Math.max(1e-6, ...hi.map((h, k) => h - lo[k]));
const ang = 0.6;
function project(v) {
  const x = (v[0] - lo[0]) / span - 0.5;
  const y = (v[1] - lo[1]) / span - 0.5;
  const z = (v[2] - lo[2]) / span - 0.5;
  // cheap isometric-ish rotation: yaw about the vertical axis, slight pitch
  const cx = Math.cos(ang), sx = Math.sin(ang);
  const rx = x * cx - y * sx, ry = x * sx + y * cx;
  const sc = Math.min(W, H) * 0.62;
  return [W / 2 + rx * sc, H / 2 + (ry * 0.75 - z * 0.55) * sc];
}
function draw() {
  ctx.clearRect(0, 0, W, H);
  const from = Math.max(0, idx - 3000);
  ctx.beginPath();
  for (let i = from; i <= idx; i++) {
    const p = project(FRAMES[i].a);
    i === from ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]);
  }
  ctx.strokeStyle = '#4fd1c5'; ctx.lineWidth = 1.4 * devicePixelRatio;
  ctx.stroke();
  // prediction ghost
  const f = FRAMES[idx];
  if (f) {
    const pa = project(f.a), pp = project(f.p);
    ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pp[0], pp[1]);
    ctx.strokeStyle = '#f6ad55'; ctx.stroke();
    ctx.beginPath(); ctx.arc(pa[0], pa[1], 3.5 * devicePixelRatio, 0, 7);
    ctx.fillStyle = '#e6edf3'; ctx.fill();
    document.getElementById('s').textContent = f.step;
    document.getElementById('t').textContent = f.t.toFixed(3);
    document.getElementById('r').textContent =
      f.r === null ? 'nan' : f.r.toFixed(4);
    document.getElementById('hz').textContent = f.rate.toFixed(0);
    document.getElementById('pos').textContent =
      (idx + 1) + ' / ' + FRAMES.length;
    document.getElementById('scrub').value =
      FRAMES.length > 1 ? (idx / (FRAMES.length - 1)) * 100 : 100;
  }
}
function tick() {
  if (playing && FRAMES.length) idx = (idx + 1) % FRAMES.length;
  draw(); requestAnimationFrame(tick);
}
document.getElementById('play').onclick = e => {
  playing = !playing; e.target.textContent = playing ? 'pause' : 'play';
};
document.getElementById('scrub').oninput = e => {
  idx = Math.round((e.target.value / 100) * (FRAMES.length - 1));
  draw();
};
resize(); tick();
</script></body></html>
"""


def to_html(frames: list[dict]) -> str:
    # Embed rounded values only; a 50k-frame export stays a manageable file.
    payload = json.dumps(frames, separators=(",", ":"))
    return HTML_TEMPLATE.replace("__FRAMES__", payload)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export MORPHOS trajectory data")
    ap.add_argument("--wal", default=os.path.join(ROOT, "data",
                                                  "trajectory.wal"))
    ap.add_argument("--format", choices=["json", "ndjson", "csv", "html"],
                    default="json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    frames = list(read_frames(args.wal, args.limit))
    if not frames:
        print(f"no records in {args.wal}", file=sys.stderr)
        return 1

    if args.format == "json":
        text = to_json(frames)
    elif args.format == "ndjson":
        text = "\n".join(json.dumps(f, separators=(",", ":"))
                         for f in frames) + "\n"
    elif args.format == "csv":
        text = to_csv(frames)
    else:
        text = to_html(frames)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {len(frames)} frames -> {args.out} "
              f"({len(text) / 1024:.1f} KiB)")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
