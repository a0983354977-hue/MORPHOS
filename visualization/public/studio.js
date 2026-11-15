/**
 * MORPHOS Studio renderer (T5.3)
 *
 * Renders four things into one shared 3D scene:
 *
 *   1. the decoded state-space trajectory (the neural ensemble's output),
 *   2. the one-step prediction as a ghost segment, showing residual direction,
 *   3. a residual ribbon whose height encodes |prediction error|,
 *   4. a counterfactual branch tree: forks spawned from a chosen step replay
 *      perturbed dynamics in a different colour, so divergences are visible as
 *      spatial separation rather than as numbers in a table.
 *
 * Data arrives either from the vault (a static frame array) or live over the
 * WebSocket hub. Both paths feed the same ring buffer, so replay and live look
 * identical to the renderer.
 */

import * as THREE from 'three';

/* ------------------------------------------------------------------ */
/* DOM handles                                                        */
/* ------------------------------------------------------------------ */
const $ = (id) => document.getElementById(id);
const ui = {
  canvas: $('gl'),
  boot: $('boot'), bootMsg: $('boot-msg'),
  dot: $('status-dot'), run: $('run-name'),
  step: $('h-step'), t: $('h-t'), resid: $('h-resid'), ema: $('h-ema'),
  rate: $('h-rate'), conf: $('h-conf'), anom: $('h-anom'), fps: $('h-fps'),
  play: $('btn-play'), live: $('btn-live'), scrub: $('scrub'), pos: $('pos'),
  speed: $('speed'), trail: $('trail'),
  tPred: $('t-pred'), tResid: $('t-resid'), tBranch: $('t-branch'),
  tGrid: $('t-grid'),
  branchMode: $('branch-mode'), branchBtn: $('btn-branch'),
};

/* ------------------------------------------------------------------ */
/* scene                                                              */
/* ------------------------------------------------------------------ */
const renderer = new THREE.WebGLRenderer({
  canvas: ui.canvas, antialias: true, alpha: false,
});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0x070910, 1);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x070910, 0.035);

const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.1, 500);
camera.position.set(3.4, 2.6, 4.2);

const controls = makeOrbit(camera, ui.canvas, new THREE.Vector3(0, 0, 0));

scene.add(new THREE.AmbientLight(0x8fa8c8, 0.85));
const key = new THREE.DirectionalLight(0xcfe4ff, 0.9);
key.position.set(4, 6, 5);
scene.add(key);
const rim = new THREE.DirectionalLight(0x4fd1c5, 0.35);
rim.position.set(-5, -2, -4);
scene.add(rim);

// grid floor: spatial reference for the 3D extent of the trajectory
const grid = new THREE.GridHelper(10, 20, 0x1b2636, 0x121a26);
grid.position.y = -1.6;
grid.material.transparent = true;
grid.material.opacity = 0.6;
scene.add(grid);

/* ------------------------------------------------------------------ */
/* orbit controls (small in-file arcball: no extra dependency)         */
/* ------------------------------------------------------------------ */
function makeOrbit(cam, dom, target) {
  const state = { theta: 0.72, phi: 1.05, radius: 6.2, dragging: false,
                  px: 0, py: 0, target };
  const apply = () => {
    const { theta, phi, radius } = state;
    cam.position.set(
      target.x + radius * Math.sin(phi) * Math.cos(theta),
      target.y + radius * Math.cos(phi),
      target.z + radius * Math.sin(phi) * Math.sin(theta),
    );
    cam.lookAt(target);
  };
  dom.addEventListener('pointerdown', (e) => {
    state.dragging = true; state.px = e.clientX; state.py = e.clientY;
    dom.setPointerCapture(e.pointerId);
  });
  dom.addEventListener('pointerup', (e) => {
    state.dragging = false;
    try { dom.releasePointerCapture(e.pointerId); } catch (_) {}
  });
  dom.addEventListener('pointermove', (e) => {
    if (!state.dragging) return;
    state.theta -= (e.clientX - state.px) * 0.006;
    state.phi = Math.max(0.12, Math.min(Math.PI - 0.12,
      state.phi - (e.clientY - state.py) * 0.006));
    state.px = e.clientX; state.py = e.clientY;
    apply();
  });
  dom.addEventListener('wheel', (e) => {
    e.preventDefault();
    state.radius = Math.max(1.6, Math.min(40,
      state.radius * (1 + Math.sign(e.deltaY) * 0.09)));
    apply();
  }, { passive: false });
  apply();
  return { state, apply, target };
}

/* ------------------------------------------------------------------ */
/* data model                                                         */
/* ------------------------------------------------------------------ */
const MAX_TRAIL = 8000;
const frames = [];            // {step,t,a[3],p[3],r,rate,conf,flags,anom}
let bounds = { min: [1e9, 1e9, 1e9], max: [-1e9, -1e9, -1e9] };
let scale = 1, center = new THREE.Vector3();

function normalize(v) {
  // Fit into a unit cube so camera framing is stable across runs.
  return [
    (v[0] - center.x) * scale,
    (v[1] - center.y) * scale,
    (v[2] - center.z) * scale,
  ];
}

function recomputeBounds() {
  if (!frames.length) return;
  bounds = { min: [1e9, 1e9, 1e9], max: [-1e9, -1e9, -1e9] };
  for (const f of frames) {
    for (let k = 0; k < 3; k++) {
      bounds.min[k] = Math.min(bounds.min[k], f.a[k]);
      bounds.max[k] = Math.max(bounds.max[k], f.a[k]);
    }
  }
  center.set(
    (bounds.min[0] + bounds.max[0]) / 2,
    (bounds.min[1] + bounds.max[1]) / 2,
    (bounds.min[2] + bounds.max[2]) / 2,
  );
  let ext = 0;
  for (let k = 0; k < 3; k++) ext = Math.max(ext, bounds.max[k] - bounds.min[k]);
  scale = ext > 1e-9 ? 2.6 / ext : 1;
}

function pushFrame(f) {
  frames.push(f);
  if (frames.length > MAX_TRAIL * 2) frames.splice(0, frames.length - MAX_TRAIL * 2);
  recomputeBounds();
}

/* ------------------------------------------------------------------ */
/* geometry: trail line                                              */
/* ------------------------------------------------------------------ */
const TRAIL_CAP = MAX_TRAIL;
const trailGeo = new THREE.BufferGeometry();
const trailPos = new Float32Array(TRAIL_CAP * 3);
const trailCol = new Float32Array(TRAIL_CAP * 3);
trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
trailGeo.setAttribute('color', new THREE.BufferAttribute(trailCol, 3));
trailGeo.setDrawRange(0, 0);
const trailMat = new THREE.LineBasicMaterial({
  vertexColors: true, linewidth: 1, transparent: true, opacity: 0.95,
});
const trail = new THREE.Line(trailGeo, trailMat);
scene.add(trail);

// prediction ghost segments (per-step short lines)
const predCap = 1500;
const predGeo = new THREE.BufferGeometry();
const predPos = new Float32Array(predCap * 6);
const predCol = new Float32Array(predCap * 6);
predGeo.setAttribute('position', new THREE.BufferAttribute(predPos, 3));
predGeo.setAttribute('color', new THREE.BufferAttribute(predCol, 3));
predGeo.setDrawRange(0, 0);
const predLines = new THREE.LineSegments(predGeo, new THREE.LineBasicMaterial({
  vertexColors: true, transparent: true, opacity: 0.55,
}));
scene.add(predLines);

// residual ribbon: one vertical segment per step, taller = bigger error
const residGeo = new THREE.BufferGeometry();
const residPos = new Float32Array(TRAIL_CAP * 6);
const residCol = new Float32Array(TRAIL_CAP * 6);
residGeo.setAttribute('position', new THREE.BufferAttribute(residPos, 3));
residGeo.setAttribute('color', new THREE.BufferAttribute(residCol, 3));
residGeo.setDrawRange(0, 0);
const residRibbon = new THREE.LineSegments(residGeo, new THREE.LineBasicMaterial({
  vertexColors: true, transparent: true, opacity: 0.28,
}));
scene.add(residRibbon);

// head marker
const head = new THREE.Mesh(
  new THREE.SphereGeometry(0.035, 16, 12),
  new THREE.MeshStandardMaterial({ color: 0xe6edf3, emissive: 0x4fd1c5,
                                   emissiveIntensity: 0.9 }),
);
scene.add(head);

/* ------------------------------------------------------------------ */
/* counterfactual branches                                            */
/* ------------------------------------------------------------------ */
const branches = [];   // {mode, originStep, pts:[{a}], line, head, geo}

function perturbSeries(origin, mode, len) {
  // Synthesise a counterfactual continuation from the origin state. This is a
  // viewer-side approximation of core/counterfactual's replay: it shows how
  // sensitive the trajectory is, and is labelled as such in the UI.
  const pts = [];
  let s = [...origin];
  let prev = { x: s[0], y: s[1], z: s[2] };
  let phase = 0.0;
  for (let i = 0; i < len; i++) {
    phase += 0.21;
    const vx = s[0], vy = s[1], vz = s[2];
    let nx = vx + (vy - vx) * 0.012;
    let ny = vy + (vx * (2.0 - vz) - vy) * 0.012;
    let nz = vz + (vx * vy - 0.3 * vz) * 0.012;
    switch (mode) {
      case 'gain':
        nx *= 1.6; ny *= 1.6; nz *= 1.6;
        break;
      case 'noise':
        nx += (Math.sin(phase * 7.3) * 0.35);
        ny += (Math.cos(phase * 5.1) * 0.35);
        nz += (Math.sin(phase * 3.7) * 0.35);
        break;
      case 'step':
        nx += 0.8; ny += 0.8;
        break;
      case 'dropout':
        if (i > 8 && i < 40) { nx = pts[pts.length - 1].a[0];
                               ny = pts[pts.length - 1].a[1];
                               nz = pts[pts.length - 1].a[2]; }
        break;
      default: break;
    }
    s = [nx, ny, nz];
    pts.push({ a: [nx, ny, nz] });
  }
  return pts;
}

const branchColors = [0xb794f4, 0xf687b3, 0x7f9cf5, 0xf6c453];

function spawnBranch() {
  if (!frames.length) return;
  const mode = ui.branchMode.value;
  if (mode === 'none') {
    ui.branchBtn.textContent = 'pick a mode';
    setTimeout(() => { ui.branchBtn.textContent = 'fork'; }, 1200);
    return;
  }
  const idx = Math.min(cursor, frames.length - 1);
  const origin = frames[idx].a;
  const pts = perturbSeries(origin, mode, 160);
  const color = branchColors[branches.length % branchColors.length];

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(
    new Float32Array((pts.length + 1) * 3), 3));
  const line = new THREE.Line(geo, new THREE.LineBasicMaterial({
    color, transparent: true, opacity: 0.85,
  }));
  geo.setDrawRange(0, 0);
  const headMesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.026, 12, 10),
    new THREE.MeshStandardMaterial({ color, emissive: color,
                                     emissiveIntensity: 0.7 }),
  );
  scene.add(line);
  scene.add(headMesh);
  branches.push({ mode, originStep: frames[idx].step, pts, line, headMesh,
                  geo, color, grow: 0 });
  if (branches.length > 4) {
    const old = branches.shift();
    scene.remove(old.line);
    scene.remove(old.headMesh);
    old.geo.dispose();
    old.line.material.dispose();
    old.headMesh.geometry.dispose();
    old.headMesh.material.dispose();
  }
}

function updateBranches() {
  for (const b of branches) {
    b.grow = Math.min(b.pts.length, b.grow + 3);
    const pos = b.geo.attributes.position.array;
    // start the line at the origin frame so the fork visibly attaches
    const originFrame = frames.find((f) => f.step === b.originStep);
    const base = originFrame ? normalize(originFrame.a) : normalize(b.pts[0].a);
    pos[0] = base[0]; pos[1] = base[1]; pos[2] = base[2];
    for (let i = 0; i < b.grow; i++) {
      const p = normalize(b.pts[i].a);
      pos[(i + 1) * 3] = p[0];
      pos[(i + 1) * 3 + 1] = p[1];
      pos[(i + 1) * 3 + 2] = p[2];
    }
    b.geo.setDrawRange(0, b.grow + 1);
    b.geo.attributes.position.needsUpdate = true;
    if (b.grow > 0) {
      const last = normalize(b.pts[b.grow - 1].a);
      b.headMesh.position.set(last[0], last[1], last[2]);
    } else {
      b.headMesh.position.set(base[0], base[1], base[2]);
    }
  }
}

/* ------------------------------------------------------------------ */
/* frame -> GPU upload                                                */
/* ------------------------------------------------------------------ */
let cursor = 0;
let playing = true;
let liveMode = false;
let trailLen = 2400;
let speed = 12;

function rebuildBuffers() {
  const n = frames.length;
  if (n === 0) {
    trailGeo.setDrawRange(0, 0);
    predGeo.setDrawRange(0, 0);
    residGeo.setDrawRange(0, 0);
    return;
  }
  const end = Math.min(n, cursor + 1);
  const start = Math.max(0, end - trailLen);

  let vp = 0;
  for (let i = start; i < end; i++) {
    const f = frames[i];
    const p = normalize(f.a);
    trailPos[vp * 3] = p[0];
    trailPos[vp * 3 + 1] = p[1];
    trailPos[vp * 3 + 2] = p[2];
    // colour: teal normally, red when flagged anomalous, amber on low conf
    let c = [0.31, 0.82, 0.77];
    if (f.anom) c = [0.99, 0.51, 0.51];
    else if (f.conf != null && f.conf < 0.2) c = [0.96, 0.68, 0.33];
    trailCol[vp * 3] = c[0];
    trailCol[vp * 3 + 1] = c[1];
    trailCol[vp * 3 + 2] = c[2];
    vp++;
  }
  trailGeo.setDrawRange(0, vp);
  trailGeo.attributes.position.needsUpdate = true;
  trailGeo.attributes.color.needsUpdate = true;

  // prediction ghosts (subsample to the last `predCap` steps)
  let sp = 0;
  const pstart = Math.max(start, end - predCap);
  for (let i = pstart; i < end; i++) {
    const f = frames[i];
    if (!f.p || !f.a) continue;
    const a = normalize(f.a), q = normalize(f.p);
    predPos[sp * 6] = a[0]; predPos[sp * 6 + 1] = a[1]; predPos[sp * 6 + 2] = a[2];
    predPos[sp * 6 + 3] = q[0]; predPos[sp * 6 + 4] = q[1]; predPos[sp * 6 + 5] = q[2];
    for (let k = 0; k < 2; k++) {
      predCol[sp * 6 + k * 3] = 0.96;
      predCol[sp * 6 + k * 3 + 1] = 0.68;
      predCol[sp * 6 + k * 3 + 2] = 0.33;
    }
    sp++;
  }
  predGeo.setDrawRange(0, sp * 2);
  predGeo.attributes.position.needsUpdate = true;
  predGeo.attributes.color.needsUpdate = true;

  // residual ribbon height = |residual| scaled to the fitted extent
  let rp = 0;
  let maxR = 1e-9;
  for (let i = start; i < end; i++) {
    const r = frames[i].r;
    if (r != null && isFinite(r)) maxR = Math.max(maxR, r);
  }
  for (let i = start; i < end; i++) {
    const f = frames[i];
    const r = f.r;
    if (r == null || !isFinite(r)) continue;
    const p = normalize(f.a);
    const h = Math.min(1.4, (r / maxR) * 1.4);
    residPos[rp * 6] = p[0]; residPos[rp * 6 + 1] = p[1]; residPos[rp * 6 + 2] = p[2];
    residPos[rp * 6 + 3] = p[0]; residPos[rp * 6 + 4] = p[1] + h; residPos[rp * 6 + 5] = p[2];
    const t = Math.min(1, r / maxR);
    for (let k = 0; k < 2; k++) {
      residCol[rp * 6 + k * 3] = 0.35 + 0.6 * t;
      residCol[rp * 6 + k * 3 + 1] = 0.45 * (1 - t);
      residCol[rp * 6 + k * 3 + 2] = 0.5 * (1 - t) + 0.2;
    }
    rp++;
  }
  residGeo.setDrawRange(0, rp * 2);
  residGeo.attributes.position.needsUpdate = true;
  residGeo.attributes.color.needsUpdate = true;

  // head + camera target follow the current point
  const cf = frames[Math.min(cursor, n - 1)];
  if (cf) {
    const p = normalize(cf.a);
    head.position.set(p[0], p[1], p[2]);
  }
}

/* ------------------------------------------------------------------ */
/* HUD                                                                */
/* ------------------------------------------------------------------ */
let emaResid = null;
let fpsAcc = 0, fpsFrames = 0, fps = 0;

function updateHud() {
  const n = frames.length;
  const f = frames[Math.min(cursor, n - 1)];
  if (!f) return;
  ui.step.textContent = f.step ?? cursor;
  ui.t.textContent = (f.t ?? 0).toFixed(3) + 's';
  ui.resid.textContent = f.r == null ? 'nan' : f.r.toFixed(4);
  ui.resid.className = 'val ' + (f.r == null ? 'bad' : f.r > 0.05 ? 'warn' : 'good');
  emaResid = emaResid == null ? (f.r ?? 0) : emaResid * 0.96 + (f.r ?? 0) * 0.04;
  ui.ema.textContent = emaResid.toFixed(4);
  ui.rate.textContent = (f.rate ?? 0).toFixed(0) + ' Hz';
  ui.conf.textContent = (f.conf ?? 0).toFixed(2);
  ui.conf.className = 'val ' + ((f.conf ?? 1) < 0.2 ? 'warn' : '');
  ui.anom.textContent = f.anom ? f.anom : 'none';
  ui.anom.className = 'val ' + (f.anom ? 'bad' : '');
  ui.pos.textContent = (cursor + 1) + ' / ' + n;
  ui.scrub.max = String(Math.max(0, n - 1));
  ui.scrub.value = String(Math.min(cursor, n - 1));
  ui.fps.textContent = fps.toFixed(0);
}

/* ------------------------------------------------------------------ */
/* data loading                                                       */
/* ------------------------------------------------------------------ */
async function loadVault() {
  ui.bootMsg.textContent = 'loading trajectory …';
  const res = await fetch('/frames');
  const data = await res.json();
  const list = data.frames || [];
  if (!list.length) {
    ui.bootMsg.innerHTML =
      'no trajectory data yet.<br>' +
      'run <b>morphos run --steps 600</b> then <b>morphos export --format json --out data/run.frames.json</b>';
    ui.dot.className = 'dot off';
    ui.run.textContent = 'empty';
    return false;
  }
  for (const raw of list) frames.push(normalizeFrame(raw));
  recomputeBounds();
  cursor = frames.length - 1;
  ui.run.textContent = data.run || 'vault';
  ui.dot.className = 'dot vault';
  ui.boot.style.display = 'none';
  return true;
}

function normalizeFrame(raw) {
  // The export tool already emits {step,t,a,p,r,rate,conf,flags,low_conf,breaker}.
  return {
    step: raw.step,
    t: raw.t,
    a: raw.a || [0, 0, 0],
    p: raw.p || raw.a || [0, 0, 0],
    r: raw.r == null ? null : raw.r,
    rate: raw.rate ?? 0,
    conf: raw.conf ?? 1,
    flags: raw.flags ?? 0,
    anom: raw.anom || (raw.low_conf ? 'low-conf' : null),
  };
}

/* ------------------------------------------------------------------ */
/* WebSocket live feed                                                */
/* ------------------------------------------------------------------ */
let ws = null;

function connectLive() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/`);
  ws.onopen = () => {
    ws.send(JSON.stringify({ type: 'hello' }));
    ui.live.classList.add('on');
  };
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (_) { return; }
    if (msg.type === 'frame' && msg.frame) {
      pushFrame(normalizeFrame(msg.frame));
      if (liveMode) cursor = frames.length - 1;
    } else if (msg.type === 'hello') {
      ui.run.textContent = msg.run || 'live';
      ui.dot.className = 'dot live';
    }
  };
  ws.onclose = () => {
    ui.live.classList.remove('on');
    if (liveMode) {
      ui.dot.className = 'dot off';
      setTimeout(connectLive, 2000);
    }
  };
  ws.onerror = () => {};
}

/* ------------------------------------------------------------------ */
/* main loop                                                          */
/* ------------------------------------------------------------------ */
let lastFrameTime = performance.now();

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = Math.min(100, now - lastFrameTime);
  lastFrameTime = now;
  fpsAcc += dt; fpsFrames++;
  if (fpsAcc >= 400) {
    fps = (fpsFrames * 1000) / fpsAcc;
    fpsAcc = 0; fpsFrames = 0;
  }

  if (playing && frames.length) {
    if (liveMode) {
      cursor = frames.length - 1;
    } else {
      // speed = steps per second
      const advance = (speed * dt) / 1000;
      cursor += advance;
      if (cursor >= frames.length - 1) cursor = frames.length - 1;
    }
  }

  updateBranches();
  rebuildBuffers();
  updateHud();

  // keep the camera looking at the trajectory centre as it grows
  controls.target.copy(new THREE.Vector3(0, 0, 0));
  controls.apply();

  renderer.render(scene, camera);
}

/* ------------------------------------------------------------------ */
/* UI wiring                                                          */
/* ------------------------------------------------------------------ */
ui.play.onclick = () => {
  playing = !playing;
  ui.play.textContent = playing ? '❚❚' : '▶';
  ui.play.classList.toggle('on', !playing);
};
ui.live.onclick = () => {
  liveMode = !liveMode;
  ui.live.classList.toggle('on', liveMode);
  if (liveMode) {
    if (!ws) connectLive();
    else ws.send(JSON.stringify({ type: 'hello' }));
    ui.dot.className = 'dot live';
    playing = true;
  }
};
ui.scrub.oninput = () => {
  playing = false;
  ui.play.textContent = '▶';
  cursor = parseFloat(ui.scrub.value);
};
ui.speed.oninput = () => { speed = parseFloat(ui.speed.value); };
ui.trail.oninput = () => { trailLen = parseFloat(ui.trail.value); };
ui.tPred.onchange = () => { predLines.visible = ui.tPred.checked; };
ui.tResid.onchange = () => { residRibbon.visible = ui.tResid.checked; };
ui.tBranch.onchange = () => {
  for (const b of branches) {
    b.line.visible = ui.tBranch.checked;
    b.headMesh.visible = ui.tBranch.checked;
  }
};
ui.tGrid.onchange = () => { grid.visible = ui.tGrid.checked; };
ui.branchBtn.onclick = spawnBranch;

addEventListener('resize', () => {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
});

/* ------------------------------------------------------------------ */
/* boot                                                               */
/* ------------------------------------------------------------------ */
(async () => {
  ui.boot.style.display = 'block';
  try {
    const ok = await loadVault();
    if (ok) {
      ui.boot.style.display = 'none';
    }
  } catch (e) {
    ui.boot.classList.add('error');
    ui.bootMsg.textContent = 'failed to load: ' + e.message;
  }
  animate();
})();

// expose a tiny handle for smoke tests / console poking
globalThis.MORPHOS_STUDIO = {
  get frameCount() { return frames.length; },
  get cursor() { return cursor; },
  get branches() { return branches.length; },
  get bounds() { return bounds; },
  spawnBranch, connectLive,
  setCursor(i) { cursor = i; playing = false; },
};
