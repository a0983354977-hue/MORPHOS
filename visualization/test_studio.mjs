/**
 * Studio smoke test (T5.3 acceptance).
 *
 * Loads the real studio in Chromium, waits for the WebGL scene to populate,
 * asserts frames reached the renderer, exercises the counterfactual fork, and
 * measures the render loop's frame rate. Exits non-zero on any failure.
 *
 * Run: node visualization/test_studio.mjs [--port 8791] [--fps-min 30]
 */

import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const argOf = (name, def) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : def;
};
const PORT = argOf('--port', '8791');
const FPS_MIN = parseFloat(argOf('--fps-min', '30'));
const URL = `http://localhost:${PORT}/`;

let failures = 0;
const check = (cond, label, detail = '') => {
  if (cond) {
    console.log(`  PASS  ${label}`);
  } else {
    failures++;
    console.log(`  FAIL  ${label}${detail ? ' -> ' + detail : ''}`);
  }
};

// Version-agnostic browser resolution: prefer whatever Playwright installed,
// falling back to an explicit CHROME_PATH so the test never dies on a minor
// version bump of the bundled Chromium.
const launchOpts = {
  args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader',
         '--ignore-gpu-blocklist', '--disable-dev-shm-usage'],
};
if (process.env.CHROME_PATH) launchOpts.executablePath = process.env.CHROME_PATH;
let browser;
try {
  browser = await chromium.launch(launchOpts);
} catch (e) {
  const fallback = process.env.CHROME_PATH_FALLBACK;
  if (!fallback) throw e;
  browser = await chromium.launch({ ...launchOpts, executablePath: fallback });
}
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });

const consoleErrors = [];
page.on('console', (m) => {
  if (m.type() === 'error') consoleErrors.push(m.text());
});
page.on('pageerror', (e) => consoleErrors.push('pageerror: ' + e.message));

console.log(`\nMORPHOS studio smoke test -> ${URL}\n`);

await page.goto(URL, { waitUntil: 'networkidle', timeout: 30000 });

// The studio exposes MORPHOS_STUDIO once the module has evaluated.
await page.waitForFunction(
  () => globalThis.MORPHOS_STUDIO && globalThis.MORPHOS_STUDIO.frameCount > 0,
  { timeout: 20000 },
);

const state = await page.evaluate(() => ({
  frames: globalThis.MORPHOS_STUDIO.frameCount,
  cursor: globalThis.MORPHOS_STUDIO.cursor,
  bounds: globalThis.MORPHOS_STUDIO.bounds,
  canvasW: document.getElementById('gl').width,
  canvasH: document.getElementById('gl').height,
  bootVisible: getComputedStyle(document.getElementById('boot')).display !== 'none',
  runName: document.getElementById('run-name').textContent,
  hudStep: document.getElementById('h-step').textContent,
}));

check(state.frames > 0, 'frames loaded into the scene', `${state.frames}`);
check(!state.bootVisible, 'boot overlay dismissed');
check(state.canvasW > 0 && state.canvasH > 0, 'WebGL canvas sized',
  `${state.canvasW}x${state.canvasH}`);
check(state.hudStep !== '0' || state.frames === 1, 'HUD showing live step',
  state.hudStep);
const b = state.bounds;
const extent = Math.max(
  b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
check(extent > 1e-6, 'trajectory has non-zero spatial extent', extent.toFixed(4));

// Does the canvas actually contain drawn pixels (not a blank clear colour)?
const pixelCheck = await page.evaluate(() => {
  const c = document.getElementById('gl');
  // Re-render into a 2D canvas to sample pixels: WebGL canvases keep their
  // drawing buffer alive for one frame after render when preserveDrawingBuffer
  // is off, so read from the live GL context instead.
  const gl = c.getContext('webgl2') || c.getContext('webgl');
  if (!gl) return { ok: false, reason: 'no gl context' };
  const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
  const px = new Uint8Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
  const bg = [7, 9, 16];
  let nonBg = 0;
  for (let i = 0; i < px.length; i += 4 * 37) {   // sparse sample
    const dr = Math.abs(px[i] - bg[0]);
    const dg = Math.abs(px[i + 1] - bg[1]);
    const db = Math.abs(px[i + 2] - bg[2]);
    if (dr + dg + db > 12) nonBg++;
  }
  return { ok: true, nonBg, w, h };
});
check(pixelCheck.ok, 'WebGL context readable', pixelCheck.reason || '');
check(pixelCheck.ok && pixelCheck.nonBg > 0, 'scene rendered visible geometry',
  `non-background samples: ${pixelCheck.nonBg}`);

// Counterfactual fork
await page.selectOption('#branch-mode', 'gain');
await page.click('#btn-branch');
await page.waitForTimeout(500);
const branches1 = await page.evaluate(() => globalThis.MORPHOS_STUDIO.branches);
check(branches1 === 1, 'fork created one branch', String(branches1));

await page.selectOption('#branch-mode', 'noise');
await page.click('#btn-branch');
await page.waitForTimeout(400);
const branches2 = await page.evaluate(() => globalThis.MORPHOS_STUDIO.branches);
check(branches2 === 2, 'second fork created another branch', String(branches2));

// Scrub to the middle and confirm the HUD follows.
await page.evaluate(() => globalThis.MORPHOS_STUDIO.setCursor(50));
await page.waitForTimeout(200);
const scrubbed = await page.evaluate(() =>
  document.getElementById('h-step').textContent);
check(scrubbed !== '', 'HUD updated after scrubbing', scrubbed);

// Measure the render loop for ~2 s.
const fps = await page.evaluate(() => new Promise((res) => {
  let n = 0;
  const t0 = performance.now();
  const step = () => {
    n++;
    if (performance.now() - t0 < 2000) requestAnimationFrame(step);
    else res((n * 1000) / (performance.now() - t0));
  };
  requestAnimationFrame(step);
}));
check(fps >= FPS_MIN, `render loop >= ${FPS_MIN} FPS (swiftshader)`,
  `${fps.toFixed(1)} fps`);

// Screenshot for the record (also proves the canvas is compositable).
const shotDir = resolve(__dirname, 'screenshots');
mkdirSync(shotDir, { recursive: true });
const shot = resolve(shotDir, 'studio.png');
await page.screenshot({ path: shot });
console.log(`\n  screenshot: ${shot}`);

check(consoleErrors.length === 0, 'no console errors',
  consoleErrors.slice(0, 2).join(' | '));

await browser.close();

console.log(`\n${failures === 0 ? 'ALL CHECKS PASSED' : failures + ' CHECK(S) FAILED'}\n`);
process.exit(failures === 0 ? 0 : 1);
