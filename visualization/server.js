#!/usr/bin/env node
/**
 * MORPHOS Studio hub (T5.3) - zero-dependency WebSocket + HTTP server.
 *
 * Serves the Three.js studio from ./public and streams live frames from the
 * neural core. The hub supports two live sources without any npm packages:
 *
 *   webhook  - `main.py --feed` POSTs one JSON frame per step.
 *   poll     - the hub tails a running process's stdout for JSON lines.
 *
 * Static datasets are served from requests to /frames?wal=... (the export
 * tool re-reads the binary WAL and converts to the JSON frame schema).
 *
 * WebSocket is implemented over Node's built-in http server (RFC 6455
 * handshake + masked client frames). Small, dependency-free, good enough to
 * stream 60 FPS of 3D state.
 */

import { createServer } from 'node:http';
import { readFile, readdir, stat } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { spawn } from 'node:child_process';
import { extname, join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomBytes, createHash } from 'node:crypto';
import { execSync } from 'node:child_process';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');

const args = process.argv.slice(2);
const arg = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const PORT = parseInt(arg('--port', process.env.MORPHOS_VIZ_PORT || '8787'), 10);
const DATA_DIR = arg('--data-dir', 'data');
const CHUNK_VAULT = arg('--frames-vault', join(ROOT, 'data'));

/* ------------------------------------------------------------------ */
/* frames vault: read .frames.json snapshots written by morphos export */
/* ------------------------------------------------------------------ */
let framesVault = [];
let lastFrame = null;
let runName = 'offline';

async function loadVault() {
  try {
    const files = (await readdir(CHUNK_VAULT))
      .filter((f) => f.endsWith('.frames.json'))
      .sort()
      .slice(-1);
    if (files.length === 0) return;
    const text = await readFile(join(CHUNK_VAULT, files[0]), 'utf8');
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed)) framesVault = parsed;
    else if (parsed.frames) framesVault = parsed.frames;
    runName = files[0];
    console.log(`[vault] loaded ${framesVault.length} frames from ${files[0]}`);
  } catch (e) {
    console.log('[vault] none found (offline mode)');
  }
}

/* ------------------------------------------------------------------ */
/* WebSocket (RFC 6455)                                               */
/* ------------------------------------------------------------------ */
const clients = new Set();

function wsAccept(key) {
  return createHash('sha1')
    .update(key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11')
    .digest('base64');
}

function handleUpgrade(req, socket) {
  const key = req.headers['sec-websocket-key'];
  if (!key) return socket.destroy();
  const handshake = [
    'HTTP/1.1 101 Switching Protocols',
    'Upgrade: websocket',
    'Connection: Upgrade',
    `Sec-WebSocket-Accept: ${wsAccept(key)}`,
    '',
    '',
  ].join('\r\n');
  socket.write(handshake);
  const client = {
    socket,
    buffer: Buffer.alloc(0),
  };
  clients.add(client);
  socket.on('data', (d) => {
    client.buffer = Buffer.concat([client.buffer, d]);
    while (client.buffer.length >= 2) {
      const b0 = client.buffer[0];
      const b1 = client.buffer[1];
      const len = b1 & 0x7f;
      const masked = (b1 & 0x80) !== 0;
      let header = 2;
      let plen = len;
      if (len === 126) {
        if (client.buffer.length < 4) break;
        plen = client.buffer.readUInt16BE(2);
        header = 4;
      } else if (len === 127) {
        if (client.buffer.length < 10) break;
        plen = client.buffer.readBigUInt64BE(2);
        header = 10;
      }
      let mask = null;
      if (masked) {
        if (client.buffer.length < header + 4) break;
        mask = client.buffer.subarray(header, header + 4);
        header += 4;
      }
      if (client.buffer.length < header + Number(plen)) break;
      const payload = Buffer.from(
        client.buffer.subarray(header, header + Number(plen)));
      if (mask) {
        for (let i = 0; i < payload.length; i++) payload[i] ^= mask[i & 3];
      }
      client.buffer = client.buffer.subarray(header + Number(plen));
      if ((b0 & 0x0f) === 0x8) {
        // close frame
        clients.delete(client);
        socket.end();
        return;
      }
      if ((b0 & 0x0f) === 0x9) {
        // ping -> pong
        sendFrame(client, Buffer.concat([Buffer.from([0x8a, payload.length]), payload]));
      } else {
        try {
          const msg = JSON.parse(payload.toString('utf8'));
          handleMessage(client, msg);
        } catch (_) { /* ignore malformed */ }
      }
    }
  });
  socket.on('close', () => clients.delete(client));
  socket.on('error', () => clients.delete(client));
}

function encodeWSPayload(payload) {
  const payloadBuf = Buffer.isBuffer(payload) ? payload : Buffer.from(payload);
  if (payloadBuf.length < 126) {
    const h = Buffer.alloc(2);
    h[0] = 0x81;
    h[1] = payloadBuf.length;
    return Buffer.concat([h, payloadBuf]);
  }
  if (payloadBuf.length <= 0xffff) {
    const h = Buffer.alloc(4);
    h[0] = 0x81;
    h[1] = 126;
    h.writeUInt16BE(payloadBuf.length, 2);
    return Buffer.concat([h, payloadBuf]);
  }
  const h = Buffer.alloc(10);
  h[0] = 0x81;
  h[1] = 127;
  h.writeBigUInt64BE(BigInt(payloadBuf.length), 2);
  return Buffer.concat([h, payloadBuf]);
}

function sendFrame(client, payload) {
  if (client.socket.writable) client.socket.write(encodeWSPayload(payload));
}

function broadcast(obj) {
  const payload = Buffer.from(JSON.stringify(obj), 'utf8');
  for (const c of clients) sendFrame(c, payload);
}

function handleMessage(client, msg) {
  switch (msg.type) {
    case 'hello':
      sendFrame(client, JSON.stringify({
        type: 'hello',
        run: runName,
        vaultSize: framesVault.length,
        last: lastFrame,
      }));
      break;
    case 'want-live':
      // begin streaming the vault to this client (bounded replay)
      if (msg.from != null && framesVault.length) {
        const start = Math.max(0, msg.from);
        for (let i = start; i < framesVault.length; i++) {
          sendFrame(client, JSON.stringify({
            type: 'frame', frame: framesVault[i],
          }));
        }
      }
      break;
    default:
      break;
  }
}

/* ------------------------------------------------------------------ */
/* frame ingestion (webhook)                                          */
/* ------------------------------------------------------------------ */
const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  if (url.pathname === '/api/ingest' && req.method === 'POST') {
    let body = '';
    for await (const chunk of req) body += chunk;
    try {
      const parsed = JSON.parse(body);
      if (parsed.frames) {
        for (const f of parsed.frames) framesVault.push(f);
        if (parsed.frames.length) lastFrame = parsed.frames[parsed.frames.length - 1];
      } else if (parsed.frame) {
        framesVault.push(parsed.frame);
        lastFrame = parsed.frame;
      }
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ ok: true, vaultSize: framesVault.length }));
      if (parsed.frame) broadcast({ type: 'frame', frame: parsed.frame });
      else if (parsed.frames) for (const f of parsed.frames) broadcast({ type: 'frame', frame: f });
    } catch {
      res.writeHead(400);
      res.end('{"ok":false}');
    }
    return;
  }

  if (url.pathname === '/api/vault') {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ run: runName, count: framesVault.length }));
    return;
  }

  if (url.pathname === '/frames' || url.pathname === '/frames/') {
    const frames = framesVault;
    res.writeHead(200, {
      'content-type': 'application/json',
      'access-control-allow-origin': '*',
    });
    res.end(JSON.stringify({
      schema: 'morphos.frames/1', run: runName, count: frames.length, frames,
    }));
    return;
  }

  // static assets
  const rel = url.pathname === '/' ? 'index.html' : url.pathname.slice(1);
  const filePath = resolve(join(__dirname, 'public'), rel);
  if (!filePath.startsWith(resolve(__dirname, 'public'))) {
    res.writeHead(403); res.end('forbidden'); return;
  }
  try {
    const info = await stat(filePath);
    if (info.isFile()) {
      const types = {
        '.html': 'text/html; charset=utf-8',
        '.js': 'text/javascript; charset=utf-8',
        '.css': 'text/css; charset=utf-8',
        '.json': 'application/json',
        '.mjs': 'text/javascript; charset=utf-8',
        '.svg': 'image/svg+xml',
      };
      res.writeHead(200, {
        'content-type': types[extname(filePath)] || 'application/octet-stream',
        'cache-control': 'no-cache',
      });
      createReadStream(filePath).pipe(res);
      return;
    }
  } catch (_) { /* fallthrough 404 */ }
  res.writeHead(404);
  res.end('not found');
});

server.on('upgrade', handleUpgrade);

/* ------------------------------------------------------------------ */
/* live process polling (no feed hook exists yet - stays dormant)     */
/* ------------------------------------------------------------------ */
function maybeStartLiveSource() {
  // Placeholder: wired for a future `morphos run --feed` that emits JSON
  // lines; the hub will spawn it here. Offline/vault mode needs nothing.
}

await loadVault();
server.listen(PORT, () => {
  console.log(`MORPHOS studio on http://localhost:${PORT}/`);
  console.log(`  vault: ${framesVault.length} frames (${runName})`);
  console.log('  press Ctrl+C to stop');
});

process.on('SIGINT', () => process.exit(0));
process.on('SIGTERM', () => process.exit(0));