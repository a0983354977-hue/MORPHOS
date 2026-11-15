"""HTTP/WebSocket bridge for the local MORPHOS daemon and WAL."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from adapters.lorenz import LorenzSource
from adapters.system import SystemSource
from core.gateway.ipc_server import DEFAULT_HOST, DEFAULT_PORT, GatewayClient
from core.encoding.projector import SubspaceProjector
from storage.wal import MAGIC, RECORD

LOGGER = logging.getLogger(__name__)
WEB_HOST = "127.0.0.1"
WEB_PORT = 8080
PUSH_INTERVAL = 0.05
MAX_PAGE_SIZE = 500


def _project_root() -> Path:
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        return Path(bundled)
    return Path(__file__).resolve().parents[2]


class BridgeService:
    def __init__(self, ipc_host: str = DEFAULT_HOST,
                 ipc_port: int = DEFAULT_PORT,
                 data_dir: str = "data_gateway",
                 dist_dir: str | None = None,
                 lorenz_enabled: bool = True):
        root = _project_root()
        self.ipc_host = ipc_host
        self.ipc_port = ipc_port
        self.data_dir = Path(data_dir).resolve()
        self.wal_path = self.data_dir / "trajectory.wal"
        self.db_path = self.data_dir / "morphos.db"
        self.dist_dir = (Path(dist_dir).resolve() if dist_dir else
                         root / "studio" / "dist")
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._snapshot: dict[str, Any] = {
            "type": "telemetry",
            "connected": False,
            "ipc_latency_ms": None,
            "core": None,
            "sources": {"lorenz": False, "system": False},
            "system": None,
            "observed_at": None,
            "error": None,
        }
        self._system_source: SystemSource | None = None
        self._system_enabled = False
        self._projector = SubspaceProjector(
            {"lorenz": 3, "system": 4}, proj_dim=16,
            mode="orthogonal", seed=0)
        self._lorenz_enabled = False
        self._lorenz_stop: threading.Event | None = None
        self._lorenz_thread: threading.Thread | None = None
        self._poller = threading.Thread(
            target=self._poll, name="morphos-web-bridge", daemon=True)
        self._poller.start()
        if lorenz_enabled:
            self.set_source("lorenz", True)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._snapshot))

    def _poll(self) -> None:
        client: GatewayClient | None = None
        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                if client is None:
                    client = GatewayClient(self.ipc_host, self.ipc_port,
                                           timeout=1.0)
                response = client.request({"op": "status"})
                if not response.get("ok"):
                    raise ConnectionError("IPC status request was rejected")
                system = None
                system_error = None
                with self._lock:
                    if self._system_enabled:
                        try:
                            if self._system_source is None:
                                self._system_source = SystemSource()
                            system = self._system_source.sample().tolist()
                        except Exception as exc:
                            system_error = str(exc)
                    core = response.get("result")
                    self._snapshot = {
                        "type": "telemetry",
                        "connected": True,
                        "ipc_latency_ms":
                            (time.perf_counter() - started) * 1000.0,
                        "core": core,
                        "sources": {
                            "lorenz": self._lorenz_enabled,
                            "system": self._system_enabled,
                        },
                        "system": system,
                        "system_error": system_error,
                        "geometry": self._geometry(core, system),
                        "observed_at": time.time(),
                        "error": None,
                    }
            except Exception as exc:
                if client is not None:
                    client.close()
                    client = None
                with self._lock:
                    self._snapshot.update({
                        "connected": False,
                        "ipc_latency_ms": None,
                        "core": None,
                        "geometry": {
                            "basis_cross_talk": self._basis_cross_talk(),
                            "lorenz_energy": None,
                            "system_energy": None,
                        },
                        "observed_at": time.time(),
                        "error": str(exc),
                    })
            self._stop.wait(PUSH_INTERVAL)
        if client is not None:
            client.close()

    def _basis_cross_talk(self) -> float:
        lorenz = self._projector.matrices["lorenz"]
        system = self._projector.matrices["system"]
        return float(np.linalg.norm(lorenz.T @ system))

    def _geometry(self, core: Any, system: list[float] | None) -> dict:
        lorenz_energy = system_energy = None
        latest = core.get("latest") if isinstance(core, dict) else None
        if isinstance(latest, dict):
            state = np.asarray(latest.get("actual", []), dtype=float)
            if state.shape == (3,):
                lorenz_energy = float(np.linalg.norm(
                    self._projector.matrices["lorenz"] @ state))
        if system is not None:
            system_state = np.asarray(system, dtype=float)
            if system_state.shape == (4,):
                scale = float(np.linalg.norm(system_state))
                if scale > 1e-12:
                    system_state = system_state / scale
                system_energy = float(np.linalg.norm(
                    self._projector.matrices["system"] @ system_state))
        return {
            "basis_cross_talk": self._basis_cross_talk(),
            "lorenz_energy": lorenz_energy,
            "system_energy": system_energy,
        }

    def set_source(self, source: str, enabled: bool) -> dict[str, Any]:
        if source == "system":
            with self._lock:
                self._system_enabled = enabled
                if not enabled:
                    self._system_source = None
            return {"source": source, "enabled": enabled}
        if source != "lorenz":
            raise ValueError("source must be lorenz or system")
        with self._lock:
            if enabled == self._lorenz_enabled:
                return {"source": source, "enabled": enabled}
            self._lorenz_enabled = enabled
            if enabled:
                stop = threading.Event()
                thread = threading.Thread(
                    target=self._run_lorenz, args=(stop,),
                    name="morphos-lorenz-source", daemon=True)
                self._lorenz_stop = stop
                self._lorenz_thread = thread
                thread.start()
            else:
                if self._lorenz_stop is not None:
                    self._lorenz_stop.set()
                self._lorenz_stop = None
                self._lorenz_thread = None
        return {"source": source, "enabled": enabled}

    def _run_lorenz(self, stop: threading.Event) -> None:
        source = LorenzSource(dt=0.005)
        while not stop.is_set() and not self._stop.is_set():
            try:
                with GatewayClient(self.ipc_host, self.ipc_port,
                                  timeout=2.0) as client:
                    status = client.request({"op": "status"})
                    if status.get("ok"):
                        source.t = (status["result"].get("steps_processed", 0)
                                    * source.dt)
                    while not stop.is_set() and not self._stop.is_set():
                        tick = source.t + source.dt
                        source.advance_to(tick)
                        for timestamp, values in source.read_pending():
                            response = client.request({
                                "op": "ingest",
                                "timestamp": timestamp,
                                "values": values.tolist(),
                            })
                            if not response.get("ok"):
                                raise RuntimeError(
                                    "IPC rejected a Lorenz sample")
                        if stop.wait(source.dt):
                            break
            except Exception as exc:
                LOGGER.warning("Lorenz source disconnected from IPC: %s", exc)
                stop.wait(0.5)

    def control(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or not isinstance(
                payload.get("action"), str):
            raise ValueError("control action is required")
        action = payload["action"]
        if action == "source":
            if set(payload) != {"action", "source", "enabled"}:
                raise ValueError("source control fields are invalid")
            source, enabled = payload.get("source"), payload.get("enabled")
            if not isinstance(source, str) or not isinstance(enabled, bool):
                raise ValueError("source and enabled have invalid types")
            return self.set_source(source, enabled)

        allowed = {
            "set_mode": {"action", "mode"},
            "reset_breaker": {"action"},
            "branch": {"action", "branch_mode", "step"},
        }
        if action not in allowed or set(payload) != allowed[action]:
            raise ValueError("unsupported control action or fields")
        request = {"op": "control", **payload}
        with GatewayClient(self.ipc_host, self.ipc_port,
                           timeout=10.0) as client:
            response = client.request(request)
        if not response.get("ok"):
            error = response.get("error", {})
            raise RuntimeError(error.get("message", "IPC control failed"))
        return response.get("result", {})

    def history(self, offset: int, limit: int) -> dict[str, Any]:
        if limit < 1 or limit > MAX_PAGE_SIZE or offset < 0:
            raise ValueError("history page must use offset >= 0 and limit 1..500")
        if not self.wal_path.exists():
            return {"offset": offset, "limit": limit, "total": 0, "records": []}
        with self.wal_path.open("rb") as wal:
            if wal.read(len(MAGIC)) != MAGIC:
                raise ValueError("WAL header is invalid")
            size = os.fstat(wal.fileno()).st_size
            total = max(0, (size - len(MAGIC)) // RECORD.size)
            wal.seek(len(MAGIC) + offset * RECORD.size)
            payload = wal.read(limit * RECORD.size)
        count = len(payload) // RECORD.size
        records = []
        for index in range(count):
            values = RECORD.unpack_from(payload, index * RECORD.size)
            records.append({
                "step": offset + index + 1,
                "timestamp": values[0],
                "confidence": values[1],
                "actual": list(values[2:5]),
                "predicted": list(values[5:8]),
                "residual": values[8],
                "rate": values[9],
                "flags": values[10],
                "wall": values[11],
            })
        return {"offset": offset, "limit": limit, "total": total,
                "records": records}

    def events(self, offset: int, limit: int) -> dict[str, Any]:
        if limit < 1 or limit > MAX_PAGE_SIZE or offset < 0:
            raise ValueError("event page must use offset >= 0 and limit 1..500")
        if not self.db_path.exists():
            return {"offset": offset, "limit": limit, "total": 0, "events": []}
        uri = self.db_path.as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=1.0) as connection:
            total = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            rows = connection.execute(
                "SELECT id, run_id, sim_t, wall, level, message, data "
                "FROM events ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset)).fetchall()
        events = []
        for row in reversed(rows):
            events.append({
                "id": row[0], "run_id": row[1], "sim_t": row[2],
                "wall": row[3], "level": row[4], "message": row[5],
                "data": json.loads(row[6]) if row[6] else None,
            })
        return {"offset": offset, "limit": limit, "total": total,
                "events": events}

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            if self._lorenz_stop is not None:
                self._lorenz_stop.set()
            thread = self._lorenz_thread
        if thread is not None:
            thread.join(timeout=3.0)
        self._poller.join(timeout=3.0)


def create_app(service: BridgeService | None = None) -> FastAPI:
    bridge = service or BridgeService(
        ipc_host=os.environ.get("MORPHOS_IPC_HOST", DEFAULT_HOST),
        ipc_port=int(os.environ.get("MORPHOS_IPC_PORT", DEFAULT_PORT)),
        data_dir=os.environ.get("MORPHOS_DATA_DIR", "data_gateway"),
        dist_dir=os.environ.get("MORPHOS_STUDIO_DIST"),
        lorenz_enabled=os.environ.get("MORPHOS_LORENZ", "1") != "0",
    )
    app = FastAPI(title="MORPHOS Web Bridge", version="1.0")
    app.state.bridge = bridge

    @app.get("/api/health")
    async def health():
        return bridge.snapshot()

    @app.get("/api/history")
    async def history(offset: int = Query(0, ge=0),
                      limit: int = Query(200, ge=1, le=MAX_PAGE_SIZE)):
        try:
            return await asyncio.to_thread(bridge.history, offset, limit)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/events")
    async def events(offset: int = Query(0, ge=0),
                     limit: int = Query(100, ge=1, le=MAX_PAGE_SIZE)):
        try:
            return await asyncio.to_thread(bridge.events, offset, limit)
        except (OSError, sqlite3.Error, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/control")
    async def control(payload: dict[str, Any]):
        try:
            return await asyncio.to_thread(bridge.control, payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                try:
                    message = await asyncio.wait_for(
                        websocket.receive_json(), timeout=PUSH_INTERVAL)
                    if isinstance(message, dict) and \
                            message.get("type") == "control":
                        try:
                            result = await asyncio.to_thread(
                                bridge.control, message.get("payload"))
                            await websocket.send_json(
                                {"type": "control", "ok": True,
                                 "result": result})
                        except Exception as exc:
                            await websocket.send_json(
                                {"type": "control", "ok": False,
                                 "error": str(exc)})
                except asyncio.TimeoutError:
                    pass
                await websocket.send_json(bridge.snapshot())
        except WebSocketDisconnect:
            return

    @app.get("/{path:path}")
    async def static_files(path: str):
        requested = (bridge.dist_dir / (path or "index.html")).resolve()
        try:
            requested.relative_to(bridge.dist_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="forbidden")
        if not requested.is_file():
            requested = bridge.dist_dir / "index.html"
        if not requested.is_file():
            return JSONResponse(
                {"error": "Studio build is missing",
                 "build": "Run npm --prefix studio run build"},
                status_code=503)
        return FileResponse(requested)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="MORPHOS HTTP/WebSocket bridge")
    parser.add_argument("--host", default=WEB_HOST)
    parser.add_argument("--port", type=int, default=WEB_PORT)
    parser.add_argument("--ipc-host", default=DEFAULT_HOST)
    parser.add_argument("--ipc-port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--data-dir", default="data_gateway")
    parser.add_argument("--dist-dir", default=None)
    parser.add_argument("--no-lorenz", action="store_true")
    args = parser.parse_args()

    import uvicorn

    service = BridgeService(
        ipc_host=args.ipc_host, ipc_port=args.ipc_port,
        data_dir=args.data_dir, dist_dir=args.dist_dir,
        lorenz_enabled=not args.no_lorenz)
    try:
        uvicorn.run(create_app(service), host=args.host, port=args.port,
                    log_level="info")
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
