"""Loopback JSON-lines IPC gateway.

Run as an isolated service process with ``python -m core.gateway.ipc_server``.
Each TCP connection is handled independently, so a client disconnect cannot
stop the gateway or its core handler.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import socket
import socketserver
import threading
from typing import Any

from core.gateway.schemas import (
    GatewayRequest,
    GatewayResponse,
    SchemaValidationError,
)
from core.ingest.ring_buffer import TimeAligner
from core.memory.thread_safe_buffer import ThreadSafeBuffer, synchronized

LOGGER = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 1 << 20
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9876


class GatewayCore(ThreadSafeBuffer):
    """Default validated backend, backed by the existing time aligner."""

    def __init__(self, dim: int = 3, dt: float = 0.005):
        if not isinstance(dim, int) or isinstance(dim, bool) or not 1 <= dim <= 1024:
            raise ValueError("dim must be between 1 and 1024")
        if (isinstance(dt, bool) or not isinstance(dt, (int, float))
                or not math.isfinite(dt) or dt <= 0):
            raise ValueError("dt must be finite and positive")
        super().__init__()
        self.aligner = TimeAligner(dt=dt, dim=dim)
        self._accepted = 0

    @synchronized
    def handle(self, request: GatewayRequest) -> GatewayResponse:
        if request.op == "ingest":
            if request.timestamp is None or request.values is None:
                return GatewayResponse.failure(
                    "invalid_request", "ingest requires timestamp and values",
                    request.request_id)
            if len(request.values) != self.aligner.dim:
                return GatewayResponse.failure(
                    "dimension_mismatch",
                    f"expected {self.aligner.dim} values",
                    request.request_id,
                )
            self.aligner.push(request.timestamp, request.values)
            self._accepted += 1
            result = {"accepted": True, "samples_received": self._accepted}
        elif request.op == "query":
            if request.timestamp is None:
                return GatewayResponse.failure(
                    "invalid_request", "query requires timestamp",
                    request.request_id)
            values, confidence, flags = self.aligner.at(request.timestamp)
            result = {
                "timestamp": request.timestamp,
                "values": values.tolist(),
                "confidence": confidence,
                "flags": int(flags),
            }
        elif request.op == "reset":
            self.aligner.reset()
            self._accepted = 0
            result = {"reset": True}
        else:
            result = {
                "dimension": self.aligner.dim,
                "samples_received": self._accepted,
                "dropped_samples": self.aligner.dropped_count,
            }
        return GatewayResponse.success(result, request.request_id)


class _GatewayTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False

    def __init__(self, address, core: Any, request_timeout: float):
        self.core = core
        self.request_timeout = request_timeout
        super().__init__(address, _GatewayRequestHandler)


class _GatewayRequestHandler(socketserver.StreamRequestHandler):
    server: _GatewayTCPServer

    def handle(self) -> None:
        self.connection.settimeout(self.server.request_timeout)
        while True:
            try:
                raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
            except (TimeoutError, socket.timeout):
                return
            if not raw:
                return
            if len(raw) > MAX_REQUEST_BYTES:
                self._send(GatewayResponse.failure(
                    "request_too_large", "request exceeds the size limit"))
                return
            try:
                payload = json.loads(raw)
            except (ValueError, UnicodeDecodeError, RecursionError):
                self._send(GatewayResponse.failure(
                    "invalid_json", "request must be valid UTF-8 JSON"))
                continue
            try:
                request = GatewayRequest.from_payload(payload)
            except SchemaValidationError as exc:
                self._send(GatewayResponse.failure(
                    "invalid_request", str(exc)))
                continue
            try:
                response = self.server.core.handle(request)
            except Exception:
                LOGGER.exception("gateway core request failed")
                response = GatewayResponse.failure(
                    "internal_error", "request could not be processed",
                    request.request_id)
            self._send(response)

    def _send(self, response: GatewayResponse) -> None:
        try:
            data = json.dumps(response.to_dict(), allow_nan=False,
                              separators=(",", ":")).encode("utf-8") + b"\n"
            self.wfile.write(data)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            return


class IPCGatewayServer:
    """Manage a threaded TCP server suitable for running in its own process."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 dim: int = 3, dt: float = 0.005,
                 core: Any | None = None, request_timeout: float = 5.0):
        if not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        self.core = core if core is not None else GatewayCore(dim=dim, dt=dt)
        self._server = _GatewayTCPServer(
            (host, port), self.core, request_timeout=request_timeout)

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def shutdown(self) -> None:
        self._server.shutdown()

    def server_close(self) -> None:
        self._server.server_close()

    def __enter__(self) -> "IPCGatewayServer":
        return self

    def __exit__(self, *_exc) -> None:
        self.shutdown()
        self.server_close()


class GatewayClient:
    """Synchronous client using one persistent newline-delimited connection."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 timeout: float = 5.0):
        self._socket = socket.create_connection((host, port), timeout=timeout)
        self._socket.settimeout(timeout)
        self._reader = self._socket.makefile("rb")
        self._lock = threading.Lock()

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, allow_nan=False,
                          separators=(",", ":")).encode("utf-8") + b"\n"
        if len(data) > MAX_REQUEST_BYTES:
            raise ValueError("request exceeds the size limit")
        with self._lock:
            self._socket.sendall(data)
            response = self._reader.readline(MAX_REQUEST_BYTES + 1)
        if not response:
            raise ConnectionError("gateway closed the connection")
        parsed = json.loads(response)
        if not isinstance(parsed, dict):
            raise ConnectionError("gateway returned an invalid response")
        return parsed

    def close(self) -> None:
        self._reader.close()
        self._socket.close()

    def __enter__(self) -> "GatewayClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="MORPHOS local IPC gateway")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="bind address (defaults to loopback only)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--backend", choices=("morphos", "aligner"),
                        default="morphos")
    parser.add_argument("--data-dir", default="data_gateway")
    parser.add_argument("--neurons", type=int, default=400)
    args = parser.parse_args()
    if args.backend == "morphos":
        from core.gateway.runtime import MORPHOSRuntime
        core = MORPHOSRuntime(data_dir=args.data_dir, neurons=args.neurons,
                              dt=args.dt, dim=args.dim)
    else:
        core = GatewayCore(dim=args.dim, dt=args.dt)
    gateway = IPCGatewayServer(host=args.host, port=args.port, core=core)
    host, port = gateway.address
    print(f"LISTENING {host} {port}", flush=True)
    try:
        gateway.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        gateway.server_close()
        close = getattr(core, "close", None)
        if close is not None:
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
