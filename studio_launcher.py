"""Supervise the isolated MORPHOS daemon and local Studio bridge."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


def _app_root() -> Path:
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        return Path(bundled)
    return Path(__file__).resolve().parent


def _data_dir() -> Path:
    configured = os.environ.get("MORPHOS_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        return base / "MORPHOS"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "MORPHOS"
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "morphos"


def _service(name: str) -> int:
    import uvicorn

    from core.gateway.ipc_server import (
        DEFAULT_HOST,
        DEFAULT_PORT,
        IPCGatewayServer,
    )
    from core.gateway.runtime import MORPHOSRuntime
    from core.gateway.web_bridge import (
        WEB_HOST,
        WEB_PORT,
        BridgeService,
        create_app,
    )

    data_dir = _data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    ipc_host = os.environ.get("MORPHOS_IPC_HOST", DEFAULT_HOST)
    ipc_port = int(os.environ.get("MORPHOS_IPC_PORT", DEFAULT_PORT))

    if name == "ipc":
        runtime = MORPHOSRuntime(
            data_dir=str(data_dir),
            neurons=int(os.environ.get("MORPHOS_NEURONS", "400")),
        )
        gateway = None
        try:
            gateway = IPCGatewayServer(
                host=ipc_host, port=ipc_port, core=runtime)
            if hasattr(signal, "SIGTERM"):
                signal.signal(
                    signal.SIGTERM,
                    lambda *_: threading.Thread(
                        target=gateway.shutdown, daemon=True).start(),
                )
            if hasattr(signal, "SIGBREAK"):
                signal.signal(
                    signal.SIGBREAK,
                    lambda *_: threading.Thread(
                        target=gateway.shutdown, daemon=True).start(),
                )
            print(f"IPC daemon listening on {ipc_host}:{ipc_port}", flush=True)
            gateway.serve_forever()
        finally:
            if gateway is not None:
                gateway.server_close()
            runtime.close()
        return 0

    if name == "web":
        web_host = os.environ.get("MORPHOS_WEB_HOST", WEB_HOST)
        web_port = int(os.environ.get("MORPHOS_WEB_PORT", WEB_PORT))
        dist_dir = Path(os.environ.get(
            "MORPHOS_STUDIO_DIST", _app_root() / "studio" / "dist")).resolve()
        bridge = BridgeService(
            ipc_host=ipc_host, ipc_port=ipc_port,
            data_dir=str(data_dir), dist_dir=str(dist_dir),
            lorenz_enabled=os.environ.get("MORPHOS_LORENZ", "1") != "0",
        )
        web_server = uvicorn.Server(uvicorn.Config(
            create_app(bridge), host=web_host, port=web_port,
            log_level="info", access_log=False))
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, lambda *_: setattr(
                web_server, "should_exit", True))
        try:
            web_server.run()
        finally:
            bridge.close()
        return 0

    raise ValueError(f"unknown MORPHOS service: {name}")


def _wait_for_daemon(process: subprocess.Popen, host: str, port: int) -> None:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"IPC daemon exited with status {process.returncode}")
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError("IPC daemon did not become ready")


def _wait_for_web(process: subprocess.Popen, url: str) -> None:
    deadline = time.monotonic() + 30.0
    health_url = f"{url}/api/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Web Bridge exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(health_url, timeout=0.5):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)
    raise TimeoutError("Web Bridge did not become ready")


def _child_command(service: str) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--service", service]
    return [sys.executable, str(Path(__file__).resolve()),
            "--service", service]


def _stop_children(children: list[subprocess.Popen]) -> None:
    for child in children:
        if child.poll() is None:
            try:
                if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
                    child.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    child.send_signal(signal.SIGTERM)
            except OSError:
                child.terminate()
    deadline = time.monotonic() + 5.0
    for child in children:
        if child.poll() is None:
            try:
                child.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2.0)


def _handle_console_break(_signum, _frame) -> None:
    raise KeyboardInterrupt


def _supervise() -> int:
    root = _app_root()
    data_dir = _data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    ipc_host = os.environ.get("MORPHOS_IPC_HOST", "127.0.0.1")
    ipc_port = int(os.environ.get("MORPHOS_IPC_PORT", "9876"))
    web_host = os.environ.get("MORPHOS_WEB_HOST", "127.0.0.1")
    web_port = int(os.environ.get("MORPHOS_WEB_PORT", "8080"))
    dist_dir = Path(os.environ.get(
        "MORPHOS_STUDIO_DIST", root / "studio" / "dist")).resolve()
    if web_host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("MORPHOS Web Bridge must bind to loopback")
    if not (dist_dir / "index.html").is_file():
        raise FileNotFoundError("Studio build is missing; build studio/dist first")
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handle_console_break)

    environment = os.environ.copy()
    environment["MORPHOS_DATA_DIR"] = str(data_dir)
    environment["MORPHOS_STUDIO_DIST"] = str(dist_dir)
    children: list[subprocess.Popen] = []
    try:
        ipc = subprocess.Popen(
            _child_command("ipc"), cwd=root, env=environment,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        children.append(ipc)
        _wait_for_daemon(ipc, ipc_host, ipc_port)

        web = subprocess.Popen(
            _child_command("web"), cwd=root, env=environment,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        children.append(web)
        url = f"http://{web_host}:{web_port}"
        _wait_for_web(web, url)
        print(f"MORPHOS Studio: {url}", flush=True)
        if os.environ.get("MORPHOS_OPEN_BROWSER", "1") != "0":
            webbrowser.open(url)

        while True:
            for child in children:
                code = child.poll()
                if code is not None:
                    if code != 0:
                        print(f"MORPHOS service exited with status {code}",
                              file=sys.stderr, flush=True)
                    return code or 1
            time.sleep(0.25)
    except KeyboardInterrupt:
        return 0
    finally:
        _stop_children(children)


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--service":
        return _service(sys.argv[2])
    return _supervise()


if __name__ == "__main__":
    raise SystemExit(main())
