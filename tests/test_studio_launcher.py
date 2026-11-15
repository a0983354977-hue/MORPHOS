import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def _free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def test_launcher_starts_and_stops_isolated_daemon_and_bridge(tmp_path):
    root = Path(__file__).resolve().parents[1]
    ipc_port, web_port = _free_port(), _free_port()
    environment = os.environ.copy()
    environment.update({
        "MORPHOS_DATA_DIR": str(tmp_path),
        "MORPHOS_IPC_PORT": str(ipc_port),
        "MORPHOS_WEB_PORT": str(web_port),
        "MORPHOS_NEURONS": "80",
        "MORPHOS_LORENZ": "0",
        "MORPHOS_OPEN_BROWSER": "0",
    })
    executable = root / "dist" / "MORPHOS_Studio.exe"
    command = [str(executable)] if executable.is_file() else [
        sys.executable, str(root / "studio_launcher.py")]
    process = subprocess.Popen(
        command,
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    try:
        deadline = time.monotonic() + 30
        response = None
        while time.monotonic() < deadline and process.poll() is None:
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{web_port}/api/health",
                        timeout=0.25) as health:
                    response = health.read()
                break
            except OSError:
                time.sleep(0.1)
    finally:
        if process.poll() is None:
            if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGINT)
        output, _ = process.communicate(timeout=20)

    assert response is not None, (
        f"Studio supervisor did not start its services. Output: {output}")
    assert process.returncode == 0
    for port in (ipc_port, web_port):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2)
        except OSError:
            continue
        raise AssertionError(f"launcher left service port {port} running")
