import json
import threading
import time
import urllib.request
from pathlib import Path

import uvicorn
from websockets.sync.client import connect

from core.gateway.ipc_server import GatewayClient, IPCGatewayServer
from core.gateway.runtime import MORPHOSRuntime
from core.gateway.web_bridge import BridgeService, create_app


def test_bridge_serves_live_core_wal_sqlite_and_static_studio(tmp_path):
    runtime = MORPHOSRuntime(data_dir=str(tmp_path), neurons=80)
    ipc = IPCGatewayServer(host="127.0.0.1", port=0, core=runtime)
    ipc_thread = threading.Thread(target=ipc.serve_forever, daemon=True)
    ipc_thread.start()
    bridge = BridgeService(
        ipc_host=ipc.address[0], ipc_port=ipc.address[1],
        data_dir=str(tmp_path),
        dist_dir=str(Path(__file__).resolve().parents[1] / "studio" / "dist"),
        lorenz_enabled=False,
    )
    web = uvicorn.Server(uvicorn.Config(
        create_app(bridge), host="127.0.0.1", port=0,
        log_level="error", access_log=False))
    web_thread = threading.Thread(target=web.run, daemon=True)
    web_thread.start()
    try:
        deadline = time.monotonic() + 10
        while not web.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert web.started
        web_port = web.servers[0].sockets[0].getsockname()[1]
        with GatewayClient(*ipc.address) as client:
            for step in range(6):
                result = client.request({
                    "op": "ingest",
                    "timestamp": step * 0.005,
                    "values": [step / 10, step / 20, step / 30],
                })
                assert result["ok"]

        base = f"http://127.0.0.1:{web_port}"
        with urllib.request.urlopen(f"{base}/api/history?offset=1&limit=3") as response:
            history = json.load(response)
        assert history["total"] == 6
        assert [row["step"] for row in history["records"]] == [2, 3, 4]
        assert len(history["records"][0]["actual"]) == 3

        with urllib.request.urlopen(f"{base}/api/events?offset=0&limit=10") as response:
            events = json.load(response)
        assert any(event["message"] == "IPC gateway core started"
                   for event in events["events"])

        request = urllib.request.Request(
            f"{base}/api/control",
            data=json.dumps({"action": "set_mode", "mode": "combat5"}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["mode"] == "combat5"

        with connect(f"ws://127.0.0.1:{web_port}/ws", open_timeout=3) as socket:
            telemetry = json.loads(socket.recv(timeout=3))
        assert telemetry["type"] == "telemetry"
        assert telemetry["connected"] is True
        assert telemetry["core"]["mode"] == "combat5"

        with urllib.request.urlopen(base) as response:
            page = response.read().decode()
        assert "MORPHOS Studio" in page
    finally:
        web.should_exit = True
        bridge.close()
        ipc.shutdown()
        ipc.server_close()
        web_thread.join(timeout=5)
        ipc_thread.join(timeout=5)
        runtime.close()
