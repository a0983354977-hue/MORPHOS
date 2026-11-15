"""Phase IV.c gateway and shared-buffer stress tests.

Run with: ``python -m pytest tests/concurrency_test.py``
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.gateway.ipc_server import GatewayClient, IPCGatewayServer
from core.dynamics.engine import NengoDynamics
from core.gateway.runtime import MORPHOSRuntime
from core.gateway.schemas import GatewayRequest
from core.ingest.ring_buffer import TimeAligner
from core.memory.buffer import RollingTemporalBuffer, TemporalConfig
from storage.wal import iter_records

CLIENTS = 10
STEPS_PER_CLIENT = 500


def _start_server():
    server = IPCGatewayServer(host="127.0.0.1", port=0, dim=3)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_server(server, thread):
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()
    assert not thread.is_alive(), "gateway server did not stop"


def test_schema_rejects_bad_payloads_without_core_contamination():
    server, thread = _start_server()
    try:
        host, port = server.address
        with GatewayClient(host, port) as client:
            bad_payloads = [
                {"op": "ingest", "timestamp": 1.0, "values": [1.0, 2.0]},
                {"op": "ingest", "values": [1.0, 2.0, 3.0]},
                {"op": "ingest", "timestamp": float("nan"),
                 "values": [1.0, 2.0, 3.0]},
                {"op": "query", "timestamp": "now"},
                {"op": "reset", "extra": "not allowed"},
                {"op": "status", "request_id": "bad\nid"},
                {"op": "control", "action": "set_mode", "mode": ["combat5"]},
            ]
            for payload in bad_payloads:
                # json.dumps intentionally rejects NaN; send it as raw JSON below.
                if payload.get("timestamp") != payload.get("timestamp"):
                    response = _raw_request(host, port, b'{"op":"ingest",'
                                            b'"timestamp":NaN,"values":[1,2,3]}\n')
                else:
                    response = client.request(payload)
                assert response["ok"] is False
                assert response["error"]["code"] in {
                    "invalid_request", "dimension_mismatch"}

            status = client.request({"op": "status"})
            assert status["ok"] is True
            assert status["result"]["samples_received"] == 0
    finally:
        _stop_server(server, thread)


def _raw_request(host: str, port: int, data: bytes) -> dict:
    import json

    with socket.create_connection((host, port), timeout=3) as connection:
        connection.sendall(data)
        with connection.makefile("rb") as reader:
            return json.loads(reader.readline())


def test_ten_clients_ingest_and_query_five_thousand_steps():
    server, thread = _start_server()
    errors = []
    completed = [0] * CLIENTS

    def run_client(client_id: int) -> None:
        host, port = server.address
        try:
            with GatewayClient(host, port, timeout=15) as client:
                for step in range(STEPS_PER_CLIENT):
                    timestamp = client_id * 1000.0 + step * 0.001
                    ingest = client.request({
                        "op": "ingest",
                        "timestamp": timestamp,
                        "values": [client_id, step, timestamp],
                    })
                    assert ingest["ok"] is True
                    query = client.request({"op": "query",
                                            "timestamp": timestamp})
                    assert query["ok"] is True
                    assert len(query["result"]["values"]) == 3
                    completed[client_id] += 1
                    time.sleep(0.001)
        except Exception as exc:
            errors.append(exc)

    started = time.monotonic()
    workers = [threading.Thread(target=run_client, args=(i,))
               for i in range(CLIENTS)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=45)
        elapsed = time.monotonic() - started
        assert all(not worker.is_alive() for worker in workers), (
            "concurrent clients deadlocked")
        assert not errors, f"client request failed: {errors!r}"
        assert completed == [STEPS_PER_CLIENT] * CLIENTS
        host, port = server.address
        with GatewayClient(host, port) as client:
            status = client.request({"op": "status"})
        assert status["result"]["samples_received"] == CLIENTS * STEPS_PER_CLIENT
        assert elapsed < 45, f"stress test exceeded its deadline: {elapsed:.1f}s"
    finally:
        _stop_server(server, thread)


def test_aligner_and_temporal_buffer_are_safe_under_shared_access():
    aligner = TimeAligner(dt=0.001, dim=3, capacity=4096)
    temporal = RollingTemporalBuffer(
        feature_dim=3,
        config=TemporalConfig(capacity=512, ema_scales=(10, 50),
                              compress_after=1024, max_compressed=64),
    )
    errors = []

    def run_worker(worker_id: int) -> None:
        try:
            for step in range(STEPS_PER_CLIENT):
                timestamp = worker_id * 1000.0 + step * 0.001
                values = np.array([worker_id, step, timestamp], dtype=float)
                aligner.push(timestamp, values)
                aligned, confidence, _flags = aligner.at(timestamp)
                assert aligned.shape == (3,) and np.all(np.isfinite(aligned))
                assert 0.0 <= confidence <= 1.0
                temporal.push_step(values, 1.0, float(step), 20.0)
                recent, residuals, steps = temporal.get_recent_history(32)
                assert recent.shape[1] == 3
                assert residuals.shape == steps.shape == (recent.shape[0],)
                temporal.state_dict()
        except Exception as exc:
            errors.append(exc)

    workers = [threading.Thread(target=run_worker, args=(i,))
               for i in range(CLIENTS)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
    assert all(not worker.is_alive() for worker in workers), (
        "buffer operations deadlocked")
    assert not errors, f"buffer operation failed: {errors!r}"
    assert aligner.dropped_count == 0
    assert len(aligner.buf) <= aligner.buf.capacity + aligner.buf.trim_batch
    assert len(temporal) == temporal.capacity
    assert np.all(np.isfinite(list(temporal.get_all_ema().values())))


def test_shared_nengo_simulator_serializes_concurrent_steps():
    engine = NengoDynamics(feature_dim=3, n_neurons=30, dt=0.001)
    errors = []

    def run_worker(worker_id: int) -> None:
        try:
            feats = np.full((1, 3), worker_id / CLIENTS)
            for _ in range(20):
                result = engine.run_chunk(feats)
                assert result["decoded"].shape == (1, 3)
                assert np.all(np.isfinite(result["decoded"]))
        except Exception as exc:
            errors.append(exc)

    workers = [threading.Thread(target=run_worker, args=(i,))
               for i in range(CLIENTS)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
    assert all(not worker.is_alive() for worker in workers), (
        "concurrent simulator operations deadlocked")
    assert not errors, f"simulator operation failed: {errors!r}"
    assert engine.steps_run == CLIENTS * 20


def test_full_runtime_connects_neural_core_dual_opinion_and_wal():
    with tempfile.TemporaryDirectory(prefix="morphos-gateway-") as data_dir:
        runtime = MORPHOSRuntime(data_dir=data_dir, neurons=30)
        try:
            for step in range(3):
                response = runtime.handle(GatewayRequest(
                    op="ingest", timestamp=step * 0.005,
                    values=(float(step), float(step + 1), float(step + 2))))
                assert response.ok
                assert response.result["step"] == step + 1
            query = runtime.handle(GatewayRequest(
                op="query", timestamp=0.01))
            assert query.ok and len(query.result["values"]) == 3
            before_reset = runtime.handle(GatewayRequest(op="status"))
            assert before_reset.result["neural_steps"] == 3
            assert before_reset.result["temporal_samples"] == 3

            reset = runtime.handle(GatewayRequest(op="reset"))
            assert reset.result["reset"] is True
            assert reset.result["wal_records"] == 3
            after_reset = runtime.handle(GatewayRequest(
                op="ingest", timestamp=0.02, values=(1.0, 2.0, 3.0)))
            assert after_reset.ok and after_reset.result["wal_records"] == 4
        finally:
            runtime.close()

        from storage.wal import iter_records
        records = list(iter_records(os.path.join(data_dir, "trajectory.wal")))
        assert len(records) == 4
        assert all(np.all(np.isfinite(record["actual"])) for record in records)


def test_runtime_restart_continues_wal_step_timeline():
    with tempfile.TemporaryDirectory() as data_dir:
        runtime = MORPHOSRuntime(data_dir=data_dir, neurons=80)
        for step in range(2):
            response = runtime.handle(GatewayRequest(
                op="ingest", timestamp=step * 0.005,
                values=(step / 10, step / 20, step / 30)))
            assert response.ok
        runtime.close()

        restarted = MORPHOSRuntime(data_dir=data_dir, neurons=80)
        try:
            status = restarted.handle(GatewayRequest(op="status"))
            assert status.result["steps_processed"] == 2
            response = restarted.handle(GatewayRequest(
                op="ingest", timestamp=0.0, values=(0.2, 0.1, 0.05)))
            assert response.ok
        finally:
            restarted.close()

        records = list(iter_records(os.path.join(data_dir, "trajectory.wal")))
        assert [row["t_sim"] for row in records] == [0.005, 0.01, 0.015]


def test_runtime_controls_drive_chaos_breaker_and_real_branch_replay():
    with tempfile.TemporaryDirectory(prefix="morphos-controls-") as data_dir:
        runtime = MORPHOSRuntime(data_dir=data_dir, neurons=20)
        try:
            mode = runtime.handle(GatewayRequest(
                op="control", action="set_mode", mode="combat5"))
            assert mode.ok and mode.result["mode"] == "combat5"
            for step in range(50):
                runtime.handle(GatewayRequest(
                    op="ingest", timestamp=step * 0.005,
                    values=(step / 10.0, step / 20.0, step / 30.0)))
                if runtime._step >= 5:
                    break
            assert runtime._step >= 5
            runtime.handle(GatewayRequest(
                op="control", action="set_mode", mode="live"))
            breaker = runtime.handle(GatewayRequest(
                op="control", action="reset_breaker"))
            assert breaker.ok and breaker.result["breaker_state"] == "OK"
            branch = runtime.handle(GatewayRequest(
                op="control", action="branch",
                branch_mode="gain", step=runtime._step))
            assert branch.ok, branch.error
            continued = runtime.handle(GatewayRequest(
                op="ingest", timestamp=1.0, values=(0.5, 0.25, 0.1)))
            assert continued.ok
            status = runtime.handle(GatewayRequest(op="status"))
            assert status.result["branches"][0]["steps"] == 1
        finally:
            runtime.close()


def test_runtime_sqlite_logging_is_safe_from_client_threads():
    with tempfile.TemporaryDirectory(prefix="morphos-runtime-threads-") as data_dir:
        runtime = MORPHOSRuntime(data_dir=data_dir, neurons=30)
        errors = []

        def run_worker(worker_id: int) -> None:
            try:
                for step in range(60):
                    response = runtime.handle(GatewayRequest(
                        op="ingest",
                        timestamp=worker_id * 1000.0 + step * 0.005,
                        values=(worker_id / 10.0, step / 60.0,
                                (worker_id + step) / 70.0)))
                    assert response.ok
            except Exception as exc:
                errors.append(exc)

        workers = [threading.Thread(target=run_worker, args=(i,))
                   for i in range(CLIENTS)]
        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=60)
            assert all(not worker.is_alive() for worker in workers), (
                "runtime processing deadlocked")
            assert not errors, f"runtime client failed: {errors!r}"
            status = runtime.handle(GatewayRequest(op="status"))
            assert status.result["steps_processed"] == 600
            assert status.result["dual_calibrated"] is True
            assert status.result["wal_records"] == 600
        finally:
            runtime.close()


def test_wal_recovery_truncates_only_an_incomplete_tail():
    from storage.wal import MAGIC, RECORD, WALWriter, iter_records

    with tempfile.TemporaryDirectory(prefix="morphos-wal-recovery-") as data_dir:
        path = os.path.join(data_dir, "trajectory.wal")
        writer = WALWriter(path)
        writer.append(0.005, 1.0, [0.1, 0.2, 0.3],
                      [0.0, 0.0, 0.0], 0.14, 10.0, 0)
        writer.close()
        with open(path, "ab") as wal_file:
            wal_file.write(b"partial-record")

        writer = WALWriter(path)
        assert writer.total_records == 1
        writer.append(0.010, 1.0, [0.2, 0.3, 0.4],
                      [0.1, 0.1, 0.1], 0.14, 11.0, 0)
        writer.close()
        records = list(iter_records(path))
        assert len(records) == 2
        assert os.path.getsize(path) == len(MAGIC) + 2 * RECORD.size


def test_disconnected_client_does_not_stop_isolated_gateway_process():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="morphos-ipc-") as data_dir:
        process = subprocess.Popen(
            [sys.executable, "-m", "core.gateway.ipc_server",
             "--host", "127.0.0.1", "--port", str(port),
             "--data-dir", data_dir, "--neurons", "30"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise AssertionError(
                        f"gateway process exited with {process.returncode}")
                try:
                    with socket.create_connection(
                            ("127.0.0.1", port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                raise AssertionError("gateway process did not start")

            with GatewayClient("127.0.0.1", port) as client:
                assert client.request({
                    "op": "ingest", "timestamp": 1.0,
                    "values": [1.0, 2.0, 3.0],
                })["ok"]
            crashed_client = socket.create_connection(
                ("127.0.0.1", port), timeout=2)
            crashed_client.sendall(
                b'{"op":"ingest","timestamp":1.5,"values":[1,2,3]}\n')
            crashed_client.close()
            with GatewayClient("127.0.0.1", port) as client:
                response = client.request({
                    "op": "ingest", "timestamp": 2.0,
                    "values": [2.0, 3.0, 4.0],
                })
                assert response["ok"] is True
                deadline = time.monotonic() + 10
                while True:
                    status = client.request({"op": "status"})
                    if status["result"]["samples_received"] == 3:
                        break
                    if time.monotonic() >= deadline:
                        raise AssertionError(
                            "disconnected request did not reach core")
                    time.sleep(0.01)
                assert status["result"]["neural_steps"] == 3
                assert status["result"]["wal_records"] == 3
            assert process.poll() is None, (
                "client disconnect terminated gateway")
        finally:
            process.terminate()
            process.wait(timeout=10)
        from storage.wal import iter_records
        records = list(iter_records(os.path.join(data_dir, "trajectory.wal")))
        assert len(records) == 3


def run_all_tests() -> None:
    tests = [
        test_schema_rejects_bad_payloads_without_core_contamination,
        test_ten_clients_ingest_and_query_five_thousand_steps,
        test_aligner_and_temporal_buffer_are_safe_under_shared_access,
        test_shared_nengo_simulator_serializes_concurrent_steps,
        test_full_runtime_connects_neural_core_dual_opinion_and_wal,
        test_runtime_controls_drive_chaos_breaker_and_real_branch_replay,
        test_runtime_sqlite_logging_is_safe_from_client_threads,
        test_wal_recovery_truncates_only_an_incomplete_tail,
        test_disconnected_client_does_not_stop_isolated_gateway_process,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    run_all_tests()
