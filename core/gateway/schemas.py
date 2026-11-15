"""Strict, dependency-free JSON contracts for gateway requests and responses."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

MAX_DIMENSION = 1024
MAX_REQUEST_ID_LENGTH = 64
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_OPERATIONS = {"ingest", "query", "reset", "status", "control"}
_BRANCH_MODES = {"gain", "noise", "step", "dropout"}


class SchemaValidationError(ValueError):
    """A payload does not conform to the public gateway contract."""


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{name} must be a number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise SchemaValidationError(f"{name} must be finite") from None
    if not math.isfinite(result):
        raise SchemaValidationError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class GatewayRequest:
    op: str
    request_id: str | None = None
    timestamp: float | None = None
    values: tuple[float, ...] | None = None
    action: str | None = None
    mode: str | None = None
    branch_mode: str | None = None
    step: int | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> "GatewayRequest":
        if not isinstance(payload, Mapping):
            raise SchemaValidationError("request must be a JSON object")
        if not all(isinstance(key, str) for key in payload):
            raise SchemaValidationError("request keys must be strings")

        op = payload.get("op")
        if not isinstance(op, str) or op not in _OPERATIONS:
            raise SchemaValidationError(
                "op must be ingest, query, reset, status, or control")

        request_id = payload.get("request_id")
        if request_id is not None and (
                not isinstance(request_id, str)
                or len(request_id) > MAX_REQUEST_ID_LENGTH
                or not _REQUEST_ID.fullmatch(request_id)):
            raise SchemaValidationError("request_id has an invalid format")

        fields = {
            "ingest": {"op", "request_id", "timestamp", "values"},
            "query": {"op", "request_id", "timestamp"},
            "reset": {"op", "request_id"},
            "status": {"op", "request_id"},
            "control": {"op", "request_id", "action", "mode",
                        "branch_mode", "step"},
        }[op]
        if set(payload) - fields:
            raise SchemaValidationError("request contains unsupported fields")

        if op in {"ingest", "query"}:
            if "timestamp" not in payload:
                raise SchemaValidationError("timestamp is required")
            timestamp = _number(payload["timestamp"], "timestamp")
        else:
            timestamp = None

        values = None
        if op == "ingest":
            raw_values = payload.get("values")
            if not isinstance(raw_values, list) or not raw_values:
                raise SchemaValidationError("values must be a non-empty number array")
            if len(raw_values) > MAX_DIMENSION:
                raise SchemaValidationError("values exceeds the maximum dimension")
            values = tuple(_number(value, "values entry") for value in raw_values)
        elif "values" in payload:
            raise SchemaValidationError("values are only valid for ingest")

        action = mode = branch_mode = None
        step = None
        if op == "control":
            action = payload.get("action")
            if action == "set_mode":
                if set(payload) - {"op", "request_id", "action", "mode"}:
                    raise SchemaValidationError("set_mode has unsupported fields")
                mode = payload.get("mode")
                if not isinstance(mode, str) or mode not in {"live", "combat5"}:
                    raise SchemaValidationError("mode must be live or combat5")
            elif action == "reset_breaker":
                if set(payload) - {"op", "request_id", "action"}:
                    raise SchemaValidationError(
                        "reset_breaker has unsupported fields")
            elif action == "branch":
                if set(payload) - {
                        "op", "request_id", "action", "branch_mode", "step"}:
                    raise SchemaValidationError("branch has unsupported fields")
                branch_mode = payload.get("branch_mode")
                if (not isinstance(branch_mode, str)
                        or branch_mode not in _BRANCH_MODES):
                    raise SchemaValidationError(
                        "branch_mode must be gain, noise, step, or dropout")
                raw_step = payload.get("step")
                if (isinstance(raw_step, bool) or not isinstance(raw_step, int)
                        or raw_step < 0):
                    raise SchemaValidationError(
                        "step must be a non-negative integer")
                step = raw_step
            else:
                raise SchemaValidationError(
                    "action must be set_mode, reset_breaker, or branch")

        return cls(op=op, request_id=request_id, timestamp=timestamp,
                   values=values, action=action, mode=mode,
                   branch_mode=branch_mode, step=step)


@dataclass(frozen=True)
class GatewayResponse:
    ok: bool
    request_id: str | None = None
    result: Mapping[str, Any] | None = None
    error: Mapping[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        response: dict[str, Any] = {"ok": self.ok}
        if self.request_id is not None:
            response["request_id"] = self.request_id
        if self.ok:
            response["result"] = dict(self.result or {})
        else:
            response["error"] = dict(self.error or {})
        return response

    @classmethod
    def success(cls, result: Mapping[str, Any],
                request_id: str | None = None) -> "GatewayResponse":
        return cls(ok=True, request_id=request_id, result=result)

    @classmethod
    def failure(cls, code: str, message: str,
                request_id: str | None = None) -> "GatewayResponse":
        return cls(ok=False, request_id=request_id,
                   error={"code": code, "message": message})
