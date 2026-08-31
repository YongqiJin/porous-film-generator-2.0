from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from porous_film.parallel.planning import ParallelExecutionPlan
from porous_film.parallel.runtime import THREAD_ENVIRONMENT
from porous_film.storage import TaskPaths

_SCHEMA_VERSION = 1


def write_parallel_plan(paths: TaskPaths, plan: ParallelExecutionPlan) -> Path:
    """Write the parent-owned parallel execution plan using an atomic replace."""
    path = paths.work / "parallel" / "parallel-plan.json"
    _atomic_write_json(path, _parallel_plan_payload(plan))
    return path


def write_parallel_summary(
    paths: TaskPaths,
    plan: ParallelExecutionPlan,
    task_results: list[object] | tuple[object, ...],
    *,
    started: datetime,
    finished: datetime,
    cancellation: object | None,
    pool_failure: object | None,
    serial_fallback_reason: str | None = None,
) -> Path:
    """Write JSON and Markdown summaries for successful or interrupted runs."""
    records = [_task_record(result) for result in task_results]
    payload = _parallel_summary_payload(
        plan,
        records,
        started=started,
        finished=finished,
        cancellation=cancellation,
        pool_failure=pool_failure,
        serial_fallback_reason=serial_fallback_reason,
    )
    json_path = paths.analysis / "parallel-summary.json"
    _atomic_write_json(json_path, payload)
    _atomic_write_text(paths.reports / "parallel-summary.md", _summary_markdown(payload))
    return json_path


def _parallel_plan_payload(plan: ParallelExecutionPlan) -> dict[str, Any]:
    resources = plan.resources
    requested_settings = dict(getattr(plan, "requested_settings", {}))
    if not requested_settings:
        requested_settings = {
            "enabled": plan.requested_enabled,
            "strategy": plan.requested_strategy,
        }
    return {
        "schema_version": _SCHEMA_VERSION,
        "command": plan.command,
        "requested_enabled": plan.requested_enabled,
        "requested_strategy": plan.requested_strategy,
        "requested_settings": requested_settings,
        "settings_sources": dict(getattr(plan, "settings_sources", {})),
        "effective_strategy": plan.effective_strategy,
        "worker_count": plan.worker_count,
        "seed_task_count": plan.seed_task_count,
        "candidate_task_count": plan.candidate_task_count,
        "parallel_task_count": plan.parallel_task_count,
        "total_scientific_task_count": plan.total_scientific_task_count,
        "estimated_worker_memory_bytes": plan.estimated_worker_memory_bytes,
        "worker_limits": dict(plan.worker_limits),
        "limiting_factors": list(plan.limiting_factors),
        "resources": {
            "platform_name": resources.platform_name,
            "allowed_logical_cpus": list(resources.allowed_logical_cpus),
            "allowed_logical_count": resources.allowed_logical_count,
            "available_physical_cores": resources.available_physical_cores,
            "physical_core_source": resources.physical_core_source,
            "available_memory_bytes": resources.available_memory_bytes,
            "effective_memory_bytes": resources.effective_memory_bytes,
            "audit_memory_cap_bytes": resources.audit_memory_cap_bytes,
            "warnings": list(resources.warnings),
        },
        "blas_environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT},
        "warnings": list(plan.warnings),
        "fallback_reason": plan.fallback_reason,
    }


def _parallel_summary_payload(
    plan: ParallelExecutionPlan,
    records: list[dict[str, Any]],
    *,
    started: datetime,
    finished: datetime,
    cancellation: object | None,
    pool_failure: object | None,
    serial_fallback_reason: str | None,
) -> dict[str, Any]:
    completed_count = sum(1 for record in records if _is_completed(record))
    failed_count = sum(1 for record in records if _is_failed(record))
    cancelled_count = sum(1 for record in records if record.get("status") == "cancelled")
    plan_payload = _parallel_plan_payload(plan)
    return {
        "schema_version": _SCHEMA_VERSION,
        "command": plan.command,
        "effective_strategy": plan.effective_strategy,
        "worker_count": plan.worker_count,
        "parallel_task_count": plan.parallel_task_count,
        "task_count": len(records),
        "completed_count": completed_count,
        "failed_count": failed_count,
        "cancelled_count": cancelled_count,
        "started_at": _isoformat(started),
        "finished_at": _isoformat(finished),
        "total_wall_time_seconds": max(0.0, (finished - started).total_seconds()),
        "plan": plan_payload,
        "tasks": records,
        "pool_failure": _failure_payload(pool_failure),
        "cancellation": _json_ready(cancellation),
        "fallback_reason": plan.fallback_reason,
        "serial_fallback_reason": serial_fallback_reason,
    }


def _task_record(result: object) -> dict[str, Any]:
    if isinstance(result, dict):
        payload = dict(result)
        payload["identity"] = _json_ready(payload.get("identity", {}))
        payload["failure"] = _json_ready(payload.get("failure"))
        return _json_ready(payload)

    identity = _identity_payload(getattr(result, "identity", None))
    status = getattr(result, "status", None)
    failure_type = getattr(result, "failure_type", None)
    failure_message = getattr(result, "failure_message", None)
    failure_reason = getattr(result, "failure_reason", None)
    if status is None:
        succeeded = getattr(result, "succeeded", None)
        status = "completed" if succeeded is True else "failed"
    failure = None
    if failure_type is not None or failure_message is not None:
        failure = {"type": failure_type, "message": failure_message}
    elif failure_reason:
        failure = {"type": "failure", "message": failure_reason}
    return {
        "identity": identity,
        "status": status,
        "wall_time_seconds": getattr(result, "wall_time_seconds", None),
        "failure": failure,
    }


def _identity_payload(identity: object) -> dict[str, Any]:
    if identity is None:
        return {}
    if is_dataclass(identity):
        return _json_ready(asdict(identity))
    if isinstance(identity, dict):
        return _json_ready(identity)
    return _json_ready(vars(identity))


def _is_completed(record: dict[str, Any]) -> bool:
    status = record.get("status")
    return status not in {"failed", "cancelled"}


def _is_failed(record: dict[str, Any]) -> bool:
    return record.get("status") == "failed"


def _failure_payload(value: object | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return _json_ready(value)
    payload = {
        "type": type(value).__name__,
        "message": str(value),
    }
    started = getattr(value, "started_task_count", None)
    if started is not None:
        payload["started_task_count"] = started
    return payload


def _summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Parallel Execution Summary",
        "",
        "## Plan",
        f"- Command: {payload['command']}",
        f"- Effective strategy: {payload['effective_strategy']}",
        f"- Workers: {payload['worker_count']}",
        f"- Parallel task count: {payload['parallel_task_count']}",
        f"- Fallback reason: {payload['fallback_reason']}",
        f"- Serial fallback reason: {payload['serial_fallback_reason']}",
        "",
        "## Timing",
        f"- Started: {payload['started_at']}",
        f"- Finished: {payload['finished_at']}",
        f"- Total wall time seconds: {payload['total_wall_time_seconds']}",
        "",
        "## Results",
        f"- Task count: {payload['task_count']}",
        f"- Completed: {payload['completed_count']}",
        f"- Failed: {payload['failed_count']}",
        f"- Cancelled: {payload['cancelled_count']}",
        "",
        "## Failure and Cancellation",
        f"- Pool failure: {payload['pool_failure']}",
        f"- Cancellation: {payload['cancellation']}",
        "",
        "## Tasks",
    ]
    for record in payload["tasks"]:
        identity = json.dumps(record["identity"], sort_keys=True)
        lines.append(
            f"- {identity}: status={record.get('status')}, "
            f"wall_time_seconds={record.get('wall_time_seconds')}, "
            f"failure={record.get('failure')}"
        )
    lines.append("")
    return "\n".join(lines)


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(
        path,
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _isoformat(value: datetime) -> str:
    return value.isoformat()


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value
