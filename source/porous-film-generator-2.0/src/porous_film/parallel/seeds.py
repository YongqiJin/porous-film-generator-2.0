from __future__ import annotations

import json
import os
import pickle
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from porous_film.config import GeneratorConfig
from porous_film.optimization import write_optimizer_exchange
from porous_film.parallel.payloads import canonical_config_payload
from porous_film.parallel.planning import ParallelExecutionPlan
from porous_film.parallel.runtime import ParallelPoolError
from porous_film.storage import TaskPaths, create_task_paths_at_root

_WORKER_FORCED_SERIAL_SOURCE = "worker-forced-serial"


@dataclass(frozen=True)
class SeedIdentity:
    panel_index: int
    seed: int
    sequence_index: int


@dataclass(frozen=True)
class SeedTask:
    identity: SeedIdentity
    config_payload: dict[str, object]
    task_root: str
    top_level_run_identifier: str
    representative: bool

    @property
    def sequence_index(self) -> int:
        return self.identity.sequence_index


@dataclass(frozen=True)
class SeedTaskResult:
    identity: SeedIdentity
    status: str
    feasible: bool
    objective: float | None
    failure_reason: str | None
    artifact_root: str
    optimizer_payload: dict[str, object]
    representative_pickle: str | None
    candidate_summary: dict[str, object]
    traceback_text: str | None
    wall_time_seconds: float

    @property
    def sequence_index(self) -> int:
        return self.identity.sequence_index


def build_seed_tasks(
    config: GeneratorConfig,
    paths: TaskPaths,
    execution_plan: ParallelExecutionPlan,
    top_level_run_identifier: str,
) -> list[SeedTask]:
    """Build isolated seed tasks under ``analysis/seed_panel/<index>-<seed>``."""
    seeds = tuple(int(seed) for seed in (config.optimization.seed_panel or (config.task.random_seed,)))
    representative_index = seeds.index(config.task.random_seed) if config.task.random_seed in seeds else 0
    force_child_serial = execution_plan.effective_strategy == "seeds"
    tasks: list[SeedTask] = []
    for panel_index, seed in enumerate(seeds):
        seed_config = _config_for_seed(config, seed)
        if force_child_serial:
            seed_config = _worker_forced_serial_config(seed_config)
        tasks.append(
            SeedTask(
                identity=SeedIdentity(
                    panel_index=panel_index,
                    seed=seed,
                    sequence_index=panel_index,
                ),
                config_payload=canonical_config_payload(seed_config),
                task_root=str(paths.analysis / "seed_panel" / f"{panel_index}-{seed}"),
                top_level_run_identifier=top_level_run_identifier,
                representative=panel_index == representative_index,
            )
        )
    return tasks


def execute_seed_task(task: SeedTask) -> SeedTaskResult:
    """Picklable process-pool wrapper for a single seed task."""
    start = time.perf_counter()
    try:
        result, representative = _execute_seed_body(task, candidate_executor=None)
        if task.representative and representative is not None:
            paths = representative.paths
            pickle_path = paths.work / "representative-run.pkl"
            temporary = pickle_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                pickle.dump(representative, handle, protocol=5)
            os.replace(temporary, pickle_path)
            return replace(result, representative_pickle=str(pickle_path))
        return result
    except ParallelPoolError:
        raise
    except Exception as exc:  # noqa: BLE001 - seed-local failures become ordered records.
        seed_root = Path(task.task_root)
        seed_config = _safe_seed_config(task)
        paths = _failure_task_paths(seed_root)
        return _failure_seed_result(task, seed_config, seed_root, paths, exc, start)


def _execute_seed_body(task: SeedTask, candidate_executor: object | None = None):
    from porous_film.pipeline import (
        _execute_single_seed,
        _failure_reason_from_result,
        _objective_value,
        optimizer_payload_for_result,
        prepare_task_inputs,
    )

    start = time.perf_counter()
    seed_root = Path(task.task_root)
    seed_config: GeneratorConfig | None = None
    paths: TaskPaths | None = None
    try:
        seed_config = GeneratorConfig.model_validate(task.config_payload)
        paths = create_task_paths_at_root(seed_root)
        prepare_task_inputs(seed_config, paths)
        run_result = _execute_single_seed(
            seed_config,
            paths,
            write_artifacts=True,
            run_identifier=task.top_level_run_identifier,
            candidate_executor=candidate_executor,
        )
        payload = optimizer_payload_for_result(run_result)
        write_optimizer_exchange(seed_root, **payload)
        feasible = run_result.status == "completed_feasible"
        seed_result = SeedTaskResult(
            identity=task.identity,
            status=run_result.status,
            feasible=feasible,
            objective=_objective_value(seed_config, run_result),
            failure_reason=None if feasible else _failure_reason_from_result(run_result),
            artifact_root=str(paths.root),
            optimizer_payload=dict(payload),
            representative_pickle=None,
            candidate_summary=_candidate_summary(run_result),
            traceback_text=None,
            wall_time_seconds=time.perf_counter() - start,
        )
        _write_seed_record(paths, seed_result)
        return seed_result, run_result if task.representative else None
    except ParallelPoolError:
        raise
    except Exception as exc:  # noqa: BLE001 - seed-local failures become ordered records.
        failure_paths = paths or _failure_task_paths(seed_root)
        return _failure_seed_result(task, seed_config, seed_root, failure_paths, exc, start), None


def _safe_seed_config(task: SeedTask) -> GeneratorConfig | None:
    try:
        return GeneratorConfig.model_validate(task.config_payload)
    except Exception:  # noqa: BLE001 - preserve the original failure when fallback fails.
        return None


def _failure_seed_result(
    task: SeedTask,
    seed_config: GeneratorConfig | None,
    seed_root: Path,
    failure_paths: TaskPaths,
    exc: Exception,
    start: float,
) -> SeedTaskResult:
    from porous_film.pipeline import optimizer_payload_for_failure

    elapsed = time.perf_counter() - start
    payload = (
        optimizer_payload_for_failure(seed_config, exc)
        if seed_config is not None
        else _fallback_failure_payload(task, exc)
    )
    write_optimizer_exchange(seed_root, **payload)
    status_payload = payload.get("calculation_status", {})
    if isinstance(status_payload, dict):
        _write_json(failure_paths.logs / "failure.json", status_payload)
    seed_result = SeedTaskResult(
        identity=task.identity,
        status="failed",
        feasible=False,
        objective=None,
        failure_reason=f"{type(exc).__name__}: {exc}",
        artifact_root=str(failure_paths.root),
        optimizer_payload=dict(payload),
        representative_pickle=None,
        candidate_summary={},
        traceback_text=traceback.format_exc(),
        wall_time_seconds=elapsed,
    )
    _write_seed_record(failure_paths, seed_result)
    return seed_result


def _fallback_failure_payload(task: SeedTask, exc: Exception) -> dict[str, object]:
    return {
        "requested": {"task_random_seed": task.identity.seed},
        "realized": {},
        "feasible": False,
        "constraints": {"audit_passed": False, "packing_completed": False},
        "calculation_status": {
            "status": "failed",
            "failure_reason": type(exc).__name__,
            "message": str(exc),
            "exception_type": type(exc).__name__,
        },
        "objectives": {},
        "uncertainty": {},
    }


def _config_for_seed(config: GeneratorConfig, seed: int) -> GeneratorConfig:
    return config.model_copy(
        update={
            "task": config.task.model_copy(update={"random_seed": int(seed)}),
            "optimization": config.optimization.model_copy(update={"seed_panel": (int(seed),)}),
        }
    )


def _worker_forced_serial_config(config: GeneratorConfig) -> GeneratorConfig:
    sources = config.parallel.sources.model_copy(
        update={
            "enabled": _WORKER_FORCED_SERIAL_SOURCE,
            "strategy": _WORKER_FORCED_SERIAL_SOURCE,
            "max_workers": _WORKER_FORCED_SERIAL_SOURCE,
        }
    )
    parallel = config.parallel.model_copy(
        update={
            "enabled": False,
            "strategy": "serial",
            "max_workers": None,
            "sources": sources,
        }
    )
    return config.model_copy(update={"parallel": parallel})


def _candidate_summary(run_result: Any) -> dict[str, object]:
    candidates = tuple(run_result.geometry_run.candidate_results)
    return {
        "candidate_count": len(candidates),
        "failed_candidate_count": sum(1 for candidate in candidates if not candidate.succeeded),
        "selected_candidate": asdict(run_result.geometry_run.selected_candidate),
    }


def _seed_record(result: SeedTaskResult) -> dict[str, object]:
    record: dict[str, object] = {
        "panel_index": result.identity.panel_index,
        "seed": result.identity.seed,
        "sequence_index": result.identity.sequence_index,
        "status": result.status,
        "feasible": result.feasible,
        "objective": result.objective,
        "failure_reason": result.failure_reason,
        "artifact_root": result.artifact_root,
    }
    if result.traceback_text is not None:
        record["traceback_file"] = "logs/seed-traceback.txt"
    return record


def _write_seed_record(paths: TaskPaths, result: SeedTaskResult) -> None:
    if result.traceback_text is not None:
        (paths.logs / "seed-traceback.txt").write_text(result.traceback_text, encoding="utf-8")
    _write_json(paths.logs / "seed-record.json", _seed_record(result))


def _failure_task_paths(seed_root: Path) -> TaskPaths:
    paths = {
        name: seed_root / name
        for name in ("inputs", "work", "outputs", "analysis", "reports", "logs", "qa_export")
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return TaskPaths(root=seed_root, **paths)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, separators=(",", ": "))
        + "\n",
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "SeedIdentity",
    "SeedTask",
    "SeedTaskResult",
    "build_seed_tasks",
    "execute_seed_task",
]
