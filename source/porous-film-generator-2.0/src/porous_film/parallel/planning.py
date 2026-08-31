from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from porous_film.config import GeneratorConfig
from porous_film.parallel.resources import ResourceSnapshot


@dataclass(frozen=True)
class ParallelExecutionPlan:
    command: Literal["generate", "generate-geometry"]
    requested_enabled: bool
    requested_strategy: str
    effective_strategy: Literal["serial", "seeds", "candidates"]
    worker_count: int
    seed_task_count: int
    candidate_task_count: int
    parallel_task_count: int
    total_scientific_task_count: int
    estimated_worker_memory_bytes: int
    worker_limits: dict[str, int]
    limiting_factors: tuple[str, ...]
    resources: ResourceSnapshot
    warnings: tuple[str, ...]
    fallback_reason: str | None
    requested_settings: dict[str, Any] = field(default_factory=dict)
    settings_sources: dict[str, str] = field(default_factory=dict)


def build_execution_plan(
    config: GeneratorConfig,
    *,
    command: Literal["generate", "generate-geometry"],
    seed_task_count: int,
    candidate_task_count: int,
    resources: ResourceSnapshot,
    estimated_worker_memory_bytes: int,
) -> ParallelExecutionPlan:
    spec = config.parallel
    requested_strategy = spec.strategy
    planned_strategy, task_count, fallback_reason = _select_strategy(
        enabled=spec.enabled,
        strategy=requested_strategy,
        seed_task_count=seed_task_count,
        candidate_task_count=candidate_task_count,
    )
    warnings = list(resources.warnings)
    safe_estimate = max(1, int(estimated_worker_memory_bytes))
    cpu_limit = max(1, math.floor(spec.cpu_fraction * resources.available_physical_cores))
    raw_memory = math.floor(
        spec.memory_fraction * resources.effective_memory_bytes / safe_estimate
    )
    memory_limit = max(1, raw_memory)
    if raw_memory < 1:
        warnings.append("memory capacity supports less than one worker; using serial fallback")
    limits = {"tasks": task_count, "cpu": cpu_limit, "memory": memory_limit}
    if spec.max_workers is not None:
        limits["user"] = spec.max_workers
    worker_count = max(1, min(limits.values()))
    effective_strategy = planned_strategy
    if planned_strategy != "serial" and worker_count == 1:
        effective_strategy = "serial"
        fallback_reason = fallback_reason or "worker count resolved to one"
    limiting_factors = tuple(
        name for name, limit in limits.items() if limit == worker_count
    )
    return ParallelExecutionPlan(
        command=command,
        requested_enabled=spec.enabled,
        requested_strategy=requested_strategy,
        effective_strategy=effective_strategy,
        worker_count=worker_count,
        seed_task_count=seed_task_count,
        candidate_task_count=candidate_task_count,
        parallel_task_count=task_count,
        total_scientific_task_count=max(1, seed_task_count) * max(1, candidate_task_count),
        estimated_worker_memory_bytes=int(estimated_worker_memory_bytes),
        worker_limits=limits,
        limiting_factors=limiting_factors,
        resources=resources,
        warnings=tuple(warnings),
        fallback_reason=fallback_reason,
        requested_settings={
            "enabled": spec.enabled,
            "strategy": spec.strategy,
            "max_workers": spec.max_workers,
            "cpu_fraction": spec.cpu_fraction,
            "memory_fraction": spec.memory_fraction,
            "worker_threads": spec.worker_threads,
            "start_method": spec.start_method,
        },
        settings_sources={
            "enabled": spec.sources.enabled,
            "strategy": spec.sources.strategy,
            "max_workers": spec.sources.max_workers,
            "cpu_fraction": "config",
            "memory_fraction": "config",
            "worker_threads": "config",
            "start_method": "config",
        },
    )


def _select_strategy(
    *,
    enabled: bool,
    strategy: str,
    seed_task_count: int,
    candidate_task_count: int,
) -> tuple[Literal["serial", "seeds", "candidates"], int, str | None]:
    if not enabled:
        return "serial", 1, None
    if strategy == "serial":
        return "serial", 1, None
    if strategy == "auto":
        if seed_task_count > 1:
            return "seeds", seed_task_count, None
        if candidate_task_count > 1:
            return "candidates", candidate_task_count, None
        return "serial", 1, "only one independent task"
    if strategy == "seeds":
        if seed_task_count > 1:
            return "seeds", seed_task_count, None
        return "serial", 1, "only one seed task"
    if strategy == "candidates":
        if candidate_task_count > 1:
            return "candidates", candidate_task_count, None
        return "serial", 1, "only one candidate task"
    raise ValueError(f"unsupported parallel strategy: {strategy}")
