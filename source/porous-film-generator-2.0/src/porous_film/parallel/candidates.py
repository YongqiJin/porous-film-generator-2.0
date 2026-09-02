from __future__ import annotations

import time
import traceback
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from porous_film.centers import generate_centers
from porous_film.config import GeneratorConfig
from porous_film.geometry import BuiltGeometry, build_units
from porous_film.geometry.scaling import scale_built_geometry, scale_solver_tolerance
from porous_film.metrics import AuditResult, audit_target_distributions
from porous_film.parallel.payloads import canonical_config_payload
from porous_film.parallel.runtime import numeric_thread_limits
from porous_film.performance import (
    PerformanceSnapshot,
    RuntimeProfiler,
    activate_runtime_profiler,
    profile_stage,
)
from porous_film.voxel import PhaseGrid, solve_scale_for_porosity
from porous_film.voxel.grid import voxelize_geometry

_WORKER_FORCED_SERIAL_SOURCE = "worker:forced-serial"


@dataclass
class GeometryAcceptanceError(RuntimeError):
    reason: str
    warnings: list[Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", (self.reason,))


@dataclass(frozen=True)
class CandidateIdentity:
    seed: int
    round_index: int
    candidate_index: int
    derived_random_seed: int
    sequence_index: int


@dataclass(frozen=True)
class CandidateTask:
    identity: CandidateIdentity
    config_payload: dict[str, object]

    @property
    def sequence_index(self) -> int:
        return self.identity.sequence_index


@dataclass(frozen=True)
class CandidateArtifacts:
    identity: CandidateIdentity
    built: BuiltGeometry
    phase_grid: PhaseGrid
    audit: AuditResult
    scale: float
    performance: PerformanceSnapshot


@dataclass(frozen=True)
class CandidateResult:
    identity: CandidateIdentity
    succeeded: bool
    audit_passed: bool
    porosity: float | None
    scale: float | None
    warnings: tuple[str, ...]
    failure_type: str | None
    failure_message: str | None
    traceback_text: str | None
    wall_time_seconds: float
    performance: PerformanceSnapshot | None = None

    @property
    def sequence_index(self) -> int:
        return self.identity.sequence_index


def build_candidate_tasks(config: GeneratorConfig) -> list[CandidateTask]:
    base_rng = np.random.default_rng(config.task.random_seed)
    base_payload = _worker_config_payload(config)
    tasks: list[CandidateTask] = []
    sequence_index = 0
    for round_index in range(config.audit.maximum_rounds):
        for candidate_index in range(config.audit.candidate_count_per_round):
            seed = int(base_rng.integers(0, np.iinfo(np.int32).max))
            identity = CandidateIdentity(
                seed=int(config.task.random_seed),
                round_index=round_index,
                candidate_index=candidate_index,
                derived_random_seed=seed,
                sequence_index=sequence_index,
            )
            tasks.append(CandidateTask(identity, deepcopy(base_payload)))
            sequence_index += 1
    return tasks


def _worker_config_payload(config: GeneratorConfig) -> dict[str, Any]:
    return canonical_config_payload(_worker_serial_config(config))


def _worker_serial_config(config: GeneratorConfig) -> GeneratorConfig:
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


def evaluate_candidate_task(task: CandidateTask) -> CandidateResult:
    start = time.perf_counter()
    profiler = RuntimeProfiler()
    try:
        config = GeneratorConfig.model_validate(task.config_payload)
        with numeric_thread_limits(1):
            artifacts = _evaluate(config, task.identity, profiler=profiler)
        return CandidateResult(
            identity=task.identity,
            succeeded=True,
            audit_passed=artifacts.audit.passed,
            porosity=float(artifacts.phase_grid.porosity),
            scale=float(artifacts.scale),
            warnings=tuple(str(warning) for warning in artifacts.audit.warnings),
            failure_type=None,
            failure_message=None,
            traceback_text=None,
            wall_time_seconds=time.perf_counter() - start,
            performance=artifacts.performance,
        )
    except Exception as exc:  # noqa: BLE001 - worker must serialize ordinary failures
        return CandidateResult(
            identity=task.identity,
            succeeded=False,
            audit_passed=False,
            porosity=None,
            scale=None,
            warnings=(),
            failure_type=type(exc).__name__,
            failure_message=str(exc),
            traceback_text=traceback.format_exc(),
            wall_time_seconds=time.perf_counter() - start,
            performance=profiler.snapshot(),
        )


def replay_candidate(
    config: GeneratorConfig,
    identity: CandidateIdentity,
    *,
    profiler: RuntimeProfiler | None = None,
) -> CandidateArtifacts:
    return _evaluate(config, identity, profiler=profiler)


def select_candidate(
    config: GeneratorConfig,
    results: Iterable[CandidateResult],
) -> CandidateResult:
    ordered = sorted(results, key=lambda result: result.sequence_index)
    for result in ordered:
        if result.succeeded and result.audit_passed:
            return result

    successful = [result for result in ordered if result.succeeded]
    if not successful:
        raise GeometryAcceptanceError("no_geometry_candidates", ordered)

    return min(
        successful,
        key=lambda result: (
            abs(float(result.porosity) - config.pores.target_porosity)
            if result.porosity is not None
            else np.inf,
            result.sequence_index,
        ),
    )


def _evaluate(
    config: GeneratorConfig,
    identity: CandidateIdentity,
    *,
    profiler: RuntimeProfiler | None = None,
) -> CandidateArtifacts:
    runtime = profiler or RuntimeProfiler()
    with activate_runtime_profiler(runtime):
        rng = np.random.default_rng(identity.derived_random_seed)
        with profile_stage("center_seed_generation"):
            center_plan = generate_centers(config, rng)
        with profile_stage("shape_generation"):
            base_built = build_units(config, center_plan, rng)

        def build_at_scale(value: float) -> BuiltGeometry:
            with profile_stage("shape_generation"):
                return scale_built_geometry(base_built, value)

        with profile_stage("scale_optimization"):
            scale, built, fine_grid = solve_scale_for_porosity(
                build_at_scale,
                config.pores.target_porosity,
                scale_solver_tolerance(config, config.audit.fine_spacing_A),
                voxel_spacing_A=config.audit.fine_spacing_A,
            )
        voxelize_geometry(
            built.geometry,
            built.geometry.target_box_A,
            config.audit.coarse_spacing_A,
        )
        audit = audit_target_distributions(config, built, center_plan, fine_grid)
    return CandidateArtifacts(
        identity,
        built,
        fine_grid,
        audit,
        float(scale),
        runtime.snapshot(),
    )


__all__ = [
    "CandidateArtifacts",
    "CandidateIdentity",
    "CandidateResult",
    "CandidateTask",
    "GeometryAcceptanceError",
    "build_candidate_tasks",
    "evaluate_candidate_task",
    "replay_candidate",
    "select_candidate",
]
