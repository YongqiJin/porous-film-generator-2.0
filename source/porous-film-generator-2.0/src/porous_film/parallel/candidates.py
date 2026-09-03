from __future__ import annotations

import time
import traceback
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from porous_film.centers import generate_centers
from porous_film.config import GeneratorConfig
from porous_film.geometry import (
    BuiltGeometry,
    ChannelUnit,
    adjust_channel_lateral_deviations_xy,
    build_units,
)
from porous_film.geometry.scaling import (
    minimum_scale_for_channels_through_z,
    scale_built_geometry,
    scale_solver_tolerance,
    separate_channel_footprints_xy,
)
from porous_film.metrics import (
    AuditResult,
    audit_target_distributions,
    compare_samples_to_distribution,
    measure_final_geometry,
)
from porous_film.parallel.payloads import canonical_config_payload
from porous_film.parallel.runtime import numeric_thread_limits
from porous_film.performance import (
    PerformanceSnapshot,
    RuntimeProfiler,
    activate_runtime_profiler,
    profile_stage,
)
from porous_film.voxel import PhaseGrid, solve_scale_for_porosity, voxelize_geometry

_WORKER_FORCED_SERIAL_SOURCE = "worker:forced-serial"
_FINAL_TORTUOSITY_CALIBRATION_ROUNDS = 2


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
    result, _artifacts = evaluate_candidate_task_with_artifacts(task)
    return result


def evaluate_candidate_task_with_artifacts(
    task: CandidateTask,
) -> tuple[CandidateResult, CandidateArtifacts | None]:
    """Evaluate one local task while retaining its exact scientific artifacts."""
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
        ), artifacts
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
        ), None


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
        require_channels_through_z = config.pore_constraints.z_connectivity == "all_components"
        minimum_scale = (
            max(
                0.1,
                minimum_scale_for_channels_through_z(base_built)
                * (1.0 + 16.0 * np.finfo(float).eps),
            )
            if require_channels_through_z
            else 0.1
        )

        def build_at_scale(value: float) -> BuiltGeometry:
            with profile_stage("shape_generation"):
                scaled = scale_built_geometry(
                    base_built,
                    value,
                    require_channels_through_z=require_channels_through_z,
                )
                shape = config.formal_targets.shape
                requires_distinct_channel_measurements = any(
                    target is not None
                    for target in (
                        shape.equivalent_diameter_A,
                        shape.orientation,
                        shape.channel_aspect_ratio,
                        shape.channel_tortuosity,
                        shape.curvature_fluctuation,
                    )
                )
                if require_channels_through_z and requires_distinct_channel_measurements:
                    return separate_channel_footprints_xy(scaled)
                return scaled

        with profile_stage("scale_optimization"):
            scale, built, fine_grid = solve_scale_for_porosity(
                build_at_scale,
                config.pores.target_porosity,
                scale_solver_tolerance(config, config.audit.fine_spacing_A),
                lower=minimum_scale,
                voxel_spacing_A=config.audit.fine_spacing_A,
            )
        built, fine_grid = _calibrate_final_channel_tortuosity(config, built, fine_grid)
        audit = audit_target_distributions(config, built, center_plan, fine_grid)
    return CandidateArtifacts(
        identity,
        built,
        fine_grid,
        audit,
        float(scale),
        runtime.snapshot(),
    )


def _calibrate_final_channel_tortuosity(
    config: GeneratorConfig,
    built: BuiltGeometry,
    fine_grid: PhaseGrid,
) -> tuple[BuiltGeometry, PhaseGrid]:
    target_spec = config.formal_targets.shape.channel_tortuosity
    if (
        config.source_schema_version != 3
        or config.pore_constraints.z_connectivity != "all_components"
        or target_spec is None
        or config.formal_targets.shape.channel_aspect_ratio is not None
    ):
        return built, fine_grid

    channels = [unit for unit in built.units if isinstance(unit, ChannelUnit)]
    if not channels:
        return built, fine_grid
    target = np.asarray([float(unit.latent_tau or 1.0) for unit in channels], dtype=float)
    observed = _measured_tortuosity_by_channel(config, built, fine_grid)
    if observed is None:
        return built, fine_grid
    comparison = compare_samples_to_distribution(observed, target_spec, 0.20, 0.20)
    if comparison.passed:
        return built, fine_grid

    straight_factors = {unit.unit_id: 0.0 for unit in channels}
    straight = adjust_channel_lateral_deviations_xy(built, straight_factors)
    straight_grid = voxelize_geometry(
        straight.geometry,
        straight.geometry.target_box_A,
        config.audit.fine_spacing_A,
    )
    baseline = _measured_tortuosity_by_channel(config, straight, straight_grid)
    if baseline is None:
        return built, fine_grid

    factors = np.ones(len(channels), dtype=float)
    current_built = built
    current_grid = fine_grid
    best_built = built
    best_grid = fine_grid
    best_score = float(comparison.ks + comparison.normalized_wasserstein)
    for _ in range(_FINAL_TORTUOSITY_CALIBRATION_ROUNDS):
        rank_matched_target = np.empty_like(target)
        rank_matched_target[np.argsort(observed)] = np.sort(target)
        factors = _updated_tortuosity_factors(
            factors,
            baseline=baseline,
            observed=observed,
            target=rank_matched_target,
        )
        factors_by_id = {
            unit.unit_id: float(factor) for unit, factor in zip(channels, factors, strict=True)
        }
        current_built = separate_channel_footprints_xy(
            adjust_channel_lateral_deviations_xy(built, factors_by_id)
        )
        current_grid = voxelize_geometry(
            current_built.geometry,
            current_built.geometry.target_box_A,
            config.audit.fine_spacing_A,
        )
        observed = _measured_tortuosity_by_channel(config, current_built, current_grid)
        if observed is None:
            break
        comparison = compare_samples_to_distribution(observed, target_spec, 0.20, 0.20)
        score = float(comparison.ks + comparison.normalized_wasserstein)
        if score < best_score:
            best_built = current_built
            best_grid = current_grid
            best_score = score
        if comparison.passed:
            return current_built, current_grid
    return best_built, best_grid


def _updated_tortuosity_factors(
    current: np.ndarray,
    *,
    baseline: np.ndarray,
    observed: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    current_values = np.asarray(current, dtype=float)
    baseline_values = np.asarray(baseline, dtype=float)
    observed_values = np.asarray(observed, dtype=float)
    target_values = np.asarray(target, dtype=float)
    if not (
        current_values.shape
        == baseline_values.shape
        == observed_values.shape
        == target_values.shape
    ):
        raise ValueError("tortuosity feedback arrays must have matching shapes")
    desired_excess = np.maximum(target_values - baseline_values, 0.0)
    observed_excess = observed_values - baseline_values
    ratio = np.divide(
        desired_excess,
        observed_excess,
        out=np.full_like(desired_excess, 4.0),
        where=observed_excess > 1.0e-8,
    )
    step = np.clip(np.sqrt(np.maximum(ratio, 0.0)), 0.5, 2.0)
    updated = current_values * step
    updated[desired_excess <= 1.0e-8] = 0.0
    return np.clip(updated, 0.0, 4.0)


def _measured_tortuosity_by_channel(
    config: GeneratorConfig,
    built: BuiltGeometry,
    grid: PhaseGrid,
) -> np.ndarray | None:
    channels = [unit for unit in built.units if isinstance(unit, ChannelUnit)]
    measurements = measure_final_geometry(grid, config.measurement)
    tracks = [track for track in measurements.centerlines if track.is_through]
    if len(tracks) < len(channels):
        return None
    geometry_by_track = {
        value.track_id: value
        for value in measurements.channel_geometries
        if value.valid and value.tortuosity is not None
    }
    if len(geometry_by_track) < len(channels):
        return None
    box_xy = np.asarray(built.geometry.target_box_A[:2], dtype=float)
    costs = np.asarray(
        [
            [_channel_track_xy_cost(unit, track.points_unwrapped_A, box_xy) for track in tracks]
            for unit in channels
        ],
        dtype=float,
    )
    row_indices, column_indices = linear_sum_assignment(costs)
    output = np.full(len(channels), np.nan, dtype=float)
    for row, column in zip(row_indices, column_indices, strict=True):
        geometry = geometry_by_track.get(tracks[int(column)].track_id)
        if geometry is not None:
            output[int(row)] = float(geometry.tortuosity)
    return output if np.all(np.isfinite(output)) else None


def _channel_track_xy_cost(
    unit: ChannelUnit,
    track_points_A: np.ndarray,
    box_xy_A: np.ndarray,
) -> float:
    centerline = np.asarray(unit.centerline_samples_A, dtype=float)
    order = np.argsort(centerline[:, 2])
    sorted_centerline = centerline[order]
    unique_z, unique_indices = np.unique(sorted_centerline[:, 2], return_index=True)
    unique_centerline = sorted_centerline[unique_indices]
    track = np.asarray(track_points_A, dtype=float)
    predicted_xy = np.column_stack(
        [np.interp(track[:, 2], unique_z, unique_centerline[:, axis]) for axis in (0, 1)]
    )
    delta = predicted_xy - track[:, :2]
    delta -= box_xy_A * np.round(delta / box_xy_A)
    return float(np.mean(np.linalg.norm(delta, axis=1)))


__all__ = [
    "CandidateArtifacts",
    "CandidateIdentity",
    "CandidateResult",
    "CandidateTask",
    "GeometryAcceptanceError",
    "build_candidate_tasks",
    "evaluate_candidate_task",
    "evaluate_candidate_task_with_artifacts",
    "replay_candidate",
    "select_candidate",
]
