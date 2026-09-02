from __future__ import annotations

import csv
import hashlib
import json
import pickle
import shutil
import time
from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import psutil
import yaml
from scipy import ndimage
from scipy.interpolate import PchipInterpolator

from porous_film.config import GeneratorConfig
from porous_film.geometry import (
    BuiltGeometry,
    ChannelUnit,
    CompactUnit,
)
from porous_film.io import (
    export_semiconductor_glb,
    export_surface_ply,
    write_qa_checksums,
    write_qa_contract,
)
from porous_film.metrics import AuditResult
from porous_film.molecules import (
    MoleculeTemplate,
    PackingConfig,
    PackingError,
    PackingResult,
    pack_molecules,
)
from porous_film.molecules.packing import InstanceTransform
from porous_film.optimization import (
    OptimizerPayload,
    aggregate_seed_results,
    write_optimizer_exchange,
)
from porous_film.parallel import (
    CandidateIdentity,
    CandidateResult,
    GeometryAcceptanceError,
    ParallelCancelled,
    ParallelExecutionPlan,
    ParallelPoolError,
    build_candidate_tasks,
    build_execution_plan,
    build_seed_tasks,
    discover_resources,
    estimate_generation_memory_bytes,
    estimate_worker_memory_bytes,
    evaluate_candidate_task,
    execute_seed_task,
    replay_candidate,
    run_spawn_tasks,
    select_candidate,
    spawn_pool,
    write_parallel_plan,
    write_parallel_summary,
)
from porous_film.parallel.seeds import SeedTask, SeedTaskResult, _execute_seed_body
from porous_film.performance import (
    PerformanceSnapshot,
    RuntimeProfiler,
    activate_runtime_profiler,
    profile_stage,
)
from porous_film.reporting.markdown import write_preflight_report, write_run_report
from porous_film.reporting.visual import write_visual_report
from porous_film.storage import (
    TaskPaths,
    create_task_directory,
    create_task_paths_at_root,
    task_paths_from_existing_root,
)
from porous_film.voxel import PhaseGrid
from porous_film.voxel.grid import _GRID_DIVISIBILITY_TOLERANCE

_DEFAULT_RESULT_ROOT = Path("C:/Calculation_results")
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_GEOMETRY_SCHEMA_VERSION = 2
_MATRIX_PHASE_ID = 0
_PORE_PHASE_ID = 1
_COMPACT_SUPERELLIPSOID_EXPONENT = 2.0


@dataclass(frozen=True)
class PreflightResult:
    passed: bool
    errors: list[str]
    warnings: list[str]
    estimated_voxels: int
    estimated_memory_bytes: int
    report_path: Path | None


@dataclass(frozen=True)
class GeometryRun:
    config: GeneratorConfig
    built: BuiltGeometry
    phase_grid: PhaseGrid
    audit: AuditResult
    paths: TaskPaths
    candidate_results: tuple[CandidateResult, ...]
    selected_candidate: CandidateIdentity
    performance: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunResult:
    paths: TaskPaths
    geometry_run: GeometryRun
    packing_result: PackingResult | None
    status: str


def preflight(
    config: GeneratorConfig,
    result_root: Path | None = None,
) -> PreflightResult:
    """Validate a configuration and write a Markdown preflight report."""
    paths = create_task_directory(
        _result_root(result_root),
        config.task.name,
        datetime.now(_SHANGHAI),
    )
    errors: list[str] = []
    warnings: list[str] = []

    target_box = _target_box(config)
    packing_box = _packing_box(config)
    if np.any(packing_box[:2] != target_box[:2]) or packing_box[2] < target_box[2]:
        errors.append("packing box must cover the target box with identical x/y lengths")

    pdb_valid = config.pore_material is None
    if config.pore_material is not None:
        try:
            MoleculeTemplate.from_pdb(config.pore_material.pdb)
            pdb_valid = True
        except (OSError, ValueError) as exc:
            errors.append(f"pore material PDB is invalid: {exc}")

    if config.seed_count < 1:
        errors.append("center target produces zero pore seeds")
    elif config.seed_count < 5:
        warnings.append("seed panel has limited sample support for distribution audits")

    if config.audit.candidate_count_per_round < 1 or config.audit.maximum_rounds < 1:
        errors.append("candidate search must request at least one candidate")

    for name, spacing in (
        ("coarse", config.audit.coarse_spacing_A),
        ("fine", config.audit.fine_spacing_A),
    ):
        if not _box_axes_are_divisible_by_spacing(target_box, float(spacing)):
            errors.append(
                f"target-box axes must be exactly divisible by {name} audit spacing_A"
            )

    spacing = float(config.audit.fine_spacing_A)
    estimated_voxels = _estimated_voxel_count(target_box, spacing)
    estimated_memory_bytes = _estimated_preflight_memory_bytes(config, target_box)
    configured_cap = config.audit.available_memory_cap_bytes
    available_memory_bytes = configured_cap or _available_memory_bytes()
    if configured_cap is not None and estimated_memory_bytes > configured_cap:
        errors.append("estimated generation memory exceeds configured memory cap")
    elif estimated_memory_bytes > 0.75 * available_memory_bytes:
        warnings.append("estimated fine-grid memory is high relative to available memory")

    passed = not errors
    report = write_preflight_report(
        paths.reports / "preflight-report.md",
        config=config,
        passed=passed,
        errors=errors,
        warnings=warnings,
        estimated_voxels=estimated_voxels,
        estimated_memory_bytes=estimated_memory_bytes,
    )
    _write_normalized_config(config, paths.inputs / "normalized_config.yaml")
    if pdb_valid and config.pore_material is not None:
        _copy_source_pdb(config, paths.inputs / config.pore_material.pdb.name)
    return PreflightResult(
        passed=passed,
        errors=errors,
        warnings=warnings,
        estimated_voxels=estimated_voxels,
        estimated_memory_bytes=estimated_memory_bytes,
        report_path=report,
    )


def generate_geometry(
    config: GeneratorConfig,
    paths: TaskPaths,
    *,
    execution_plan: ParallelExecutionPlan | None = None,
    candidate_executor: Executor | None = None,
) -> GeometryRun:
    """Generate and audit a pore geometry without packing molecules."""
    run_started_at = time.perf_counter()
    tasks = build_candidate_tasks(config)
    plan = execution_plan or _candidate_execution_plan(config, len(tasks))
    write_parallel_plan(paths, plan)
    started = datetime.now(_SHANGHAI)
    candidate_search_started_at = time.perf_counter()
    serial_fallback_reason: str | None = None
    if candidate_executor is not None:
        try:
            results = run_spawn_tasks(
                tasks,
                evaluate_candidate_task,
                worker_count=plan.worker_count,
                worker_threads=config.parallel.worker_threads,
                executor=candidate_executor,
                result_callback=lambda result: _write_parallel_progress(paths, result),
            )
        except ParallelCancelled as exc:
            finished = datetime.now(_SHANGHAI)
            records = _results_with_cancelled_tasks(
                tasks,
                exc.completed_results,
                "cancelled by KeyboardInterrupt",
            )
            write_parallel_summary(
                paths,
                plan,
                records,
                started=started,
                finished=finished,
                cancellation=_cancellation_record(exc, tasks, exc.completed_results),
                pool_failure=None,
            )
            raise
        except ParallelPoolError as exc:
            finished = datetime.now(_SHANGHAI)
            records = _results_with_cancelled_tasks(
                tasks,
                exc.completed_results,
                "missing after process pool failure",
            )
            write_parallel_summary(
                paths,
                plan,
                records,
                started=started,
                finished=finished,
                cancellation=None,
                pool_failure=exc,
            )
            raise
    elif plan.effective_strategy == "candidates":
        pool_entered = False
        try:
            with spawn_pool(plan.worker_count, config.parallel.worker_threads) as executor:
                pool_entered = True
                results = run_spawn_tasks(
                    tasks,
                    evaluate_candidate_task,
                    worker_count=plan.worker_count,
                    worker_threads=config.parallel.worker_threads,
                    executor=executor,
                    result_callback=lambda result: _write_parallel_progress(paths, result),
                )
        except ParallelCancelled as exc:
            finished = datetime.now(_SHANGHAI)
            records = _results_with_cancelled_tasks(
                tasks,
                exc.completed_results,
                "cancelled by KeyboardInterrupt",
            )
            write_parallel_summary(
                paths,
                plan,
                records,
                started=started,
                finished=finished,
                cancellation=_cancellation_record(exc, tasks, exc.completed_results),
                pool_failure=None,
            )
            raise
        except ParallelPoolError as exc:
            finished = datetime.now(_SHANGHAI)
            records = _results_with_cancelled_tasks(
                tasks,
                exc.completed_results,
                "missing after process pool failure",
            )
            write_parallel_summary(
                paths,
                plan,
                records,
                started=started,
                finished=finished,
                cancellation=None,
                pool_failure=exc,
            )
            raise
        except Exception as exc:
            if pool_entered:
                raise
            serial_fallback_reason = f"{type(exc).__name__}: {exc}"
            results = []
            for task in tasks:
                result = evaluate_candidate_task(task)
                results.append(result)
                _write_parallel_progress(paths, result)
    else:
        results = []
        for task in tasks:
            result = evaluate_candidate_task(task)
            results.append(result)
            _write_parallel_progress(paths, result)

    ordered_results = tuple(sorted(results, key=lambda result: result.sequence_index))
    candidate_search_wall_time_seconds = time.perf_counter() - candidate_search_started_at
    write_parallel_summary(
        paths,
        plan,
        ordered_results,
        started=started,
        finished=datetime.now(_SHANGHAI),
        cancellation=None,
        pool_failure=None,
        serial_fallback_reason=serial_fallback_reason,
    )
    _write_candidate_records(paths, ordered_results)

    selected = select_candidate(config, ordered_results)
    replay_profiler = RuntimeProfiler()
    replay = replay_candidate(config, selected.identity, profiler=replay_profiler)
    _require_replay_matches_summary(replay, selected)
    run = GeometryRun(
        config=config,
        built=replay.built,
        phase_grid=replay.phase_grid,
        audit=replay.audit,
        paths=paths,
        candidate_results=ordered_results,
        selected_candidate=selected.identity,
    )
    return _write_geometry_artifacts(
        config,
        run,
        runtime_profiler=replay_profiler,
        run_started_at=run_started_at,
        candidate_search_wall_time_seconds=candidate_search_wall_time_seconds,
        selected_replay_wall_time_seconds=replay.performance.wall_time_seconds,
    )


def _candidate_execution_plan(
    config: GeneratorConfig,
    candidate_task_count: int,
) -> ParallelExecutionPlan:
    return build_execution_plan(
        config,
        command="generate-geometry",
        seed_task_count=1,
        candidate_task_count=candidate_task_count,
        resources=discover_resources(
            audit_memory_cap_bytes=config.audit.available_memory_cap_bytes
        ),
        estimated_worker_memory_bytes=estimate_worker_memory_bytes(config),
    )


def _write_candidate_records(
    paths: TaskPaths,
    results: Sequence[CandidateResult],
) -> None:
    output = paths.qa_export / "unit_candidates.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    _json_ready(_candidate_record(result)),
                    sort_keys=True,
                )
                + "\n"
            )
    _write_candidate_tracebacks(paths, results)


def _candidate_record(result: CandidateResult) -> dict[str, Any]:
    return {
        "identity": _candidate_identity_record(result.identity),
        "scale": result.scale,
        "porosity": result.porosity,
        "succeeded": result.succeeded,
        "audit_passed": result.audit_passed,
        "warnings": list(result.warnings),
        "failure_type": result.failure_type,
        "failure_message": result.failure_message,
        "wall_time_seconds": result.wall_time_seconds,
    }


def _geometry_performance_record(
    snapshot: PerformanceSnapshot,
    *,
    candidates: Sequence[CandidateResult],
    candidate_search_wall_time_seconds: float | None,
    selected_replay_wall_time_seconds: float | None,
    total_wall_time_seconds: float,
) -> dict[str, Any]:
    worker_peaks = [
        float(result.performance.peak_rss_mib)
        for result in candidates
        if result.performance is not None
    ]
    return {
        "measurement_scope": (
            "wall-clock timings through scientific artifact export; visual-report rendering "
            "and final checksum refresh are excluded"
        ),
        "total_wall_time_seconds": float(total_wall_time_seconds),
        "candidate_search_wall_time_seconds": candidate_search_wall_time_seconds,
        "selected_replay_wall_time_seconds": selected_replay_wall_time_seconds,
        "candidate_wall_time_sum_seconds": float(
            sum(result.wall_time_seconds for result in candidates)
        ),
        "stage_timings_seconds": snapshot.stage_seconds,
        "stage_inclusive_seconds": snapshot.stage_inclusive_seconds,
        "stage_call_counts": snapshot.stage_call_counts,
        "rss_start_mib": float(snapshot.rss_start_mib),
        "peak_rss_mib": float(snapshot.peak_rss_mib),
        "worker_peak_rss_mib": max(worker_peaks) if worker_peaks else None,
    }


def _candidate_identity_record(identity: CandidateIdentity) -> dict[str, int]:
    return {
        "seed": identity.seed,
        "round_index": identity.round_index,
        "candidate_index": identity.candidate_index,
        "derived_random_seed": identity.derived_random_seed,
        "sequence_index": identity.sequence_index,
    }


def _write_candidate_tracebacks(
    paths: TaskPaths,
    results: Sequence[CandidateResult],
) -> None:
    traceback_dir = paths.logs / "candidates"
    for result in results:
        if result.traceback_text is None:
            continue
        traceback_dir.mkdir(parents=True, exist_ok=True)
        (traceback_dir / f"{result.sequence_index}.traceback.txt").write_text(
            result.traceback_text,
            encoding="utf-8",
        )


def _require_replay_matches_summary(
    replay: Any,
    summary: CandidateResult,
) -> None:
    if (
        summary.scale is None
        or summary.porosity is None
        or replay.scale != summary.scale
        or replay.phase_grid.porosity != summary.porosity
        or replay.audit.passed != summary.audit_passed
        or tuple(str(warning) for warning in replay.audit.warnings)
        != summary.warnings
    ):
        raise RuntimeError("candidate replay diverged from worker summary")


def run_full(
    config: GeneratorConfig,
    result_root: Path,
    *,
    config_path: Path | None = None,
) -> RunResult:
    """Run geometry generation, molecule packing, handoff export, and reporting."""
    paths = create_task_directory(result_root, config.task.name, datetime.now(_SHANGHAI))
    return _run_full_with_paths(config, paths, config_path=config_path)


def run_full_at_root(
    config: GeneratorConfig,
    task_root: Path,
    *,
    config_path: Path | None = None,
) -> RunResult:
    """Run the full pipeline in exactly ``task_root`` without dated parent folders."""
    return _run_full_with_paths(
        config,
        create_task_paths_at_root(task_root),
        config_path=config_path,
    )


def _run_full_with_paths(
    config: GeneratorConfig,
    paths: TaskPaths,
    *,
    config_path: Path | None = None,
    run_identifier: str | None = None,
) -> RunResult:
    """Run the full pipeline in a preallocated task directory."""
    identifier = run_identifier or paths.root.name
    try:
        return _run_seed_panel(
            config,
            paths,
            config_path=config_path,
            run_identifier=identifier,
        )
    except Exception as exc:
        if not (paths.outputs / "calculation_status.json").is_file():
            _record_failure(paths, exc, config)
        raise


def prepare_task_inputs(
    config: GeneratorConfig,
    paths: TaskPaths,
    config_path: Path | None = None,
) -> None:
    _write_normalized_config(config, paths.inputs / "normalized_config.yaml")
    _write_normalized_config(config, paths.qa_export / "normalized_config.yaml")
    if config_path is not None:
        shutil.copy2(config_path, paths.inputs / Path(config_path).name)
    if config.pore_material is not None:
        _copy_source_pdb(config, paths.inputs / config.pore_material.pdb.name)


def fill_pore_run(run_root: Path) -> Path:
    root = Path(run_root)
    _require_run_artifact(root / "outputs" / "packing_metrics.json")
    _require_run_artifact(root / "outputs" / "pore_reference_coordinates.cif")
    output = root / "analysis" / "fill-pore-status.json"
    _write_json(output, {"status": "existing_pore_fill_available", "run": root.resolve()})
    return output


def audit_run(run_root: Path) -> Path:
    root = Path(run_root)
    feasibility = _read_json(_require_run_artifact(root / "outputs" / "feasibility.json"))
    status = _read_json(_require_run_artifact(root / "outputs" / "calculation_status.json"))
    output = root / "analysis" / "audit-status.json"
    _write_json(
        output,
        {
            "status": "audited_existing_run",
            "run_status": status.get("status"),
            "feasible": bool(feasibility.get("feasible")),
            "constraints": feasibility.get("constraints", {}),
        },
    )
    return output


def audit_packmol_output_run(run_root: Path, structure: Path) -> Path:
    root = Path(run_root)
    structure_path = _require_run_artifact(Path(structure))
    _require_run_artifact(root / "outputs" / "pore_atom_indices.ndx")
    phase = _read_packmol_audit_phase(root)
    atoms = _read_structure_atoms(structure_path)
    pore_indices = _read_pore_indices(root / "outputs" / "pore_atom_indices.ndx")
    non_pore_atoms = [atom for atom in atoms if atom["serial"] not in pore_indices]
    interface_fraction = _packmol_interface_mixing_layer_fraction(root)
    atom_metrics = _intrusion_metrics(
        [atom["position_A"] for atom in non_pore_atoms],
        phase,
        interface_fraction=interface_fraction,
    )
    com_metrics = _intrusion_metrics(
        _molecule_centers(non_pore_atoms),
        phase,
        interface_fraction=interface_fraction,
    )
    payload = {
        "status": "audited_packmol_output",
        "structure": str(structure_path.resolve()),
        "atom_count": len(atoms),
        "non_pore_atom_count": len(non_pore_atoms),
        "atom_intrusion_count": atom_metrics["intrusion_count"],
        "atom_intrusion_fraction": atom_metrics["intrusion_fraction"],
        "com_count": com_metrics["sample_count"],
        "com_intrusion_count": com_metrics["intrusion_count"],
        "com_intrusion_fraction": com_metrics["intrusion_fraction"],
        "maximum_intrusion_depth_A": max(
            atom_metrics["maximum_intrusion_depth_A"],
            com_metrics["maximum_intrusion_depth_A"],
        ),
        "interface_mixing_layer_fraction": interface_fraction,
        "interface_mixing_layer_A": atom_metrics["interface_mixing_layer_A"],
        "deep_intrusion_atom_count": atom_metrics["deep_intrusion_count"],
        "deep_intrusion_com_count": com_metrics["deep_intrusion_count"],
        "deep_intrusion_cluster_count": atom_metrics["deep_intrusion_cluster_count"],
    }
    payload["passed"] = bool(
        payload["deep_intrusion_atom_count"] == 0
        and payload["deep_intrusion_com_count"] == 0
        and payload["deep_intrusion_cluster_count"] == 0
    )
    output = root / "analysis" / "packmol-output-audit.json"
    _write_json(output, payload)
    _write_packmol_output_audit_markdown(root / "analysis" / "packmol-output-audit.md", payload)
    return output


@dataclass(frozen=True)
class _PackmolAuditPhase:
    pore_mask: np.ndarray
    spacing_A: float
    origin_A: np.ndarray
    target_box_A: np.ndarray
    pore_depth_A: np.ndarray


def _read_packmol_audit_phase(root: Path) -> _PackmolAuditPhase:
    final_phase = root / "qa_export" / "final_phase.h5"
    if final_phase.is_file():
        with h5py.File(final_phase, "r") as handle:
            mask = np.asarray(handle["pore_mask"], dtype=bool)
            spacing = float(handle.attrs["spacing_A"])
            origin = np.asarray(handle.attrs.get("origin_A", [0.0, 0.0, 0.0]), dtype=float)
            target_box = np.asarray(handle.attrs["target_box_A"], dtype=float)
    else:
        geometry_h5 = _require_run_artifact(root / "outputs" / "pore_geometry.h5")
        with h5py.File(geometry_h5, "r") as handle:
            mask = np.asarray(handle["final_phase/pore_mask"], dtype=bool)
            target_box = np.asarray(handle.attrs["target_box_A"], dtype=float)
            origin = np.zeros(3, dtype=float)
            spacing = float(target_box[0] / mask.shape[2])
    if mask.ndim != 3:
        raise ValueError("final_phase pore_mask must be three-dimensional")
    depth = ndimage.distance_transform_edt(mask, sampling=spacing)
    return _PackmolAuditPhase(
        pore_mask=mask,
        spacing_A=spacing,
        origin_A=origin,
        target_box_A=target_box,
        pore_depth_A=depth,
    )


def _read_pore_indices(path: Path) -> set[int]:
    indices: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("["):
            continue
        for token in line.split():
            try:
                indices.add(int(token))
            except ValueError:
                continue
    return indices


def _read_structure_atoms(path: Path) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atoms.append(_parse_structure_atom_line(line))
    return atoms


def _parse_structure_atom_line(line: str) -> dict[str, Any]:
    try:
        serial = int(line[6:11])
        position = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=float,
        )
        residue = (line[21:22].strip(), line[17:20].strip(), line[22:26].strip())
    except ValueError:
        parts = line.split()
        if len(parts) < 10:
            raise ValueError(f"could not parse atom coordinates from structure line: {line}")
        serial = int(parts[1])
        position = np.array([float(parts[7]), float(parts[8]), float(parts[9])], dtype=float)
        residue = (parts[5], parts[4], parts[6])
    return {"serial": serial, "position_A": position, "residue_key": residue}


def _molecule_centers(atoms: list[dict[str, Any]]) -> list[np.ndarray]:
    groups: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for atom in atoms:
        groups.setdefault(atom["residue_key"], []).append(atom["position_A"])
    return [np.mean(np.vstack(positions), axis=0) for positions in groups.values()]


def _intrusion_metrics(
    points_A: list[np.ndarray],
    phase: _PackmolAuditPhase,
    *,
    interface_fraction: float,
) -> dict[str, Any]:
    sample_count = len(points_A)
    mixing_layer_A = float(interface_fraction) * float(phase.spacing_A)
    if sample_count == 0:
        return {
            "sample_count": 0,
            "intrusion_count": 0,
            "intrusion_fraction": 0.0,
            "maximum_intrusion_depth_A": 0.0,
            "interface_mixing_layer_A": mixing_layer_A,
            "deep_intrusion_count": 0,
            "deep_intrusion_cluster_count": 0,
        }

    intrusion_count = 0
    deep_count = 0
    max_depth = 0.0
    deep_mask = np.zeros_like(phase.pore_mask, dtype=bool)
    for point in points_A:
        index = _phase_index_for_point(point, phase)
        if index is None:
            continue
        z_index, y_index, x_index = index
        if not phase.pore_mask[z_index, y_index, x_index]:
            continue
        intrusion_count += 1
        depth = float(phase.pore_depth_A[z_index, y_index, x_index])
        max_depth = max(max_depth, depth)
        if depth > mixing_layer_A + 1.0e-12:
            deep_count += 1
            deep_mask[z_index, y_index, x_index] = True
    return {
        "sample_count": sample_count,
        "intrusion_count": intrusion_count,
        "intrusion_fraction": intrusion_count / sample_count,
        "maximum_intrusion_depth_A": max_depth,
        "interface_mixing_layer_A": mixing_layer_A,
        "deep_intrusion_count": deep_count,
        "deep_intrusion_cluster_count": _connected_cluster_count(deep_mask),
    }


def _phase_index_for_point(
    point_A: np.ndarray,
    phase: _PackmolAuditPhase,
) -> tuple[int, int, int] | None:
    point = np.asarray(point_A, dtype=float).copy()
    lower = phase.origin_A
    upper = phase.origin_A + phase.target_box_A
    point[0] = lower[0] + np.mod(point[0] - lower[0], phase.target_box_A[0])
    point[1] = lower[1] + np.mod(point[1] - lower[1], phase.target_box_A[1])
    if point[2] < lower[2] or point[2] >= upper[2]:
        return None
    indices_xyz = np.floor((point - lower) / phase.spacing_A).astype(int)
    nx = phase.pore_mask.shape[2]
    ny = phase.pore_mask.shape[1]
    nz = phase.pore_mask.shape[0]
    if indices_xyz[0] < 0 or indices_xyz[0] >= nx or indices_xyz[1] < 0 or indices_xyz[1] >= ny:
        return None
    if indices_xyz[2] < 0 or indices_xyz[2] >= nz:
        return None
    return int(indices_xyz[2]), int(indices_xyz[1]), int(indices_xyz[0])


def _connected_cluster_count(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    _labels, count = ndimage.label(mask)
    return int(count)


def _packmol_interface_mixing_layer_fraction(root: Path) -> float:
    for path in (root / "inputs" / "normalized_config.yaml", root / "qa_export" / "normalized_config.yaml"):
        if not path.is_file():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            continue
        audit = data.get("audit", data.get("geometry_audit", {}))
        if isinstance(audit, dict) and "interface_mixing_layer_fraction" in audit:
            return float(audit["interface_mixing_layer_fraction"])
    return 0.0


def _write_packmol_output_audit_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Packmol output intrusion audit",
        "",
        f"Status: {'PASS' if payload['passed'] else 'FAIL'}",
        "",
        f"- Atom intrusion fraction: {payload['atom_intrusion_fraction']:.6g}",
        f"- COM intrusion fraction: {payload['com_intrusion_fraction']:.6g}",
        f"- Maximum intrusion depth (A): {payload['maximum_intrusion_depth_A']:.6g}",
        f"- Interface mixing layer fraction: {payload['interface_mixing_layer_fraction']:.6g}",
        f"- Deep intrusion atoms: {payload['deep_intrusion_atom_count']}",
        f"- Deep intrusion COMs: {payload['deep_intrusion_com_count']}",
        f"- Deep intrusion clusters: {payload['deep_intrusion_cluster_count']}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _execute_single_seed(
    config: GeneratorConfig,
    paths: TaskPaths,
    *,
    write_artifacts: bool,
    run_identifier: str | None = None,
    candidate_executor: Executor | None = None,
) -> RunResult:
    geometry_run = generate_geometry(config, paths, candidate_executor=candidate_executor)
    if not _accepted_for_packing(
        geometry_run.audit,
        allow_small_sample_warnings=not config.audit.enabled,
    ):
        result = RunResult(
            paths=paths,
            geometry_run=geometry_run,
            packing_result=None,
            status="completed_infeasible",
        )
        _write_reports(config, geometry_run, paths, result.status)
        return result

    if config.pore_material is None:
        result = RunResult(
            paths=paths,
            geometry_run=geometry_run,
            packing_result=None,
            status="completed_feasible",
        )
        _write_reports(config, geometry_run, paths, result.status)
        return result

    template = MoleculeTemplate.from_pdb(config.pore_material.pdb)
    packing_result = pack_molecules(
        template,
        geometry_run.built.geometry,
        _packing_config(config),
        np.random.default_rng(config.task.random_seed),
    )
    if write_artifacts:
        _write_all_outputs(
            config,
            geometry_run,
            packing_result,
            run_identifier=run_identifier or paths.root.name,
        )
    result = RunResult(
        paths=paths,
        geometry_run=geometry_run,
        packing_result=packing_result,
        status="completed_feasible",
    )
    _write_reports(config, geometry_run, paths, result.status)
    return result


def _write_all_outputs(
    config: GeneratorConfig,
    geometry_run: GeometryRun,
    packing_result: PackingResult,
    *,
    run_identifier: str | None = None,
) -> None:
    paths = geometry_run.paths
    outputs = paths.outputs
    qa = paths.qa_export
    identifier = run_identifier or paths.root.name
    qa_molecules = qa / "molecules"
    qa_molecules.mkdir(parents=True, exist_ok=True)

    source_hash = _file_sha256(config.pore_material.pdb)
    shutil.copy2(config.pore_material.pdb, outputs / "pore_material.pdb")
    source_dir = qa_molecules / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.pore_material.pdb, source_dir / config.pore_material.pdb.name)

    export_semiconductor_glb(
        geometry_run.phase_grid,
        outputs / "semiconductor_solid_target.glb",
        _qa_contract(
            config,
            geometry_run.phase_grid,
            run_identifier=identifier,
        ),
    )
    shutil.copy2(outputs / "semiconductor_solid_target.glb", qa / "semiconductor_solid_target.glb")
    geometry_run.phase_grid.write_hdf5(qa / "final_phase.h5")
    export_surface_ply(geometry_run.phase_grid, qa / "final_surface.ply")
    _write_pore_geometry_hdf5(config, geometry_run)
    _write_phase_mapping(outputs / "phase_mapping.json")

    packing_result.write_mmcif(outputs / "pore_material_high_precision.cif")
    packing_frame_result = _packing_frame_result(config, packing_result)
    packing_frame_result.write_mmcif(outputs / "pore_reference_coordinates.cif")
    packing_result.write_hdf5(qa_molecules / "placed_atoms.h5")
    packing_result.write_mmcif(qa_molecules / "placed_structure.cif")
    packing_result.write_metrics_json(outputs / "packing_metrics.json")
    _write_required_instances_csv(
        packing_result,
        outputs / "molecule_instances.csv",
        source_hash=source_hash,
    )
    shutil.copy2(outputs / "molecule_instances.csv", qa_molecules / "instances.csv")
    _write_packmol_handoff(
        config,
        outputs / "packmol_handoff.inp",
        outputs / "pore_material.pdb",
        packing_count=packing_result.count,
    )
    _write_pore_ndx(outputs / "pore_atom_indices.ndx", packing_result.atom_positions_A.shape[0])
    _write_compression_metadata(
        config,
        outputs / "compression_metadata.json",
        coordinate_hash=_array_hash(packing_frame_result.atom_positions_A),
    )
    _write_main_metrics(config, geometry_run, packing_result)
    _write_unit_metrics(geometry_run.built, qa / "main_unit_metrics.csv")
    _write_channel_curves(geometry_run.built, qa / "channel_curves.h5")
    write_qa_contract(
        _qa_contract(
            config,
            geometry_run.phase_grid,
            run_identifier=identifier,
        ),
        qa,
    )


def _packing_frame_result(
    config: GeneratorConfig,
    packing_result: PackingResult,
) -> PackingResult:
    offset = np.asarray(config.film.target_origin_in_packing_A, dtype=float)
    shifted_transforms = tuple(
        InstanceTransform(
            translation_A=transform.translation_A + offset,
            quaternion_xyzw=transform.quaternion_xyzw,
        )
        for transform in packing_result.instance_transforms
    )
    return PackingResult(
        count=packing_result.count,
        atom_positions_A=packing_result.atom_positions_A + offset[np.newaxis, :],
        instance_transforms=shifted_transforms,
        minimum_interatomic_distance_A=packing_result.minimum_interatomic_distance_A,
        actual_density_g_cm3=packing_result.actual_density_g_cm3,
        protrusion_metrics=packing_result.protrusion_metrics,
        status=packing_result.status,
        template=packing_result.template,
        target_box_A=_packing_box(config),
        pore_volume_A3=packing_result.pore_volume_A3,
    )


def _write_geometry_artifacts(
    config: GeneratorConfig,
    run: GeometryRun,
    *,
    runtime_profiler: RuntimeProfiler | None = None,
    run_started_at: float | None = None,
    candidate_search_wall_time_seconds: float | None = None,
    selected_replay_wall_time_seconds: float | None = None,
) -> GeometryRun:
    profiler = runtime_profiler or RuntimeProfiler()
    with activate_runtime_profiler(profiler), profile_stage("export"):
        run.paths.qa_export.mkdir(parents=True, exist_ok=True)
        for unit in run.built.units:
            _append_jsonl(run.paths.qa_export / "unit_geometry.jsonl", unit.to_record())
        run.phase_grid.write_hdf5(run.paths.qa_export / "final_phase.h5")
        _write_normalized_config(config, run.paths.qa_export / "normalized_config.yaml")
        _write_final_measurement_artifacts(run)
        export_surface_ply(run.phase_grid, run.paths.qa_export / "final_surface.ply")
        run.paths.outputs.mkdir(parents=True, exist_ok=True)
        export_semiconductor_glb(
            run.phase_grid,
            run.paths.outputs / "semiconductor_solid_target.glb",
            _qa_contract(config, run.phase_grid),
        )
        shutil.copy2(
            run.paths.outputs / "semiconductor_solid_target.glb",
            run.paths.qa_export / "semiconductor_solid_target.glb",
        )
        _write_pore_geometry_hdf5(config, run)
        _write_main_metrics(config, run, None)
        _write_unit_metrics(run.built, run.paths.qa_export / "main_unit_metrics.csv")
        _write_channel_curves(run.built, run.paths.qa_export / "channel_curves.h5")
        write_qa_contract(
            _qa_contract(config, run.phase_grid),
            run.paths.qa_export,
        )
        write_run_report(
            run.paths.reports / "geometry-report.md",
            config=config,
            status="accepted",
            warnings=list(run.audit.warnings),
            convergence={
                "audit_passed": run.audit.passed,
                "porosity": run.phase_grid.porosity,
                "target_porosity": config.pores.target_porosity,
            },
            output_paths=[run.paths.qa_export / "unit_geometry.jsonl"],
        )

    snapshot = profiler.snapshot()
    performance = _geometry_performance_record(
        snapshot,
        candidates=run.candidate_results,
        candidate_search_wall_time_seconds=candidate_search_wall_time_seconds,
        selected_replay_wall_time_seconds=selected_replay_wall_time_seconds,
        total_wall_time_seconds=(
            snapshot.wall_time_seconds
            if run_started_at is None
            else max(0.0, time.perf_counter() - run_started_at)
        ),
    )
    run = replace(run, performance=performance)
    _write_json(run.paths.qa_export / "performance.json", performance)
    write_qa_checksums(run.paths.qa_export)
    if config.output.write_plots:
        write_visual_report(
            run.paths.outputs / "visual-report" / "index.html",
            config=config,
            built=run.built,
            grid=run.phase_grid,
            audit=run.audit,
            candidates=run.candidate_results,
            selected_sequence_index=run.selected_candidate.sequence_index,
            performance=performance,
        )
    return run


def _write_final_measurement_artifacts(run: GeometryRun) -> None:
    measurements = run.audit.formal_measurements
    if measurements is None:
        return
    qa = run.paths.qa_export
    _write_json(
        qa / "final_measurements.json",
        _json_ready(asdict(measurements)),
    )
    with h5py.File(qa / "final_centerlines.h5", "w") as handle:
        handle.attrs["schema_version"] = 3
        handle.attrs["length_unit"] = "angstrom"
        handle.attrs["target_box_A"] = run.phase_grid.target_box_A
        handle.attrs["periodic_axes"] = "x,y"
        group = handle.create_group("centerlines")
        for track in measurements.centerlines:
            track_group = group.create_group(str(track.track_id))
            track_group.create_dataset("slice_indices", data=track.slice_indices)
            track_group.create_dataset("points_wrapped_A", data=track.points_wrapped_A)
            track_group.create_dataset(
                "points_unwrapped_A",
                data=track.points_unwrapped_A,
            )
            track_group.create_dataset(
                "wall_distances_A",
                data=track.wall_distances_A,
            )
            track_group.attrs["touches_z_lower"] = bool(track.touches_z_lower)
            track_group.attrs["touches_z_upper"] = bool(track.touches_z_upper)
            track_group.attrs["is_through"] = bool(track.is_through)
            track_group.attrs["has_branch_neighborhood"] = bool(
                track.has_branch_neighborhood
            )

    fields = [
        "track_id",
        "arc_position_A",
        "center_x_A",
        "center_y_A",
        "center_z_A",
        "tangent_x",
        "tangent_y",
        "tangent_z",
        "valid",
        "invalid_reason",
        "area_A2",
        "equivalent_diameter_A",
        "curvature_fluctuation",
    ]
    with (qa / "final_cross_sections.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for section in measurements.cross_sections:
            writer.writerow(
                {
                    "track_id": section.track_id,
                    "arc_position_A": section.arc_position_A,
                    "center_x_A": section.center_A[0],
                    "center_y_A": section.center_A[1],
                    "center_z_A": section.center_A[2],
                    "tangent_x": section.tangent[0],
                    "tangent_y": section.tangent[1],
                    "tangent_z": section.tangent[2],
                    "valid": section.valid,
                    "invalid_reason": section.invalid_reason or "",
                    "area_A2": section.area_A2,
                    "equivalent_diameter_A": section.equivalent_diameter_A,
                    "curvature_fluctuation": section.curvature_fluctuation,
                }
            )


def _write_reports(
    config: GeneratorConfig,
    geometry_run: GeometryRun,
    paths: TaskPaths,
    status: str,
) -> None:
    output_paths = [
        paths.outputs / "semiconductor_solid_target.glb",
        paths.outputs / "pore_geometry.h5",
        paths.outputs / "pore_reference_coordinates.cif",
        paths.qa_export / "contract.json",
    ]
    write_run_report(
        paths.reports / "generation-report.md",
        config=config,
        status=status,
        warnings=list(geometry_run.audit.warnings),
        convergence={
            "audit_passed": geometry_run.audit.passed,
            "porosity": geometry_run.phase_grid.porosity,
            "minimum_cross_section_fraction": geometry_run.audit.minimum_cross_section_fraction,
        },
        output_paths=output_paths,
    )
    write_run_report(
        paths.reports / "final-summary.md",
        config=config,
        status=status,
        warnings=list(geometry_run.audit.warnings),
        convergence={
            "audit_passed": geometry_run.audit.passed,
            "porosity": geometry_run.phase_grid.porosity,
            "target_porosity": config.pores.target_porosity,
            "minimum_cross_section_fraction": geometry_run.audit.minimum_cross_section_fraction,
        },
        output_paths=output_paths,
    )


def _write_pore_geometry_hdf5(config: GeneratorConfig, run: GeometryRun) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    path = run.paths.outputs / "pore_geometry.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = _GEOMETRY_SCHEMA_VERSION
        handle.attrs["target_box_A"] = _target_box(config)
        handle.attrs["packing_box_A"] = _packing_box(config)
        handle.attrs["target_origin_A"] = config.film.target_origin_in_packing_A
        handle.attrs["final_phase_reference"] = "../qa_export/final_phase.h5"
        records = [
            json.dumps(_json_ready(unit.to_record()), sort_keys=True)
            for unit in run.built.units
        ]
        handle.create_dataset("unit_records", data=np.asarray(records, dtype=object), dtype=string_dtype)
        final_phase = handle.create_group("final_phase")
        final_phase.attrs["authoritative_reference"] = "../qa_export/final_phase.h5"
        final_phase.create_dataset(
            "pore_mask",
            data=run.phase_grid.pore_mask.astype(np.uint8),
            compression="gzip",
            shuffle=True,
        )


def _write_required_instances_csv(
    packing_result: PackingResult,
    path: Path,
    *,
    source_hash: str,
) -> None:
    atoms_per_instance = len(packing_result.template.elements)
    residue = packing_result.template.residue_names[0] if atoms_per_instance else "UNK"
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "instance_id",
                "source_hash",
                "residue_name",
                "translation_x_A",
                "translation_y_A",
                "translation_z_A",
                "quaternion_w",
                "quaternion_x",
                "quaternion_y",
                "quaternion_z",
                "periodic_image_x",
                "periodic_image_y",
            ]
        )
        for index, transform in enumerate(packing_result.instance_transforms):
            qx, qy, qz, qw = transform.quaternion_xyzw
            writer.writerow(
                [
                    index,
                    source_hash,
                    residue,
                    f"{transform.translation_A[0]:.10f}",
                    f"{transform.translation_A[1]:.10f}",
                    f"{transform.translation_A[2]:.10f}",
                    f"{qw:.12f}",
                    f"{qx:.12f}",
                    f"{qy:.12f}",
                    f"{qz:.12f}",
                    0,
                    0,
                ]
            )


def _write_packmol_handoff(
    config: GeneratorConfig,
    path: Path,
    pore_material: Path,
    *,
    packing_count: int,
) -> None:
    lines = [
        "# Porous-film Packmol handoff",
        f"# target_box_A: {_vector_list(_target_box(config))}",
        f"# packing_box_A: {_vector_list(_packing_box(config))}",
        "tolerance 2.0",
        "filetype pdb",
        "output packed_pore_material.pdb",
        f"structure {pore_material.resolve()}",
        f"  number {int(packing_count)}",
        (
            "  inside box "
            f"0.0 0.0 0.0 {_packing_box(config)[0]:.10f} "
            f"{_packing_box(config)[1]:.10f} {_packing_box(config)[2]:.10f}"
        ),
        "end structure",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pore_ndx(path: Path, atom_count: int) -> None:
    indices = " ".join(str(index) for index in range(1, atom_count + 1))
    path.write_text(f"[ PORE ]\n{indices}\n", encoding="utf-8")


def _write_phase_mapping(path: Path) -> None:
    _write_json(path, {"MATRIX": _MATRIX_PHASE_ID, "PORE": _PORE_PHASE_ID})


def _write_compression_metadata(
    config: GeneratorConfig,
    path: Path,
    *,
    coordinate_hash: str,
) -> None:
    _write_json(
        path,
        {
            "target_box_A": _vector_list(_target_box(config)),
            "packing_box_A": _vector_list(_packing_box(config)),
            "target_origin_A": _vector_list(config.film.target_origin_in_packing_A),
            "absolute_lock_mode": "target_origin_in_packing_A",
            "pore_coordinate_hash": coordinate_hash,
        },
    )


def _write_main_metrics(
    config: GeneratorConfig,
    run: GeometryRun,
    packing_result: PackingResult | None,
) -> None:
    metrics = {
        "target_porosity": config.pores.target_porosity,
        "realized_porosity": run.phase_grid.porosity,
        "porosity_absolute_error": abs(run.phase_grid.porosity - config.pores.target_porosity),
        "audit_passed": run.audit.passed,
        "packing_count": None if packing_result is None else packing_result.count,
        "actual_density_g_cm3": None
        if packing_result is None
        else packing_result.actual_density_g_cm3,
        "minimum_interatomic_distance_A": None
        if packing_result is None
        else packing_result.minimum_interatomic_distance_A,
        "warnings": list(run.audit.warnings),
        "formal_measurements_available": run.audit.formal_measurements is not None,
    }
    _write_json(run.paths.qa_export / "main_metrics.json", metrics)
    _write_json(run.paths.analysis / "target_vs_actual_metrics.json", metrics)


def _write_unit_metrics(built: BuiltGeometry, path: Path) -> None:
    fieldnames = [
        "unit_id",
        "kind",
        "shape_model",
        "shape_seed",
        "anchor_x_A",
        "anchor_y_A",
        "anchor_z_A",
        "latent_target_volume_A3",
        "roughness",
        "eta",
        "tau",
        "lobe_count",
        "envelope_fill_fraction",
        "centroid_offset_A",
        "lobes_connected",
        "radius_cv",
        "minimum_to_maximum_radius_ratio",
        "bend_count",
        "nonplanarity",
        "minimum_self_clearance_A",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for unit in built.units:
            writer.writerow(_unit_shape_metrics(unit))


def _unit_shape_metrics(unit: CompactUnit | ChannelUnit) -> dict[str, Any]:
    anchor = unit.anchor_A
    base: dict[str, Any] = {
        "unit_id": unit.unit_id,
        "kind": "compact" if isinstance(unit, CompactUnit) else "channel",
        "shape_model": unit.shape_model or "",
        "shape_seed": "" if unit.shape_seed is None else int(unit.shape_seed),
        "anchor_x_A": f"{anchor[0]:.10f}",
        "anchor_y_A": f"{anchor[1]:.10f}",
        "anchor_z_A": f"{anchor[2]:.10f}",
        "latent_target_volume_A3": ""
        if unit.latent_target_volume_A3 is None
        else f"{unit.latent_target_volume_A3:.10f}",
        "roughness": f"{unit.roughness:.10f}",
        "eta": "",
        "tau": "",
        "lobe_count": "",
        "envelope_fill_fraction": "",
        "centroid_offset_A": "",
        "lobes_connected": "",
        "radius_cv": "",
        "minimum_to_maximum_radius_ratio": "",
        "bend_count": "",
        "nonplanarity": "",
        "minimum_self_clearance_A": "",
    }
    if isinstance(unit, CompactUnit):
        base["eta"] = f"{unit.radii_A[2] / unit.radii_A[0]:.10f}"
        if unit.is_multilobe:
            base.update(
                {
                    "lobe_count": int(unit.lobe_centers_local_A.shape[0]),
                    "envelope_fill_fraction": f"{unit.envelope_fill_fraction:.10f}",
                    "centroid_offset_A": f"{unit.centroid_offset_A:.10f}",
                    "lobes_connected": bool(unit.lobes_connected),
                }
            )
        return base

    base["eta"] = f"{unit.eta:.10f}"
    base["tau"] = f"{unit.tortuosity:.10f}"
    if unit.is_variable_radius:
        dense_radii = np.asarray(
            PchipInterpolator(unit.radius_profile_s, unit.radius_profile_A)(
                np.linspace(0.0, 1.0, 1025)
            ),
            dtype=float,
        )
        base.update(
            {
                "radius_cv": f"{np.std(dense_radii) / np.mean(dense_radii):.10f}",
                "minimum_to_maximum_radius_ratio": (
                    f"{np.min(dense_radii) / np.max(dense_radii):.10f}"
                ),
                "bend_count": int(unit.bend_count or 0),
                "nonplanarity": f"{float(unit.nonplanarity or 0.0):.10f}",
                "minimum_self_clearance_A": (
                    f"{float(unit.minimum_self_clearance_A or 0.0):.10f}"
                ),
            }
        )
    return base


def _shape_complexity_summary(built: BuiltGeometry) -> dict[str, Any]:
    compact_metrics = [
        _unit_shape_metrics(unit)
        for unit in built.units
        if isinstance(unit, CompactUnit) and unit.is_multilobe
    ]
    channel_metrics = [
        _unit_shape_metrics(unit)
        for unit in built.units
        if isinstance(unit, ChannelUnit) and unit.is_variable_radius
    ]
    return {
        "compact": {
            "count": len(compact_metrics),
            "shape_model_counts": _value_counts(
                metric["shape_model"] for metric in compact_metrics
            ),
            "lobe_count": _numeric_summary(
                metric["lobe_count"] for metric in compact_metrics
            ),
            "envelope_fill_fraction": _numeric_summary(
                metric["envelope_fill_fraction"] for metric in compact_metrics
            ),
            "centroid_offset_A": _numeric_summary(
                metric["centroid_offset_A"] for metric in compact_metrics
            ),
            "connected_fraction": (
                float(
                    np.mean(
                        [bool(metric["lobes_connected"]) for metric in compact_metrics]
                    )
                )
                if compact_metrics
                else None
            ),
        },
        "channel": {
            "count": len(channel_metrics),
            "shape_model_counts": _value_counts(
                metric["shape_model"] for metric in channel_metrics
            ),
            "radius_cv": _numeric_summary(
                metric["radius_cv"] for metric in channel_metrics
            ),
            "minimum_to_maximum_radius_ratio": _numeric_summary(
                metric["minimum_to_maximum_radius_ratio"]
                for metric in channel_metrics
            ),
            "bend_count": _numeric_summary(
                metric["bend_count"] for metric in channel_metrics
            ),
            "nonplanarity": _numeric_summary(
                metric["nonplanarity"] for metric in channel_metrics
            ),
            "minimum_self_clearance_A": _numeric_summary(
                metric["minimum_self_clearance_A"] for metric in channel_metrics
            ),
        },
    }


def _numeric_summary(values: Any) -> dict[str, float | int | None]:
    array = np.asarray([float(value) for value in values], dtype=float)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)) if array.size else None,
        "mean": float(np.mean(array)) if array.size else None,
        "maximum": float(np.max(array)) if array.size else None,
    }


def _value_counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _write_channel_curves(built: BuiltGeometry, path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = _GEOMETRY_SCHEMA_VERSION
        for unit in built.units:
            if not isinstance(unit, ChannelUnit):
                continue
            group = handle.create_group(unit.unit_id)
            group.attrs["shape_model"] = unit.shape_model or "constant-radius-polyline-v1"
            if unit.shape_seed is not None:
                group.attrs["shape_seed"] = int(unit.shape_seed)
            group.attrs["equivalent_radius_A"] = float(unit.cross_radius_A)
            group.create_dataset("centerline_A", data=unit.centerline_samples_A)
            sample_radii = np.concatenate(
                (unit.segment_start_radii_A, unit.segment_end_radii_A[-1:])
            )
            group.create_dataset("radius_A", data=sample_radii)


def _write_parallel_progress(paths: TaskPaths, result: object) -> None:
    sequence_index = getattr(result, "sequence_index", None)
    if sequence_index is None:
        return
    _write_json(
        paths.work / "parallel" / "progress" / f"{int(sequence_index)}.json",
        _parallel_progress_record(result),
    )


def _parallel_progress_record(result: object) -> dict[str, Any]:
    failure = None
    failure_type = getattr(result, "failure_type", None)
    failure_message = getattr(result, "failure_message", None)
    failure_reason = getattr(result, "failure_reason", None)
    if failure_type is not None or failure_message is not None:
        failure = {"type": failure_type, "message": failure_message}
    elif failure_reason:
        failure = {"type": "failure", "message": failure_reason}
    return {
        "identity": _identity_record(getattr(result, "identity", None)),
        "status": _parallel_result_status(result),
        "wall_time_seconds": getattr(result, "wall_time_seconds", None),
        "failure": failure,
    }


def _parallel_result_status(result: object) -> str:
    status = getattr(result, "status", None)
    if status is not None:
        return str(status)
    succeeded = getattr(result, "succeeded", None)
    return "completed" if succeeded is True else "failed"


def _results_with_cancelled_tasks(
    tasks: Sequence[object],
    completed_results: Sequence[object],
    message: str,
) -> list[object]:
    completed_sequences = {
        int(sequence)
        for result in completed_results
        if (sequence := getattr(result, "sequence_index", None)) is not None
    }
    records: list[object] = list(completed_results)
    for task in sorted(tasks, key=lambda item: int(item.sequence_index)):
        sequence_index = int(task.sequence_index)
        if sequence_index in completed_sequences:
            continue
        records.append(
            {
                "identity": _identity_record(getattr(task, "identity", None)),
                "status": "cancelled",
                "wall_time_seconds": None,
                "failure": {"type": "cancelled", "message": message},
            }
        )
    return sorted(records, key=_parallel_summary_sequence)


def _cancellation_record(
    exc: ParallelCancelled,
    tasks: Sequence[object],
    completed_results: Sequence[object],
    *,
    diagnostic_results: Sequence[object] = (),
) -> dict[str, Any]:
    completed_sequences = {
        int(sequence)
        for result in completed_results
        if (sequence := getattr(result, "sequence_index", None)) is not None
    }
    missing = [
        _identity_record(getattr(task, "identity", None))
        for task in sorted(tasks, key=lambda item: int(item.sequence_index))
        if int(task.sequence_index) not in completed_sequences
    ]
    return {
        "type": "KeyboardInterrupt",
        "message": "parallel execution cancelled",
        "started_task_count": exc.started_task_count,
        "missing_identities": missing,
        "completed_results": [_parallel_progress_record(result) for result in diagnostic_results],
    }


def _pool_failure_record(exc: ParallelPoolError) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "started_task_count": exc.started_task_count,
        "completed_results": [
            _parallel_progress_record(result) for result in exc.completed_results
        ],
    }


def _parallel_summary_sequence(record: object) -> int:
    if isinstance(record, dict):
        identity = record.get("identity", {})
        if isinstance(identity, dict):
            return int(identity.get("sequence_index", 0))
    return int(getattr(record, "sequence_index", 0))


def _identity_record(identity: object) -> dict[str, Any]:
    if identity is None:
        return {}
    if isinstance(identity, dict):
        return dict(identity)
    return asdict(identity)


def _run_seed_panel(
    config: GeneratorConfig,
    paths: TaskPaths,
    *,
    config_path: Path | None = None,
    run_identifier: str | None = None,
) -> RunResult:
    prepare_task_inputs(config, paths, config_path=config_path)
    identifier = run_identifier or paths.root.name
    plan = _seed_panel_execution_plan(config)
    write_parallel_plan(paths, plan)
    started = datetime.now(_SHANGHAI)
    tasks = build_seed_tasks(config, paths, plan, identifier)
    parent_results: dict[int, RunResult] = {}
    serial_fallback_reason: str | None = None

    if plan.effective_strategy == "seeds":
        pool_entered = False
        try:
            with spawn_pool(plan.worker_count, config.parallel.worker_threads) as executor:
                pool_entered = True
                compact_results = run_spawn_tasks(
                    tasks,
                    execute_seed_task,
                    worker_count=plan.worker_count,
                    worker_threads=config.parallel.worker_threads,
                    executor=executor,
                    result_callback=lambda result: _write_parallel_progress(paths, result),
                )
        except ParallelCancelled as exc:
            finished = datetime.now(_SHANGHAI)
            records = _results_with_cancelled_tasks(
                tasks,
                exc.completed_results,
                "cancelled by KeyboardInterrupt",
            )
            write_parallel_summary(
                paths,
                plan,
                records,
                started=started,
                finished=finished,
                cancellation=_cancellation_record(exc, tasks, exc.completed_results),
                pool_failure=None,
            )
            raise
        except ParallelPoolError as exc:
            finished = datetime.now(_SHANGHAI)
            records = _results_with_cancelled_tasks(
                tasks,
                exc.completed_results,
                "missing after process pool failure",
            )
            write_parallel_summary(
                paths,
                plan,
                records,
                started=started,
                finished=finished,
                cancellation=None,
                pool_failure=exc,
            )
            raise
        except Exception as exc:
            if pool_entered:
                raise
            serial_fallback_reason = f"{type(exc).__name__}: {exc}"
            compact_results = run_spawn_tasks(
                tasks,
                execute_seed_task,
                worker_count=1,
                worker_threads=config.parallel.worker_threads,
                result_callback=lambda result: _write_parallel_progress(paths, result),
            )
    elif plan.effective_strategy == "candidates":
        compact_results = []
        pool_entered = False
        try:
            with spawn_pool(plan.worker_count, config.parallel.worker_threads) as executor:
                pool_entered = True
                for task in tasks:
                    compact, run_result = _execute_seed_body(task, candidate_executor=executor)
                    compact_results.append(compact)
                    _write_parallel_progress(paths, compact)
                    if run_result is not None:
                        parent_results[task.sequence_index] = run_result
        except ParallelCancelled as exc:
            finished = datetime.now(_SHANGHAI)
            records = _results_with_cancelled_tasks(
                tasks,
                compact_results,
                "cancelled by KeyboardInterrupt",
            )
            write_parallel_summary(
                paths,
                plan,
                records,
                started=started,
                finished=finished,
                cancellation=_cancellation_record(
                    exc,
                    tasks,
                    compact_results,
                    diagnostic_results=exc.completed_results,
                ),
                pool_failure=None,
            )
            raise
        except ParallelPoolError as exc:
            finished = datetime.now(_SHANGHAI)
            records = _results_with_cancelled_tasks(
                tasks,
                compact_results,
                "missing after process pool failure",
            )
            write_parallel_summary(
                paths,
                plan,
                records,
                started=started,
                finished=finished,
                cancellation=None,
                pool_failure=_pool_failure_record(exc),
            )
            raise
        except Exception as exc:
            if pool_entered:
                raise
            serial_fallback_reason = f"{type(exc).__name__}: {exc}"
            for task in tasks:
                compact, run_result = _execute_seed_body(task, candidate_executor=None)
                compact_results.append(compact)
                _write_parallel_progress(paths, compact)
                if run_result is not None:
                    parent_results[task.sequence_index] = run_result
    else:
        compact_results = []
        for task in tasks:
            compact, run_result = _execute_seed_body(task, candidate_executor=None)
            compact_results.append(compact)
            _write_parallel_progress(paths, compact)
            if run_result is not None:
                parent_results[task.sequence_index] = run_result

    results = sorted(compact_results, key=lambda result: result.sequence_index)
    write_parallel_summary(
        paths,
        plan,
        results,
        started=started,
        finished=datetime.now(_SHANGHAI),
        cancellation=None,
        pool_failure=None,
        serial_fallback_reason=serial_fallback_reason,
    )
    records = [_seed_aggregate_record(result) for result in results]
    aggregate = aggregate_seed_results(records)
    seed_dir = paths.analysis / "seed_panel"
    seed_dir.mkdir(parents=True, exist_ok=True)
    _write_json(seed_dir / "aggregate.json", aggregate)

    task_by_sequence = {task.sequence_index: task for task in tasks}
    representative_result = _representative_seed_result(results, task_by_sequence)
    representative_task = task_by_sequence[representative_result.sequence_index]

    if representative_result.status == "failed":
        _write_optimizer_exchange_from_payload(
            representative_result.optimizer_payload,
            output_dir=paths.outputs,
            seed_panel_summary=aggregate,
        )
        raise RuntimeError(
            representative_result.failure_reason or "representative seed failed"
        )

    representative_paths = task_paths_from_existing_root(Path(representative_result.artifact_root))
    if plan.effective_strategy == "seeds":
        representative = _load_representative_from_pickle(
            representative_result,
            representative_task,
            paths,
            config,
        )
    else:
        representative = parent_results.get(representative_result.sequence_index)
        if representative is None:
            raise RuntimeError("representative seed did not return an in-memory result")
        representative = _rebind_run_result(representative, paths, config)

    _publish_representative(representative_paths, paths, config)
    _write_normalized_config(config, paths.inputs / "normalized_config.yaml")
    _write_normalized_config(config, paths.qa_export / "normalized_config.yaml")
    write_qa_checksums(paths.qa_export)
    _write_optimizer_exchange_from_payload(
        representative_result.optimizer_payload,
        output_dir=paths.outputs,
        seed_panel_summary=aggregate,
    )
    return representative


def _seed_panel_execution_plan(config: GeneratorConfig) -> ParallelExecutionPlan:
    seed_count = len(config.optimization.seed_panel or (config.task.random_seed,))
    candidate_count = config.audit.maximum_rounds * config.audit.candidate_count_per_round
    return build_execution_plan(
        config,
        command="generate",
        seed_task_count=seed_count,
        candidate_task_count=candidate_count,
        resources=discover_resources(
            audit_memory_cap_bytes=config.audit.available_memory_cap_bytes
        ),
        estimated_worker_memory_bytes=estimate_worker_memory_bytes(config),
    )


def _representative_seed_result(
    results: Sequence[SeedTaskResult],
    tasks: dict[int, SeedTask],
) -> SeedTaskResult:
    for result in results:
        if tasks[result.sequence_index].representative:
            return result
    raise RuntimeError("seed panel did not include a representative task")


def _seed_aggregate_record(result: SeedTaskResult) -> dict[str, Any]:
    return {
        "panel_index": result.identity.panel_index,
        "seed": result.identity.seed,
        "sequence_index": result.identity.sequence_index,
        "status": result.status,
        "feasible": result.feasible,
        "objective": result.objective,
        "failure_reason": result.failure_reason,
    }


def _load_representative_from_pickle(
    seed_result: SeedTaskResult,
    task: SeedTask,
    top_paths: TaskPaths,
    top_config: GeneratorConfig,
) -> RunResult:
    if seed_result.representative_pickle is None:
        raise RuntimeError("representative pickle was not returned by seed worker")
    seed_root = Path(task.task_root).resolve()
    expected_work = (seed_root / "work").resolve()
    pickle_path = Path(seed_result.representative_pickle)
    resolved = pickle_path.resolve()
    validated = False
    try:
        if (
            not resolved.is_relative_to(seed_root)
            or resolved.parent != expected_work
            or resolved.name != "representative-run.pkl"
            or not resolved.is_file()
        ):
            raise RuntimeError("representative pickle path failed validation")
        validated = True
        with resolved.open("rb") as handle:
            result = pickle.load(handle)
        if not isinstance(result, RunResult):
            raise TypeError("representative pickle did not contain a RunResult")
        return _rebind_run_result(result, top_paths, top_config)
    finally:
        if validated:
            try:
                resolved.unlink()
            except FileNotFoundError:
                pass


def _rebind_run_result(
    result: RunResult,
    top_paths: TaskPaths,
    top_config: GeneratorConfig,
) -> RunResult:
    geometry_run = replace(
        result.geometry_run,
        paths=top_paths,
        config=top_config,
    )
    return replace(result, paths=top_paths, geometry_run=geometry_run)


def _publish_representative(
    seed_paths: TaskPaths,
    top_paths: TaskPaths,
    top_config: GeneratorConfig,
) -> None:
    del top_config
    _copy_tree_files(seed_paths.outputs, top_paths.outputs)
    representative_metrics = seed_paths.analysis / "target_vs_actual_metrics.json"
    if representative_metrics.is_file():
        top_paths.analysis.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            representative_metrics,
            top_paths.analysis / "target_vs_actual_metrics.json",
        )
    _copy_tree_files(
        seed_paths.qa_export,
        top_paths.qa_export,
        skip_relative_paths={Path("normalized_config.yaml")},
    )
    _copy_tree_files(
        seed_paths.reports,
        top_paths.reports,
        skip_relative_paths={Path("parallel-summary.md")},
    )


def _copy_tree_files(
    source: Path,
    destination: Path,
    *,
    skip_relative_paths: set[Path] | None = None,
) -> None:
    if not source.is_dir():
        return
    skips = skip_relative_paths or set()
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if relative in skips:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _seed_panel_task_paths(seed_root: Path) -> tuple[TaskPaths, Path]:
    """Create a complete seed task layout while preserving root optimizer JSON."""
    return create_task_paths_at_root(seed_root), seed_root


def optimizer_payload_for_result(result: RunResult) -> OptimizerPayload:
    """Return the compact optimizer exchange payload for a completed single run."""
    feasible = result.status == "completed_feasible"
    return {
        "requested": _requested_parameters(result.geometry_run.config),
        "realized": _realized_parameters(result.geometry_run, result.packing_result),
        "feasible": feasible,
        "constraints": _constraint_status(
            result.geometry_run.audit,
            packing_result=result.packing_result,
        ),
        "calculation_status": {
            "status": result.status,
            "warnings": list(result.geometry_run.audit.warnings),
            "task_random_seed": result.geometry_run.config.task.random_seed,
        },
        "objectives": _objectives_from_result(result) if feasible else {},
        "uncertainty": {"geometry_replicate_variance": None},
    }


def optimizer_payload_for_failure(
    config: GeneratorConfig,
    exc: Exception,
) -> OptimizerPayload:
    """Return the compact optimizer exchange payload for a failed single run."""
    reason = (
        exc.reason
        if isinstance(exc, PackingError | GeometryAcceptanceError)
        else type(exc).__name__
    )
    return {
        "requested": _requested_parameters(config),
        "realized": {},
        "feasible": False,
        "constraints": {"audit_passed": False, "packing_completed": False},
        "calculation_status": {
            "status": "failed",
            "failure_reason": reason,
            "message": str(exc),
            "exception_type": type(exc).__name__,
        },
        "objectives": {},
        "uncertainty": {},
    }


def _write_optimizer_exchange_for_result(
    result: RunResult,
    *,
    output_dir: Path,
    seed_panel_summary: dict[str, Any] | None = None,
) -> None:
    payload = optimizer_payload_for_result(result)
    _write_optimizer_exchange_from_payload(
        payload,
        output_dir=output_dir,
        seed_panel_summary=seed_panel_summary,
    )


def _write_optimizer_exchange_from_payload(
    payload: OptimizerPayload,
    *,
    output_dir: Path,
    seed_panel_summary: dict[str, Any] | None = None,
) -> None:
    payload = dict(payload)
    if seed_panel_summary is not None:
        status_payload = payload["calculation_status"]
        if not isinstance(status_payload, dict):
            raise TypeError("calculation_status payload must be a dictionary")
        status_payload = dict(status_payload)
        status_payload["seed_panel"] = seed_panel_summary
        payload["calculation_status"] = status_payload
        uncertainty = payload["uncertainty"]
        if not isinstance(uncertainty, dict):
            raise TypeError("uncertainty payload must be a dictionary")
        uncertainty = dict(uncertainty)
        uncertainty["seed_panel"] = seed_panel_summary
        payload["uncertainty"] = uncertainty
    write_optimizer_exchange(output_dir, **payload)


def _config_for_seed(config: GeneratorConfig, seed: int) -> GeneratorConfig:
    return config.model_copy(
        update={
            "task": config.task.model_copy(update={"random_seed": int(seed)}),
            "optimization": config.optimization.model_copy(update={"seed_panel": (int(seed),)}),
        },
    )


def _objective_value(config: GeneratorConfig, result: RunResult) -> float | None:
    if result.status != "completed_feasible":
        return None
    return abs(result.geometry_run.phase_grid.porosity - config.pores.target_porosity)


def _failure_reason_from_result(result: RunResult) -> str:
    if result.status == "completed_infeasible":
        return _failure_reason_from_audit(result.geometry_run.audit)
    if result.status == "failed":
        return "failed"
    return ""


def _objectives_from_result(result: RunResult) -> dict[str, Any]:
    return _objectives(result.geometry_run.config, result.geometry_run)


def _record_failure(
    paths: TaskPaths,
    exc: Exception,
    config: GeneratorConfig,
    realized: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> None:
    payload = optimizer_payload_for_failure(config, exc)
    if realized is not None:
        payload["realized"] = realized
    status = payload["calculation_status"]
    if not isinstance(status, dict):
        raise TypeError("calculation_status payload must be a dictionary")
    paths.logs.mkdir(parents=True, exist_ok=True)
    exchange_dir = output_dir or paths.outputs
    exchange_dir.mkdir(parents=True, exist_ok=True)
    _write_json(paths.logs / "failure.json", status)
    write_optimizer_exchange(exchange_dir, **payload)


def _accepted_for_packing(
    audit: AuditResult,
    *,
    allow_small_sample_warnings: bool = False,
) -> bool:
    if audit.passed:
        return True
    if not allow_small_sample_warnings:
        return False
    fatal_phrases = (
        "does not percolate",
        "cross-section",
        "overlap fraction",
        "seed_count scalar error",
        "rdf weighted loss",
    )
    return not any(
        any(phrase in warning for phrase in fatal_phrases)
        for warning in audit.warnings
    )


def _packing_config(config: GeneratorConfig) -> PackingConfig:
    return PackingConfig(
        exact_count=config.pore_material.molecule_count,
        target_density_g_cm3=config.pore_material.target_density_g_cm3,
        minimum_distance_A=2.0,
        wall_clearance_A=0.0,
        max_attempts=100_000,
    )


def _requested_parameters(config: GeneratorConfig) -> dict[str, Any]:
    payload = config.model_dump(mode="json", exclude_none=True)
    payload["compact"] = {
        **payload.get("compact", {}),
        "superellipsoid_exponent": _COMPACT_SUPERELLIPSOID_EXPONENT,
    }
    if config.pore_material is not None:
        packing_config = _packing_config(config)
        payload["packing"] = {
            "exact_count": packing_config.exact_count,
            "target_density_g_cm3": packing_config.target_density_g_cm3,
            "minimum_distance_A": packing_config.minimum_distance_A,
            "wall_clearance_A": packing_config.wall_clearance_A,
            "max_attempts": packing_config.max_attempts,
        }
    payload.update(
        {
            "target_porosity": config.pores.target_porosity,
            "seed_number_density_A3": config.pores.seed_number_density_A3,
            "target_box_A": _vector_list(_target_box(config)),
            "packing_box_A": _vector_list(_packing_box(config)),
            "seed_panel": list(config.optimization.seed_panel or (config.task.random_seed,)),
        }
    )
    return payload


def _realized_parameters(
    run: GeometryRun,
    packing_result: PackingResult | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "porosity": run.phase_grid.porosity,
        "seed_count": len(run.built.units),
        "unit_counts": _unit_counts(run.built),
        "target_box_A": _vector_list(_target_box(run.config)),
        "packing_box_A": _vector_list(_packing_box(run.config)),
        "target_origin_A": _vector_list(run.config.film.target_origin_in_packing_A),
        "audit_passed": run.audit.passed,
        "minimum_cross_section_fraction": run.audit.minimum_cross_section_fraction,
        "audit_results": _audit_summary(run.audit),
        "shape_complexity_summary": _shape_complexity_summary(run.built),
    }
    if packing_result is not None:
        payload.update(
            {
                "packed_molecule_count": packing_result.count,
                "actual_density_g_cm3": packing_result.actual_density_g_cm3,
                "packing_status": packing_result.status,
                "minimum_interatomic_distance_A": packing_result.minimum_interatomic_distance_A,
                "pore_volume_A3": packing_result.pore_volume_A3,
            }
        )
    return payload


def _unit_counts(built: BuiltGeometry) -> dict[str, int]:
    compact_count = sum(isinstance(unit, CompactUnit) for unit in built.units)
    channel_count = sum(isinstance(unit, ChannelUnit) for unit in built.units)
    return {
        "total": len(built.units),
        "compact": int(compact_count),
        "channel": int(channel_count),
    }


def _audit_summary(audit: AuditResult) -> dict[str, Any]:
    return {
        "passed": audit.passed,
        "warnings": list(audit.warnings),
        "scalar_errors": audit.scalar_errors,
        "distribution_results": {
            name: _distribution_comparison_summary(result)
            for name, result in audit.distribution_results.items()
        },
        "rdf_result": audit.rdf_result,
        "theta_result": _optional_distribution_summary(audit.theta_result),
        "compact_eta_result": _optional_distribution_summary(audit.compact_eta_result),
        "channel_eta_result": _optional_distribution_summary(audit.channel_eta_result),
        "compact_relative_volume_result": _optional_distribution_summary(
            audit.compact_relative_volume_result
        ),
        "channel_relative_volume_result": _optional_distribution_summary(
            audit.channel_relative_volume_result
        ),
        "roughness_result": _optional_distribution_summary(audit.roughness_result),
        "tau_result": _optional_distribution_summary(audit.tau_result),
        "center_distance_xy_result": audit.center_distance_xy_result,
        "equivalent_diameter_result": _optional_distribution_summary(
            audit.equivalent_diameter_result
        ),
        "theta_xz_result": _optional_distribution_summary(audit.theta_xz_result),
        "theta_xy_result": _optional_distribution_summary(audit.theta_xy_result),
        "curvature_fluctuation_result": _optional_distribution_summary(
            audit.curvature_fluctuation_result
        ),
        "formal_measurements": None
        if audit.formal_measurements is None
        else _json_ready(asdict(audit.formal_measurements)),
        "channel_fraction_error": audit.channel_fraction_error,
        "realized_mean_volume_ratio": audit.realized_mean_volume_ratio,
        "mean_volume_ratio_relative_error": audit.mean_volume_ratio_relative_error,
        "unit_volume_summary": audit.unit_volume_summary,
        "mixture_weight_errors": audit.mixture_weight_errors,
        "overlap_fraction": audit.overlap_fraction,
        "connected_pore_domains": audit.connected_pore_domains,
        "largest_pore_fraction": audit.largest_pore_fraction,
        "x_surface_openings": list(audit.x_surface_openings),
        "y_surface_openings": list(audit.y_surface_openings),
        "z_lower_opening_fraction": audit.z_lower_opening_fraction,
        "z_upper_opening_fraction": audit.z_upper_opening_fraction,
        "minimum_cross_section_fraction": audit.minimum_cross_section_fraction,
        "minimum_cross_section_index": audit.minimum_cross_section_index,
        "local_thickness_stability": {
            "passed": audit.local_thickness_stability_result.passed,
            "max_quantile_error_A": audit.local_thickness_stability_result.max_quantile_error_A,
            "mean_error_A": audit.local_thickness_stability_result.mean_error_A,
            "histogram_l1": audit.local_thickness_stability_result.histogram_l1,
            "tolerance_A": audit.local_thickness_stability_result.tolerance_A,
            "warning": audit.local_thickness_stability_result.warning,
        },
    }


def _optional_distribution_summary(
    result: Any | None,
) -> dict[str, Any] | None:
    return None if result is None else _distribution_comparison_summary(result)


def _distribution_comparison_summary(result: Any) -> dict[str, Any]:
    return {
        "passed": bool(result.passed),
        "ks": float(result.ks),
        "normalized_wasserstein": float(result.normalized_wasserstein),
    }


def _constraint_status(
    audit: AuditResult,
    *,
    packing_result: PackingResult | None = None,
) -> dict[str, Any]:
    return {
        "audit_passed": audit.passed,
        "packing_completed": packing_result is not None and packing_result.status == "packed",
        "minimum_cross_section_fraction": audit.minimum_cross_section_fraction,
        "overlap_fraction": audit.overlap_fraction,
        "warnings": list(audit.warnings),
    }


def _objectives(config: GeneratorConfig, run: GeometryRun) -> dict[str, Any]:
    return {
        "porosity_absolute_error": abs(run.phase_grid.porosity - config.pores.target_porosity),
        "objective": abs(run.phase_grid.porosity - config.pores.target_porosity),
    }


def _failure_reason_from_audit(audit: AuditResult) -> str:
    return next(iter(audit.warnings), "audit_failed")


def _qa_contract(
    config: GeneratorConfig,
    grid: PhaseGrid,
    *,
    run_identifier: str | None = None,
) -> dict[str, Any]:
    return {
        "length_unit": "angstrom",
        "task_id": config.task.name,
        "run_identifier": run_identifier or config.task.name,
        "origin_A": _vector_list(grid.origin_A),
        "target_box_A": _vector_list(_target_box(config)),
        "periodic_axes": ["x", "y"],
        "axis_order": "zyx",
        "phase_encoding": {"semiconductor": _MATRIX_PHASE_ID, "pore": _PORE_PHASE_ID},
        "target_origin_A": _vector_list(config.film.target_origin_in_packing_A),
    }


def _copy_source_pdb(config: GeneratorConfig, destination: Path) -> None:
    if config.pore_material is None:
        raise ValueError("pore_material is required for molecule-packing output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.pore_material.pdb, destination)


def _write_normalized_config(config: GeneratorConfig, path: Path) -> None:
    data = config.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_ready(payload), sort_keys=True) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, separators=(",", ": "))
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def _require_run_artifact(path: Path) -> Path:
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"required run artifact is missing: {artifact}")
    return artifact


def _target_box(config: GeneratorConfig) -> np.ndarray:
    target = config.film.target_box_A
    return np.array([target.x, target.y, target.z], dtype=float)


def _packing_box(config: GeneratorConfig) -> np.ndarray:
    packing = config.film.packing_box_A
    if packing is None:
        raise ValueError("packing_box_A could not be normalized")
    return np.array([packing.x, packing.y, packing.z], dtype=float)


def _estimated_voxel_count(target_box: np.ndarray, spacing: float) -> int:
    counts = np.ceil(target_box / float(spacing)).astype(np.int64)
    return int(np.prod(counts))


def _estimated_preflight_memory_bytes(config: GeneratorConfig, target_box: np.ndarray) -> int:
    return estimate_generation_memory_bytes(config)


def _box_axes_are_divisible_by_spacing(target_box: np.ndarray, spacing: float) -> bool:
    raw_counts = np.asarray(target_box, dtype=float) / float(spacing)
    rounded = np.rint(raw_counts)
    return bool(
        np.all(
            np.isclose(
                raw_counts,
                rounded,
                rtol=0.0,
                atol=_GRID_DIVISIBILITY_TOLERANCE,
            )
        )
    )


def _available_memory_bytes() -> int:
    return int(psutil.virtual_memory().available)


def _result_root(result_root: Path | None) -> Path:
    return Path(result_root) if result_root is not None else _DEFAULT_RESULT_ROOT


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _vector_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


def _json_ready(value: Any) -> Any:
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
