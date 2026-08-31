from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from porous_film.config import GeneratorConfig, load_config
from porous_film.parallel import (
    CandidateIdentity,
    CandidateResult,
    ParallelExecutionPlan,
    ParallelPoolError,
    ResourceSnapshot,
    SeedIdentity,
    SeedTaskResult,
)
from porous_film.pipeline import (
    GeometryRun,
    RunResult,
    generate_geometry,
    prepare_task_inputs,
    run_full_at_root,
)
from porous_film.storage import create_task_paths_at_root

_RUN_HEAVY = os.environ.get("POROUS_FILM_RUN_HEAVY") == "1"


def candidate_config(path: Path, strategy: str, workers: int | None):
    config = load_config(path)
    return config.model_copy(
        update={
            "audit": config.audit.model_copy(
                update={
                    "candidate_count_per_round": 2,
                    "maximum_rounds": 2,
                }
            ),
            "parallel": config.parallel.model_copy(
                update={
                    "strategy": strategy,
                    "max_workers": workers,
                }
            ),
        }
    )


def read_records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def seed_config(path: Path, strategy: str, workers: int | None) -> GeneratorConfig:
    config = load_config(path)
    return config.model_copy(
        update={
            "optimization": config.optimization.model_copy(update={"seed_panel": (11, 22)}),
            "parallel": config.parallel.model_copy(
                update={
                    "strategy": strategy,
                    "max_workers": workers,
                }
            ),
        }
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execution_plan(strategy: str, *, seed_count: int = 2, candidate_count: int = 2):
    return ParallelExecutionPlan(
        command="generate",
        requested_enabled=True,
        requested_strategy=strategy,
        effective_strategy=strategy,
        worker_count=2,
        seed_task_count=seed_count,
        candidate_task_count=candidate_count,
        parallel_task_count=seed_count if strategy == "seeds" else candidate_count,
        total_scientific_task_count=seed_count * candidate_count,
        estimated_worker_memory_bytes=1,
        worker_limits={"tasks": 2, "cpu": 2, "memory": 2, "user": 2},
        limiting_factors=("tasks",),
        resources=ResourceSnapshot.for_test(physical_cores=2, memory_bytes=10**9),
        warnings=(),
        fallback_reason=None,
    )


def optimizer_payload(status: str = "completed_feasible"):
    return {
        "requested": {},
        "realized": {},
        "feasible": status != "failed",
        "constraints": {},
        "calculation_status": {"status": status},
        "objectives": {},
        "uncertainty": {},
    }


def seed_result_for(task, paths, *, status: str = "completed_feasible") -> SeedTaskResult:
    feasible = status != "failed"
    return SeedTaskResult(
        identity=task.identity,
        status=status,
        feasible=feasible,
        objective=float(task.identity.panel_index) if feasible else None,
        failure_reason=None if feasible else "injected failure",
        artifact_root=str(paths.root),
        optimizer_payload=optimizer_payload(status),
        representative_pickle=None,
        candidate_summary={},
        traceback_text=None,
        wall_time_seconds=0.01,
    )


def representative_run_for(task, paths, config: GeneratorConfig) -> RunResult:
    return RunResult(
        paths=paths,
        geometry_run=GeometryRun(
            config=config,
            built=SimpleNamespace(units=[]),
            phase_grid=SimpleNamespace(porosity=0.3),
            audit=SimpleNamespace(passed=True, warnings=[]),
            paths=paths,
            candidate_results=(),
            selected_candidate=CandidateIdentity(
                seed=config.task.random_seed,
                round_index=0,
                candidate_index=0,
                derived_random_seed=config.task.random_seed,
                sequence_index=0,
            ),
        ),
        packing_result=None,
        status="completed_feasible",
    )


def test_build_seed_tasks_uses_index_seed_roots_and_forces_seed_workers_serial(
    feasible_config_path: Path,
    tmp_path: Path,
) -> None:
    from porous_film.parallel.seeds import build_seed_tasks

    top_paths = create_task_paths_at_root(tmp_path / "top")
    config = seed_config(feasible_config_path, "seeds", 2)

    tasks = build_seed_tasks(config, top_paths, execution_plan("seeds"), "top-run")

    assert [Path(task.task_root).name for task in tasks] == ["0-11", "1-22"]
    assert [task.representative for task in tasks] == [True, False]
    assert [task.identity.sequence_index for task in tasks] == [0, 1]
    assert [task.identity.panel_index for task in tasks] == [0, 1]
    assert [task.identity.seed for task in tasks] == [11, 22]

    child_configs = [GeneratorConfig.model_validate(task.config_payload) for task in tasks]
    assert [child.task.random_seed for child in child_configs] == [11, 22]
    assert [child.optimization.seed_panel for child in child_configs] == [(11,), (22,)]
    for child in child_configs:
        assert child.parallel.enabled is False
        assert child.parallel.strategy == "serial"
        assert child.parallel.max_workers is None
        assert child.parallel.sources.enabled == "worker-forced-serial"
        assert child.parallel.sources.strategy == "worker-forced-serial"
        assert child.parallel.sources.max_workers == "worker-forced-serial"


def test_build_seed_tasks_preserves_candidate_strategy_for_parent_candidate_pool(
    feasible_config_path: Path,
    tmp_path: Path,
) -> None:
    from porous_film.parallel.seeds import build_seed_tasks

    top_paths = create_task_paths_at_root(tmp_path / "top")
    config = seed_config(feasible_config_path, "candidates", 2)

    tasks = build_seed_tasks(config, top_paths, execution_plan("candidates"), "top-run")

    child_configs = [GeneratorConfig.model_validate(task.config_payload) for task in tasks]
    for child in child_configs:
        assert child.parallel.enabled is True
        assert child.parallel.strategy == "candidates"
        assert child.parallel.max_workers == 2


def test_z_padding_seed_payload_round_trips_without_derived_packing_box(
    sample_config_path: Path,
    tmp_path: Path,
) -> None:
    from porous_film.parallel.seeds import build_seed_tasks

    config_data = load_config(sample_config_path).model_dump(mode="json")
    film_data = dict(config_data["film"])
    film_data.pop("packing_box_A")
    film_data["z_padding_A"] = {"lower": 7.0, "upper": 3.0}
    config = GeneratorConfig.model_validate({**config_data, "film": film_data})
    top_paths = create_task_paths_at_root(tmp_path / "top")

    tasks = build_seed_tasks(config, top_paths, execution_plan("seeds"), "top-run")

    payload_film = tasks[0].config_payload["film"]
    assert payload_film == {
        "target_box_A": {"x": 20.0, "y": 20.0, "z": 20.0},
        "z_padding_A": {"lower": 7.0, "upper": 3.0},
    }
    child = GeneratorConfig.model_validate(tasks[0].config_payload)
    assert child.film.z_padding_A == config.film.z_padding_A
    assert child.film.packing_box_A == config.film.packing_box_A


def test_explicit_packing_box_seed_payload_round_trips_without_z_padding_key(
    sample_config_path: Path,
    tmp_path: Path,
) -> None:
    from porous_film.parallel.seeds import build_seed_tasks

    config = load_config(sample_config_path)
    top_paths = create_task_paths_at_root(tmp_path / "top")

    tasks = build_seed_tasks(config, top_paths, execution_plan("seeds"), "top-run")

    payload_film = tasks[0].config_payload["film"]
    assert payload_film == {
        "target_box_A": {"x": 20.0, "y": 20.0, "z": 20.0},
        "packing_box_A": {"x": 20.0, "y": 20.0, "z": 30.0},
    }
    child = GeneratorConfig.model_validate(tasks[0].config_payload)
    assert child.film.z_padding_A is None
    assert child.film.packing_box_A == config.film.packing_box_A


def test_parallel_plan_and_summary_have_required_metadata(tmp_path: Path) -> None:
    from porous_film.parallel.reporting import write_parallel_plan, write_parallel_summary
    from porous_film.parallel.runtime import THREAD_ENVIRONMENT

    paths = create_task_paths_at_root(tmp_path / "metadata")
    plan = execution_plan("seeds")
    started = datetime(2026, 8, 13, 1, 2, 3, tzinfo=UTC)
    finished = datetime(2026, 8, 13, 1, 2, 8, tzinfo=UTC)
    first = SeedTaskResult(
        identity=SeedIdentity(panel_index=0, seed=11, sequence_index=0),
        status="completed_feasible",
        feasible=True,
        objective=0.0,
        failure_reason=None,
        artifact_root=str(paths.analysis / "seed_panel" / "0-11"),
        optimizer_payload={},
        representative_pickle=None,
        candidate_summary={},
        traceback_text=None,
        wall_time_seconds=1.25,
    )
    cancelled = {
        "identity": {"panel_index": 1, "seed": 22, "sequence_index": 1},
        "status": "cancelled",
        "wall_time_seconds": None,
        "failure": {"type": "cancelled", "message": "missing after interruption"},
    }

    plan_path = write_parallel_plan(paths, plan)
    summary_path = write_parallel_summary(
        paths,
        plan,
        [first, cancelled],
        started=started,
        finished=finished,
        cancellation={"type": "KeyboardInterrupt", "missing_identities": [cancelled["identity"]]},
        pool_failure=None,
        serial_fallback_reason=None,
    )

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["effective_strategy"] == "seeds"
    assert payload["worker_count"] == 2
    assert payload["worker_limits"] == {"cpu": 2, "memory": 2, "tasks": 2, "user": 2}
    assert payload["limiting_factors"] == ["tasks"]
    assert payload["resources"]["allowed_logical_cpus"] == [0, 1]
    assert payload["resources"]["allowed_logical_count"] == 2
    assert payload["resources"]["available_physical_cores"] == 2
    assert payload["resources"]["effective_memory_bytes"] == 10**9
    assert payload["blas_environment"] == {
        name: os.environ.get(name) for name in THREAD_ENVIRONMENT
    }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["task_count"] == 2
    assert summary["plan"]["requested_settings"] == payload["requested_settings"]
    assert summary["plan"]["settings_sources"] == payload["settings_sources"]
    assert summary["plan"]["resources"] == payload["resources"]
    assert summary["plan"]["worker_limits"] == payload["worker_limits"]
    assert summary["plan"]["limiting_factors"] == payload["limiting_factors"]
    assert summary["plan"]["blas_environment"] == payload["blas_environment"]
    assert summary["plan"]["seed_task_count"] == 2
    assert summary["plan"]["candidate_task_count"] == 2
    assert summary["plan"]["total_scientific_task_count"] == 4
    assert summary["completed_count"] == 1
    assert summary["failed_count"] == 0
    assert summary["cancelled_count"] == 1
    assert summary["total_wall_time_seconds"] == 5.0
    assert summary["tasks"][0]["identity"] == {
        "panel_index": 0,
        "seed": 11,
        "sequence_index": 0,
    }
    assert summary["tasks"][0]["status"] == "completed_feasible"
    assert summary["tasks"][1]["status"] == "cancelled"
    assert summary["cancellation"]["type"] == "KeyboardInterrupt"
    markdown = (paths.reports / "parallel-summary.md").read_text(encoding="utf-8")
    assert "## Tasks" in markdown
    assert "|" not in markdown


def test_seed_pool_failure_summary_records_cancelled_missing_identity(
    feasible_config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from porous_film.parallel import ParallelPoolError
    from porous_film.pipeline import _run_seed_panel

    paths = create_task_paths_at_root(tmp_path / "pool-failure")
    config = seed_config(feasible_config_path, "seeds", 2)
    completed = SeedTaskResult(
        identity=SeedIdentity(panel_index=0, seed=11, sequence_index=0),
        status="completed_feasible",
        feasible=True,
        objective=0.0,
        failure_reason=None,
        artifact_root=str(paths.analysis / "seed_panel" / "0-11"),
        optimizer_payload={},
        representative_pickle=None,
        candidate_summary={},
        traceback_text=None,
        wall_time_seconds=1.0,
    )

    def raise_pool_error(*args, **kwargs):
        raise ParallelPoolError("seed worker died", (completed,), started_task_count=2)

    monkeypatch.setattr("porous_film.pipeline._seed_panel_execution_plan", lambda config: execution_plan("seeds"))
    monkeypatch.setattr("porous_film.pipeline.run_spawn_tasks", raise_pool_error)

    with pytest.raises(ParallelPoolError):
        _run_seed_panel(config, paths)

    summary = json.loads((paths.analysis / "parallel-summary.json").read_text())
    assert summary["task_count"] == 2
    assert summary["completed_count"] == 1
    assert summary["cancelled_count"] == 1
    assert summary["tasks"][1]["identity"] == {
        "panel_index": 1,
        "seed": 22,
        "sequence_index": 1,
    }
    assert summary["tasks"][1]["status"] == "cancelled"
    assert summary["pool_failure"]["started_task_count"] == 2


def test_candidate_strategy_pool_failure_summarizes_seed_identities_only(
    feasible_config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from porous_film.config import GeneratorConfig
    from porous_film.pipeline import _run_seed_panel
    from porous_film.storage import create_task_paths_at_root

    paths = create_task_paths_at_root(tmp_path / "candidate-pool-failure")
    config = seed_config(feasible_config_path, "candidates", 2)
    candidate_result = CandidateResult(
        identity=CandidateIdentity(
            seed=11,
            round_index=4,
            candidate_index=5,
            derived_random_seed=123,
            sequence_index=0,
        ),
        succeeded=True,
        audit_passed=True,
        porosity=0.3,
        scale=1.0,
        warnings=(),
        failure_type=None,
        failure_message=None,
        traceback_text=None,
        wall_time_seconds=0.02,
    )

    def fake_execute_seed_body(task, *, candidate_executor=None):
        seed_paths = create_task_paths_at_root(Path(task.task_root))
        seed_config = GeneratorConfig.model_validate(task.config_payload)
        if task.identity.seed == 22:
            raise ParallelPoolError(
                "candidate pool died",
                (candidate_result,),
                started_task_count=1,
            )
        result = seed_result_for(task, seed_paths)
        return result, representative_run_for(task, seed_paths, seed_config)

    monkeypatch.setattr(
        "porous_film.pipeline._seed_panel_execution_plan",
        lambda config: execution_plan("candidates"),
    )
    monkeypatch.setattr("porous_film.pipeline._execute_seed_body", fake_execute_seed_body)

    with pytest.raises(ParallelPoolError):
        _run_seed_panel(config, paths)

    summary = json.loads((paths.analysis / "parallel-summary.json").read_text())
    assert summary["task_count"] == 2
    assert summary["completed_count"] == 1
    assert summary["cancelled_count"] == 1
    assert [task["identity"]["seed"] for task in summary["tasks"]] == [11, 22]
    assert "round_index" not in summary["tasks"][0]["identity"]
    assert summary["pool_failure"]["completed_results"][0]["identity"] == {
        "candidate_index": 5,
        "derived_random_seed": 123,
        "round_index": 4,
        "seed": 11,
        "sequence_index": 0,
    }


def test_seed_task_converts_keyerror_to_structured_failure_with_traceback_file(
    feasible_config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from porous_film.parallel.seeds import SeedTask, _execute_seed_body

    config = seed_config(feasible_config_path, "serial", None)
    seed_root = tmp_path / "seed-root"
    task = SeedTask(
        identity=SeedIdentity(panel_index=0, seed=config.task.random_seed, sequence_index=0),
        config_payload=config.model_dump(mode="json"),
        task_root=str(seed_root),
        top_level_run_identifier="top",
        representative=True,
    )

    def raise_key_error(*args, **kwargs):
        raise KeyError("ordinary seed failure")

    monkeypatch.setattr("porous_film.pipeline._execute_single_seed", raise_key_error)

    result, representative = _execute_seed_body(task, candidate_executor=None)

    assert representative is None
    assert result.status == "failed"
    assert result.sequence_index == 0
    assert "KeyError: 'ordinary seed failure'" in (result.traceback_text or "")
    record = json.loads((seed_root / "logs" / "seed-record.json").read_text(encoding="utf-8"))
    assert record["failure_reason"] == "KeyError: 'ordinary seed failure'"
    assert record["traceback_file"] == "logs/seed-traceback.txt"
    traceback_path = seed_root / record["traceback_file"]
    assert "KeyError: 'ordinary seed failure'" in traceback_path.read_text(encoding="utf-8")


def test_representative_pickle_failure_becomes_seed_failure_record(
    feasible_config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pickle

    from porous_film.parallel import seeds
    from porous_film.parallel.seeds import SeedTask, execute_seed_task

    config = seed_config(feasible_config_path, "serial", None)
    seed_root = tmp_path / "seed-root"
    paths = create_task_paths_at_root(seed_root)
    task = SeedTask(
        identity=SeedIdentity(panel_index=0, seed=config.task.random_seed, sequence_index=0),
        config_payload=config.model_dump(mode="json"),
        task_root=str(seed_root),
        top_level_run_identifier="top",
        representative=True,
    )
    successful_result = seed_result_for(task, paths)
    representative = representative_run_for(task, paths, config)

    def fake_execute_seed_body(task, candidate_executor=None):
        return successful_result, representative

    def raise_pickling_error(*args, **kwargs):
        raise pickle.PicklingError("cannot serialize representative")

    monkeypatch.setattr(seeds, "_execute_seed_body", fake_execute_seed_body)
    monkeypatch.setattr(pickle, "dump", raise_pickling_error)

    result = execute_seed_task(task)

    assert result.status == "failed"
    assert result.representative_pickle is None
    assert "PicklingError: cannot serialize representative" in (result.traceback_text or "")
    record = json.loads((seed_root / "logs" / "seed-record.json").read_text(encoding="utf-8"))
    assert record["traceback_file"] == "logs/seed-traceback.txt"
    assert "PicklingError" in (seed_root / record["traceback_file"]).read_text(encoding="utf-8")


def test_serial_seed_failure_injection_produces_aggregate_without_escaping(
    feasible_config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from porous_film import pipeline
    from porous_film.pipeline import _run_seed_panel
    from porous_film.storage import create_task_paths_at_root

    paths = create_task_paths_at_root(tmp_path / "serial-failure")
    config = seed_config(feasible_config_path, "serial", None)

    def fake_execute_single_seed(seed_config, paths, **kwargs):
        if seed_config.task.random_seed == 22:
            raise ValueError("injected seed failure")
        return representative_run_for(
            SimpleNamespace(),
            paths,
            seed_config,
        )

    monkeypatch.setattr(pipeline, "_execute_single_seed", fake_execute_single_seed)
    monkeypatch.setattr(pipeline, "optimizer_payload_for_result", lambda result: optimizer_payload())
    _run_seed_panel(config, paths)

    aggregate = json.loads((paths.analysis / "seed_panel" / "aggregate.json").read_text())
    assert aggregate["seed_count"] == 2
    assert "22" in aggregate["failures"]
    failed_record = json.loads(
        (paths.analysis / "seed_panel" / "1-22" / "logs" / "seed-record.json").read_text()
    )
    assert failed_record["status"] == "failed"
    assert "ValueError: injected seed failure" in failed_record["failure_reason"]


def test_seed_panel_parent_shared_files_survive_worker_and_publication_boundaries(
    feasible_config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from porous_film.config import GeneratorConfig
    from porous_film.parallel.reporting import write_parallel_plan as real_write_plan
    from porous_film.parallel.reporting import write_parallel_summary as real_write_summary
    from porous_film.pipeline import _publish_representative as real_publish
    from porous_film.pipeline import _run_seed_panel
    from porous_film.storage import create_task_paths_at_root

    paths = create_task_paths_at_root(tmp_path / "shared-boundaries")
    config = seed_config(feasible_config_path, "serial", None)
    boundary_events: list[str] = []
    plan_hash: str | None = None
    summary_hash_after_write: str | None = None

    def fake_write_plan(paths_arg, plan):
        nonlocal plan_hash
        result = real_write_plan(paths_arg, plan)
        plan_hash = sha256(result)
        boundary_events.append("plan")
        assert not (paths_arg.analysis / "parallel-summary.json").exists()
        assert not (paths_arg.reports / "parallel-summary.md").exists()
        return result

    def fake_execute_seed_body(task, *, candidate_executor=None):
        assert plan_hash == sha256(paths.work / "parallel" / "parallel-plan.json")
        assert not (paths.analysis / "parallel-summary.json").exists()
        assert not (paths.reports / "parallel-summary.md").exists()
        seed_paths = create_task_paths_at_root(Path(task.task_root))
        seed_config = GeneratorConfig.model_validate(task.config_payload)
        (seed_paths.reports / "parallel-summary.md").write_text(
            f"child summary {task.identity.seed}",
            encoding="utf-8",
        )
        (seed_paths.outputs / "representative.txt").write_text("artifact", encoding="utf-8")
        result = seed_result_for(task, seed_paths)
        resolved = Path(result.artifact_root).resolve()
        assert resolved.is_relative_to(Path(task.task_root).resolve())
        boundary_events.append(f"worker-{task.identity.seed}")
        run_result = (
            representative_run_for(task, seed_paths, seed_config)
            if task.representative
            else None
        )
        return result, run_result

    def fake_write_summary(paths_arg, plan, task_results, **kwargs):
        nonlocal summary_hash_after_write
        result = real_write_summary(paths_arg, plan, task_results, **kwargs)
        summary_hash_after_write = sha256(paths_arg.reports / "parallel-summary.md")
        boundary_events.append("summary")
        return result

    def fake_publish(seed_paths, top_paths, top_config):
        assert summary_hash_after_write == sha256(top_paths.reports / "parallel-summary.md")
        before = (top_paths.reports / "parallel-summary.md").read_text(encoding="utf-8")
        real_publish(seed_paths, top_paths, top_config)
        after = (top_paths.reports / "parallel-summary.md").read_text(encoding="utf-8")
        assert after == before
        assert "child summary" not in after
        boundary_events.append("publish")

    monkeypatch.setattr("porous_film.pipeline.write_parallel_plan", fake_write_plan)
    monkeypatch.setattr("porous_film.pipeline.write_parallel_summary", fake_write_summary)
    monkeypatch.setattr("porous_film.pipeline._execute_seed_body", fake_execute_seed_body)
    monkeypatch.setattr("porous_film.pipeline._publish_representative", fake_publish)

    _run_seed_panel(config, paths)

    assert boundary_events == ["plan", "worker-11", "worker-22", "summary", "publish"]
    assert plan_hash == sha256(paths.work / "parallel" / "parallel-plan.json")


def test_publish_representative_copies_metrics_without_overwriting_parent_analysis(
    feasible_config_path: Path,
    tmp_path: Path,
) -> None:
    from porous_film.pipeline import _publish_representative

    top_paths = create_task_paths_at_root(tmp_path / "top")
    seed_paths = create_task_paths_at_root(top_paths.analysis / "seed_panel" / "0-11")
    config = load_config(feasible_config_path)
    representative_metrics = {
        "target_porosity": 0.30,
        "realized_porosity": 0.299,
        "porosity_absolute_error": 0.001,
    }
    parent_parallel_summary = {"owner": "parent-parallel-summary"}
    parent_seed_aggregate = {"owner": "parent-seed-aggregate"}
    child_parallel_summary = {"owner": "child-parallel-summary"}
    child_seed_aggregate = {"owner": "child-seed-aggregate"}

    (seed_paths.analysis / "target_vs_actual_metrics.json").write_text(
        json.dumps(representative_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (top_paths.analysis / "parallel-summary.json").write_text(
        json.dumps(parent_parallel_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (top_paths.analysis / "seed_panel").mkdir(exist_ok=True)
    (top_paths.analysis / "seed_panel" / "aggregate.json").write_text(
        json.dumps(parent_seed_aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (seed_paths.analysis / "parallel-summary.json").write_text(
        json.dumps(child_parallel_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (seed_paths.analysis / "seed_panel").mkdir(exist_ok=True)
    (seed_paths.analysis / "seed_panel" / "aggregate.json").write_text(
        json.dumps(child_seed_aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _publish_representative(seed_paths, top_paths, config)

    assert json.loads(
        (top_paths.analysis / "target_vs_actual_metrics.json").read_text(encoding="utf-8")
    ) == representative_metrics
    assert json.loads(
        (top_paths.analysis / "parallel-summary.json").read_text(encoding="utf-8")
    ) == parent_parallel_summary
    assert json.loads(
        (top_paths.analysis / "seed_panel" / "aggregate.json").read_text(encoding="utf-8")
    ) == parent_seed_aggregate


@pytest.mark.skipif(
    not _RUN_HEAVY,
    reason="Task 8 metadata generation test requires POROUS_FILM_RUN_HEAVY=1",
)
def test_parallel_metadata_has_required_resource_and_task_fields(
    feasible_config_path: Path, tmp_path: Path
) -> None:
    result = run_full_at_root(
        seed_config(feasible_config_path, "seeds", 2), tmp_path / "metadata"
    )
    plan = json.loads((result.paths.work / "parallel" / "parallel-plan.json").read_text())
    summary = json.loads((result.paths.analysis / "parallel-summary.json").read_text())
    assert plan["effective_strategy"] == "seeds"
    assert plan["worker_count"] == 2
    assert plan["resources"]["allowed_logical_count"] >= 1
    assert plan["resources"]["effective_memory_bytes"] >= 1
    assert summary["task_count"] == 2
    assert summary["completed_count"] == 2
    assert summary["cancelled_count"] == 0
    assert (result.paths.reports / "parallel-summary.md").is_file()


@pytest.mark.skipif(
    not _RUN_HEAVY,
    reason="Task 8 seed failure-isolation test requires POROUS_FILM_RUN_HEAVY=1",
)
def test_one_seed_failure_does_not_remove_other_seed_record(
    feasible_config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from porous_film import pipeline

    original = pipeline._execute_single_seed

    def injected(seed_config, *args, **kwargs):
        if seed_config.task.random_seed == 22:
            raise ValueError("injected seed failure")
        return original(seed_config, *args, **kwargs)

    monkeypatch.setattr(pipeline, "_execute_single_seed", injected)
    config = seed_config(feasible_config_path, "serial", None)
    result = run_full_at_root(config, tmp_path / "isolated")
    aggregate = json.loads(
        (result.paths.analysis / "seed_panel" / "aggregate.json").read_text()
    )
    assert aggregate["seed_count"] == 2
    assert "22" in aggregate["failures"]
    assert (
        result.paths.analysis / "seed_panel" / "0-11" / "logs" / "seed-record.json"
    ).is_file()


@pytest.mark.skipif(
    not _RUN_HEAVY,
    reason="Task 8 parent-only shared-write test requires POROUS_FILM_RUN_HEAVY=1",
)
def test_seed_worker_results_stay_under_seed_roots_and_top_shared_writes_are_parent_only(
    feasible_config_path: Path, tmp_path: Path
) -> None:
    result = run_full_at_root(
        seed_config(feasible_config_path, "seeds", 2), tmp_path / "parent-only"
    )
    shared_files = [
        result.paths.work / "parallel" / "parallel-plan.json",
        result.paths.analysis / "parallel-summary.json",
        result.paths.reports / "parallel-summary.md",
    ]
    assert all(path.is_file() for path in shared_files)
    for seed_root in sorted((result.paths.analysis / "seed_panel").iterdir()):
        if not seed_root.is_dir():
            continue
        record = json.loads((seed_root / "logs" / "seed-record.json").read_text())
        artifact_root = Path(str(record["artifact_root"])).resolve()
        assert artifact_root.is_relative_to(seed_root.resolve())


@pytest.mark.skipif(
    not _RUN_HEAVY,
    reason="Task 7 heavy pore-generation equivalence test requires POROUS_FILM_RUN_HEAVY=1",
)
def test_serial_and_two_worker_seed_panels_match_without_duplicate_representative(
    feasible_config_path: Path,
    tmp_path: Path,
) -> None:
    from porous_film.pipeline import run_full_at_root

    serial = run_full_at_root(
        seed_config(feasible_config_path, "serial", None),
        tmp_path / "serial",
        config_path=feasible_config_path,
    )
    parallel = run_full_at_root(
        seed_config(feasible_config_path, "seeds", 2),
        tmp_path / "parallel",
        config_path=feasible_config_path,
    )
    for result in (serial, parallel):
        panel = result.paths.analysis / "seed_panel"
        assert sorted(path.name for path in panel.iterdir() if path.is_dir()) == ["0-11", "1-22"]
        assert len(list(panel.rglob("seed-record.json"))) == 2
        assert not list(panel.rglob("representative-run.pkl"))
    serial_aggregate = json.loads(
        (serial.paths.analysis / "seed_panel" / "aggregate.json").read_text()
    )
    parallel_aggregate = json.loads(
        (parallel.paths.analysis / "seed_panel" / "aggregate.json").read_text()
    )
    assert serial_aggregate == parallel_aggregate
    assert sha256(serial.paths.outputs / "pore_geometry.h5") == sha256(
        parallel.paths.outputs / "pore_geometry.h5"
    )
    assert sha256(serial.paths.outputs / "pore_reference_coordinates.cif") == sha256(
        parallel.paths.outputs / "pore_reference_coordinates.cif"
    )


def test_serial_and_two_worker_candidates_match(
    sample_config_path: Path,
    tmp_path: Path,
) -> None:
    serial_paths = create_task_paths_at_root(tmp_path / "serial")
    parallel_paths = create_task_paths_at_root(tmp_path / "parallel")
    serial_config = candidate_config(sample_config_path, "serial", None)
    parallel_config = candidate_config(sample_config_path, "candidates", 2)
    prepare_task_inputs(serial_config, serial_paths, config_path=sample_config_path)
    prepare_task_inputs(parallel_config, parallel_paths, config_path=sample_config_path)

    serial = generate_geometry(serial_config, serial_paths)
    parallel = generate_geometry(parallel_config, parallel_paths)

    assert serial.selected_candidate == parallel.selected_candidate
    assert serial.phase_grid.porosity == parallel.phase_grid.porosity
    assert [unit.to_record() for unit in serial.built.units] == [
        unit.to_record() for unit in parallel.built.units
    ]
    serial_records = read_records(serial_paths.qa_export / "unit_candidates.jsonl")
    parallel_records = read_records(
        parallel_paths.qa_export / "unit_candidates.jsonl"
    )
    for records in (serial_records, parallel_records):
        for record in records:
            record.pop("wall_time_seconds")
    assert serial_records == parallel_records
