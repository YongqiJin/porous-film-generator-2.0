from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from conftest import write_config

from porous_film.config import GeneratorConfig, load_config
from porous_film.parallel.candidates import (
    CandidateResult,
    CandidateTask,
    build_candidate_tasks,
    evaluate_candidate_task,
    replay_candidate,
    select_candidate,
)
from porous_film.pipeline import GeometryAcceptanceError


def test_identities_keep_existing_rng_order(sample_config_path: Path) -> None:
    config = load_config(sample_config_path)
    config = config.model_copy(
        update={
            "audit": config.audit.model_copy(
                update={"candidate_count_per_round": 3, "maximum_rounds": 2}
            )
        }
    )
    rng = np.random.default_rng(config.task.random_seed)
    expected = [int(rng.integers(0, np.iinfo(np.int32).max)) for _ in range(6)]
    tasks = build_candidate_tasks(config)
    assert [task.identity.derived_random_seed for task in tasks] == expected
    assert [task.sequence_index for task in tasks] == list(range(6))


def test_candidate_tasks_are_picklable_and_json_sized(sample_config_path: Path) -> None:
    task = build_candidate_tasks(load_config(sample_config_path))[0]

    restored = pickle.loads(pickle.dumps(task))

    assert restored == task
    assert isinstance(task.config_payload, dict)
    assert isinstance(task.config_payload["film"], dict)


def test_candidate_task_payload_forces_serial_without_mutating_caller(
    sample_config_path: Path,
) -> None:
    config = load_config(sample_config_path)
    parallel = config.parallel.model_copy(
        update={
            "enabled": True,
            "strategy": "candidates",
            "max_workers": 4,
        }
    )
    config = config.model_copy(update={"parallel": parallel})

    task = build_candidate_tasks(config)[0]

    assert config.parallel.enabled is True
    assert config.parallel.strategy == "candidates"
    assert config.parallel.max_workers == 4
    assert task.config_payload["parallel"] == {
        "enabled": False,
        "strategy": "serial",
        "max_workers": None,
        "cpu_fraction": config.parallel.cpu_fraction,
        "memory_fraction": config.parallel.memory_fraction,
        "worker_threads": config.parallel.worker_threads,
        "start_method": config.parallel.start_method,
        "sources": {
            "enabled": "worker:forced-serial",
            "strategy": "worker:forced-serial",
            "max_workers": "worker:forced-serial",
        },
    }


def test_z_padding_candidate_payload_round_trips_without_derived_packing_box(
    tmp_path: Path,
) -> None:
    config = load_config(write_config(tmp_path, name="z-padding", lower_padding=7.0))

    task = build_candidate_tasks(config)[0]
    film_payload = task.config_payload["film"]

    assert film_payload == {
        "target_box_A": {"x": 20.0, "y": 20.0, "z": 20.0},
        "z_padding_A": {"lower": 7.0, "upper": 3.0},
    }
    round_tripped = GeneratorConfig.model_validate(task.config_payload)
    assert round_tripped.film.z_padding_A == config.film.z_padding_A
    assert round_tripped.film.packing_box_A == config.film.packing_box_A
    assert config.film.z_padding_A is not None
    assert config.film.packing_box_A is not None


def test_explicit_packing_box_candidate_payload_round_trips_with_packing_box(
    sample_config_path: Path,
) -> None:
    config = load_config(sample_config_path)

    task = build_candidate_tasks(config)[0]
    film_payload = task.config_payload["film"]

    assert film_payload == {
        "target_box_A": {"x": 20.0, "y": 20.0, "z": 20.0},
        "packing_box_A": {"x": 20.0, "y": 20.0, "z": 30.0},
    }
    round_tripped = GeneratorConfig.model_validate(task.config_payload)
    assert round_tripped.film.z_padding_A is None
    assert round_tripped.film.packing_box_A == config.film.packing_box_A


def test_worker_result_is_compact(sample_config_path: Path) -> None:
    result = evaluate_candidate_task(build_candidate_tasks(load_config(sample_config_path))[0])
    assert result.succeeded is True
    assert isinstance(result.porosity, float)
    assert not any(isinstance(value, np.ndarray) for value in result.__dict__.values())


def test_replay_matches_summary(sample_config_path: Path) -> None:
    config = load_config(sample_config_path)
    task = build_candidate_tasks(config)[0]
    summary = evaluate_candidate_task(task)
    replay = replay_candidate(config, task.identity)
    assert replay.phase_grid.porosity == summary.porosity
    assert replay.scale == summary.scale
    assert replay.audit.passed == summary.audit_passed


def test_worker_serializes_validation_failure(sample_config_path: Path) -> None:
    task = build_candidate_tasks(load_config(sample_config_path))[0]
    payload = dict(task.config_payload)
    payload["film"] = dict(payload["film"])
    payload["film"]["target_box_A"] = {"x": 0, "y": 20, "z": 20}
    result = evaluate_candidate_task(CandidateTask(task.identity, payload))
    assert result.succeeded is False
    assert result.failure_type == "ValidationError"
    assert "target_box_A" in (result.traceback_text or "")


def test_select_candidate_prefers_first_passing_sequence(sample_config_path: Path) -> None:
    config = load_config(sample_config_path)
    tasks = build_candidate_tasks(
        config.model_copy(
            update={
                "audit": config.audit.model_copy(
                    update={"candidate_count_per_round": 3, "maximum_rounds": 1}
                )
            }
        )
    )
    earlier_failed = _result(tasks[0], succeeded=True, audit_passed=False, porosity=0.12)
    first_passing = _result(tasks[1], succeeded=True, audit_passed=True, porosity=0.16)
    later_passing = _result(tasks[2], succeeded=True, audit_passed=True, porosity=0.11)

    selected = select_candidate(config, [later_passing, first_passing, earlier_failed])

    assert selected == first_passing


def test_select_candidate_falls_back_to_nearest_porosity_then_sequence(
    sample_config_path: Path,
) -> None:
    config = load_config(sample_config_path)
    tasks = build_candidate_tasks(
        config.model_copy(
            update={
                "audit": config.audit.model_copy(
                    update={"candidate_count_per_round": 3, "maximum_rounds": 1}
                )
            }
        )
    )
    target = config.pores.target_porosity
    earlier_tie = _result(tasks[0], succeeded=True, audit_passed=False, porosity=target + 0.02)
    nearest = _result(tasks[1], succeeded=True, audit_passed=False, porosity=target + 0.01)
    later_tie = _result(tasks[2], succeeded=True, audit_passed=False, porosity=target - 0.01)

    selected = select_candidate(config, [later_tie, nearest, earlier_tie])

    assert selected == nearest


def test_select_candidate_raises_with_ordered_failures_when_all_failed(
    sample_config_path: Path,
) -> None:
    config = load_config(sample_config_path)
    tasks = build_candidate_tasks(
        config.model_copy(
            update={
                "audit": config.audit.model_copy(
                    update={"candidate_count_per_round": 2, "maximum_rounds": 1}
                )
            }
        )
    )
    failure_one = _result(tasks[0], succeeded=False, audit_passed=False, porosity=None)
    failure_two = _result(tasks[1], succeeded=False, audit_passed=False, porosity=None)

    with pytest.raises(GeometryAcceptanceError) as exc_info:
        select_candidate(config, [failure_two, failure_one])

    assert exc_info.value.reason == "no_geometry_candidates"
    assert exc_info.value.warnings == [failure_one, failure_two]


def _result(
    task: CandidateTask,
    *,
    succeeded: bool,
    audit_passed: bool,
    porosity: float | None,
) -> CandidateResult:
    return CandidateResult(
        identity=task.identity,
        succeeded=succeeded,
        audit_passed=audit_passed,
        porosity=porosity,
        scale=1.0 if succeeded else None,
        warnings=(),
        failure_type=None if succeeded else "RuntimeError",
        failure_message=None if succeeded else "failed",
        traceback_text=None if succeeded else "traceback",
        wall_time_seconds=0.0,
    )
