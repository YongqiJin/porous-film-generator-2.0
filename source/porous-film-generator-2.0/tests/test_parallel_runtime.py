from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

import pytest

from porous_film.parallel.runtime import (
    THREAD_ENVIRONMENT,
    ParallelCancelled,
    ParallelPoolError,
    ensure_pool_allowed,
    numeric_thread_limits,
    run_spawn_tasks,
    spawn_pool,
)
from porous_film.pipeline import GeometryAcceptanceError


@dataclass(frozen=True)
class ProbeTask:
    sequence_index: int


def runtime_probe(task: ProbeTask) -> tuple[int, str | None, tuple[str | None, ...]]:
    return (
        task.sequence_index,
        os.environ.get("POROUS_FILM_WORKER"),
        tuple(os.environ.get(name) for name in THREAD_ENVIRONMENT),
    )


def echo_sequence(task: ProbeTask) -> int:
    return task.sequence_index


def pool_error_probe(task: ProbeTask) -> str:
    if task.sequence_index == 1:
        time.sleep(0.05)
        raise BrokenProcessPool("worker died")
    time.sleep(0.01 if task.sequence_index == 0 else 0.005)
    return f"done-{task.sequence_index}"


def interrupt_probe(task: ProbeTask) -> str:
    if task.sequence_index == 1:
        marker_dir_text = os.environ.get("POROUS_FILM_INTERRUPT_MARKER_DIR")
        if marker_dir_text is not None:
            marker_dir = Path(marker_dir_text)
            expected = (marker_dir / "done-0", marker_dir / "done-2")
            deadline = time.monotonic() + 10.0
            while not all(path.is_file() for path in expected):
                if time.monotonic() >= deadline:
                    raise RuntimeError("timed out waiting for completed-result callbacks")
                time.sleep(0.005)
        else:
            time.sleep(0.05)
        raise KeyboardInterrupt
    time.sleep(0.01 if task.sequence_index == 0 else 0.005)
    return f"done-{task.sequence_index}"


def abrupt_exit_probe(task: ProbeTask) -> str:
    if task.sequence_index == 1:
        time.sleep(0.30)
        os._exit(7)
    time.sleep(0.01 if task.sequence_index == 0 else 0.005)
    return f"done-{task.sequence_index}"


class RecordingExecutor:
    def __init__(self) -> None:
        self.submitted_sequence_indices: list[int] = []

    def submit(
        self, worker: Callable[[ProbeTask], object], task: ProbeTask
    ) -> Future[object]:
        self.submitted_sequence_indices.append(task.sequence_index)
        future: Future[object] = Future()
        future.set_result(worker(task))
        return future


def test_spawn_is_ordered_and_thread_limited() -> None:
    results = run_spawn_tasks(
        [ProbeTask(2), ProbeTask(0), ProbeTask(1)],
        runtime_probe,
        worker_count=2,
        worker_threads=1,
    )
    assert [item[0] for item in results] == [0, 1, 2]
    assert all(item[1] == "1" for item in results)
    assert all(set(item[2]) == {"1"} for item in results)


def test_parent_environment_is_restored() -> None:
    before = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    run_spawn_tasks([ProbeTask(0)], runtime_probe, worker_count=1, worker_threads=1)
    assert {name: os.environ.get(name) for name in THREAD_ENVIRONMENT} == before


def test_worker_sentinel_rejects_nested_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POROUS_FILM_WORKER", "1")
    with pytest.raises(RuntimeError, match="nested process pools are forbidden"):
        ensure_pool_allowed()


def test_parallel_pool_error_str_omits_completed_results() -> None:
    large_result = "x" * 1000
    exc = ParallelPoolError("worker died", (large_result,))

    assert str(exc) == "worker died"
    assert exc.args == ("worker died",)
    assert exc.completed_results == (large_result,)


def test_geometry_acceptance_error_propagates_through_numeric_thread_limits() -> None:
    warnings = ["candidate-0", "candidate-1"]

    with pytest.raises(GeometryAcceptanceError) as exc_info, numeric_thread_limits(1):
        raise GeometryAcceptanceError("no_geometry_candidates", warnings)

    exc = exc_info.value
    assert exc.reason == "no_geometry_candidates"
    assert exc.warnings == warnings
    assert str(exc) == "no_geometry_candidates"
    assert exc.args == ("no_geometry_candidates",)
    assert exc.__traceback__ is not None


def test_bare_caller_owned_executor_is_rejected() -> None:
    executor = RecordingExecutor()

    with pytest.raises(ValueError, match="executor must be created by spawn_pool"):
        run_spawn_tasks(
            [ProbeTask(0)],
            echo_sequence,
            worker_count=1,
            worker_threads=1,
            executor=executor,
        )

    assert executor.submitted_sequence_indices == []


def test_spawn_pool_executor_can_be_reused_by_caller() -> None:
    with spawn_pool(worker_count=2, worker_threads=1) as executor:
        assert executor is not None
        first = run_spawn_tasks(
            [ProbeTask(2), ProbeTask(0), ProbeTask(1)],
            runtime_probe,
            worker_count=2,
            worker_threads=1,
            executor=executor,
        )
        second = run_spawn_tasks(
            [ProbeTask(3)],
            runtime_probe,
            worker_count=2,
            worker_threads=1,
            executor=executor,
        )

    assert [item[0] for item in first] == [0, 1, 2]
    assert [item[0] for item in second] == [3]
    assert all(item[1] == "1" for item in first + second)


def test_broken_pool_error_keeps_completed_results_ordered() -> None:
    callbacks: list[str] = []
    with (
        spawn_pool(worker_count=3, worker_threads=1) as executor,
        pytest.raises(ParallelPoolError) as exc_info,
    ):
        assert executor is not None
        run_spawn_tasks(
            [ProbeTask(2), ProbeTask(0), ProbeTask(1)],
            pool_error_probe,
            worker_count=3,
            worker_threads=1,
            executor=executor,
            result_callback=callbacks.append,
        )

    assert exc_info.value.completed_results == ("done-0", "done-2")
    assert exc_info.value.started_task_count == 3
    assert sorted(callbacks) == ["done-0", "done-2"]
    assert "worker died" in exc_info.value.message


def test_keyboard_interrupt_keeps_completed_results_ordered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callbacks: list[str] = []
    monkeypatch.setenv("POROUS_FILM_INTERRUPT_MARKER_DIR", str(tmp_path))

    def record_completed(result: str) -> None:
        callbacks.append(result)
        (tmp_path / result).touch()

    with (
        spawn_pool(worker_count=3, worker_threads=1) as executor,
        pytest.raises(ParallelCancelled) as exc_info,
    ):
        assert executor is not None
        run_spawn_tasks(
            [ProbeTask(2), ProbeTask(0), ProbeTask(1)],
            interrupt_probe,
            worker_count=3,
            worker_threads=1,
            executor=executor,
            result_callback=record_completed,
        )

    assert exc_info.value.completed_results == ("done-0", "done-2")
    assert exc_info.value.started_task_count == 3
    assert sorted(callbacks) == ["done-0", "done-2"]


def test_abrupt_worker_exit_keeps_completed_results_ordered() -> None:
    callbacks: list[str] = []
    with (
        spawn_pool(worker_count=3, worker_threads=1) as executor,
        pytest.raises(ParallelPoolError) as exc_info,
    ):
        assert executor is not None
        run_spawn_tasks(
            [ProbeTask(2), ProbeTask(0), ProbeTask(1)],
            abrupt_exit_probe,
            worker_count=3,
            worker_threads=1,
            executor=executor,
            result_callback=callbacks.append,
        )

    assert exc_info.value.completed_results == ("done-0", "done-2")
    assert exc_info.value.started_task_count == 3
    assert sorted(callbacks) == ["done-0", "done-2"]
