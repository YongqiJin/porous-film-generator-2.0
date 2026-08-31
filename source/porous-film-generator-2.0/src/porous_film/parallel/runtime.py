from __future__ import annotations

import logging
import multiprocessing
import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from threadpoolctl import threadpool_limits

THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

_LOGGER = logging.getLogger(__name__)
_WORKER_THREADPOOL_LIMITS: Any | None = None
_SPAWN_POOL_MARKER = "_porous_film_spawn_pool_executor"


class SequencedTask(Protocol):
    sequence_index: int


@dataclass
class ParallelCancelled(Exception):
    completed_results: tuple[object, ...]
    started_task_count: int = 0


@dataclass
class ParallelPoolError(RuntimeError):
    message: str
    completed_results: tuple[object, ...]
    started_task_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", (self.message,))


@contextmanager
def numeric_thread_limits(worker_threads: int) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    try:
        for name in THREAD_ENVIRONMENT:
            os.environ[name] = str(worker_threads)
        with threadpool_limits(limits=worker_threads):
            yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def initialize_worker(worker_threads: int) -> None:
    global _WORKER_THREADPOOL_LIMITS
    os.environ["POROUS_FILM_WORKER"] = "1"
    for name in THREAD_ENVIRONMENT:
        os.environ[name] = str(worker_threads)
    _WORKER_THREADPOOL_LIMITS = threadpool_limits(limits=worker_threads)


def ensure_pool_allowed() -> None:
    if os.environ.get("POROUS_FILM_WORKER") == "1":
        raise RuntimeError("nested process pools are forbidden")


@contextmanager
def spawn_pool(
    worker_count: int, worker_threads: int
) -> Iterator[ProcessPoolExecutor | None]:
    """Create the only supported reusable executor for `run_spawn_tasks`.

    A multi-worker pool uses Windows-safe spawn semantics, initializes every
    worker with the porous-film worker sentinel and numeric thread limits, and
    keeps parent thread-limit environment changes scoped to this context.
    """
    if worker_count == 1:
        with numeric_thread_limits(worker_threads):
            yield None
        return

    ensure_pool_allowed()
    with numeric_thread_limits(worker_threads):
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(worker_threads,),
        )
        setattr(executor, _SPAWN_POOL_MARKER, True)
        try:
            yield executor
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


def run_spawn_tasks[TaskT: SequencedTask, ResultT](
    tasks: Iterable[TaskT],
    worker: Callable[[TaskT], ResultT],
    *,
    worker_count: int,
    worker_threads: int,
    executor: Executor | None = None,
    result_callback: Callable[[ResultT], None] | None = None,
) -> list[ResultT]:
    """Run sequenced tasks and return results ordered by `sequence_index`.

    When `executor` is supplied, it must be the caller-owned executor yielded by
    `spawn_pool`. This preserves one-pool reuse while ensuring the executor has
    already received the required spawn context, worker initialization, nested
    pool protection, and parent numeric-thread-limit scope.
    """
    sorted_tasks = sorted(tasks, key=lambda task: task.sequence_index)
    if executor is not None:
        _ensure_spawn_pool_executor(executor)
        return _run_on_executor(sorted_tasks, worker, executor, result_callback)

    with spawn_pool(worker_count, worker_threads) as owned_executor:
        if owned_executor is None:
            return _run_in_parent(sorted_tasks, worker, result_callback)
        return _run_on_executor(sorted_tasks, worker, owned_executor, result_callback)


def _ensure_spawn_pool_executor(executor: Executor) -> None:
    ensure_pool_allowed()
    if not getattr(executor, _SPAWN_POOL_MARKER, False):
        raise ValueError("executor must be created by spawn_pool")


def _run_in_parent[TaskT: SequencedTask, ResultT](
    tasks: list[TaskT],
    worker: Callable[[TaskT], ResultT],
    result_callback: Callable[[ResultT], None] | None,
) -> list[ResultT]:
    completed: dict[int, ResultT] = {}
    started = 0
    try:
        for task in tasks:
            started += 1
            result = worker(task)
            completed[task.sequence_index] = result
            if result_callback is not None:
                result_callback(result)
    except KeyboardInterrupt as exc:
        raise ParallelCancelled(_ordered_results(completed), started) from exc
    except BrokenProcessPool as exc:
        raise ParallelPoolError(str(exc), _ordered_results(completed), started) from exc
    return list(_ordered_results(completed))


def _run_on_executor[TaskT: SequencedTask, ResultT](
    tasks: list[TaskT],
    worker: Callable[[TaskT], ResultT],
    executor: Executor,
    result_callback: Callable[[ResultT], None] | None,
) -> list[ResultT]:
    futures: dict[Future[ResultT], int] = {}
    completed: dict[int, ResultT] = {}
    submitted = 0
    try:
        for task in tasks:
            futures[executor.submit(worker, task)] = task.sequence_index
            submitted += 1
        for future in as_completed(futures):
            _record_completed_future(future, futures, completed, result_callback)
    except KeyboardInterrupt as exc:
        _collect_ready_successes(futures, completed, result_callback)
        _cancel_pending(futures)
        raise ParallelCancelled(_ordered_results(completed), submitted) from exc
    except BrokenProcessPool as exc:
        _collect_ready_successes(futures, completed, result_callback)
        _cancel_pending(futures)
        raise ParallelPoolError(str(exc), _ordered_results(completed), submitted) from exc
    return list(_ordered_results(completed))


def _record_completed_future[ResultT](
    future: Future[ResultT],
    futures: dict[Future[ResultT], int],
    completed: dict[int, ResultT],
    result_callback: Callable[[ResultT], None] | None,
) -> None:
    sequence_index = futures[future]
    if sequence_index in completed:
        return
    result = future.result()
    completed[sequence_index] = result
    if result_callback is not None:
        result_callback(result)


def _collect_ready_successes[ResultT](
    futures: dict[Future[ResultT], int],
    completed: dict[int, ResultT],
    result_callback: Callable[[ResultT], None] | None,
) -> None:
    for future, sequence_index in sorted(futures.items(), key=lambda item: item[1]):
        if sequence_index in completed or not future.done() or future.cancelled():
            continue
        try:
            _record_completed_future(future, futures, completed, result_callback)
        except BaseException:
            _LOGGER.debug("skipping unsuccessful future during interruption harvest", exc_info=True)


def _cancel_pending[ResultT](futures: dict[Future[ResultT], int]) -> None:
    for future in futures:
        if not future.done():
            future.cancel()


def _ordered_results[ResultT](results: dict[int, ResultT]) -> tuple[ResultT, ...]:
    return tuple(result for _, result in sorted(results.items()))
