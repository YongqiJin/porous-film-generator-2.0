from __future__ import annotations

import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import psutil

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on Windows
    resource = None


@dataclass(frozen=True)
class PerformanceSnapshot:
    stage_seconds: dict[str, float]
    stage_inclusive_seconds: dict[str, float]
    stage_call_counts: dict[str, int]
    wall_time_seconds: float
    rss_start_mib: float
    peak_rss_mib: float


@dataclass
class _ActiveStage:
    name: str
    started_at: float
    child_seconds: float = 0.0


class RuntimeProfiler:
    """Accumulate low-overhead wall-clock timings for nested pipeline stages."""

    def __init__(self) -> None:
        self._started_at = time.perf_counter()
        self._stage_seconds: defaultdict[str, float] = defaultdict(float)
        self._stage_inclusive_seconds: defaultdict[str, float] = defaultdict(float)
        self._stage_call_counts: defaultdict[str, int] = defaultdict(int)
        self._stack: list[_ActiveStage] = []
        self._rss_start_mib = _current_rss_mib()
        self._peak_rss_mib = max(self._rss_start_mib, _process_peak_rss_mib())

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        frame = _ActiveStage(name=str(name), started_at=time.perf_counter())
        self._stack.append(frame)
        try:
            yield
        finally:
            elapsed = max(0.0, time.perf_counter() - frame.started_at)
            active = self._stack.pop()
            if active is not frame:
                raise RuntimeError("runtime profiler stage stack is inconsistent")
            exclusive = max(0.0, elapsed - frame.child_seconds)
            self._stage_seconds[frame.name] += exclusive
            self._stage_inclusive_seconds[frame.name] += elapsed
            self._stage_call_counts[frame.name] += 1
            if self._stack:
                self._stack[-1].child_seconds += elapsed
            self._sample_memory()

    def snapshot(self) -> PerformanceSnapshot:
        self._sample_memory()
        return PerformanceSnapshot(
            stage_seconds=dict(sorted(self._stage_seconds.items())),
            stage_inclusive_seconds=dict(sorted(self._stage_inclusive_seconds.items())),
            stage_call_counts=dict(sorted(self._stage_call_counts.items())),
            wall_time_seconds=max(0.0, time.perf_counter() - self._started_at),
            rss_start_mib=float(self._rss_start_mib),
            peak_rss_mib=float(self._peak_rss_mib),
        )

    def _sample_memory(self) -> None:
        self._peak_rss_mib = max(
            self._peak_rss_mib,
            _current_rss_mib(),
            _process_peak_rss_mib(),
        )


_ACTIVE_PROFILER: ContextVar[RuntimeProfiler | None] = ContextVar(
    "porous_film_runtime_profiler",
    default=None,
)


@contextmanager
def activate_runtime_profiler(profiler: RuntimeProfiler) -> Iterator[RuntimeProfiler]:
    token = _ACTIVE_PROFILER.set(profiler)
    try:
        yield profiler
    finally:
        _ACTIVE_PROFILER.reset(token)


@contextmanager
def profile_stage(name: str) -> Iterator[None]:
    profiler = _ACTIVE_PROFILER.get()
    if profiler is None:
        yield
        return
    with profiler.stage(name):
        yield


def _current_rss_mib() -> float:
    return float(psutil.Process().memory_info().rss / (1024.0**2))


def _process_peak_rss_mib() -> float:
    if resource is None:
        return _current_rss_mib()
    maximum_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return maximum_rss / (1024.0**2)
    return maximum_rss / 1024.0


__all__ = [
    "PerformanceSnapshot",
    "RuntimeProfiler",
    "activate_runtime_profiler",
    "profile_stage",
]
