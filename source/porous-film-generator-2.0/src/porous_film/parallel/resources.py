from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil

from porous_film.config import GeneratorConfig

_MIN_WORKER_MEMORY = 256 * 1024**2


@dataclass(frozen=True)
class ResourceSnapshot:
    platform_name: str
    allowed_logical_cpus: tuple[int, ...]
    allowed_logical_count: int
    available_physical_cores: int
    physical_core_source: str
    available_memory_bytes: int
    effective_memory_bytes: int
    audit_memory_cap_bytes: int | None
    warnings: tuple[str, ...] = ()

    @classmethod
    def for_test(cls, physical_cores: int, memory_bytes: int) -> ResourceSnapshot:
        cpus = tuple(range(physical_cores))
        return cls(
            platform_name="test",
            allowed_logical_cpus=cpus,
            allowed_logical_count=physical_cores,
            available_physical_cores=physical_cores,
            physical_core_source="test",
            available_memory_bytes=memory_bytes,
            effective_memory_bytes=memory_bytes,
            audit_memory_cap_bytes=None,
        )


def _parse_cpu_list(value: str) -> set[int]:
    result: set[int] = set()
    for token in value.strip().split(","):
        if not token:
            continue
        if "-" in token:
            lower, upper = (int(item) for item in token.split("-", 1))
            result.update(range(lower, upper + 1))
        else:
            result.add(int(token))
    return result


def _allowed_logical_cpus() -> set[int]:
    if hasattr(os, "sched_getaffinity"):
        return set(os.sched_getaffinity(0))
    process = psutil.Process()
    if hasattr(process, "cpu_affinity"):
        affinity = process.cpu_affinity()
        if affinity:
            return set(affinity)
    return set(range(psutil.cpu_count(logical=True) or 1))


def discover_resources(
    *,
    audit_memory_cap_bytes: int | None = None,
    affinity: set[int] | None = None,
    available_memory_bytes: int | None = None,
    platform_name: str | None = None,
    sysfs_cpu_root: Path = Path("/sys/devices/system/cpu"),
) -> ResourceSnapshot:
    system = (platform_name or platform.system()).lower()
    allowed = set(affinity) if affinity is not None else _allowed_logical_cpus()
    warnings: list[str] = []
    if system == "linux":
        groups: set[tuple[int, ...]] = set()
        for cpu in sorted(allowed):
            path = sysfs_cpu_root / f"cpu{cpu}" / "topology" / "thread_siblings_list"
            try:
                groups.add(tuple(sorted(_parse_cpu_list(path.read_text(encoding="utf-8")))))
            except (OSError, ValueError):
                groups.clear()
                break
        physical = len(groups) if groups else len(allowed)
        source = "linux-thread-siblings" if groups else "allowed-logical-fallback"
        if not groups:
            warnings.append("physical CPU topology unavailable; allowed logical CPUs used")
    else:
        physical = min(psutil.cpu_count(logical=False) or len(allowed), len(allowed))
        source = "psutil-physical-clamped-to-affinity"
    available = int(
        available_memory_bytes
        if available_memory_bytes is not None
        else psutil.virtual_memory().available
    )
    effective = min(available, audit_memory_cap_bytes) if audit_memory_cap_bytes else available
    return ResourceSnapshot(
        platform_name=system,
        allowed_logical_cpus=tuple(sorted(allowed)),
        allowed_logical_count=len(allowed),
        available_physical_cores=max(1, physical),
        physical_core_source=source,
        available_memory_bytes=available,
        effective_memory_bytes=effective,
        audit_memory_cap_bytes=audit_memory_cap_bytes,
        warnings=tuple(warnings),
    )


def estimate_generation_memory_bytes(config: GeneratorConfig) -> int:
    box = np.array(
        [
            config.film.target_box_A.x,
            config.film.target_box_A.y,
            config.film.target_box_A.z,
        ]
    )
    fine = int(np.prod(np.ceil(box / config.audit.fine_spacing_A).astype(np.int64)))
    coarse = int(np.prod(np.ceil(box / config.audit.coarse_spacing_A).astype(np.int64)))
    return int((5 * fine + coarse) * np.dtype(np.float64).itemsize)


def estimate_worker_memory_bytes(config: GeneratorConfig) -> int:
    return max(_MIN_WORKER_MEMORY, estimate_generation_memory_bytes(config))
