from __future__ import annotations

from porous_film.config import GeneratorConfig


def apply_parallel_cli_overrides(
    config: GeneratorConfig,
    *,
    workers: int | None,
    no_parallel: bool,
) -> GeneratorConfig:
    if workers is not None and workers < 1:
        raise ValueError("--workers must be a positive integer")
    if workers is not None and no_parallel:
        raise ValueError("--workers and --no-parallel cannot be used together")
    updates: dict[str, object] = {}
    source_updates: dict[str, str] = {}
    if workers is not None:
        updates.update(enabled=True, max_workers=int(workers))
        source_updates.update(enabled="cli:--workers", max_workers="cli:--workers")
    if no_parallel:
        updates.update(enabled=False, strategy="serial")
        source_updates.update(enabled="cli:--no-parallel", strategy="cli:--no-parallel")
    if not updates:
        return config
    sources = config.parallel.sources.model_copy(update=source_updates)
    parallel = config.parallel.model_copy(update={**updates, "sources": sources})
    return config.model_copy(update={"parallel": parallel})
