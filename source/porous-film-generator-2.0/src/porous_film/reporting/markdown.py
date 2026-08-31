from __future__ import annotations

from pathlib import Path
from typing import Any

from porous_film import __version__
from porous_film.config import GeneratorConfig


def write_preflight_report(
    path: Path,
    *,
    config: GeneratorConfig,
    passed: bool,
    errors: list[str],
    warnings: list[str],
    estimated_voxels: int,
    estimated_memory_bytes: int,
) -> Path:
    return _write_report(
        path,
        title="Porous-film preflight report",
        config=config,
        status="passed" if passed else "failed",
        warnings=warnings,
        extra=[
            ("Estimated voxels", str(estimated_voxels)),
            ("Estimated memory bytes", str(estimated_memory_bytes)),
            ("Errors", ", ".join(errors) if errors else "none"),
        ],
    )


def write_run_report(
    path: Path,
    *,
    config: GeneratorConfig,
    status: str,
    warnings: list[str],
    convergence: dict[str, Any],
    output_paths: list[Path],
) -> Path:
    return _write_report(
        path,
        title="Porous-film generation report",
        config=config,
        status=status,
        warnings=warnings,
        extra=[
            ("Convergence", _format_mapping(convergence)),
            ("Output paths", "\n".join(f"- {item.resolve()}" for item in output_paths)),
        ],
    )


def _write_report(
    path: Path,
    *,
    title: str,
    config: GeneratorConfig,
    status: str,
    warnings: list[str],
    extra: list[tuple[str, str]],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        f"- Software version: porous-film {__version__}",
        f"- Task: {config.task.name}",
        f"- Status: {status}",
        f"- Target box (A): {config.film.target_box_A.model_dump()}",
        f"- Packing box (A): {config.film.packing_box_A.model_dump() if config.film.packing_box_A else None}",
        f"- Target porosity: {config.pores.target_porosity}",
        f"- Seed count: {config.seed_count}",
        "",
        "## Parameters",
        "",
        "```yaml",
        config.model_dump_json(indent=2),
        "```",
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {warning}" for warning in warnings] or ["- none"])
    for heading, body in extra:
        lines.extend(["", f"## {heading}", "", body])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _format_mapping(mapping: dict[str, Any]) -> str:
    if not mapping:
        return "none"
    return "\n".join(f"- {key}: {value}" for key, value in mapping.items())
