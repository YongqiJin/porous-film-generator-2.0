from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

import typer

from porous_film import __version__
from porous_film.config import GeneratorConfig, load_config
from porous_film.parallel import apply_parallel_cli_overrides
from porous_film.pipeline import (
    GeometryRun,
    audit_packmol_output_run,
    audit_run,
    fill_pore_run,
    generate_geometry,
    preflight,
    prepare_task_inputs,
    run_full,
)
from porous_film.storage import create_task_directory

app = typer.Typer(no_args_is_help=True)
_SHANGHAI = ZoneInfo("Asia/Shanghai")


@app.callback()
def main() -> None:
    pass


@app.command()
def version() -> None:
    typer.echo(f"porous-film {__version__}")


def _load_config_with_parallel_overrides(
    config: Path,
    *,
    workers: int | None,
    no_parallel: bool,
) -> GeneratorConfig:
    loaded = load_config(config)
    try:
        return apply_parallel_cli_overrides(
            loaded,
            workers=workers,
            no_parallel=no_parallel,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _echo_geometry_summary(result: GeometryRun) -> None:
    performance = result.performance or {}
    runtime_seconds = float(performance["total_wall_time_seconds"])
    typer.echo(f"Result directory: {result.paths.root.resolve()}")
    typer.echo(f"Runtime: {runtime_seconds:.2f} s")
    typer.echo(f"Audit result: {'PASS' if result.audit.passed else 'FAIL'}")


@app.command("preflight")
def preflight_cmd(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    result_root: Annotated[
        Path,
        typer.Option("--result-root", help="Root for dated calculation output directories."),
    ] = Path("C:/Calculation_results"),
) -> None:
    result = preflight(load_config(config), result_root)
    if result.report_path is not None:
        typer.echo(str(result.report_path.resolve()))
    if not result.passed:
        raise typer.Exit(1)

@app.command("generate-geometry")
def generate_geometry_cmd(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    result_root: Annotated[
        Path,
        typer.Option("--result-root", help="Root for dated calculation output directories."),
    ] = Path("C:/Calculation_results"),
    workers: Annotated[
        int | None,
        typer.Option("--workers", min=1, help="Maximum worker processes."),
    ] = None,
    no_parallel: Annotated[
        bool,
        typer.Option("--no-parallel", help="Force serial execution."),
    ] = False,
) -> None:
    loaded = _load_config_with_parallel_overrides(
        config,
        workers=workers,
        no_parallel=no_parallel,
    )
    paths = create_task_directory(result_root, loaded.task.name, datetime.now(_SHANGHAI))
    prepare_task_inputs(loaded, paths, config_path=config)
    result = generate_geometry(loaded, paths)
    _echo_geometry_summary(result)


@app.command("fill-pore")
def fill_pore_cmd(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
) -> None:
    typer.echo(str(fill_pore_run(run).resolve()))


@app.command("generate")
def generate_cmd(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    result_root: Annotated[
        Path,
        typer.Option("--result-root", help="Root for dated calculation output directories."),
    ] = Path("C:/Calculation_results"),
    workers: Annotated[
        int | None,
        typer.Option("--workers", min=1, help="Maximum worker processes."),
    ] = None,
    no_parallel: Annotated[
        bool,
        typer.Option("--no-parallel", help="Force serial execution."),
    ] = False,
) -> None:
    loaded = _load_config_with_parallel_overrides(
        config,
        workers=workers,
        no_parallel=no_parallel,
    )
    result = run_full(loaded, result_root, config_path=config)
    typer.echo(str(result.paths.root.resolve()))


@app.command("audit")
def audit_cmd(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
) -> None:
    typer.echo(str(audit_run(run).resolve()))


@app.command("audit-packmol-output")
def audit_packmol_output_cmd(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    structure: Annotated[Path, typer.Option("--structure", exists=True, dir_okay=False)],
) -> None:
    output = audit_packmol_output_run(run, structure)
    typer.echo(str(output.resolve()))
    audit = json.loads(output.read_text(encoding="utf-8"))
    if not bool(audit.get("passed")):
        raise typer.Exit(1)
