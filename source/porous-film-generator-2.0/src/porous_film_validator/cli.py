from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer

from porous_film_validator.validate import validate_export

app = typer.Typer(no_args_is_help=True)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    qa_export: Annotated[
        Path | None,
        typer.Argument(help="Path to a generated qa_export directory."),
    ] = None,
) -> None:
    """Validate porous-film QA exports independently of the generator package."""
    if ctx.invoked_subcommand is not None:
        return
    if qa_export is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)

    started_at = time.perf_counter()
    report = validate_export(qa_export)
    validation_seconds = max(0.0, time.perf_counter() - started_at)
    typer.echo(report.status)
    typer.echo(f"Validation time: {validation_seconds:.2f} s")
    if report.errors:
        for error in report.errors:
            typer.echo(f"ERROR: {error}", err=True)
    if report.warnings:
        for warning in report.warnings:
            typer.echo(f"WARNING: {warning}", err=True)
    if report.status in {"FAIL", "NOT_EVALUABLE"}:
        raise typer.Exit(1)
