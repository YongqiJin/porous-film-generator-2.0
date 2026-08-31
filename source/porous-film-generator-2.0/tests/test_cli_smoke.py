from typer.testing import CliRunner

from porous_film.cli import app

runner = CliRunner()


def test_version_command_reports_package_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "porous-film 0.4.0.dev1"


def test_generate_help_exposes_parallel_options() -> None:
    result = runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0
    assert "--workers" in result.stdout
    assert "--no-parallel" in result.stdout


def test_generate_geometry_help_exposes_parallel_options() -> None:
    result = runner.invoke(app, ["generate-geometry", "--help"])
    assert result.exit_code == 0
    assert "--workers" in result.stdout
    assert "--no-parallel" in result.stdout
