import json
from pathlib import Path

from porous_film.optimization import write_optimizer_exchange


def test_seed_panel_aggregation_reports_noise_and_failures() -> None:
    from porous_film.optimization import aggregate_seed_results

    summary = aggregate_seed_results(
        [
            {"seed": 11, "feasible": True, "objective": 2.0},
            {"seed": 22, "feasible": True, "objective": 4.0},
            {"seed": 33, "feasible": False, "failure_reason": "x_not_percolating"},
        ]
    )

    assert summary["objective_mean"] == 3.0
    assert summary["objective_variance"] == 1.0
    assert summary["feasible_fraction"] == 2 / 3
    assert summary["failures"] == {"33": "x_not_percolating"}


def test_optimizer_exchange_separates_requested_and_realized(tmp_path: Path) -> None:
    write_optimizer_exchange(
        tmp_path,
        requested={"target_porosity": 0.25},
        realized={"porosity": 0.247},
        feasible=True,
        constraints={"x_percolates": True},
        objectives={},
        uncertainty={"geometry_replicate_variance": None},
    )

    requested = json.loads((tmp_path / "requested_design_parameters.json").read_text())
    realized = json.loads((tmp_path / "realized_geometry_parameters.json").read_text())

    assert requested["target_porosity"] == 0.25
    assert realized["porosity"] == 0.247


def test_optimizer_exchange_writes_machine_readable_failure_reason(tmp_path: Path) -> None:
    write_optimizer_exchange(
        tmp_path,
        requested={"target_porosity": 0.40},
        realized={},
        feasible=False,
        constraints={"x_percolates": False},
        calculation_status={"status": "failed", "failure_reason": "x_not_percolating"},
        objectives={"score": None},
        uncertainty={"geometry_replicate_variance": None},
    )

    feasibility = json.loads((tmp_path / "feasibility.json").read_text(encoding="utf-8"))
    status = json.loads((tmp_path / "calculation_status.json").read_text(encoding="utf-8"))

    assert feasibility == {"feasible": False, "constraints": {"x_percolates": False}}
    assert status["failure_reason"] == "x_not_percolating"
