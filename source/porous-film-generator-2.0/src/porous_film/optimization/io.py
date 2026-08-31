from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_JSON_FILES = {
    "requested": "requested_design_parameters.json",
    "realized": "realized_geometry_parameters.json",
    "feasibility": "feasibility.json",
    "calculation_status": "calculation_status.json",
    "objectives": "objectives.json",
    "uncertainty": "uncertainty.json",
}

OptimizerPayload = dict[str, object]


def write_optimizer_exchange(
    output_dir: Path,
    requested: dict,
    realized: dict,
    feasible: bool,
    constraints: dict,
    calculation_status: dict | None = None,
    objectives: dict | None = None,
    uncertainty: dict | None = None,
) -> None:
    """Write optimizer-facing JSON files with requested and realized values separated."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    status = {"status": "completed" if feasible else "failed"}
    if calculation_status:
        status.update(calculation_status)
    payloads = {
        _JSON_FILES["requested"]: requested,
        _JSON_FILES["realized"]: realized,
        _JSON_FILES["feasibility"]: {
            "feasible": bool(feasible),
            "constraints": constraints,
        },
        _JSON_FILES["calculation_status"]: status,
        _JSON_FILES["objectives"]: objectives or {},
        _JSON_FILES["uncertainty"]: uncertainty or {},
    }
    for filename, payload in payloads.items():
        _write_json(directory / filename, payload)


def aggregate_seed_results(seed_results: list[dict]) -> dict:
    """Aggregate deterministic seed-panel records for optimizer noise accounting."""
    feasible_results = [result for result in seed_results if bool(result.get("feasible"))]
    objective_values = [
        float(result["objective"])
        for result in feasible_results
        if result.get("objective") is not None
    ]
    failures = {
        str(result.get("seed")): str(result.get("failure_reason", "unknown_failure"))
        for result in seed_results
        if not bool(result.get("feasible"))
    }
    if objective_values:
        objectives = np.asarray(objective_values, dtype=float)
        objective_mean = float(np.mean(objectives))
        objective_variance = float(np.var(objectives))
    else:
        objective_mean = None
        objective_variance = None
    seed_count = len(seed_results)
    feasible_count = len(feasible_results)
    return {
        "seed_count": seed_count,
        "feasible_count": feasible_count,
        "objective_mean": objective_mean,
        "objective_variance": objective_variance,
        "feasible_fraction": feasible_count / seed_count if seed_count else 0.0,
        "failures": failures,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, separators=(",", ": "))
        + "\n",
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value
