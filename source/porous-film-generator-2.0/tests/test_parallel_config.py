from pathlib import Path

import pytest

from porous_film.config import GeneratorConfig, load_config
from porous_film.parallel import apply_parallel_cli_overrides


def test_parallel_defaults(sample_config_path: Path) -> None:
    parallel = load_config(sample_config_path).parallel
    assert parallel.model_dump() == {
        "enabled": True,
        "strategy": "auto",
        "max_workers": None,
        "cpu_fraction": 0.80,
        "memory_fraction": 0.75,
        "worker_threads": 1,
        "start_method": "spawn",
        "sources": {"enabled": "config", "strategy": "config", "max_workers": "config"},
    }


@pytest.mark.parametrize("value", [0, -1])
def test_max_workers_must_be_positive(sample_config_path: Path, value: int) -> None:
    data = load_config(sample_config_path).model_dump(mode="json")
    data["parallel"] = {"max_workers": value}
    with pytest.raises(ValueError, match="greater than 0"):
        GeneratorConfig.model_validate(data)


@pytest.mark.parametrize(
    "parallel",
    [
        {"cpu_fraction": 0},
        {"memory_fraction": 1.1},
        {"worker_threads": 2},
        {"start_method": "fork"},
        {"strategy": "threads"},
    ],
)
def test_parallel_rejects_unsupported_values(
    sample_config_path: Path, parallel: dict[str, object]
) -> None:
    data = load_config(sample_config_path).model_dump(mode="json")
    data["parallel"] = parallel
    with pytest.raises(ValueError):
        GeneratorConfig.model_validate(data)


def test_workers_override_enables_and_records_source(sample_config_path: Path) -> None:
    config = apply_parallel_cli_overrides(
        load_config(sample_config_path), workers=6, no_parallel=False
    )
    assert config.parallel.enabled is True
    assert config.parallel.max_workers == 6
    assert config.parallel.sources.enabled == "cli:--workers"
    assert config.parallel.sources.max_workers == "cli:--workers"


def test_no_parallel_forces_serial(sample_config_path: Path) -> None:
    config = apply_parallel_cli_overrides(
        load_config(sample_config_path), workers=None, no_parallel=True
    )
    assert config.parallel.enabled is False
    assert config.parallel.strategy == "serial"
    assert config.parallel.sources.enabled == "cli:--no-parallel"
    assert config.parallel.sources.strategy == "cli:--no-parallel"


def test_overrides_are_mutually_exclusive(sample_config_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be used together"):
        apply_parallel_cli_overrides(
            load_config(sample_config_path), workers=2, no_parallel=True
        )


def test_schema_v3_canonical_payload_round_trips_through_worker_validation() -> None:
    from test_pipeline import _pipeline_v3_config_dict

    from porous_film.parallel.payloads import canonical_config_payload

    config = GeneratorConfig.model_validate(_pipeline_v3_config_dict())
    payload = canonical_config_payload(config)
    restored = GeneratorConfig.model_validate(payload)

    assert restored.source_schema_version == 3
    assert restored.formal_targets == config.formal_targets
    assert restored.film.target_box_A == config.film.target_box_A
    assert payload["film"] == {
        "target_box_A": {"x": 10.0, "y": 10.0, "z": 10.0}
    }
