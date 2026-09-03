import json
import struct
import sys
from importlib import util
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from conftest import write_config
from typer.testing import CliRunner

from porous_film.cli import app

runner = CliRunner()


def test_preflight_creates_report(sample_config_path: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["preflight", "--config", str(sample_config_path), "--result-root", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "preflight-report.md" in result.stdout
    assert not (Path("C:/Calculation_results") / "preflight-report.md").exists()


def test_missing_pdb_preflight_returns_failed_result_and_report(tmp_path: Path) -> None:
    from porous_film.config import load_config
    from porous_film.pipeline import preflight

    missing = tmp_path / "missing.pdb"
    config_path = write_config(tmp_path, name="missing-pdb", pdb_path=missing)

    result = preflight(load_config(config_path), tmp_path)
    cli_result = runner.invoke(
        app,
        ["preflight", "--config", str(config_path), "--result-root", str(tmp_path)],
    )

    assert result.passed is False
    assert result.report_path is not None
    assert result.report_path.name == "preflight-report.md"
    assert "pore material PDB is invalid" in result.report_path.read_text(encoding="utf-8")
    assert not list(result.report_path.parents[1].joinpath("inputs").glob("missing.pdb"))
    assert cli_result.exit_code == 1
    assert "FileNotFoundError" not in cli_result.stdout


def test_preflight_rejects_target_box_not_divisible_by_audit_spacing(tmp_path: Path) -> None:
    from porous_film.config import load_config
    from porous_film.pipeline import preflight

    config_path = write_config(tmp_path, name="nondivisible-spacing")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["film"]["target_box_A"]["x"] = 20.25
    data["film"]["packing_box_A"]["x"] = 20.25
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = preflight(load_config(config_path), tmp_path)

    assert result.passed is False
    assert any("divisible" in error for error in result.errors)


def test_preflight_fails_when_estimated_memory_exceeds_configured_cap(tmp_path: Path) -> None:
    from porous_film.config import load_config
    from porous_film.pipeline import preflight

    config_path = write_config(tmp_path, name="memory-cap")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["geometry_audit"]["available_memory_cap_bytes"] = 1
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = preflight(load_config(config_path), tmp_path)

    assert result.passed is False
    assert any("memory" in error for error in result.errors)


def test_run_full_writes_every_required_handoff_file(
    feasible_config_path: Path,
    tmp_path: Path,
) -> None:
    from porous_film.config import load_config
    from porous_film.pipeline import run_full

    result = run_full(load_config(feasible_config_path), tmp_path)
    expected = [
        result.paths.outputs / "semiconductor_solid_target.glb",
        result.paths.outputs / "pore_material.pdb",
        result.paths.outputs / "pore_material_high_precision.cif",
        result.paths.outputs / "molecule_instances.csv",
        result.paths.outputs / "pore_geometry.h5",
        result.paths.outputs / "packing_metrics.json",
        result.paths.outputs / "packmol_handoff.inp",
        result.paths.outputs / "pore_reference_coordinates.cif",
        result.paths.outputs / "phase_mapping.json",
        result.paths.outputs / "pore_atom_indices.ndx",
        result.paths.outputs / "compression_metadata.json",
    ]

    assert all(path.exists() for path in expected)


def test_run_full_writes_mandatory_qa_and_optimizer_files(
    feasible_config_path: Path,
    tmp_path: Path,
) -> None:
    from porous_film.config import load_config
    from porous_film.pipeline import run_full

    result = run_full(load_config(feasible_config_path), tmp_path)
    qa_expected = [
        result.paths.qa_export / "contract.json",
        result.paths.qa_export / "normalized_config.yaml",
        result.paths.qa_export / "unit_candidates.jsonl",
        result.paths.qa_export / "unit_geometry.jsonl",
        result.paths.qa_export / "channel_curves.h5",
        result.paths.qa_export / "final_phase.h5",
        result.paths.qa_export / "final_surface.ply",
        result.paths.qa_export / "main_unit_metrics.csv",
        result.paths.qa_export / "main_metrics.json",
        result.paths.qa_export / "molecules" / "source",
        result.paths.qa_export / "molecules" / "instances.csv",
        result.paths.qa_export / "molecules" / "placed_atoms.h5",
        result.paths.qa_export / "molecules" / "placed_structure.cif",
        result.paths.qa_export / "checksums.sha256",
    ]
    optimizer_expected = [
        result.paths.outputs / "requested_design_parameters.json",
        result.paths.outputs / "realized_geometry_parameters.json",
        result.paths.outputs / "feasibility.json",
        result.paths.outputs / "calculation_status.json",
        result.paths.outputs / "objectives.json",
        result.paths.outputs / "uncertainty.json",
    ]

    assert all(path.exists() for path in qa_expected + optimizer_expected)

    header = (
        (result.paths.outputs / "molecule_instances.csv")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert header.split(",") == [
        "instance_id",
        "source_hash",
        "residue_name",
        "translation_x_A",
        "translation_y_A",
        "translation_z_A",
        "quaternion_w",
        "quaternion_x",
        "quaternion_y",
        "quaternion_z",
        "periodic_image_x",
        "periodic_image_y",
    ]

    with h5py.File(result.paths.outputs / "pore_geometry.h5", "r") as handle:
        assert handle.attrs["schema_version"] == 2
        assert handle.attrs["final_phase_reference"] == "../qa_export/final_phase.h5"
        assert "unit_records" in handle
        assert "final_phase" in handle

    handoff = (result.paths.outputs / "packmol_handoff.inp").read_text(encoding="utf-8")
    assert "pore_material.pdb" in handoff
    assert "blocker" not in handoff.lower()
    assert "target_box_A" in handoff
    assert "packing_box_A" in handoff

    assert json.loads((result.paths.outputs / "phase_mapping.json").read_text()) == {
        "MATRIX": 0,
        "PORE": 1,
    }
    assert "[ PORE ]" in (result.paths.outputs / "pore_atom_indices.ndx").read_text()
    compression = json.loads((result.paths.outputs / "compression_metadata.json").read_text())
    assert compression["absolute_lock_mode"] == "target_origin_in_packing_A"
    assert compression["pore_coordinate_hash"]

    status = json.loads((result.paths.outputs / "calculation_status.json").read_text())
    realized = json.loads((result.paths.outputs / "realized_geometry_parameters.json").read_text())
    feasibility = json.loads((result.paths.outputs / "feasibility.json").read_text())
    report = (result.paths.reports / "generation-report.md").read_text(encoding="utf-8")

    assert result.status == "completed_feasible"
    assert status["status"] == "completed_feasible"
    assert feasibility["feasible"] is True
    assert feasibility["constraints"]["audit_passed"] is True
    assert realized["packed_molecule_count"] == 1
    assert "actual_density_g_cm3" in realized
    assert "shape_complexity_summary" in realized
    assert "Status: completed_feasible" in report


def test_pipeline_glb_raw_extras_use_task_name_and_run_identifier(
    feasible_config_path: Path,
    tmp_path: Path,
) -> None:
    from porous_film.config import load_config
    from porous_film.pipeline import run_full

    config = load_config(feasible_config_path)
    result = run_full(config, tmp_path)

    glb_json = _read_glb_json(result.paths.outputs / "semiconductor_solid_target.glb")
    extras = glb_json["scenes"][0]["extras"]

    assert extras["task_id"] == config.task.name
    assert extras["run_identifier"] == result.paths.root.name
    assert extras["task_id"] != "task-7"


def test_run_full_at_root_delegates_with_exact_task_paths_without_date_layer(
    sample_config_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    from porous_film.config import load_config
    from porous_film.pipeline import run_full_at_root

    captured = {}
    sentinel = object()

    def fake_run_full_with_paths(config, paths, *, config_path=None):
        captured["config"] = config
        captured["paths"] = paths
        captured["config_path"] = config_path
        return sentinel

    monkeypatch.setattr("porous_film.pipeline._run_full_with_paths", fake_run_full_with_paths)
    config = load_config(sample_config_path)
    target = tmp_path / "remote-policy-root"

    result = run_full_at_root(config, target, config_path=sample_config_path)

    assert result is sentinel
    assert captured["config"] is config
    assert captured["config_path"] == sample_config_path
    assert captured["paths"].root == target
    assert (target / "outputs").is_dir()
    assert not any(path.name == "python_results" for path in target.rglob("*"))


def test_optimizer_failure_payload_is_json_compact_and_uses_domain_reason(
    sample_config_path: Path,
) -> None:
    from porous_film.config import load_config
    from porous_film.molecules import PackingError
    from porous_film.pipeline import optimizer_payload_for_failure

    payload = optimizer_payload_for_failure(
        load_config(sample_config_path),
        PackingError("no_free_pore_volume", "could not pack molecule"),
    )

    json.dumps(payload)
    assert set(payload) == {
        "requested",
        "realized",
        "feasible",
        "constraints",
        "calculation_status",
        "objectives",
        "uncertainty",
    }
    assert payload["feasible"] is False
    assert payload["realized"] == {}
    assert payload["constraints"] == {"audit_passed": False, "packing_completed": False}
    assert payload["calculation_status"] == {
        "status": "failed",
        "failure_reason": "no_free_pore_volume",
        "message": "could not pack molecule",
        "exception_type": "PackingError",
    }
    assert payload["objectives"] == {}
    assert payload["uncertainty"] == {}


def test_run_full_at_root_uses_exact_directory(
    feasible_config_path: Path,
    tmp_path: Path,
) -> None:
    from porous_film.config import load_config
    from porous_film.pipeline import run_full_at_root

    target = tmp_path / "remote-policy-root"

    result = run_full_at_root(
        load_config(feasible_config_path),
        target,
        config_path=feasible_config_path,
    )

    assert result.paths.root == target
    assert (target / "outputs" / "calculation_status.json").is_file()
    assert not any(path.name == "python_results" for path in target.rglob("*"))


def test_write_qa_checksums_is_public_deterministic_and_excludes_itself(tmp_path: Path) -> None:
    from porous_film.io import write_qa_checksums

    qa = tmp_path / "qa"
    (qa / "nested").mkdir(parents=True)
    (qa / "b.txt").write_text("second", encoding="utf-8")
    (qa / "nested" / "a.txt").write_text("first", encoding="utf-8")
    (qa / "checksums.sha256").write_text("stale\n", encoding="utf-8")

    first = write_qa_checksums(qa).read_text(encoding="utf-8")
    second = write_qa_checksums(qa).read_text(encoding="utf-8")

    assert first == second
    assert first.splitlines() == [
        "16367aacb67a4a017c8da8ab95682ccb390863780f7114dda0a0e0c55644c7c4  b.txt",
        "a7937b64b8caa58f03721bb6bacf5c78cb235febe0e70b1b84cd99541461a08e  nested/a.txt",
    ]


def test_parent_candidate_strategy_reuses_one_candidate_pool_for_all_seeds(
    feasible_config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager
    from types import SimpleNamespace

    from porous_film.config import GeneratorConfig, load_config
    from porous_film.parallel.seeds import SeedTaskResult
    from porous_film.pipeline import GeometryRun, RunResult, _run_seed_panel
    from porous_film.storage import create_task_paths_at_root

    top_paths = create_task_paths_at_root(tmp_path / "top")
    config = load_config(feasible_config_path)
    config = config.model_copy(
        update={
            "audit": config.audit.model_copy(
                update={"candidate_count_per_round": 2, "maximum_rounds": 1}
            ),
            "optimization": config.optimization.model_copy(update={"seed_panel": (11, 22)}),
            "parallel": config.parallel.model_copy(
                update={"strategy": "candidates", "max_workers": 2}
            ),
        }
    )
    seen_executors: list[object | None] = []
    pool_markers: list[object] = []

    @contextmanager
    def fake_spawn_pool(worker_count: int, worker_threads: int):
        marker = object()
        pool_markers.append(marker)
        yield marker

    def fake_execute_seed_body(task, *, candidate_executor=None):
        seed_paths = create_task_paths_at_root(Path(task.task_root))
        seed_config = GeneratorConfig.model_validate(task.config_payload)
        seen_executors.append(candidate_executor)
        optimizer_payload = {
            "requested": {},
            "realized": {},
            "feasible": True,
            "constraints": {},
            "calculation_status": {"status": "completed_feasible"},
            "objectives": {"porosity_error": float(task.identity.panel_index)},
            "uncertainty": {},
        }
        seed_result = SeedTaskResult(
            identity=task.identity,
            status="completed_feasible",
            feasible=True,
            objective=float(task.identity.panel_index),
            failure_reason=None,
            artifact_root=str(seed_paths.root),
            optimizer_payload=optimizer_payload,
            representative_pickle=None,
            candidate_summary={},
            traceback_text=None,
            wall_time_seconds=0.0,
        )
        run_result = None
        if task.representative:
            run_result = RunResult(
                paths=seed_paths,
                geometry_run=GeometryRun(
                    config=seed_config,
                    built=SimpleNamespace(),
                    phase_grid=SimpleNamespace(porosity=0.3),
                    audit=SimpleNamespace(passed=True, warnings=[]),
                    paths=seed_paths,
                    candidate_results=(),
                    selected_candidate=SimpleNamespace(),
                ),
                packing_result=None,
                status="completed_feasible",
            )
        return seed_result, run_result

    published: list[tuple[Path, Path]] = []

    monkeypatch.setattr("porous_film.pipeline.spawn_pool", fake_spawn_pool)
    monkeypatch.setattr("porous_film.pipeline._execute_seed_body", fake_execute_seed_body)
    monkeypatch.setattr(
        "porous_film.pipeline._publish_representative",
        lambda seed_paths, top_paths, top_config: published.append(
            (seed_paths.root, top_paths.root)
        ),
    )

    result = _run_seed_panel(
        config,
        top_paths,
        config_path=feasible_config_path,
        run_identifier="top",
    )

    assert len(pool_markers) == 1
    assert seen_executors == [pool_markers[0], pool_markers[0]]
    assert result.paths == top_paths
    assert result.geometry_run.paths == top_paths
    assert result.geometry_run.config == config
    assert published == [(top_paths.analysis / "seed_panel" / "0-11", top_paths.root)]


def test_representative_pickle_path_must_be_inside_seed_work_directory(
    feasible_config_path: Path,
    tmp_path: Path,
) -> None:
    from porous_film.config import load_config
    from porous_film.parallel.seeds import SeedIdentity, SeedTask, SeedTaskResult
    from porous_film.pipeline import _load_representative_from_pickle
    from porous_film.storage import create_task_paths_at_root

    top_paths = create_task_paths_at_root(tmp_path / "top")
    seed_root = top_paths.analysis / "seed_panel" / "0-11"
    create_task_paths_at_root(seed_root)
    outside_pickle = tmp_path / "representative-run.pkl"
    outside_pickle.write_bytes(b"not-a-trusted-pickle")
    config = load_config(feasible_config_path)
    task = SeedTask(
        identity=SeedIdentity(panel_index=0, seed=11, sequence_index=0),
        config_payload=config.model_dump(mode="json"),
        task_root=str(seed_root),
        top_level_run_identifier="top",
        representative=True,
    )
    seed_result = SeedTaskResult(
        identity=task.identity,
        status="completed_feasible",
        feasible=True,
        objective=0.0,
        failure_reason=None,
        artifact_root=str(seed_root),
        optimizer_payload={},
        representative_pickle=str(outside_pickle),
        candidate_summary={},
        traceback_text=None,
        wall_time_seconds=0.0,
    )

    with pytest.raises(RuntimeError, match="representative pickle"):
        _load_representative_from_pickle(seed_result, task, top_paths, config)
    assert outside_pickle.is_file()


def test_parallel_pool_error_from_candidate_executor_propagates_without_seed_record(
    feasible_config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from porous_film.config import load_config
    from porous_film.parallel import ParallelPoolError
    from porous_film.parallel.seeds import SeedIdentity, SeedTask, _execute_seed_body

    config = load_config(feasible_config_path)
    seed_root = tmp_path / "seed-root"
    task = SeedTask(
        identity=SeedIdentity(panel_index=0, seed=config.task.random_seed, sequence_index=0),
        config_payload=config.model_dump(mode="json"),
        task_root=str(seed_root),
        top_level_run_identifier="top",
        representative=True,
    )
    pool_error = ParallelPoolError("candidate worker died", completed_results=())

    def raise_pool_error(*args, **kwargs):
        raise pool_error

    monkeypatch.setattr("porous_film.pipeline._execute_single_seed", raise_pool_error)

    with pytest.raises(ParallelPoolError) as raised:
        _execute_seed_body(task, candidate_executor=object())

    assert raised.value is pool_error
    assert not (seed_root / "logs" / "seed-record.json").exists()


def test_heavy_seed_equivalence_marker_is_enabled_by_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POROUS_FILM_RUN_HEAVY", "1")
    module_path = Path(__file__).with_name("test_parallel_integration.py")
    module_name = "_task7_heavy_marker_probe"
    sys.modules.pop(module_name, None)
    spec = util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    test_function = (
        module.test_serial_and_two_worker_seed_panels_match_without_duplicate_representative
    )
    marks = list(getattr(test_function, "pytestmark", ()))

    assert not any(mark.name == "skip" for mark in marks)
    heavy_mark = next(mark for mark in marks if mark.name == "skipif")
    assert heavy_mark.args == (False,)


def test_optimizer_payload_is_json_compact(
    feasible_config_path: Path,
    tmp_path: Path,
) -> None:
    from porous_film.config import load_config
    from porous_film.pipeline import optimizer_payload_for_result, run_full_at_root

    result = run_full_at_root(load_config(feasible_config_path), tmp_path / "run")
    payload = optimizer_payload_for_result(result)

    json.dumps(payload)
    assert set(payload) == {
        "requested",
        "realized",
        "feasible",
        "constraints",
        "calculation_status",
        "objectives",
        "uncertainty",
    }


def test_seed_panel_run_uses_main_path_schema_without_overwriting_top_level(
    feasible_config_path: Path,
    tmp_path: Path,
) -> None:
    from porous_film.config import load_config
    from porous_film.pipeline import run_full

    data = yaml.safe_load(feasible_config_path.read_text(encoding="utf-8"))
    data["optimization"] = {"seed_panel": [11, 22]}
    feasible_config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = run_full(load_config(feasible_config_path), tmp_path)
    standalone_config_path = write_config(
        tmp_path,
        name="standalone-22",
        target_porosity=0.30,
        seed_density=0.001,
        random_seed=22,
        audit_enabled=False,
        minimum_cross_section_fraction=0.0,
    )
    standalone = run_full(load_config(standalone_config_path), tmp_path)

    seed_dir = result.paths.analysis / "seed_panel"
    aggregate = json.loads((seed_dir / "aggregate.json").read_text(encoding="utf-8"))
    top_level_realized = json.loads(
        (result.paths.outputs / "realized_geometry_parameters.json").read_text(encoding="utf-8")
    )
    seed_22_realized = json.loads(
        (seed_dir / "1-22" / "realized_geometry_parameters.json").read_text(encoding="utf-8")
    )
    standalone_realized = json.loads(
        (standalone.paths.outputs / "realized_geometry_parameters.json").read_text(encoding="utf-8")
    )

    for seed in ("0-11", "1-22"):
        seed_path = seed_dir / seed
        assert sorted(path.name for path in seed_path.glob("*.json")) == [
            "calculation_status.json",
            "feasibility.json",
            "objectives.json",
            "realized_geometry_parameters.json",
            "requested_design_parameters.json",
            "uncertainty.json",
        ]
        assert (seed_path / "logs").is_dir()

    assert aggregate["seed_count"] == 2
    assert "objective_mean" in aggregate
    assert "objective_variance" in aggregate
    assert "failures" in aggregate
    assert top_level_realized["packed_molecule_count"] == 1
    assert seed_22_realized["porosity"] == standalone_realized["porosity"]
    assert seed_22_realized["packed_molecule_count"] == standalone_realized["packed_molecule_count"]


def test_optimizer_json_serializes_representative_requested_and_realized_design_values(
    feasible_config_path: Path,
    tmp_path: Path,
) -> None:
    from porous_film.config import load_config
    from porous_film.pipeline import run_full

    data = yaml.safe_load(feasible_config_path.read_text(encoding="utf-8"))
    data["center_distribution"]["position_jitter"] = 0.125
    data["orientation"] = {
        "distribution": {"family": "beta", "alpha": 3.0, "beta": 4.0},
        "azimuth": "uniform",
    }
    data["compact"]["relative_volume"] = {
        "family": "mixture",
        "components": [
            {"weight": 0.25, "family": "constant", "value": 0.75},
            {"weight": 0.75, "family": "constant", "value": 1.25},
        ],
    }
    data["channel"] = {
        "relative_volume": {"family": "constant", "value": 1.0},
        "eta": {"family": "constant", "value": 2.0},
        "tau": {"family": "constant", "value": 1.0},
        "roughness": {"family": "constant", "value": 0.0},
    }
    feasible_config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = run_full(load_config(feasible_config_path), tmp_path)

    requested = json.loads((result.paths.outputs / "requested_design_parameters.json").read_text())
    realized = json.loads((result.paths.outputs / "realized_geometry_parameters.json").read_text())

    assert (
        requested["pores"]["channel_fraction_by_count"]
        == data["pores"]["channel_fraction_by_count"]
    )
    assert (
        requested["pores"]["channel_to_compact_mean_volume_ratio"]
        == data["pores"]["channel_to_compact_mean_volume_ratio"]
    )
    assert requested["center_distribution"]["position_jitter"] == 0.125
    assert requested["orientation"]["distribution"]["alpha"] == 3.0
    assert requested["compact"]["relative_volume"]["components"][1]["value"] == 1.25
    assert requested["compact"]["superellipsoid_exponent"] == 2.0
    assert requested["pore_material"]["molecule_count"] == data["pore_material"]["molecule_count"]
    assert requested["packing"]["minimum_distance_A"] == 2.0
    assert realized["audit_results"]["unit_volume_summary"]["compact"]["count"] >= 1
    assert "compact_relative_volume" in realized["audit_results"]["distribution_results"]
    assert "mean_volume_ratio_relative_error" in realized["audit_results"]


def test_packmol_handoff_uses_realized_packing_count_for_density_mode(tmp_path: Path) -> None:
    from porous_film.config import GeneratorConfig
    from porous_film.pipeline import _write_packmol_handoff

    config = GeneratorConfig.model_validate(
        {
            "task": {"name": "density-handoff", "random_seed": 1},
            "film": {
                "target_box_A": {"x": 10, "y": 10, "z": 10},
                "packing_box_A": {"x": 10, "y": 10, "z": 12},
            },
            "pores": {
                "seed_number_density_A3": 0.001,
                "target_porosity": 0.2,
                "channel_fraction_by_count": 0.0,
                "channel_to_compact_mean_volume_ratio": 1.0,
            },
            "center_distribution": {
                "mode": "lattice_jitter",
                "lattice": "simple_cubic",
                "position_jitter": 0.0,
            },
            "compact": {
                "relative_volume": {"family": "constant", "value": 1.0},
                "aspect_ratio": {"family": "constant", "value": 1.0},
                "roughness": {"family": "constant", "value": 0.0},
            },
            "pore_material": {
                "pdb": str(Path("tests/fixtures/argon.pdb").resolve()),
                "target_density_g_cm3": 0.5,
            },
        }
    )
    output = tmp_path / "packmol_handoff.inp"

    _write_packmol_handoff(config, output, tmp_path / "pore_material.pdb", packing_count=7)

    assert "  number 7\n" in output.read_text(encoding="utf-8")


def test_audit_packmol_output_allows_configured_interface_mixing_layer(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    pore_mask = np.zeros((3, 3, 3), dtype=bool)
    pore_mask[1, 1, 1] = True
    _write_packmol_audit_run(run_root, pore_mask, interface_fraction=1.0)
    structure = tmp_path / "packed.pdb"
    structure.write_text(
        "\n".join(
            [
                _pdb_atom(1, "AR", "POR", 1, 0.5, 0.5, 0.5),
                _pdb_atom(2, "SI", "SEM", 2, 1.5, 1.5, 1.5),
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["audit-packmol-output", "--run", str(run_root), "--structure", str(structure)],
    )

    audit = json.loads((run_root / "analysis" / "packmol-output-audit.json").read_text())
    assert result.exit_code == 0
    assert audit["passed"] is True
    assert audit["atom_intrusion_fraction"] > 0.0
    assert audit["deep_intrusion_cluster_count"] == 0
    assert audit["interface_mixing_layer_fraction"] == 1.0
    assert (run_root / "analysis" / "packmol-output-audit.md").exists()


def test_audit_packmol_output_fails_for_connected_deep_intrusion_cluster(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    pore_mask = np.zeros((5, 5, 5), dtype=bool)
    pore_mask[1:4, 1:4, 1:4] = True
    _write_packmol_audit_run(run_root, pore_mask, interface_fraction=0.0, pore_indices=[])
    structure = tmp_path / "packed.pdb"
    structure.write_text(
        "\n".join(
            [
                _pdb_atom(1, "SI", "SEM", 1, 2.5, 2.5, 2.5),
                _pdb_atom(2, "SI", "SEM", 1, 2.6, 2.5, 2.5),
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["audit-packmol-output", "--run", str(run_root), "--structure", str(structure)],
    )

    audit = json.loads((run_root / "analysis" / "packmol-output-audit.json").read_text())
    assert result.exit_code == 1
    assert audit["passed"] is False
    assert audit["deep_intrusion_atom_count"] == 2
    assert audit["deep_intrusion_cluster_count"] == 1
    assert audit["maximum_intrusion_depth_A"] > 1.0


def test_run_full_infeasible_audit_writes_all_exchange_files_without_packing(
    sample_config_path: Path,
    tmp_path: Path,
) -> None:
    from porous_film.config import load_config
    from porous_film.pipeline import run_full

    data = yaml.safe_load(sample_config_path.read_text(encoding="utf-8"))
    data["matrix_constraints"]["minimum_cross_section_fraction"] = 0.99
    sample_config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = run_full(load_config(sample_config_path), tmp_path)
    outputs = result.paths.outputs
    expected = [
        "requested_design_parameters.json",
        "realized_geometry_parameters.json",
        "feasibility.json",
        "calculation_status.json",
        "objectives.json",
        "uncertainty.json",
    ]
    status = json.loads((outputs / "calculation_status.json").read_text(encoding="utf-8"))
    feasibility = json.loads((outputs / "feasibility.json").read_text(encoding="utf-8"))

    assert all((outputs / name).exists() for name in expected)
    assert result.status == "completed_infeasible"
    assert status["status"] == "completed_infeasible"
    assert feasibility["feasible"] is False
    assert not (outputs / "pore_reference_coordinates.cif").exists()


def test_failed_run_writes_all_optimizer_json_and_logs(tmp_path: Path) -> None:
    from porous_film.config import load_config
    from porous_film.pipeline import run_full

    config_path = write_config(tmp_path, name="failed-run", pdb_path=tmp_path / "missing.pdb")

    try:
        run_full(load_config(config_path), tmp_path)
    except FileNotFoundError:
        pass

    run_root = next(tmp_path.glob("*/python_results/failed-run*"))
    outputs = run_root / "outputs"
    expected = [
        "requested_design_parameters.json",
        "realized_geometry_parameters.json",
        "feasibility.json",
        "calculation_status.json",
        "objectives.json",
        "uncertainty.json",
    ]

    assert all((outputs / name).exists() for name in expected)
    assert (run_root / "logs" / "failure.json").exists()
    assert json.loads((outputs / "calculation_status.json").read_text())["status"] == "failed"


def _write_packmol_audit_run(
    run_root: Path,
    pore_mask: object,
    *,
    interface_fraction: float,
    pore_indices: list[int] | None = None,
) -> None:
    (run_root / "qa_export").mkdir(parents=True)
    (run_root / "outputs").mkdir(parents=True)
    (run_root / "inputs").mkdir(parents=True)
    (run_root / "analysis").mkdir(parents=True)
    mask = np.asarray(pore_mask, dtype=bool)
    target_box = [float(mask.shape[2]), float(mask.shape[1]), float(mask.shape[0])]
    with h5py.File(run_root / "qa_export" / "final_phase.h5", "w") as handle:
        handle.create_dataset("pore_mask", data=mask.astype("uint8"))
        handle.attrs["spacing_A"] = 1.0
        handle.attrs["origin_A"] = [0.0, 0.0, 0.0]
        handle.attrs["target_box_A"] = target_box
        handle.attrs["axis_order"] = "zyx"
    indices = [1] if pore_indices is None else pore_indices
    (run_root / "outputs" / "pore_atom_indices.ndx").write_text(
        "[ PORE ]\n" + " ".join(str(index) for index in indices) + "\n",
        encoding="utf-8",
    )
    (run_root / "inputs" / "normalized_config.yaml").write_text(
        f"audit:\n  interface_mixing_layer_fraction: {interface_fraction}\n",
        encoding="utf-8",
    )


def _pdb_atom(
    serial: int,
    atom_name: str,
    residue_name: str,
    residue_number: int,
    x_A: float,
    y_A: float,
    z_A: float,
) -> str:
    return (
        f"HETATM{serial:5d} {atom_name[:4]:>4s} {residue_name[:3]:>3s} A"
        f"{residue_number:4d}    {x_A:8.3f}{y_A:8.3f}{z_A:8.3f}"
        "  1.00  0.00          "
        f"{atom_name[:2]:>2s}"
    )


def test_generate_geometry_cli_retains_inputs(sample_config_path: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["generate-geometry", "--config", str(sample_config_path), "--result-root", str(tmp_path)],
    )

    assert result.exit_code == 0
    output_lines = result.stdout.strip().splitlines()
    assert output_lines[0].startswith("Result directory: ")
    assert output_lines[1].startswith("Runtime: ")
    assert output_lines[1].endswith(" s")
    assert float(output_lines[1].removeprefix("Runtime: ").removesuffix(" s")) >= 0.0
    assert output_lines[2] in {"Audit result: PASS", "Audit result: FAIL"}
    run_root = Path(output_lines[0].removeprefix("Result directory: "))
    assert (run_root / "inputs" / "normalized_config.yaml").exists()
    assert (run_root / "inputs" / "argon.pdb").exists()


def test_generate_cli_copies_original_yaml_byte_identically(
    feasible_config_path: Path,
    tmp_path: Path,
) -> None:
    original = feasible_config_path.read_bytes() + b"# source comment retained verbatim\r\n"
    feasible_config_path.write_bytes(original)

    result = runner.invoke(
        app,
        ["generate", "--config", str(feasible_config_path), "--result-root", str(tmp_path)],
    )

    assert result.exit_code == 0, result.stdout
    run_root = Path(result.stdout.strip())
    copied = run_root / "inputs" / feasible_config_path.name
    assert copied.read_bytes() == original


def test_run_directory_cli_commands_use_existing_artifacts(
    feasible_config_path: Path,
    tmp_path: Path,
) -> None:
    from porous_film.config import load_config
    from porous_film.pipeline import run_full

    result = run_full(load_config(feasible_config_path), tmp_path)

    fill = runner.invoke(app, ["fill-pore", "--run", str(result.paths.root)])
    audit = runner.invoke(app, ["audit", "--run", str(result.paths.root)])
    packmol = runner.invoke(
        app,
        [
            "audit-packmol-output",
            "--run",
            str(result.paths.root),
            "--structure",
            str(result.paths.outputs / "pore_reference_coordinates.cif"),
        ],
    )

    assert fill.exit_code == 0
    assert audit.exit_code == 0
    assert packmol.exit_code == 0
    assert (result.paths.analysis / "fill-pore-status.json").exists()
    assert (result.paths.analysis / "audit-status.json").exists()
    assert (result.paths.analysis / "packmol-output-audit.json").exists()


def test_asymmetric_padding_shifts_coordinates_to_packing_frame(tmp_path: Path) -> None:
    from porous_film.config import load_config
    from porous_film.molecules import MoleculeTemplate
    from porous_film.molecules.packing import InstanceTransform, PackingResult
    from porous_film.pipeline import _packing_frame_result

    config_path = write_config(
        tmp_path,
        name="asymmetric",
        target_porosity=0.30,
        seed_density=0.004,
        lower_padding=7.0,
        audit_enabled=False,
        minimum_cross_section_fraction=0.0,
    )

    config = load_config(config_path)
    template = MoleculeTemplate.from_pdb(Path("tests/fixtures/argon.pdb"))
    target_position = np.array([[1.0, 2.0, 3.0]])
    packing_result = PackingResult(
        count=1,
        atom_positions_A=target_position,
        instance_transforms=(
            InstanceTransform(
                translation_A=target_position[0],
                quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            ),
        ),
        minimum_interatomic_distance_A=float("inf"),
        actual_density_g_cm3=0.0,
        protrusion_metrics={},
        status="accepted",
        template=template,
        target_box_A=np.array([20.0, 20.0, 20.0]),
        pore_volume_A3=1.0,
    )
    shifted = _packing_frame_result(config, packing_result)

    assert shifted.atom_positions_A == pytest.approx(target_position + [0.0, 0.0, 7.0])
    assert shifted.instance_transforms[0].translation_A == pytest.approx([1.0, 2.0, 10.0])
    assert shifted.target_box_A == pytest.approx([20.0, 20.0, 30.0])


def _read_glb_json(path: Path) -> dict:
    data = path.read_bytes()
    magic, version, _length = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF"
    assert version == 2
    json_length, chunk_type = struct.unpack_from("<I4s", data, 12)
    assert chunk_type == b"JSON"
    return json.loads(data[20 : 20 + json_length].rstrip(b" \x00"))


def test_channel_curves_h5_v2_uses_group_with_centerline_and_radius(tmp_path: Path) -> None:
    from porous_film.geometry import BuiltGeometry, ChannelUnit, PoreGeometry
    from porous_film.pipeline import _write_channel_curves

    channel = ChannelUnit.from_polyline(
        "channel-v2",
        np.array([[0.0, 0.0, 0.0], [5.0, 1.0, 0.5], [10.0, 0.0, 0.0]]),
        1.0,
        0.0,
        shape_model="variable-radius-spline-v1",
        shape_seed=101,
        radius_profile_s=np.array([0.0, 0.5, 1.0]),
        radius_profile_A=np.array([0.7, 1.3, 0.8]),
        bend_count=2,
        nonplanarity=0.1,
        minimum_self_clearance_A=2.0,
    )
    built = BuiltGeometry(
        geometry=PoreGeometry([channel], np.array([20.0, 20.0, 20.0])),
        units=[channel],
        realized_anchors_A=channel.anchor_A[np.newaxis, :],
        latent_to_realized_ids={"latent-0000": channel.unit_id},
    )
    path = tmp_path / "channel_curves.h5"

    _write_channel_curves(built, path)

    with h5py.File(path, "r") as handle:
        assert handle.attrs["schema_version"] == 2
        group = handle[channel.unit_id]
        assert isinstance(group, h5py.Group)
        centerline = np.asarray(group["centerline_A"])
        radii = np.asarray(group["radius_A"])
        assert centerline.shape == channel.centerline_samples_A.shape
        assert radii.shape == (centerline.shape[0],)
        assert np.all(radii > 0.0)
        assert group.attrs["shape_model"] == "variable-radius-spline-v1"
        assert int(group.attrs["shape_seed"]) == 101


def test_shape_complexity_metrics_are_written_per_unit_and_summarized(
    tmp_path: Path,
) -> None:
    import csv

    from scipy.spatial.transform import Rotation

    from porous_film.geometry import BuiltGeometry, ChannelUnit, CompactUnit, PoreGeometry
    from porous_film.geometry.complex_shapes import (
        generate_multilobe_profile,
        generate_variable_radius_channel_profile,
    )
    from porous_film.pipeline import _shape_complexity_summary, _write_unit_metrics

    compact_profile = generate_multilobe_profile(2_000.0, 2.2, 3101)
    compact = CompactUnit(
        unit_id="compact-v2",
        center_A=np.array([2.0, 3.0, 4.0]),
        radii_A=compact_profile.envelope_radii_A,
        orientation=Rotation.identity(),
        exponent=2.0,
        roughness=0.0,
        shape_model="multilobe-v1",
        shape_seed=3101,
        lobe_centers_local_A=compact_profile.lobe_centers_local_A,
        lobe_radii_A=compact_profile.lobe_radii_A,
        smooth_length_A=compact_profile.smooth_length_A,
        envelope_fill_fraction=compact_profile.envelope_fill_fraction,
        centroid_offset_A=compact_profile.centroid_offset_A,
        lobes_connected=compact_profile.connected,
    )
    channel_profile = generate_variable_radius_channel_profile(4_000.0, 7.0, 1.4, 3102)
    channel = ChannelUnit.from_polyline(
        "channel-v2",
        channel_profile.control_points_local_A + np.array([8.0, 9.0, 10.0]),
        channel_profile.equivalent_radius_A,
        0.0,
        shape_model="variable-radius-spline-v1",
        shape_seed=3102,
        radius_profile_s=channel_profile.radius_profile_s,
        radius_profile_A=channel_profile.radius_profile_A,
        bend_count=channel_profile.bend_count,
        nonplanarity=channel_profile.nonplanarity,
        minimum_self_clearance_A=channel_profile.minimum_self_clearance_A,
    )
    built = BuiltGeometry(
        geometry=PoreGeometry([compact, channel], np.array([30.0, 30.0, 30.0])),
        units=[compact, channel],
        realized_anchors_A=np.vstack([compact.anchor_A, channel.anchor_A]),
        latent_to_realized_ids={
            "latent-0000": compact.unit_id,
            "latent-0001": channel.unit_id,
        },
    )
    metrics_path = tmp_path / "main_unit_metrics.csv"

    _write_unit_metrics(built, metrics_path)
    summary = _shape_complexity_summary(built)

    with metrics_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    compact_row = next(row for row in rows if row["unit_id"] == compact.unit_id)
    channel_row = next(row for row in rows if row["unit_id"] == channel.unit_id)
    assert compact_row["shape_model"] == "multilobe-v1"
    assert int(compact_row["lobe_count"]) == compact_profile.lobe_count
    assert float(compact_row["envelope_fill_fraction"]) >= 0.50
    assert compact_row["lobes_connected"] == "True"
    assert channel_row["shape_model"] == "variable-radius-spline-v1"
    assert 0.15 <= float(channel_row["radius_cv"]) <= 0.30
    assert int(channel_row["bend_count"]) >= 2
    assert float(channel_row["nonplanarity"]) > 0.0
    assert summary["compact"]["count"] == 1
    assert summary["channel"]["count"] == 1
    assert summary["compact"]["lobe_count"]["mean"] == compact_profile.lobe_count
    assert summary["channel"]["bend_count"]["minimum"] >= 2


def _pipeline_v3_config_dict() -> dict:
    return {
        "schema_version": 3,
        "task": {"name": "phase-field-only", "random_seed": 29},
        "film": {"target_box_A": {"x": 10.0, "y": 10.0, "z": 10.0}},
        "formal_targets": {
            "position_quantity": {"center_distance_xy": {"components": []}},
            "shape": {
                "equivalent_diameter_A": {"family": "constant", "value": 4.0},
                "orientation": {
                    "model": "paired_projected_planes",
                    "components": [
                        {
                            "weight": 1.0,
                            "theta_xz_deg": {
                                "family": "beta",
                                "alpha": 2.0,
                                "beta": 2.0,
                                "lower": 60.0,
                                "upper": 90.0,
                            },
                            "theta_xy_deg": {
                                "family": "beta",
                                "alpha": 2.0,
                                "beta": 2.0,
                                "lower": 0.0,
                                "upper": 30.0,
                            },
                        }
                    ],
                },
                "channel_aspect_ratio": {"family": "constant", "value": 4.0},
                "channel_tortuosity": {"family": "constant", "value": 1.1},
                "curvature_fluctuation": {"family": "constant", "value": 0.2},
            },
            "proportion": {"porosity": 0.10},
        },
        "generation_controls": {
            "seed_number_density_A3": 0.001,
            "channel_fraction_by_count": 1.0,
            "channel_to_compact_mean_volume_ratio": 1.0,
        },
        "measurement": {
            "z_slice_spacing_A": 1.0,
            "center_min_separation_A": 1.0,
            "center_tracking_max_displacement_A": 2.0,
            "center_distance_bin_width_A": 1.0,
            "center_distance_reference_samples": 1024,
            "centerline_sample_spacing_A": 1.0,
            "cross_section_spacing_A": 1.0,
            "boundary_resample_spacing_A": 0.5,
            "curvature_smoothing_length_A": 0.5,
            "branch_exclusion_length_A": 1.0,
            "surface_exclusion_length_A": 1.0,
            "orientation_projection_min_fraction": 0.05,
        },
        "matrix_constraints": {
            "enabled": False,
            "require_x_percolation": False,
            "minimum_cross_section_fraction": 0.0,
            "maximum_overlap_fraction": 1.0,
        },
        "audit": {
            "enabled": False,
            "candidate_count_per_round": 1,
            "maximum_rounds": 1,
            "coarse_spacing_A": 2.0,
            "fine_spacing_A": 1.0,
        },
        "parallel": {"enabled": False, "strategy": "serial"},
    }


def test_schema_v3_preflight_and_input_copy_do_not_require_pdb_or_padding(
    tmp_path: Path,
) -> None:
    from porous_film.config import GeneratorConfig
    from porous_film.pipeline import preflight, prepare_task_inputs
    from porous_film.storage import create_task_paths_at_root

    config = GeneratorConfig.model_validate(_pipeline_v3_config_dict())
    paths = create_task_paths_at_root(tmp_path / "prepared")

    prepare_task_inputs(config, paths)
    result = preflight(config, tmp_path / "preflight")

    assert result.passed
    assert (paths.inputs / "normalized_config.yaml").is_file()
    assert not list(paths.inputs.glob("*.pdb"))
    assert config.film.packing_box_A == config.film.target_box_A


def _pipeline_v3_phase_grid():
    from porous_film.voxel import PhaseGrid

    _z, y, x = np.indices((10, 10, 10), dtype=float)
    pore_mask = (x - 3.5) ** 2 + (y - 5.5) ** 2 <= 1.8**2
    return PhaseGrid(
        pore_mask=pore_mask,
        origin_A=np.zeros(3),
        spacing_A=1.0,
        target_box_A=np.array([10.0, 10.0, 10.0]),
    )


def test_geometry_artifacts_write_final_phase_centerlines_and_cross_sections(
    tmp_path: Path,
) -> None:
    from porous_film.centers import CenterSeedPlan
    from porous_film.config import GeneratorConfig
    from porous_film.geometry import BuiltGeometry, PoreGeometry
    from porous_film.metrics import audit_target_distributions
    from porous_film.parallel import CandidateIdentity
    from porous_film.pipeline import GeometryRun, _write_geometry_artifacts
    from porous_film.storage import create_task_paths_at_root

    config_data = _pipeline_v3_config_dict()
    config_data["formal_targets"]["proportion"]["porosity"] = 0.12
    config = GeneratorConfig.model_validate(config_data)
    grid = _pipeline_v3_phase_grid()
    built = BuiltGeometry(
        geometry=PoreGeometry([], grid.target_box_A),
        units=[],
        realized_anchors_A=np.empty((0, 3)),
        latent_to_realized_ids={},
    )
    center_plan = CenterSeedPlan(
        intended_points_A=np.empty((0, 3)),
        target_rdf_xi=np.array([0.0]),
        target_rdf_values=np.array([1.0]),
        starting_loss=0.0,
        initialization_loss=0.0,
    )
    audit = audit_target_distributions(config, built, center_plan, grid)
    paths = create_task_paths_at_root(tmp_path / "geometry-artifacts")
    identity = CandidateIdentity(
        seed=29,
        round_index=0,
        candidate_index=0,
        derived_random_seed=31,
        sequence_index=0,
    )
    run = GeometryRun(
        config=config,
        built=built,
        phase_grid=grid,
        audit=audit,
        paths=paths,
        candidate_results=(),
        selected_candidate=identity,
    )

    _write_geometry_artifacts(config, run)

    assert (paths.qa_export / "final_phase.h5").is_file()
    assert (paths.qa_export / "final_centerlines.h5").is_file()
    assert (paths.qa_export / "final_cross_sections.csv").is_file()
    assert (paths.qa_export / "final_measurements.json").is_file()
    assert (paths.qa_export / "contract.json").is_file()
    assert (paths.qa_export / "main_metrics.json").is_file()
    assert (paths.qa_export / "final_surface.ply").is_file()
    assert (paths.qa_export / "semiconductor_solid_target.glb").is_file()
    assert (paths.qa_export / "checksums.sha256").is_file()
    assert (paths.qa_export / "performance.json").is_file()
    assert (paths.outputs / "semiconductor_solid_target.glb").is_file()
    assert (paths.outputs / "visual-report" / "index.html").is_file()
    performance = json.loads((paths.qa_export / "performance.json").read_text())
    assert performance["stage_timings_seconds"]["export"] >= 0.0
    assert performance["peak_rss_mib"] > 0.0
    with h5py.File(paths.qa_export / "final_centerlines.h5", "r") as handle:
        assert handle.attrs["schema_version"] == 3
        assert len(handle["centerlines"]) >= 1
    header = (
        (paths.qa_export / "final_cross_sections.csv").read_text(encoding="utf-8").splitlines()[0]
    )
    assert "equivalent_diameter_A" in header
    assert "curvature_fluctuation" in header


def test_serial_geometry_generation_reuses_selected_candidate_artifacts(
    sample_config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from porous_film.config import load_config
    from porous_film.pipeline import generate_geometry
    from porous_film.storage import create_task_paths_at_root

    def unexpected_replay(*args, **kwargs):
        raise AssertionError("serial generation must not replay the selected candidate")

    monkeypatch.setattr("porous_film.pipeline.replay_candidate", unexpected_replay)
    paths = create_task_paths_at_root(tmp_path / "single-pass")

    run = generate_geometry(load_config(sample_config_path), paths)

    assert run.phase_grid.porosity == run.candidate_results[0].porosity
    assert run.performance is not None


def test_execute_single_seed_schema_v3_without_pore_material_skips_packing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from porous_film.config import GeneratorConfig
    from porous_film.pipeline import GeometryRun, _execute_single_seed
    from porous_film.storage import create_task_paths_at_root

    config = GeneratorConfig.model_validate(_pipeline_v3_config_dict())
    paths = create_task_paths_at_root(tmp_path / "phase-only")
    geometry_run = GeometryRun(
        config=config,
        built=SimpleNamespace(),
        phase_grid=SimpleNamespace(porosity=0.10),
        audit=SimpleNamespace(passed=True, warnings=[], minimum_cross_section_fraction=0.5),
        paths=paths,
        candidate_results=(),
        selected_candidate=SimpleNamespace(),
    )
    monkeypatch.setattr(
        "porous_film.pipeline.generate_geometry", lambda *args, **kwargs: geometry_run
    )
    monkeypatch.setattr("porous_film.pipeline._write_reports", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "porous_film.pipeline.MoleculeTemplate.from_pdb",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("packing must be skipped")),
    )

    result = _execute_single_seed(config, paths, write_artifacts=True)

    assert result.status == "completed_feasible"
    assert result.packing_result is None


def test_schema_v3_requested_parameters_without_pore_material_omit_packing() -> None:
    from porous_film.config import GeneratorConfig
    from porous_film.pipeline import _requested_parameters

    config = GeneratorConfig.model_validate(_pipeline_v3_config_dict())

    requested = _requested_parameters(config)

    assert "pore_material" not in requested
    assert "packing" not in requested
    assert requested["target_box_A"] == [10.0, 10.0, 10.0]
