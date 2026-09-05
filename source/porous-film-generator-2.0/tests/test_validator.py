import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import trimesh
import yaml
from scipy import stats
from scipy.interpolate import PchipInterpolator
from typer.testing import CliRunner

from porous_film.metrics import compare_samples_to_distribution
from porous_film_validator.cli import app as validator_app
from porous_film_validator.validate import (
    _compare_v3_json_compacts,
    _config_schema_version,
    _independent_overlap_fraction,
    _PhaseData,
    _points_inside_triangle_mesh,
    _points_inside_triangle_mesh_brute_force,
    _target_compliance,
    _v3_compare_paired_orientation_pairs,
    _v3_distribution_target_result,
    _v3_measure_final_geometry,
    _v3_measurement_contract,
    _V3CompactGeometryMeasurement,
    _V3MeasurementContract,
    validate_export,
)

runner = CliRunner()


def test_indexed_mesh_occupancy_matches_brute_force() -> None:
    mesh = trimesh.creation.box(extents=[4.0, 5.0, 6.0])
    rng = np.random.default_rng(20260904)
    points = rng.uniform(-4.0, 4.0, size=(257, 3))

    expected = _points_inside_triangle_mesh_brute_force(mesh, points)
    actual = _points_inside_triangle_mesh(mesh, points)

    assert np.array_equal(actual, expected)


def test_validator_uses_source_schema_version_for_translated_legacy_config() -> None:
    assert _config_schema_version({"schema_version": 3, "source_schema_version": 2}) == 2
    assert _config_schema_version({"schema_version": 3, "source_schema_version": 3}) == 3
    assert _config_schema_version({"schema_version": 3}) == 3


def test_validator_console_command_is_installed() -> None:
    result = subprocess.run(
        ["porous-film-validate", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_validator_imports_when_main_package_is_blocked() -> None:
    script = """
import builtins

real_import = builtins.__import__


def guarded_import(name, *args, **kwargs):
    if name == "porous_film" or name.startswith("porous_film."):
        raise RuntimeError("validator imported main package")
    return real_import(name, *args, **kwargs)


builtins.__import__ = guarded_import
import porous_film_validator.validate
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_validator_rejects_incomplete_mandatory_export(
    incomplete_qa_export: Path,
) -> None:
    report = validate_export(incomplete_qa_export)

    assert report.status in {"FAIL", "NOT_EVALUABLE"}
    assert "mandatory" in " ".join(report.errors).lower()


def test_validator_reports_contract_schema_mismatch(
    incomplete_qa_export: Path,
) -> None:
    (incomplete_qa_export / "contract.json").write_text(
        json.dumps(
            {
                "format_version": 99,
                "length_unit": "angstrom",
                "origin_A": [0.0, 0.0, 0.0],
                "target_box_A": [2.0, 2.0, 2.0],
                "periodic_axes": ["x", "y"],
                "axis_order": "zyx",
                "phase_encoding": {"semiconductor": 0, "pore": 1},
                "final_phase_file": "final_phase.h5",
            }
        ),
        encoding="utf-8",
    )

    report = validate_export(incomplete_qa_export)

    assert "format_version" in " ".join(report.errors)


def test_validator_cli_reports_nonzero_for_incomplete_export(
    incomplete_qa_export: Path,
) -> None:
    result = runner.invoke(validator_app, [str(incomplete_qa_export)])

    assert result.exit_code == 1
    assert "NOT_EVALUABLE" in result.stdout or "FAIL" in result.stdout
    timing_line = next(
        line for line in result.stdout.splitlines() if line.startswith("Validation time: ")
    )
    assert timing_line.endswith(" s")
    assert float(timing_line.removeprefix("Validation time: ").removesuffix(" s")) >= 0.0
    assert (incomplete_qa_export / "independent-validation.json").exists()


def test_validator_fails_when_independent_porosity_misses_target(
    tmp_path: Path,
) -> None:
    qa = _write_complete_qa_export(tmp_path, target_porosity=0.50)

    report = validate_export(qa)

    assert report.status == "FAIL"
    assert "porosity" in " ".join(report.errors).lower()
    assert report.target_compliance["porosity_within_tolerance"] is False


def test_validator_uses_same_voxel_aware_porosity_tolerance_as_main_audit() -> None:
    errors: list[str] = []

    compliance = _target_compliance(
        {
            "film": {"target_box_A": {"x": 2.0, "y": 2.0, "z": 2.0}},
            "pores": {"target_porosity": 0.05},
        },
        {"target_box_A": [2.0, 2.0, 2.0]},
        {
            "target_box_A": [2.0, 2.0, 2.0],
            "porosity": 0.125,
            "pore_voxels": 1,
            "semiconductor_voxels": 7,
        },
        {},
        {},
        {},
        errors,
    )

    assert compliance["porosity_tolerance"] == 0.125
    assert compliance["porosity_within_tolerance"] is True
    assert not errors


def test_validator_fails_when_recomputed_molecule_count_misses_target(
    tmp_path: Path,
) -> None:
    qa = _write_complete_qa_export(tmp_path, target_molecule_count=1)

    report = validate_export(qa)

    assert report.status == "FAIL"
    assert "molecule count" in " ".join(report.errors).lower()
    assert report.target_compliance["molecule_count_match"] is False


def test_validator_fails_when_recomputed_density_misses_target_density_mode(
    tmp_path: Path,
) -> None:
    qa = _write_complete_qa_export(
        tmp_path,
        target_density_g_cm3=99.0,
        molecule_instance_ids=[0],
    )

    report = validate_export(qa)

    assert report.status == "FAIL"
    assert "density" in " ".join(report.errors).lower()
    assert report.target_compliance["density_within_tolerance"] is False


def test_validator_ignores_parallel_source_metadata_in_normalized_config(
    tmp_path: Path,
) -> None:
    qa = _write_complete_qa_export(tmp_path)
    normalized_path = qa / "normalized_config.yaml"
    normalized = yaml.safe_load(normalized_path.read_text(encoding="utf-8"))
    normalized["parallel"] = {
        "enabled": True,
        "strategy": "auto",
        "max_workers": None,
        "cpu_fraction": 0.80,
        "memory_fraction": 0.75,
        "worker_threads": 1,
        "start_method": "spawn",
        "sources": {
            "enabled": "config",
            "strategy": "config",
            "max_workers": "config",
        },
    }
    normalized_path.write_text(yaml.safe_dump(normalized, sort_keys=True), encoding="utf-8")
    _write_checksums(qa)

    report = validate_export(qa)

    assert report.status == "PASS", report.errors
    assert report.target_compliance["target_box_match"] is True
    assert report.target_compliance["porosity_within_tolerance"] is True
    assert report.target_compliance["molecule_count_match"] is True
    assert report.independent_metrics["molecules"]["instance_count"] == 0


def test_validator_recomputes_molecule_count_instead_of_trusting_claim(
    tmp_path: Path,
) -> None:
    qa = _write_complete_qa_export(
        tmp_path,
        target_molecule_count=2,
        molecule_instance_ids=[0, 1],
        claimed_count=1,
        claimed_density_count=1,
        main_metrics_count=1,
    )

    report = validate_export(qa)

    assert report.status == "FAIL"
    assert report.independent_metrics["molecules"]["instance_count"] == 2
    assert report.independent_metrics["molecules"]["claimed_count"] == 1
    assert "claimed molecule count" in " ".join(report.errors).lower()


def test_validator_rejects_incomplete_and_unsafe_checksums(
    tmp_path: Path,
) -> None:
    qa = _write_complete_qa_export(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    digest = _sha256(outside)
    (qa / "checksums.sha256").write_text(
        "\n".join(
            [
                f"{_sha256(qa / 'contract.json')}  contract.json",
                f"{_sha256(qa / 'contract.json')}  contract.json",
                f"{digest}  {outside}",
                f"{digest}  ../outside.txt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_export(qa)

    assert report.status in {"FAIL", "NOT_EVALUABLE"}
    errors = " ".join(report.errors).lower()
    assert "checksum" in errors
    assert "duplicate" in errors
    assert "outside qa_export" in errors
    assert "missing" in errors


def test_validator_warns_for_wrong_glb_spatial_occupancy_with_matching_volume(
    tmp_path: Path,
) -> None:
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[:, :, :2] = True
    wrong_glb = _wrong_half_volume_glb()
    qa = _write_complete_qa_export(
        tmp_path,
        mask=mask,
        target_box_A=(4.0, 4.0, 4.0),
        target_porosity=0.5,
        glb_mesh=wrong_glb,
    )

    report = validate_export(qa)

    assert report.status in {"WARNING", "FAIL"}
    assert report.independent_metrics["glb"]["occupancy_mismatch_fraction"] > 0.25
    assert "occupancy" in " ".join(report.warnings + report.errors).lower()


def test_validator_uses_channel_curves_for_channel_volume_statistics(
    tmp_path: Path,
) -> None:
    unit_id = "channel-1"
    qa = _write_complete_qa_export(
        tmp_path,
        unit_records=[_channel_record(unit_id, cross_radius_A=0.5)],
        channel_curves={unit_id: np.array([[0.0, 1.0, 1.0], [4.0, 1.0, 1.0]])},
        target_box_A=(4.0, 4.0, 4.0),
    )

    report = validate_export(qa)

    assert report.status == "PASS", report.errors
    assert report.independent_metrics["units"]["channel_count"] == 1
    assert report.independent_metrics["units"]["volume_sample_count"] == 1
    assert report.independent_metrics["units"]["volume_mean_A3"] == 7.0 * math.pi / 6.0


def test_validator_errors_when_channel_curve_data_is_missing(
    tmp_path: Path,
) -> None:
    qa = _write_complete_qa_export(
        tmp_path,
        unit_records=[_channel_record("channel-1", cross_radius_A=0.5)],
        channel_curves={},
    )

    report = validate_export(qa)

    assert report.status == "FAIL"
    assert "channel_curves" in " ".join(report.errors)


def _write_complete_qa_export(
    tmp_path: Path,
    *,
    mask: np.ndarray | None = None,
    target_box_A: tuple[float, float, float] = (2.0, 2.0, 2.0),
    target_porosity: float = 0.0,
    target_molecule_count: int | None = 0,
    target_density_g_cm3: float | None = None,
    molecule_instance_ids: list[int] | None = None,
    claimed_count: int | None = None,
    claimed_density_count: int | None = None,
    main_metrics_count: int | None = None,
    unit_records: list[dict] | None = None,
    channel_curves: dict[str, np.ndarray] | None = None,
    glb_mesh: trimesh.Trimesh | trimesh.Scene | None = None,
) -> Path:
    qa = tmp_path / "qa_export"
    qa.mkdir()
    (qa / "molecules" / "source").mkdir(parents=True)

    target_box = np.asarray(target_box_A, dtype=float)
    if mask is None:
        counts = tuple(int(value) for value in target_box)
        mask = np.zeros((counts[2], counts[1], counts[0]), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    spacing = float(target_box[0] / mask.shape[2])
    instance_ids = molecule_instance_ids or []
    independent_count = len(set(instance_ids))
    claimed = independent_count if claimed_count is None else claimed_count
    density_count = independent_count if claimed_density_count is None else claimed_density_count
    metrics_count = independent_count if main_metrics_count is None else main_metrics_count
    template_mass = 39.948
    pore_volume_A3 = 1000.0
    claimed_density = density_count * template_mass / 6.022_140_76e23 / (pore_volume_A3 * 1.0e-24)

    _write_phase_h5(qa / "final_phase.h5", mask, target_box, spacing)
    _write_json(
        qa / "contract.json",
        {
            "format_version": 1,
            "length_unit": "angstrom",
            "origin_A": [0.0, 0.0, 0.0],
            "target_box_A": target_box.tolist(),
            "periodic_axes": ["x", "y"],
            "axis_order": "zyx",
            "phase_encoding": {"semiconductor": 0, "pore": 1},
            "final_phase_file": "final_phase.h5",
        },
    )
    material_amount = (
        f"  target_density_g_cm3: {target_density_g_cm3}"
        if target_density_g_cm3 is not None
        else f"  molecule_count: {target_molecule_count}"
    )
    (qa / "normalized_config.yaml").write_text(
        "\n".join(
            [
                "film:",
                (f"  target_box_A: {{x: {target_box[0]}, y: {target_box[1]}, z: {target_box[2]}}}"),
                "pores:",
                f"  target_porosity: {target_porosity}",
                "pore_material:",
                material_amount,
                "",
            ]
        ),
        encoding="utf-8",
    )
    (qa / "unit_candidates.jsonl").write_text("", encoding="utf-8")
    records = unit_records or []
    (qa / "unit_geometry.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    _write_channel_curves(qa / "channel_curves.h5", channel_curves or {})
    (qa / "final_surface.ply").write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    _export_glb(qa / "semiconductor_solid_target.glb", glb_mesh or _box_mesh(target_box))
    (qa / "main_unit_metrics.csv").write_text(
        "unit_id,kind,anchor_x_A,anchor_y_A,anchor_z_A\n", encoding="utf-8"
    )
    _write_json(
        qa / "main_metrics.json",
        {
            "realized_porosity": float(np.mean(mask)),
            "packing_count": metrics_count,
            "actual_density_g_cm3": claimed_density,
            "minimum_interatomic_distance_A": math.inf,
        },
    )
    (qa / "molecules" / "source" / "argon.pdb").write_text(
        "HETATM    1 AR   ARG A   1       0.000   0.000   0.000  1.00  0.00          Ar\nEND\n",
        encoding="utf-8",
    )
    _write_instances_csv(qa / "molecules" / "instances.csv", instance_ids)
    _write_placed_atoms_h5(
        qa / "molecules" / "placed_atoms.h5",
        instance_ids=instance_ids,
        claimed_count=claimed,
        density=claimed_density,
        pore_volume_A3=pore_volume_A3,
        target_box=target_box,
        template_mass=template_mass,
    )
    _write_cif(qa / "molecules" / "placed_structure.cif", len(instance_ids))
    _write_checksums(qa)
    return qa


def _write_phase_h5(path: Path, mask: np.ndarray, target_box: np.ndarray, spacing: float) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("pore_mask", data=mask.astype(np.uint8), compression="gzip")
        handle.attrs["schema_version"] = 1
        handle.attrs["axis_order"] = "zyx"
        handle.attrs["periodic_axes"] = "x,y"
        handle.attrs["phase_encoding"] = json.dumps({"semiconductor": 0, "pore": 1})
        handle.attrs["pore_mask_compression"] = "gzip"
        handle.attrs["spacing_A"] = spacing
        handle.attrs["origin_A"] = [0.0, 0.0, 0.0]
        handle.attrs["target_box_A"] = target_box


def _write_channel_curves(path: Path, channel_curves: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = 1
        for unit_id, samples in channel_curves.items():
            handle.create_dataset(unit_id, data=np.asarray(samples, dtype=float))


def _write_instances_csv(path: Path, instance_ids: list[int]) -> None:
    rows = ["instance_id,translation_x_A,translation_y_A,translation_z_A\n"]
    rows.extend(f"{instance_id},0.0,0.0,0.0\n" for instance_id in instance_ids)
    path.write_text("".join(rows), encoding="utf-8")


def _write_placed_atoms_h5(
    path: Path,
    *,
    instance_ids: list[int],
    claimed_count: int,
    density: float,
    pore_volume_A3: float,
    target_box: np.ndarray,
    template_mass: float,
) -> None:
    positions = np.asarray([[float(index), 0.0, 0.0] for index in instance_ids], dtype=float)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = 1
        handle.attrs["status"] = "packed"
        handle.attrs["count"] = claimed_count
        handle.attrs["minimum_interatomic_distance_A"] = math.inf
        handle.attrs["actual_density_g_cm3"] = density
        handle.attrs["pore_volume_A3"] = pore_volume_A3
        handle.attrs["target_box_A"] = target_box
        handle.create_dataset("atom_positions_A", data=positions.reshape((-1, 3)))
        handle.create_dataset("template_positions_A", data=np.asarray([[0.0, 0.0, 0.0]]))
        handle.create_dataset("template_masses_g_mol", data=np.asarray([template_mass]))
        handle.create_dataset("atom_instance_index", data=np.asarray(instance_ids, dtype=np.int64))
        transforms = handle.create_group("instance_transforms")
        unique_ids = sorted(set(instance_ids))
        transforms.create_dataset("translations_A", data=np.zeros((len(unique_ids), 3)))
        transforms.create_dataset(
            "quaternions_xyzw",
            data=np.tile(np.asarray([[0.0, 0.0, 0.0, 1.0]]), (len(unique_ids), 1)),
        )


def _write_cif(path: Path, atom_count: int) -> None:
    lines = ["data_test", "#"]
    lines.extend(
        f"HETATM {index + 1} Ar AR ARG A 1 {float(index):.10f} 0.0000000000 0.0000000000 1 {index} 0 {index + 1}"
        for index in range(atom_count)
    )
    lines.append("#")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _channel_record(unit_id: str, *, cross_radius_A: float) -> dict:
    return {
        "schema_version": 1,
        "unit_id": unit_id,
        "kind": "channel",
        "latent_parameters": {"cross_radius_A": cross_radius_A},
        "realized_geometry": {"anchor_A": [2.0, 1.0, 1.0], "arc_length_A": 4.0},
    }


def _box_mesh(extents: np.ndarray, *, transform: np.ndarray | None = None) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(extents / 2.0)
    if transform is not None:
        mesh.apply_transform(transform)
    return mesh


def _wrong_half_volume_glb() -> trimesh.Trimesh:
    top_half = trimesh.creation.box(extents=(4.0, 4.0, 2.0))
    top_half.apply_translation((2.0, 2.0, 3.0))
    tiny_lower_marker = trimesh.creation.box(extents=(1.0e-6, 1.0e-6, 2.0))
    tiny_lower_marker.apply_translation((5.0e-7, 5.0e-7, 1.0))
    return trimesh.util.concatenate([top_half, tiny_lower_marker])


def _export_glb(path: Path, mesh: trimesh.Trimesh | trimesh.Scene) -> None:
    scene = mesh if isinstance(mesh, trimesh.Scene) else trimesh.Scene(mesh)
    scene.export(path)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_checksums(qa: Path) -> None:
    rows = [
        f"{_sha256(path)}  {path.relative_to(qa).as_posix()}"
        for path in sorted(qa.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (qa / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validator_independently_recomputes_v2_multilobe_compact_volume(
    tmp_path: Path,
) -> None:
    from porous_film.geometry.complex_shapes import generate_multilobe_profile

    target_volume = 4_000.0
    profile = generate_multilobe_profile(target_volume, 2.8, 8811)
    record = {
        "schema_version": 2,
        "unit_id": "compact-v2",
        "kind": "compact",
        "shape_model": "multilobe-v1",
        "shape_seed": 8811,
        "latent_parameters": {"target_volume_A3": target_volume, "eta": 2.8},
        "realized_geometry": {
            "anchor_A": [1.0, 1.0, 1.0],
            "radii_A": profile.envelope_radii_A.tolist(),
            "envelope_radii_A": profile.envelope_radii_A.tolist(),
            "lobe_centers_local_A": profile.lobe_centers_local_A.tolist(),
            "lobe_radii_A": profile.lobe_radii_A.tolist(),
            "smooth_length_A": profile.smooth_length_A,
            "envelope_fill_fraction": profile.envelope_fill_fraction,
            "centroid_offset_A": profile.centroid_offset_A,
            "lobes_connected": profile.connected,
            "lobe_count": profile.lobe_count,
        },
    }
    qa = _write_complete_qa_export(tmp_path, unit_records=[record])

    report = validate_export(qa)

    assert report.status == "PASS", report.errors
    assert report.independent_metrics["units"]["volume_sample_count"] == 1
    assert np.isclose(
        report.independent_metrics["units"]["volume_mean_A3"],
        target_volume,
        rtol=0.03,
    )


def test_validator_reads_v2_channel_group_and_integrates_variable_radius_volume(
    tmp_path: Path,
) -> None:
    unit_id = "channel-v2"
    centerline = np.column_stack([np.linspace(0.0, 6.0, 7), np.ones(7), np.ones(7)])
    radii = np.array([0.60, 0.72, 1.00, 0.88, 0.55, 0.96, 0.70])
    equivalent_radius = 0.8
    record = _channel_v2_record(
        unit_id,
        centerline=centerline,
        radii=radii,
        equivalent_radius_A=equivalent_radius,
    )
    qa = _write_complete_qa_export(tmp_path, unit_records=[record])
    _write_channel_curves_v2(
        qa / "channel_curves.h5",
        {unit_id: (centerline, radii)},
    )
    _write_checksums(qa)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(centerline, axis=0), axis=1)))
    )
    dense_s = np.linspace(0.0, 1.0, 4097)
    dense_radii = PchipInterpolator(
        np.linspace(0.0, 1.0, radii.size),
        radii,
    )(dense_s)
    expected = math.pi * float(cumulative[-1]) * float(np.trapezoid(dense_radii**2, dense_s))
    expected += 2.0 * math.pi * (float(radii[0]) ** 3 + float(radii[-1]) ** 3) / 3.0

    report = validate_export(qa)

    assert report.status == "PASS", report.errors
    assert np.isclose(
        report.independent_metrics["units"]["volume_mean_A3"],
        expected,
        rtol=1.0e-10,
    )


def test_validator_rejects_v2_channel_with_tampered_realized_eta(tmp_path: Path) -> None:
    unit_id = "channel-v2-bad-eta"
    centerline = np.column_stack([np.linspace(0.0, 6.0, 7), np.ones(7), np.ones(7)])
    radii = np.array([0.60, 0.72, 1.00, 0.88, 0.55, 0.96, 0.70])
    record = _channel_v2_record(
        unit_id,
        centerline=centerline,
        radii=radii,
        equivalent_radius_A=0.8,
    )
    record["realized_geometry"]["eta"] = 99.0
    qa = _write_complete_qa_export(tmp_path, unit_records=[record])
    _write_channel_curves_v2(
        qa / "channel_curves.h5",
        {unit_id: (centerline, radii)},
    )
    _write_checksums(qa)

    report = validate_export(qa)

    assert report.status == "FAIL"
    assert "eta" in " ".join(report.errors).lower()


def _channel_v2_record(
    unit_id: str,
    *,
    centerline: np.ndarray,
    radii: np.ndarray,
    equivalent_radius_A: float,
) -> dict:
    arc_length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())
    end_distance = float(np.linalg.norm(centerline[-1] - centerline[0]))
    return {
        "schema_version": 2,
        "unit_id": unit_id,
        "kind": "channel",
        "shape_model": "variable-radius-spline-v1",
        "shape_seed": 2468,
        "latent_parameters": {
            "cross_radius_A": equivalent_radius_A,
            "target_volume_A3": (
                math.pi
                * arc_length
                * float(
                    np.trapezoid(
                        PchipInterpolator(
                            np.linspace(0.0, 1.0, radii.size),
                            radii,
                        )(np.linspace(0.0, 1.0, 4097))
                        ** 2,
                        np.linspace(0.0, 1.0, 4097),
                    )
                )
                + 2.0 * math.pi * (float(radii[0]) ** 3 + float(radii[-1]) ** 3) / 3.0
            ),
            "eta": arc_length / (2.0 * equivalent_radius_A),
            "tau": arc_length / end_distance,
        },
        "realized_geometry": {
            "anchor_A": np.mean(centerline, axis=0).tolist(),
            "control_points_unwrapped_A": centerline.tolist(),
            "arc_length_A": arc_length,
            "end_distance_A": end_distance,
            "eta": arc_length / (2.0 * equivalent_radius_A),
            "tortuosity": arc_length / end_distance,
            "equivalent_radius_A": equivalent_radius_A,
            "radius_profile_s": np.linspace(0.0, 1.0, radii.size).tolist(),
            "radius_profile_A": radii.tolist(),
            "radius_cv": float(
                np.std(
                    PchipInterpolator(
                        np.linspace(0.0, 1.0, radii.size),
                        radii,
                    )(np.linspace(0.0, 1.0, 4097))
                )
                / np.mean(
                    PchipInterpolator(
                        np.linspace(0.0, 1.0, radii.size),
                        radii,
                    )(np.linspace(0.0, 1.0, 4097))
                )
            ),
            "minimum_to_maximum_radius_ratio": float(
                np.min(
                    PchipInterpolator(
                        np.linspace(0.0, 1.0, radii.size),
                        radii,
                    )(np.linspace(0.0, 1.0, 4097))
                )
                / np.max(
                    PchipInterpolator(
                        np.linspace(0.0, 1.0, radii.size),
                        radii,
                    )(np.linspace(0.0, 1.0, 4097))
                )
            ),
            "bend_count": 0,
            "nonplanarity": 0.0,
            "minimum_self_clearance_A": 0.0,
        },
    }


def _write_channel_curves_v2(
    path: Path,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = 2
        for unit_id, (centerline, radii) in curves.items():
            group = handle.create_group(unit_id)
            group.attrs["shape_model"] = "variable-radius-spline-v1"
            group.attrs["shape_seed"] = 2468
            group.create_dataset("centerline_A", data=np.asarray(centerline, dtype=float))
            group.create_dataset("radius_A", data=np.asarray(radii, dtype=float))


def test_validator_rejects_v2_compact_target_volume_mismatch(tmp_path: Path) -> None:
    from porous_film.geometry.complex_shapes import generate_multilobe_profile

    profile = generate_multilobe_profile(4_000.0, 2.8, 8811)
    record = {
        "schema_version": 2,
        "unit_id": "compact-v2-bad-volume",
        "kind": "compact",
        "shape_model": "multilobe-v1",
        "shape_seed": 8811,
        "latent_parameters": {"target_volume_A3": 1.0, "eta": 2.8},
        "realized_geometry": {
            "anchor_A": [1.0, 1.0, 1.0],
            "radii_A": profile.envelope_radii_A.tolist(),
            "envelope_radii_A": profile.envelope_radii_A.tolist(),
            "lobe_centers_local_A": profile.lobe_centers_local_A.tolist(),
            "lobe_radii_A": profile.lobe_radii_A.tolist(),
            "smooth_length_A": profile.smooth_length_A,
            "envelope_fill_fraction": profile.envelope_fill_fraction,
            "centroid_offset_A": profile.centroid_offset_A,
            "lobes_connected": profile.connected,
            "lobe_count": profile.lobe_count,
        },
    }
    qa = _write_complete_qa_export(tmp_path, unit_records=[record])

    report = validate_export(qa)

    assert report.status == "FAIL"
    assert "target_volume" in " ".join(report.errors)


def test_validator_rejects_v2_channel_target_volume_mismatch(tmp_path: Path) -> None:
    unit_id = "channel-v2-bad-volume"
    centerline = np.column_stack([np.linspace(0.0, 6.0, 7), np.ones(7), np.ones(7)])
    radii = np.array([0.60, 0.72, 1.00, 0.88, 0.55, 0.96, 0.70])
    record = _channel_v2_record(
        unit_id,
        centerline=centerline,
        radii=radii,
        equivalent_radius_A=0.8,
    )
    record["latent_parameters"]["target_volume_A3"] = 1.0
    qa = _write_complete_qa_export(tmp_path, unit_records=[record])
    _write_channel_curves_v2(
        qa / "channel_curves.h5",
        {unit_id: (centerline, radii)},
    )
    _write_checksums(qa)

    report = validate_export(qa)

    assert report.status == "FAIL"
    assert "target_volume" in " ".join(report.errors)


def test_validator_rejects_v1_compact_target_volume_mismatch(tmp_path: Path) -> None:
    record = {
        "schema_version": 1,
        "unit_id": "compact-v1-bad-volume",
        "kind": "compact",
        "latent_parameters": {"target_volume_A3": 1.0},
        "realized_geometry": {
            "anchor_A": [1.0, 1.0, 1.0],
            "radii_A": [2.0, 2.0, 2.0],
        },
    }
    qa = _write_complete_qa_export(tmp_path, unit_records=[record])

    report = validate_export(qa)

    assert report.status == "FAIL"
    assert "target_volume" in " ".join(report.errors)


def test_validator_uses_continuous_pchip_radius_cv_not_sparse_nodes(
    tmp_path: Path,
) -> None:
    from porous_film.geometry import ChannelUnit
    from porous_film.geometry.complex_shapes import generate_variable_radius_channel_profile

    target_volume = 8_000.0
    shape_seed = 1
    profile = generate_variable_radius_channel_profile(
        target_volume,
        5.0,
        1.0,
        shape_seed,
    )
    channel = ChannelUnit.from_polyline(
        "channel-pchip-cv",
        profile.control_points_local_A,
        profile.equivalent_radius_A,
        0.0,
        latent_eta=5.0,
        latent_tau=1.0,
        latent_target_volume_A3=target_volume,
        shape_model="variable-radius-spline-v1",
        shape_seed=shape_seed,
        radius_profile_s=profile.radius_profile_s,
        radius_profile_A=profile.radius_profile_A,
        bend_count=profile.bend_count,
        nonplanarity=profile.nonplanarity,
        minimum_self_clearance_A=profile.minimum_self_clearance_A,
    )
    qa = _write_complete_qa_export(
        tmp_path,
        unit_records=[channel.to_record()],
    )
    with h5py.File(qa / "channel_curves.h5", "w") as handle:
        handle.attrs["schema_version"] = 2
        group = handle.create_group(channel.unit_id)
        group.attrs["shape_model"] = channel.shape_model
        group.attrs["shape_seed"] = shape_seed
        group.attrs["equivalent_radius_A"] = channel.cross_radius_A
        group.create_dataset("centerline_A", data=channel.centerline_samples_A)
        group.create_dataset(
            "radius_A",
            data=np.concatenate(
                (
                    channel.segment_start_radii_A,
                    channel.segment_end_radii_A[-1:],
                )
            ),
        )
    _write_checksums(qa)

    report = validate_export(qa)

    assert report.status == "PASS", report.errors


def _write_schema_v3_geometry_export(tmp_path: Path) -> Path:
    from test_pipeline import _pipeline_v3_config_dict, _pipeline_v3_phase_grid

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
    plan = CenterSeedPlan(
        intended_points_A=np.empty((0, 3)),
        target_rdf_xi=np.array([0.0]),
        target_rdf_values=np.array([1.0]),
        starting_loss=0.0,
        initialization_loss=0.0,
    )
    audit = audit_target_distributions(config, built, plan, grid)
    paths = create_task_paths_at_root(tmp_path / "v3")
    identity = CandidateIdentity(29, 0, 0, 31, 0)
    _write_geometry_artifacts(
        config,
        GeometryRun(
            config=config,
            built=built,
            phase_grid=grid,
            audit=audit,
            paths=paths,
            candidate_results=(),
            selected_candidate=identity,
        ),
    )
    return paths.qa_export


def test_validator_accepts_geometry_only_schema_v3_without_molecule_artifacts(
    tmp_path: Path,
) -> None:
    qa = _write_schema_v3_geometry_export(tmp_path)

    report = validate_export(qa)

    errors = " ".join(report.errors).lower()
    assert "molecules" not in errors
    assert "placed_atoms" not in errors
    assert "instances.csv" not in errors
    assert "final_geometry" in report.independent_metrics
    assert report.independent_metrics["final_geometry"]["through_network_count"] >= 1


def test_validator_schema_v3_does_not_gate_constant_radius_generation_control(
    tmp_path: Path,
) -> None:
    qa = _write_schema_v3_geometry_export(tmp_path)
    unit_id = "constant-radius-channel"
    centerline = np.column_stack([np.linspace(0.0, 6.0, 7), np.ones(7), np.ones(7)])
    radii = np.ones(7, dtype=float)
    record = _channel_v2_record(
        unit_id,
        centerline=centerline,
        radii=radii,
        equivalent_radius_A=1.0,
    )
    (qa / "unit_geometry.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    _write_channel_curves_v2(
        qa / "channel_curves.h5",
        {unit_id: (centerline, radii)},
    )
    _write_checksums(qa)

    report = validate_export(qa)

    errors = " ".join(report.errors).lower()
    assert "radius cv" not in errors
    assert "neck or bulge" not in errors


def test_validator_rejects_schema_v3_evidence_tampered_without_phase_change(
    tmp_path: Path,
) -> None:
    qa = _write_schema_v3_geometry_export(tmp_path)
    original_phase_digest = _sha256(qa / "final_phase.h5")

    _shift_schema_v3_centerline_evidence(qa, dx_A=0.25)
    _write_checksums(qa)

    report = validate_export(qa)

    assert _sha256(qa / "final_phase.h5") == original_phase_digest
    assert report.status == "FAIL"
    errors = " ".join(report.errors).lower()
    assert "final_centerlines.h5" in errors
    assert "final_measurements.json" in errors


def _shift_schema_v3_centerline_evidence(qa: Path, *, dx_A: float) -> None:
    with h5py.File(qa / "final_centerlines.h5", "r+") as handle:
        for track in handle["centerlines"].values():
            track["points_wrapped_A"][:, 0] += dx_A
            track["points_unwrapped_A"][:, 0] += dx_A

    measurements_path = qa / "final_measurements.json"
    measurements = json.loads(measurements_path.read_text(encoding="utf-8"))
    for track in measurements["centerlines"]:
        for key in ("points_wrapped_A", "points_unwrapped_A"):
            for point in track[key]:
                point[0] += dx_A
    for section in measurements["cross_sections"]:
        section["center_A"][0] += dx_A
    measurements_path.write_text(
        json.dumps(measurements, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = (
        (qa / "final_cross_sections.csv")
        .read_text(
            encoding="utf-8",
        )
        .splitlines()
    )
    header = rows[0]
    shifted_rows = [header]
    for row in csv.DictReader(rows):
        row["center_x_A"] = str(float(row["center_x_A"]) + dx_A)
        shifted_rows.append(",".join(row[field] for field in (header.split(","))))
    (qa / "final_cross_sections.csv").write_text(
        "\n".join(shifted_rows) + "\n",
        encoding="utf-8",
    )


def test_validator_rejects_empty_schema_v3_final_measurements(tmp_path: Path) -> None:
    qa = _write_schema_v3_geometry_export(tmp_path)
    (qa / "final_measurements.json").write_text("{}\n", encoding="utf-8")
    _write_checksums(qa)

    report = validate_export(qa)

    assert report.status == "FAIL"
    assert "final_measurements.json" in " ".join(report.errors)


def test_validator_requires_final_compact_geometry_evidence(tmp_path: Path) -> None:
    qa = _write_schema_v3_geometry_export(tmp_path)
    measurements_path = qa / "final_measurements.json"
    measurements = json.loads(measurements_path.read_text(encoding="utf-8"))
    measurements.pop("compact_geometries")
    measurements_path.write_text(
        json.dumps(measurements, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_checksums(qa)

    report = validate_export(qa)

    assert report.status == "FAIL"
    assert "compact_geometries" in " ".join(report.errors)


def test_validator_compares_compact_validity_evidence() -> None:
    rebuilt = (
        _V3CompactGeometryMeasurement(
            component_id=1,
            voxel_count=3,
            eta=None,
            valid=False,
            invalid_reason="insufficient_component_voxels",
        ),
    )
    errors: list[str] = []

    _compare_v3_json_compacts(
        [
            {
                "component_id": 1,
                "voxel_count": 3,
                "eta": None,
                "valid": True,
                "invalid_reason": None,
            }
        ],
        rebuilt,
        errors,
    )

    combined = " ".join(errors)
    assert "valid" in combined
    assert "invalid_reason" in combined


def test_validator_rejects_phase_shape_inconsistent_with_spacing_metadata(
    tmp_path: Path,
) -> None:
    qa = _write_schema_v3_geometry_export(tmp_path)
    with h5py.File(qa / "final_phase.h5", "r+") as handle:
        handle.attrs["spacing_A"] = 2.0
    _write_checksums(qa)

    report = validate_export(qa)

    assert report.status == "FAIL"
    errors = " ".join(report.errors).lower()
    assert "mask.shape" in errors
    assert "target_box_a" in errors


def test_validator_rejects_single_tampered_schema_v3_formal_measurement(
    tmp_path: Path,
) -> None:
    qa = _write_schema_v3_geometry_export(tmp_path)
    measurements_path = qa / "final_measurements.json"
    measurements = json.loads(measurements_path.read_text(encoding="utf-8"))
    measurements["cross_sections"][0]["equivalent_diameter_A"] *= 2.0
    measurements_path.write_text(
        json.dumps(measurements, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_checksums(qa)

    report = validate_export(qa)

    assert report.status == "FAIL"
    errors = " ".join(report.errors).lower()
    assert "final_measurements.json" in errors
    assert "equivalent_diameter_a" in errors


def test_validator_rebuilds_gxy_from_through_centerlines_only() -> None:
    phase = _phase_from_slice_centers(
        [
            [(3.0, 3.0)],
            [(3.0, 3.0), (9.0, 9.0)],
            [(3.0, 3.0), (9.0, 9.0)],
            [(3.0, 3.0)],
        ],
        radius_A=1.1,
    )

    measured = _v3_measure_final_geometry(
        phase,
        _V3MeasurementContract(
            center_min_separation_A=1.0,
            center_tracking_max_displacement_A=2.0,
        ),
    )

    assert measured.through_centerline_count == 1
    assert measured.center_distance_xy.pair_count == 0


def test_validator_uses_orientation_aspect_ratio_tolerance() -> None:
    phase = _phase_from_slice_centers([[(6.0, 6.0)]] * 8, radius_A=1.8)

    strict = _v3_measure_final_geometry(
        phase,
        _V3MeasurementContract(orientation_aspect_ratio_tolerance=0.0),
    )
    tolerant = _v3_measure_final_geometry(
        phase,
        _V3MeasurementContract(orientation_aspect_ratio_tolerance=2.0),
    )

    assert strict.projected_orientations[0].theta_xz_identifiable
    assert not tolerant.projected_orientations[0].theta_xz_identifiable


def test_validator_reads_legacy_audit_orientation_aspect_ratio_tolerance() -> None:
    errors: list[str] = []

    contract = _v3_measurement_contract(
        {"audit": {"orientation_aspect_ratio_tolerance": 0.25}},
        errors,
    )

    assert contract.orientation_aspect_ratio_tolerance == 0.25
    assert not errors


def test_validator_excludes_only_local_branch_neighborhood_sections() -> None:
    phase = _phase_from_slice_centers(
        [
            [(6.0, 6.0)],
            [(6.0, 6.0)],
            [(6.0, 6.0)],
            [(6.0, 6.0)],
            [(6.0, 6.0)],
            [(6.0, 6.0)],
            [(5.0, 6.0), (7.0, 6.0)],
            [(4.0, 6.0), (8.0, 6.0)],
            [(4.0, 6.0), (8.0, 6.0)],
        ],
        radius_A=1.4,
    )

    measured = _v3_measure_final_geometry(
        phase,
        _V3MeasurementContract(
            center_min_separation_A=1.0,
            center_tracking_max_displacement_A=3.5,
            cross_section_spacing_A=1.0,
            branch_exclusion_length_A=1.5,
            surface_exclusion_length_A=0.0,
        ),
    )

    branched_ids = {
        track.track_id for track in measured.centerlines if track.has_branch_neighborhood
    }
    assert branched_ids
    assert any(
        section.track_id in branched_ids
        and not section.valid
        and section.invalid_reason == "branch_neighborhood"
        for section in measured.cross_sections
    )
    assert any(
        section.track_id in branched_ids and section.valid and section.invalid_reason is None
        for section in measured.cross_sections
    )
    assert all(
        not channel.valid and channel.invalid_reason == "branch_neighborhood"
        for channel in measured.channel_geometries
        if channel.track_id in branched_ids
    )


def test_validator_rejects_explicit_invalid_schema_v3_measurement_contract(
    tmp_path: Path,
) -> None:
    qa = _write_schema_v3_geometry_export(tmp_path)
    normalized_path = qa / "normalized_config.yaml"
    normalized = yaml.safe_load(normalized_path.read_text(encoding="utf-8"))
    normalized["measurement"]["centerline_sample_spacing_A"] = 0.0
    normalized["measurement"]["branch_exclusion_length_A"] = -1.0
    normalized_path.write_text(yaml.safe_dump(normalized, sort_keys=True), encoding="utf-8")
    _write_checksums(qa)

    report = validate_export(qa)

    assert report.status == "FAIL"
    errors = " ".join(report.errors)
    assert "centerline_sample_spacing_A" in errors
    assert "branch_exclusion_length_A" in errors


def test_validator_v3_target_compliance_uses_rebuilt_diameter_distribution(
    tmp_path: Path,
) -> None:
    qa = _write_schema_v3_geometry_export(tmp_path)
    _sync_schema_v3_targets_to_reported_measurements(qa)
    normalized_path = qa / "normalized_config.yaml"
    normalized = yaml.safe_load(normalized_path.read_text(encoding="utf-8"))
    normalized["formal_targets"]["shape"]["equivalent_diameter_A"]["value"] = 99.0
    normalized["pore_constraints"] = {"z_connectivity": "all_components"}
    normalized_path.write_text(yaml.safe_dump(normalized, sort_keys=True), encoding="utf-8")
    _write_checksums(qa)

    report = validate_export(qa)

    assert report.status == "FAIL"
    assert report.target_compliance["equivalent_diameter_A_within_tolerance"] is False
    assert "equivalent_diameter_A" in " ".join(report.errors)


def test_validator_distribution_contract_supports_weibull_like_main_audit() -> None:
    probabilities = (np.arange(64, dtype=float) + 0.5) / 64.0
    samples = stats.weibull_min(c=2.5, loc=1.0, scale=4.0).ppf(probabilities)

    result = _v3_distribution_target_result(
        samples,
        {"family": "weibull", "shape": 2.5, "loc": 1.0, "scale": 4.0},
    )

    assert result["passed"]


def test_validator_distribution_metrics_match_main_audit_contract() -> None:
    probabilities = (np.arange(64, dtype=float) + 0.5) / 64.0
    cases = [
        (
            stats.lognorm(s=0.4, loc=1.0, scale=3.0).ppf(probabilities),
            {"family": "lognormal", "s": 0.4, "loc": 1.0, "scale": 3.0},
        ),
        (
            stats.gamma(a=2.5, loc=0.5, scale=1.5).ppf(probabilities),
            {"family": "gamma", "alpha": 2.5, "loc": 0.5, "theta": 1.5},
        ),
        (
            stats.weibull_min(c=2.0, loc=1.0, scale=4.0).ppf(probabilities),
            {"family": "weibull_min", "k": 2.0, "loc": 1.0, "scale": 4.0},
        ),
        (
            stats.truncnorm(a=-1.0, b=2.0, loc=5.0, scale=2.0).ppf(probabilities),
            {
                "family": "truncated_normal",
                "mean": 5.0,
                "sigma": 2.0,
                "lower": 3.0,
                "upper": 9.0,
            },
        ),
        (
            stats.beta(a=2.0, b=3.0, loc=10.0, scale=20.0).ppf(probabilities),
            {
                "family": "beta",
                "alpha": 2.0,
                "beta": 3.0,
                "minimum": 10.0,
                "maximum": 30.0,
            },
        ),
    ]

    for samples, target in cases:
        main = compare_samples_to_distribution(samples, target, 0.20, 0.20)
        independent = _v3_distribution_target_result(samples, target)

        assert independent["passed"] == main.passed
        assert np.isclose(independent["ks"], main.ks)
        assert np.isclose(
            independent["normalized_wasserstein"],
            main.normalized_wasserstein,
        )


def test_validator_joint_orientation_rejects_crossed_component_pairs() -> None:
    target = {
        "components": [
            {
                "weight": 0.5,
                "theta_xz_deg": {
                    "family": "beta",
                    "alpha": 2.0,
                    "beta": 2.0,
                    "lower": 70.0,
                    "upper": 80.0,
                },
                "theta_xy_deg": {
                    "family": "beta",
                    "alpha": 2.0,
                    "beta": 2.0,
                    "lower": 0.0,
                    "upper": 10.0,
                },
            },
            {
                "weight": 0.5,
                "theta_xz_deg": {
                    "family": "beta",
                    "alpha": 2.0,
                    "beta": 2.0,
                    "lower": 0.0,
                    "upper": 10.0,
                },
                "theta_xy_deg": {
                    "family": "beta",
                    "alpha": 2.0,
                    "beta": 2.0,
                    "lower": 70.0,
                    "upper": 80.0,
                },
            },
        ]
    }

    result = _v3_compare_paired_orientation_pairs(np.array([[75.0, 75.0], [5.0, 5.0]]), target)

    assert not result["passed"]
    assert result["unassigned_pair_count"] == 2


def test_validator_enforces_matrix_and_final_sample_constraints(tmp_path: Path) -> None:
    qa = _write_schema_v3_geometry_export(tmp_path)
    normalized_path = qa / "normalized_config.yaml"
    normalized = yaml.safe_load(normalized_path.read_text(encoding="utf-8"))
    normalized["matrix_constraints"] = {
        "enabled": True,
        "require_x_percolation": False,
        "minimum_cross_section_fraction": 1.0,
        "maximum_overlap_fraction": 1.0,
    }
    normalized["pore_constraints"] = {
        "z_connectivity": "unrestricted",
        "minimum_through_centerlines": 2,
        "minimum_valid_cross_sections": 100,
    }
    normalized_path.write_text(yaml.safe_dump(normalized, sort_keys=True), encoding="utf-8")
    _write_checksums(qa)

    report = validate_export(qa)

    errors = " ".join(report.errors)
    assert report.status == "FAIL"
    assert "minimum semiconductor cross-section" in errors
    assert "minimum through centerlines" in errors
    assert "minimum valid cross-sections" in errors


def test_validator_unrestricted_skips_through_dependent_target_gates(tmp_path: Path) -> None:
    qa = _write_schema_v3_geometry_export(tmp_path)

    report = validate_export(qa)

    assert not any("schema-v3 target" in error for error in report.errors)
    for name in (
        "equivalent_diameter_A",
        "theta_xz_deg",
        "theta_xy_deg",
        "channel_eta",
        "channel_tau",
        "curvature_fluctuation",
    ):
        assert report.target_compliance[f"{name}_evaluated"] is False
        assert report.target_compliance[f"{name}_within_tolerance"] is None
    assert report.target_compliance["paired_orientation_evaluated"] is False
    assert report.target_compliance["paired_orientation_within_tolerance"] is None
    assert report.target_compliance["g_xy_evaluated"] is False
    assert report.target_compliance["g_xy_within_tolerance"] is None
    assert report.target_compliance["pore_z_connectivity_within_constraint"] is None
    assert report.target_compliance["minimum_through_centerlines_within_constraint"] is None
    assert report.target_compliance["minimum_valid_cross_sections_within_constraint"] is None


def test_validator_enforces_all_components_z_connectivity(tmp_path: Path) -> None:
    qa = _write_schema_v3_geometry_export(tmp_path)
    normalized_path = qa / "normalized_config.yaml"
    normalized = yaml.safe_load(normalized_path.read_text(encoding="utf-8"))
    normalized["pore_constraints"] = {"z_connectivity": "all_components"}
    normalized_path.write_text(yaml.safe_dump(normalized, sort_keys=True), encoding="utf-8")
    with h5py.File(qa / "final_phase.h5", "r+") as handle:
        mask = np.asarray(handle["pore_mask"], dtype=np.uint8)
        mask[5, 0, 9] = 1
        handle["pore_mask"][...] = mask
    _write_checksums(qa)

    report = validate_export(qa)

    assert report.status == "FAIL"
    assert report.target_compliance["pore_z_connectivity_within_constraint"] is False
    assert "all pore components" in " ".join(report.errors)


def test_validator_applies_skeleton_thickness_before_matrix_percolation(
    tmp_path: Path,
) -> None:
    qa = _write_schema_v3_geometry_export(tmp_path)
    normalized_path = qa / "normalized_config.yaml"
    normalized = yaml.safe_load(normalized_path.read_text(encoding="utf-8"))
    normalized["matrix_constraints"] = {
        "enabled": True,
        "require_x_percolation": True,
        "minimum_cross_section_fraction": 0.0,
        "maximum_overlap_fraction": 1.0,
        "minimum_skeleton_thickness_A": 100.0,
    }
    normalized_path.write_text(yaml.safe_dump(normalized, sort_keys=True), encoding="utf-8")
    _write_checksums(qa)

    report = validate_export(qa)

    assert report.status == "FAIL"
    assert report.target_compliance["matrix_x_percolation_within_constraint"] is False
    assert "percolate along periodic x" in " ".join(report.errors)


def test_validator_independently_enforces_maximum_overlap_fraction(tmp_path: Path) -> None:
    qa = _write_schema_v3_geometry_export(tmp_path)
    normalized_path = qa / "normalized_config.yaml"
    normalized = yaml.safe_load(normalized_path.read_text(encoding="utf-8"))
    normalized["matrix_constraints"] = {
        "enabled": True,
        "require_x_percolation": False,
        "minimum_cross_section_fraction": 0.0,
        "maximum_overlap_fraction": 0.0,
    }
    normalized_path.write_text(yaml.safe_dump(normalized, sort_keys=True), encoding="utf-8")
    record = {
        "schema_version": 1,
        "unit_id": "compact-overlap-a",
        "kind": "compact",
        "latent_parameters": {
            "radius_A": 3.0,
            "radii_A": [3.0, 3.0, 3.0],
            "superellipsoid_exponent": 2.0,
            "roughness": 0.0,
            "target_volume_A3": None,
        },
        "realized_geometry": {
            "anchor_A": [3.5, 5.5, 5.0],
            "center_A": [3.5, 5.5, 5.0],
            "radii_A": [3.0, 3.0, 3.0],
            "orientation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }
    duplicate = json.loads(json.dumps(record))
    duplicate["unit_id"] = "compact-overlap-b"
    (qa / "unit_geometry.jsonl").write_text(
        json.dumps(record) + "\n" + json.dumps(duplicate) + "\n",
        encoding="utf-8",
    )
    _write_checksums(qa)

    report = validate_export(qa)

    assert report.status == "FAIL"
    assert report.independent_metrics["units"]["overlap_fraction"] > 0.0
    assert report.target_compliance["pore_overlap_within_constraint"] is False
    assert "overlap fraction" in " ".join(report.errors)


def test_validator_overlap_reconstruction_matches_main_audit() -> None:
    from porous_film.geometry import BuiltGeometry, CompactUnit, PoreGeometry
    from porous_film.metrics.audit import _overlap_fraction
    from porous_film.voxel import voxelize_geometry

    units = [
        CompactUnit.sphere("compact-a", np.array([5.0, 5.0, 5.0]), 3.0),
        CompactUnit.sphere("compact-b", np.array([7.0, 5.0, 5.0]), 3.0),
    ]
    geometry = PoreGeometry(units, np.array([12.0, 12.0, 12.0]))
    built = BuiltGeometry(
        geometry=geometry,
        units=units,
        realized_anchors_A=np.vstack([unit.anchor_A for unit in units]),
        latent_to_realized_ids={},
    )
    grid = voxelize_geometry(geometry, geometry.target_box_A, 1.0)
    phase = _PhaseData(
        mask=grid.pore_mask,
        spacing_A=grid.spacing_A,
        target_box_A=grid.target_box_A,
        origin_A=grid.origin_A,
    )

    main = _overlap_fraction(built, grid, [])
    independent = _independent_overlap_fraction(
        [unit.to_record() for unit in units],
        {},
        phase,
        [],
    )

    assert independent == main


def test_validator_overlap_reconstruction_culls_points_outside_unit_support(
    monkeypatch,
) -> None:
    from porous_film.geometry import CompactUnit
    from porous_film_validator import validate as validator

    units = [
        CompactUnit.sphere("compact-a", np.array([5.0, 5.0, 5.0]), 1.0),
        CompactUnit.sphere("compact-b", np.array([15.0, 15.0, 5.0]), 1.0),
    ]
    phase = _PhaseData(
        mask=np.ones((10, 20, 20), dtype=bool),
        spacing_A=1.0,
        target_box_A=np.array([20.0, 20.0, 10.0]),
        origin_A=np.zeros(3),
    )
    evaluated_point_count = 0
    reference = validator._independent_periodic_unit_field

    def counting_field(*args, **kwargs):
        nonlocal evaluated_point_count
        points = args[2]
        evaluated_point_count += int(points.shape[0])
        return reference(*args, **kwargs)

    monkeypatch.setattr(validator, "_independent_periodic_unit_field", counting_field)

    validator._independent_overlap_fraction(
        [unit.to_record() for unit in units],
        {},
        phase,
        [],
    )

    assert evaluated_point_count < 2 * phase.mask.size


def test_validator_full_schema_v3_execution_does_not_import_main_package(
    tmp_path: Path,
) -> None:
    qa = _write_schema_v3_geometry_export(tmp_path)
    _sync_schema_v3_targets_to_reported_measurements(qa)
    script = f"""
import builtins
from pathlib import Path

real_import = builtins.__import__


def guarded_import(name, *args, **kwargs):
    if name == "porous_film" or name.startswith("porous_film."):
        raise RuntimeError("validator imported main package")
    return real_import(name, *args, **kwargs)


builtins.__import__ = guarded_import
from porous_film_validator.validate import validate_export

report = validate_export(Path({str(qa)!r}))
raise SystemExit(0 if report.status == "PASS" else 2)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _phase_from_slice_centers(
    slice_centers: list[list[tuple[float, float]]],
    *,
    radius_A: float,
    box_A: tuple[float, float] = (12.0, 12.0),
    spacing_A: float = 1.0,
) -> _PhaseData:
    nz = len(slice_centers)
    ny = round(box_A[1] / spacing_A)
    nx = round(box_A[0] / spacing_A)
    yx = np.indices((ny, nx), dtype=float)
    x_A = (yx[1] + 0.5) * spacing_A
    y_A = (yx[0] + 0.5) * spacing_A
    mask = np.zeros((nz, ny, nx), dtype=bool)
    box_xy = np.asarray(box_A, dtype=float)
    for z_index, centers in enumerate(slice_centers):
        for center_x, center_y in centers:
            dx = x_A - center_x
            dy = y_A - center_y
            dx -= box_xy[0] * np.round(dx / box_xy[0])
            dy -= box_xy[1] * np.round(dy / box_xy[1])
            mask[z_index] |= np.hypot(dx, dy) <= radius_A
    return _PhaseData(
        mask=mask,
        spacing_A=spacing_A,
        target_box_A=np.asarray([box_A[0], box_A[1], nz * spacing_A], dtype=float),
        origin_A=np.zeros(3, dtype=float),
    )


def _sync_schema_v3_targets_to_reported_measurements(qa: Path) -> None:
    measurements = json.loads((qa / "final_measurements.json").read_text(encoding="utf-8"))
    normalized_path = qa / "normalized_config.yaml"
    normalized = yaml.safe_load(normalized_path.read_text(encoding="utf-8"))
    shape = normalized["formal_targets"]["shape"]
    normalized["formal_targets"]["position_quantity"]["center_distance_xy"] = None
    valid_sections = [section for section in measurements["cross_sections"] if section["valid"]]
    valid_channels = [channel for channel in measurements["channel_geometries"] if channel["valid"]]
    shape["equivalent_diameter_A"] = {
        "family": "constant",
        "value": valid_sections[0]["equivalent_diameter_A"],
    }
    shape["curvature_fluctuation"] = {
        "family": "constant",
        "value": valid_sections[0]["curvature_fluctuation"],
    }
    shape["channel_aspect_ratio"] = {
        "family": "constant",
        "value": valid_channels[0]["eta"],
    }
    shape["channel_tortuosity"] = {
        "family": "constant",
        "value": valid_channels[0]["tortuosity"],
    }
    shape["orientation"] = None
    normalized_path.write_text(yaml.safe_dump(normalized, sort_keys=True), encoding="utf-8")
    _write_checksums(qa)
