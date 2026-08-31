import os
from pathlib import Path

import pytest
import trimesh
from typer.testing import CliRunner

from porous_film.cli import app

runner = CliRunner()
_RUN_HEAVY = os.environ.get("POROUS_FILM_RUN_HEAVY") == "1"


@pytest.mark.skipif(
    not _RUN_HEAVY,
    reason="end-to-end pore generation requires POROUS_FILM_RUN_HEAVY=1 on the CPU server",
)
def test_minimal_generate_produces_required_outputs(
    minimal_config_path: Path,
    temporary_result_root: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "generate",
            "--config",
            str(minimal_config_path),
            "--result-root",
            str(temporary_result_root),
        ],
    )

    assert result.exit_code == 0
    run_root = Path(result.stdout.strip().splitlines()[-1])
    required = [
        run_root / "outputs" / "semiconductor_solid_target.glb",
        run_root / "outputs" / "pore_material.pdb",
        run_root / "outputs" / "pore_material_high_precision.cif",
        run_root / "outputs" / "molecule_instances.csv",
        run_root / "outputs" / "pore_geometry.h5",
        run_root / "outputs" / "packing_metrics.json",
        run_root / "outputs" / "packmol_handoff.inp",
        run_root / "outputs" / "pore_reference_coordinates.cif",
        run_root / "outputs" / "phase_mapping.json",
        run_root / "outputs" / "pore_atom_indices.ndx",
        run_root / "outputs" / "compression_metadata.json",
        run_root / "analysis" / "target_vs_actual_metrics.json",
        run_root / "reports" / "final-summary.md",
        run_root / "qa_export" / "contract.json",
        run_root / "qa_export" / "normalized_config.yaml",
        run_root / "qa_export" / "unit_candidates.jsonl",
        run_root / "qa_export" / "unit_geometry.jsonl",
        run_root / "qa_export" / "channel_curves.h5",
        run_root / "qa_export" / "final_phase.h5",
        run_root / "qa_export" / "final_surface.ply",
        run_root / "qa_export" / "semiconductor_solid_target.glb",
        run_root / "qa_export" / "main_unit_metrics.csv",
        run_root / "qa_export" / "main_metrics.json",
        run_root / "qa_export" / "molecules" / "source",
        run_root / "qa_export" / "molecules" / "instances.csv",
        run_root / "qa_export" / "molecules" / "placed_atoms.h5",
        run_root / "qa_export" / "molecules" / "placed_structure.cif",
        run_root / "qa_export" / "checksums.sha256",
    ]
    assert all(path.exists() for path in required)
    assert (run_root / "work" / "parallel" / "parallel-plan.json").is_file()
    assert (run_root / "analysis" / "parallel-summary.json").is_file()
    assert (run_root / "reports" / "parallel-summary.md").is_file()

    output_glb = run_root / "outputs" / "semiconductor_solid_target.glb"
    scene = trimesh.load(output_glb, force="scene")
    assert scene.geometry

    output_glb.rename(output_glb.with_suffix(".glb.hidden"))

    from porous_film_validator.validate import validate_export

    validation = validate_export(run_root / "qa_export")
    assert validation.status == "PASS", validation.errors
    assert validation.independent_metrics["phase"]["porosity"] > 0.0
    assert validation.independent_metrics["phase"]["connected_pore_components"] >= 1
    assert validation.independent_metrics["phase"]["semiconductor_x_percolates"] is True
    assert validation.independent_metrics["units"]["total_count"] == 1
    assert validation.independent_metrics["units"]["channel_fraction"] == 0.0
    assert validation.independent_metrics["molecules"]["instance_count"] == 1
    assert validation.independent_metrics["molecules"]["rigid_template_rmsd_A"] == 0.0
    assert validation.independent_metrics["glb"]["bounds_match"] is True
    assert validation.report_consistency["main_metrics_match"] is True
    assert validation.target_compliance["target_box_match"] is True
    assert (run_root / "qa_export" / "independent-validation.json").exists()
    assert (run_root / "qa_export" / "independent-validation-report.md").exists()
