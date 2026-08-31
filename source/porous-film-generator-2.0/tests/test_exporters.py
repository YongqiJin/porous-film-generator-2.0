import json
import struct
from pathlib import Path

import numpy as np
import pytest
import trimesh

from porous_film.io import (
    export_semiconductor_glb,
    export_surface_ply,
    voxelize_exported_glb,
    write_qa_contract,
)
from porous_film.voxel import PhaseGrid


def test_glb_uses_target_box_and_contains_pore_cavity(tmp_path: Path) -> None:
    pore = np.zeros((10, 10, 10), dtype=bool)
    pore[3:7, 3:7, 3:7] = True
    grid = PhaseGrid(
        pore_mask=pore,
        origin_A=np.zeros(3),
        spacing_A=1.0,
        target_box_A=np.array([10.0, 10.0, 10.0]),
    )
    path = tmp_path / "semiconductor_solid_target.glb"

    export_semiconductor_glb(grid, path, {"length_unit": "angstrom", "task_id": "task-7"})
    scene = trimesh.load(path, force="scene")
    bounds = scene.bounds
    glb_json = _read_glb_json(path)
    mesh = scene.geometry["SEMICONDUCTOR_SOLID_TARGET"]
    material = glb_json["materials"][0]
    base_color = material["pbrMetallicRoughness"]["baseColorFactor"]

    assert path.exists()
    assert np.allclose(bounds[0], [0.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(bounds[1], [10.0, 10.0, 10.0], atol=1e-6)
    solid_volume = sum(abs(mesh.volume) for mesh in scene.geometry.values())
    assert 0.0 < solid_volume < 1000.0
    assert "SEMICONDUCTOR_SOLID_TARGET" in scene.geometry
    assert scene.metadata["length_unit"] == "angstrom"
    assert scene.metadata["target_box_A"] == [10.0, 10.0, 10.0]
    assert scene.metadata["periodic_axes"] == ["x", "y"]
    assert scene.metadata["porosity"] == grid.porosity
    assert scene.metadata["mesh_resolution_A"] == 1.0
    assert scene.metadata["task_id"] == "task-7"
    assert glb_json["meshes"][0]["name"] == "SEMICONDUCTOR_SOLID_TARGET"
    assert glb_json["nodes"][0]["name"] == "SEMICONDUCTOR_SOLID_TARGET"
    assert glb_json["scenes"][0]["extras"] == scene.metadata
    assert material["alphaMode"] == "BLEND"
    assert material["doubleSided"] is True
    assert 0.0 < base_color[3] < 1.0
    assert np.all(mesh.nondegenerate_faces())
    assert np.all(mesh.unique_faces())
    assert mesh.is_watertight
    assert len(trimesh.repair.broken_faces(mesh)) == 0


def test_glb_preserves_open_z_and_periodic_x_pore_openings(tmp_path: Path) -> None:
    pore = np.zeros((10, 10, 10), dtype=bool)
    pore[:, 4:6, 4:6] = True
    pore[4:6, 4:6, 0] = True
    pore[4:6, 4:6, -1] = True
    pore[4:6, 0, 4:6] = True
    pore[4:6, -1, 4:6] = True
    grid = PhaseGrid(
        pore_mask=pore,
        origin_A=np.zeros(3),
        spacing_A=1.0,
        target_box_A=np.array([10.0, 10.0, 10.0]),
    )
    path = tmp_path / "open-periodic.glb"

    export_semiconductor_glb(
        grid,
        path,
        {
            "length_unit": "angstrom",
            "periodic_axes": ["x", "y"],
            "target_box_A": [10.0, 10.0, 10.0],
        },
    )
    reconstructed_semiconductor = voxelize_exported_glb(path, grid)
    expected_semiconductor = np.logical_not(pore)
    mismatch = np.mean(reconstructed_semiconductor != expected_semiconductor)

    assert mismatch <= 0.05
    assert not reconstructed_semiconductor[0, 4:6, 4:6].any()
    assert not reconstructed_semiconductor[-1, 4:6, 4:6].any()
    assert not reconstructed_semiconductor[:, 4:6, 4:6].any()
    assert not reconstructed_semiconductor[4:6, 4:6, 0].any()
    assert not reconstructed_semiconductor[4:6, 4:6, -1].any()
    assert not reconstructed_semiconductor[4:6, 0, 4:6].any()
    assert not reconstructed_semiconductor[4:6, -1, 4:6].any()


def test_surface_ply_and_qa_contract_are_neutral_files(tmp_path: Path) -> None:
    pore = np.zeros((4, 4, 4), dtype=bool)
    pore[1:3, 1:3, 1:3] = True
    grid = PhaseGrid(
        pore_mask=pore,
        origin_A=np.zeros(3),
        spacing_A=1.0,
        target_box_A=np.array([4.0, 4.0, 4.0]),
    )

    ply_path = export_surface_ply(grid, tmp_path / "final_surface.ply")
    loaded = trimesh.load(ply_path, force="mesh")
    qa_dir = tmp_path / "qa_export"
    qa_dir.mkdir()
    grid.write_hdf5(qa_dir / "final_phase.h5")
    contract_path = write_qa_contract(
        {
            "length_unit": "angstrom",
            "origin_A": [0.0, 0.0, 0.0],
            "target_box_A": [4.0, 4.0, 4.0],
            "periodic_axes": ["x", "y"],
            "axis_order": "zyx",
            "phase_encoding": {"semiconductor": 0, "pore": 1},
        },
        qa_dir,
    )

    assert ply_path.exists()
    assert loaded.vertices.shape[0] > 0
    assert loaded.faces.shape[0] > 0
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    checksum_lines = (qa_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines()

    assert contract_path.name == "contract.json"
    assert contract_path.read_text(encoding="utf-8").endswith("\n")
    assert contract["final_phase_file"] == "final_phase.h5"
    assert any(line.endswith("  contract.json") for line in checksum_lines)
    assert any(line.endswith("  final_phase.h5") for line in checksum_lines)
    assert any(line.endswith("  unit_candidates.jsonl") for line in checksum_lines)
    assert any(line.endswith("  unit_geometry.jsonl") for line in checksum_lines)


def test_write_qa_contract_requires_final_phase_and_schema_fields(tmp_path: Path) -> None:
    qa_dir = tmp_path / "qa_export"
    qa_dir.mkdir()
    valid_contract = {
        "length_unit": "angstrom",
        "origin_A": [0.0, 0.0, 0.0],
        "target_box_A": [4.0, 4.0, 4.0],
        "periodic_axes": ["x", "y"],
        "axis_order": "zyx",
        "phase_encoding": {"semiconductor": 0, "pore": 1},
    }

    with pytest.raises(FileNotFoundError, match="final_phase.h5"):
        write_qa_contract(valid_contract, qa_dir)

    (qa_dir / "final_phase.h5").write_bytes(b"placeholder")
    incomplete = dict(valid_contract)
    incomplete.pop("periodic_axes")

    with pytest.raises(ValueError, match="periodic_axes"):
        write_qa_contract(incomplete, qa_dir)


def test_voxelize_exported_glb_rejects_unbounded_work(tmp_path: Path) -> None:
    pore = np.zeros((4, 4, 4), dtype=bool)
    pore[1:3, 1:3, 1:3] = True
    grid = PhaseGrid(
        pore_mask=pore,
        origin_A=np.zeros(3),
        spacing_A=1.0,
        target_box_A=np.array([4.0, 4.0, 4.0]),
    )
    path = export_semiconductor_glb(grid, tmp_path / "bounded.glb", {"length_unit": "angstrom"})

    with pytest.raises(ValueError, match="max_intersection_tests"):
        voxelize_exported_glb(path, grid, max_intersection_tests=1)


def _read_glb_json(path: Path) -> dict:
    data = path.read_bytes()
    magic, version, _length = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF"
    assert version == 2
    json_length, chunk_type = struct.unpack_from("<I4s", data, 12)
    assert chunk_type == b"JSON"
    return json.loads(data[20 : 20 + json_length].rstrip(b" \x00"))
