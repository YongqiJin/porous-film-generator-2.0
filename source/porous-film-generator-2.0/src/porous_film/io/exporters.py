from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from skimage import measure
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from porous_film.voxel import PhaseGrid

_SOLID_NAME = "SEMICONDUCTOR_SOLID_TARGET"
_POINT_CHUNK_SIZE = 4096
_FACE_CHUNK_SIZE = 128
_DEFAULT_MAX_INTERSECTION_TESTS = 10_000_000
_REQUIRED_CONTRACT_FIELDS = (
    "length_unit",
    "origin_A",
    "target_box_A",
    "periodic_axes",
    "axis_order",
    "phase_encoding",
)


def export_semiconductor_glb(grid: PhaseGrid, output_path: Path, metadata: dict[str, Any]) -> Path:
    """Export the semiconductor phase as a Blender-readable GLB mesh."""
    mesh = _semiconductor_mesh(grid)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    scene = trimesh.Scene()
    scene.metadata.update(_scene_metadata(grid, metadata))
    scene.add_geometry(mesh, geom_name=_SOLID_NAME, node_name=_SOLID_NAME)
    scene.export(output)
    return output


def export_surface_ply(grid: PhaseGrid, output_path: Path) -> Path:
    """Export the semiconductor surface to neutral PLY for QA tooling."""
    mesh = _semiconductor_mesh(grid)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output)
    return output


def write_qa_contract(contract: dict[str, Any], output_dir: Path) -> Path:
    """Write a stable QA contract and basic neutral sidecar files."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    final_phase_path = directory / "final_phase.h5"
    if not final_phase_path.is_file():
        raise FileNotFoundError("QA contract requires final_phase.h5 in the QA directory")

    normalized = {
        "format_version": 1,
        **_validated_contract(contract),
        "final_phase_file": "final_phase.h5",
    }
    contract_path = directory / "contract.json"
    contract_path.write_text(
        json.dumps(normalized, indent=2, sort_keys=True, separators=(",", ": ")) + "\n",
        encoding="utf-8",
    )

    for name in ("unit_candidates.jsonl", "unit_geometry.jsonl"):
        path = directory / name
        if not path.exists():
            path.write_text("", encoding="utf-8")

    write_qa_checksums(directory)
    return contract_path


def voxelize_exported_glb(
    path: Path,
    grid: PhaseGrid,
    *,
    max_intersection_tests: int = _DEFAULT_MAX_INTERSECTION_TESTS,
) -> np.ndarray:
    """Reconstruct a semiconductor mask by sampling exported mesh normals."""
    scene = trimesh.load(Path(path), force="scene")
    meshes = [geometry for geometry in scene.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
    if not meshes:
        raise ValueError("GLB does not contain a mesh geometry")
    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0].copy()
    if mesh.faces.size == 0:
        return np.zeros_like(grid.pore_mask, dtype=bool)

    work = int(grid.pore_mask.size) * int(mesh.faces.shape[0])
    if work > int(max_intersection_tests):
        raise ValueError(
            "voxelize_exported_glb would require "
            f"{work} point-triangle intersection tests, exceeding "
            f"max_intersection_tests={max_intersection_tests}"
        )

    points = _voxel_centers(grid)
    reconstructed = np.empty(points.shape[0], dtype=bool)
    triangles = np.asarray(mesh.triangles, dtype=float)
    normals = np.asarray(mesh.face_normals, dtype=float)

    for start in range(0, points.shape[0], _POINT_CHUNK_SIZE):
        stop = min(start + _POINT_CHUNK_SIZE, points.shape[0])
        chunk = points[start:stop]
        closest, face_index = _closest_surface_points(chunk, triangles)
        vectors = chunk - closest
        signed = np.einsum("ij,ij->i", vectors, normals[face_index])
        reconstructed[start:stop] = signed < 0.0

    return reconstructed.reshape(grid.pore_mask.shape)


def _validated_contract(contract: dict[str, Any]) -> dict[str, Any]:
    normalized = _json_ready(contract)
    missing = [field for field in _REQUIRED_CONTRACT_FIELDS if field not in normalized]
    if missing:
        raise ValueError(f"QA contract missing required field: {missing[0]}")
    if normalized["length_unit"] != "angstrom":
        raise ValueError("QA contract length_unit must be 'angstrom'")
    _validate_vector(normalized["origin_A"], "origin_A", positive=False)
    _validate_vector(normalized["target_box_A"], "target_box_A", positive=True)
    if list(normalized["periodic_axes"]) != ["x", "y"]:
        raise ValueError("QA contract periodic_axes must be ['x', 'y']")
    if normalized["axis_order"] != "zyx":
        raise ValueError("QA contract axis_order must be 'zyx'")
    if normalized["phase_encoding"] != {"semiconductor": 0, "pore": 1}:
        raise ValueError("QA contract phase_encoding must map semiconductor=0 and pore=1")
    normalized.pop("final_phase_file", None)
    return normalized


def _validate_vector(values: Any, name: str, *, positive: bool) -> None:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"QA contract {name} must contain three finite numbers") from exc
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"QA contract {name} must contain three finite numbers")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"QA contract {name} must contain positive lengths")


def _semiconductor_mesh(grid: PhaseGrid) -> trimesh.Trimesh:
    semiconductor = np.logical_not(np.asarray(grid.pore_mask, dtype=bool))
    if not np.any(semiconductor):
        mesh = trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64))
        mesh.metadata["name"] = _SOLID_NAME
        return mesh

    padded = np.pad(semiconductor.astype(np.float32), 1, mode="constant", constant_values=0.0)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Setting the shape on a NumPy array has been deprecated in NumPy 2\.5\..*",
            category=DeprecationWarning,
            module=r"skimage\.measure\._marching_cubes_lewiner",
        )
        vertices_zyx, faces, _normals, _values = measure.marching_cubes(
            padded,
            level=0.5,
            spacing=(grid.spacing_A, grid.spacing_A, grid.spacing_A),
            allow_degenerate=False,
        )
    vertices_xyz = vertices_zyx[:, [2, 1, 0]] - 0.5 * grid.spacing_A
    vertices_xyz += grid.origin_A
    vertices_xyz = _snap_to_target_box(vertices_xyz, grid)

    mesh = trimesh.Trimesh(vertices=vertices_xyz, faces=faces, process=False)
    mesh.metadata["name"] = _SOLID_NAME
    _clean_mesh(mesh)
    mesh.visual = TextureVisuals(material=_semiconductor_material())
    return mesh


def _clean_mesh(mesh: trimesh.Trimesh) -> None:
    if mesh.faces.size == 0:
        return
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()


def _semiconductor_material() -> PBRMaterial:
    return PBRMaterial(
        name="semi-transparent semiconductor",
        baseColorFactor=[0.25, 0.55, 1.0, 0.42],
        metallicFactor=0.0,
        roughnessFactor=0.35,
        alphaMode="BLEND",
        doubleSided=True,
    )


def _scene_metadata(grid: PhaseGrid, user_metadata: dict[str, Any]) -> dict[str, Any]:
    metadata = _json_ready(user_metadata)
    metadata.setdefault("length_unit", "angstrom")
    metadata["target_box_A"] = _vector_list(grid.target_box_A)
    metadata["periodic_axes"] = list(grid.periodic_axes)
    metadata["porosity"] = float(grid.porosity)
    metadata["mesh_resolution_A"] = float(grid.spacing_A)
    return metadata


def _snap_to_target_box(vertices_xyz: np.ndarray, grid: PhaseGrid) -> np.ndarray:
    lower = np.asarray(grid.origin_A, dtype=float)
    upper = lower + np.asarray(grid.target_box_A, dtype=float)
    snapped = vertices_xyz.copy()
    tolerance = max(float(grid.spacing_A) * 1e-6, 1e-9)
    for axis in range(3):
        near_lower = np.isclose(snapped[:, axis], lower[axis], rtol=0.0, atol=tolerance)
        near_upper = np.isclose(snapped[:, axis], upper[axis], rtol=0.0, atol=tolerance)
        snapped[near_lower, axis] = lower[axis]
        snapped[near_upper, axis] = upper[axis]
    return np.clip(snapped, lower, upper)


def _voxel_centers(grid: PhaseGrid) -> np.ndarray:
    nz, ny, nx = grid.pore_mask.shape
    z, y, x = np.indices((nz, ny, nx), dtype=float)
    return np.column_stack(
        [
            grid.origin_A[0] + (x.ravel() + 0.5) * grid.spacing_A,
            grid.origin_A[1] + (y.ravel() + 0.5) * grid.spacing_A,
            grid.origin_A[2] + (z.ravel() + 0.5) * grid.spacing_A,
        ]
    )


def _closest_surface_points(
    points: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    best_distance = np.full(points.shape[0], np.inf, dtype=float)
    best_points = np.zeros_like(points, dtype=float)
    best_faces = np.zeros(points.shape[0], dtype=np.int64)

    for face_start in range(0, triangles.shape[0], _FACE_CHUNK_SIZE):
        face_stop = min(face_start + _FACE_CHUNK_SIZE, triangles.shape[0])
        tri_chunk = triangles[face_start:face_stop]
        repeated_triangles = np.repeat(tri_chunk, points.shape[0], axis=0)
        tiled_points = np.tile(points, (tri_chunk.shape[0], 1))
        closest = trimesh.triangles.closest_point(repeated_triangles, tiled_points)
        closest = closest.reshape(tri_chunk.shape[0], points.shape[0], 3)
        delta = closest - points[np.newaxis, :, :]
        distances = np.einsum("fpi,fpi->fp", delta, delta)
        local_faces = np.argmin(distances, axis=0)
        local_distances = distances[local_faces, np.arange(points.shape[0])]
        improved = local_distances < best_distance
        if np.any(improved):
            best_distance[improved] = local_distances[improved]
            best_points[improved] = closest[local_faces[improved], np.nonzero(improved)[0]]
            best_faces[improved] = face_start + local_faces[improved]

    return best_points, best_faces


def write_qa_checksums(directory: Path) -> Path:
    checksum_path = Path(directory) / "checksums.sha256"
    entries = sorted(
        path
        for path in Path(directory).rglob("*")
        if path.is_file() and path != checksum_path
    )
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
        f"{path.relative_to(directory).as_posix()}"
        for path in entries
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value


def _vector_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]
