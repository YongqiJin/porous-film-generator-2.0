from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from porous_film.geometry import PoreGeometry
from porous_film.metrics import local_thickness_field
from porous_film.molecules.template import (
    MoleculeTemplate,
    density_for_count,
    molecule_count_for_density,
)
from porous_film.voxel import voxelize_geometry

_PACKING_SCHEMA_VERSION = 1
_DEFAULT_GRID_DIVISIONS = 32
_BOND_CENTERLINE_MAX_STEP_A = 0.2


class PackingError(RuntimeError):
    """Raised when rigid molecule packing cannot produce the requested phase."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class PackingConfig:
    exact_count: int | None = None
    target_density_g_cm3: float | None = None
    minimum_distance_A: float = 2.0
    wall_clearance_A: float = 0.0
    max_attempts: int = 100_000

    def __post_init__(self) -> None:
        if (self.exact_count is None) == (self.target_density_g_cm3 is None):
            raise ValueError("PackingConfig requires exactly one of exact_count or target_density_g_cm3")
        if self.exact_count is not None:
            exact_count = int(self.exact_count)
            if exact_count < 0:
                raise ValueError("exact_count must be nonnegative")
            object.__setattr__(self, "exact_count", exact_count)
        if self.target_density_g_cm3 is not None:
            target_density = _nonnegative_float(
                self.target_density_g_cm3,
                "target_density_g_cm3",
            )
            object.__setattr__(self, "target_density_g_cm3", target_density)
        object.__setattr__(
            self,
            "minimum_distance_A",
            _nonnegative_float(self.minimum_distance_A, "minimum_distance_A"),
        )
        object.__setattr__(
            self,
            "wall_clearance_A",
            _nonnegative_float(self.wall_clearance_A, "wall_clearance_A"),
        )
        max_attempts = int(self.max_attempts)
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        object.__setattr__(self, "max_attempts", max_attempts)


@dataclass(frozen=True)
class InstanceTransform:
    translation_A: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self) -> None:
        translation = _as_vector(self.translation_A, "translation_A")
        quaternion = np.asarray(self.quaternion_xyzw, dtype=float)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("quaternion_xyzw must have shape (4,) and be finite")
        norm = float(np.linalg.norm(quaternion))
        if norm <= 0.0:
            raise ValueError("quaternion_xyzw must have nonzero norm")
        object.__setattr__(self, "translation_A", translation.copy())
        object.__setattr__(self, "quaternion_xyzw", quaternion / norm)


@dataclass(frozen=True)
class PackingResult:
    count: int
    atom_positions_A: np.ndarray
    instance_transforms: tuple[InstanceTransform, ...]
    minimum_interatomic_distance_A: float
    actual_density_g_cm3: float
    protrusion_metrics: dict[str, Any]
    status: str
    template: MoleculeTemplate
    target_box_A: np.ndarray
    pore_volume_A3: float

    def __post_init__(self) -> None:
        positions = np.asarray(self.atom_positions_A, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or not np.all(np.isfinite(positions)):
            raise ValueError("atom_positions_A must have shape (n, 3) and be finite")
        count = int(self.count)
        if count < 0 or len(self.instance_transforms) != count:
            raise ValueError("count must match instance_transforms length")
        if positions.shape[0] != count * self.template.positions_A.shape[0]:
            raise ValueError("atom_positions_A length must match count and template atom count")
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "atom_positions_A", positions.copy())
        object.__setattr__(self, "target_box_A", _as_vector(self.target_box_A, "target_box_A"))
        object.__setattr__(self, "minimum_interatomic_distance_A", float(self.minimum_interatomic_distance_A))
        object.__setattr__(self, "actual_density_g_cm3", float(self.actual_density_g_cm3))
        object.__setattr__(self, "pore_volume_A3", float(self.pore_volume_A3))

    def write_pdb(self, path: Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        atom_index = 0
        atoms_per_instance = len(self.template.elements)
        written_serials: list[list[int]] = []
        for instance_index in range(self.count):
            instance_serials: list[int] = []
            for template_index in range(atoms_per_instance):
                atom_index += 1
                position = self.atom_positions_A[
                    instance_index * atoms_per_instance + template_index
                ]
                serial = (
                    self.template.serial_numbers[template_index]
                    if self.count == 1
                    else atom_index
                )
                instance_serials.append(serial)
                lines.append(
                    _pdb_atom_line(
                        serial=serial,
                        atom_name=self.template.atom_names[template_index],
                        residue_name=self.template.residue_names[template_index],
                        chain_id=self.template.chain_ids[template_index],
                        residue_number=self.template.residue_numbers[template_index],
                        position_A=position,
                        element=self.template.elements[template_index],
                    )
                )
            written_serials.append(instance_serials)
        for instance_index in range(self.count):
            for first, second in self.template.conect_pairs:
                lines.append(
                    f"CONECT{written_serials[instance_index][first]:5d}"
                    f"{written_serials[instance_index][second]:5d}"
                )
        lines.append("END")
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output

    def write_mmcif(self, path: Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        atoms_per_instance = len(self.template.elements)
        lines = [
            "data_porous_film_packed_molecules",
            "#",
            "loop_",
            "_atom_site.group_PDB",
            "_atom_site.id",
            "_atom_site.type_symbol",
            "_atom_site.label_atom_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.label_seq_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.pdbx_PDB_model_num",
            "_atom_site.porous_film_instance_id",
            "_atom_site.porous_film_template_atom_index",
            "_atom_site.porous_film_source_atom_index",
        ]
        atom_id = 0
        for instance_index in range(self.count):
            for template_index in range(atoms_per_instance):
                atom_id += 1
                position = self.atom_positions_A[
                    instance_index * atoms_per_instance + template_index
                ]
                lines.append(
                    "HETATM "
                    f"{atom_id} "
                    f"{self.template.elements[template_index]} "
                    f"{self.template.atom_names[template_index]} "
                    f"{self.template.residue_names[template_index]} "
                    f"{self.template.chain_ids[template_index]} "
                    f"{self.template.residue_numbers[template_index]} "
                    f"{position[0]:.10f} {position[1]:.10f} {position[2]:.10f} "
                    f"1 {instance_index} {template_index} "
                    f"{self.template.source_atom_indices[template_index]}"
                )
        lines.append("#")
        if self.template.conect_pairs:
            lines.extend(
                [
                    "loop_",
                    "_struct_conn.id",
                    "_struct_conn.conn_type_id",
                    "_struct_conn.ptnr1_label_asym_id",
                    "_struct_conn.ptnr1_label_comp_id",
                    "_struct_conn.ptnr1_label_seq_id",
                    "_struct_conn.ptnr1_label_atom_id",
                    "_struct_conn.ptnr2_label_asym_id",
                    "_struct_conn.ptnr2_label_comp_id",
                    "_struct_conn.ptnr2_label_seq_id",
                    "_struct_conn.ptnr2_label_atom_id",
                    "_struct_conn.porous_film_instance_id",
                    "_struct_conn.porous_film_template_atom_index_1",
                    "_struct_conn.porous_film_template_atom_index_2",
                ]
            )
            bond_id = 0
            for instance_index in range(self.count):
                for first, second in self.template.conect_pairs:
                    bond_id += 1
                    lines.append(
                        f"bond{bond_id} covale "
                        f"{self.template.chain_ids[first]} "
                        f"{self.template.residue_names[first]} "
                        f"{self.template.residue_numbers[first]} "
                        f"{self.template.atom_names[first]} "
                        f"{self.template.chain_ids[second]} "
                        f"{self.template.residue_names[second]} "
                        f"{self.template.residue_numbers[second]} "
                        f"{self.template.atom_names[second]} "
                        f"{instance_index} {first} {second}"
                    )
            lines.append("#")
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output

    def write_instances_csv(self, path: Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "instance_index",
                    "translation_x_A",
                    "translation_y_A",
                    "translation_z_A",
                    "quaternion_x",
                    "quaternion_y",
                    "quaternion_z",
                    "quaternion_w",
                ]
            )
            for index, transform in enumerate(self.instance_transforms):
                writer.writerow(
                    [
                        index,
                        *[f"{value:.10f}" for value in transform.translation_A],
                        *[f"{value:.12f}" for value in transform.quaternion_xyzw],
                    ]
                )
        return output

    def write_hdf5(self, path: Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(output, "w") as handle:
            handle.attrs["schema_version"] = _PACKING_SCHEMA_VERSION
            handle.attrs["status"] = self.status
            handle.attrs["count"] = self.count
            handle.attrs["minimum_interatomic_distance_A"] = self.minimum_interatomic_distance_A
            handle.attrs["actual_density_g_cm3"] = self.actual_density_g_cm3
            handle.attrs["pore_volume_A3"] = self.pore_volume_A3
            handle.attrs["target_box_A"] = self.target_box_A
            handle.create_dataset("atom_positions_A", data=self.atom_positions_A)
            handle.create_dataset("template_positions_A", data=self.template.positions_A)
            handle.create_dataset(
                "template_original_positions_A",
                data=self.template.original_positions_A,
            )
            handle.create_dataset(
                "elements",
                data=np.asarray(self.template.elements, dtype=object),
                dtype=string_dtype,
            )
            handle.create_dataset(
                "template_atom_names",
                data=np.asarray(self.template.atom_names, dtype=object),
                dtype=string_dtype,
            )
            handle.create_dataset(
                "template_residue_names",
                data=np.asarray(self.template.residue_names, dtype=object),
                dtype=string_dtype,
            )
            handle.create_dataset(
                "template_chain_ids",
                data=np.asarray(self.template.chain_ids, dtype=object),
                dtype=string_dtype,
            )
            handle.create_dataset("template_residue_numbers", data=self.template.residue_numbers)
            handle.create_dataset("template_serial_numbers", data=self.template.serial_numbers)
            handle.create_dataset("template_masses_g_mol", data=self.template.masses_g_mol)
            handle.create_dataset("template_radii_A", data=self.template.radii_A)
            handle.create_dataset(
                "template_source_atom_indices",
                data=np.asarray(self.template.source_atom_indices, dtype=np.int64),
            )
            handle.create_dataset(
                "template_conect_pairs",
                data=np.asarray(self.template.conect_pairs, dtype=np.int64).reshape(-1, 2),
            )
            atoms_per_instance = len(self.template.elements)
            handle.create_dataset(
                "atom_instance_index",
                data=np.repeat(np.arange(self.count, dtype=np.int64), atoms_per_instance),
            )
            handle.create_dataset(
                "atom_template_index",
                data=np.tile(np.arange(atoms_per_instance, dtype=np.int64), self.count),
            )
            group = handle.create_group("instance_transforms")
            group.create_dataset(
                "translations_A",
                data=np.vstack([transform.translation_A for transform in self.instance_transforms])
                if self.instance_transforms
                else np.empty((0, 3), dtype=float),
            )
            group.create_dataset(
                "quaternions_xyzw",
                data=np.vstack([transform.quaternion_xyzw for transform in self.instance_transforms])
                if self.instance_transforms
                else np.empty((0, 4), dtype=float),
            )
        return output

    def write_metrics_json(self, path: Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            "schema_version": _PACKING_SCHEMA_VERSION,
            "status": self.status,
            "count": self.count,
            "minimum_interatomic_distance_A": self.minimum_interatomic_distance_A,
            "actual_density_g_cm3": self.actual_density_g_cm3,
            "pore_volume_A3": self.pore_volume_A3,
            "protrusion_metrics": _json_ready(self.protrusion_metrics),
        }
        output.write_text(
            json.dumps(metrics, indent=2, sort_keys=True, separators=(",", ": ")) + "\n",
            encoding="utf-8",
        )
        return output


def pack_molecules(
    template: MoleculeTemplate,
    geometry: PoreGeometry,
    config: PackingConfig,
    rng: np.random.Generator,
) -> PackingResult:
    target_box = _as_vector(geometry.target_box_A, "geometry.target_box_A")
    has_empty_units = hasattr(geometry, "units") and not geometry.units
    if has_empty_units and config.exact_count != 0:
        raise PackingError("geometrically_infeasible", "geometry contains no accessible pore")
    sdf_lipschitz_bound = (
        None if has_empty_units else _require_sdf_lipschitz_bound(geometry)
    )
    packing_grid = None if has_empty_units else _PackingSdfGrid.build(geometry, target_box)
    target_count, pore_volume = _target_count_and_volume(
        template,
        geometry,
        config,
        packing_grid=packing_grid,
    )
    if target_count == 0:
        return _empty_result(template, target_box, pore_volume)
    if sdf_lipschitz_bound is None or packing_grid is None:
        raise PackingError("geometrically_infeasible", "geometry contains no accessible pore")
    pore_access = _PoreAccessibilityGrid.build(
        geometry,
        target_box,
        sdf_lipschitz_bound=sdf_lipschitz_bound,
        packing_grid=packing_grid,
    )

    candidates = _candidate_sampler(
        template,
        _PackingGeometryView(geometry, packing_grid),
        config,
        rng,
    )
    placed_positions: list[np.ndarray] = []
    transforms: list[InstanceTransform] = []
    total_attempts = 0
    per_instance_limit = max(config.max_attempts // max(target_count, 1), 1)
    while len(transforms) < target_count and total_attempts < config.max_attempts:
        placement = _sample_valid_placement(
            template=template,
            geometry=geometry,
            config=config,
            rng=rng,
            candidate_sampler=candidates,
            placed_positions=placed_positions,
            target_box=target_box,
            pore_access=pore_access,
            max_attempts=per_instance_limit,
        )
        total_attempts += placement.attempts
        if placement.positions_A is None:
            break
        placed_positions.append(placement.positions_A)
        transforms.append(
            InstanceTransform(
                translation_A=placement.translation_A,
                quaternion_xyzw=placement.rotation.as_quat(),
            )
        )

    if len(transforms) != target_count:
        reason = "algorithm_not_converged"
        raise PackingError(reason, f"failed to pack {target_count} molecule(s): {reason}")

    atom_positions = np.vstack(placed_positions)
    instance_indices = np.repeat(np.arange(target_count, dtype=int), len(template.elements))
    minimum_distance = _minimum_interatomic_distance(atom_positions, instance_indices, target_box)
    return PackingResult(
        count=target_count,
        atom_positions_A=atom_positions,
        instance_transforms=tuple(transforms),
        minimum_interatomic_distance_A=minimum_distance,
        actual_density_g_cm3=density_for_count(template, pore_volume, target_count),
        protrusion_metrics=_protrusion_metrics(atom_positions, target_box),
        status="packed",
        template=template,
        target_box_A=target_box,
        pore_volume_A3=pore_volume,
    )


@dataclass(frozen=True)
class _PlacementAttempt:
    attempts: int
    positions_A: np.ndarray | None
    translation_A: np.ndarray
    rotation: Rotation


@dataclass(frozen=True)
class _PackingSdfGrid:
    sdf_zyx_A: np.ndarray
    spacing_A: np.ndarray
    origin_A: np.ndarray
    target_box_A: np.ndarray

    @staticmethod
    def build(
        geometry: PoreGeometry,
        target_box_A: np.ndarray,
    ) -> _PackingSdfGrid:
        target_box = _as_vector(target_box_A, "target_box_A")
        desired_spacing = _grid_spacing(target_box)
        counts = np.maximum(np.ceil(target_box / desired_spacing).astype(int), 1)
        spacing = target_box / counts
        x = (np.arange(counts[0], dtype=float) + 0.5) * spacing[0]
        y = (np.arange(counts[1], dtype=float) + 0.5) * spacing[1]
        z = (np.arange(counts[2], dtype=float) + 0.5) * spacing[2]
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        sdf = np.asarray(geometry.sdf(points), dtype=float)
        return _PackingSdfGrid(
            sdf_zyx_A=sdf.reshape(
                (int(counts[2]), int(counts[1]), int(counts[0]))
            ),
            spacing_A=spacing,
            origin_A=np.zeros(3, dtype=float),
            target_box_A=target_box,
        )

    @property
    def pore_mask(self) -> np.ndarray:
        return self.sdf_zyx_A < 0.0

    @property
    def voxel_volume_A3(self) -> float:
        return float(np.prod(self.spacing_A))

    def centers_A(self) -> np.ndarray:
        return _voxel_centers(
            self.sdf_zyx_A.shape,
            self.spacing_A,
            self.origin_A,
        )


@dataclass(frozen=True)
class _PackingGeometryView:
    geometry: PoreGeometry
    packing_grid: _PackingSdfGrid

    @property
    def target_box_A(self) -> np.ndarray:
        return self.geometry.target_box_A

    @property
    def units(self) -> Any:
        return self.geometry.units

    @property
    def sdf_lipschitz_bound(self) -> float:
        return float(self.geometry.sdf_lipschitz_bound)

    def sdf(self, points_A: np.ndarray) -> np.ndarray:
        return self.geometry.sdf(points_A)


@dataclass(frozen=True)
class _PoreAccessibilityGrid:
    labels_zyx: np.ndarray
    spacing_A: np.ndarray
    origin_A: np.ndarray
    target_box_A: np.ndarray
    lower_open_labels: frozenset[int]
    upper_open_labels: frozenset[int]

    @staticmethod
    def build(
        geometry: PoreGeometry,
        target_box_A: np.ndarray,
        *,
        sdf_lipschitz_bound: float | None = None,
        packing_grid: _PackingSdfGrid | None = None,
    ) -> _PoreAccessibilityGrid:
        target_box = _as_vector(target_box_A, "target_box_A")
        bound = (
            _require_sdf_lipschitz_bound(geometry)
            if sdf_lipschitz_bound is None
            else _validate_sdf_lipschitz_bound(sdf_lipschitz_bound)
        )
        pore_mask, spacing = _axis_specific_pore_mask(
            geometry,
            target_box,
            bound,
            packing_grid=packing_grid,
        )
        labels, lower_open, upper_open = _label_periodic_xy_components(pore_mask)
        return _PoreAccessibilityGrid(
            labels_zyx=labels,
            spacing_A=spacing,
            origin_A=np.zeros(3, dtype=float),
            target_box_A=target_box,
            lower_open_labels=frozenset(lower_open),
            upper_open_labels=frozenset(upper_open),
        )

    def is_open_to_face(self, point_A: np.ndarray, *, lower: bool) -> bool:
        point = np.asarray(point_A, dtype=float)
        indices_xyz = np.floor((point - self.origin_A) / self.spacing_A).astype(int)
        nx = self.labels_zyx.shape[2]
        ny = self.labels_zyx.shape[1]
        nz = self.labels_zyx.shape[0]
        x = int(np.mod(indices_xyz[0], nx))
        y = int(np.mod(indices_xyz[1], ny))
        z = int(indices_xyz[2])
        if z < 0 or z >= nz:
            return False
        label = int(self.labels_zyx[z, y, x])
        if label < 0:
            return False
        open_labels = self.lower_open_labels if lower else self.upper_open_labels
        return label in open_labels


def _axis_specific_pore_mask(
    geometry: PoreGeometry,
    target_box_A: np.ndarray,
    sdf_lipschitz_bound: float,
    *,
    packing_grid: _PackingSdfGrid | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    target_box = _as_vector(target_box_A, "target_box_A")
    bound = _validate_sdf_lipschitz_bound(sdf_lipschitz_bound)
    grid = packing_grid or _PackingSdfGrid.build(geometry, target_box)
    if not np.allclose(grid.target_box_A, target_box):
        raise ValueError("packing_grid target box does not match geometry target box")
    half_cell_diagonal = 0.5 * float(np.linalg.norm(grid.spacing_A))
    mask = grid.sdf_zyx_A <= -bound * half_cell_diagonal
    return mask, grid.spacing_A.copy()


def _label_periodic_xy_components(mask_zyx: np.ndarray) -> tuple[np.ndarray, set[int], set[int]]:
    mask = np.asarray(mask_zyx, dtype=bool)
    labels = np.full(mask.shape, -1, dtype=np.int64)
    lower_open: set[int] = set()
    upper_open: set[int] = set()
    label = 0
    nz, ny, nx = mask.shape
    for start in np.argwhere(mask):
        z0, y0, x0 = (int(value) for value in start)
        if labels[z0, y0, x0] >= 0:
            continue
        touches_lower = False
        touches_upper = False
        queue: deque[tuple[int, int, int]] = deque([(z0, y0, x0)])
        labels[z0, y0, x0] = label
        while queue:
            z, y, x = queue.popleft()
            touches_lower = touches_lower or z == 0
            touches_upper = touches_upper or z == nz - 1
            for next_z, next_y, next_x in _periodic_xy_neighbors(z, y, x, nz, ny, nx):
                if not mask[next_z, next_y, next_x] or labels[next_z, next_y, next_x] >= 0:
                    continue
                labels[next_z, next_y, next_x] = label
                queue.append((next_z, next_y, next_x))
        if touches_lower:
            lower_open.add(label)
        if touches_upper:
            upper_open.add(label)
        label += 1
    return labels, lower_open, upper_open


def _periodic_xy_neighbors(
    z: int,
    y: int,
    x: int,
    nz: int,
    ny: int,
    nx: int,
) -> tuple[tuple[int, int, int], ...]:
    neighbors = [
        (z, y, (x - 1) % nx),
        (z, y, (x + 1) % nx),
        (z, (y - 1) % ny, x),
        (z, (y + 1) % ny, x),
    ]
    if z > 0:
        neighbors.append((z - 1, y, x))
    if z < nz - 1:
        neighbors.append((z + 1, y, x))
    return tuple(neighbors)


def _target_count_and_volume(
    template: MoleculeTemplate,
    geometry: PoreGeometry,
    config: PackingConfig,
    *,
    packing_grid: _PackingSdfGrid | None = None,
) -> tuple[int, float]:
    pore_volume = _estimate_pore_volume_A3(geometry, packing_grid=packing_grid)
    if config.exact_count is not None:
        return config.exact_count, pore_volume
    return (
        molecule_count_for_density(template, pore_volume, config.target_density_g_cm3),
        pore_volume,
    )


def _estimate_pore_volume_A3(
    geometry: PoreGeometry,
    *,
    packing_grid: _PackingSdfGrid | None = None,
) -> float:
    if packing_grid is not None:
        return float(
            np.count_nonzero(packing_grid.pore_mask)
            * packing_grid.voxel_volume_A3
        )
    target_box = _as_vector(geometry.target_box_A, "geometry.target_box_A")
    spacing = _grid_spacing(target_box)
    try:
        grid = voxelize_geometry(geometry, target_box, spacing)
        return float(np.count_nonzero(grid.pore_mask) * grid.spacing_A**3)
    except ValueError:
        samples_per_axis = 40
        coordinates = [
            np.linspace(0.0, target_box[axis], samples_per_axis, endpoint=False)
            + 0.5 * target_box[axis] / samples_per_axis
            for axis in range(3)
        ]
        x, y, z = np.meshgrid(*coordinates, indexing="ij")
        points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        fraction = float(np.mean(geometry.sdf(points) < 0.0))
        return fraction * float(np.prod(target_box))


def _candidate_sampler(
    template: MoleculeTemplate,
    geometry: PoreGeometry,
    config: PackingConfig,
    rng: np.random.Generator,
) -> Any:
    target_box = _as_vector(geometry.target_box_A, "geometry.target_box_A")

    def sample_uniform() -> np.ndarray:
        return rng.uniform(np.zeros(3, dtype=float), target_box)

    try:
        grid = getattr(geometry, "packing_grid", None) or _PackingSdfGrid.build(
            geometry, target_box
        )
        pore_mask = grid.pore_mask
        if not np.any(pore_mask):
            return sample_uniform
        centers = grid.centers_A()
        pore_centers = centers[pore_mask.ravel()]
        minimum_spacing = float(np.min(grid.spacing_A))
        if len(template.elements) == 1:
            pore_clearance = -grid.sdf_zyx_A.ravel()[pore_mask.ravel()]
            needed_radius = _template_extent_A(template) + config.wall_clearance_A
            weights = np.maximum(
                pore_clearance - needed_radius + minimum_spacing,
                0.0,
            )
        else:
            thickness = local_thickness_field(
                pore_mask,
                minimum_spacing,
                periodic_xy=True,
            )
            pore_thickness = thickness.ravel()[pore_mask.ravel()]
            needed_diameter = (
                2.0 * _template_extent_A(template)
                + 2.0 * config.wall_clearance_A
            )
            weights = np.maximum(
                pore_thickness - needed_diameter + minimum_spacing,
                0.0,
            )
        if not np.any(weights > 0.0):
            weights = np.ones(pore_centers.shape[0], dtype=float)
        weights = weights / float(np.sum(weights))

        def sample_from_grid() -> np.ndarray:
            index = int(rng.choice(pore_centers.shape[0], p=weights))
            jitter = rng.uniform(-0.5 * grid.spacing_A, 0.5 * grid.spacing_A, 3)
            jittered = pore_centers[index] + jitter
            jittered[:2] = np.mod(jittered[:2], target_box[:2])
            jittered[2] = float(np.clip(jittered[2], 0.0, target_box[2]))
            return jittered

        return sample_from_grid
    except ValueError:
        return sample_uniform


def _sample_valid_placement(
    *,
    template: MoleculeTemplate,
    geometry: PoreGeometry,
    config: PackingConfig,
    rng: np.random.Generator,
    candidate_sampler: Any,
    placed_positions: list[np.ndarray],
    target_box: np.ndarray,
    pore_access: _PoreAccessibilityGrid | None,
    max_attempts: int,
) -> _PlacementAttempt:
    identity = Rotation.identity()
    for attempt in range(1, max_attempts + 1):
        translation = candidate_sampler()
        rotation = Rotation.random(random_state=rng)
        positions = _transformed_positions(template, rotation, translation)
        repaired = _repair_if_needed(
            template=template,
            geometry=geometry,
            config=config,
            rng=rng,
            positions=positions,
            translation=translation,
            rotation=rotation,
            placed_positions=placed_positions,
            target_box=target_box,
            pore_access=pore_access,
        )
        if repaired is not None:
            return _PlacementAttempt(
                attempts=attempt,
                positions_A=repaired[0],
                translation_A=repaired[1],
                rotation=repaired[2],
            )
    return _PlacementAttempt(
        attempts=max_attempts,
        positions_A=None,
        translation_A=np.zeros(3, dtype=float),
        rotation=identity,
    )


def _repair_if_needed(
    *,
    template: MoleculeTemplate,
    geometry: PoreGeometry,
    config: PackingConfig,
    rng: np.random.Generator,
    positions: np.ndarray,
    translation: np.ndarray,
    rotation: Rotation,
    placed_positions: list[np.ndarray],
    target_box: np.ndarray,
    pore_access: _PoreAccessibilityGrid | None,
) -> tuple[np.ndarray, np.ndarray, Rotation] | None:
    current_positions = positions
    current_translation = translation
    current_rotation = rotation
    repair_steps = 8
    for step in range(repair_steps + 1):
        if _placement_is_valid(
            positions_A=current_positions,
            translation_A=current_translation,
            template=template,
            geometry=geometry,
            config=config,
            placed_positions=placed_positions,
            target_box=target_box,
            pore_access=pore_access,
        ):
            return current_positions, current_translation, current_rotation
        if step == repair_steps:
            break
        trial_translation = current_translation + rng.normal(0.0, max(config.minimum_distance_A, 1.0), 3)
        trial_translation[:2] = np.mod(trial_translation[:2], target_box[:2])
        trial_translation[2] = float(np.clip(trial_translation[2], 0.0, target_box[2]))
        trial_rotation = Rotation.random(random_state=rng)
        current_translation = trial_translation
        current_rotation = trial_rotation
        current_positions = _transformed_positions(template, trial_rotation, trial_translation)
    return None


def _placement_is_valid(
    *,
    positions_A: np.ndarray,
    translation_A: np.ndarray,
    template: MoleculeTemplate,
    geometry: PoreGeometry,
    config: PackingConfig,
    placed_positions: list[np.ndarray],
    target_box: np.ndarray,
    pore_access: _PoreAccessibilityGrid | None = None,
) -> bool:
    sdf_lipschitz_bound = _require_sdf_lipschitz_bound(geometry)
    if geometry.sdf(translation_A[np.newaxis, :])[0] >= (
        -sdf_lipschitz_bound * config.wall_clearance_A
    ):
        return False
    if not _template_envelopes_respect_walls_and_open_z(
        positions_A,
        template,
        translation_A,
        geometry,
        target_box,
        config.wall_clearance_A,
        pore_access,
    ):
        return False
    if not placed_positions:
        return True
    existing = np.vstack(placed_positions)
    return _no_collision(positions_A, existing, target_box, config.minimum_distance_A)


def _template_envelopes_respect_walls_and_open_z(
    positions_A: np.ndarray,
    template: MoleculeTemplate,
    translation_A: np.ndarray,
    geometry: PoreGeometry,
    target_box: np.ndarray,
    wall_clearance_A: float,
    pore_access: _PoreAccessibilityGrid | None = None,
) -> bool:
    return _atom_envelopes_respect_walls_and_open_z(
        positions_A,
        template.radii_A,
        translation_A,
        geometry,
        target_box,
        wall_clearance_A,
        pore_access,
    ) and _bond_envelopes_respect_walls_and_open_z(
        positions_A,
        template,
        translation_A=translation_A,
        geometry=geometry,
        target_box=target_box,
        wall_clearance_A=wall_clearance_A,
        pore_access=pore_access,
    )


def _atom_envelopes_respect_walls_and_open_z(
    positions_A: np.ndarray,
    radii_A: np.ndarray,
    translation_A: np.ndarray,
    geometry: PoreGeometry,
    target_box: np.ndarray,
    wall_clearance_A: float,
    pore_access: _PoreAccessibilityGrid | None = None,
) -> bool:
    z = positions_A[:, 2]
    sdf_lipschitz_bound = _require_sdf_lipschitz_bound(geometry)
    required_clearance = np.asarray(radii_A, dtype=float) + wall_clearance_A
    if np.any(geometry.sdf(positions_A) > -sdf_lipschitz_bound * required_clearance):
        return False
    if np.any(z - required_clearance < 0.0) and not _com_open_to_z_face(
        translation_A,
        geometry,
        target_box,
        lower=True,
        pore_access=pore_access,
    ):
        return False
    return not (
        np.any(z + required_clearance > target_box[2])
        and not _com_open_to_z_face(
            translation_A,
            geometry,
            target_box,
            lower=False,
            pore_access=pore_access,
        )
    )


def _sampled_atom_envelope_intersection_is_in_pore(
    position_A: np.ndarray,
    radius_A: float,
    geometry: PoreGeometry,
    target_box: np.ndarray,
    wall_clearance_A: float,
) -> bool:
    samples = _atom_envelope_samples_intersecting_target_z(position_A, radius_A, target_box)
    if samples.size == 0:
        return True
    return bool(np.all(geometry.sdf(samples) <= -wall_clearance_A))


def _atom_envelope_samples_intersecting_target_z(
    position_A: np.ndarray,
    radius_A: float,
    target_box: np.ndarray,
) -> np.ndarray:
    directions = _sphere_sample_directions()
    surface = position_A[np.newaxis, :] + radius_A * directions
    samples = [surface[(surface[:, 2] >= 0.0) & (surface[:, 2] <= target_box[2])]]
    lower_cap = _sphere_plane_intersection_samples(position_A, radius_A, z_A=0.0)
    upper_cap = _sphere_plane_intersection_samples(position_A, radius_A, z_A=target_box[2])
    if lower_cap.size:
        samples.append(lower_cap)
    if upper_cap.size:
        samples.append(upper_cap)
    return np.vstack([sample for sample in samples if sample.size]) if any(sample.size for sample in samples) else np.empty((0, 3), dtype=float)


def _sphere_sample_directions() -> np.ndarray:
    axes = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=float,
    )
    return axes / np.linalg.norm(axes, axis=1)[:, np.newaxis]


def _sphere_plane_intersection_samples(
    position_A: np.ndarray,
    radius_A: float,
    *,
    z_A: float,
) -> np.ndarray:
    offset_z = z_A - float(position_A[2])
    if abs(offset_z) > radius_A:
        return np.empty((0, 3), dtype=float)
    circle_radius = float(np.sqrt(max(radius_A**2 - offset_z**2, 0.0)))
    if circle_radius <= 1e-12:
        return np.array([[position_A[0], position_A[1], z_A]], dtype=float)
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    return np.column_stack(
        [
            position_A[0] + circle_radius * np.cos(angles),
            position_A[1] + circle_radius * np.sin(angles),
            np.full(angles.shape, z_A, dtype=float),
        ]
    )


def _bond_envelopes_respect_walls_and_open_z(
    positions_A: np.ndarray,
    template: MoleculeTemplate,
    *,
    translation_A: np.ndarray,
    geometry: PoreGeometry,
    target_box: np.ndarray,
    wall_clearance_A: float,
    pore_access: _PoreAccessibilityGrid | None = None,
) -> bool:
    for first, second in template.conect_pairs:
        if not _bond_segment_envelope_respects_walls_and_open_z(
            start_A=positions_A[first],
            end_A=positions_A[second],
            start_radius_A=float(template.radii_A[first]),
            end_radius_A=float(template.radii_A[second]),
            translation_A=translation_A,
            geometry=geometry,
            target_box=target_box,
            wall_clearance_A=wall_clearance_A,
            pore_access=pore_access,
        ):
            return False
    return True


def _bond_segment_envelope_respects_walls_and_open_z(
    *,
    start_A: np.ndarray,
    end_A: np.ndarray,
    start_radius_A: float,
    end_radius_A: float,
    translation_A: np.ndarray,
    geometry: PoreGeometry,
    target_box: np.ndarray,
    wall_clearance_A: float,
    pore_access: _PoreAccessibilityGrid | None = None,
) -> bool:
    segment = end_A - start_A
    length = float(np.linalg.norm(segment))
    if length <= 1e-12:
        samples = start_A[np.newaxis, :]
        radii = np.array([max(start_radius_A, end_radius_A)], dtype=float)
        sample_spacing = 0.0
    else:
        intervals = max(int(np.ceil(length / _BOND_CENTERLINE_MAX_STEP_A)), 1)
        fractions = np.linspace(0.0, 1.0, intervals + 1)
        samples = start_A[np.newaxis, :] + fractions[:, np.newaxis] * segment[np.newaxis, :]
        radii = (1.0 - fractions) * start_radius_A + fractions * end_radius_A
        sample_spacing = length / intervals

    z = samples[:, 2]
    sdf_lipschitz_bound = _require_sdf_lipschitz_bound(geometry)
    capsule_radii = radii + wall_clearance_A + 0.5 * sample_spacing
    conservative_clearance = sdf_lipschitz_bound * capsule_radii
    if np.any(geometry.sdf(samples) > -conservative_clearance):
        return False
    if np.any(z - capsule_radii < 0.0) and not _com_open_to_z_face(
        translation_A,
        geometry,
        target_box,
        lower=True,
        pore_access=pore_access,
    ):
        return False
    return not (
        np.any(z + capsule_radii > target_box[2])
        and not _com_open_to_z_face(
            translation_A,
            geometry,
            target_box,
            lower=False,
            pore_access=pore_access,
        )
    )


def _bond_plane_intersection_samples(
    start_A: np.ndarray,
    end_A: np.ndarray,
    start_radius_A: float,
    end_radius_A: float,
    *,
    face_z_A: float,
) -> np.ndarray:
    z0 = float(start_A[2])
    z1 = float(end_A[2])
    if abs(z1 - z0) <= 1e-12:
        return np.empty((0, 3), dtype=float)
    fraction = (face_z_A - z0) / (z1 - z0)
    if fraction < 0.0 or fraction > 1.0:
        return np.empty((0, 3), dtype=float)
    center = (1.0 - fraction) * start_A + fraction * end_A
    radius = (1.0 - fraction) * start_radius_A + fraction * end_radius_A
    tangent = end_A - start_A
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-12 or radius <= 1e-12:
        return center[np.newaxis, :]
    tangent = tangent / tangent_norm
    reference = np.array([1.0, 0.0, 0.0]) if abs(tangent[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first_axis = np.cross(tangent, reference)
    first_axis /= np.linalg.norm(first_axis)
    second_axis = np.cross(tangent, first_axis)
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    return center[np.newaxis, :] + radius * (
        np.cos(angles)[:, np.newaxis] * first_axis[np.newaxis, :]
        + np.sin(angles)[:, np.newaxis] * second_axis[np.newaxis, :]
    )


def _com_open_to_z_face(
    translation_A: np.ndarray,
    geometry: PoreGeometry,
    target_box: np.ndarray,
    *,
    lower: bool,
    pore_access: _PoreAccessibilityGrid | None = None,
) -> bool:
    if pore_access is not None:
        return pore_access.is_open_to_face(translation_A, lower=lower)
    return _PoreAccessibilityGrid.build(geometry, target_box).is_open_to_face(
        translation_A,
        lower=lower,
    )


def _no_collision(
    candidate_positions_A: np.ndarray,
    existing_positions_A: np.ndarray,
    target_box: np.ndarray,
    minimum_distance_A: float,
) -> bool:
    if minimum_distance_A == 0.0 or existing_positions_A.size == 0:
        return True
    delta = candidate_positions_A[:, np.newaxis, :] - existing_positions_A[np.newaxis, :, :]
    delta[:, :, 0] -= target_box[0] * np.rint(delta[:, :, 0] / target_box[0])
    delta[:, :, 1] -= target_box[1] * np.rint(delta[:, :, 1] / target_box[1])
    distances = np.linalg.norm(delta, axis=2)
    return bool(np.all(distances >= minimum_distance_A))


def _minimum_interatomic_distance(
    atom_positions_A: np.ndarray,
    instance_indices: np.ndarray,
    target_box: np.ndarray,
) -> float:
    best = np.inf
    for index in range(atom_positions_A.shape[0]):
        delta = atom_positions_A[index + 1 :] - atom_positions_A[index]
        if delta.size == 0:
            continue
        other_instances = instance_indices[index + 1 :] != instance_indices[index]
        if not np.any(other_instances):
            continue
        delta = delta[other_instances]
        delta[:, 0] -= target_box[0] * np.rint(delta[:, 0] / target_box[0])
        delta[:, 1] -= target_box[1] * np.rint(delta[:, 1] / target_box[1])
        distances = np.linalg.norm(delta, axis=1)
        best = min(best, float(np.min(distances)))
    return best


def _transformed_positions(
    template: MoleculeTemplate,
    rotation: Rotation,
    translation_A: np.ndarray,
) -> np.ndarray:
    return rotation.apply(template.positions_A) + translation_A


def _template_extent_A(template: MoleculeTemplate) -> float:
    return float(np.max(np.linalg.norm(template.positions_A, axis=1) + template.radii_A))


def _protrusion_metrics(atom_positions_A: np.ndarray, target_box: np.ndarray) -> dict[str, Any]:
    z = atom_positions_A[:, 2]
    lower = np.maximum(-z, 0.0)
    upper = np.maximum(z - target_box[2], 0.0)
    return {
        "atoms_below_z": int(np.count_nonzero(lower > 0.0)),
        "atoms_above_z": int(np.count_nonzero(upper > 0.0)),
        "max_lower_protrusion_A": float(np.max(lower)) if lower.size else 0.0,
        "max_upper_protrusion_A": float(np.max(upper)) if upper.size else 0.0,
    }


def _empty_result(template: MoleculeTemplate, target_box_A: np.ndarray, pore_volume_A3: float) -> PackingResult:
    return PackingResult(
        count=0,
        atom_positions_A=np.empty((0, 3), dtype=float),
        instance_transforms=(),
        minimum_interatomic_distance_A=np.inf,
        actual_density_g_cm3=0.0,
        protrusion_metrics={
            "atoms_below_z": 0,
            "atoms_above_z": 0,
            "max_lower_protrusion_A": 0.0,
            "max_upper_protrusion_A": 0.0,
        },
        status="packed",
        template=template,
        target_box_A=target_box_A,
        pore_volume_A3=pore_volume_A3,
    )


def _voxel_centers(
    shape_or_mask_zyx: tuple[int, int, int] | np.ndarray,
    spacing_A: float | np.ndarray,
    origin_A: np.ndarray,
) -> np.ndarray:
    shape = (
        shape_or_mask_zyx.shape
        if isinstance(shape_or_mask_zyx, np.ndarray)
        else tuple(int(value) for value in shape_or_mask_zyx)
    )
    nz, ny, nx = shape
    spacing = np.asarray(spacing_A, dtype=float)
    if spacing.ndim == 0:
        spacing = np.full(3, float(spacing), dtype=float)
    z, y, x = np.indices((nz, ny, nx), dtype=float)
    return np.column_stack(
        [
            origin_A[0] + (x.ravel() + 0.5) * spacing[0],
            origin_A[1] + (y.ravel() + 0.5) * spacing[1],
            origin_A[2] + (z.ravel() + 0.5) * spacing[2],
        ]
    )


def _grid_spacing(target_box_A: np.ndarray) -> float:
    return float(np.min(target_box_A) / _DEFAULT_GRID_DIVISIONS)


def _pdb_atom_line(
    *,
    serial: int,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_number: int,
    position_A: np.ndarray,
    element: str,
) -> str:
    return (
        f"HETATM{serial:5d} {atom_name[:4]:>4s} {residue_name[:3]:>3s} "
        f"{(chain_id or 'A')[:1]:1s}{residue_number:4d}    "
        f"{position_A[0]:8.3f}{position_A[1]:8.3f}{position_A[2]:8.3f}"
        f"  1.00  0.00          {element:>2s}"
    )


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


def _require_sdf_lipschitz_bound(geometry: PoreGeometry) -> float:
    if not hasattr(geometry, "sdf_lipschitz_bound"):
        raise ValueError("geometry.sdf_lipschitz_bound must be a positive finite value")
    return _validate_sdf_lipschitz_bound(geometry.sdf_lipschitz_bound)


def _validate_sdf_lipschitz_bound(value: float) -> float:
    bound = float(value)
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("geometry.sdf_lipschitz_bound must be a positive finite value")
    return bound


def _as_vector(vector_A: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(vector_A, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    if np.any(vector <= 0.0) and "box" in name:
        raise ValueError(f"{name} must contain positive lengths")
    return vector


def _nonnegative_float(value: float, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite value")
    return parsed
