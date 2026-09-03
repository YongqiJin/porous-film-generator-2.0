from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import h5py
import numpy as np

from porous_film.geometry import BuiltGeometry, PoreGeometry
from porous_film.performance import profile_stage

_SCHEMA_VERSION = 1
_AXIS_ORDER = "zyx"
_PERIODIC_AXES = ("x", "y")
_PHASE_ENCODING = {"semiconductor": 0, "pore": 1}
_SCALE_SOLVER_GRID_DIVISIONS = 40
_SCALE_SOLVER_MAX_ITERATIONS = 64
_GRID_DIVISIBILITY_TOLERANCE = 1e-9


class PorosityResolutionError(ValueError):
    """Raised when the voxel grid cannot resolve the requested porosity tolerance."""


@dataclass(frozen=True)
class PhaseGrid:
    pore_mask: np.ndarray
    origin_A: np.ndarray
    spacing_A: float
    target_box_A: np.ndarray
    axis_order: str = _AXIS_ORDER
    periodic_axes: tuple[str, ...] = _PERIODIC_AXES
    phase_encoding: dict[str, int] | None = None
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        mask = np.asarray(self.pore_mask)
        if mask.ndim != 3:
            raise ValueError("pore_mask must have shape (z, y, x)")
        _validate_binary_mask_values(mask, "pore_mask")
        origin = _as_vector(self.origin_A, "origin_A")
        target_box = _as_box(self.target_box_A)
        spacing = _positive_float(self.spacing_A, "spacing_A")
        counts_xyz = _grid_counts_xyz(target_box, spacing)
        expected_shape = (int(counts_xyz[2]), int(counts_xyz[1]), int(counts_xyz[0]))
        if mask.shape != expected_shape:
            raise ValueError(
                "pore_mask shape in z,y,x order must match target_box_A/spacing_A counts; "
                f"expected {expected_shape}, got {mask.shape}"
            )
        if self.axis_order != _AXIS_ORDER:
            raise ValueError("PhaseGrid axis_order must be 'zyx'")
        if tuple(self.periodic_axes) != _PERIODIC_AXES:
            raise ValueError("PhaseGrid periodic_axes must be ('x', 'y')")
        phase_encoding = self.phase_encoding or _PHASE_ENCODING.copy()
        if phase_encoding != _PHASE_ENCODING:
            raise ValueError("PhaseGrid phase_encoding must map semiconductor=0 and pore=1")
        if int(self.schema_version) != _SCHEMA_VERSION:
            raise ValueError(f"unsupported PhaseGrid schema_version: {self.schema_version}")

        object.__setattr__(self, "pore_mask", mask.astype(bool, copy=True))
        object.__setattr__(self, "origin_A", origin.copy())
        object.__setattr__(self, "spacing_A", spacing)
        object.__setattr__(self, "target_box_A", target_box.copy())
        object.__setattr__(self, "periodic_axes", _PERIODIC_AXES)
        object.__setattr__(self, "phase_encoding", phase_encoding.copy())
        object.__setattr__(self, "schema_version", _SCHEMA_VERSION)

    @property
    def porosity(self) -> float:
        return float(np.mean(self.pore_mask))

    def write_hdf5(self, path: Path) -> None:
        output_path = Path(path)
        with h5py.File(output_path, "w") as handle:
            handle.create_dataset(
                "pore_mask",
                data=self.pore_mask.astype(np.uint8),
                compression="gzip",
                shuffle=True,
            )
            handle.attrs["schema_version"] = self.schema_version
            handle.attrs["axis_order"] = self.axis_order
            handle.attrs["periodic_axes"] = ",".join(self.periodic_axes)
            handle.attrs["phase_encoding"] = json.dumps(self.phase_encoding, sort_keys=True)
            handle.attrs["pore_mask_compression"] = "gzip"
            handle.attrs["spacing_A"] = self.spacing_A
            handle.attrs["origin_A"] = self.origin_A
            handle.attrs["target_box_A"] = self.target_box_A

    @staticmethod
    def read_hdf5(path: Path) -> PhaseGrid:
        input_path = Path(path)
        with h5py.File(input_path, "r") as handle:
            if "pore_mask" not in handle:
                raise ValueError("malformed PhaseGrid HDF5: missing pore_mask dataset")
            dataset = handle["pore_mask"]
            if dataset.compression != "gzip":
                raise ValueError("malformed PhaseGrid HDF5: pore_mask compression must be gzip")
            raw_mask = np.asarray(dataset)
            _validate_binary_mask_values(raw_mask, "malformed PhaseGrid HDF5: pore_mask")

            schema_version = int(_required_attr(handle, "schema_version"))
            axis_order = _read_text_attr(_required_attr(handle, "axis_order"))
            periodic_axes = tuple(
                _read_text_attr(_required_attr(handle, "periodic_axes")).split(",")
            )
            phase_encoding = json.loads(_read_text_attr(_required_attr(handle, "phase_encoding")))
            compression_metadata = _read_text_attr(_required_attr(handle, "pore_mask_compression"))
            if compression_metadata != "gzip":
                raise ValueError(
                    "malformed PhaseGrid HDF5: pore_mask compression metadata must be gzip"
                )
            return PhaseGrid(
                pore_mask=raw_mask.astype(bool),
                origin_A=np.asarray(_required_attr(handle, "origin_A"), dtype=float),
                spacing_A=float(_required_attr(handle, "spacing_A")),
                target_box_A=np.asarray(_required_attr(handle, "target_box_A"), dtype=float),
                axis_order=axis_order,
                periodic_axes=periodic_axes,
                phase_encoding={str(key): int(value) for key, value in phase_encoding.items()},
                schema_version=schema_version,
            )


def voxelize_geometry(
    geometry: PoreGeometry,
    box_A: np.ndarray,
    spacing_A: float,
    max_points_per_chunk: int = 1_000_000,
) -> PhaseGrid:
    with profile_stage("voxelization"):
        return _voxelize_geometry(geometry, box_A, spacing_A, max_points_per_chunk)


def _voxelize_geometry(
    geometry: PoreGeometry,
    box_A: np.ndarray,
    spacing_A: float,
    max_points_per_chunk: int,
) -> PhaseGrid:
    target_box = _as_box(box_A)
    spacing = _positive_float(spacing_A, "spacing_A")
    chunk_size = _positive_int(max_points_per_chunk, "max_points_per_chunk")
    counts_xyz = _grid_counts_xyz(target_box, spacing)
    backend = _voxel_backend()
    if backend == "cuda":
        from porous_film.voxel import cupy_backend

        if not cupy_backend.cuda_backend_available():
            raise RuntimeError("CUDA voxel backend was requested but CuPy CUDA is unavailable")
        pore_mask = cupy_backend.voxelize_geometry_cupy(
            geometry,
            counts_xyz=counts_xyz,
            spacing_A=spacing,
            max_points_per_chunk=chunk_size,
        )
        return PhaseGrid(
            pore_mask=pore_mask,
            origin_A=np.zeros(3, dtype=float),
            spacing_A=spacing,
            target_box_A=target_box,
        )
    nx, ny, nz = (int(value) for value in counts_xyz)
    total_points = int(nx * ny * nz)
    flat_mask = np.empty(total_points, dtype=bool)

    for start in range(0, total_points, chunk_size):
        stop = min(start + chunk_size, total_points)
        flat_indices = np.arange(start, stop, dtype=np.int64)
        x_indices = flat_indices % nx
        y_indices = (flat_indices // nx) % ny
        z_indices = flat_indices // (nx * ny)
        points = np.column_stack(
            [
                (x_indices.astype(float) + 0.5) * spacing,
                (y_indices.astype(float) + 0.5) * spacing,
                (z_indices.astype(float) + 0.5) * spacing,
            ]
        )
        flat_mask[start:stop] = geometry.sdf(points) < 0.0

    return PhaseGrid(
        pore_mask=flat_mask.reshape((nz, ny, nx)),
        origin_A=np.zeros(3, dtype=float),
        spacing_A=spacing,
        target_box_A=target_box,
    )


def _voxel_backend() -> str:
    requested = os.environ.get("POROUS_FILM_VOXEL_BACKEND", "cpu").strip().lower()
    if requested not in {"cpu", "cuda", "auto"}:
        raise ValueError("POROUS_FILM_VOXEL_BACKEND must be one of: cpu, cuda, auto")
    if requested != "auto":
        return requested
    from porous_film.voxel import cupy_backend

    return "cuda" if cupy_backend.cuda_backend_available() else "cpu"


def solve_scale_for_porosity(
    build_at_linear_scale: Callable[[float], BuiltGeometry],
    target_phi: float,
    tolerance: float,
    lower: float = 0.1,
    upper: float = 10.0,
    *,
    voxel_spacing_A: float | None = None,
    voxel_divisions: int | None = None,
) -> tuple[float, BuiltGeometry, PhaseGrid]:
    target = _porosity_float(target_phi)
    tol = _positive_float(tolerance, "tolerance")
    low = _positive_float(lower, "lower")
    high = _positive_float(upper, "upper")
    if low >= high:
        raise ValueError("lower must be smaller than upper")
    if voxel_spacing_A is not None and voxel_divisions is not None:
        raise ValueError("provide only one of voxel_spacing_A or voxel_divisions")
    if voxel_divisions is not None:
        _positive_int(voxel_divisions, "voxel_divisions")

    low_built = _build_geometry(build_at_linear_scale, low)
    spacing = _solver_spacing_A(
        low_built.geometry.target_box_A,
        voxel_spacing_A=voxel_spacing_A,
        voxel_divisions=voxel_divisions,
    )
    low_result = _evaluate_built_geometry(low, low_built, spacing)
    low_phi = low_result[2].porosity

    nvox = _voxel_count(low_built.geometry.target_box_A, spacing)
    minimum_step = 1.0 / float(nvox)
    resolution_is_sufficient = tol >= minimum_step
    if resolution_is_sufficient and abs(low_phi - target) <= tol:
        return low_result

    best = low_result
    probe_scale = min(max(1.0, low), high)
    probe_result = low_result
    if resolution_is_sufficient and probe_scale > low and probe_scale < high:
        probe_result = _evaluate_scaled_geometry(build_at_linear_scale, probe_scale, spacing)
        probe_phi = probe_result[2].porosity
        if probe_phi < low_phi:
            raise ValueError("porosity must increase monotonically with linear scale")
        best = min((best, probe_result), key=lambda result: abs(result[2].porosity - target))
        if abs(probe_phi - target) <= tol:
            return probe_result
        if low_phi <= target <= probe_phi:
            high = probe_scale
            high_result = probe_result
            high_phi = probe_phi
        else:
            low = probe_scale
            low_result = probe_result
            low_phi = probe_phi

    bracketed = "high_result" in locals()
    if resolution_is_sufficient and not bracketed and low_phi < target and low < high:
        for _ in range(4):
            if low_phi > 0.0:
                predicted = low * (target / low_phi) ** (1.0 / 3.0)
            else:
                predicted = low * 2.0
            minimum_progress = low + 0.02 * (high - low)
            next_scale = min(high, max(predicted, minimum_progress))
            if next_scale >= high:
                break
            next_result = _evaluate_scaled_geometry(
                build_at_linear_scale,
                next_scale,
                spacing,
            )
            next_phi = next_result[2].porosity
            if next_phi < low_phi:
                raise ValueError("porosity must increase monotonically with linear scale")
            if abs(next_phi - target) < abs(best[2].porosity - target):
                best = next_result
            if abs(next_phi - target) <= tol:
                return next_result
            if next_phi >= target:
                high = next_scale
                high_result = next_result
                high_phi = next_phi
                bracketed = True
                break
            low = next_scale
            low_result = next_result
            low_phi = next_phi

    if not bracketed:
        high_result = _evaluate_scaled_geometry(build_at_linear_scale, high, spacing)
        high_phi = high_result[2].porosity
        if high_phi < low_phi:
            raise ValueError("porosity must increase monotonically with linear scale")
        if abs(high_phi - target) < abs(best[2].porosity - target):
            best = high_result
    if target < low_phi or target > high_phi:
        raise ValueError("target porosity is not bracketed by lower and upper scales")
    if not resolution_is_sufficient:
        raise PorosityResolutionError(
            "tolerance is finer than the minimum resolvable porosity step "
            f"1/nvox={minimum_step:g} for nvox={nvox}"
        )

    best = min(
        (best, low_result, high_result),
        key=lambda result: abs(result[2].porosity - target),
    )
    for _ in range(_SCALE_SOLVER_MAX_ITERATIONS):
        mid = 0.5 * (low + high)
        mid_result = _evaluate_scaled_geometry(build_at_linear_scale, mid, spacing)
        mid_phi = mid_result[2].porosity
        if abs(mid_phi - target) < abs(best[2].porosity - target):
            best = mid_result
        if abs(mid_phi - target) <= tol:
            return mid_result
        if mid_phi < target:
            low = mid
            low_phi = mid_phi
        else:
            high = mid
            high_phi = mid_phi
        if high - low <= np.finfo(float).eps * max(high, 1.0):
            break
        if high_phi - low_phi <= tol and abs(best[2].porosity - target) <= tol:
            break

    if abs(best[2].porosity - target) > tol:
        raise ValueError("failed to solve target porosity within tolerance")
    return best


def _evaluate_scaled_geometry(
    build_at_linear_scale: Callable[[float], BuiltGeometry],
    scale: float,
    spacing_A: float,
) -> tuple[float, BuiltGeometry, PhaseGrid]:
    built = _build_geometry(build_at_linear_scale, scale)
    return _evaluate_built_geometry(scale, built, spacing_A)


def _build_geometry(
    build_at_linear_scale: Callable[[float], BuiltGeometry],
    scale: float,
) -> BuiltGeometry:
    built = build_at_linear_scale(scale)
    if not isinstance(built, BuiltGeometry):
        raise TypeError("build_at_linear_scale must return BuiltGeometry")
    return built


def _evaluate_built_geometry(
    scale: float,
    built: BuiltGeometry,
    spacing_A: float,
) -> tuple[float, BuiltGeometry, PhaseGrid]:
    grid = voxelize_geometry(built.geometry, built.geometry.target_box_A, spacing_A)
    return scale, built, grid


def _solver_spacing_A(
    target_box_A: np.ndarray,
    *,
    voxel_spacing_A: float | None,
    voxel_divisions: int | None,
) -> float:
    target_box = _as_box(target_box_A)
    if voxel_spacing_A is not None:
        return _positive_float(voxel_spacing_A, "voxel_spacing_A")
    divisions = (
        _positive_int(voxel_divisions, "voxel_divisions")
        if voxel_divisions is not None
        else _SCALE_SOLVER_GRID_DIVISIONS
    )
    return float(np.min(target_box) / divisions)


def _grid_counts_xyz(box_A: np.ndarray, spacing_A: float) -> np.ndarray:
    raw_counts = box_A / spacing_A
    rounded = np.rint(raw_counts)
    if not np.all(np.isclose(raw_counts, rounded, rtol=0.0, atol=_GRID_DIVISIBILITY_TOLERANCE)):
        raise ValueError("each target-box length must be divisible by spacing_A")
    counts = rounded.astype(int)
    if np.any(counts <= 0):
        raise ValueError("box_A must contain at least one voxel along every axis")
    return counts


def _voxel_count(box_A: np.ndarray, spacing_A: float) -> int:
    counts = _grid_counts_xyz(_as_box(box_A), _positive_float(spacing_A, "spacing_A"))
    return int(np.prod(counts))


def _validate_binary_mask_values(mask: np.ndarray, name: str) -> None:
    if mask.dtype == np.dtype(bool):
        return
    if not np.issubdtype(mask.dtype, np.number):
        raise ValueError(f"{name} values must be exactly 0 or 1")
    if not np.all(np.isfinite(mask)):
        raise ValueError(f"{name} values must be exactly 0 or 1")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{name} values must be exactly 0 or 1")


def _as_vector(vector_A: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(vector_A, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def _as_box(box_A: np.ndarray) -> np.ndarray:
    box = _as_vector(box_A, "box_A")
    if np.any(box <= 0.0):
        raise ValueError("box_A must contain positive lengths")
    return box


def _positive_float(value: float, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return parsed


def _porosity_float(value: float) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise ValueError("target_phi must be between 0 and 1")
    return parsed


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _read_text_attr(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _required_attr(handle: h5py.File, name: str) -> object:
    if name not in handle.attrs:
        raise ValueError(f"malformed PhaseGrid HDF5: missing {name} attribute")
    return handle.attrs[name]
