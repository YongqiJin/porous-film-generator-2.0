from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class ThicknessResult:
    bin_edges_A: np.ndarray
    probabilities: np.ndarray
    field_A: np.ndarray
    uncertainty_A: float


@dataclass(frozen=True)
class ThicknessStabilityResult:
    passed: bool
    max_quantile_error_A: float
    mean_error_A: float
    histogram_l1: float
    tolerance_A: float
    warning: str | None


def local_thickness_field(
    mask_zyx: np.ndarray,
    spacing_A: float,
    periodic_xy: bool,
    max_voxels: int = 64_000_000,
) -> np.ndarray:
    """Return local-thickness diameters for the true phase in Angstrom."""

    mask = _as_bool_zyx(mask_zyx)
    spacing = _positive_float(spacing_A, "spacing_A")
    limit = _positive_int(max_voxels, "max_voxels")
    if mask.size > limit:
        raise ValueError(f"mask contains {mask.size} voxels, exceeding max_voxels={limit}")
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=float)

    if not periodic_xy:
        temporary_voxels = int(np.prod(np.asarray(mask.shape) + 2))
        if temporary_voxels > limit:
            raise ValueError(
                f"local-thickness temporaries exceed max_voxels={limit}: need {temporary_voxels}"
            )
        return _local_thickness_field_full(mask, spacing, periodic_xy=False, max_voxels=limit)

    full_temporary_voxels = int((mask.shape[0] + 2) * (3 * mask.shape[1]) * (3 * mask.shape[2]))
    if full_temporary_voxels <= limit:
        return _local_thickness_field_full(mask, spacing, periodic_xy=True, max_voxels=limit)
    return _local_thickness_field_periodic_slabs(mask, spacing, max_voxels=limit)


def local_thickness_distribution(
    mask_zyx: np.ndarray,
    spacing_A: float,
    periodic_xy: bool,
) -> ThicknessResult:
    mask = _as_bool_zyx(mask_zyx)
    spacing = _positive_float(spacing_A, "spacing_A")
    field = local_thickness_field(mask, spacing, periodic_xy)
    phase_values = field[mask]
    if phase_values.size == 0:
        return ThicknessResult(
            bin_edges_A=np.array([0.0, spacing], dtype=float),
            probabilities=np.array([0.0], dtype=float),
            field_A=field,
            uncertainty_A=spacing,
        )

    upper = max(float(np.max(phase_values)), spacing)
    bin_edges = np.arange(0.0, upper + 2.0 * spacing, spacing, dtype=float)
    counts, edges = np.histogram(phase_values, bins=bin_edges)
    probabilities = counts.astype(float) / float(phase_values.size)
    return ThicknessResult(
        bin_edges_A=edges.astype(float),
        probabilities=probabilities,
        field_A=field,
        uncertainty_A=spacing,
    )


def compare_local_thickness_coarse_fine(
    coarse_mask_zyx: np.ndarray,
    fine_mask_zyx: np.ndarray,
    *,
    coarse_spacing_A: float,
    fine_spacing_A: float,
    periodic_xy: bool,
) -> ThicknessStabilityResult:
    coarse_mask = _as_bool_zyx(coarse_mask_zyx)
    fine_mask = _as_bool_zyx(fine_mask_zyx)
    coarse_spacing = _positive_float(coarse_spacing_A, "coarse_spacing_A")
    fine_spacing = _positive_float(fine_spacing_A, "fine_spacing_A")
    tolerance = 2.0 * fine_spacing

    coarse = local_thickness_distribution(coarse_mask, coarse_spacing, periodic_xy)
    fine = local_thickness_distribution(fine_mask, fine_spacing, periodic_xy)
    coarse_values = coarse.field_A[coarse_mask]
    fine_values = fine.field_A[fine_mask]
    if coarse_values.size == 0 and fine_values.size == 0:
        return ThicknessStabilityResult(True, 0.0, 0.0, 0.0, tolerance, None)
    if coarse_values.size == 0 or fine_values.size == 0:
        warning = "coarse/fine local-thickness comparison has an empty phase on one grid"
        return ThicknessStabilityResult(False, np.inf, np.inf, np.inf, tolerance, warning)

    quantiles = np.array([0.0, 0.1, 0.5, 0.9, 1.0], dtype=float)
    coarse_quantiles = np.quantile(coarse_values, quantiles)
    fine_quantiles = np.quantile(fine_values, quantiles)
    max_quantile_error = float(np.max(np.abs(coarse_quantiles - fine_quantiles)))
    mean_error = float(abs(np.mean(coarse_values) - np.mean(fine_values)))
    histogram_l1 = _histogram_l1(coarse_values, fine_values, fine_spacing)
    passed = bool(max(max_quantile_error, mean_error) <= tolerance)
    warning = None if passed else "coarse/fine local-thickness agreement exceeds two fine voxels"
    return ThicknessStabilityResult(
        passed=passed,
        max_quantile_error_A=max_quantile_error,
        mean_error_A=mean_error,
        histogram_l1=histogram_l1,
        tolerance_A=tolerance,
        warning=warning,
    )


def _local_thickness_field_full(
    mask: np.ndarray,
    spacing: float,
    *,
    periodic_xy: bool,
    max_voxels: int,
) -> np.ndarray:
    if periodic_xy:
        analysis_mask = np.pad(np.tile(mask, (1, 3, 3)), ((1, 1), (0, 0), (0, 0)))
        central = (
            slice(1, 1 + mask.shape[0]),
            slice(mask.shape[1], 2 * mask.shape[1]),
            slice(mask.shape[2], 2 * mask.shape[2]),
        )
    else:
        analysis_mask = np.pad(mask, 1)
        central = (
            slice(1, 1 + mask.shape[0]),
            slice(1, 1 + mask.shape[1]),
            slice(1, 1 + mask.shape[2]),
        )

    edt_voxels = ndimage.distance_transform_edt(analysis_mask)
    field = np.zeros(mask.shape, dtype=float)
    max_radius = int(np.floor(np.max(edt_voxels[central])))
    for radius in range(max_radius, 0, -1):
        footprint = _sphere_footprint(radius, max_voxels=max_voxels)
        eligible_centers = ndimage.binary_erosion(
            analysis_mask,
            structure=footprint,
            border_value=0,
        )
        if periodic_xy:
            central_centers = eligible_centers[central]
            tiled_centers = np.zeros_like(analysis_mask, dtype=bool)
            tiled_centers[1 : 1 + mask.shape[0]] = np.tile(central_centers, (1, 3, 3))
            coverage = ndimage.binary_dilation(
                tiled_centers,
                structure=footprint,
                border_value=0,
            )[central]
        else:
            coverage = ndimage.binary_dilation(
                eligible_centers,
                structure=footprint,
                border_value=0,
            )[central]
        diameter = 2.0 * radius * spacing
        better = coverage & mask & (field < diameter)
        field[better] = diameter
    return field


def _local_thickness_field_periodic_slabs(
    mask: np.ndarray,
    spacing: float,
    *,
    max_voxels: int,
) -> np.ndarray:
    nz, ny, nx = mask.shape
    field = np.zeros(mask.shape, dtype=float)
    max_radius = _z_open_radius_upper_bound(mask)
    tiled_yx = 9 * ny * nx

    for radius in range(max_radius, 0, -1):
        footprint = _sphere_footprint(radius, max_voxels=max_voxels)
        slab_depth = _periodic_slab_depth_for_radius(
            radius,
            tiled_yx=tiled_yx,
            max_voxels=max_voxels,
        )
        diameter = 2.0 * radius * spacing
        for z0 in range(0, nz, slab_depth):
            z1 = min(nz, z0 + slab_depth)
            source0, source1, coverage = _periodic_radius_coverage_for_core_center_slab(
                mask,
                radius=radius,
                footprint=footprint,
                z0=z0,
                z1=z1,
                max_voxels=max_voxels,
            )
            target = field[source0:source1]
            better = coverage & (target < diameter)
            target[better] = diameter
    return field


def _periodic_radius_coverage_for_core_center_slab(
    mask: np.ndarray,
    *,
    radius: int,
    footprint: np.ndarray,
    z0: int,
    z1: int,
    max_voxels: int,
) -> tuple[int, int, np.ndarray]:
    _nz, ny, nx = mask.shape
    source0 = max(0, z0 - radius)
    source1 = min(mask.shape[0], z1 + radius)
    pad_before = max(0, radius - z0)
    pad_after = max(0, z1 + radius - mask.shape[0])
    analysis_mask = np.pad(
        np.tile(mask[source0:source1], (1, 3, 3)),
        ((pad_before, pad_after), (0, 0), (0, 0)),
        constant_values=False,
    )
    _raise_if_temporary_exceeds_limit(
        analysis_mask.size,
        radius=radius,
        max_voxels=max_voxels,
    )

    eroded = ndimage.binary_erosion(analysis_mask, structure=footprint, border_value=0)
    core_start = pad_before + (z0 - source0)
    core_stop = core_start + (z1 - z0)
    valid_core_centers = eroded[core_start:core_stop, ny : 2 * ny, nx : 2 * nx]
    centers = np.zeros_like(analysis_mask, dtype=bool)
    centers[core_start:core_stop] = np.tile(valid_core_centers, (1, 3, 3))
    tiled_centers = centers
    _raise_if_temporary_exceeds_limit(
        tiled_centers.size,
        radius=radius,
        max_voxels=max_voxels,
    )

    dilated = ndimage.binary_dilation(tiled_centers, structure=footprint, border_value=0)
    source_start = pad_before
    source_stop = source_start + (source1 - source0)
    coverage = dilated[source_start:source_stop, ny : 2 * ny, nx : 2 * nx]
    return source0, source1, coverage & mask[source0:source1]


def _write_sphere(
    field: np.ndarray,
    mask: np.ndarray,
    center_zyx: np.ndarray,
    offsets: np.ndarray,
    diameter: float,
    *,
    periodic_xy: bool,
) -> None:
    positions = center_zyx[np.newaxis, :] + offsets
    z = positions[:, 0]
    valid = (z >= 0) & (z < mask.shape[0])
    if periodic_xy:
        z_valid = z[valid]
        y_valid = np.mod(positions[valid, 1], mask.shape[1])
        x_valid = np.mod(positions[valid, 2], mask.shape[2])
    else:
        valid &= (
            (positions[:, 1] >= 0)
            & (positions[:, 1] < mask.shape[1])
            & (positions[:, 2] >= 0)
            & (positions[:, 2] < mask.shape[2])
        )
        z_valid = z[valid]
        y_valid = positions[valid, 1]
        x_valid = positions[valid, 2]

    inside = mask[z_valid, y_valid, x_valid]
    if not np.any(inside):
        return
    z_inside = z_valid[inside]
    y_inside = y_valid[inside]
    x_inside = x_valid[inside]
    better = field[z_inside, y_inside, x_inside] < diameter
    field[z_inside[better], y_inside[better], x_inside[better]] = diameter


def _sphere_offsets(radius_voxels: int, *, max_voxels: int) -> np.ndarray:
    radius = _positive_int(radius_voxels, "radius_voxels")
    _raise_if_footprint_exceeds_limit(radius, max_voxels)
    grid = np.indices((2 * radius + 1, 2 * radius + 1, 2 * radius + 1))
    offsets = np.moveaxis(grid, 0, -1).reshape(-1, 3) - radius
    distances = np.linalg.norm(offsets, axis=1)
    return offsets[distances < radius].astype(np.int64)


def _sphere_footprint(radius_voxels: int, *, max_voxels: int) -> np.ndarray:
    radius = _positive_int(radius_voxels, "radius_voxels")
    _raise_if_footprint_exceeds_limit(radius, max_voxels)
    grid = np.indices((2 * radius + 1, 2 * radius + 1, 2 * radius + 1))
    offsets = np.moveaxis(grid, 0, -1) - radius
    distances = np.linalg.norm(offsets, axis=-1)
    return distances < radius


def _raise_if_footprint_exceeds_limit(radius: int, max_voxels: int) -> None:
    footprint_voxels = (2 * radius + 1) ** 3
    if footprint_voxels > max_voxels:
        raise ValueError(
            "periodic local-thickness memory limit exceeded for "
            f"radius {radius}: spherical footprint needs {footprint_voxels} "
            f"voxels, max_voxels={max_voxels}"
        )


def _periodic_slab_depth_for_radius(
    radius: int,
    *,
    tiled_yx: int,
    max_voxels: int,
) -> int:
    available_z = max_voxels // tiled_yx
    slab_depth = available_z - 2 * radius
    if slab_depth < 1:
        needed = (2 * radius + 1) * tiled_yx
        raise ValueError(
            "periodic local-thickness memory limit exceeded for "
            f"radius {radius}: one core z slab with halo needs {needed} "
            f"voxels, max_voxels={max_voxels}"
        )
    return int(slab_depth)


def _raise_if_temporary_exceeds_limit(
    temporary_voxels: int,
    *,
    radius: int,
    max_voxels: int,
) -> None:
    if temporary_voxels > max_voxels:
        raise ValueError(
            "periodic local-thickness memory limit exceeded for "
            f"radius {radius}: temporary needs {temporary_voxels} "
            f"voxels, max_voxels={max_voxels}"
        )


def _z_open_radius_upper_bound(mask: np.ndarray) -> int:
    max_run = 0
    nz, ny, nx = mask.shape
    for y in range(ny):
        for x in range(nx):
            run = 0
            for z in range(nz):
                if mask[z, y, x]:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 0
    return (max_run + 1) // 2


def _histogram_l1(first: np.ndarray, second: np.ndarray, bin_width: float) -> float:
    upper = max(float(np.max(first)), float(np.max(second)), bin_width)
    edges = np.arange(0.0, upper + 2.0 * bin_width, bin_width, dtype=float)
    first_counts, _ = np.histogram(first, bins=edges)
    second_counts, _ = np.histogram(second, bins=edges)
    first_prob = first_counts.astype(float) / float(first.size)
    second_prob = second_counts.astype(float) / float(second.size)
    return float(np.sum(np.abs(first_prob - second_prob)))


def _as_bool_zyx(mask_zyx: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask_zyx, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("mask_zyx must have shape (z, y, x)")
    return mask


def _positive_float(value: float, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return parsed


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed
