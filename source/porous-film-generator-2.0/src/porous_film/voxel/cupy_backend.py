from __future__ import annotations

import importlib
import os
from typing import Any

import numpy as np

from porous_film.geometry import ChannelUnit, CompactUnit, PoreGeometry
from porous_film.geometry.sdf import (
    _compact_minimum_image_safe,
    _roughness_parameters,
    _smooth_min_skip_margin_A,
    _unit_axis_extent,
)

_SMOOTH_UNION_SHARPNESS = 32.0
_DEFAULT_GPU_MAX_POINTS_PER_CHUNK = 262_144
_DEFAULT_SEGMENT_BATCH_SIZE = 32


def _cupy_module() -> Any:
    try:
        return importlib.import_module("cupy")
    except ImportError as exc:
        raise RuntimeError(
            "the CUDA voxel backend requires CuPy; install the project GPU extra"
        ) from exc


def cuda_backend_available() -> bool:
    try:
        cupy = _cupy_module()
    except RuntimeError:
        return False
    try:
        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except cupy.cuda.runtime.CUDARuntimeError:
        return False


def cuda_device() -> int:
    raw = os.environ.get("POROUS_FILM_CUDA_DEVICE", "0").strip()
    if raw.startswith("cuda:"):
        raw = raw.split(":", 1)[1]
    try:
        index = int(raw)
    except ValueError as exc:
        raise ValueError("POROUS_FILM_CUDA_DEVICE must be a nonnegative integer") from exc
    if index < 0:
        raise ValueError("POROUS_FILM_CUDA_DEVICE must be a nonnegative integer")
    return index


def recommended_gpu_workload(
    *,
    device_name: str,
    free_memory_bytes: int,
) -> tuple[int, int]:
    """Choose a conservative voxel chunk and channel-segment batch from free VRAM.

    The thresholds cover the tested A100/A800 and RTX 4090 configurations while
    retaining the previous conservative defaults on lower-memory devices.
    ``device_name`` is accepted so this public policy can evolve for device-specific
    tuning without changing its API.
    """
    free_memory = max(0, int(free_memory_bytes))
    gib = 1024**3
    if free_memory >= 48 * gib:
        return 4_000_000, 64
    if free_memory >= 28 * gib:
        return 2_000_000, 64
    if free_memory >= 18 * gib:
        return 1_000_000, 64
    return _DEFAULT_GPU_MAX_POINTS_PER_CHUNK, _DEFAULT_SEGMENT_BATCH_SIZE


def _resolve_gpu_workload(
    *,
    automatic: tuple[int, int],
    max_points_per_chunk: int | None,
) -> tuple[int, int]:
    automatic_chunk, automatic_segment_batch = automatic
    configured_chunk = _positive_environment_int(
        "POROUS_FILM_GPU_MAX_POINTS_PER_CHUNK",
        automatic_chunk,
    )
    if max_points_per_chunk is not None:
        explicit_chunk = _positive_int(max_points_per_chunk, "max_points_per_chunk")
        configured_chunk = min(configured_chunk, explicit_chunk)
    segment_batch_size = _positive_environment_int(
        "POROUS_FILM_GPU_SEGMENT_BATCH_SIZE",
        automatic_segment_batch,
    )
    return configured_chunk, segment_batch_size


def evaluate_sdf_cupy(
    geometry: PoreGeometry,
    points_A: np.ndarray,
    *,
    device: int | None = None,
) -> np.ndarray:
    cupy = _cupy_module()
    points = np.asarray(points_A, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_A must have shape (n, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_A must be finite")
    selected_device = cuda_device() if device is None else int(device)
    with cupy.cuda.Device(selected_device):
        point_array = cupy.asarray(points, dtype=cupy.float64)
        result = _geometry_sdf_array(cupy, geometry, point_array)
        return cupy.asnumpy(result)


def voxelize_geometry_cupy(
    geometry: PoreGeometry,
    *,
    counts_xyz: np.ndarray,
    spacing_A: float,
    max_points_per_chunk: int | None,
    device: int | None = None,
) -> np.ndarray:
    cupy = _cupy_module()
    selected_device = cuda_device() if device is None else int(device)
    if not cuda_backend_available():
        raise RuntimeError("CUDA voxel backend was requested but CuPy CUDA is unavailable")
    nx, ny, nz = (int(value) for value in np.asarray(counts_xyz, dtype=np.int64))
    total_points = int(nx * ny * nz)
    flat_mask = np.empty(total_points, dtype=bool)
    with cupy.cuda.Device(selected_device):
        chunk_size, segment_batch_size = _selected_gpu_workload(
            cupy,
            selected_device,
            max_points_per_chunk=max_points_per_chunk,
        )
        for start in range(0, total_points, chunk_size):
            stop = min(start + chunk_size, total_points)
            flat_indices = cupy.arange(start, stop, dtype=cupy.int64)
            points = _voxel_center_points(
                cupy,
                flat_indices,
                nx=nx,
                ny=ny,
                spacing_A=spacing_A,
            )
            occupied = _geometry_occupancy_array(
                cupy,
                geometry,
                points,
                segment_batch_size=segment_batch_size,
            )
            flat_mask[start:stop] = cupy.asnumpy(occupied)
    return flat_mask.reshape((nz, ny, nx))


def voxelize_unit_masks_cupy(
    geometry: PoreGeometry,
    *,
    counts_xyz: np.ndarray,
    spacing_A: float,
    candidate_mask: np.ndarray,
    max_points_per_chunk: int | None = None,
    device: int | None = None,
) -> tuple[np.ndarray, ...]:
    """Voxelize each unit only where the already-final pore phase is occupied."""
    cupy = _cupy_module()
    selected_device = cuda_device() if device is None else int(device)
    if not cuda_backend_available():
        raise RuntimeError("CUDA voxel backend was requested but CuPy CUDA is unavailable")
    nx, ny, nz = (int(value) for value in np.asarray(counts_xyz, dtype=np.int64))
    expected_shape = (nz, ny, nx)
    candidates = np.asarray(candidate_mask, dtype=bool)
    if candidates.shape != expected_shape:
        raise ValueError(f"candidate_mask must have shape {expected_shape}, got {candidates.shape}")
    if not geometry.units:
        return ()
    total_points = int(nx * ny * nz)
    candidate_indices = np.flatnonzero(candidates.ravel())
    masks = np.zeros((len(geometry.units), total_points), dtype=bool)
    box = np.asarray(geometry.target_box_A, dtype=float)
    margin_A = _smooth_min_skip_margin_A(9)
    with cupy.cuda.Device(selected_device):
        chunk_size, segment_batch_size = _selected_gpu_workload(
            cupy,
            selected_device,
            max_points_per_chunk=max_points_per_chunk,
        )
        for start in range(0, candidate_indices.size, chunk_size):
            host_indices = candidate_indices[start : start + chunk_size]
            flat_indices = cupy.asarray(host_indices, dtype=cupy.int64)
            points = _voxel_center_points(
                cupy,
                flat_indices,
                nx=nx,
                ny=ny,
                spacing_A=spacing_A,
            )
            for unit_index, unit in enumerate(geometry.units):
                active = _unit_support_mask(cupy, unit, points, box, margin_A=margin_A)
                if not bool(cupy.any(active).item()):
                    continue
                occupied = cupy.zeros(points.shape[0], dtype=cupy.bool_)
                occupied[active] = (
                    _single_unit_sdf_array(
                        cupy,
                        unit,
                        points[active],
                        box,
                        segment_batch_size=segment_batch_size,
                    )
                    < 0.0
                )
                masks[unit_index, host_indices] = cupy.asnumpy(occupied)
    return tuple(masks[index].reshape(expected_shape) for index in range(len(geometry.units)))


def _selected_gpu_workload(
    cupy: Any,
    selected_device: int,
    *,
    max_points_per_chunk: int | None,
) -> tuple[int, int]:
    properties = cupy.cuda.runtime.getDeviceProperties(selected_device)
    raw_device_name = properties.get("name", "unknown CUDA device")
    if isinstance(raw_device_name, bytes):
        device_name = raw_device_name.decode("utf-8", errors="replace")
    else:
        device_name = str(raw_device_name)
    free_memory_bytes, _ = cupy.cuda.runtime.memGetInfo()
    return _resolve_gpu_workload(
        automatic=recommended_gpu_workload(
            device_name=device_name,
            free_memory_bytes=int(free_memory_bytes),
        ),
        max_points_per_chunk=max_points_per_chunk,
    )


def _voxel_center_points(
    cupy: Any,
    flat_indices: Any,
    *,
    nx: int,
    ny: int,
    spacing_A: float,
) -> Any:
    x_indices = flat_indices % nx
    y_indices = (flat_indices // nx) % ny
    z_indices = flat_indices // (nx * ny)
    return cupy.stack(
        (
            (x_indices.astype(cupy.float64) + 0.5) * float(spacing_A),
            (y_indices.astype(cupy.float64) + 0.5) * float(spacing_A),
            (z_indices.astype(cupy.float64) + 0.5) * float(spacing_A),
        ),
        axis=1,
    )


def _geometry_occupancy_array(
    cupy: Any,
    geometry: PoreGeometry,
    points: Any,
    *,
    segment_batch_size: int,
) -> Any:
    """Evaluate only points that can lie in a unit's zero-level support.

    Voxelization needs the sign of the final smooth-union field, not its positive
    distance far from every pore.  Restricting each unit evaluation to its
    conservative periodic support avoids evaluating every channel at every grid
    point while retaining the exact field calculation near the phase boundary.
    """
    if not geometry.units:
        return cupy.zeros(points.shape[0], dtype=cupy.bool_)
    result = cupy.full(points.shape[0], cupy.inf, dtype=cupy.float64)
    box = np.asarray(geometry.target_box_A, dtype=float)
    # A single periodic unit can contribute up to nine xy images.  The margin
    # bounds the aggregate log-sum-exp tail omitted outside all unit supports.
    margin_A = _smooth_min_skip_margin_A(max(1, 9 * len(geometry.units)))
    for unit in geometry.units:
        active = _unit_support_mask(cupy, unit, points, box, margin_A=margin_A)
        if not bool(cupy.any(active).item()):
            continue
        field = _single_unit_sdf_array(
            cupy,
            unit,
            points[active],
            box,
            segment_batch_size=segment_batch_size,
        )
        result[active] = _smooth_min_pair(cupy, result[active], field)
    return result < 0.0


def _single_unit_sdf_array(
    cupy: Any,
    unit: CompactUnit | ChannelUnit,
    points: Any,
    box: np.ndarray,
    *,
    segment_batch_size: int,
) -> Any:
    if isinstance(unit, CompactUnit) and _compact_minimum_image_safe(unit, box):
        return _compact_periodic_sdf(cupy, unit, points, box)
    query_min = cupy.asnumpy(cupy.amin(points, axis=0))
    query_max = cupy.asnumpy(cupy.amax(points, axis=0))
    result = cupy.full(points.shape[0], cupy.inf, dtype=cupy.float64)
    x_shifts = _periodic_shifts(
        unit,
        axis=0,
        query_min=float(query_min[0]),
        query_max=float(query_max[0]),
        box=box,
    )
    y_shifts = _periodic_shifts(
        unit,
        axis=1,
        query_min=float(query_min[1]),
        query_max=float(query_max[1]),
        box=box,
    )
    for x_shift in x_shifts:
        for y_shift in y_shifts:
            offset = cupy.asarray([x_shift, y_shift, 0.0], dtype=cupy.float64)
            result = _smooth_min_pair(
                cupy,
                result,
                _unit_sdf(
                    cupy,
                    unit,
                    points - offset,
                    segment_batch_size=segment_batch_size,
                ),
            )
    return result


def _unit_support_mask(
    cupy: Any,
    unit: CompactUnit | ChannelUnit,
    points: Any,
    box: np.ndarray,
    *,
    margin_A: float,
) -> Any:
    if isinstance(unit, CompactUnit):
        support = float(np.max(unit.radii_A)) * (1.0 + float(unit.roughness))
        lower = np.asarray(unit.center_A, dtype=float) - support
        upper = np.asarray(unit.center_A, dtype=float) + support
    else:
        lower = np.min(np.asarray(unit.segment_aabb_min_A, dtype=float), axis=0)
        upper = np.max(np.asarray(unit.segment_aabb_max_A, dtype=float), axis=0)

    active = cupy.ones(points.shape[0], dtype=cupy.bool_)
    margin = float(margin_A)
    for axis in (0, 1):
        span = float(upper[axis] - lower[axis])
        if span >= float(box[axis]):
            continue
        center = 0.5 * float(lower[axis] + upper[axis])
        half_width = 0.5 * span
        delta = cupy.mod(points[:, axis] - center + 0.5 * box[axis], box[axis]) - 0.5 * box[axis]
        active &= cupy.abs(delta) <= half_width + margin
    active &= points[:, 2] >= float(lower[2]) - margin
    active &= points[:, 2] <= float(upper[2]) + margin
    return active


def _geometry_sdf_array(cupy: Any, geometry: PoreGeometry, points: Any) -> Any:
    if not geometry.units:
        return cupy.full(points.shape[0], cupy.inf, dtype=cupy.float64)
    query_min = cupy.asnumpy(cupy.amin(points, axis=0))
    query_max = cupy.asnumpy(cupy.amax(points, axis=0))
    box = np.asarray(geometry.target_box_A, dtype=float)
    result = cupy.full(points.shape[0], cupy.inf, dtype=cupy.float64)
    for unit in geometry.units:
        if isinstance(unit, CompactUnit) and _compact_minimum_image_safe(unit, box):
            unit_field = _compact_periodic_sdf(cupy, unit, points, box)
        else:
            unit_field = cupy.full_like(result, cupy.inf)
            x_shifts = _periodic_shifts(
                unit,
                axis=0,
                query_min=float(query_min[0]),
                query_max=float(query_max[0]),
                box=box,
            )
            y_shifts = _periodic_shifts(
                unit,
                axis=1,
                query_min=float(query_min[1]),
                query_max=float(query_max[1]),
                box=box,
            )
            for x_shift in x_shifts:
                for y_shift in y_shifts:
                    offset = cupy.asarray(
                        [x_shift, y_shift, 0.0],
                        dtype=cupy.float64,
                    )
                    field = _unit_sdf(
                        cupy,
                        unit,
                        points - offset,
                        segment_batch_size=_DEFAULT_SEGMENT_BATCH_SIZE,
                    )
                    unit_field = _smooth_min_pair(cupy, unit_field, field)
        result = _smooth_min_pair(cupy, result, unit_field)
    return result


def _periodic_shifts(
    unit: CompactUnit | ChannelUnit,
    *,
    axis: int,
    query_min: float,
    query_max: float,
    box: np.ndarray,
) -> np.ndarray:
    unit_min, unit_max = _unit_axis_extent(unit, axis)
    box_length = float(box[axis])
    first = int(np.floor((query_min - unit_max) / box_length))
    last = int(np.ceil((query_max - unit_min) / box_length))
    return np.arange(first, last + 1, dtype=float) * box_length


def _unit_sdf(
    cupy: Any,
    unit: CompactUnit | ChannelUnit,
    points: Any,
    *,
    segment_batch_size: int,
) -> Any:
    if isinstance(unit, CompactUnit):
        return _compact_sdf(cupy, unit, points)
    if isinstance(unit, ChannelUnit):
        return _channel_sdf(cupy, unit, points, segment_batch_size=segment_batch_size)
    raise TypeError(f"unsupported pore unit type: {type(unit)!r}")


def _compact_periodic_sdf(
    cupy: Any,
    unit: CompactUnit,
    points: Any,
    box_A: np.ndarray,
) -> Any:
    center = _array(cupy, unit.center_A)
    box = _array(cupy, box_A)
    delta = points - center
    delta = delta.copy()
    delta[:, 0] -= box[0] * cupy.rint(delta[:, 0] / box[0])
    delta[:, 1] -= box[1] * cupy.rint(delta[:, 1] / box[1])
    wrapped = center + delta
    result = _compact_sdf(cupy, unit, wrapped)
    support = float(np.max(unit.radii_A)) * (1.0 + float(unit.roughness))
    skip_margin = _smooth_min_skip_margin_A(9)
    for x_image in (-1.0, 0.0, 1.0):
        for y_image in (-1.0, 0.0, 1.0):
            if x_image == 0.0 and y_image == 0.0:
                continue
            offset = cupy.asarray(
                [x_image * box_A[0], y_image * box_A[1], 0.0],
                dtype=cupy.float64,
            )
            alternative_points = wrapped + offset
            lower_bound = cupy.linalg.norm(alternative_points - center, axis=1) - support
            active = lower_bound < result + skip_margin
            if not bool(cupy.any(active).item()):
                continue
            result[active] = _smooth_min_pair(
                cupy,
                result[active],
                _compact_sdf(cupy, unit, alternative_points[active]),
            )
    return result


def _compact_sdf(cupy: Any, unit: CompactUnit, points: Any) -> Any:
    center = _array(cupy, unit.center_A)
    rotation = _array(cupy, unit.orientation.as_matrix())
    local = (points - center) @ rotation
    radii = _array(cupy, unit.radii_A)
    if unit.is_multilobe:
        lobe_centers = _array(cupy, unit.lobe_centers_local_A)
        lobe_radii = _array(cupy, unit.lobe_radii_A)
        offsets = local[cupy.newaxis, :, :] - lobe_centers[:, cupy.newaxis, :]
        implicit = cupy.sqrt(cupy.sum((offsets / lobe_radii[:, cupy.newaxis, :]) ** 2, axis=2))
        fields = (implicit - 1.0) * cupy.amin(lobe_radii, axis=1)[:, cupy.newaxis]
        smooth_length = float(unit.smooth_length_A)
        if smooth_length <= 0.0 or fields.shape[0] == 1:
            base = cupy.amin(fields, axis=0)
        else:
            base = -smooth_length * _logsumexp(cupy, -fields / smooth_length, axis=0)
    elif float(unit.exponent) == 2.0 and np.allclose(unit.radii_A, unit.radii_A[0]):
        base = cupy.linalg.norm(local, axis=1) - float(unit.radii_A[0])
    else:
        radial_xy = cupy.sqrt((local[:, 0] / radii[0]) ** 2 + (local[:, 1] / radii[1]) ** 2)
        axial_z = cupy.abs(local[:, 2] / radii[2])
        exponent = float(unit.exponent)
        implicit = (radial_xy**exponent + axial_z**exponent) ** (1.0 / exponent)
        base = (implicit - 1.0) * float(np.min(unit.radii_A))
    normalized = local / cupy.maximum(radii, 1.0e-12)
    return base - _roughness_perturbation(
        cupy,
        normalized,
        unit_id=unit.unit_id,
        roughness=float(unit.roughness),
        length_scale_A=float(np.min(unit.radii_A)),
    )


def _channel_sdf(
    cupy: Any,
    unit: ChannelUnit,
    points: Any,
    *,
    segment_batch_size: int,
) -> Any:
    result = cupy.full(points.shape[0], cupy.inf, dtype=cupy.float64)
    segment_count = int(unit.segment_lengths_A.size)
    for first in range(0, segment_count, segment_batch_size):
        last = min(first + segment_batch_size, segment_count)
        starts = _array(cupy, unit.segment_starts_A[first:last])
        ends = _array(cupy, unit.segment_ends_A[first:last])
        tangents = _array(cupy, unit.segment_tangents_A[first:last])
        start_normals = _array(cupy, unit.segment_start_plane_normals_A[first:last])
        end_normals = _array(cupy, unit.segment_end_plane_normals_A[first:last])
        lengths = _array(cupy, unit.segment_lengths_A[first:last])
        cumulative = _array(cupy, unit.segment_cumulative_starts_A[first:last])
        matrices = _array(
            cupy,
            np.stack([frame.as_matrix() for frame in unit.local_segment_frames[first:last]]),
        )
        offsets = points[cupy.newaxis, :, :] - starts[:, cupy.newaxis, :]
        axial_raw = cupy.sum(offsets * tangents[:, cupy.newaxis, :], axis=2)
        axial = cupy.maximum(axial_raw, 0.0)
        axial = cupy.minimum(axial, lengths[:, cupy.newaxis])
        normalized_s = (cumulative[:, cupy.newaxis] + axial) / max(
            float(unit.arc_length_A), 1.0e-12
        )
        local_radius = _channel_radius(cupy, unit, normalized_s)
        closest = (
            starts[:, cupy.newaxis, :] + axial[:, :, cupy.newaxis] * tangents[:, cupy.newaxis, :]
        )
        radial = cupy.linalg.norm(points[cupy.newaxis, :, :] - closest, axis=2) - local_radius
        start_signed = -cupy.sum(offsets * start_normals[:, cupy.newaxis, :], axis=2)
        end_signed = cupy.sum(
            (points[cupy.newaxis, :, :] - ends[:, cupy.newaxis, :])
            * end_normals[:, cupy.newaxis, :],
            axis=2,
        )
        inside = (start_signed <= 0.0) & (end_signed <= 0.0)
        base = cupy.where(
            inside,
            radial,
            cupy.maximum(radial, cupy.maximum(start_signed, end_signed)),
        )
        local = cupy.einsum("bnj,bjk->bnk", offsets, matrices)
        roughness_coordinates = cupy.stack(
            (
                normalized_s,
                local[:, :, 1] / float(unit.cross_radius_A),
                local[:, :, 2] / float(unit.cross_radius_A),
            ),
            axis=2,
        )
        fields = base - _roughness_perturbation(
            cupy,
            roughness_coordinates,
            unit_id=unit.unit_id,
            roughness=float(unit.roughness),
            length_scale_A=float(unit.cross_radius_A),
        )
        batch_field = (
            -_logsumexp(
                cupy,
                -_SMOOTH_UNION_SHARPNESS * fields,
                axis=0,
            )
            / _SMOOTH_UNION_SHARPNESS
        )
        result = _smooth_min_pair(cupy, result, batch_field)
    first_cap, last_cap = _endpoint_cap_fields(cupy, unit, points)
    result = _smooth_min_pair(cupy, result, first_cap)
    return _smooth_min_pair(cupy, result, last_cap)


def _channel_radius(cupy: Any, unit: ChannelUnit, normalized_s: Any) -> Any:
    if (
        unit.radius_profile_s is None
        or unit.radius_profile_A is None
        or unit.radius_profile_coefficients is None
    ):
        return cupy.full_like(normalized_s, float(unit.cross_radius_A))
    nodes = _array(cupy, unit.radius_profile_s)
    coefficients = _array(cupy, unit.radius_profile_coefficients)
    indices = cupy.searchsorted(nodes, normalized_s, side="right") - 1
    indices = cupy.clip(indices, 0, nodes.size - 2)
    delta = normalized_s - nodes[indices]
    return (
        (coefficients[0][indices] * delta + coefficients[1][indices]) * delta
        + coefficients[2][indices]
    ) * delta + coefficients[3][indices]


def _endpoint_cap_fields(cupy: Any, unit: ChannelUnit, points: Any) -> tuple[Any, Any]:
    first_local = (points - _array(cupy, unit.segment_starts_A[0])) @ _array(
        cupy,
        unit.local_segment_frames[0].as_matrix(),
    )
    first = cupy.linalg.norm(first_local, axis=1) - float(unit.segment_start_radii_A[0])
    first = cupy.where(first_local[:, 0] <= 0.0, first, cupy.inf)
    first_coordinates = cupy.stack(
        (
            cupy.zeros(points.shape[0], dtype=cupy.float64),
            first_local[:, 1] / float(unit.cross_radius_A),
            first_local[:, 2] / float(unit.cross_radius_A),
        ),
        axis=1,
    )
    first -= _roughness_perturbation(
        cupy,
        first_coordinates,
        unit_id=unit.unit_id,
        roughness=float(unit.roughness),
        length_scale_A=float(unit.cross_radius_A),
    )

    last_local = (points - _array(cupy, unit.segment_ends_A[-1])) @ _array(
        cupy,
        unit.local_segment_frames[-1].as_matrix(),
    )
    last = cupy.linalg.norm(last_local, axis=1) - float(unit.segment_end_radii_A[-1])
    last = cupy.where(last_local[:, 0] >= 0.0, last, cupy.inf)
    last_coordinates = cupy.stack(
        (
            cupy.ones(points.shape[0], dtype=cupy.float64),
            last_local[:, 1] / float(unit.cross_radius_A),
            last_local[:, 2] / float(unit.cross_radius_A),
        ),
        axis=1,
    )
    last -= _roughness_perturbation(
        cupy,
        last_coordinates,
        unit_id=unit.unit_id,
        roughness=float(unit.roughness),
        length_scale_A=float(unit.cross_radius_A),
    )
    return first, last


def _roughness_perturbation(
    cupy: Any,
    local_coordinates: Any,
    *,
    unit_id: str,
    roughness: float,
    length_scale_A: float,
) -> Any:
    if roughness == 0.0:
        return cupy.zeros(local_coordinates.shape[:-1], dtype=cupy.float64)
    parameters = _roughness_parameters(unit_id, roughness)
    coordinate = cupy.sum(local_coordinates, axis=-1)
    perturbation = cupy.zeros_like(coordinate)
    for frequency, phase, amplitude in zip(
        parameters["frequencies"],
        parameters["phases_rad"],
        parameters["amplitudes"],
        strict=True,
    ):
        perturbation += float(amplitude) * cupy.sin(
            2.0 * np.pi * float(frequency) * coordinate + float(phase)
        )
    perturbation /= max(float(parameters["mode_count"]), 1.0)
    return float(roughness) * float(length_scale_A) * perturbation


def _logsumexp(cupy: Any, values: Any, *, axis: int) -> Any:
    maximum = cupy.amax(values, axis=axis, keepdims=True)
    finite_maximum = cupy.where(cupy.isfinite(maximum), maximum, 0.0)
    total = cupy.sum(cupy.exp(values - finite_maximum), axis=axis)
    result = cupy.squeeze(finite_maximum, axis=axis) + cupy.log(total)
    return cupy.where(cupy.squeeze(cupy.isfinite(maximum), axis=axis), result, -cupy.inf)


def _smooth_min_pair(cupy: Any, first: Any, second: Any) -> Any:
    return (
        -cupy.logaddexp(
            -_SMOOTH_UNION_SHARPNESS * first,
            -_SMOOTH_UNION_SHARPNESS * second,
        )
        / _SMOOTH_UNION_SHARPNESS
    )


def _array(cupy: Any, value: Any) -> Any:
    return cupy.asarray(value, dtype=cupy.float64)


def _positive_environment_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_int(value: int, name: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if integer <= 0 or integer != value:
        raise ValueError(f"{name} must be a positive integer")
    return integer


__all__ = [
    "cuda_backend_available",
    "cuda_device",
    "evaluate_sdf_cupy",
    "recommended_gpu_workload",
    "voxelize_geometry_cupy",
    "voxelize_unit_masks_cupy",
]
