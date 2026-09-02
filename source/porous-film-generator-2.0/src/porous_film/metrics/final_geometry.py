from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.stats import qmc
from skimage.feature import peak_local_max
from skimage.measure import find_contours

from porous_film.config import MeasurementSpec
from porous_film.performance import profile_stage
from porous_film.voxel import PhaseGrid


@dataclass(frozen=True)
class SliceCenter:
    xy_A: np.ndarray
    wall_distance_A: float


@dataclass(frozen=True)
class SliceCenterRecord:
    z_index: int
    z_A: float
    centers: tuple[SliceCenter, ...]


@dataclass(frozen=True)
class CenterlineTrack:
    track_id: int
    slice_indices: np.ndarray
    points_wrapped_A: np.ndarray
    points_unwrapped_A: np.ndarray
    wall_distances_A: np.ndarray
    touches_z_lower: bool
    touches_z_upper: bool
    is_through: bool
    has_branch_neighborhood: bool


@dataclass(frozen=True)
class XYCenterDistanceDistribution:
    bin_edges_A: np.ndarray
    bin_centers_A: np.ndarray
    observed_pair_counts: np.ndarray
    reference_pair_counts: np.ndarray
    g_xy: np.ndarray
    pair_count: int
    valid_slice_count: int


@dataclass(frozen=True)
class CrossSectionMeasurement:
    track_id: int
    arc_position_A: float
    center_A: np.ndarray
    tangent: np.ndarray
    area_A2: float | None
    equivalent_diameter_A: float | None
    curvature_fluctuation: float | None
    valid: bool
    invalid_reason: str | None


@dataclass(frozen=True)
class ProjectedOrientationMeasurement:
    track_id: int
    axis: np.ndarray
    theta_xz_deg: float | None
    theta_xy_deg: float | None
    theta_xz_identifiable: bool
    theta_xy_identifiable: bool


@dataclass(frozen=True)
class ChannelGeometryMeasurement:
    track_id: int
    arc_length_A: float | None
    end_distance_A: float | None
    equivalent_diameter_A: float | None
    eta: float | None
    tortuosity: float | None
    valid: bool
    invalid_reason: str | None


@dataclass(frozen=True)
class FinalGeometryMeasurements:
    porosity: float
    slice_centers: tuple[SliceCenterRecord, ...]
    centerlines: tuple[CenterlineTrack, ...]
    through_centerline_count: int
    branch_event_count: int
    center_distance_xy: XYCenterDistanceDistribution
    cross_sections: tuple[CrossSectionMeasurement, ...]
    projected_orientations: tuple[ProjectedOrientationMeasurement, ...]
    channel_geometries: tuple[ChannelGeometryMeasurement, ...]


@dataclass
class _MutableTrack:
    track_id: int
    slice_indices: list[int]
    wrapped_points_A: list[np.ndarray]
    unwrapped_points_A: list[np.ndarray]
    wall_distances_A: list[float]
    branch_z_A: list[float]


def measure_final_geometry(
    grid: PhaseGrid,
    contract: MeasurementSpec,
) -> FinalGeometryMeasurements:
    with profile_stage("final_measurement"):
        return _measure_final_geometry(grid, contract)


def _measure_final_geometry(
    grid: PhaseGrid,
    contract: MeasurementSpec,
) -> FinalGeometryMeasurements:
    with profile_stage("centerline_generation"):
        sampled_indices = _sampled_z_indices(
            grid.pore_mask.shape[0],
            grid.spacing_A,
            contract.z_slice_spacing_A,
        )
        slices = tuple(
            _measure_slice_centers(grid, int(z_index), contract)
            for z_index in sampled_indices
        )
        centerlines, branch_count, branch_z_by_track = _track_slice_centers(
            slices,
            target_box_A=grid.target_box_A,
            maximum_displacement_A=contract.center_tracking_max_displacement_A,
            lower_index=int(sampled_indices[0]),
            upper_index=int(sampled_indices[-1]),
        )
    through_slices = _through_center_slices(slices, centerlines)
    center_distance = _measure_xy_center_distance_distribution(
        through_slices,
        box_xy_A=np.asarray(grid.target_box_A[:2], dtype=float),
        bin_width_A=contract.center_distance_bin_width_A,
        maximum_distance_A=contract.center_distance_max_A,
        reference_samples=contract.center_distance_reference_samples,
    )
    cross_sections = _measure_normal_cross_sections(
        grid,
        centerlines,
        contract,
        branch_z_by_track=branch_z_by_track,
    )
    projected_orientations = tuple(
        _projected_orientation(
            track,
            minimum_projection_fraction=contract.orientation_projection_min_fraction,
        )
        for track in centerlines
    )
    channel_geometries = tuple(
        _channel_geometry(track, cross_sections)
        for track in centerlines
    )
    return FinalGeometryMeasurements(
        porosity=grid.porosity,
        slice_centers=slices,
        centerlines=centerlines,
        through_centerline_count=sum(track.is_through for track in centerlines),
        branch_event_count=branch_count,
        center_distance_xy=center_distance,
        cross_sections=cross_sections,
        projected_orientations=projected_orientations,
        channel_geometries=channel_geometries,
    )


def _projected_orientation(
    track: CenterlineTrack,
    *,
    minimum_projection_fraction: float,
) -> ProjectedOrientationMeasurement:
    if track.points_unwrapped_A.shape[0] < 2:
        return ProjectedOrientationMeasurement(
            track_id=track.track_id,
            axis=np.array([np.nan, np.nan, np.nan]),
            theta_xz_deg=None,
            theta_xy_deg=None,
            theta_xz_identifiable=False,
            theta_xy_identifiable=False,
        )
    axis = track.points_unwrapped_A[-1] - track.points_unwrapped_A[0]
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        return ProjectedOrientationMeasurement(
            track_id=track.track_id,
            axis=np.array([np.nan, np.nan, np.nan]),
            theta_xz_deg=None,
            theta_xy_deg=None,
            theta_xz_identifiable=False,
            theta_xy_identifiable=False,
        )
    axis = axis / norm
    xz_projection = float(np.hypot(axis[0], axis[2]))
    xy_projection = float(np.hypot(axis[0], axis[1]))
    xz_identifiable = xz_projection >= float(minimum_projection_fraction)
    xy_identifiable = xy_projection >= float(minimum_projection_fraction)
    theta_xz = (
        float(np.degrees(np.arctan2(abs(axis[2]), abs(axis[0]))))
        if xz_identifiable
        else None
    )
    theta_xy = (
        float(np.degrees(np.arctan2(abs(axis[1]), abs(axis[0]))))
        if xy_identifiable
        else None
    )
    return ProjectedOrientationMeasurement(
        track_id=track.track_id,
        axis=axis,
        theta_xz_deg=theta_xz,
        theta_xy_deg=theta_xy,
        theta_xz_identifiable=xz_identifiable,
        theta_xy_identifiable=xy_identifiable,
    )


def _channel_geometry(
    track: CenterlineTrack,
    cross_sections: tuple[CrossSectionMeasurement, ...],
) -> ChannelGeometryMeasurement:
    if track.has_branch_neighborhood:
        return _invalid_channel_geometry(track.track_id, "branch_neighborhood")
    if track.points_unwrapped_A.shape[0] < 2:
        return _invalid_channel_geometry(track.track_id, "insufficient_centerline_points")
    segment_lengths = np.linalg.norm(np.diff(track.points_unwrapped_A, axis=0), axis=1)
    arc_length = float(np.sum(segment_lengths))
    end_distance = float(
        np.linalg.norm(track.points_unwrapped_A[-1] - track.points_unwrapped_A[0])
    )
    valid_sections = [
        section
        for section in cross_sections
        if section.track_id == track.track_id
        and section.valid
        and section.area_A2 is not None
    ]
    if not valid_sections:
        return _invalid_channel_geometry(track.track_id, "no_valid_cross_sections")
    mean_area = float(np.mean([float(section.area_A2) for section in valid_sections]))
    equivalent_diameter = 2.0 * float(np.sqrt(mean_area / np.pi))
    if arc_length <= 0.0 or end_distance <= 0.0 or equivalent_diameter <= 0.0:
        return _invalid_channel_geometry(track.track_id, "nonpositive_channel_measure")
    return ChannelGeometryMeasurement(
        track_id=track.track_id,
        arc_length_A=arc_length,
        end_distance_A=end_distance,
        equivalent_diameter_A=equivalent_diameter,
        eta=arc_length / equivalent_diameter,
        tortuosity=arc_length / end_distance,
        valid=True,
        invalid_reason=None,
    )


def _invalid_channel_geometry(
    track_id: int,
    reason: str,
) -> ChannelGeometryMeasurement:
    return ChannelGeometryMeasurement(
        track_id=track_id,
        arc_length_A=None,
        end_distance_A=None,
        equivalent_diameter_A=None,
        eta=None,
        tortuosity=None,
        valid=False,
        invalid_reason=reason,
    )


def _measure_normal_cross_sections(
    grid: PhaseGrid,
    tracks: tuple[CenterlineTrack, ...],
    contract: MeasurementSpec,
    *,
    branch_z_by_track: dict[int, np.ndarray],
) -> tuple[CrossSectionMeasurement, ...]:
    output: list[CrossSectionMeasurement] = []
    for track in tracks:
        points = _smoothed_track_points(track.points_unwrapped_A)
        if points.shape[0] < 2:
            continue
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total_length = float(cumulative[-1])
        if total_length <= 0.0:
            continue
        branch_z = branch_z_by_track.get(track.track_id, np.empty(0, dtype=float))
        branch_arc_positions = (
            np.interp(branch_z, points[:, 2], cumulative)
            if branch_z.size
            else np.empty(0, dtype=float)
        )
        lower = float(contract.surface_exclusion_length_A)
        upper = total_length - float(contract.surface_exclusion_length_A)
        if upper < lower:
            positions = np.array([0.5 * total_length], dtype=float)
        else:
            positions = np.arange(
                lower,
                upper + 0.5 * contract.cross_section_spacing_A,
                contract.cross_section_spacing_A,
                dtype=float,
            )
            positions = positions[positions <= upper + 1.0e-9]
        for arc_position in positions:
            center, tangent, wall_distance = _interpolate_track(
                points,
                track.wall_distances_A,
                cumulative,
                float(arc_position),
            )
            if (
                branch_arc_positions.size
                and contract.branch_exclusion_length_A > 0.0
                and np.min(np.abs(branch_arc_positions - arc_position))
                <= contract.branch_exclusion_length_A + 1.0e-9
            ):
                output.append(
                    _invalid_cross_section(
                        track.track_id,
                        float(arc_position),
                        center,
                        tangent,
                        "branch_neighborhood",
                    )
                )
                continue
            output.append(
                _measure_one_cross_section(
                    grid,
                    track_id=track.track_id,
                    arc_position_A=float(arc_position),
                    center_A=center,
                    tangent=tangent,
                    wall_distance_A=wall_distance,
                    contract=contract,
                )
            )
    return tuple(output)


def _smoothed_track_points(points_A: np.ndarray) -> np.ndarray:
    points = np.asarray(points_A, dtype=float)
    if points.shape[0] < 4:
        return points.copy()
    smoothed = np.column_stack(
        [
            ndimage.gaussian_filter1d(points[:, axis], sigma=1.0, mode="nearest")
            for axis in range(3)
        ]
    )
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return smoothed


def _interpolate_track(
    points_A: np.ndarray,
    wall_distances_A: np.ndarray,
    cumulative_A: np.ndarray,
    arc_position_A: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if arc_position_A <= 0.0:
        index = 0
        fraction = 0.0
    elif arc_position_A >= cumulative_A[-1]:
        index = points_A.shape[0] - 2
        fraction = 1.0
    else:
        index = int(np.searchsorted(cumulative_A, arc_position_A, side="right") - 1)
        span = float(cumulative_A[index + 1] - cumulative_A[index])
        fraction = (arc_position_A - float(cumulative_A[index])) / max(span, 1.0e-12)
    center = (1.0 - fraction) * points_A[index] + fraction * points_A[index + 1]
    tangent = points_A[index + 1] - points_A[index]
    tangent /= np.linalg.norm(tangent)
    wall = (1.0 - fraction) * wall_distances_A[index] + fraction * wall_distances_A[
        min(index + 1, wall_distances_A.size - 1)
    ]
    return center, tangent, float(wall)


def _measure_one_cross_section(
    grid: PhaseGrid,
    *,
    track_id: int,
    arc_position_A: float,
    center_A: np.ndarray,
    tangent: np.ndarray,
    wall_distance_A: float,
    contract: MeasurementSpec,
) -> CrossSectionMeasurement:
    first_axis, second_axis = _normal_plane_axes(tangent)
    plane_spacing = min(
        0.5 * float(grid.spacing_A),
        float(contract.boundary_resample_spacing_A),
    )
    plane_spacing = max(plane_spacing, 0.25 * float(grid.spacing_A))
    base_extent = max(
        3.0 * float(wall_distance_A),
        4.0 * float(grid.spacing_A),
        2.0 * float(contract.center_min_separation_A),
    )
    maximum_extent = 0.75 * float(np.linalg.norm(grid.target_box_A))
    extent = min(base_extent, maximum_extent)
    for _attempt in range(4):
        component, coordinates = _sample_normal_component(
            grid,
            center_A=center_A,
            first_axis=first_axis,
            second_axis=second_axis,
            half_extent_A=extent,
            plane_spacing_A=plane_spacing,
        )
        if component is None:
            return _invalid_cross_section(
                track_id,
                arc_position_A,
                center_A,
                tangent,
                "center_not_in_pore",
            )
        if not _component_touches_border(component):
            return _cross_section_from_component(
                component,
                coordinates,
                track_id=track_id,
                arc_position_A=arc_position_A,
                center_A=center_A,
                tangent=tangent,
                plane_spacing_A=plane_spacing,
                contract=contract,
            )
        if extent >= maximum_extent:
            break
        extent = min(1.75 * extent, maximum_extent)
    return _invalid_cross_section(
        track_id,
        arc_position_A,
        center_A,
        tangent,
        "cross_section_not_bounded",
    )


def _normal_plane_axes(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(tangent, dtype=float)
    direction /= np.linalg.norm(direction)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(direction, reference))) > 0.85:
        reference = np.array([0.0, 1.0, 0.0])
    first = np.cross(direction, reference)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    second /= np.linalg.norm(second)
    return first, second


def _sample_normal_component(
    grid: PhaseGrid,
    *,
    center_A: np.ndarray,
    first_axis: np.ndarray,
    second_axis: np.ndarray,
    half_extent_A: float,
    plane_spacing_A: float,
) -> tuple[np.ndarray | None, np.ndarray]:
    coordinate = np.arange(
        -half_extent_A,
        half_extent_A + 0.5 * plane_spacing_A,
        plane_spacing_A,
        dtype=float,
    )
    vv, uu = np.meshgrid(coordinate, coordinate, indexing="ij")
    world = (
        center_A[np.newaxis, np.newaxis, :]
        + uu[:, :, np.newaxis] * first_axis
        + vv[:, :, np.newaxis] * second_axis
    )
    nx = grid.pore_mask.shape[2]
    ny = grid.pore_mask.shape[1]
    x_index = (world[:, :, 0] - grid.origin_A[0]) / grid.spacing_A - 0.5
    y_index = (world[:, :, 1] - grid.origin_A[1]) / grid.spacing_A - 0.5
    z_index = (world[:, :, 2] - grid.origin_A[2]) / grid.spacing_A - 0.5
    x_index = np.mod(x_index, nx)
    y_index = np.mod(y_index, ny)
    sampled = ndimage.map_coordinates(
        grid.pore_mask.astype(float),
        [z_index, y_index, x_index],
        order=1,
        mode="constant",
        cval=0.0,
    )
    plane_mask = sampled >= 0.5
    labels, _ = ndimage.label(plane_mask, structure=np.ones((3, 3), dtype=int))
    middle = coordinate.size // 2
    label = int(labels[middle, middle])
    if label == 0:
        window = labels[
            max(0, middle - 1) : min(labels.shape[0], middle + 2),
            max(0, middle - 1) : min(labels.shape[1], middle + 2),
        ]
        candidates = window[window > 0]
        if candidates.size:
            label = int(candidates[0])
    if label == 0:
        return None, coordinate
    return labels == label, coordinate


def _component_touches_border(component: np.ndarray) -> bool:
    return bool(
        np.any(component[0])
        or np.any(component[-1])
        or np.any(component[:, 0])
        or np.any(component[:, -1])
    )


def _cross_section_from_component(
    component: np.ndarray,
    coordinates_A: np.ndarray,
    *,
    track_id: int,
    arc_position_A: float,
    center_A: np.ndarray,
    tangent: np.ndarray,
    plane_spacing_A: float,
    contract: MeasurementSpec,
) -> CrossSectionMeasurement:
    contours = find_contours(component.astype(float), 0.5)
    closed = [contour for contour in contours if contour.shape[0] >= 8]
    if not closed:
        return _invalid_cross_section(
            track_id,
            arc_position_A,
            center_A,
            tangent,
            "boundary_not_found",
        )
    contour = max(closed, key=lambda item: item.shape[0])
    plane_xy = np.column_stack(
        [
            coordinates_A[0] + contour[:, 1] * plane_spacing_A,
            coordinates_A[0] + contour[:, 0] * plane_spacing_A,
        ]
    )
    if np.linalg.norm(plane_xy[0] - plane_xy[-1]) > 1.5 * plane_spacing_A:
        return _invalid_cross_section(
            track_id,
            arc_position_A,
            center_A,
            tangent,
            "open_boundary",
        )
    area = abs(_polygon_area_A2(plane_xy))
    if area <= 0.0:
        return _invalid_cross_section(
            track_id,
            arc_position_A,
            center_A,
            tangent,
            "nonpositive_area",
        )
    equivalent_diameter = 2.0 * float(np.sqrt(area / np.pi))
    curvature_fluctuation = _closed_curve_curvature_fluctuation(
        plane_xy,
        resample_spacing_A=contract.boundary_resample_spacing_A,
        smoothing_length_A=contract.curvature_smoothing_length_A,
    )
    return CrossSectionMeasurement(
        track_id=track_id,
        arc_position_A=arc_position_A,
        center_A=np.asarray(center_A, dtype=float),
        tangent=np.asarray(tangent, dtype=float),
        area_A2=area,
        equivalent_diameter_A=equivalent_diameter,
        curvature_fluctuation=curvature_fluctuation,
        valid=True,
        invalid_reason=None,
    )


def _polygon_area_A2(points_xy_A: np.ndarray) -> float:
    x = points_xy_A[:, 0]
    y = points_xy_A[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _closed_curve_curvature_fluctuation(
    points_xy_A: np.ndarray,
    *,
    resample_spacing_A: float,
    smoothing_length_A: float,
) -> float:
    points = np.asarray(points_xy_A, dtype=float)
    if np.linalg.norm(points[0] - points[-1]) <= 1.0e-12:
        points = points[:-1]
    closed = np.vstack([points, points[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(np.sum(lengths))
    if perimeter <= 0.0:
        return float("nan")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    sample_count = max(16, int(np.ceil(perimeter / float(resample_spacing_A))))
    sample_s = np.linspace(0.0, perimeter, sample_count, endpoint=False)
    sampled = np.column_stack(
        [
            np.interp(sample_s, cumulative, closed[:, axis])
            for axis in range(2)
        ]
    )
    sigma = float(smoothing_length_A) / max(float(resample_spacing_A), 1.0e-12)
    if sigma > 0.0:
        sampled = np.column_stack(
            [
                ndimage.gaussian_filter1d(sampled[:, axis], sigma=sigma, mode="wrap")
                for axis in range(2)
            ]
        )
    step = perimeter / sample_count
    first = (np.roll(sampled, -1, axis=0) - np.roll(sampled, 1, axis=0)) / (
        2.0 * step
    )
    second = (
        np.roll(sampled, -1, axis=0)
        - 2.0 * sampled
        + np.roll(sampled, 1, axis=0)
    ) / (step**2)
    denominator = np.maximum(np.sum(first**2, axis=1) ** 1.5, 1.0e-12)
    curvature = (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / denominator
    mean_curvature = float(np.mean(curvature))
    return float(np.std(curvature) / max(abs(mean_curvature), 1.0e-12))


def _invalid_cross_section(
    track_id: int,
    arc_position_A: float,
    center_A: np.ndarray,
    tangent: np.ndarray,
    reason: str,
) -> CrossSectionMeasurement:
    return CrossSectionMeasurement(
        track_id=track_id,
        arc_position_A=arc_position_A,
        center_A=np.asarray(center_A, dtype=float),
        tangent=np.asarray(tangent, dtype=float),
        area_A2=None,
        equivalent_diameter_A=None,
        curvature_fluctuation=None,
        valid=False,
        invalid_reason=reason,
    )


def _measure_xy_center_distance_distribution(
    slices: tuple[SliceCenterRecord, ...],
    *,
    box_xy_A: np.ndarray,
    bin_width_A: float,
    maximum_distance_A: float | None,
    reference_samples: int,
) -> XYCenterDistanceDistribution:
    maximum = (
        float(maximum_distance_A)
        if maximum_distance_A is not None
        else float(np.linalg.norm(0.5 * np.asarray(box_xy_A, dtype=float)))
    )
    bin_width = float(bin_width_A)
    bin_count = max(1, int(np.ceil(maximum / bin_width)))
    edges = np.linspace(0.0, bin_count * bin_width, bin_count + 1)
    observed = np.zeros(bin_count, dtype=np.int64)
    pair_count = 0
    valid_slice_count = 0
    for slice_record in slices:
        if len(slice_record.centers) < 2:
            continue
        xy = np.vstack([center.xy_A for center in slice_record.centers])
        distances = _unique_periodic_xy_distances(xy, box_xy_A)
        observed += np.histogram(distances, bins=edges)[0]
        pair_count += int(distances.size)
        valid_slice_count += 1

    reference = np.zeros(bin_count, dtype=float)
    if pair_count:
        reference_probability = _uniform_periodic_xy_reference(
            box_xy_A,
            edges,
            sample_count=reference_samples,
        )
        reference = reference_probability * float(pair_count)

    g_xy = np.zeros(bin_count, dtype=float)
    valid = reference > 0.0
    g_xy[valid] = observed[valid] / reference[valid]
    return XYCenterDistanceDistribution(
        bin_edges_A=edges,
        bin_centers_A=0.5 * (edges[:-1] + edges[1:]),
        observed_pair_counts=observed,
        reference_pair_counts=reference,
        g_xy=g_xy,
        pair_count=pair_count,
        valid_slice_count=valid_slice_count,
    )


def _unique_periodic_xy_distances(
    points_xy_A: np.ndarray,
    box_xy_A: np.ndarray,
) -> np.ndarray:
    count = points_xy_A.shape[0]
    row, col = np.triu_indices(count, k=1)
    delta = points_xy_A[col] - points_xy_A[row]
    delta -= box_xy_A * np.round(delta / box_xy_A)
    return np.linalg.norm(delta, axis=1)


def _uniform_periodic_xy_reference(
    box_xy_A: np.ndarray,
    edges_A: np.ndarray,
    *,
    sample_count: int,
) -> np.ndarray:
    power = max(1, int(np.ceil(np.log2(max(int(sample_count), 2)))))
    samples = qmc.Sobol(d=4, scramble=False).random_base2(power)
    first = samples[:, :2] * box_xy_A
    second = samples[:, 2:] * box_xy_A
    delta = second - first
    delta -= box_xy_A * np.round(delta / box_xy_A)
    distances = np.linalg.norm(delta, axis=1)
    counts = np.histogram(distances, bins=edges_A)[0].astype(float)
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.zeros(edges_A.size - 1, dtype=float)
    return counts / total


def _sampled_z_indices(count: int, spacing_A: float, requested_spacing_A: float) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=int)
    stride = max(1, round(float(requested_spacing_A) / float(spacing_A)))
    indices = np.arange(0, count, stride, dtype=int)
    if indices[-1] != count - 1:
        indices = np.append(indices, count - 1)
    return indices


def _measure_slice_centers(
    grid: PhaseGrid,
    z_index: int,
    contract: MeasurementSpec,
) -> SliceCenterRecord:
    mask = np.asarray(grid.pore_mask[z_index], dtype=bool)
    centers = _periodic_slice_centers(
        mask,
        spacing_A=grid.spacing_A,
        origin_xy_A=grid.origin_A[:2],
        minimum_separation_A=contract.center_min_separation_A,
    )
    z_A = float(grid.origin_A[2] + (z_index + 0.5) * grid.spacing_A)
    return SliceCenterRecord(z_index=z_index, z_A=z_A, centers=centers)


def _periodic_slice_centers(
    mask_yx: np.ndarray,
    *,
    spacing_A: float,
    origin_xy_A: np.ndarray,
    minimum_separation_A: float,
) -> tuple[SliceCenter, ...]:
    mask = np.asarray(mask_yx, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("slice mask must have shape (y, x)")
    if not np.any(mask):
        return ()

    ny, nx = mask.shape
    tiled = np.tile(mask, (3, 3))
    distance = ndimage.distance_transform_edt(tiled, sampling=float(spacing_A))
    min_distance_voxels = max(1, int(np.ceil(minimum_separation_A / spacing_A)))
    coordinates = peak_local_max(
        distance,
        min_distance=min_distance_voxels,
        threshold_abs=0.5 * float(spacing_A),
        exclude_border=False,
    )
    in_center = coordinates[
        (coordinates[:, 0] >= ny)
        & (coordinates[:, 0] < 2 * ny)
        & (coordinates[:, 1] >= nx)
        & (coordinates[:, 1] < 2 * nx)
    ]
    if in_center.size == 0:
        return ()

    candidates: list[tuple[float, np.ndarray]] = []
    for tiled_y, tiled_x in in_center:
        local_y = int(tiled_y - ny)
        local_x = int(tiled_x - nx)
        xy = np.array(
            [
                float(origin_xy_A[0] + (local_x + 0.5) * spacing_A),
                float(origin_xy_A[1] + (local_y + 0.5) * spacing_A),
            ]
        )
        candidates.append((float(distance[tiled_y, tiled_x]), xy))
    candidates.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))

    box_xy = np.array([nx * spacing_A, ny * spacing_A], dtype=float)
    candidates = _merge_plateau_candidates(
        candidates,
        box_xy_A=box_xy,
        spacing_A=spacing_A,
    )
    accepted: list[SliceCenter] = []
    for wall_distance, xy in candidates:
        if any(
            _periodic_xy_distance(xy, center.xy_A, box_xy)
            <= minimum_separation_A + 1.0e-12
            for center in accepted
        ):
            continue
        accepted.append(SliceCenter(xy_A=xy, wall_distance_A=wall_distance))
    accepted = _refine_slice_centers(
        mask,
        distance[ny : 2 * ny, nx : 2 * nx],
        accepted,
        spacing_A=spacing_A,
        origin_xy_A=origin_xy_A,
        box_xy_A=box_xy,
    )
    accepted.sort(key=lambda center: (float(center.xy_A[0]), float(center.xy_A[1])))
    return tuple(accepted)


def _merge_plateau_candidates(
    candidates: list[tuple[float, np.ndarray]],
    *,
    box_xy_A: np.ndarray,
    spacing_A: float,
) -> list[tuple[float, np.ndarray]]:
    count = len(candidates)
    if count < 2:
        return candidates
    parents = np.arange(count, dtype=int)

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first in range(count):
        first_distance, first_xy = candidates[first]
        for second in range(first + 1, count):
            second_distance, second_xy = candidates[second]
            if not np.isclose(first_distance, second_distance, atol=1.0e-12):
                continue
            delta = second_xy - first_xy
            delta -= box_xy_A * np.round(delta / box_xy_A)
            if np.max(np.abs(delta)) <= float(spacing_A) + 1.0e-12:
                union(first, second)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)

    merged: list[tuple[float, np.ndarray]] = []
    for members in groups.values():
        reference = candidates[members[0]][1]
        unwrapped = []
        for member in members:
            xy = candidates[member][1]
            delta = xy - reference
            delta -= box_xy_A * np.round(delta / box_xy_A)
            unwrapped.append(reference + delta)
        merged_xy = np.mod(np.mean(np.vstack(unwrapped), axis=0), box_xy_A)
        merged.append(
            (
                max(float(candidates[member][0]) for member in members),
                merged_xy,
            )
        )
    merged.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
    return merged


def _refine_slice_centers(
    mask_yx: np.ndarray,
    distance_A: np.ndarray,
    centers: list[SliceCenter],
    *,
    spacing_A: float,
    origin_xy_A: np.ndarray,
    box_xy_A: np.ndarray,
) -> list[SliceCenter]:
    if not centers:
        return centers
    pixel_y, pixel_x = np.nonzero(mask_yx)
    pixel_xy = np.column_stack(
        [
            origin_xy_A[0] + (pixel_x.astype(float) + 0.5) * spacing_A,
            origin_xy_A[1] + (pixel_y.astype(float) + 0.5) * spacing_A,
        ]
    )
    center_xy = np.vstack([center.xy_A for center in centers])
    distances = _periodic_xy_distance_matrix(center_xy, pixel_xy, box_xy_A).T
    assignments = np.argmin(distances, axis=1)
    refined: list[SliceCenter] = []
    for index, center in enumerate(centers):
        selected = assignments == index
        if not np.any(selected):
            refined.append(center)
            continue
        deltas = pixel_xy[selected] - center.xy_A
        deltas -= box_xy_A * np.round(deltas / box_xy_A)
        weights = np.asarray(distance_A[pixel_y[selected], pixel_x[selected]], dtype=float) ** 2
        if not np.any(weights > 0.0):
            weights = np.ones_like(weights)
        xy = np.mod(center.xy_A + np.average(deltas, axis=0, weights=weights), box_xy_A)
        refined.append(SliceCenter(xy_A=xy, wall_distance_A=center.wall_distance_A))
    return refined


def _track_slice_centers(
    slices: tuple[SliceCenterRecord, ...],
    *,
    target_box_A: np.ndarray,
    maximum_displacement_A: float,
    lower_index: int,
    upper_index: int,
) -> tuple[tuple[CenterlineTrack, ...], int, dict[int, np.ndarray]]:
    box_xy = np.asarray(target_box_A[:2], dtype=float)
    tracks: dict[int, _MutableTrack] = {}
    active: dict[int, int] = {}
    branch_tracks: set[int] = set()
    branch_events = 0
    next_track_id = 0

    for slice_record in slices:
        centers = slice_record.centers
        if not centers:
            active = {}
            continue
        current_xy = np.vstack([center.xy_A for center in centers])
        previous_ids = sorted(active)
        current_to_track: dict[int, int] = {}

        if previous_ids:
            previous_xy = np.vstack(
                [tracks[track_id].wrapped_points_A[-1][:2] for track_id in previous_ids]
            )
            costs = _periodic_xy_distance_matrix(previous_xy, current_xy, box_xy)
            neighbor_mask = costs <= float(maximum_displacement_A)
            row_branch = np.sum(neighbor_mask, axis=1) > 1
            col_branch = np.sum(neighbor_mask, axis=0) > 1
            branch_events += int(np.count_nonzero(row_branch))
            branch_events += int(np.count_nonzero(col_branch))
            for row in np.flatnonzero(row_branch):
                track_id = previous_ids[int(row)]
                branch_tracks.add(track_id)
                tracks[track_id].branch_z_A.append(float(slice_record.z_A))
            for col in np.flatnonzero(col_branch):
                for row in np.flatnonzero(neighbor_mask[:, int(col)]):
                    track_id = previous_ids[int(row)]
                    branch_tracks.add(track_id)
                    tracks[track_id].branch_z_A.append(float(slice_record.z_A))
            row_indices, col_indices = linear_sum_assignment(costs)
            for row, col in zip(row_indices, col_indices, strict=True):
                if costs[row, col] > maximum_displacement_A:
                    continue
                track_id = previous_ids[int(row)]
                current_to_track[int(col)] = track_id
                if np.sum(neighbor_mask[row]) > 1 or np.sum(neighbor_mask[:, col]) > 1:
                    branch_tracks.add(track_id)

        for center_index, center in enumerate(centers):
            track_id = current_to_track.get(center_index)
            point_wrapped = np.array(
                [center.xy_A[0], center.xy_A[1], slice_record.z_A],
                dtype=float,
            )
            if track_id is None:
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = _MutableTrack(
                    track_id=track_id,
                    slice_indices=[],
                    wrapped_points_A=[],
                    unwrapped_points_A=[],
                    wall_distances_A=[],
                    branch_z_A=[],
                )
                if previous_ids:
                    distances = _periodic_xy_distance_matrix(
                        np.vstack(
                            [tracks[previous_id].wrapped_points_A[-1][:2] for previous_id in previous_ids]
                        ),
                        center.xy_A[np.newaxis, :],
                        box_xy,
                    )
                    if np.any(distances[:, 0] <= maximum_displacement_A):
                        branch_tracks.add(track_id)
                        tracks[track_id].branch_z_A.append(float(slice_record.z_A))

            track = tracks[track_id]
            if track.unwrapped_points_A:
                previous_unwrapped = track.unwrapped_points_A[-1]
                delta_xy = center.xy_A - track.wrapped_points_A[-1][:2]
                delta_xy -= box_xy * np.round(delta_xy / box_xy)
                point_unwrapped = np.array(
                    [
                        previous_unwrapped[0] + delta_xy[0],
                        previous_unwrapped[1] + delta_xy[1],
                        slice_record.z_A,
                    ],
                    dtype=float,
                )
            else:
                point_unwrapped = point_wrapped.copy()
            track.slice_indices.append(slice_record.z_index)
            track.wrapped_points_A.append(point_wrapped)
            track.unwrapped_points_A.append(point_unwrapped)
            track.wall_distances_A.append(center.wall_distance_A)
            current_to_track[center_index] = track_id
        active = {track_id: track_id for track_id in current_to_track.values()}

    output = []
    for track_id in sorted(tracks):
        track = tracks[track_id]
        indices = np.asarray(track.slice_indices, dtype=int)
        wrapped = np.vstack(track.wrapped_points_A)
        unwrapped = np.vstack(track.unwrapped_points_A)
        touches_lower = bool(indices.size and indices[0] == lower_index)
        touches_upper = bool(indices.size and indices[-1] == upper_index)
        output.append(
            CenterlineTrack(
                track_id=track_id,
                slice_indices=indices,
                points_wrapped_A=wrapped,
                points_unwrapped_A=unwrapped,
                wall_distances_A=np.asarray(track.wall_distances_A, dtype=float),
                touches_z_lower=touches_lower,
                touches_z_upper=touches_upper,
                is_through=touches_lower and touches_upper,
                has_branch_neighborhood=track_id in branch_tracks,
            )
        )
    branch_z_by_track = {
        track_id: np.unique(np.asarray(track.branch_z_A, dtype=float))
        for track_id, track in tracks.items()
        if track.branch_z_A
    }
    return tuple(output), branch_events, branch_z_by_track


def _through_center_slices(
    slices: tuple[SliceCenterRecord, ...],
    tracks: tuple[CenterlineTrack, ...],
) -> tuple[SliceCenterRecord, ...]:
    centers_by_slice: dict[int, list[SliceCenter]] = {
        record.z_index: [] for record in slices
    }
    for track in tracks:
        if not track.is_through:
            continue
        for z_index, point, wall_distance in zip(
            track.slice_indices,
            track.points_wrapped_A,
            track.wall_distances_A,
            strict=True,
        ):
            centers_by_slice[int(z_index)].append(
                SliceCenter(
                    xy_A=np.asarray(point[:2], dtype=float),
                    wall_distance_A=float(wall_distance),
                )
            )
    output = []
    for record in slices:
        centers = centers_by_slice[record.z_index]
        centers.sort(key=lambda center: (float(center.xy_A[0]), float(center.xy_A[1])))
        output.append(
            SliceCenterRecord(
                z_index=record.z_index,
                z_A=record.z_A,
                centers=tuple(centers),
            )
        )
    return tuple(output)


def _periodic_xy_distance(
    first_xy_A: np.ndarray,
    second_xy_A: np.ndarray,
    box_xy_A: np.ndarray,
) -> float:
    delta = np.asarray(second_xy_A, dtype=float) - np.asarray(first_xy_A, dtype=float)
    delta -= box_xy_A * np.round(delta / box_xy_A)
    return float(np.linalg.norm(delta))


def _periodic_xy_distance_matrix(
    first_xy_A: np.ndarray,
    second_xy_A: np.ndarray,
    box_xy_A: np.ndarray,
) -> np.ndarray:
    delta = second_xy_A[np.newaxis, :, :] - first_xy_A[:, np.newaxis, :]
    delta -= box_xy_A[np.newaxis, np.newaxis, :] * np.round(
        delta / box_xy_A[np.newaxis, np.newaxis, :]
    )
    return np.linalg.norm(delta, axis=2)


