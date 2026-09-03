from __future__ import annotations

import numpy as np

from porous_film.config import MeasurementSpec
from porous_film.metrics import measure_final_geometry
from porous_film.metrics.final_geometry import _resample_centerline, _smoothed_track_points
from porous_film.voxel import PhaseGrid


def _grid_from_slice_centers(
    centers_by_z: list[list[tuple[float, float]]],
    *,
    box_xy_A: tuple[float, float] = (12.0, 12.0),
    spacing_A: float = 1.0,
    radius_A: float = 1.8,
) -> PhaseGrid:
    nz = len(centers_by_z)
    nx = round(box_xy_A[0] / spacing_A)
    ny = round(box_xy_A[1] / spacing_A)
    y, x = np.indices((ny, nx), dtype=float)
    x = (x + 0.5) * spacing_A
    y = (y + 0.5) * spacing_A
    mask = np.zeros((nz, ny, nx), dtype=bool)
    for z_index, centers in enumerate(centers_by_z):
        for center_x, center_y in centers:
            dx = np.abs(x - center_x)
            dy = np.abs(y - center_y)
            dx = np.minimum(dx, box_xy_A[0] - dx)
            dy = np.minimum(dy, box_xy_A[1] - dy)
            mask[z_index] |= dx**2 + dy**2 <= radius_A**2
    return PhaseGrid(
        pore_mask=mask,
        origin_A=np.zeros(3),
        spacing_A=spacing_A,
        target_box_A=np.array([box_xy_A[0], box_xy_A[1], nz * spacing_A]),
    )


def _measurement_spec(**updates: float) -> MeasurementSpec:
    values = {
        "z_slice_spacing_A": 1.0,
        "center_min_separation_A": 2.0,
        "center_tracking_max_displacement_A": 3.0,
        "centerline_sample_spacing_A": 1.0,
        "cross_section_spacing_A": 1.0,
        "boundary_resample_spacing_A": 0.5,
        "curvature_smoothing_length_A": 1.0,
        "branch_exclusion_length_A": 2.0,
        "surface_exclusion_length_A": 1.0,
        "orientation_projection_min_fraction": 0.05,
    }
    values.update(updates)
    return MeasurementSpec.model_validate(values)


def test_centerline_resampling_uses_configured_physical_arc_spacing() -> None:
    points = np.array(
        [
            [1.0, 2.0, 0.0],
            [1.0, 2.0, 1.0],
            [1.0, 2.0, 2.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 4.0],
        ]
    )
    wall_distances = np.arange(1.0, 6.0)

    sampled_points, sampled_walls = _resample_centerline(
        points,
        wall_distances,
        sample_spacing_A=2.5,
    )

    np.testing.assert_allclose(sampled_points[:, 2], [0.0, 2.5, 4.0])
    np.testing.assert_allclose(sampled_walls, [1.0, 3.5, 5.0])


def test_centerline_smoothing_preserves_resolved_tortuosity() -> None:
    z = np.arange(12, dtype=float)
    points = np.column_stack(
        (
            2.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, z.size)),
            np.zeros_like(z),
            z,
        )
    )

    def tortuosity(values: np.ndarray) -> float:
        arc_length = float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum())
        end_distance = float(np.linalg.norm(values[-1] - values[0]))
        return arc_length / end_distance

    original_excess = tortuosity(points) - 1.0
    smoothed_excess = tortuosity(_smoothed_track_points(points)) - 1.0

    assert smoothed_excess >= 0.9 * original_excess


def test_centerline_sample_spacing_controls_final_channel_arc_measurement() -> None:
    centers = [[(4.0 if z_index % 2 == 0 else 8.0, 6.0)] for z_index in range(12)]
    grid = _grid_from_slice_centers(centers, radius_A=1.4)

    fine = measure_final_geometry(
        grid,
        _measurement_spec(
            center_tracking_max_displacement_A=5.0,
            centerline_sample_spacing_A=0.5,
            cross_section_spacing_A=1.0,
            surface_exclusion_length_A=0.0,
        ),
    )
    coarse = measure_final_geometry(
        grid,
        _measurement_spec(
            center_tracking_max_displacement_A=5.0,
            centerline_sample_spacing_A=20.0,
            cross_section_spacing_A=1.0,
            surface_exclusion_length_A=0.0,
        ),
    )

    assert fine.channel_geometries[0].valid
    assert coarse.channel_geometries[0].valid
    assert fine.channel_geometries[0].arc_length_A > coarse.channel_geometries[0].arc_length_A


def test_final_geometry_extracts_one_straight_through_centerline() -> None:
    grid = _grid_from_slice_centers([[(6.0, 6.0)]] * 8)

    measured = measure_final_geometry(grid, _measurement_spec())

    assert len(measured.slice_centers) == 8
    assert all(len(slice_record.centers) == 1 for slice_record in measured.slice_centers)
    assert len(measured.centerlines) == 1
    track = measured.centerlines[0]
    assert track.touches_z_lower
    assert track.touches_z_upper
    assert track.is_through
    np.testing.assert_allclose(
        track.points_unwrapped_A[:, :2],
        np.tile(np.array([6.0, 6.0]), (8, 1)),
        atol=0.6,
    )


def test_orientation_aspect_ratio_tolerance_controls_final_identifiability() -> None:
    grid = _grid_from_slice_centers([[(6.0, 6.0)]] * 8)

    strict = measure_final_geometry(
        grid,
        _measurement_spec(orientation_aspect_ratio_tolerance=0.0),
    )
    tolerant = measure_final_geometry(
        grid,
        _measurement_spec(orientation_aspect_ratio_tolerance=2.0),
    )

    assert strict.projected_orientations[0].theta_xz_identifiable
    assert not tolerant.projected_orientations[0].theta_xz_identifiable


def test_final_geometry_keeps_two_periodic_edge_centerlines_distinct() -> None:
    grid = _grid_from_slice_centers(
        [[(0.5, 4.0), (11.5, 8.0)]] * 7,
        radius_A=1.1,
    )

    measured = measure_final_geometry(
        grid,
        _measurement_spec(center_min_separation_A=1.0),
    )

    assert len(measured.centerlines) == 2
    assert sum(track.is_through for track in measured.centerlines) == 2
    assert all(len(slice_record.centers) == 2 for slice_record in measured.slice_centers)


def test_final_geometry_unwraps_a_tilted_track_across_periodic_x() -> None:
    centers = [[((9.0 + 0.8 * z) % 12.0, 6.0)] for z in range(8)]
    grid = _grid_from_slice_centers(centers, radius_A=1.4)

    measured = measure_final_geometry(grid, _measurement_spec())
    track = measured.centerlines[0]

    assert track.is_through
    assert np.all(np.diff(track.points_unwrapped_A[:, 0]) >= 0.0)
    assert track.points_unwrapped_A[-1, 0] - track.points_unwrapped_A[0, 0] >= 5.0
    assert track.points_unwrapped_A[-1, 0] > 12.0


def test_final_geometry_marks_branch_neighborhoods() -> None:
    centers = [
        [(6.0, 6.0)],
        [(6.0, 6.0)],
        [(6.0, 6.0)],
        [(5.0, 6.0), (7.0, 6.0)],
        [(4.0, 6.0), (8.0, 6.0)],
        [(4.0, 6.0), (8.0, 6.0)],
    ]
    grid = _grid_from_slice_centers(centers, radius_A=1.4)

    measured = measure_final_geometry(
        grid,
        _measurement_spec(
            center_min_separation_A=1.0,
            center_tracking_max_displacement_A=3.5,
        ),
    )

    assert measured.branch_event_count >= 1
    assert any(track.has_branch_neighborhood for track in measured.centerlines)


def test_final_geometry_does_not_mark_internal_cavity_as_through() -> None:
    centers = [[], [], [(6.0, 6.0)], [(6.0, 6.0)], [(6.0, 6.0)], [], []]
    grid = _grid_from_slice_centers(centers)

    measured = measure_final_geometry(grid, _measurement_spec())

    assert len(measured.centerlines) == 1
    assert not measured.centerlines[0].is_through
    assert measured.through_centerline_count == 0


def test_final_geometry_measures_compact_eta_from_internal_final_component() -> None:
    z, y, x = np.indices((24, 24, 24), dtype=float)
    pore = ((x + 0.5 - 12.0) / 4.5) ** 2 + ((y + 0.5 - 12.0) / 2.5) ** 2 + (
        (z + 0.5 - 12.0) / 2.5
    ) ** 2 <= 1.0
    grid = PhaseGrid(
        pore_mask=pore,
        origin_A=np.zeros(3),
        spacing_A=1.0,
        target_box_A=np.array([24.0, 24.0, 24.0]),
    )

    measured = measure_final_geometry(grid, _measurement_spec())

    assert len(measured.compact_geometries) == 1
    compact = measured.compact_geometries[0]
    assert compact.valid
    assert compact.eta is not None
    assert 1.5 < compact.eta < 2.5


def test_final_geometry_excludes_z_through_component_from_compact_population() -> None:
    grid = _grid_from_slice_centers([[(6.0, 6.0)]] * 8)

    measured = measure_final_geometry(grid, _measurement_spec())

    assert measured.compact_geometries == ()


def test_xy_center_distance_uses_same_slice_periodic_distances_and_is_deterministic() -> None:
    grid = _grid_from_slice_centers(
        [[(1.0, 6.0), (5.0, 6.0)]] * 6,
        radius_A=1.2,
    )
    contract = _measurement_spec(
        center_min_separation_A=1.0,
        center_distance_bin_width_A=1.0,
        center_distance_max_A=6.0,
        center_distance_reference_samples=4096,
    )

    first = measure_final_geometry(grid, contract)
    second = measure_final_geometry(grid, contract)
    distribution = first.center_distance_xy

    np.testing.assert_array_equal(
        distribution.observed_pair_counts, second.center_distance_xy.observed_pair_counts
    )
    np.testing.assert_allclose(distribution.g_xy, second.center_distance_xy.g_xy)
    assert distribution.pair_count == 6
    peak_index = int(np.argmax(distribution.observed_pair_counts))
    assert 4.0 <= distribution.bin_centers_A[peak_index] < 5.0
    assert distribution.g_xy[peak_index] > 1.0


def test_xy_center_distance_uses_minimum_image_across_periodic_boundary() -> None:
    grid = _grid_from_slice_centers(
        [[(0.5, 5.0), (10.5, 5.0)]] * 5,
        radius_A=0.7,
    )
    measured = measure_final_geometry(
        grid,
        _measurement_spec(
            center_min_separation_A=0.5,
            center_distance_bin_width_A=0.5,
            center_distance_max_A=6.0,
        ),
    )

    distribution = measured.center_distance_xy
    peak_index = int(np.argmax(distribution.observed_pair_counts))
    assert 1.0 <= distribution.bin_centers_A[peak_index] < 2.5


def test_xy_center_distance_excludes_non_through_slice_centers() -> None:
    grid = _grid_from_slice_centers(
        [
            [(3.0, 3.0)],
            [(3.0, 3.0), (9.0, 9.0)],
            [(3.0, 3.0), (9.0, 9.0)],
            [(3.0, 3.0)],
        ],
        radius_A=1.1,
    )

    measured = measure_final_geometry(
        grid,
        _measurement_spec(
            center_min_separation_A=1.0,
            center_tracking_max_displacement_A=2.0,
        ),
    )

    assert measured.through_centerline_count == 1
    assert measured.center_distance_xy.pair_count == 0


def test_branch_neighborhoods_are_excluded_from_formal_section_and_channel_metrics() -> None:
    centers = [
        [(6.0, 6.0)],
        [(6.0, 6.0)],
        [(6.0, 6.0)],
        [(5.0, 6.0), (7.0, 6.0)],
        [(4.0, 6.0), (8.0, 6.0)],
        [(4.0, 6.0), (8.0, 6.0)],
    ]
    grid = _grid_from_slice_centers(centers, radius_A=1.4)

    measured = measure_final_geometry(
        grid,
        _measurement_spec(
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
    assert all(
        not channel.valid and channel.invalid_reason == "branch_neighborhood"
        for channel in measured.channel_geometries
        if channel.track_id in branched_ids
    )


def _tube_grid(
    *,
    tangent: np.ndarray,
    radius_A: float,
    spacing_A: float = 0.5,
    box_A: tuple[float, float, float] = (24.0, 24.0, 24.0),
    ellipse_axes_A: tuple[float, float] | None = None,
    angular_corrugation: float = 0.0,
) -> PhaseGrid:
    nx, ny, nz = [round(length / spacing_A) for length in box_A]
    z, y, x = np.indices((nz, ny, nx), dtype=float)
    points = np.column_stack(
        [
            (x.ravel() + 0.5) * spacing_A,
            (y.ravel() + 0.5) * spacing_A,
            (z.ravel() + 0.5) * spacing_A,
        ]
    )
    direction = np.asarray(tangent, dtype=float)
    direction /= np.linalg.norm(direction)
    center = 0.5 * np.asarray(box_A, dtype=float)
    relative = points - center
    axial = relative @ direction
    radial = relative - axial[:, None] * direction

    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(reference, direction)) > 0.8:
        reference = np.array([0.0, 1.0, 0.0])
    first_axis = np.cross(direction, reference)
    first_axis /= np.linalg.norm(first_axis)
    second_axis = np.cross(direction, first_axis)
    first = radial @ first_axis
    second = radial @ second_axis

    if ellipse_axes_A is not None:
        a, b = ellipse_axes_A
        inside = (first / a) ** 2 + (second / b) ** 2 <= 1.0
    elif angular_corrugation:
        angle = np.arctan2(second, first)
        local_radius = radius_A * (1.0 + angular_corrugation * np.cos(4.0 * angle))
        inside = np.sqrt(first**2 + second**2) <= local_radius
    else:
        inside = np.sqrt(first**2 + second**2) <= radius_A

    return PhaseGrid(
        pore_mask=inside.reshape((nz, ny, nx)),
        origin_A=np.zeros(3),
        spacing_A=spacing_A,
        target_box_A=np.asarray(box_A, dtype=float),
    )


def _valid_cross_sections(measured):
    return [section for section in measured.cross_sections if section.valid]


def test_normal_cross_sections_measure_straight_cylinder_equivalent_diameter() -> None:
    grid = _tube_grid(tangent=np.array([0.0, 0.0, 1.0]), radius_A=3.0)

    measured = measure_final_geometry(
        grid,
        _measurement_spec(
            z_slice_spacing_A=0.5,
            center_min_separation_A=2.0,
            center_tracking_max_displacement_A=1.0,
            cross_section_spacing_A=2.0,
            boundary_resample_spacing_A=0.25,
            surface_exclusion_length_A=2.0,
        ),
    )
    sections = _valid_cross_sections(measured)

    assert len(sections) >= 6
    assert np.isclose(
        np.median([section.equivalent_diameter_A for section in sections]), 6.0, atol=0.7
    )


def test_normal_cross_sections_remove_tilt_bias_from_equivalent_diameter() -> None:
    vertical = measure_final_geometry(
        _tube_grid(tangent=np.array([0.0, 0.0, 1.0]), radius_A=3.0),
        _measurement_spec(z_slice_spacing_A=0.5, cross_section_spacing_A=2.0),
    )
    tilted = measure_final_geometry(
        _tube_grid(tangent=np.array([0.35, 0.0, 1.0]), radius_A=3.0),
        _measurement_spec(
            z_slice_spacing_A=0.5,
            center_tracking_max_displacement_A=1.5,
            cross_section_spacing_A=2.0,
        ),
    )

    vertical_diameter = np.median(
        [section.equivalent_diameter_A for section in _valid_cross_sections(vertical)]
    )
    tilted_diameter = np.median(
        [section.equivalent_diameter_A for section in _valid_cross_sections(tilted)]
    )

    assert np.isclose(tilted_diameter, vertical_diameter, atol=0.7)


def test_normal_cross_sections_use_equal_area_diameter_for_ellipse() -> None:
    grid = _tube_grid(
        tangent=np.array([0.0, 0.0, 1.0]),
        radius_A=3.0,
        ellipse_axes_A=(2.0, 4.0),
    )

    measured = measure_final_geometry(
        grid,
        _measurement_spec(z_slice_spacing_A=0.5, cross_section_spacing_A=2.0),
    )
    diameter = np.median(
        [section.equivalent_diameter_A for section in _valid_cross_sections(measured)]
    )

    assert np.isclose(diameter, 2.0 * np.sqrt(8.0), atol=0.8)


def test_normal_cross_section_curvature_detects_angular_corrugation() -> None:
    smooth = measure_final_geometry(
        _tube_grid(tangent=np.array([0.0, 0.0, 1.0]), radius_A=3.0),
        _measurement_spec(
            z_slice_spacing_A=0.5,
            cross_section_spacing_A=2.0,
            boundary_resample_spacing_A=0.25,
            curvature_smoothing_length_A=0.75,
        ),
    )
    corrugated = measure_final_geometry(
        _tube_grid(
            tangent=np.array([0.0, 0.0, 1.0]),
            radius_A=3.0,
            angular_corrugation=0.22,
        ),
        _measurement_spec(
            z_slice_spacing_A=0.5,
            cross_section_spacing_A=2.0,
            boundary_resample_spacing_A=0.25,
            curvature_smoothing_length_A=0.75,
        ),
    )

    smooth_w = np.median(
        [section.curvature_fluctuation for section in _valid_cross_sections(smooth)]
    )
    corrugated_w = np.median(
        [section.curvature_fluctuation for section in _valid_cross_sections(corrugated)]
    )

    assert corrugated_w > smooth_w + 0.15


def test_projected_orientation_recovers_xz_and_xy_angles() -> None:
    direction = np.array([0.42, 0.11, 0.90])
    direction /= np.linalg.norm(direction)
    grid = _tube_grid(tangent=direction, radius_A=2.5)

    measured = measure_final_geometry(
        grid,
        _measurement_spec(
            z_slice_spacing_A=0.5,
            center_tracking_max_displacement_A=1.5,
            cross_section_spacing_A=2.0,
        ),
    )
    orientation = measured.projected_orientations[0]

    expected_xz = np.degrees(np.arctan2(abs(direction[2]), abs(direction[0])))
    expected_xy = np.degrees(np.arctan2(abs(direction[1]), abs(direction[0])))
    assert orientation.theta_xz_identifiable
    assert orientation.theta_xy_identifiable
    assert np.isclose(orientation.theta_xz_deg, expected_xz, atol=5.0)
    assert np.isclose(orientation.theta_xy_deg, expected_xy, atol=5.0)


def test_projected_orientation_marks_xy_angle_unidentifiable_for_pure_z() -> None:
    grid = _tube_grid(tangent=np.array([0.0, 0.0, 1.0]), radius_A=2.5)

    measured = measure_final_geometry(
        grid,
        _measurement_spec(
            z_slice_spacing_A=0.5,
            orientation_projection_min_fraction=0.05,
        ),
    )
    orientation = measured.projected_orientations[0]

    assert orientation.theta_xz_identifiable
    assert np.isclose(orientation.theta_xz_deg, 90.0, atol=1.0)
    assert not orientation.theta_xy_identifiable
    assert orientation.theta_xy_deg is None


def test_channel_geometry_uses_mean_cross_section_area_for_eta() -> None:
    grid = _tube_grid(tangent=np.array([0.0, 0.0, 1.0]), radius_A=3.0)

    measured = measure_final_geometry(
        grid,
        _measurement_spec(
            z_slice_spacing_A=0.5,
            cross_section_spacing_A=1.5,
            surface_exclusion_length_A=1.5,
        ),
    )
    channel = measured.channel_geometries[0]

    assert channel.valid
    assert np.isclose(channel.equivalent_diameter_A, 6.0, atol=0.7)
    assert np.isclose(
        channel.eta,
        channel.arc_length_A / channel.equivalent_diameter_A,
    )
    assert np.isclose(channel.tortuosity, 1.0, atol=0.03)
