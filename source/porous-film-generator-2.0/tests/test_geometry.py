from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from porous_film.centers import CenterSeedPlan
from porous_film.config import GeneratorConfig
from porous_film.geometry import (
    BuiltGeometry,
    ChannelUnit,
    CompactUnit,
    PoreGeometry,
    adjust_channel_lateral_deviations_xy,
    build_units,
    minimum_scale_for_channels_through_z,
    scale_built_geometry,
    separate_channel_footprints_xy,
)
from porous_film.metrics import pore_z_connectivity_summary
from porous_film.voxel import voxelize_geometry


def _geometry_config(
    *,
    channel_fraction: float = 0.0,
    compact_relative_volume: dict | None = None,
    compact_eta: float = 1.0,
    channel_eta: float = 2.0,
    channel_tau: float = 1.0,
    channel_roughness: float = 0.0,
    orientation_distribution: dict | None = None,
) -> GeneratorConfig:
    return GeneratorConfig.model_validate(
        {
            "task": {"name": "geometry-test", "random_seed": 7},
            "film": {
                "target_box_A": {"x": 20, "y": 20, "z": 20},
                "packing_box_A": {"x": 20, "y": 20, "z": 30},
            },
            "pores": {
                "seed_number_density_A3": 0.000125,
                "target_porosity": 0.10,
                "channel_fraction_by_count": channel_fraction,
                "channel_to_compact_mean_volume_ratio": 1.0,
            },
            "center_distribution": {
                "mode": "lattice_jitter",
                "lattice": "simple_cubic",
                "position_jitter": 0.0,
            },
            "orientation": {
                "distribution": orientation_distribution
                or {"family": "beta", "alpha": 2.0, "beta": 2.0},
                "azimuth": "uniform",
            },
            "compact": {
                "relative_volume": compact_relative_volume or {"family": "constant", "value": 1.0},
                "aspect_ratio": {"family": "constant", "value": compact_eta},
                "roughness": {"family": "constant", "value": 0.0},
            },
            "channel": {
                "relative_volume": {"family": "constant", "value": 1.0},
                "eta": {"family": "constant", "value": channel_eta},
                "tau": {"family": "constant", "value": channel_tau},
                "roughness": {"family": "constant", "value": channel_roughness},
            },
            "pore_material": {"pdb": "argon.pdb", "molecule_count": 1},
        }
    )


def _single_latent_plan(point_A: np.ndarray) -> CenterSeedPlan:
    return CenterSeedPlan(
        intended_points_A=np.asarray([point_A], dtype=float),
        target_rdf_xi=np.array([0.0]),
        target_rdf_values=np.array([1.0]),
        starting_loss=0.0,
        initialization_loss=0.0,
    )


def test_spherical_compact_sdf_matches_radius() -> None:
    unit = CompactUnit.sphere(
        unit_id="compact-0001",
        center_A=np.array([0.0, 0.0, 0.0]),
        radius_A=2.0,
    )

    values = unit.sdf(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
    )

    assert np.allclose(values, [-2.0, 0.0, 1.0], atol=1e-7)
    assert unit.sdf_lipschitz_bound == 1.0


def test_pore_geometry_exposes_finite_smooth_union_lipschitz_bound() -> None:
    sphere = CompactUnit.sphere(
        unit_id="compact-lipschitz",
        center_A=np.array([3.0, 3.0, 3.0]),
        radius_A=2.0,
    )
    channel = ChannelUnit.from_polyline(
        unit_id="channel-lipschitz",
        control_points_unwrapped_A=np.array([[0.0, 6.0, 6.0], [10.0, 6.0, 6.0]]),
        cross_radius_A=1.0,
        roughness=0.0,
    )
    geometry = PoreGeometry([sphere, channel], np.array([10.0, 10.0, 10.0]))

    assert channel.sdf_lipschitz_bound == 1.0
    assert np.isfinite(geometry.sdf_lipschitz_bound)
    assert geometry.sdf_lipschitz_bound == 1.0


def test_roughness_adds_to_unit_lipschitz_bound() -> None:
    smooth_channel = ChannelUnit.from_polyline(
        unit_id="roughness-bound",
        control_points_unwrapped_A=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        cross_radius_A=1.0,
        roughness=0.0,
    )
    rough_channel = ChannelUnit.from_polyline(
        unit_id="roughness-bound",
        control_points_unwrapped_A=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        cross_radius_A=1.0,
        roughness=0.2,
    )

    assert rough_channel.sdf_lipschitz_bound > smooth_channel.sdf_lipschitz_bound
    assert np.isfinite(rough_channel.sdf_lipschitz_bound)


def test_straight_channel_has_tau_one_and_continuous_negative_sdf() -> None:
    channel = ChannelUnit.from_polyline(
        unit_id="channel-0001",
        control_points_unwrapped_A=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        cross_radius_A=1.0,
        roughness=0.0,
    )

    sample = np.column_stack([np.linspace(0.0, 10.0, 21), np.zeros(21), np.zeros(21)])

    assert np.isclose(channel.tortuosity, 1.0)
    assert np.all(channel.sdf(sample) < 0.0)


def test_channel_anchor_is_unfolded_arclength_centroid() -> None:
    channel = ChannelUnit.from_polyline(
        unit_id="channel-0002",
        control_points_unwrapped_A=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        cross_radius_A=1.0,
        roughness=0.0,
    )

    assert np.allclose(channel.anchor_A, [5.0, 0.0, 0.0])


def test_pore_geometry_wraps_periodic_images_in_x_and_y_only() -> None:
    unit = CompactUnit.sphere(
        unit_id="compact-edge",
        center_A=np.array([0.5, 5.0, 5.0]),
        radius_A=1.0,
    )
    geometry = PoreGeometry([unit], np.array([10.0, 10.0, 10.0]))

    values = geometry.sdf(np.array([[9.75, 5.0, 5.0], [0.5, 5.0, 9.5]]))

    assert values[0] < 0.0
    assert values[1] > 0.0


def test_build_units_reports_realized_channel_anchor_not_latent_seed() -> None:
    latent_anchor_A = np.array([1.0, 2.0, 3.0])

    built = build_units(
        config=_geometry_config(channel_fraction=1.0),
        center_plan=_single_latent_plan(latent_anchor_A),
        rng=np.random.default_rng(11),
    )

    assert built.realized_anchors_A.shape == (1, 3)
    assert np.allclose(built.realized_anchors_A[0], built.units[0].anchor_A)
    assert not np.allclose(built.realized_anchors_A[0], latent_anchor_A)
    assert built.latent_to_realized_ids == {"latent-0000": "channel-0000"}


def test_build_units_realizes_sampled_channel_eta_and_tau() -> None:
    built = build_units(
        config=_geometry_config(channel_fraction=1.0, channel_eta=3.0, channel_tau=1.4),
        center_plan=_single_latent_plan(np.array([4.0, 5.0, 6.0])),
        rng=np.random.default_rng(31),
    )

    channel = built.units[0]

    assert np.isclose(channel.arc_length_A / (2.0 * channel.cross_radius_A), 3.0, atol=2e-3)
    assert np.isclose(channel.eta, 3.0, atol=2e-3)
    assert np.isclose(channel.tortuosity, 1.4, atol=2e-3)


def test_theta_zero_channel_orientation_aligns_with_x_axis() -> None:
    built = build_units(
        config=_geometry_config(
            channel_fraction=1.0,
            orientation_distribution={
                "family": "beta",
                "alpha": 1.0,
                "beta": 100_000.0,
            },
        ),
        center_plan=_single_latent_plan(np.array([0.0, 0.0, 0.0])),
        rng=np.random.default_rng(37),
    )

    channel = built.units[0]
    end_direction = (
        channel.centerline_samples_A[-1] - channel.centerline_samples_A[0]
    ) / channel.end_distance_A

    assert end_direction[0] > 0.99
    assert abs(end_direction[2]) < 0.01


def test_theta_zero_elongated_compact_major_axis_aligns_with_x_axis() -> None:
    built = build_units(
        config=_geometry_config(
            compact_eta=4.0,
            orientation_distribution={
                "family": "beta",
                "alpha": 1.0,
                "beta": 100_000.0,
            },
        ),
        center_plan=_single_latent_plan(np.array([5.0, 5.0, 5.0])),
        rng=np.random.default_rng(43),
    )

    compact = built.units[0]
    major_axis = compact.orientation.apply(np.eye(3)[int(np.argmax(compact.radii_A))])

    assert major_axis[0] > 0.99
    assert abs(major_axis[1]) < 0.01
    assert abs(major_axis[2]) < 0.01


def test_rough_channel_sdf_is_continuous_and_rotation_invariant() -> None:
    control_points = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 4.0, 0.0]])
    channel = ChannelUnit.from_polyline(
        unit_id="rough-channel",
        control_points_unwrapped_A=control_points,
        cross_radius_A=0.7,
        roughness=0.15,
    )
    rotation = Rotation.from_euler("z", 90.0, degrees=True)
    rotated_channel = ChannelUnit.from_polyline(
        unit_id="rough-channel",
        control_points_unwrapped_A=rotation.apply(control_points),
        cross_radius_A=0.7,
        roughness=0.15,
    )
    sample = np.array(
        [
            [3.95, 0.20, 0.30],
            [4.00, 0.25, 0.30],
            [4.05, 0.30, 0.30],
        ]
    )

    values = channel.sdf(sample)
    rotated_values = rotated_channel.sdf(rotation.apply(sample))

    assert np.max(np.abs(np.diff(values))) < 0.08
    assert np.allclose(values, rotated_values, atol=1e-7)


def test_channel_join_uses_smooth_minimum_across_segments() -> None:
    channel = ChannelUnit.from_polyline(
        unit_id="smooth-join",
        control_points_unwrapped_A=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        cross_radius_A=0.5,
        roughness=0.0,
    )

    value = channel.sdf(np.array([[1.0, 1.0, 0.0]]))[0]

    assert value < 0.499


def test_periodic_geometry_checks_images_spanning_multiple_boxes() -> None:
    channel = ChannelUnit.from_polyline(
        unit_id="long-unfolded",
        control_points_unwrapped_A=np.array([[25.5, 5.0, 5.0], [30.5, 5.0, 5.0]]),
        cross_radius_A=1.0,
        roughness=0.0,
    )
    geometry = PoreGeometry([channel], np.array([10.0, 10.0, 10.0]))

    assert geometry.sdf(np.array([[0.25, 5.0, 5.0]]))[0] < 0.0


def test_channel_record_schema_keeps_latent_and_realized_geometry_values() -> None:
    built = build_units(
        config=_geometry_config(channel_fraction=1.0, channel_eta=3.0, channel_tau=1.25),
        center_plan=_single_latent_plan(np.array([2.0, 3.0, 4.0])),
        rng=np.random.default_rng(41),
    )

    record = built.units[0].to_record()

    assert record["latent_parameters"]["eta"] == 3.0
    assert record["latent_parameters"]["tau"] == 1.25
    assert "theta_rad" in record["latent_parameters"]["orientation"]
    assert "phi_rad" in record["latent_parameters"]["orientation"]
    assert np.isclose(record["realized_geometry"]["eta"], built.units[0].eta)
    assert np.isclose(record["realized_geometry"]["tortuosity"], built.units[0].tortuosity)
    assert "theta_rad" in record["realized_geometry"]["orientation"]
    assert "quaternion_xyzw" in record["realized_geometry"]["orientation"]
    assert len(record["realized_geometry"]["segment_frame_quaternions_xyzw"]) >= 1


def test_pore_geometry_copies_target_box_without_aliasing_caller_array() -> None:
    caller_box = np.array([10.0, 20.0, 30.0])
    geometry = PoreGeometry([], caller_box)

    geometry.target_box_A[0] = 99.0

    assert np.array_equal(caller_box, np.array([10.0, 20.0, 30.0]))


def test_build_units_uses_supplied_rng_for_relative_volume_sampling() -> None:
    config = _geometry_config(
        compact_relative_volume={
            "family": "beta",
            "alpha": 1.0,
            "beta": 1.0,
            "lower": 1.0,
            "upper": 2.0,
        }
    )
    plan = CenterSeedPlan(
        intended_points_A=np.array([[5.0, 5.0, 5.0], [15.0, 15.0, 15.0]]),
        target_rdf_xi=np.array([0.0]),
        target_rdf_values=np.array([1.0]),
        starting_loss=0.0,
        initialization_loss=0.0,
    )

    first = build_units(config, plan, np.random.default_rng(101))
    second = build_units(config, plan, np.random.default_rng(202))

    first_radii = np.vstack([unit.radii_A for unit in first.units])
    second_radii = np.vstack([unit.radii_A for unit in second.units])
    assert not np.allclose(first_radii, second_radii)


def test_unit_records_separate_latent_parameters_from_realized_geometry() -> None:
    unit = CompactUnit.sphere(
        unit_id="compact-record",
        center_A=np.array([1.0, 2.0, 3.0]),
        radius_A=2.0,
    )

    record = unit.to_record()

    assert record["kind"] == "compact"
    assert record["latent_parameters"]["radius_A"] == 2.0
    assert record["realized_geometry"]["anchor_A"] == [1.0, 2.0, 3.0]


def test_built_unit_records_include_latent_target_volume() -> None:
    config = _geometry_config()
    built = build_units(
        config=config,
        center_plan=_single_latent_plan(np.array([5.0, 5.0, 5.0])),
        rng=np.random.default_rng(47),
    )

    record = built.units[0].to_record()

    assert record["latent_parameters"]["target_volume_A3"] == config.pores.target_porosity * (
        config.film.target_box_A.volume_A3
    )


def test_build_units_generates_multilobe_compact_schema_v2() -> None:
    built = build_units(
        config=_geometry_config(compact_eta=2.5),
        center_plan=_single_latent_plan(np.array([5.0, 5.0, 5.0])),
        rng=np.random.default_rng(501),
    )
    compact = built.units[0]
    record = compact.to_record()

    assert isinstance(compact, CompactUnit)
    assert compact.shape_model == "multilobe-v1"
    assert compact.shape_seed is not None
    assert compact.lobe_centers_local_A is not None
    assert compact.lobe_radii_A is not None
    assert 2 <= compact.lobe_centers_local_A.shape[0] <= 4
    assert np.isclose(compact.radii_A[2] / compact.radii_A[0], 2.5, rtol=0.05)
    assert record["schema_version"] == 2
    assert record["shape_model"] == "multilobe-v1"
    assert record["shape_seed"] == compact.shape_seed
    assert np.isclose(record["latent_parameters"]["eta"], 2.5)
    assert len(record["realized_geometry"]["lobe_centers_local_A"]) >= 2
    assert 0.50 <= record["realized_geometry"]["envelope_fill_fraction"] <= 0.85


def test_build_units_generates_variable_radius_multibend_channel_schema_v2() -> None:
    built = build_units(
        config=_geometry_config(channel_fraction=1.0, channel_eta=7.0, channel_tau=1.45),
        center_plan=_single_latent_plan(np.array([2.0, 3.0, 4.0])),
        rng=np.random.default_rng(503),
    )
    channel = built.units[0]
    record = channel.to_record()
    radii = channel.radius_profile_A

    assert isinstance(channel, ChannelUnit)
    assert channel.shape_model == "variable-radius-spline-v1"
    assert channel.shape_seed is not None
    assert channel.control_points_unwrapped_A.shape == (7, 3)
    assert channel.radius_profile_s is not None
    assert radii is not None
    assert radii.shape == (7,)
    assert 0.15 <= float(np.std(radii) / np.mean(radii)) <= 0.30
    assert channel.bend_count >= 2
    assert channel.nonplanarity > 0.0
    assert channel.minimum_self_clearance_A >= 0.0
    assert np.isclose(channel.eta, 7.0, rtol=0.01)
    assert np.isclose(channel.tortuosity, 1.45, rtol=0.01)
    assert record["schema_version"] == 2
    assert record["shape_model"] == "variable-radius-spline-v1"
    assert len(record["realized_geometry"]["radius_profile_A"]) == 7


def test_complex_shapes_are_reproducible_for_same_build_rng() -> None:
    config = _geometry_config(channel_fraction=0.5, channel_eta=5.0, channel_tau=1.3)
    plan = CenterSeedPlan(
        intended_points_A=np.array([[4.0, 5.0, 6.0], [14.0, 15.0, 16.0]]),
        target_rdf_xi=np.array([0.0]),
        target_rdf_values=np.array([1.0]),
        starting_loss=0.0,
        initialization_loss=0.0,
    )
    first = build_units(config, plan, np.random.default_rng(509))
    second = build_units(config, plan, np.random.default_rng(509))

    assert [unit.to_record() for unit in first.units] == [unit.to_record() for unit in second.units]


def test_manual_simple_unit_constructors_keep_schema_v1() -> None:
    compact = CompactUnit.sphere("manual-sphere", np.zeros(3), 1.0)
    channel = ChannelUnit.from_polyline(
        "manual-channel",
        np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        0.5,
        0.0,
    )

    assert compact.to_record()["schema_version"] == 1
    assert channel.to_record()["schema_version"] == 1
    assert getattr(compact, "shape_model", None) is None
    assert getattr(channel, "shape_model", None) is None


def test_scale_unit_preserves_complex_shape_similarity() -> None:
    from porous_film.geometry import scale_unit

    config = _geometry_config(channel_fraction=0.5, channel_eta=6.0, channel_tau=1.4)
    plan = CenterSeedPlan(
        intended_points_A=np.array([[3.0, 4.0, 5.0], [13.0, 14.0, 15.0]]),
        target_rdf_xi=np.array([0.0]),
        target_rdf_values=np.array([1.0]),
        starting_loss=0.0,
        initialization_loss=0.0,
    )
    built = build_units(config, plan, np.random.default_rng(607))
    factor = 1.75

    for unit in built.units:
        scaled = scale_unit(unit, factor)
        if isinstance(unit, CompactUnit):
            assert scaled.lobe_centers_local_A is not None
            assert scaled.lobe_radii_A is not None
            assert np.allclose(scaled.lobe_centers_local_A, unit.lobe_centers_local_A * factor)
            assert np.allclose(scaled.lobe_radii_A, unit.lobe_radii_A * factor)
            assert np.isclose(scaled.smooth_length_A, unit.smooth_length_A * factor)
            assert np.allclose(scaled.radii_A, unit.radii_A * factor)
        else:
            assert isinstance(unit, ChannelUnit)
            assert scaled.radius_profile_A is not None
            assert np.allclose(scaled.radius_profile_A, unit.radius_profile_A * factor)
            assert np.isclose(scaled.cross_radius_A, unit.cross_radius_A * factor)
            assert np.isclose(scaled.eta, unit.eta)
            assert np.isclose(scaled.tortuosity, unit.tortuosity)


def test_scale_unit_does_not_rewrite_formal_target_provenance() -> None:
    config = _geometry_v3_config(seed_count=1)
    built = build_units(
        config,
        _single_latent_plan(np.array([50.0, 50.0, 50.0])),
        np.random.default_rng(303),
    )
    channel = built.units[0]

    scaled = scale_built_geometry(built, 1.5).units[0]

    assert isinstance(channel, ChannelUnit)
    assert isinstance(scaled, ChannelUnit)
    assert scaled.latent_equivalent_diameter_A == channel.latent_equivalent_diameter_A
    assert scaled.latent_target_volume_A3 == channel.latent_target_volume_A3
    assert not np.isclose(scaled.cross_radius_A, channel.cross_radius_A)


def test_variable_radius_channel_uses_internal_profile_nodes_along_straight_centerline() -> None:
    channel = ChannelUnit.from_polyline(
        unit_id="variable-straight",
        control_points_unwrapped_A=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        cross_radius_A=1.0,
        roughness=0.0,
        shape_model="variable-radius-polyline-v1",
        shape_seed=17,
        radius_profile_s=np.array([0.0, 0.5, 1.0]),
        radius_profile_A=np.array([0.5, 1.5, 0.5]),
    )

    values = channel.sdf_bruteforce(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [5.0, 1.0, 0.0],
                [10.0, 1.0, 0.0],
            ]
        )
    )

    assert values[0] > 0.45
    assert values[1] < -0.45
    assert values[2] > 0.45


def test_channel_aabb_culling_matches_bruteforce_within_one_e_minus_nine_A() -> None:
    built = build_units(
        config=_geometry_config(
            channel_fraction=1.0,
            channel_eta=8.0,
            channel_tau=1.5,
            channel_roughness=0.2,
        ),
        center_plan=_single_latent_plan(np.array([2.0, 3.0, 4.0])),
        rng=np.random.default_rng(701),
    )
    channel = built.units[0]
    rng = np.random.default_rng(702)
    points = rng.uniform(-25.0, 35.0, size=(2500, 3))

    optimized = channel.sdf(points)
    reference = channel.sdf_bruteforce(points)

    assert np.max(np.abs(optimized - reference)) <= 1.0e-9


def test_variable_radius_channel_sdf_is_rotation_invariant() -> None:
    controls = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.8, 0.3],
            [4.0, -0.6, 0.9],
            [6.0, 0.0, 0.0],
        ]
    )
    profile_s = np.array([0.0, 0.5, 1.0])
    profile_A = np.array([0.7, 1.2, 0.8])
    channel = ChannelUnit.from_polyline(
        "variable-rotation",
        controls,
        0.9,
        0.0,
        shape_model="variable-radius-polyline-v1",
        shape_seed=23,
        radius_profile_s=profile_s,
        radius_profile_A=profile_A,
    )
    rotation = Rotation.from_euler("xyz", [25.0, -15.0, 70.0], degrees=True)
    rotated = ChannelUnit.from_polyline(
        "variable-rotation",
        rotation.apply(controls),
        0.9,
        0.0,
        shape_model="variable-radius-polyline-v1",
        shape_seed=23,
        radius_profile_s=profile_s,
        radius_profile_A=profile_A,
    )
    points = np.array([[1.0, 0.4, 0.2], [3.0, -0.1, 0.8], [5.5, 0.2, -0.3]])

    assert np.allclose(channel.sdf(points), rotated.sdf(rotation.apply(points)), atol=1.0e-8)


def test_periodic_compact_sdf_uses_single_minimum_image_evaluation(
    monkeypatch,
) -> None:
    from scipy.special import logsumexp

    unit = CompactUnit.sphere(
        "periodic-counted",
        np.array([1.0, 9.0, 5.0]),
        1.5,
    )
    geometry = PoreGeometry([unit], np.array([20.0, 20.0, 10.0]))
    points = np.array(
        [
            [19.8, 9.0, 5.0],
            [0.2, 9.0, 5.0],
            [10.0, 19.8, 5.0],
            [10.0, 0.2, 5.0],
        ]
    )
    original = CompactUnit.sdf
    call_count = 0

    def counted(self, values):
        nonlocal call_count
        call_count += 1
        return original(self, values)

    monkeypatch.setattr(CompactUnit, "sdf", counted)
    optimized = geometry.sdf(points)
    brute_fields = []
    for x_shift in (-20.0, 0.0, 20.0):
        for y_shift in (-20.0, 0.0, 20.0):
            brute_fields.append(original(unit, points - np.array([x_shift, y_shift, 0.0])))
    expected = -logsumexp(-32.0 * np.vstack(brute_fields), axis=0) / 32.0

    assert call_count == 1
    assert np.allclose(optimized, expected, atol=1.0e-9)


def test_variable_radius_channel_runtime_sdf_volume_matches_target() -> None:
    from scipy.stats import qmc

    from porous_film.geometry.complex_shapes import generate_variable_radius_channel_profile

    target_volume = 12_000.0
    profile = generate_variable_radius_channel_profile(target_volume, 2.0, 1.0, 43)
    channel = ChannelUnit.from_polyline(
        "low-eta-volume",
        profile.control_points_local_A,
        profile.equivalent_radius_A,
        0.0,
        shape_model="variable-radius-spline-v1",
        shape_seed=43,
        radius_profile_s=profile.radius_profile_s,
        radius_profile_A=profile.radius_profile_A,
        bend_count=profile.bend_count,
        nonplanarity=profile.nonplanarity,
        minimum_self_clearance_A=profile.minimum_self_clearance_A,
    )
    support = float(np.max(profile.radius_profile_A) + 1.0)
    lower = np.min(channel.centerline_samples_A, axis=0) - support
    upper = np.max(channel.centerline_samples_A, axis=0) + support
    samples = qmc.Sobol(d=3, scramble=False).random_base2(19)
    points = lower + (upper - lower) * samples
    volume = float(np.mean(channel.sdf(points) < 0.0) * np.prod(upper - lower))

    assert np.isclose(volume, target_volume, rtol=0.03)


def test_compact_record_preserves_sampled_eta_separately_from_realized_envelope() -> None:
    from scipy.spatial.transform import Rotation

    from porous_film.geometry.complex_shapes import generate_multilobe_profile

    profile = generate_multilobe_profile(2_000.0, 2.0, 987)
    compact = CompactUnit(
        unit_id="compact-latent-eta",
        center_A=np.zeros(3),
        radii_A=profile.envelope_radii_A,
        orientation=Rotation.identity(),
        exponent=2.0,
        roughness=0.0,
        latent_eta=2.5,
        shape_model="multilobe-v1",
        shape_seed=987,
        lobe_centers_local_A=profile.lobe_centers_local_A,
        lobe_radii_A=profile.lobe_radii_A,
        smooth_length_A=profile.smooth_length_A,
        envelope_fill_fraction=profile.envelope_fill_fraction,
        centroid_offset_A=profile.centroid_offset_A,
        lobes_connected=profile.connected,
    )

    record = compact.to_record()

    assert record["latent_parameters"]["eta"] == 2.5
    assert np.isclose(
        record["realized_geometry"]["radii_A"][2] / record["realized_geometry"]["radii_A"][0], 2.0
    )


def _geometry_v3_config(*, seed_count: int = 10) -> GeneratorConfig:
    box_length = 100.0
    density = seed_count / box_length**3
    return GeneratorConfig.model_validate(
        {
            "schema_version": 3,
            "task": {"name": "geometry-v3", "random_seed": 23},
            "film": {
                "target_box_A": {
                    "x": box_length,
                    "y": box_length,
                    "z": box_length,
                }
            },
            "formal_targets": {
                "position_quantity": {
                    "center_distance_xy": {
                        "components": [
                            {
                                "kind": "exclusion",
                                "amplitude": 0.5,
                                "center_A": 0.0,
                                "width_A": 8.0,
                            }
                        ]
                    }
                },
                "shape": {
                    "equivalent_diameter_A": {
                        "family": "constant",
                        "value": 8.0,
                    },
                    "orientation": {
                        "model": "paired_projected_planes",
                        "components": [
                            {
                                "weight": 0.6,
                                "theta_xz_deg": {
                                    "family": "beta",
                                    "alpha": 2.0,
                                    "beta": 2.0,
                                    "lower": 60.0,
                                    "upper": 80.0,
                                },
                                "theta_xy_deg": {
                                    "family": "beta",
                                    "alpha": 2.0,
                                    "beta": 2.0,
                                    "lower": 0.0,
                                    "upper": 20.0,
                                },
                            },
                            {
                                "weight": 0.4,
                                "theta_xz_deg": {
                                    "family": "beta",
                                    "alpha": 2.0,
                                    "beta": 2.0,
                                    "lower": 10.0,
                                    "upper": 30.0,
                                },
                                "theta_xy_deg": {
                                    "family": "beta",
                                    "alpha": 2.0,
                                    "beta": 2.0,
                                    "lower": 60.0,
                                    "upper": 80.0,
                                },
                            },
                        ],
                    },
                    "compact_aspect_ratio": {"family": "constant", "value": 1.5},
                    "channel_aspect_ratio": {"family": "constant", "value": 4.0},
                    "channel_tortuosity": {"family": "constant", "value": 1.2},
                    "curvature_fluctuation": {"family": "constant", "value": 0.4},
                },
                "proportion": {"porosity": 0.02},
            },
            "generation_controls": {
                "seed_number_density_A3": density,
                "channel_fraction_by_count": 1.0,
                "channel_to_compact_mean_volume_ratio": 1.0,
            },
            "measurement": {
                "z_slice_spacing_A": 2.0,
                "center_min_separation_A": 4.0,
                "center_tracking_max_displacement_A": 8.0,
                "centerline_sample_spacing_A": 2.0,
                "cross_section_spacing_A": 2.0,
                "boundary_resample_spacing_A": 0.5,
                "curvature_smoothing_length_A": 1.0,
                "branch_exclusion_length_A": 4.0,
                "surface_exclusion_length_A": 2.0,
                "orientation_projection_min_fraction": 0.05,
            },
        }
    )


def _multi_latent_plan(count: int) -> CenterSeedPlan:
    x = np.linspace(5.0, 95.0, count)
    points = np.column_stack([x, np.full(count, 25.0), np.full(count, 25.0)])
    return CenterSeedPlan(
        intended_points_A=points,
        target_rdf_xi=np.array([0.0]),
        target_rdf_values=np.array([1.0]),
        starting_loss=0.0,
        initialization_loss=0.0,
    )


def test_schema_v3_paired_orientation_sampling_is_deterministic_and_grouped() -> None:
    config = _geometry_v3_config(seed_count=10)
    plan = _multi_latent_plan(10)

    first = build_units(config, plan, np.random.default_rng(101))
    second = build_units(config, plan, np.random.default_rng(101))
    records = [unit.to_record()["latent_parameters"]["orientation"] for unit in first.units]
    repeated = [unit.to_record()["latent_parameters"]["orientation"] for unit in second.units]

    assert records == repeated
    assert [record["paired_component_index"] for record in records].count(0) == 6
    assert [record["paired_component_index"] for record in records].count(1) == 4

    for unit, record in zip(first.units, records, strict=True):
        direction = unit.orientation.apply(np.array([1.0, 0.0, 0.0]))
        theta_xz = np.degrees(np.arctan2(abs(direction[2]), abs(direction[0])))
        theta_xy = np.degrees(np.arctan2(abs(direction[1]), abs(direction[0])))
        assert np.isclose(theta_xz, record["theta_xz_deg"])
        assert np.isclose(theta_xy, record["theta_xy_deg"])
        if record["paired_component_index"] == 0:
            assert 60.0 <= theta_xz <= 80.0
            assert 0.0 <= theta_xy <= 20.0
        else:
            assert 10.0 <= theta_xz <= 30.0
            assert 60.0 <= theta_xy <= 80.0


def test_schema_v3_channel_uses_absolute_equivalent_diameter_target() -> None:
    config = _geometry_v3_config(seed_count=1)
    built = build_units(
        config,
        _single_latent_plan(np.array([50.0, 50.0, 50.0])),
        np.random.default_rng(303),
    )
    channel = built.units[0]
    dense_radii = channel._radius_at_normalized_arclength(np.linspace(0.0, 1.0, 4097))
    measured_diameter = 2.0 * np.sqrt(np.trapezoid(dense_radii**2, np.linspace(0.0, 1.0, 4097)))
    record = channel.to_record()

    assert np.isclose(measured_diameter, 8.0, rtol=1e-3)
    assert np.ptp(dense_radii) <= 1.0e-12
    assert np.isclose(channel.arc_length_A / measured_diameter, 4.0, rtol=1e-2)
    assert record["latent_parameters"]["equivalent_diameter_A"] == 8.0
    assert record["latent_parameters"]["curvature_fluctuation_target"] == 0.4
    assert record["latent_parameters"]["target_volume_A3"] is None


def test_unrestricted_schema_v3_builds_without_through_only_targets() -> None:
    raw = _geometry_v3_config(seed_count=1).model_dump(mode="python")
    raw["formal_targets"]["position_quantity"].pop("center_distance_xy")
    for name in (
        "equivalent_diameter_A",
        "orientation",
        "channel_aspect_ratio",
        "channel_tortuosity",
        "curvature_fluctuation",
    ):
        raw["formal_targets"]["shape"].pop(name)
    for compatibility_key in (
        "pores",
        "center_distribution",
        "compact",
        "channel",
        "orientation",
    ):
        raw.pop(compatibility_key)
    raw["pore_constraints"] = {"z_connectivity": "unrestricted"}
    config = GeneratorConfig.model_validate(raw)

    built = build_units(
        config,
        _single_latent_plan(np.array([50.0, 50.0, 50.0])),
        np.random.default_rng(307),
    )

    channel = built.units[0]
    record = channel.to_record()["latent_parameters"]
    assert isinstance(channel, ChannelUnit)
    assert record["equivalent_diameter_A"] is None
    assert record["curvature_fluctuation_target"] is None
    assert record["orientation"]["paired_component_index"] is None


def test_all_components_mode_places_generated_channel_through_finite_z() -> None:
    raw = _geometry_v3_config(seed_count=1).model_dump(mode="python")
    raw["film"]["target_box_A"]["z"] = 20.0
    raw["film"]["packing_box_A"]["z"] = 20.0
    raw["generation_controls"]["seed_number_density_A3"] = 1.0 / (100.0 * 100.0 * 20.0)
    raw["formal_targets"]["shape"]["compact_aspect_ratio"] = None
    for name in (
        "equivalent_diameter_A",
        "orientation",
        "channel_aspect_ratio",
        "channel_tortuosity",
        "curvature_fluctuation",
    ):
        raw["formal_targets"]["shape"][name] = None
    raw["pore_constraints"] = {"z_connectivity": "all_components"}
    config = GeneratorConfig.model_validate(raw)
    plan = CenterSeedPlan(
        intended_points_A=np.array([[50.0, 50.0, 3.0]]),
        target_rdf_xi=np.array([0.0]),
        target_rdf_values=np.array([1.0]),
        starting_loss=0.0,
        initialization_loss=0.0,
    )

    built = build_units(config, plan, np.random.default_rng(303))
    grid = voxelize_geometry(built.geometry, np.array([100.0, 100.0, 20.0]), 2.0)

    assert pore_z_connectivity_summary(grid.pore_mask).all_components_through


def test_all_components_couples_formal_channel_metrics_to_finite_z() -> None:
    raw = _geometry_v3_config(seed_count=12).model_dump(mode="python")
    raw["film"] = {"target_box_A": {"x": 1000.0, "y": 1000.0, "z": 800.0}}
    raw["generation_controls"]["seed_number_density_A3"] = 12.0 / (
        1000.0 * 1000.0 * 800.0
    )
    raw["generation_controls"]["channel_fraction_by_count"] = 1.0
    raw["formal_targets"]["shape"]["compact_aspect_ratio"] = None
    raw["formal_targets"]["shape"]["equivalent_diameter_A"] = {
        "family": "beta",
        "alpha": 2.5,
        "beta": 3.5,
        "lower": 250.0,
        "upper": 360.0,
    }
    raw["formal_targets"]["shape"]["channel_aspect_ratio"] = {
        "family": "beta",
        "alpha": 2.5,
        "beta": 3.0,
        "lower": 2.5,
        "upper": 3.5,
    }
    raw["formal_targets"]["shape"]["channel_tortuosity"] = {
        "family": "beta",
        "alpha": 2.5,
        "beta": 3.5,
        "lower": 1.04,
        "upper": 1.15,
    }
    raw["formal_targets"]["shape"]["orientation"] = {
        "model": "paired_projected_planes",
        "components": [
            {
                "weight": 1.0,
                "theta_xz_deg": {
                    "family": "beta",
                    "alpha": 4.0,
                    "beta": 2.0,
                    "lower": 82.0,
                    "upper": 89.5,
                },
                "theta_xy_deg": {
                    "family": "beta",
                    "alpha": 2.0,
                    "beta": 2.0,
                    "lower": 0.0,
                    "upper": 30.0,
                },
            }
        ],
    }
    raw["pore_constraints"] = {"z_connectivity": "all_components"}
    for compatibility_key in ("pores", "center_distribution", "compact", "channel", "orientation"):
        raw.pop(compatibility_key)
    config = GeneratorConfig.model_validate(raw)
    points = np.column_stack(
        [
            np.linspace(100.0, 900.0, 12),
            np.full(12, 500.0),
            np.full(12, 400.0),
        ]
    )
    plan = CenterSeedPlan(
        intended_points_A=points,
        target_rdf_xi=np.array([0.0]),
        target_rdf_values=np.array([1.0]),
        starting_loss=0.0,
        initialization_loss=0.0,
    )

    built = build_units(config, plan, np.random.default_rng(311))

    for channel in built.units:
        assert isinstance(channel, ChannelUnit)
        assert np.isclose(np.ptp(channel.centerline_samples_A[:, 2]), 800.0, rtol=1.0e-6)
        assert 2.5 <= float(channel.latent_eta) <= 3.5
        assert np.isclose(channel.eta, float(channel.latent_eta), rtol=0.01)
        assert np.isclose(channel.tortuosity, float(channel.latent_tau), rtol=0.01)


def test_all_components_without_orientation_target_uses_z_spanning_default() -> None:
    raw = _geometry_v3_config(seed_count=1).model_dump(mode="python")
    raw["film"]["target_box_A"]["z"] = 20.0
    raw["film"]["packing_box_A"]["z"] = 20.0
    raw["generation_controls"]["seed_number_density_A3"] = 1.0 / (100.0 * 100.0 * 20.0)
    raw["generation_controls"]["channel_fraction_by_count"] = 1.0
    raw["formal_targets"]["position_quantity"]["center_distance_xy"] = None
    for name in (
        "equivalent_diameter_A",
        "orientation",
        "compact_aspect_ratio",
        "channel_aspect_ratio",
        "channel_tortuosity",
        "curvature_fluctuation",
    ):
        raw["formal_targets"]["shape"][name] = None
    for compatibility_key in (
        "pores",
        "center_distribution",
        "compact",
        "channel",
        "orientation",
    ):
        raw.pop(compatibility_key)
    raw["pore_constraints"] = {"z_connectivity": "all_components"}
    config = GeneratorConfig.model_validate(raw)
    plan = CenterSeedPlan(
        intended_points_A=np.array([[50.0, 50.0, 3.0]]),
        target_rdf_xi=np.array([0.0]),
        target_rdf_values=np.array([1.0]),
        starting_loss=0.0,
        initialization_loss=0.0,
    )

    built = build_units(config, plan, np.random.default_rng(311))
    channel = built.units[0]

    assert isinstance(channel, ChannelUnit)
    assert np.ptp(channel.centerline_samples_A[:, 2]) >= 20.0


def test_all_components_lengthens_channel_when_eta_is_not_a_formal_target() -> None:
    raw = _geometry_v3_config(seed_count=1).model_dump(mode="python")
    raw["film"]["target_box_A"]["z"] = 100.0
    raw["film"]["packing_box_A"]["z"] = 100.0
    raw["generation_controls"]["seed_number_density_A3"] = 1.0 / (100.0**3)
    raw["formal_targets"]["shape"]["compact_aspect_ratio"] = None
    raw["formal_targets"]["shape"]["channel_aspect_ratio"] = None
    raw["pore_constraints"] = {"z_connectivity": "all_components"}
    config = GeneratorConfig.model_validate(raw)
    plan = CenterSeedPlan(
        intended_points_A=np.array([[50.0, 50.0, 50.0]]),
        target_rdf_xi=np.array([0.0]),
        target_rdf_values=np.array([1.0]),
        starting_loss=0.0,
        initialization_loss=0.0,
    )

    built = build_units(config, plan, np.random.default_rng(313))
    channel = built.units[0]
    dense_s = np.linspace(0.0, 1.0, 4097)
    radii = channel._radius_at_normalized_arclength(dense_s)
    equivalent_diameter = 2.0 * np.sqrt(np.trapezoid(radii**2, dense_s))

    assert isinstance(channel, ChannelUnit)
    assert np.ptp(channel.centerline_samples_A[:, 2]) >= 100.0
    assert np.isclose(equivalent_diameter, 8.0, rtol=1.0e-3)
    assert np.isclose(channel.tortuosity, 1.2, rtol=1.0e-2)
    assert channel.eta > 4.0


def test_z_through_channel_is_repositioned_after_global_porosity_scaling() -> None:
    channel = ChannelUnit.from_polyline(
        "channel-through",
        np.array([[10.0, 10.0, -2.5], [35.0, 10.0, 17.5], [10.0, 10.0, 22.5]]),
        2.0,
        0.0,
    )
    built = BuiltGeometry(
        geometry=PoreGeometry([channel], np.array([40.0, 40.0, 20.0])),
        units=[channel],
        realized_anchors_A=channel.anchor_A[np.newaxis, :],
        latent_to_realized_ids={"latent-0000": channel.unit_id},
    )
    minimum_scale = minimum_scale_for_channels_through_z(built)

    scaled = scale_built_geometry(
        built,
        minimum_scale * (1.0 + 1.0e-10),
        require_channels_through_z=True,
    )
    scaled_channel = scaled.units[0]

    assert isinstance(scaled_channel, ChannelUnit)
    assert np.min(scaled_channel.centerline_samples_A[:, 2]) <= 0.0
    assert np.max(scaled_channel.centerline_samples_A[:, 2]) >= 20.0


def test_periodic_channel_footprint_relaxation_separates_overlapping_channels() -> None:
    first = ChannelUnit.from_polyline(
        "channel-a",
        np.array([[5.0, 5.0, -1.0], [5.0, 5.0, 21.0]]),
        2.0,
        0.0,
    )
    second = ChannelUnit.from_polyline(
        "channel-b",
        np.array([[6.0, 5.0, -1.0], [6.0, 5.0, 21.0]]),
        2.0,
        0.0,
    )
    built = BuiltGeometry(
        geometry=PoreGeometry([first, second], np.array([20.0, 20.0, 20.0])),
        units=[first, second],
        realized_anchors_A=np.vstack([first.anchor_A, second.anchor_A]),
        latent_to_realized_ids={"latent-0000": first.unit_id, "latent-0001": second.unit_id},
    )

    separated = separate_channel_footprints_xy(built)
    box_xy = np.array([20.0, 20.0])
    delta = separated.units[1].anchor_A[:2] - separated.units[0].anchor_A[:2]
    delta -= box_xy * np.round(delta / box_xy)

    assert np.linalg.norm(delta) >= 4.0 - 1.0e-6
    assert np.allclose(
        separated.units[1].control_points_unwrapped_A - separated.units[1].anchor_A,
        second.control_points_unwrapped_A - second.anchor_A,
    )


def test_channel_lateral_deviation_adjustment_preserves_z_and_straightens_xy() -> None:
    channel = ChannelUnit.from_polyline(
        "channel-bent",
        np.array(
            [
                [2.0, 3.0, -5.0],
                [6.0, 9.0, 5.0],
                [8.0, 5.0, 15.0],
                [12.0, 11.0, 25.0],
            ]
        ),
        2.0,
        0.0,
        latent_tau=1.2,
    )
    built = BuiltGeometry(
        geometry=PoreGeometry([channel], np.array([40.0, 40.0, 20.0])),
        units=[channel],
        realized_anchors_A=channel.anchor_A[np.newaxis, :],
        latent_to_realized_ids={"latent-0000": channel.unit_id},
    )

    adjusted = adjust_channel_lateral_deviations_xy(built, {channel.unit_id: 0.0})
    result = adjusted.units[0]

    assert isinstance(result, ChannelUnit)
    np.testing.assert_allclose(
        result.control_points_unwrapped_A[:, 2],
        channel.control_points_unwrapped_A[:, 2],
    )
    relative_z = (
        result.control_points_unwrapped_A[:, 2] - result.control_points_unwrapped_A[0, 2]
    ) / (result.control_points_unwrapped_A[-1, 2] - result.control_points_unwrapped_A[0, 2])
    expected_xy = (1.0 - relative_z[:, np.newaxis]) * result.control_points_unwrapped_A[
        0, :2
    ] + relative_z[:, np.newaxis] * result.control_points_unwrapped_A[-1, :2]
    np.testing.assert_allclose(result.control_points_unwrapped_A[:, :2], expected_xy)
    np.testing.assert_allclose(result.anchor_A, channel.anchor_A)
    assert result.latent_tau == channel.latent_tau
