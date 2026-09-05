from __future__ import annotations

import numpy as np
from scipy import stats

from porous_film.centers import CenterSeedPlan
from porous_film.config import GeneratorConfig
from porous_film.geometry import BuiltGeometry, ChannelUnit, CompactUnit, PoreGeometry
from porous_film.metrics import (
    audit_target_distributions,
    compare_samples_to_distribution,
    measure_final_geometry,
)
from porous_film.metrics.audit import (
    _compare_paired_orientation_pairs,
    _grid_points,
    _local_thickness_stability_result,
    _porosity_tolerance,
    _primary_targets_passed,
    _unit_occupancy_masks,
)
from porous_film.metrics.local_thickness import ThicknessStabilityResult
from porous_film.voxel import PhaseGrid, voxelize_geometry


def test_audit_result_exposes_every_required_metric() -> None:
    from porous_film.metrics import AuditResult

    fields = set(AuditResult.__dataclass_fields__)
    required = {
        "theta_result",
        "compact_eta_result",
        "channel_eta_result",
        "roughness_result",
        "tau_result",
        "compact_relative_volume_result",
        "channel_relative_volume_result",
        "channel_fraction_error",
        "realized_mean_volume_ratio",
        "mean_volume_ratio_relative_error",
        "unit_volume_summary",
        "mixture_weight_errors",
        "overlap_fraction",
        "connected_pore_domains",
        "largest_pore_fraction",
        "z_lower_opening_fraction",
        "z_upper_opening_fraction",
        "minimum_cross_section_index",
        "local_thickness_stability_result",
    }

    assert required <= fields


def test_distribution_audit_rejects_shifted_constant_samples() -> None:
    result = compare_samples_to_distribution(
        samples=np.array([2.0, 2.0, 2.0, 2.0]),
        target={"family": "constant", "value": 1.0},
        ks_limit=0.05,
        normalized_wasserstein_limit=0.03,
    )

    assert not result.passed
    assert result.normalized_wasserstein > 0.03


def test_audit_passed_gates_scalar_errors_and_required_missing_samples() -> None:
    config = _audit_config(seed_density=0.002, target_porosity=0.10, channel_fraction=1.0)
    grid = _phase_grid(config, np.zeros((10, 10, 10), dtype=bool))
    built = BuiltGeometry(
        geometry=PoreGeometry([], np.array([10.0, 10.0, 10.0])),
        units=[],
        realized_anchors_A=np.empty((0, 3), dtype=float),
        latent_to_realized_ids={},
    )

    result = audit_target_distributions(config, built, _center_plan([]), grid)

    assert not result.passed
    assert any("porosity" in warning for warning in result.warnings)
    assert any("seed_count" in warning for warning in result.warnings)
    assert any("theta" in warning for warning in result.warnings)
    assert any("channel_eta" in warning for warning in result.warnings)
    assert any("tau" in warning for warning in result.warnings)


def test_audit_passed_gates_channel_fraction_rdf_and_mixture_weight_errors() -> None:
    config = _audit_config(
        seed_density=0.002,
        target_porosity=0.10,
        channel_fraction=0.0,
        compact_eta={
            "family": "mixture",
            "components": [
                {"weight": 0.5, "family": "constant", "value": 1.0},
                {"weight": 0.5, "family": "constant", "value": 3.0},
            ],
        },
    )
    pore_mask = np.zeros((10, 10, 10), dtype=bool)
    pore_mask.ravel()[:100] = True
    units = [
        CompactUnit(
            unit_id="compact-0000",
            center_A=np.array([1.0, 1.0, 1.0]),
            radii_A=np.array([1.0, 1.0, 1.0]),
            orientation=_identity_rotation(),
            exponent=2.0,
            roughness=0.0,
            latent_theta_rad=0.0,
            latent_phi_rad=0.0,
        ),
        CompactUnit(
            unit_id="compact-0001",
            center_A=np.array([8.0, 8.0, 8.0]),
            radii_A=np.array([1.0, 1.0, 1.0]),
            orientation=_identity_rotation(),
            exponent=2.0,
            roughness=0.0,
            latent_theta_rad=0.0,
            latent_phi_rad=0.0,
        ),
    ]
    built = _built(config, units)
    plan = CenterSeedPlan(
        intended_points_A=built.realized_anchors_A,
        target_rdf_xi=np.array([0.25, 0.75]),
        target_rdf_values=np.array([99.0, 99.0]),
        starting_loss=0.0,
        initialization_loss=0.0,
    )

    result = audit_target_distributions(config, built, plan, _phase_grid(config, pore_mask))

    assert not result.passed
    assert any("compact_eta mixture" in warning for warning in result.warnings)
    assert any("rdf" in warning for warning in result.warnings)


def test_audit_porosity_tolerance_uses_the_scale_solver_floor() -> None:
    config = _audit_config(
        box=(100.0, 100.0, 100.0),
        seed_density=1.0e-6,
        target_porosity=0.10,
        channel_fraction=0.0,
    )
    grid = _phase_grid(config, np.zeros((100, 100, 100), dtype=bool))

    assert np.isclose(_porosity_tolerance(grid), 0.01)


def test_small_realized_mixture_weight_error_is_not_fatal() -> None:
    config = _audit_config(
        seed_density=0.05,
        target_porosity=0.10,
        channel_fraction=0.0,
    )
    pore_mask = np.zeros((10, 10, 10), dtype=bool)
    pore_mask.ravel()[:100] = True

    assert _primary_targets_passed(
        warnings=[],
        scalar_errors={"porosity_absolute": 0.0, "seed_count_absolute": 0.0},
        grid=_phase_grid(config, pore_mask),
        channel_fraction_error=0.0,
        expected_seed_count=50,
        mixture_weight_errors={
            "compact_relative_volume": {"component_0": 0.04, "component_1": -0.04}
        },
        rdf_result={"weighted_loss": 0.0},
        local_thickness_stability_result=ThicknessStabilityResult(
            passed=True,
            max_quantile_error_A=0.0,
            mean_error_A=0.0,
            histogram_l1=0.0,
            tolerance_A=2.0,
            warning=None,
        ),
        mean_volume_ratio_relative_error=None,
    )


def test_primary_mixture_count_tolerance_uses_each_mixture_sample_count() -> None:
    config = _audit_config(
        seed_density=0.10,
        target_porosity=0.10,
        channel_fraction=0.0,
    )
    pore_mask = np.zeros((10, 10, 10), dtype=bool)
    pore_mask.ravel()[:100] = True

    assert _primary_targets_passed(
        warnings=[],
        scalar_errors={"porosity_absolute": 0.0, "seed_count_absolute": 0.0},
        grid=_phase_grid(config, pore_mask),
        channel_fraction_error=0.0,
        expected_seed_count=100,
        mixture_weight_errors={
            "compact_relative_volume": {"component_0": 0.10, "component_1": -0.10}
        },
        mixture_sample_counts={"compact_relative_volume": 10},
        rdf_result={"weighted_loss": 0.0},
        local_thickness_stability_result=ThicknessStabilityResult(
            passed=True,
            max_quantile_error_A=0.0,
            mean_error_A=0.0,
            histogram_l1=0.0,
            tolerance_A=2.0,
            warning=None,
        ),
        mean_volume_ratio_relative_error=None,
    )


def test_audit_passed_gates_each_failed_distribution_target() -> None:
    config = _audit_config(seed_density=0.002, target_porosity=0.10, channel_fraction=0.5)
    pore_mask = np.zeros((10, 10, 10), dtype=bool)
    pore_mask.ravel()[:100] = True
    compact = CompactUnit(
        unit_id="compact-0000",
        center_A=np.array([1.0, 1.0, 1.0]),
        radii_A=np.array([1.0, 1.0, 9.0]),
        orientation=_identity_rotation(),
        exponent=2.0,
        roughness=5.0,
        latent_theta_rad=np.pi,
        latent_phi_rad=0.0,
    )
    channel = ChannelUnit.from_polyline(
        unit_id="channel-0000",
        control_points_unwrapped_A=np.array([[4.0, 4.0, 4.0], [7.0, 8.0, 4.0], [9.0, 4.0, 4.0]]),
        cross_radius_A=1.0,
        roughness=5.0,
        latent_theta_rad=np.pi,
        latent_phi_rad=0.0,
    )
    built = _built(config, [compact, channel])

    result = audit_target_distributions(
        config,
        built,
        _center_plan(built.realized_anchors_A.tolist()),
        _phase_grid(config, pore_mask),
    )

    assert not result.passed
    for name in ("theta", "compact_eta", "channel_eta", "roughness", "tau"):
        assert any(name in warning and "distribution" in warning for warning in result.warnings)


def test_audit_uses_realized_compact_major_axis_instead_of_latent_theta() -> None:
    count = 64
    config = _audit_config(
        seed_density=count / 1000.0,
        target_porosity=0.10,
        channel_fraction=0.0,
        compact_eta={"family": "constant", "value": 4.0},
    )
    pore_mask = np.zeros((10, 10, 10), dtype=bool)
    pore_mask.ravel()[:100] = True
    latent_theta = np.pi * stats.beta(a=1.0, b=100_000.0).ppf(
        (np.arange(count, dtype=float) + 0.5) / count
    )
    centers = np.column_stack(
        [
            0.5 + (np.arange(count, dtype=float) % 8),
            0.5 + ((np.arange(count, dtype=float) // 8) % 8),
            np.full(count, 5.0),
        ]
    )
    units = [
        CompactUnit(
            unit_id=f"compact-{index:04d}",
            center_A=center,
            radii_A=np.array([1.0, 1.0, 4.0]),
            orientation=_identity_rotation(),
            exponent=2.0,
            roughness=0.0,
            latent_theta_rad=float(theta),
            latent_phi_rad=0.0,
        )
        for index, (center, theta) in enumerate(zip(centers, latent_theta, strict=True))
    ]

    result = audit_target_distributions(
        config,
        _built(config, units),
        _center_plan(centers.tolist()),
        _phase_grid(config, pore_mask),
    )

    assert not result.passed
    assert result.theta_result is not None
    assert not result.theta_result.passed


def test_audit_excludes_near_spherical_compacts_from_theta_distribution() -> None:
    config = _audit_config(
        seed_density=0.002,
        target_porosity=0.10,
        channel_fraction=0.0,
        compact_eta={"family": "constant", "value": 1.01},
        orientation_aspect_ratio_tolerance=0.05,
    )
    pore_mask = np.zeros((10, 10, 10), dtype=bool)
    pore_mask.ravel()[:100] = True
    units = [
        CompactUnit(
            unit_id=f"compact-{index:04d}",
            center_A=np.array([2.0 + index, 5.0, 5.0]),
            radii_A=np.array([1.0, 1.0, 1.01]),
            orientation=_identity_rotation(),
            exponent=2.0,
            roughness=0.0,
            latent_theta_rad=np.pi,
            latent_phi_rad=0.0,
        )
        for index in range(2)
    ]

    result = audit_target_distributions(
        config,
        _built(config, units),
        _center_plan([[2.0, 5.0, 5.0], [3.0, 5.0, 5.0]]),
        _phase_grid(config, pore_mask),
    )

    assert result.theta_result is None
    assert "theta" not in " ".join(result.warnings)


def test_audit_gates_compact_relative_volume_distribution() -> None:
    config = _audit_config(
        seed_density=0.002,
        target_porosity=0.10,
        channel_fraction=0.0,
        compact_relative_volume={
            "family": "mixture",
            "components": [
                {"weight": 0.5, "family": "constant", "value": 0.5},
                {"weight": 0.5, "family": "constant", "value": 1.5},
            ],
        },
    )
    pore_mask = np.zeros((10, 10, 10), dtype=bool)
    pore_mask.ravel()[:100] = True
    units = [
        CompactUnit.sphere(f"compact-{index:04d}", np.array([2.0 + index * 4.0, 5.0, 5.0]), 1.0)
        for index in range(2)
    ]

    result = audit_target_distributions(
        config,
        _built(config, units),
        _center_plan([[2.0, 5.0, 5.0], [6.0, 5.0, 5.0]]),
        _phase_grid(config, pore_mask),
    )

    assert not result.passed
    assert result.compact_relative_volume_result is not None
    assert not result.compact_relative_volume_result.passed
    assert any("compact_relative_volume distribution" in warning for warning in result.warnings)
    assert result.unit_volume_summary["per_unit"][0]["realized_clipped_volume_A3"] >= 0.0


def test_audit_gates_realized_channel_to_compact_mean_volume_ratio() -> None:
    config = _audit_config(
        seed_density=0.002,
        target_porosity=0.10,
        channel_fraction=0.5,
        mean_volume_ratio=3.0,
    )
    pore_mask = np.zeros((10, 10, 10), dtype=bool)
    pore_mask.ravel()[:100] = True
    compact = CompactUnit.sphere("compact-0000", np.array([2.0, 5.0, 5.0]), 1.0)
    channel = ChannelUnit.from_polyline(
        unit_id="channel-0000",
        control_points_unwrapped_A=np.array([[6.0, 5.0, 5.0], [7.0, 5.0, 5.0]]),
        cross_radius_A=1.0,
        roughness=0.0,
        latent_theta_rad=0.0,
        latent_phi_rad=0.0,
    )

    result = audit_target_distributions(
        config,
        _built(config, [compact, channel]),
        _center_plan([[2.0, 5.0, 5.0], [6.5, 5.0, 5.0]]),
        _phase_grid(config, pore_mask),
    )

    assert not result.passed
    assert result.realized_mean_volume_ratio is not None
    assert result.mean_volume_ratio_relative_error is not None
    assert any("channel/compact mean volume ratio" in warning for warning in result.warnings)


def test_audit_disabled_matrix_constraints_do_not_gate_blocked_matrix() -> None:
    config = _audit_config(
        box=(2.0, 2.0, 2.0),
        seed_density=0.001,
        target_porosity=0.5,
        channel_fraction=0.0,
        matrix_enabled=False,
        require_x_percolation=True,
        minimum_cross_section_fraction=1.0,
    )
    pore_mask = np.zeros((2, 2, 2), dtype=bool)
    pore_mask.ravel()[:4] = True

    result = audit_target_distributions(
        config,
        _built(config, []),
        _center_plan([]),
        _phase_grid(config, pore_mask),
    )

    assert result.passed
    assert not any("percolate" in warning for warning in result.warnings)
    assert not any("cross-section" in warning for warning in result.warnings)


def test_audit_uses_uneroded_matrix_when_no_skeleton_thickness_is_configured() -> None:
    config = _audit_config(
        box=(4.0, 3.0, 3.0),
        seed_density=0.001,
        target_porosity=32 / 36,
        channel_fraction=0.0,
        matrix_enabled=True,
        require_x_percolation=True,
    )
    semiconductor = np.zeros((3, 3, 4), dtype=bool)
    semiconductor[1, 1, :] = True

    result = audit_target_distributions(
        config,
        _built(config, []),
        _center_plan([]),
        _phase_grid(config, ~semiconductor),
    )

    assert result.passed


def test_audit_erodes_matrix_only_when_skeleton_thickness_is_configured() -> None:
    config = _audit_config(
        box=(4.0, 3.0, 3.0),
        seed_density=0.001,
        target_porosity=32 / 36,
        channel_fraction=0.0,
        matrix_enabled=True,
        require_x_percolation=True,
        minimum_skeleton_thickness_A=3.0,
    )
    semiconductor = np.zeros((3, 3, 4), dtype=bool)
    semiconductor[1, 1, :] = True

    result = audit_target_distributions(
        config,
        _built(config, []),
        _center_plan([]),
        _phase_grid(config, ~semiconductor),
    )

    assert not result.passed
    assert any("percolate" in warning for warning in result.warnings)


def test_audit_merges_pore_components_across_periodic_x_and_y_faces() -> None:
    config = _audit_config(
        box=(4.0, 3.0, 1.0),
        seed_density=0.001,
        target_porosity=0.25,
        channel_fraction=0.0,
        matrix_enabled=False,
    )
    pore_mask = np.zeros((1, 3, 4), dtype=bool)
    pore_mask[0, 0, 0] = True
    pore_mask[0, 2, 3] = True
    pore_mask[0, 0, 3] = True

    result = audit_target_distributions(
        config,
        _built(config, []),
        _center_plan([]),
        _phase_grid(config, pore_mask),
    )

    assert result.connected_pore_domains == 1
    assert np.isclose(result.largest_pore_fraction, 1.0)


def test_cached_unit_occupancy_masks_match_direct_brute_force_phase() -> None:
    box = np.array([8.0, 8.0, 8.0])
    units = [
        CompactUnit.sphere("left", np.array([0.5, 4.0, 4.0]), 1.5),
        CompactUnit.sphere("middle", np.array([4.0, 4.0, 4.0]), 1.25),
    ]
    built = BuiltGeometry(
        geometry=PoreGeometry(units, box),
        units=units,
        realized_anchors_A=np.vstack([unit.anchor_A for unit in units]),
        latent_to_realized_ids={},
    )
    grid = voxelize_geometry(built.geometry, box, 0.5)
    indices = np.arange(grid.pore_mask.size, dtype=np.int64)
    points = _grid_points(indices, grid.pore_mask.shape, grid.spacing_A, grid.origin_A)

    actual = _unit_occupancy_masks(built, grid)

    for unit, unit_mask in zip(units, actual, strict=True):
        expected = PoreGeometry([unit], box).sdf(points).reshape(grid.pore_mask.shape) < 0.0
        assert np.array_equal(unit_mask, expected)


def test_local_thickness_stability_reuses_supplied_fine_grid(monkeypatch) -> None:
    config = _audit_config(
        box=(8.0, 8.0, 8.0),
        seed_density=1.0 / 512.0,
        target_porosity=0.01,
        channel_fraction=0.0,
    )
    config = config.model_copy(
        update={
            "audit": config.audit.model_copy(
                update={"coarse_spacing_A": 2.0, "fine_spacing_A": 1.0}
            )
        }
    )
    built = _built(config, [])
    fine_grid = PhaseGrid(
        np.zeros((8, 8, 8), dtype=bool),
        np.zeros(3),
        1.0,
        np.array([8.0, 8.0, 8.0]),
    )
    spacings: list[float] = []

    def recording_voxelize(geometry, box_A, spacing_A):
        spacings.append(float(spacing_A))
        return voxelize_geometry(geometry, box_A, spacing_A)

    monkeypatch.setattr("porous_film.metrics.audit.voxelize_geometry", recording_voxelize)

    result = _local_thickness_stability_result(config, built, [], fine_grid=fine_grid)

    assert result.passed
    assert spacings == [2.0]


def _audit_config(
    *,
    box: tuple[float, float, float] = (10.0, 10.0, 10.0),
    seed_density: float,
    target_porosity: float,
    channel_fraction: float,
    matrix_enabled: bool = False,
    require_x_percolation: bool = False,
    minimum_cross_section_fraction: float = 0.0,
    compact_eta: dict | None = None,
    compact_relative_volume: dict | None = None,
    channel_relative_volume: dict | None = None,
    channel_eta: dict | None = None,
    channel_tau: dict | None = None,
    mean_volume_ratio: float = 1.0,
    minimum_skeleton_thickness_A: float | None = None,
    orientation_aspect_ratio_tolerance: float | None = None,
) -> GeneratorConfig:
    matrix_constraints = {
        "enabled": matrix_enabled,
        "require_x_percolation": require_x_percolation,
        "minimum_cross_section_fraction": minimum_cross_section_fraction,
        "maximum_overlap_fraction": 1.0,
    }
    if minimum_skeleton_thickness_A is not None:
        matrix_constraints["minimum_skeleton_thickness_A"] = minimum_skeleton_thickness_A
    data = {
        "task": {"name": "audit-test", "random_seed": 11},
        "film": {
            "target_box_A": {"x": box[0], "y": box[1], "z": box[2]},
            "packing_box_A": {"x": box[0], "y": box[1], "z": box[2]},
        },
        "pores": {
            "seed_number_density_A3": seed_density,
            "target_porosity": target_porosity,
            "channel_fraction_by_count": channel_fraction,
            "channel_to_compact_mean_volume_ratio": mean_volume_ratio,
        },
        "center_distribution": {
            "mode": "lattice_jitter",
            "lattice": "simple_cubic",
            "position_jitter": 0.0,
        },
        "orientation": {
            "distribution": {"family": "beta", "alpha": 1.0, "beta": 100_000.0},
            "azimuth": "uniform",
        },
        "compact": {
            "relative_volume": compact_relative_volume or {"family": "constant", "value": 1.0},
            "aspect_ratio": compact_eta or {"family": "constant", "value": 1.0},
            "roughness": {"family": "constant", "value": 0.0},
        },
        "channel": {
            "relative_volume": channel_relative_volume or {"family": "constant", "value": 1.0},
            "eta": channel_eta or {"family": "constant", "value": 2.0},
            "tau": channel_tau or {"family": "constant", "value": 1.0},
            "roughness": {"family": "constant", "value": 0.0},
        },
        "matrix_constraints": matrix_constraints,
        "audit": {"coarse_spacing_A": 1.0, "fine_spacing_A": 1.0},
        "pore_material": {"pdb": "argon.pdb", "molecule_count": 1},
    }
    if orientation_aspect_ratio_tolerance is not None:
        data["audit"]["orientation_aspect_ratio_tolerance"] = orientation_aspect_ratio_tolerance
    return GeneratorConfig.model_validate(data)


def _phase_grid(config: GeneratorConfig, pore_mask: np.ndarray) -> PhaseGrid:
    return PhaseGrid(
        pore_mask=pore_mask,
        origin_A=np.zeros(3, dtype=float),
        spacing_A=1.0,
        target_box_A=np.array(
            [config.film.target_box_A.x, config.film.target_box_A.y, config.film.target_box_A.z],
            dtype=float,
        ),
    )


def _built(config: GeneratorConfig, units: list) -> BuiltGeometry:
    box = np.array(
        [config.film.target_box_A.x, config.film.target_box_A.y, config.film.target_box_A.z],
        dtype=float,
    )
    anchors = np.vstack([unit.anchor_A for unit in units]) if units else np.empty((0, 3))
    return BuiltGeometry(
        geometry=PoreGeometry(units, box),
        units=units,
        realized_anchors_A=anchors,
        latent_to_realized_ids={},
    )


def _center_plan(points: list[list[float]]) -> CenterSeedPlan:
    return CenterSeedPlan(
        intended_points_A=np.asarray(points, dtype=float).reshape((-1, 3)),
        target_rdf_xi=np.array([0.0]),
        target_rdf_values=np.array([1.0]),
        starting_loss=0.0,
        initialization_loss=0.0,
    )


def _identity_rotation():
    from scipy.spatial.transform import Rotation

    return Rotation.identity()


def test_complex_compact_volume_distribution_uses_continuous_unit_geometry() -> None:
    from scipy.spatial.transform import Rotation

    from porous_film.geometry.complex_shapes import generate_multilobe_profile

    config = _audit_config(
        seed_density=0.004,
        target_porosity=0.10,
        channel_fraction=0.0,
        compact_relative_volume={"family": "constant", "value": 1.0},
        compact_eta={"family": "constant", "value": 1.0},
    )
    centers = np.array(
        [
            [2.5, 2.5, 2.5],
            [2.5, 7.5, 7.5],
            [7.5, 2.5, 7.5],
            [7.5, 7.5, 2.5],
        ]
    )
    units = []
    for index, center in enumerate(centers):
        profile = generate_multilobe_profile(100.0, 1.0, 8100 + index)
        units.append(
            CompactUnit(
                unit_id=f"compact-{index:04d}",
                center_A=center,
                radii_A=profile.envelope_radii_A,
                orientation=Rotation.from_euler(
                    "xyz", [17 * index, 11 * index, 23 * index], degrees=True
                ),
                exponent=2.0,
                roughness=0.0,
                latent_theta_rad=0.0,
                latent_phi_rad=0.0,
                latent_target_volume_A3=100.0,
                shape_model="multilobe-v1",
                shape_seed=8100 + index,
                lobe_centers_local_A=profile.lobe_centers_local_A,
                lobe_radii_A=profile.lobe_radii_A,
                smooth_length_A=profile.smooth_length_A,
                envelope_fill_fraction=profile.envelope_fill_fraction,
                centroid_offset_A=profile.centroid_offset_A,
                lobes_connected=profile.connected,
            )
        )
    pore_mask = np.zeros((10, 10, 10), dtype=bool)
    pore_mask.ravel()[:100] = True

    result = audit_target_distributions(
        config,
        _built(config, units),
        _center_plan(centers.tolist()),
        _phase_grid(config, pore_mask),
    )

    clipped = [
        item["realized_clipped_volume_A3"] for item in result.unit_volume_summary["per_unit"]
    ]
    continuous = [item["realized_volume_A3"] for item in result.unit_volume_summary["per_unit"]]
    assert len(set(clipped)) > 1
    assert np.allclose(continuous, 100.0, rtol=0.03)
    assert result.compact_relative_volume_result is not None
    assert result.compact_relative_volume_result.passed


def test_channel_eta_tau_constant_targets_use_geometry_relative_tolerance() -> None:
    from porous_film.geometry import build_units

    config = _audit_config(
        box=(20.0, 20.0, 20.0),
        seed_density=0.0005,
        target_porosity=0.10,
        channel_fraction=1.0,
        channel_eta={"family": "constant", "value": 5.0},
        channel_tau={"family": "constant", "value": 1.3},
    )
    centers = np.array(
        [
            [4.0, 4.0, 4.0],
            [4.0, 16.0, 16.0],
            [16.0, 4.0, 16.0],
            [16.0, 16.0, 4.0],
        ]
    )
    plan = _center_plan(centers.tolist())
    built = build_units(config, plan, np.random.default_rng(9123))
    pore_mask = np.zeros((20, 20, 20), dtype=bool)
    pore_mask.ravel()[:800] = True

    result = audit_target_distributions(
        config,
        built,
        plan,
        _phase_grid(config, pore_mask),
    )

    eta_errors = np.abs(np.asarray([unit.eta for unit in built.units]) / 5.0 - 1.0)
    tau_errors = np.abs(np.asarray([unit.tortuosity for unit in built.units]) / 1.3 - 1.0)
    assert np.max(eta_errors) <= 0.01
    assert np.max(tau_errors) <= 0.01
    assert result.channel_eta_result is not None
    assert result.channel_eta_result.passed
    assert result.tau_result is not None
    assert result.tau_result.passed


def test_identical_compact_and_channel_roughness_targets_are_not_duplicated_mixture() -> None:
    config = _audit_config(
        seed_density=0.002,
        target_porosity=0.10,
        channel_fraction=0.5,
    )
    roughness = {"family": "constant", "value": 0.04}
    config = config.model_copy(
        update={
            "compact": config.compact.model_copy(
                update={"roughness": config.compact.roughness.model_validate(roughness)}
            ),
            "channel": config.channel.model_copy(
                update={"roughness": config.channel.roughness.model_validate(roughness)}
            ),
        }
    )
    compact = CompactUnit(
        unit_id="compact-roughness",
        center_A=np.array([2.0, 5.0, 5.0]),
        radii_A=np.ones(3),
        orientation=_identity_rotation(),
        exponent=2.0,
        roughness=0.04,
    )
    channel = ChannelUnit.from_polyline(
        "channel-roughness",
        np.array([[6.0, 5.0, 5.0], [8.0, 5.0, 5.0]]),
        1.0,
        0.04,
    )
    pore_mask = np.zeros((10, 10, 10), dtype=bool)
    pore_mask.ravel()[:100] = True

    result = audit_target_distributions(
        config,
        _built(config, [compact, channel]),
        _center_plan([[2.0, 5.0, 5.0], [7.0, 5.0, 5.0]]),
        _phase_grid(config, pore_mask),
    )

    assert result.roughness_result is not None
    assert result.roughness_result.passed
    assert "roughness" not in result.mixture_weight_errors


def _schema_v3_audit_config() -> GeneratorConfig:
    return GeneratorConfig.model_validate(
        {
            "schema_version": 3,
            "task": {"name": "v3-audit", "random_seed": 91},
            "film": {"target_box_A": {"x": 20.0, "y": 20.0, "z": 20.0}},
            "formal_targets": {
                "position_quantity": {
                    "center_distance_xy": {
                        "components": [
                            {
                                "kind": "peak",
                                "amplitude": 0.5,
                                "center_A": 8.0,
                                "width_A": 2.0,
                            }
                        ]
                    }
                },
                "shape": {
                    "equivalent_diameter_A": {
                        "family": "beta",
                        "alpha": 2.0,
                        "beta": 2.0,
                        "lower": 3.0,
                        "upper": 8.0,
                    },
                    "orientation": {
                        "model": "paired_projected_planes",
                        "components": [
                            {
                                "weight": 1.0,
                                "theta_xz_deg": {
                                    "family": "beta",
                                    "alpha": 2.0,
                                    "beta": 2.0,
                                    "lower": 70.0,
                                    "upper": 90.0,
                                },
                                "theta_xy_deg": {
                                    "family": "beta",
                                    "alpha": 2.0,
                                    "beta": 2.0,
                                    "lower": 0.0,
                                    "upper": 20.0,
                                },
                            }
                        ],
                    },
                    "channel_aspect_ratio": {
                        "family": "beta",
                        "alpha": 2.0,
                        "beta": 2.0,
                        "lower": 2.0,
                        "upper": 10.0,
                    },
                    "channel_tortuosity": {
                        "family": "beta",
                        "alpha": 2.0,
                        "beta": 2.0,
                        "lower": 1.0,
                        "upper": 1.3,
                    },
                    "curvature_fluctuation": {
                        "family": "beta",
                        "alpha": 2.0,
                        "beta": 2.0,
                        "lower": 0.0,
                        "upper": 2.0,
                    },
                },
                "proportion": {"porosity": 0.15},
            },
            "generation_controls": {
                "seed_number_density_A3": 0.00025,
                "channel_fraction_by_count": 1.0,
                "channel_to_compact_mean_volume_ratio": 1.0,
            },
            "measurement": {
                "z_slice_spacing_A": 1.0,
                "center_min_separation_A": 2.0,
                "center_tracking_max_displacement_A": 2.0,
                "center_distance_bin_width_A": 1.0,
                "center_distance_max_A": 12.0,
                "center_distance_reference_samples": 4096,
                "centerline_sample_spacing_A": 1.0,
                "cross_section_spacing_A": 2.0,
                "boundary_resample_spacing_A": 0.5,
                "curvature_smoothing_length_A": 1.0,
                "branch_exclusion_length_A": 2.0,
                "surface_exclusion_length_A": 1.0,
                "orientation_projection_min_fraction": 0.05,
            },
            "matrix_constraints": {
                "enabled": False,
                "require_x_percolation": False,
                "minimum_cross_section_fraction": 0.0,
                "maximum_overlap_fraction": 1.0,
            },
            "audit": {
                "enabled": False,
                "candidate_count_per_round": 1,
                "maximum_rounds": 1,
                "coarse_spacing_A": 2.0,
                "fine_spacing_A": 1.0,
            },
        }
    )


def _two_vertical_pore_grid() -> PhaseGrid:
    _z, y, x = np.indices((20, 20, 20), dtype=float)
    first = (x - 5.5) ** 2 + (y - 10.5) ** 2 <= 2.5**2
    second = (x - 13.5) ** 2 + (y - 10.5) ** 2 <= 2.5**2
    return PhaseGrid(
        pore_mask=first | second,
        origin_A=np.zeros(3),
        spacing_A=1.0,
        target_box_A=np.array([20.0, 20.0, 20.0]),
    )


def test_schema_v3_audit_uses_final_phase_instead_of_generation_unit_metrics() -> None:
    config = _schema_v3_audit_config()
    grid = _two_vertical_pore_grid()
    empty_built = BuiltGeometry(
        geometry=PoreGeometry([], grid.target_box_A),
        units=[],
        realized_anchors_A=np.empty((0, 3)),
        latent_to_realized_ids={},
    )
    misleading_unit = CompactUnit.sphere(
        "misleading-generation-unit",
        np.array([2.0, 2.0, 2.0]),
        0.5,
    )
    misleading_built = BuiltGeometry(
        geometry=PoreGeometry([misleading_unit], grid.target_box_A),
        units=[misleading_unit],
        realized_anchors_A=np.array([misleading_unit.anchor_A]),
        latent_to_realized_ids={"latent-0000": misleading_unit.unit_id},
    )
    center_plan = _center_plan([])

    first = audit_target_distributions(config, empty_built, center_plan, grid)
    second = audit_target_distributions(config, misleading_built, center_plan, grid)

    assert first.formal_measurements is not None
    assert second.formal_measurements is not None
    assert first.distribution_results == second.distribution_results
    np.testing.assert_array_equal(
        first.formal_measurements.center_distance_xy.observed_pair_counts,
        second.formal_measurements.center_distance_xy.observed_pair_counts,
    )
    assert not any("seed_count" in warning for warning in first.warnings)


def test_schema_v3_audit_reports_final_geometry_distribution_results() -> None:
    raw = _schema_v3_audit_config().model_dump(mode="python")
    raw["pore_constraints"] = {"z_connectivity": "all_components"}
    config = GeneratorConfig.model_validate(raw)
    grid = _two_vertical_pore_grid()
    built = BuiltGeometry(
        geometry=PoreGeometry([], grid.target_box_A),
        units=[],
        realized_anchors_A=np.empty((0, 3)),
        latent_to_realized_ids={},
    )

    result = audit_target_distributions(config, built, _center_plan([]), grid)

    assert {
        "equivalent_diameter",
        "theta_xz",
        "channel_eta",
        "channel_tau",
        "curvature_fluctuation",
    } <= set(result.distribution_results)
    assert result.center_distance_xy_result is not None
    assert result.theta_xz_result is not None
    assert result.theta_xy_result is None
    assert result.equivalent_diameter_result is not None
    assert result.curvature_fluctuation_result is not None


def test_schema_v3_unrestricted_skips_through_dependent_target_gates() -> None:
    base = _two_vertical_pore_grid()
    mask = base.pore_mask.copy()
    mask[10:] = False
    grid = PhaseGrid(mask, base.origin_A, base.spacing_A, base.target_box_A)
    raw = _schema_v3_audit_config().model_dump(mode="python")
    raw["formal_targets"]["proportion"]["porosity"] = grid.porosity
    raw["pore_constraints"] = {"z_connectivity": "unrestricted"}
    config = GeneratorConfig.model_validate(raw)
    built = BuiltGeometry(
        geometry=PoreGeometry([], grid.target_box_A),
        units=[],
        realized_anchors_A=np.empty((0, 3)),
        latent_to_realized_ids={},
    )

    result = audit_target_distributions(config, built, _center_plan([]), grid)

    assert result.passed
    assert result.through_centerline_count == 0
    assert result.equivalent_diameter_result is None
    assert result.theta_xz_result is None
    assert result.theta_xy_result is None
    assert result.channel_eta_result is None
    assert result.tau_result is None
    assert result.curvature_fluctuation_result is None
    assert result.paired_orientation_result is None
    assert result.center_distance_xy_result is not None
    assert result.center_distance_xy_result["evaluated"] is False
    assert result.center_distance_xy_result["passed"] is None
    assert not any("no valid samples" in warning for warning in result.warnings)


def test_schema_v3_audit_gates_configured_minimum_final_samples() -> None:
    raw = _schema_v3_audit_config().model_dump(mode="python")
    raw["pore_constraints"] = {
        "minimum_through_centerlines": 2,
        "minimum_valid_cross_sections": 100,
    }
    config = GeneratorConfig.model_validate(raw)
    base = _two_vertical_pore_grid()
    grid = PhaseGrid(
        base.pore_mask & (np.indices(base.pore_mask.shape)[2] < 10),
        base.origin_A,
        base.spacing_A,
        base.target_box_A,
    )
    built = BuiltGeometry(
        geometry=PoreGeometry([], grid.target_box_A),
        units=[],
        realized_anchors_A=np.empty((0, 3)),
        latent_to_realized_ids={},
    )

    result = audit_target_distributions(config, built, _center_plan([]), grid)

    assert not result.passed
    assert result.through_centerline_count == 1
    assert result.valid_through_cross_section_count < 100
    assert any("minimum through centerlines" in warning for warning in result.warnings)
    assert any("minimum valid cross-sections" in warning for warning in result.warnings)


def test_schema_v3_audit_gates_all_final_pore_components_through_z() -> None:
    raw = _schema_v3_audit_config().model_dump(mode="python")
    raw["pore_constraints"] = {"z_connectivity": "all_components"}
    raw["formal_targets"]["shape"]["compact_aspect_ratio"] = None
    config = GeneratorConfig.model_validate(raw)
    base = _two_vertical_pore_grid()
    mask = base.pore_mask.copy()
    mask[10, 2, 10] = True
    grid = PhaseGrid(mask, base.origin_A, base.spacing_A, base.target_box_A)
    built = BuiltGeometry(
        geometry=PoreGeometry([], grid.target_box_A),
        units=[],
        realized_anchors_A=np.empty((0, 3)),
        latent_to_realized_ids={},
    )

    result = audit_target_distributions(config, built, _center_plan([]), grid)

    assert not result.passed
    assert result.through_pore_domain_count < result.connected_pore_domains
    assert any("all pore components" in warning for warning in result.warnings)


def test_all_components_marks_unconfigured_final_distributions_not_applicable() -> None:
    grid = _two_vertical_pore_grid()
    raw = _schema_v3_audit_config().model_dump(mode="python")
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
    raw["formal_targets"]["proportion"]["porosity"] = grid.porosity
    raw["pore_constraints"] = {"z_connectivity": "all_components"}
    config = GeneratorConfig.model_validate(raw)
    built = BuiltGeometry(
        geometry=PoreGeometry([], grid.target_box_A),
        units=[],
        realized_anchors_A=np.empty((0, 3)),
        latent_to_realized_ids={},
    )

    result = audit_target_distributions(config, built, _center_plan([]), grid)

    assert result.passed
    assert result.center_distance_xy_result["evaluated"] is False
    assert result.center_distance_xy_result["passed"] is None
    assert result.distribution_results == {}
    assert result.paired_orientation_result is None


def test_paired_orientation_audit_rejects_wrong_joint_pairing_with_matching_marginals() -> None:
    target = {
        "components": [
            {
                "weight": 0.5,
                "theta_xz_deg": {
                    "family": "beta",
                    "alpha": 2.0,
                    "beta": 2.0,
                    "lower": 70.0,
                    "upper": 80.0,
                },
                "theta_xy_deg": {
                    "family": "beta",
                    "alpha": 2.0,
                    "beta": 2.0,
                    "lower": 0.0,
                    "upper": 10.0,
                },
            },
            {
                "weight": 0.5,
                "theta_xz_deg": {
                    "family": "beta",
                    "alpha": 2.0,
                    "beta": 2.0,
                    "lower": 0.0,
                    "upper": 10.0,
                },
                "theta_xy_deg": {
                    "family": "beta",
                    "alpha": 2.0,
                    "beta": 2.0,
                    "lower": 70.0,
                    "upper": 80.0,
                },
            },
        ]
    }

    matching = _compare_paired_orientation_pairs(np.array([[75.0, 5.0], [5.0, 75.0]]), target)
    crossed = _compare_paired_orientation_pairs(np.array([[75.0, 75.0], [5.0, 5.0]]), target)

    assert matching["passed"]
    assert not crossed["passed"]
    assert crossed["unassigned_pair_count"] == 2


def test_paired_orientation_audit_skips_redundant_factorized_gate() -> None:
    shared_xz = {
        "family": "beta",
        "alpha": 2.0,
        "beta": 2.0,
        "lower": 60.0,
        "upper": 80.0,
    }
    target = {
        "components": [
            {
                "weight": 0.5,
                "theta_xz_deg": shared_xz,
                "theta_xy_deg": {
                    "family": "beta",
                    "alpha": 2.0,
                    "beta": 2.0,
                    "lower": 0.0,
                    "upper": 30.0,
                },
            },
            {
                "weight": 0.5,
                "theta_xz_deg": shared_xz,
                "theta_xy_deg": {
                    "family": "beta",
                    "alpha": 2.0,
                    "beta": 2.0,
                    "lower": 30.0,
                    "upper": 60.0,
                },
            },
        ]
    }

    result = _compare_paired_orientation_pairs(
        np.array([[81.0, 10.0], [75.0, 40.0], [72.0, 45.0]]),
        target,
    )

    assert result is not None
    assert result["passed"]
    assert result["joint_gate_required"] is False


def test_schema_v3_audit_compares_final_compact_component_eta() -> None:
    z, y, x = np.indices((20, 20, 20), dtype=float)
    mask = ((x + 0.5 - 10.0) / 4.5) ** 2 + ((y + 0.5 - 10.0) / 2.5) ** 2 + (
        (z + 0.5 - 10.0) / 2.5
    ) ** 2 <= 1.0
    base_config = _schema_v3_audit_config()
    probe_grid = PhaseGrid(mask, np.zeros(3), 1.0, np.array([20.0, 20.0, 20.0]))
    probe = measure_final_geometry(probe_grid, base_config.measurement)
    eta = probe.compact_geometries[0].eta
    raw = base_config.model_dump(mode="python")
    raw["formal_targets"]["shape"]["compact_aspect_ratio"] = {
        "family": "constant",
        "value": eta,
    }
    raw["formal_targets"]["proportion"]["porosity"] = probe_grid.porosity
    config = GeneratorConfig.model_validate(raw)
    built = BuiltGeometry(
        geometry=PoreGeometry([], probe_grid.target_box_A),
        units=[],
        realized_anchors_A=np.empty((0, 3)),
        latent_to_realized_ids={},
    )

    result = audit_target_distributions(config, built, _center_plan([]), probe_grid)

    assert result.compact_eta_result is not None
    assert result.compact_eta_result.passed


def test_pipeline_audit_summary_serializes_final_geometry_measurements() -> None:
    from porous_film.pipeline import _audit_summary

    raw = _schema_v3_audit_config().model_dump(mode="python")
    raw["pore_constraints"] = {"z_connectivity": "all_components"}
    config = GeneratorConfig.model_validate(raw)
    grid = _two_vertical_pore_grid()
    built = BuiltGeometry(
        geometry=PoreGeometry([], grid.target_box_A),
        units=[],
        realized_anchors_A=np.empty((0, 3)),
        latent_to_realized_ids={},
    )
    result = audit_target_distributions(config, built, _center_plan([]), grid)

    summary = _audit_summary(result)

    assert summary["center_distance_xy_result"]["pair_count"] > 0
    assert summary["equivalent_diameter_result"] is not None
    assert summary["formal_measurements"]["through_centerline_count"] == 2
    assert len(summary["formal_measurements"]["cross_sections"]) > 0
