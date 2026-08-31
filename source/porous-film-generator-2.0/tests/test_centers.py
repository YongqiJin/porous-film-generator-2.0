from __future__ import annotations

import numpy as np
import pytest

from porous_film import centers
from porous_film.centers import (
    DEFAULT_MAX_REFERENCE_PAIRS,
    CenterSeedPlan,
    evaluate_rdf_target,
    generate_center_seeds,
    generate_lattice_jitter,
    pair_distances_periodic_xy,
)
from porous_film.centers.generation import _sobol_reference_histogram
from porous_film.config import GeneratorConfig


def _minimal_config(
    *,
    mode: str = "lattice_jitter",
    density: float = 0.000125,
    lattice: str | None = "simple_cubic",
    rdf: list[dict] | None = None,
) -> GeneratorConfig:
    center_distribution = {
        "mode": mode,
        "position_jitter": 0.0,
    }
    if lattice is not None:
        center_distribution["lattice"] = lattice
    if rdf is not None:
        center_distribution["rdf"] = rdf

    return GeneratorConfig.model_validate(
        {
            "task": {"name": "center-test", "random_seed": 7},
            "film": {
                "target_box_A": {"x": 20, "y": 20, "z": 20},
                "packing_box_A": {"x": 20, "y": 20, "z": 30},
            },
            "pores": {
                "seed_number_density_A3": density,
                "target_porosity": 0.10,
                "channel_fraction_by_count": 0.0,
                "channel_to_compact_mean_volume_ratio": 1.0,
            },
            "center_distribution": center_distribution,
            "compact": {
                "relative_volume": {"family": "constant", "value": 1.0},
                "aspect_ratio": {"family": "constant", "value": 1.0},
                "roughness": {"family": "constant", "value": 0.0},
            },
            "pore_material": {"pdb": "argon.pdb", "molecule_count": 1},
        }
    )


def test_pair_distance_wraps_x_and_y_but_not_z() -> None:
    points = np.array([[0.5, 0.5, 0.5], [9.5, 9.5, 9.5]])

    distances = pair_distances_periodic_xy(points, np.array([10.0, 10.0, 10.0]))

    assert np.allclose(distances, [np.sqrt(83.0)])


def test_simple_cubic_lattice_is_reproducible() -> None:
    first = generate_lattice_jitter(
        count=8,
        box_A=np.array([20.0, 20.0, 20.0]),
        lattice="simple_cubic",
        jitter_fraction=0.0,
        rng=np.random.default_rng(3),
    )
    second = generate_lattice_jitter(
        count=8,
        box_A=np.array([20.0, 20.0, 20.0]),
        lattice="simple_cubic",
        jitter_fraction=0.0,
        rng=np.random.default_rng(3),
    )

    assert np.array_equal(first, second)
    assert first.shape == (8, 3)


def test_bcc_and_fcc_lattices_return_requested_count_inside_box() -> None:
    box = np.array([15.0, 20.0, 25.0])

    for lattice in ("bcc", "fcc"):
        points = generate_lattice_jitter(
            count=7,
            box_A=box,
            lattice=lattice,
            jitter_fraction=0.0,
            rng=np.random.default_rng(11),
        )

        assert points.shape == (7, 3)
        assert np.all(points >= 0.0)
        assert np.all(points <= box)


def test_rdf_linear_components_return_one_at_long_range() -> None:
    xi = np.array([0.0, 1.0, 10.0])
    components = [
        {"kind": "exclusion", "amplitude": 1.0, "center": 0.0, "width": 0.3},
        {"kind": "peak", "amplitude": 1.2, "center": 1.0, "width": 0.1},
    ]

    values = evaluate_rdf_target(xi, components)

    assert values[0] >= 0.0
    assert values[1] > 1.0
    assert np.isclose(values[-1], 1.0, atol=1e-6)


def test_rdf_target_rejects_negative_below_tolerance_and_clips_roundoff() -> None:
    clipped = evaluate_rdf_target(
        np.array([0.0]),
        [{"kind": "exclusion", "amplitude": 1.0 + 5e-11, "center": 0.0, "width": 0.2}],
    )

    assert np.array_equal(clipped, np.array([0.0]))

    with pytest.raises(ValueError, match="nonnegative"):
        evaluate_rdf_target(
            np.array([0.0]),
            [{"kind": "dip", "amplitude": 1.0 + 2e-10, "center": 0.0, "width": 0.2}],
        )


def test_rdf_target_rejects_nonfinite_xi_parameters_and_values() -> None:
    with pytest.raises(ValueError, match="xi"):
        evaluate_rdf_target(
            np.array([0.0, np.nan]),
            [{"kind": "peak", "amplitude": 1.0, "center": 0.5, "width": 0.1}],
        )

    with pytest.raises(ValueError, match="parameters"):
        evaluate_rdf_target(
            np.array([0.0]),
            [{"kind": "peak", "amplitude": np.inf, "center": 0.5, "width": 0.1}],
        )

    with pytest.raises(ValueError, match="finite"):
        evaluate_rdf_target(
            np.array([np.inf]),
            [{"kind": "peak", "amplitude": 1.0, "center": 0.5, "width": 0.1}],
        )


def test_config_rdf_feature_location_is_dimensionless_xi() -> None:
    plan = generate_center_seeds(
        _minimal_config(
            mode="rdf",
            density=0.001,
            lattice=None,
            rdf=[{"kind": "peak", "amplitude": 2.0, "center_xi": 0.5, "width_xi": 0.03}],
        ),
        np.random.default_rng(13),
    )

    peak_xi = plan.target_rdf_xi[int(np.argmax(plan.target_rdf_values))]
    assert abs(peak_xi - 0.5) < 0.04


def test_config_rdf_supports_dip_and_exclusion_semantics() -> None:
    plan = generate_center_seeds(
        _minimal_config(
            mode="rdf",
            density=0.001,
            lattice=None,
            rdf=[
                {"kind": "dip", "amplitude": 0.4, "center_xi": 0.45, "width_xi": 0.04},
                {"kind": "exclusion", "amplitude": 0.6, "center_xi": 0.05, "width_xi": 0.05},
            ],
        ),
        np.random.default_rng(17),
    )

    dip_index = int(np.argmin(np.abs(plan.target_rdf_xi - 0.45)))
    exclusion_index = int(np.argmin(np.abs(plan.target_rdf_xi - 0.05)))
    assert plan.target_rdf_values[dip_index] < 1.0
    assert plan.target_rdf_values[exclusion_index] < 1.0


def test_config_parsed_oscillation_rdf_alternates_around_baseline() -> None:
    config = _minimal_config(
        mode="rdf",
        density=0.001,
        lattice=None,
        rdf=[{"kind": "oscillation", "amplitude": 0.4, "center_xi": 0.5, "width_xi": 0.5}],
    )
    hand_chosen_xi = np.array([0.5, 0.75, 1.0, 1.25])

    values = evaluate_rdf_target(hand_chosen_xi, list(config.center_distribution.rdf))

    assert values[0] > 1.0
    assert values[1] < 1.0
    assert values[2] > 1.0
    assert values[3] < 1.0


def test_generate_center_seeds_dispatches_lattice_mode_from_config() -> None:
    plan = generate_center_seeds(_minimal_config(), np.random.default_rng(5))

    assert isinstance(plan, CenterSeedPlan)
    assert np.array_equal(plan.intended_points_A, np.array([[10.0, 10.0, 10.0]]))
    assert plan.target_rdf_xi.ndim == 1
    assert plan.target_rdf_values.shape == plan.target_rdf_xi.shape
    assert plan.starting_loss == 0.0
    assert plan.initialization_loss == 0.0


def test_generate_center_seeds_dispatches_rdf_mode_with_finite_z_points() -> None:
    plan = generate_center_seeds(
        _minimal_config(
            mode="rdf",
            density=0.001,
            lattice=None,
            rdf=[{"amplitude": 0.5, "center_xi": 0.5, "width_xi": 0.1}],
        ),
        np.random.default_rng(19),
    )

    assert plan.intended_points_A.shape == (8, 3)
    assert np.all(plan.intended_points_A[:, :2] >= 0.0)
    assert np.all(plan.intended_points_A[:, :2] < 20.0)
    assert np.all((plan.intended_points_A[:, 2] >= 0.0) & (plan.intended_points_A[:, 2] <= 20.0))
    assert np.all(plan.target_rdf_values >= 0.0)
    assert np.isfinite(plan.initialization_loss)


def test_rdf_generation_is_deterministic_for_same_rng_seed() -> None:
    config = _minimal_config(
        mode="rdf",
        density=0.001,
        lattice=None,
        rdf=[{"kind": "peak", "amplitude": 0.5, "center_xi": 0.5, "width_xi": 0.1}],
    )

    first = generate_center_seeds(config, np.random.default_rng(23))
    second = generate_center_seeds(config, np.random.default_rng(23))

    assert np.array_equal(first.intended_points_A, second.intended_points_A)
    assert first.starting_loss == second.starting_loss
    assert first.initialization_loss == second.initialization_loss


def test_rdf_monte_carlo_final_loss_is_no_worse_than_starting_loss() -> None:
    plan = generate_center_seeds(
        _minimal_config(
            mode="rdf",
            density=0.001,
            lattice=None,
            rdf=[{"kind": "peak", "amplitude": 0.8, "center_xi": 0.35, "width_xi": 0.08}],
        ),
        np.random.default_rng(29),
    )

    assert plan.initialization_loss <= plan.starting_loss


def test_bcc_and_fcc_lattices_have_deterministic_truncated_layouts() -> None:
    box = np.array([20.0, 20.0, 20.0])

    bcc = generate_lattice_jitter(3, box, "bcc", 0.0, np.random.default_rng(1))
    fcc = generate_lattice_jitter(5, box, "fcc", 0.0, np.random.default_rng(1))

    assert np.array_equal(
        bcc,
        np.array([[2.5, 2.5, 2.5], [7.5, 7.5, 7.5], [2.5, 2.5, 12.5]]),
    )
    assert np.array_equal(
        fcc,
        np.array(
            [
                [2.5, 2.5, 2.5],
                [2.5, 7.5, 7.5],
                [7.5, 2.5, 7.5],
                [7.5, 7.5, 2.5],
                [2.5, 2.5, 12.5],
            ]
        ),
    )


def test_zero_centers_return_empty_plan_with_zero_losses() -> None:
    plan = generate_center_seeds(_minimal_config(density=1e-9), np.random.default_rng(31))

    assert plan.intended_points_A.shape == (0, 3)
    assert plan.starting_loss == 0.0
    assert plan.initialization_loss == 0.0


def test_invalid_lattice_and_box_are_rejected() -> None:
    with pytest.raises(ValueError, match="lattice"):
        generate_lattice_jitter(
            count=1,
            box_A=np.array([10.0, 10.0, 10.0]),
            lattice="hexagonal",
            jitter_fraction=0.0,
            rng=np.random.default_rng(1),
        )

    with pytest.raises(ValueError, match="box_A"):
        generate_lattice_jitter(
            count=1,
            box_A=np.array([10.0, np.inf, 10.0]),
            lattice="simple_cubic",
            jitter_fraction=0.0,
            rng=np.random.default_rng(1),
        )


def test_sobol_reference_histogram_uses_bounded_pair_sampling_for_large_counts() -> None:
    edges = np.linspace(0.0, 50.0, 8)

    hist = _sobol_reference_histogram(
        point_count=10_000,
        box=np.array([20.0, 20.0, 20.0]),
        distance_edges=edges,
        max_reference_pairs=128,
    )

    assert DEFAULT_MAX_REFERENCE_PAIRS == 65_536
    assert hist.shape == (edges.size - 1,)
    assert np.isclose(hist.sum(), 10_000 * 9_999 / 2)


def test_public_generate_centers_alias_dispatches_from_config() -> None:
    plan = centers.generate_centers(_minimal_config(), np.random.default_rng(5))

    assert isinstance(plan, CenterSeedPlan)
    assert np.array_equal(plan.intended_points_A, np.array([[10.0, 10.0, 10.0]]))


def test_pair_distance_in_xy_plane_ignores_z_separation() -> None:
    distances = centers.pair_distances_periodic_xy_plane(
        np.array([[0.5, 0.5, 0.5], [9.5, 9.5, 9.5]]),
        np.array([10.0, 10.0]),
    )

    assert np.allclose(distances, [np.sqrt(2.0)])


def test_schema_v3_center_target_uses_absolute_xy_distance() -> None:
    from test_pipeline import _pipeline_v3_config_dict

    data = _pipeline_v3_config_dict()
    data["generation_controls"]["seed_number_density_A3"] = 0.008
    data["formal_targets"]["position_quantity"]["center_distance_xy"]["components"] = [
        {
            "kind": "peak",
            "amplitude": 1.0,
            "center_A": 5.0,
            "width_A": 0.2,
        }
    ]
    config = GeneratorConfig.model_validate(data)

    plan = generate_center_seeds(config, np.random.default_rng(73))

    peak_distance = plan.target_rdf_xi[int(np.argmax(plan.target_rdf_values))]
    assert plan.distance_coordinate == "angstrom_xy"
    assert abs(peak_distance - 5.0) <= 0.6
