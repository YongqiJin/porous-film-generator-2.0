from __future__ import annotations

import importlib

import numpy as np
import pytest

from porous_film.metrics import (
    compare_local_thickness_coarse_fine,
    local_thickness_distribution,
    local_thickness_field,
)


def test_cylindrical_channel_reports_diameter_not_wall_distance() -> None:
    z, y, _x = np.indices((9, 9, 9))
    mask = (y - 4) ** 2 + (z - 4) ** 2 <= 4

    field = local_thickness_field(mask, spacing_A=1.0, periodic_xy=False)

    assert field[4, 4, 4] >= 4.0
    assert field[4, 4, 1] >= 4.0


def test_local_thickness_rejects_masks_above_explicit_voxel_limit() -> None:
    mask = np.ones((2, 2, 2), dtype=bool)

    with pytest.raises(ValueError, match="max_voxels"):
        local_thickness_field(mask, spacing_A=1.0, periodic_xy=False, max_voxels=7)


def test_periodic_xy_thickness_uses_wrapped_neighbors_and_returns_central_shape() -> None:
    y, x = np.indices((7, 7))
    xy_disk = np.minimum(x, 7 - x) ** 2 + (y - 3) ** 2 <= 4
    mask = np.broadcast_to(xy_disk, (7, 7, 7)).copy()

    nonperiodic = local_thickness_field(mask, spacing_A=1.0, periodic_xy=False)
    periodic = local_thickness_field(mask, spacing_A=1.0, periodic_xy=True)

    assert periodic.shape == mask.shape
    assert periodic[3, 3, 0] >= 4.0
    assert periodic[3, 3, 0] > nonperiodic[3, 3, 0]


def test_periodic_xy_thickness_rejects_when_slab_temporaries_exceed_limit() -> None:
    mask = np.ones((3, 3, 3), dtype=bool)

    with pytest.raises(ValueError, match="max_voxels"):
        local_thickness_field(mask, spacing_A=1.0, periodic_xy=True, max_voxels=80)


def test_periodic_xy_thickness_uses_slabs_when_full_tile_exceeds_limit() -> None:
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 0] = True
    mask[2, 2, 4] = True

    slabbed = local_thickness_field(mask, spacing_A=1.0, periodic_xy=True, max_voxels=700)
    unbounded = local_thickness_field(mask, spacing_A=1.0, periodic_xy=True, max_voxels=10_000)

    np.testing.assert_allclose(slabbed, unbounded)


def test_periodic_xy_full_and_slab_paths_match_for_tapered_channel() -> None:
    z, y, x = np.indices((12, 20, 20))
    radius = 2.0 + 3.0 * z / 11.0
    mask = (x - 10) ** 2 + (y - 10) ** 2 <= radius**2

    full = local_thickness_field(mask, spacing_A=1.0, periodic_xy=True, max_voxels=100_000)
    slabbed = local_thickness_field(mask, spacing_A=1.0, periodic_xy=True, max_voxels=48_000)

    np.testing.assert_array_equal(full, slabbed)
    assert np.all(full[mask] > 0.0)


def test_periodic_xy_bounded_slabs_preserve_large_z_radius_feature() -> None:
    mask = np.zeros((15, 9, 9), dtype=bool)
    mask[4:11, :, :] = True

    full = local_thickness_field(mask, spacing_A=1.0, periodic_xy=True, max_voxels=20_000)
    bounded = local_thickness_field(mask, spacing_A=1.0, periodic_xy=True, max_voxels=7_000)

    np.testing.assert_array_equal(bounded, full)
    assert bounded[7, 0, 0] == 8.0


def test_periodic_xy_radius_halo_matches_full_tile_below_tile_memory_limit() -> None:
    mask = np.ones((9, 12, 14), dtype=bool)

    full = local_thickness_field(mask, spacing_A=1.0, periodic_xy=True, max_voxels=100_000)
    halo_bounded = local_thickness_field(
        mask,
        spacing_A=1.0,
        periodic_xy=True,
        max_voxels=7_000,
    )

    np.testing.assert_array_equal(halo_bounded, full)
    assert halo_bounded[4, 0, 0] == 10.0


@pytest.mark.parametrize("max_voxels", [7_000, 100_000])
def test_periodic_xy_path_avoids_large_binary_morphology(
    monkeypatch,
    max_voxels: int,
) -> None:
    local_thickness_module = importlib.import_module("porous_film.metrics.local_thickness")
    mask = np.ones((9, 12, 14), dtype=bool)

    def fail_large_binary_morphology(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("bounded periodic path must use fixed-memory distance transforms")

    monkeypatch.setattr(
        local_thickness_module.ndimage,
        "binary_erosion",
        fail_large_binary_morphology,
    )
    monkeypatch.setattr(
        local_thickness_module.ndimage,
        "binary_dilation",
        fail_large_binary_morphology,
    )

    bounded = local_thickness_field(
        mask,
        spacing_A=1.0,
        periodic_xy=True,
        max_voxels=max_voxels,
    )

    assert bounded[4, 0, 0] == 10.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_periodic_xy_distance_transform_matches_full_tile_for_random_masks(seed: int) -> None:
    local_thickness_module = importlib.import_module("porous_film.metrics.local_thickness")
    rng = np.random.default_rng(seed)
    mask = rng.random((11, 16, 17)) < 0.45
    mask[:, 7:10, :2] = True
    mask[:, 7:10, -2:] = True

    full = local_thickness_module._local_thickness_field_full(
        mask,
        1.0,
        periodic_xy=True,
        max_voxels=100_000,
    )
    bounded = local_thickness_field(mask, spacing_A=1.0, periodic_xy=True, max_voxels=12_000)

    np.testing.assert_array_equal(bounded, full)


def test_periodic_xy_bounded_slabs_raise_clear_error_when_radius_cannot_fit() -> None:
    mask = np.ones((9, 9, 9), dtype=bool)

    with pytest.raises(ValueError, match="memory limit.*radius 5"):
        local_thickness_field(mask, spacing_A=1.0, periodic_xy=True, max_voxels=1_000)


def test_local_thickness_distribution_probabilities_are_phase_normalized() -> None:
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[1, 1, 1] = True

    result = local_thickness_distribution(mask, spacing_A=2.0, periodic_xy=False)

    assert result.uncertainty_A == 2.0
    assert np.isclose(np.sum(result.probabilities), 1.0)
    assert np.count_nonzero(result.field_A) == 1


def test_coarse_fine_thickness_comparison_passes_within_two_fine_voxels() -> None:
    coarse = np.zeros((3, 3, 3), dtype=bool)
    coarse[1, 1, 1] = True
    fine = np.repeat(np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1), 2, axis=2)

    result = compare_local_thickness_coarse_fine(
        coarse,
        fine,
        coarse_spacing_A=2.0,
        fine_spacing_A=1.0,
        periodic_xy=False,
    )

    assert result.passed
    assert result.tolerance_A == 2.0
    assert result.warning is None


def test_coarse_fine_thickness_comparison_fails_when_quantiles_disagree() -> None:
    coarse = np.zeros((3, 3, 3), dtype=bool)
    coarse[1, 1, 1] = True
    fine = np.ones((8, 8, 8), dtype=bool)

    result = compare_local_thickness_coarse_fine(
        coarse,
        fine,
        coarse_spacing_A=2.0,
        fine_spacing_A=1.0,
        periodic_xy=False,
    )

    assert not result.passed
    assert result.max_quantile_error_A > result.tolerance_A
    assert result.warning is not None
