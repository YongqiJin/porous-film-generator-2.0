from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest

from porous_film.geometry import BuiltGeometry, ChannelUnit, CompactUnit, PoreGeometry
from porous_film.voxel import (
    PhaseGrid,
    PorosityResolutionError,
    solve_scale_for_porosity,
    voxelize_geometry,
)


def test_sphere_voxel_porosity_converges_with_resolution() -> None:
    unit = CompactUnit.sphere(
        unit_id="sphere",
        center_A=np.array([5.0, 5.0, 5.0]),
        radius_A=2.0,
    )
    geometry = PoreGeometry([unit], np.array([10.0, 10.0, 10.0]))

    coarse = voxelize_geometry(geometry, np.array([10.0, 10.0, 10.0]), 0.5)
    fine = voxelize_geometry(geometry, np.array([10.0, 10.0, 10.0]), 0.25)

    exact = 4.0 * np.pi * 2.0**3 / 3.0 / 1000.0
    assert abs(fine.porosity - exact) < abs(coarse.porosity - exact)


def test_voxelization_is_independent_of_chunk_size() -> None:
    unit = CompactUnit.sphere(
        unit_id="chunked",
        center_A=np.array([3.0, 3.0, 3.0]),
        radius_A=1.5,
    )
    geometry = PoreGeometry([unit], np.array([6.0, 6.0, 6.0]))

    unchunked = voxelize_geometry(
        geometry,
        np.array([6.0, 6.0, 6.0]),
        0.5,
        max_points_per_chunk=10_000,
    )
    chunked = voxelize_geometry(
        geometry,
        np.array([6.0, 6.0, 6.0]),
        0.5,
        max_points_per_chunk=7,
    )

    assert unchunked.pore_mask.shape == (12, 12, 12)
    assert np.array_equal(chunked.pore_mask, unchunked.pore_mask)


def test_voxelization_rejects_spacing_that_does_not_tile_target_box() -> None:
    geometry = PoreGeometry([], np.array([10.0, 10.0, 10.0]))

    with pytest.raises(ValueError, match="divisible"):
        voxelize_geometry(geometry, np.array([10.0, 10.0, 10.0]), 0.3)


def test_voxelization_samples_voxel_center_near_positive_faces() -> None:
    unit = CompactUnit.sphere(
        unit_id="positive-face",
        center_A=np.array([9.75, 9.75, 9.75]),
        radius_A=0.2,
    )
    geometry = PoreGeometry([unit], np.array([10.0, 10.0, 10.0]))

    grid = voxelize_geometry(geometry, np.array([10.0, 10.0, 10.0]), 0.5)

    assert grid.pore_mask[-1, -1, -1]


def test_voxelization_rejects_non_integer_chunk_size() -> None:
    geometry = PoreGeometry([], np.array([2.0, 2.0, 2.0]))

    with pytest.raises(TypeError, match="max_points_per_chunk"):
        voxelize_geometry(
            geometry,
            np.array([2.0, 2.0, 2.0]),
            1.0,
            max_points_per_chunk=1.5,
        )


def test_phase_grid_hdf5_round_trip(tmp_path: Path) -> None:
    grid = PhaseGrid(
        pore_mask=np.array([[[False, True], [True, False]]], dtype=bool),
        origin_A=np.array([0.0, 0.0, 0.0]),
        spacing_A=0.5,
        target_box_A=np.array([1.0, 1.0, 0.5]),
    )
    path = tmp_path / "phase.h5"

    grid.write_hdf5(path)
    restored = PhaseGrid.read_hdf5(path)

    assert np.array_equal(restored.pore_mask, grid.pore_mask)
    assert np.array_equal(restored.target_box_A, grid.target_box_A)
    assert restored.axis_order == "zyx"
    assert restored.periodic_axes == ("x", "y")
    assert np.isclose(restored.porosity, 0.5)
    with h5py.File(path, "r") as handle:
        assert handle["pore_mask"].compression == "gzip"
        assert handle.attrs["pore_mask_compression"] == "gzip"


def test_phase_grid_rejects_shape_that_does_not_match_target_box_counts() -> None:
    with pytest.raises(ValueError, match="pore_mask shape"):
        PhaseGrid(
            pore_mask=np.zeros((1, 2, 2), dtype=bool),
            origin_A=np.zeros(3),
            spacing_A=1.0,
            target_box_A=np.array([2.0, 2.0, 2.0]),
        )


@pytest.mark.parametrize(
    "pore_mask",
    [
        np.array([[[False, True], [True, False]]], dtype=bool),
        np.array([[[0, 1], [1, 0]]], dtype=np.uint8),
        np.array([[[0.0, 1.0], [1.0, 0.0]]], dtype=float),
    ],
)
def test_phase_grid_accepts_bool_or_exact_numeric_binary_masks(pore_mask: np.ndarray) -> None:
    grid = PhaseGrid(
        pore_mask=pore_mask,
        origin_A=np.zeros(3),
        spacing_A=0.5,
        target_box_A=np.array([1.0, 1.0, 0.5]),
    )

    assert grid.pore_mask.dtype == bool
    assert np.array_equal(grid.pore_mask, np.array([[[False, True], [True, False]]]))


@pytest.mark.parametrize(
    "bad_value",
    [2, -1, 0.5, np.nan, np.inf],
)
def test_phase_grid_rejects_direct_constructor_non_binary_mask_values(
    bad_value: float,
) -> None:
    pore_mask = np.array([[[0.0, 1.0], [bad_value, 0.0]]])

    with pytest.raises(ValueError, match="0 or 1"):
        PhaseGrid(
            pore_mask=pore_mask,
            origin_A=np.zeros(3),
            spacing_A=0.5,
            target_box_A=np.array([1.0, 1.0, 0.5]),
        )


def _write_phase_hdf5(
    path: Path,
    *,
    pore_mask: np.ndarray | None = None,
    attrs: dict[str, object] | None = None,
    compression: str | None = "gzip",
) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "pore_mask",
            data=np.zeros((1, 2, 2), dtype=np.uint8) if pore_mask is None else pore_mask,
            compression=compression,
        )
        default_attrs: dict[str, object] = {
            "schema_version": 1,
            "axis_order": "zyx",
            "periodic_axes": "x,y",
            "phase_encoding": '{"pore": 1, "semiconductor": 0}',
            "spacing_A": 0.5,
            "origin_A": [0.0, 0.0, 0.0],
            "target_box_A": [1.0, 1.0, 0.5],
            "pore_mask_compression": "gzip",
        }
        if attrs is not None:
            for key, value in attrs.items():
                if value is None:
                    default_attrs.pop(key, None)
                else:
                    default_attrs[key] = value
        for key, value in default_attrs.items():
            handle.attrs[key] = value


@pytest.mark.parametrize(
    ("pore_mask", "attrs", "compression", "message"),
    [
        (np.array([[[0, 2], [1, 0]]], dtype=np.uint8), None, "gzip", "0 or 1"),
        (np.zeros((1, 2, 2), dtype=np.uint8), {"target_box_A": [1.0, 1.0, 1.0]}, "gzip", "shape"),
        (None, {"axis_order": None}, "gzip", "missing"),
        (None, {"schema_version": 2}, "gzip", "schema_version"),
        (None, {"axis_order": "xyz"}, "gzip", "axis_order"),
        (None, {"phase_encoding": '{"pore": 2, "semiconductor": 0}'}, "gzip", "phase"),
        (None, {"periodic_axes": "x,z"}, "gzip", "periodic"),
        (None, {"pore_mask_compression": "gzip"}, None, "compression"),
        (None, {"pore_mask_compression": "lzf"}, "gzip", "compression"),
    ],
)
def test_phase_grid_hdf5_rejects_malformed_values_shape_attrs_and_compression(
    tmp_path: Path,
    pore_mask: np.ndarray | None,
    attrs: dict[str, object] | None,
    compression: str | None,
    message: str,
) -> None:
    path = tmp_path / "malformed.h5"
    _write_phase_hdf5(path, pore_mask=pore_mask, attrs=attrs, compression=compression)

    with pytest.raises(ValueError, match=message):
        PhaseGrid.read_hdf5(path)


def _sphere_builder(box_A: np.ndarray) -> Callable[[float], BuiltGeometry]:
    def build_at_linear_scale(scale: float) -> BuiltGeometry:
        unit = CompactUnit.sphere(
            unit_id="compact-0000",
            center_A=0.5 * box_A,
            radius_A=scale,
        )
        geometry = PoreGeometry([unit], box_A)
        return BuiltGeometry(
            geometry=geometry,
            units=[unit],
            realized_anchors_A=np.array([unit.anchor_A]),
            latent_to_realized_ids={"latent-0000": "compact-0000"},
        )

    return build_at_linear_scale


def test_porosity_scale_solver_rejects_unbracketed_target() -> None:
    with pytest.raises(ValueError, match="bracket"):
        solve_scale_for_porosity(
            _sphere_builder(np.array([10.0, 10.0, 10.0])),
            target_phi=0.8,
            tolerance=0.01,
            lower=0.5,
            upper=1.0,
            voxel_spacing_A=0.5,
        )


def test_porosity_scale_solver_rejects_tolerance_finer_than_grid_resolution() -> None:
    with pytest.raises(PorosityResolutionError, match="minimum resolvable porosity step"):
        solve_scale_for_porosity(
            _sphere_builder(np.array([4.0, 4.0, 4.0])),
            target_phi=0.1,
            tolerance=0.001,
            lower=0.5,
            upper=2.0,
            voxel_spacing_A=1.0,
        )


def test_porosity_scale_solver_reports_unbracketed_before_resolution_limit() -> None:
    with pytest.raises(ValueError, match="bracket"):
        solve_scale_for_porosity(
            _sphere_builder(np.array([4.0, 4.0, 4.0])),
            target_phi=0.8,
            tolerance=0.001,
            lower=0.5,
            upper=1.0,
            voxel_spacing_A=1.0,
        )


def test_porosity_scale_solver_uses_caller_voxel_spacing() -> None:
    _, _, grid = solve_scale_for_porosity(
        _sphere_builder(np.array([4.0, 4.0, 4.0])),
        target_phi=0.125,
        tolerance=0.02,
        lower=0.5,
        upper=2.0,
        voxel_spacing_A=1.0,
    )

    assert grid.spacing_A == 1.0
    assert grid.pore_mask.shape == (4, 4, 4)


def test_porosity_scale_solver_rejects_non_built_geometry_return() -> None:
    def build_bad(_scale: float) -> object:
        return object()

    with pytest.raises(TypeError, match="BuiltGeometry"):
        solve_scale_for_porosity(
            build_bad,
            target_phi=0.1,
            tolerance=0.01,
            lower=0.5,
            upper=2.0,
            voxel_spacing_A=0.5,
        )


def test_porosity_scale_solver_preserves_mixed_unit_common_linear_scale_records() -> None:
    box_A = np.array([12.0, 12.0, 12.0])
    base_channel_points = np.array([[2.0, 2.0, 2.0], [4.0, 2.0, 2.0]])

    def build_at_linear_scale(scale: float) -> BuiltGeometry:
        compact = CompactUnit.sphere(
            unit_id="compact-0000",
            center_A=np.array([8.0, 8.0, 8.0]),
            radius_A=0.7 * scale,
        )
        channel = ChannelUnit.from_polyline(
            unit_id="channel-0000",
            control_points_unwrapped_A=base_channel_points * scale,
            cross_radius_A=0.4 * scale,
            roughness=0.0,
            latent_eta=2.5,
            latent_tau=1.0,
        )
        geometry = PoreGeometry([compact, channel], box_A)
        return BuiltGeometry(
            geometry=geometry,
            units=[compact, channel],
            realized_anchors_A=np.vstack([compact.anchor_A, channel.anchor_A]),
            latent_to_realized_ids={
                "latent-0000": "compact-0000",
                "latent-0001": "channel-0000",
            },
        )

    scale, built, _ = solve_scale_for_porosity(
        build_at_linear_scale,
        target_phi=0.006,
        tolerance=0.003,
        lower=0.5,
        upper=2.0,
        voxel_spacing_A=0.5,
    )
    compact_record = built.units[0].to_record()
    channel_record = built.units[1].to_record()

    assert compact_record["realized_geometry"]["radii_A"] == [0.7 * scale] * 3
    assert channel_record["latent_parameters"]["eta"] == 2.5
    assert channel_record["latent_parameters"]["tau"] == 1.0
    assert np.isclose(channel_record["latent_parameters"]["cross_radius_A"], 0.4 * scale)
    assert np.isclose(channel_record["realized_geometry"]["arc_length_A"], 2.0 * scale)


def test_porosity_scale_solver_rebuilds_geometry_and_preserves_records() -> None:
    box_A = np.array([10.0, 10.0, 10.0])
    target_phi = 4.0 * np.pi * 2.0**3 / 3.0 / np.prod(box_A)
    latent_to_realized = {"latent-0000": "compact-0000"}
    requested_scales: list[float] = []

    def build_at_linear_scale(scale: float) -> BuiltGeometry:
        requested_scales.append(scale)
        unit = CompactUnit.sphere(
            unit_id="compact-0000",
            center_A=np.array([5.0, 5.0, 5.0]),
            radius_A=scale,
        )
        geometry = PoreGeometry([unit], box_A)
        return BuiltGeometry(
            geometry=geometry,
            units=[unit],
            realized_anchors_A=np.array([unit.anchor_A]),
            latent_to_realized_ids=latent_to_realized.copy(),
        )

    scale, built, grid = solve_scale_for_porosity(
        build_at_linear_scale,
        target_phi=float(target_phi),
        tolerance=0.01,
        lower=0.5,
        upper=3.0,
        voxel_spacing_A=0.25,
    )

    assert abs(grid.porosity - target_phi) <= 0.01
    assert 1.5 < scale < 2.5
    assert built.latent_to_realized_ids == latent_to_realized
    assert built.units[0].to_record()["realized_geometry"]["radii_A"] == [scale, scale, scale]
    assert requested_scales[0] == 0.5


def test_porosity_scale_solver_uses_unit_scale_probe_without_changing_phase() -> None:
    box_A = np.array([10.0, 10.0, 10.0])
    reference_built = _sphere_builder(box_A)(1.0)
    reference = voxelize_geometry(reference_built.geometry, box_A, 0.25)
    requested_scales: list[float] = []
    base_builder = _sphere_builder(box_A)

    def recording_builder(scale: float) -> BuiltGeometry:
        requested_scales.append(scale)
        return base_builder(scale)

    scale, _built, actual = solve_scale_for_porosity(
        recording_builder,
        target_phi=reference.porosity,
        tolerance=0.001,
        lower=0.5,
        upper=3.0,
        voxel_spacing_A=0.25,
    )

    assert scale == 1.0
    assert np.array_equal(actual.pore_mask, reference.pore_mask)
    assert requested_scales == [0.5, 1.0]
