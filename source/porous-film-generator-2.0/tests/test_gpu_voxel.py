from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from porous_film.geometry import ChannelUnit, CompactUnit, PoreGeometry
from porous_film.voxel import voxelize_geometry


def _mixed_geometry() -> PoreGeometry:
    compact = CompactUnit(
        unit_id="compact-gpu",
        center_A=np.array([0.35, 3.0, 3.0]),
        radii_A=np.array([1.25, 0.95, 1.4]),
        orientation=Rotation.from_euler("zyx", [12.0, -8.0, 5.0], degrees=True),
        exponent=2.0,
        roughness=0.015,
        shape_model="multilobe-v1",
        shape_seed=17,
        lobe_centers_local_A=np.array(
            [
                [-0.35, 0.0, -0.25],
                [0.25, 0.1, 0.2],
                [0.0, -0.2, 0.45],
            ]
        ),
        lobe_radii_A=np.array(
            [
                [0.75, 0.65, 0.8],
                [0.7, 0.75, 0.65],
                [0.6, 0.7, 0.75],
            ]
        ),
        smooth_length_A=0.08,
        envelope_fill_fraction=0.7,
        centroid_offset_A=0.0,
        lobes_connected=True,
    )
    channel = ChannelUnit.from_polyline(
        unit_id="channel-gpu",
        control_points_unwrapped_A=np.array(
            [
                [1.0, 1.2, 1.1],
                [2.5, 1.8, 2.8],
                [4.2, 4.5, 3.7],
                [6.4, 5.4, 5.2],
            ]
        ),
        cross_radius_A=0.55,
        roughness=0.02,
        shape_model="variable-radius-spline-v1",
        shape_seed=23,
        radius_profile_s=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        radius_profile_A=np.array([0.45, 0.62, 0.5, 0.68, 0.48]),
    )
    return PoreGeometry([compact, channel], np.array([7.0, 7.0, 7.0]))


@pytest.mark.parametrize(
    ("device_name", "free_memory_gib", "expected"),
    [
        ("NVIDIA A100-SXM4-80GB", 79.0, (4_000_000, 64)),
        ("NVIDIA A100-SXM4-40GB", 39.0, (2_000_000, 64)),
        ("NVIDIA GeForce RTX 4090", 23.0, (1_000_000, 64)),
        ("generic CUDA device", 8.0, (262_144, 32)),
    ],
)
def test_gpu_workload_defaults_cover_a100_and_rtx4090(
    device_name: str,
    free_memory_gib: float,
    expected: tuple[int, int],
) -> None:
    from porous_film.voxel.cupy_backend import recommended_gpu_workload

    assert (
        recommended_gpu_workload(
            device_name=device_name,
            free_memory_bytes=int(free_memory_gib * 1024**3),
        )
        == expected
    )


def test_cuda_voxelization_uses_automatic_chunking_when_unspecified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from porous_film.voxel import cupy_backend

    captured: dict[str, object] = {}

    def fake_voxelize(*args, **kwargs):
        captured.update(kwargs)
        return np.zeros((2, 2, 2), dtype=bool)

    monkeypatch.setenv("POROUS_FILM_VOXEL_BACKEND", "cuda")
    monkeypatch.setattr(cupy_backend, "cuda_backend_available", lambda: True)
    monkeypatch.setattr(cupy_backend, "voxelize_geometry_cupy", fake_voxelize)

    voxelize_geometry(PoreGeometry([], np.array([2.0, 2.0, 2.0])), np.array([2.0] * 3), 1.0)

    assert captured["max_points_per_chunk"] is None


def test_gpu_workload_environment_overrides_automatic_values_and_api_caps_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from porous_film.voxel.cupy_backend import _resolve_gpu_workload

    monkeypatch.setenv("POROUS_FILM_GPU_MAX_POINTS_PER_CHUNK", "1500000")
    monkeypatch.setenv("POROUS_FILM_GPU_SEGMENT_BATCH_SIZE", "48")

    assert _resolve_gpu_workload(
        automatic=(4_000_000, 64),
        max_points_per_chunk=1_000_000,
    ) == (1_000_000, 48)


def test_cupy_sdf_matches_numpy_reference_on_cuda() -> None:
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA is not available")
    from porous_film.voxel.cupy_backend import evaluate_sdf_cupy

    geometry = _mixed_geometry()
    rng = np.random.default_rng(20260902)
    random_points = rng.random((2048, 3)) * geometry.target_box_A
    boundary_points = np.array(
        [
            [0.01, 3.0, 3.0],
            [6.99, 3.0, 3.0],
            [3.0, 0.01, 3.0],
            [3.0, 6.99, 3.0],
        ]
    )
    points = np.vstack([random_points, boundary_points])

    expected = geometry.sdf(points)
    actual = evaluate_sdf_cupy(geometry, points, device=0)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-9)
    assert np.array_equal(actual < 0.0, expected < 0.0)


def test_cuda_voxelization_matches_cpu_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA is not available")

    geometry = _mixed_geometry()
    monkeypatch.setenv("POROUS_FILM_VOXEL_BACKEND", "cpu")
    cpu = voxelize_geometry(geometry, geometry.target_box_A, 0.25)
    monkeypatch.setenv("POROUS_FILM_VOXEL_BACKEND", "cuda")
    gpu = voxelize_geometry(
        geometry,
        geometry.target_box_A,
        0.25,
        max_points_per_chunk=997,
    )

    assert np.array_equal(gpu.pore_mask, cpu.pore_mask)
    assert gpu.porosity == cpu.porosity


def test_cuda_unit_masks_on_final_pore_candidates_match_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA is not available")
    from porous_film.voxel.cupy_backend import voxelize_unit_masks_cupy

    geometry = _mixed_geometry()
    monkeypatch.setenv("POROUS_FILM_VOXEL_BACKEND", "cpu")
    combined = voxelize_geometry(geometry, geometry.target_box_A, 0.25)
    expected = tuple(
        voxelize_geometry(
            PoreGeometry([unit], geometry.target_box_A),
            geometry.target_box_A,
            0.25,
        ).pore_mask
        for unit in geometry.units
    )
    counts_xyz = np.rint(geometry.target_box_A / 0.25).astype(int)

    actual = voxelize_unit_masks_cupy(
        geometry,
        counts_xyz=counts_xyz,
        spacing_A=0.25,
        candidate_mask=combined.pore_mask,
        device=0,
    )

    assert len(actual) == len(expected)
    for actual_mask, expected_mask in zip(actual, expected, strict=True):
        assert np.array_equal(actual_mask, expected_mask)


def test_requested_cuda_backend_fails_clearly_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cupy")
    from porous_film.voxel import cupy_backend

    geometry = PoreGeometry([], np.array([2.0, 2.0, 2.0]))
    monkeypatch.setenv("POROUS_FILM_VOXEL_BACKEND", "cuda")
    monkeypatch.setattr(cupy_backend, "cuda_backend_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA voxel backend was requested"):
        voxelize_geometry(geometry, geometry.target_box_A, 1.0)
