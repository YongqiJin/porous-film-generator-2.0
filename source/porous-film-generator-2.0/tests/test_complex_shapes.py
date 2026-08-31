import numpy as np

from porous_film.geometry.complex_shapes import (
    estimate_multilobe_volume_A3,
    estimate_variable_channel_volume_A3,
    generate_multilobe_profile,
    generate_variable_radius_channel_profile,
    multilobe_sdf_local,
    sample_channel_centerline,
    sample_radius_profile,
)


def test_multilobe_profile_is_deterministic_and_seed_sensitive() -> None:
    first = generate_multilobe_profile(2_500.0, 2.4, 12345)
    repeated = generate_multilobe_profile(2_500.0, 2.4, 12345)
    different = generate_multilobe_profile(2_500.0, 2.4, 12346)

    assert np.array_equal(first.lobe_centers_local_A, repeated.lobe_centers_local_A)
    assert np.array_equal(first.lobe_radii_A, repeated.lobe_radii_A)
    assert first.smooth_length_A == repeated.smooth_length_A
    assert not np.array_equal(first.lobe_centers_local_A, different.lobe_centers_local_A)


def test_multilobe_profile_preserves_volume_envelope_and_connectivity() -> None:
    target_volume = 4_000.0
    target_eta = 2.8
    profile = generate_multilobe_profile(target_volume, target_eta, 8811)

    assert 2 <= profile.lobe_count <= 4
    assert profile.lobe_centers_local_A.shape == (profile.lobe_count, 3)
    assert profile.lobe_radii_A.shape == (profile.lobe_count, 3)
    assert np.all(profile.lobe_radii_A > 0.0)
    assert profile.connected
    assert profile.centroid_offset_A < 1.0e-10
    assert 0.50 <= profile.envelope_fill_fraction <= 0.85
    assert np.isclose(profile.envelope_radii_A[0], profile.envelope_radii_A[1])
    assert np.isclose(
        profile.envelope_radii_A[2] / profile.envelope_radii_A[0],
        target_eta,
        rtol=0.05,
    )
    realized_volume = estimate_multilobe_volume_A3(profile)
    assert np.isclose(realized_volume, target_volume, rtol=0.03)


def test_variable_radius_channel_profile_is_deterministic_and_complex() -> None:
    first = generate_variable_radius_channel_profile(8_000.0, 7.0, 1.45, 7711)
    repeated = generate_variable_radius_channel_profile(8_000.0, 7.0, 1.45, 7711)
    different = generate_variable_radius_channel_profile(8_000.0, 7.0, 1.45, 7712)

    assert np.array_equal(first.control_points_local_A, repeated.control_points_local_A)
    assert np.array_equal(first.radius_profile_A, repeated.radius_profile_A)
    assert not np.array_equal(first.control_points_local_A, different.control_points_local_A)
    assert first.control_points_local_A.shape == (7, 3)
    assert first.radius_profile_s.shape == (7,)
    assert first.radius_profile_A.shape == (7,)
    assert first.bend_count >= 2
    assert first.nonplanarity > 0.0
    assert first.minimum_self_clearance_A >= 0.0


def test_variable_radius_channel_profile_preserves_eta_tau_and_volume() -> None:
    target_volume = 12_000.0
    target_eta = 8.0
    target_tau = 1.55
    profile = generate_variable_radius_channel_profile(
        target_volume,
        target_eta,
        target_tau,
        9117,
    )
    centerline = sample_channel_centerline(profile.control_points_local_A, sample_count=4097)
    arc_length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())
    end_distance = float(np.linalg.norm(centerline[-1] - centerline[0]))
    radii = sample_radius_profile(profile, sample_count=4097)
    radius_cv = float(np.std(radii) / np.mean(radii))

    assert 0.15 <= radius_cv <= 0.30
    assert np.min(radii) >= 0.60 * profile.equivalent_radius_A
    assert np.max(radii) <= 1.45 * profile.equivalent_radius_A
    assert np.min(radii) < 0.85 * profile.equivalent_radius_A
    assert np.max(radii) > 1.15 * profile.equivalent_radius_A
    assert np.isclose(arc_length / (2.0 * profile.equivalent_radius_A), target_eta, rtol=0.01)
    assert np.isclose(arc_length / end_distance, target_tau, rtol=0.01)
    assert np.isclose(
        estimate_variable_channel_volume_A3(profile),
        target_volume,
        rtol=0.03,
    )


def test_tau_one_channel_remains_straight_but_keeps_variable_radius_profile() -> None:
    profile = generate_variable_radius_channel_profile(800.0, 2.0, 1.0, 649515172919821594)

    centerline = sample_channel_centerline(profile.control_points_local_A, sample_count=1025)
    arc_length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())
    end_distance = float(np.linalg.norm(centerline[-1] - centerline[0]))

    assert profile.bend_count == 0
    assert np.isclose(arc_length / end_distance, 1.0, atol=1.0e-10)
    assert float(np.std(profile.radius_profile_A) / np.mean(profile.radius_profile_A)) >= 0.15

def test_tau_just_above_one_is_valid_without_forced_multibend() -> None:
    profile = generate_variable_radius_channel_profile(8_000.0, 5.0, 1.001, 314159)
    centerline = sample_channel_centerline(profile.control_points_local_A, sample_count=2049)
    arc_length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())
    end_distance = float(np.linalg.norm(centerline[-1] - centerline[0]))

    assert np.isclose(arc_length / end_distance, 1.001, rtol=0.01)


def test_multilobe_recorded_envelope_tracks_actual_smooth_union_bounds() -> None:
    from scipy.optimize import brentq

    profile = generate_multilobe_profile(4_000.0, 2.2, 147)
    realized = []
    for axis in range(3):
        extent = 0.0
        for center, radii in zip(
            profile.lobe_centers_local_A,
            profile.lobe_radii_A,
            strict=True,
        ):
            for sign in (-1.0, 1.0):
                boundary = center.copy()
                boundary[axis] += sign * radii[axis]

                def field(
                    distance: float,
                    origin: np.ndarray = boundary,
                    direction_axis: int = axis,
                    direction_sign: float = sign,
                ) -> float:
                    point = origin.copy()
                    point[direction_axis] += direction_sign * distance
                    return float(multilobe_sdf_local(point[np.newaxis, :], profile)[0])

                upper = float(np.max(profile.lobe_radii_A) + profile.smooth_length_A)
                while field(upper) < 0.0:
                    upper *= 2.0
                distance = brentq(field, 0.0, upper)
                extent = max(extent, abs(float(boundary[axis] + sign * distance)))
        realized.append(extent)
    realized = np.asarray(realized)

    assert np.allclose(profile.envelope_radii_A, realized, rtol=0.01)
    assert np.isclose(realized[0], realized[1], rtol=0.01)
    assert np.isclose(realized[2] / realized[0], 2.2, rtol=0.05)

