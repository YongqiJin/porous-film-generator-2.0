from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import qmc

_SOBOL_POWER = 15
_MAX_SHAPE_ATTEMPTS = 16
_COMPACT_SMOOTH_FRACTION = 0.12
_CHANNEL_PROFILE_NODE_COUNT = 7
_CHANNEL_CURVE_SAMPLE_COUNT = 2049


@dataclass(frozen=True)
class CompactShapeProfile:
    shape_seed: int
    lobe_centers_local_A: np.ndarray
    lobe_radii_A: np.ndarray
    smooth_length_A: float
    envelope_radii_A: np.ndarray
    envelope_fill_fraction: float
    centroid_offset_A: float
    connected: bool

    @property
    def lobe_count(self) -> int:
        return int(self.lobe_centers_local_A.shape[0])


@dataclass(frozen=True)
class VariableRadiusChannelProfile:
    shape_seed: int
    control_points_local_A: np.ndarray
    radius_profile_s: np.ndarray
    radius_profile_A: np.ndarray
    equivalent_radius_A: float
    bend_count: int
    nonplanarity: float
    minimum_self_clearance_A: float


def generate_multilobe_profile(
    target_volume_A3: float,
    aspect_ratio: float,
    shape_seed: int,
) -> CompactShapeProfile:
    target_volume = _positive_float(target_volume_A3, "target_volume_A3")
    eta = _positive_float(aspect_ratio, "aspect_ratio")
    if eta < 1.0:
        raise ValueError("aspect_ratio must be at least 1")
    rng = np.random.default_rng(int(shape_seed))
    for _attempt in range(_MAX_SHAPE_ATTEMPTS):
        count = int(rng.choice(np.array([2, 3, 4]), p=np.array([0.25, 0.50, 0.25])))
        scalar_radii = rng.uniform(0.65, 1.15, count)
        centers = np.zeros((count, 3), dtype=float)
        parent_edges: list[tuple[int, int]] = []
        for index in range(1, count):
            parent = int(rng.integers(0, index))
            direction = _random_unit_vector(rng)
            overlap_fraction = float(rng.uniform(0.58, 0.82))
            distance = overlap_fraction * (scalar_radii[parent] + scalar_radii[index])
            centers[index] = centers[parent] + direction * distance
            parent_edges.append((parent, index))

        weights = scalar_radii**3
        centers -= np.average(centers, axis=0, weights=weights)
        centers = _align_cluster_principal_axis(centers, weights)
        hard_extents = np.max(np.abs(centers) + scalar_radii[:, np.newaxis], axis=0)
        if np.any(hard_extents <= 0.0):
            continue
        scale_x = 1.0 / hard_extents[0]
        scale_y = 1.0 / hard_extents[1]
        minimum_transverse_radius = float(
            np.min(scalar_radii * min(scale_x, scale_y))
        )
        smooth_length = _COMPACT_SMOOTH_FRACTION * minimum_transverse_radius
        smooth_expansion = smooth_length * np.log(float(count))
        target_hard_z = eta * (1.0 + smooth_expansion) - smooth_expansion
        if target_hard_z <= 0.0:
            continue
        scale_z = target_hard_z / hard_extents[2]
        affine = np.array([scale_x, scale_y, scale_z], dtype=float)
        scaled_centers = centers * affine
        scaled_radii = scalar_radii[:, np.newaxis] * affine[np.newaxis, :]
        for _ in range(16):
            smooth_length = _COMPACT_SMOOTH_FRACTION * float(
                np.min(scaled_radii[:, :2])
            )
            provisional = CompactShapeProfile(
                shape_seed=int(shape_seed),
                lobe_centers_local_A=scaled_centers,
                lobe_radii_A=scaled_radii,
                smooth_length_A=float(smooth_length),
                envelope_radii_A=np.max(
                    np.abs(scaled_centers) + scaled_radii,
                    axis=0,
                ),
                envelope_fill_fraction=0.0,
                centroid_offset_A=0.0,
                connected=_lobe_graph_connected(
                    scaled_centers,
                    scaled_radii,
                    parent_edges,
                ),
            )
            envelope = _multilobe_envelope_radii_A(provisional)
            if (
                np.isclose(envelope[0], envelope[1], rtol=1.0e-8)
                and np.isclose(envelope[2] / envelope[0], eta, rtol=1.0e-8)
            ):
                break
            correction = np.array(
                [1.0 / envelope[0], 1.0 / envelope[1], eta / envelope[2]],
                dtype=float,
            )
            scaled_centers *= correction
            scaled_radii *= correction
        smooth_length = _COMPACT_SMOOTH_FRACTION * float(
            np.min(scaled_radii[:, :2])
        )
        provisional = CompactShapeProfile(
            shape_seed=int(shape_seed),
            lobe_centers_local_A=scaled_centers,
            lobe_radii_A=scaled_radii,
            smooth_length_A=float(smooth_length),
            envelope_radii_A=np.max(
                np.abs(scaled_centers) + scaled_radii,
                axis=0,
            ),
            envelope_fill_fraction=0.0,
            centroid_offset_A=0.0,
            connected=_lobe_graph_connected(scaled_centers, scaled_radii, parent_edges),
        )
        envelope = _multilobe_envelope_radii_A(provisional)
        provisional = CompactShapeProfile(
            shape_seed=provisional.shape_seed,
            lobe_centers_local_A=provisional.lobe_centers_local_A,
            lobe_radii_A=provisional.lobe_radii_A,
            smooth_length_A=provisional.smooth_length_A,
            envelope_radii_A=envelope,
            envelope_fill_fraction=0.0,
            centroid_offset_A=0.0,
            connected=provisional.connected,
        )
        estimated_volume = estimate_multilobe_volume_A3(provisional)
        if estimated_volume <= 0.0 or not provisional.connected:
            continue
        fill_fraction = estimated_volume / (4.0 * np.pi * float(np.prod(envelope)) / 3.0)
        if not 0.50 <= fill_fraction <= 0.85:
            continue
        common_scale = float((target_volume / estimated_volume) ** (1.0 / 3.0))
        final_centers = scaled_centers * common_scale
        final_radii = scaled_radii * common_scale
        final_smooth = smooth_length * common_scale
        final_envelope = envelope * common_scale
        centroid = np.average(final_centers, axis=0, weights=np.prod(final_radii, axis=1))
        return CompactShapeProfile(
            shape_seed=int(shape_seed),
            lobe_centers_local_A=final_centers,
            lobe_radii_A=final_radii,
            smooth_length_A=float(final_smooth),
            envelope_radii_A=final_envelope,
            envelope_fill_fraction=float(fill_fraction),
            centroid_offset_A=float(np.linalg.norm(centroid)),
            connected=True,
        )
    raise ValueError("failed to generate connected multilobe profile within constraints")


def multilobe_sdf_local(points_local_A: np.ndarray, profile: CompactShapeProfile) -> np.ndarray:
    points = _as_points(points_local_A, "points_local_A")
    fields = []
    for center, radii in zip(
        profile.lobe_centers_local_A,
        profile.lobe_radii_A,
        strict=True,
    ):
        local = points - center
        implicit = np.sqrt(np.sum((local / radii) ** 2, axis=1))
        fields.append((implicit - 1.0) * float(np.min(radii)))
    values = np.vstack(fields)
    smooth = float(profile.smooth_length_A)
    if smooth <= 0.0 or values.shape[0] == 1:
        return np.min(values, axis=0)
    return -smooth * logsumexp(-values / smooth, axis=0)


def estimate_multilobe_volume_A3(profile: CompactShapeProfile) -> float:
    envelope = np.asarray(profile.envelope_radii_A, dtype=float)
    sampler = qmc.Sobol(d=3, scramble=False)
    samples = sampler.random_base2(_SOBOL_POWER)
    points = (2.0 * samples - 1.0) * envelope
    occupied_fraction = float(np.mean(multilobe_sdf_local(points, profile) < 0.0))
    return occupied_fraction * 8.0 * float(np.prod(envelope))


def generate_variable_radius_channel_profile(
    target_volume_A3: float,
    eta: float,
    tau: float,
    shape_seed: int,
    *,
    target_equivalent_diameter_A: float | None = None,
) -> VariableRadiusChannelProfile:
    target_volume = _positive_float(target_volume_A3, "target_volume_A3")
    target_diameter = (
        None
        if target_equivalent_diameter_A is None
        else _positive_float(target_equivalent_diameter_A, "target_equivalent_diameter_A")
    )
    eta_value = _positive_float(eta, "eta")
    tau_value = _positive_float(tau, "tau")
    if eta_value < 1.0 or tau_value < 1.0:
        raise ValueError("eta and tau must be at least 1")
    rng = np.random.default_rng(int(shape_seed))
    for _attempt in range(_MAX_SHAPE_ATTEMPTS):
        profile_s, relative_radii = _generate_relative_radius_profile(rng)
        dense_s = np.linspace(0.0, 1.0, 4097)
        relative_dense = PchipInterpolator(profile_s, relative_radii)(dense_s)
        mean_q2 = float(np.trapezoid(relative_dense**2, dense_s))
        cap_factor = (2.0 / 3.0) * (
            float(relative_radii[0]) ** 3 + float(relative_radii[-1]) ** 3
        )
        denominator = np.pi * (2.0 * eta_value * mean_q2 + cap_factor)
        if target_diameter is None:
            equivalent_radius = float((target_volume / denominator) ** (1.0 / 3.0))
            effective_diameter = 2.0 * equivalent_radius * np.sqrt(mean_q2)
        else:
            equivalent_radius = float(target_diameter / (2.0 * np.sqrt(mean_q2)))
            effective_diameter = float(target_diameter)
        target_arc_length = eta_value * effective_diameter
        end_distance = target_arc_length / tau_value
        controls = _generate_multibend_controls(
            rng,
            end_distance_A=end_distance,
            target_arc_length_A=target_arc_length,
            require_multibend=tau_value > 1.05,
        )
        centerline = sample_channel_centerline(controls, sample_count=1025)
        arc_length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())
        end_distance_actual = float(np.linalg.norm(centerline[-1] - centerline[0]))
        if not np.isclose(arc_length / effective_diameter, eta_value, rtol=0.01):
            continue
        if not np.isclose(arc_length / end_distance_actual, tau_value, rtol=0.01):
            continue
        profile = VariableRadiusChannelProfile(
            shape_seed=int(shape_seed),
            control_points_local_A=controls,
            radius_profile_s=profile_s,
            radius_profile_A=relative_radii * equivalent_radius,
            equivalent_radius_A=equivalent_radius,
            bend_count=_bend_count(controls),
            nonplanarity=_curve_nonplanarity(centerline),
            minimum_self_clearance_A=0.0,
        )
        clearance = _minimum_channel_self_clearance_A(profile)
        profile = VariableRadiusChannelProfile(
            shape_seed=profile.shape_seed,
            control_points_local_A=profile.control_points_local_A,
            radius_profile_s=profile.radius_profile_s,
            radius_profile_A=profile.radius_profile_A,
            equivalent_radius_A=profile.equivalent_radius_A,
            bend_count=profile.bend_count,
            nonplanarity=profile.nonplanarity,
            minimum_self_clearance_A=float(clearance),
        )
        if tau_value > 1.05 and profile.bend_count < 2:
            continue
        if tau_value > 1.05 and profile.nonplanarity <= 1.0e-3:
            continue
        if clearance < 0.0:
            continue
        return profile
    raise ValueError("failed to generate non-self-intersecting variable-radius channel")


def sample_channel_centerline(
    control_points_local_A: np.ndarray,
    *,
    sample_count: int,
) -> np.ndarray:
    controls = _as_points(control_points_local_A, "control_points_local_A")
    if controls.shape[0] < 2:
        raise ValueError("control_points_local_A must contain at least two points")
    chord_lengths = np.linalg.norm(np.diff(controls, axis=0), axis=1)
    parameters = np.concatenate(([0.0], np.cumsum(chord_lengths)))
    parameters /= float(parameters[-1])
    target = np.linspace(0.0, 1.0, int(sample_count))
    if controls.shape[0] == 2:
        return (1.0 - target[:, np.newaxis]) * controls[0] + target[:, np.newaxis] * controls[1]
    splines = [CubicSpline(parameters, controls[:, axis]) for axis in range(3)]
    return np.column_stack([spline(target) for spline in splines])


def sample_radius_profile(
    profile: VariableRadiusChannelProfile,
    *,
    sample_count: int,
) -> np.ndarray:
    target = np.linspace(0.0, 1.0, int(sample_count))
    return np.asarray(
        PchipInterpolator(profile.radius_profile_s, profile.radius_profile_A)(target),
        dtype=float,
    )


def estimate_variable_channel_volume_A3(profile: VariableRadiusChannelProfile) -> float:
    centerline = sample_channel_centerline(profile.control_points_local_A, sample_count=4097)
    arc_length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())
    radii = sample_radius_profile(profile, sample_count=4097)
    s = np.linspace(0.0, 1.0, radii.size)
    mean_radius_squared = float(np.trapezoid(radii**2, s))
    body = np.pi * arc_length * mean_radius_squared
    caps = (2.0 * np.pi / 3.0) * (float(radii[0]) ** 3 + float(radii[-1]) ** 3)
    return float(body + caps)


def _generate_relative_radius_profile(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.linspace(0.0, 1.0, _CHANNEL_PROFILE_NODE_COUNT)
    dense = np.linspace(0.0, 1.0, 4097)
    for _attempt in range(_MAX_SHAPE_ATTEMPTS):
        coefficients = rng.normal(size=3)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=3)
        raw = sum(
            coefficient * np.sin(2.0 * np.pi * mode * nodes + phase)
            for mode, (coefficient, phase) in enumerate(
                zip(coefficients, phases, strict=True),
                start=1,
            )
        )
        raw_dense = PchipInterpolator(nodes, raw)(dense)
        std = float(np.std(raw_dense))
        if std <= 1.0e-12:
            continue
        target_cv = float(rng.uniform(0.18, 0.26))
        relative = 1.0 + target_cv * raw / std
        dense_relative = PchipInterpolator(nodes, relative)(dense)
        rms = float(np.sqrt(np.trapezoid(dense_relative**2, dense)))
        relative /= rms
        dense_relative = PchipInterpolator(nodes, relative)(dense)
        cv = float(np.std(dense_relative) / np.mean(dense_relative))
        if (
            0.15 <= cv <= 0.30
            and float(np.min(dense_relative)) >= 0.60
            and float(np.max(dense_relative)) <= 1.45
            and float(np.min(dense_relative)) < 0.85
            and float(np.max(dense_relative)) > 1.15
        ):
            return nodes, relative
    raise ValueError("failed to generate radius profile within constraints")


def _generate_multibend_controls(
    rng: np.random.Generator,
    *,
    end_distance_A: float,
    target_arc_length_A: float,
    require_multibend: bool,
) -> np.ndarray:
    s = np.linspace(0.0, 1.0, _CHANNEL_PROFILE_NODE_COUNT)
    if target_arc_length_A <= end_distance_A * (1.0 + 1.0e-10):
        return np.column_stack([end_distance_A * s, np.zeros_like(s), np.zeros_like(s)])
    for _attempt in range(_MAX_SHAPE_ATTEMPTS):
        coefficients_y = rng.normal(size=3)
        coefficients_z = rng.normal(size=3)
        base_y = sum(
            coefficient * np.sin(np.pi * mode * s)
            for mode, coefficient in enumerate(coefficients_y, start=1)
        )
        base_z = sum(
            coefficient * np.sin(np.pi * mode * s)
            for mode, coefficient in enumerate(coefficients_z, start=1)
        )
        scale = max(float(np.max(np.abs(base_y))), float(np.max(np.abs(base_z))), 1.0e-12)
        base_y /= scale
        base_z /= scale

        def controls(
            amplitude_A: float,
            lateral_y: np.ndarray = base_y,
            lateral_z: np.ndarray = base_z,
        ) -> np.ndarray:
            return np.column_stack(
                [end_distance_A * s, amplitude_A * lateral_y, amplitude_A * lateral_z]
            )

        low = 0.0
        high = max(end_distance_A, 1.0)
        while _curve_arc_length(controls(high)) < target_arc_length_A:
            high *= 2.0
            if high > 1.0e6 * max(end_distance_A, 1.0):
                break
        else:
            for _ in range(56):
                middle = 0.5 * (low + high)
                if _curve_arc_length(controls(middle)) < target_arc_length_A:
                    low = middle
                else:
                    high = middle
            result = controls(0.5 * (low + high))
            if not require_multibend:
                return result
            if _bend_count(result) >= 2 and _curve_nonplanarity(
                sample_channel_centerline(result, sample_count=513)
            ) > 1.0e-3:
                return result
    raise ValueError("failed to generate nonplanar multibend controls")


def _curve_arc_length(controls: np.ndarray) -> float:
    samples = sample_channel_centerline(controls, sample_count=_CHANNEL_CURVE_SAMPLE_COUNT)
    return float(np.linalg.norm(np.diff(samples, axis=0), axis=1).sum())


def _bend_count(controls: np.ndarray) -> int:
    vectors = np.diff(controls, axis=0)
    vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    angles = np.arccos(np.clip(np.sum(vectors[:-1] * vectors[1:], axis=1), -1.0, 1.0))
    return int(np.count_nonzero(angles > 0.08))


def _curve_nonplanarity(centerline: np.ndarray) -> float:
    centered = centerline - np.mean(centerline, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    return float(singular[-1] / max(singular[0], 1.0e-12))


def _minimum_channel_self_clearance_A(profile: VariableRadiusChannelProfile) -> float:
    centerline = sample_channel_centerline(profile.control_points_local_A, sample_count=129)
    radii = sample_radius_profile(profile, sample_count=129)
    segment_lengths = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    minimum = np.inf
    for first in range(centerline.shape[0] - 1):
        for second in range(first + 2, centerline.shape[0] - 1):
            arc_separation = cumulative[second] - cumulative[first + 1]
            local_support = 3.0 * max(radii[first], radii[first + 1], radii[second], radii[second + 1])
            if arc_separation <= local_support:
                continue
            distance = _segment_segment_distance(
                centerline[first],
                centerline[first + 1],
                centerline[second],
                centerline[second + 1],
            )
            clearance = distance - max(radii[first], radii[first + 1]) - max(
                radii[second], radii[second + 1]
            )
            minimum = min(minimum, float(clearance))
    return float(minimum if np.isfinite(minimum) else 0.0)


def _segment_segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    u = q1 - p1
    v = q2 - p2
    w = p1 - p2
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denominator = a * c - b * b
    if denominator <= 1.0e-15:
        first_parameter = 0.0
        second_parameter = np.clip(e / max(c, 1.0e-15), 0.0, 1.0)
    else:
        first_parameter = np.clip((b * e - c * d) / denominator, 0.0, 1.0)
        second_parameter = np.clip((a * e - b * d) / denominator, 0.0, 1.0)
    first_parameter = np.clip((b * second_parameter - d) / max(a, 1.0e-15), 0.0, 1.0)
    second_parameter = np.clip((b * first_parameter + e) / max(c, 1.0e-15), 0.0, 1.0)
    return float(np.linalg.norm((p1 + first_parameter * u) - (p2 + second_parameter * v)))


def _multilobe_envelope_radii_A(
    profile: CompactShapeProfile,
) -> np.ndarray:
    extents = np.zeros(3, dtype=float)
    maximum_radius = float(np.max(profile.lobe_radii_A))
    for axis in range(3):
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

                upper = maximum_radius + float(profile.smooth_length_A)
                while field(upper) < 0.0:
                    upper *= 2.0
                distance = brentq(field, 0.0, upper)
                extents[axis] = max(
                    extents[axis],
                    abs(float(boundary[axis] + sign * distance)),
                )
    return extents


def _align_cluster_principal_axis(centers: np.ndarray, weights: np.ndarray) -> np.ndarray:
    covariance = np.einsum("i,ij,ik->jk", weights, centers, centers) / float(np.sum(weights))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    basis = np.column_stack(
        [eigenvectors[:, order[1]], eigenvectors[:, order[0]], eigenvectors[:, order[2]]]
    )
    if np.linalg.det(basis) < 0.0:
        basis[:, 0] *= -1.0
    return centers @ basis


def _lobe_graph_connected(
    centers: np.ndarray,
    radii: np.ndarray,
    parent_edges: list[tuple[int, int]],
) -> bool:
    if centers.shape[0] <= 1:
        return True
    scale = radii[0] / max(float(np.mean(radii[0])), 1.0e-12)
    for first, second in parent_edges:
        normalized_delta = (centers[first] - centers[second]) / scale
        first_radius = float(np.mean(radii[first] / scale))
        second_radius = float(np.mean(radii[second] / scale))
        if np.linalg.norm(normalized_delta) > first_radius + second_radius:
            return False
    return True


def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return np.array([1.0, 0.0, 0.0])
    return vector / norm


def _positive_float(value: float, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return parsed


def _as_points(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape (n, 3) with finite values")
    return array