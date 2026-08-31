from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import blake2b
from itertools import pairwise
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp

from porous_film.centers import CenterSeedPlan
from porous_film.config import GeneratorConfig
from porous_film.distributions import allocate_largest_remainder, stratified_sample
from porous_film.geometry.complex_shapes import (
    CompactShapeProfile,
    generate_multilobe_profile,
    generate_variable_radius_channel_profile,
    multilobe_sdf_local,
)

_SUPERELLIPSOID_EXPONENT = 2.0
_SMOOTH_UNION_SHARPNESS = 32.0
_ROUGHNESS_MODE_COUNT = 4
_CHANNEL_REALIZATION_TOLERANCE = 1e-4
_SPLINE_CHORD_TOLERANCE_FRACTION = 0.10
_SPLINE_MAX_TURN_RAD = 0.15
_SPLINE_MAX_REFINEMENT_DEPTH = 16


@dataclass(frozen=True)
class OrientationSample:
    rotation: Rotation
    theta_rad: float
    phi_rad: float
    theta_xz_deg: float | None = None
    theta_xy_deg: float | None = None
    paired_component_index: int | None = None


class PoreUnit(ABC):
    unit_id: str
    anchor_A: np.ndarray

    @property
    @abstractmethod
    def sdf_lipschitz_bound(self) -> float:
        """Return a conservative global Lipschitz bound for this unit's SDF."""

    @abstractmethod
    def sdf(self, points_A: np.ndarray) -> np.ndarray:
        """Return signed distance values in Angstrom for points with shape ``(n, 3)``."""

    @abstractmethod
    def to_record(self) -> dict[str, Any]:
        """Return an auditable unit record with separate latent and realized values."""


@dataclass(frozen=True)
class CompactUnit(PoreUnit):
    unit_id: str
    center_A: np.ndarray
    radii_A: np.ndarray
    orientation: Rotation
    exponent: float
    roughness: float = 0.0
    latent_theta_rad: float | None = None
    latent_phi_rad: float | None = None
    latent_theta_xz_deg: float | None = None
    latent_theta_xy_deg: float | None = None
    latent_orientation_component_index: int | None = None
    latent_equivalent_diameter_A: float | None = None
    latent_curvature_fluctuation_target: float | None = None
    latent_target_volume_A3: float | None = None
    latent_eta: float | None = None
    shape_model: str | None = None
    shape_seed: int | None = None
    lobe_centers_local_A: np.ndarray | None = None
    lobe_radii_A: np.ndarray | None = None
    smooth_length_A: float | None = None
    envelope_fill_fraction: float | None = None
    centroid_offset_A: float | None = None
    lobes_connected: bool | None = None

    @staticmethod
    def sphere(unit_id: str, center_A: np.ndarray, radius_A: float) -> CompactUnit:
        radius = _positive_float(radius_A, "radius_A")
        center = _as_vector(center_A, "center_A")
        return CompactUnit(
            unit_id=unit_id,
            center_A=center,
            radii_A=np.array([radius, radius, radius], dtype=float),
            orientation=Rotation.identity(),
            exponent=_SUPERELLIPSOID_EXPONENT,
        )

    @property
    def anchor_A(self) -> np.ndarray:
        return self.center_A.copy()

    @property
    def is_multilobe(self) -> bool:
        return (
            self.shape_model == "multilobe-v1"
            and self.lobe_centers_local_A is not None
            and self.lobe_radii_A is not None
            and self.smooth_length_A is not None
        )

    def _multilobe_profile(self) -> CompactShapeProfile:
        if not self.is_multilobe:
            raise ValueError("compact unit does not contain a multilobe profile")
        return CompactShapeProfile(
            shape_seed=int(self.shape_seed if self.shape_seed is not None else 0),
            lobe_centers_local_A=np.asarray(self.lobe_centers_local_A, dtype=float),
            lobe_radii_A=np.asarray(self.lobe_radii_A, dtype=float),
            smooth_length_A=float(self.smooth_length_A),
            envelope_radii_A=np.asarray(self.radii_A, dtype=float),
            envelope_fill_fraction=float(self.envelope_fill_fraction or 0.0),
            centroid_offset_A=float(self.centroid_offset_A or 0.0),
            connected=bool(self.lobes_connected),
        )

    @property
    def sdf_lipschitz_bound(self) -> float:
        radii = np.asarray(self.radii_A, dtype=float)
        if radii.shape != (3,) or not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
            return np.inf
        if self.is_multilobe:
            lobe_radii = np.asarray(self.lobe_radii_A, dtype=float)
            base_bound = max(
                _compact_base_lipschitz_bound(values, float(self.exponent))
                for values in lobe_radii
            )
        else:
            base_bound = _compact_base_lipschitz_bound(radii, float(self.exponent))
        roughness_bound = _roughness_lipschitz_bound(
            unit_id=self.unit_id,
            roughness=float(self.roughness),
            length_scale_A=float(np.min(radii)),
            coordinate_derivative_norm=float(np.linalg.norm(1.0 / radii)),
        )
        return base_bound + roughness_bound

    def sdf(self, points_A: np.ndarray) -> np.ndarray:
        points = _as_points(points_A, "points_A")
        local = self.orientation.inv().apply(points - self.center_A)
        if self.is_multilobe:
            base = multilobe_sdf_local(local, self._multilobe_profile())
        elif self.exponent == 2.0 and np.allclose(self.radii_A, self.radii_A[0]):
            base = np.linalg.norm(local, axis=1) - float(self.radii_A[0])
        else:
            radial_xy = np.sqrt(
                (local[:, 0] / self.radii_A[0]) ** 2
                + (local[:, 1] / self.radii_A[1]) ** 2
            )
            axial_z = np.abs(local[:, 2] / self.radii_A[2])
            implicit = (radial_xy**self.exponent + axial_z**self.exponent) ** (
                1.0 / self.exponent
            )
            base = (implicit - 1.0) * float(np.min(self.radii_A))
        return base - _roughness_perturbation(
            local_coordinates=local / np.maximum(self.radii_A, 1e-12),
            unit_id=self.unit_id,
            roughness=self.roughness,
            length_scale_A=float(np.min(self.radii_A)),
        )

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": 2 if self.is_multilobe else 1,
            "unit_id": self.unit_id,
            "kind": "compact",
            "latent_parameters": {
                "radius_A": float(self.radii_A[0]),
                "radii_A": self.radii_A.tolist(),
                "superellipsoid_exponent": float(self.exponent),
                "roughness": float(self.roughness),
                "target_volume_A3": None
                if self.latent_target_volume_A3 is None
                else float(self.latent_target_volume_A3),
                "orientation": _latent_orientation_record(
                    self.latent_theta_rad,
                    self.latent_phi_rad,
                    theta_xz_deg=self.latent_theta_xz_deg,
                    theta_xy_deg=self.latent_theta_xy_deg,
                    paired_component_index=self.latent_orientation_component_index,
                ),
                "equivalent_diameter_A": None
                if self.latent_equivalent_diameter_A is None
                else float(self.latent_equivalent_diameter_A),
                "curvature_fluctuation_target": None
                if self.latent_curvature_fluctuation_target is None
                else float(self.latent_curvature_fluctuation_target),
            },
            "realized_geometry": {
                "anchor_A": self.anchor_A.tolist(),
                "center_A": self.center_A.tolist(),
                "radii_A": self.radii_A.tolist(),
                "orientation": _realized_orientation_record(
                    self.orientation,
                    local_axis=np.eye(3)[int(np.argmax(self.radii_A))],
                ),
                "orientation_quaternion_xyzw": self.orientation.as_quat().tolist(),
            },
        }
        if self.is_multilobe:
            record["shape_model"] = self.shape_model
            record["shape_seed"] = int(self.shape_seed)
            record["latent_parameters"]["eta"] = float(
                self.latent_eta
                if self.latent_eta is not None
                else self.radii_A[2] / self.radii_A[0]
            )
            record["realized_geometry"].update(
                {
                    "lobe_centers_local_A": np.asarray(
                        self.lobe_centers_local_A, dtype=float
                    ).tolist(),
                    "lobe_radii_A": np.asarray(self.lobe_radii_A, dtype=float).tolist(),
                    "smooth_length_A": float(self.smooth_length_A),
                    "envelope_radii_A": self.radii_A.tolist(),
                    "envelope_fill_fraction": float(self.envelope_fill_fraction),
                    "centroid_offset_A": float(self.centroid_offset_A),
                    "lobes_connected": bool(self.lobes_connected),
                    "lobe_count": int(np.asarray(self.lobe_centers_local_A).shape[0]),
                }
            )
        return record


@dataclass(frozen=True)
class ChannelUnit(PoreUnit):
    unit_id: str
    control_points_unwrapped_A: np.ndarray
    cross_radius_A: float
    roughness: float
    centerline_samples_A: np.ndarray
    segment_starts_A: np.ndarray
    segment_ends_A: np.ndarray
    segment_tangents_A: np.ndarray
    segment_start_plane_normals_A: np.ndarray
    segment_end_plane_normals_A: np.ndarray
    local_segment_frames: tuple[Rotation, ...]
    roughness_parameters: dict[str, Any]
    segment_cumulative_starts_A: np.ndarray
    segment_lengths_A: np.ndarray
    segment_start_radii_A: np.ndarray
    segment_end_radii_A: np.ndarray
    segment_max_radii_A: np.ndarray
    segment_support_radii_A: np.ndarray
    segment_aabb_min_A: np.ndarray
    segment_aabb_max_A: np.ndarray
    arc_length_A: float
    end_distance_A: float
    eta: float
    tortuosity: float
    anchor_A: np.ndarray
    latent_eta: float | None = None
    latent_tau: float | None = None
    latent_theta_rad: float | None = None
    latent_phi_rad: float | None = None
    latent_theta_xz_deg: float | None = None
    latent_theta_xy_deg: float | None = None
    latent_orientation_component_index: int | None = None
    latent_equivalent_diameter_A: float | None = None
    latent_curvature_fluctuation_target: float | None = None
    orientation: Rotation | None = None
    latent_target_volume_A3: float | None = None
    shape_model: str | None = None
    shape_seed: int | None = None
    radius_profile_s: np.ndarray | None = None
    radius_profile_A: np.ndarray | None = None
    radius_profile_coefficients: np.ndarray | None = None
    bend_count: int | None = None
    nonplanarity: float | None = None
    minimum_self_clearance_A: float | None = None

    @staticmethod
    def from_polyline(
        unit_id: str,
        control_points_unwrapped_A: np.ndarray,
        cross_radius_A: float,
        roughness: float,
        *,
        latent_eta: float | None = None,
        latent_tau: float | None = None,
        latent_theta_rad: float | None = None,
        latent_phi_rad: float | None = None,
        latent_theta_xz_deg: float | None = None,
        latent_theta_xy_deg: float | None = None,
        latent_orientation_component_index: int | None = None,
        latent_equivalent_diameter_A: float | None = None,
        latent_curvature_fluctuation_target: float | None = None,
        orientation: Rotation | None = None,
        latent_target_volume_A3: float | None = None,
        shape_model: str | None = None,
        shape_seed: int | None = None,
        radius_profile_s: np.ndarray | None = None,
        radius_profile_A: np.ndarray | None = None,
        bend_count: int | None = None,
        nonplanarity: float | None = None,
        minimum_self_clearance_A: float | None = None,
    ) -> ChannelUnit:
        control_points = _as_points(control_points_unwrapped_A, "control_points_unwrapped_A")
        if control_points.shape[0] < 2:
            raise ValueError("control_points_unwrapped_A must contain at least two points")
        radius = _positive_float(cross_radius_A, "cross_radius_A")
        roughness_value = _nonnegative_float(roughness, "roughness")
        profile_s, profile_radii = _validate_channel_radius_profile(
            radius_profile_s,
            radius_profile_A,
        )
        if profile_s is not None and shape_model is None:
            shape_model = "variable-radius-polyline-v1"

        chord_lengths = np.linalg.norm(np.diff(control_points, axis=0), axis=1)
        if np.any(chord_lengths <= 0.0):
            raise ValueError("control point segments must have positive length")
        samples = _sample_adaptive_spline(control_points, radius)
        if profile_s is not None:
            samples = _insert_profile_arclength_samples(samples, profile_s)

        segment_starts = samples[:-1]
        segment_ends = samples[1:]
        segment_lengths = np.linalg.norm(segment_ends - segment_starts, axis=1)
        valid = segment_lengths > 1e-12
        segment_starts = segment_starts[valid]
        segment_ends = segment_ends[valid]
        segment_lengths = segment_lengths[valid]
        if segment_lengths.size == 0:
            raise ValueError("sampled channel centerline has zero length")
        samples = np.vstack((segment_starts[0], segment_ends))

        arc_length = float(segment_lengths.sum())
        end_distance = float(np.linalg.norm(samples[-1] - samples[0]))
        if end_distance <= 0.0:
            raise ValueError("channel end distance must be positive")
        anchor = _arclength_centroid(segment_starts, segment_ends, segment_lengths)
        tangents = (segment_ends - segment_starts) / segment_lengths[:, np.newaxis]
        start_plane_normals = tangents.copy()
        end_plane_normals = tangents.copy()
        for index in range(1, tangents.shape[0]):
            start_plane_normals[index] = _safe_bisector(
                tangents[index - 1],
                tangents[index],
            )
        for index in range(tangents.shape[0] - 1):
            end_plane_normals[index] = _safe_bisector(
                tangents[index],
                tangents[index + 1],
            )
        frames = tuple(_segment_frame(tangent) for tangent in tangents)
        cumulative_starts = np.concatenate(([0.0], np.cumsum(segment_lengths[:-1])))
        normalized_starts = cumulative_starts / arc_length
        normalized_ends = (cumulative_starts + segment_lengths) / arc_length
        radius_coefficients = None
        if profile_s is None or profile_radii is None:
            start_radii = np.full(segment_lengths.shape, radius, dtype=float)
            end_radii = start_radii.copy()
            max_radii = start_radii.copy()
        else:
            interpolator = PchipInterpolator(profile_s, profile_radii)
            radius_coefficients = np.asarray(interpolator.c, dtype=float)
            start_radii = np.asarray(interpolator(normalized_starts), dtype=float)
            end_radii = np.asarray(interpolator(normalized_ends), dtype=float)
            max_radii = _channel_segment_max_radii(
                interpolator,
                profile_s,
                normalized_starts,
                normalized_ends,
            )
        roughness_support = _roughness_amplitude_support_A(
            unit_id=unit_id,
            roughness=roughness_value,
            length_scale_A=radius,
        )
        segment_support = max_radii + roughness_support
        segment_aabb_min = (
            np.minimum(segment_starts, segment_ends) - segment_support[:, np.newaxis]
        )
        segment_aabb_max = (
            np.maximum(segment_starts, segment_ends) + segment_support[:, np.newaxis]
        )
        overall_orientation = orientation or _segment_frame(samples[-1] - samples[0])

        return ChannelUnit(
            unit_id=unit_id,
            control_points_unwrapped_A=control_points,
            cross_radius_A=radius,
            roughness=roughness_value,
            centerline_samples_A=samples,
            segment_starts_A=segment_starts,
            segment_ends_A=segment_ends,
            segment_tangents_A=tangents,
            segment_start_plane_normals_A=start_plane_normals,
            segment_end_plane_normals_A=end_plane_normals,
            local_segment_frames=frames,
            roughness_parameters=_roughness_parameters(unit_id, roughness_value),
            segment_cumulative_starts_A=cumulative_starts,
            segment_lengths_A=segment_lengths,
            segment_start_radii_A=start_radii,
            segment_end_radii_A=end_radii,
            segment_max_radii_A=max_radii,
            segment_support_radii_A=segment_support,
            segment_aabb_min_A=segment_aabb_min,
            segment_aabb_max_A=segment_aabb_max,
            arc_length_A=arc_length,
            end_distance_A=end_distance,
            eta=arc_length
            / _channel_equivalent_diameter_A(profile_s, profile_radii, radius),
            tortuosity=arc_length / end_distance,
            anchor_A=anchor,
            latent_eta=latent_eta,
            latent_tau=latent_tau,
            latent_theta_rad=latent_theta_rad,
            latent_phi_rad=latent_phi_rad,
            latent_theta_xz_deg=latent_theta_xz_deg,
            latent_theta_xy_deg=latent_theta_xy_deg,
            latent_orientation_component_index=latent_orientation_component_index,
            latent_equivalent_diameter_A=latent_equivalent_diameter_A,
            latent_curvature_fluctuation_target=latent_curvature_fluctuation_target,
            orientation=overall_orientation,
            latent_target_volume_A3=latent_target_volume_A3,
            shape_model=shape_model,
            shape_seed=shape_seed,
            radius_profile_s=None if profile_s is None else profile_s.copy(),
            radius_profile_A=None if profile_radii is None else profile_radii.copy(),
            radius_profile_coefficients=None
            if radius_coefficients is None
            else radius_coefficients.copy(),
            bend_count=bend_count,
            nonplanarity=nonplanarity,
            minimum_self_clearance_A=minimum_self_clearance_A,
        )

    @property
    def is_variable_radius(self) -> bool:
        return self.radius_profile_s is not None and self.radius_profile_A is not None

    @property
    def sdf_lipschitz_bound(self) -> float:
        radius = float(self.cross_radius_A)
        arc_length = float(self.arc_length_A)
        if (
            not np.isfinite(radius)
            or radius <= 0.0
            or not np.isfinite(arc_length)
            or arc_length <= 0.0
        ):
            return np.inf
        radius_slope = float(
            np.max(
                np.abs(self.segment_end_radii_A - self.segment_start_radii_A)
                / np.maximum(self.segment_lengths_A, 1.0e-12)
            )
        )
        roughness_bound = _roughness_lipschitz_bound(
            unit_id=self.unit_id,
            roughness=float(self.roughness),
            length_scale_A=radius,
            coordinate_derivative_norm=float(
                np.sqrt((1.0 / arc_length) ** 2 + 2.0 / radius**2)
            ),
        )
        return 1.0 + radius_slope + roughness_bound

    def sdf(self, points_A: np.ndarray) -> np.ndarray:
        return self._sdf_impl(points_A, use_aabb_culling=True)

    def sdf_bruteforce(self, points_A: np.ndarray) -> np.ndarray:
        return self._sdf_impl(points_A, use_aabb_culling=False)

    def _sdf_impl(
        self,
        points_A: np.ndarray,
        *,
        use_aabb_culling: bool,
    ) -> np.ndarray:
        points = _as_points(points_A, "points_A")
        field_count = self.segment_lengths_A.size + 2
        if not use_aabb_culling:
            fields = [
                self._segment_field(index, points)
                for index in range(self.segment_lengths_A.size)
            ]
            fields.extend(self._endpoint_cap_fields(points))
            return _smooth_min(np.vstack(fields), axis=0)

        result = np.full(points.shape[0], np.inf, dtype=float)
        skip_margin = _smooth_min_skip_margin_A(field_count)
        for index in range(self.segment_lengths_A.size):
            lower_bound = _segment_aabb_field_lower_bound_A(
                points,
                expanded_min_A=self.segment_aabb_min_A[index],
                expanded_max_A=self.segment_aabb_max_A[index],
                support_radius_A=float(self.segment_support_radii_A[index]),
            )
            active = ~np.isfinite(result) | (lower_bound < result + skip_margin)
            if not np.any(active):
                continue
            field = self._segment_field(index, points[active])
            result[active] = _smooth_min_pair(result[active], field)
        for cap_field in self._endpoint_cap_fields(points):
            result = _smooth_min_pair(result, cap_field)
        return result

    def _radius_at_normalized_arclength(
        self,
        normalized_s: np.ndarray,
    ) -> np.ndarray:
        if (
            self.radius_profile_s is None
            or self.radius_profile_A is None
            or self.radius_profile_coefficients is None
        ):
            return np.full(np.asarray(normalized_s).shape, self.cross_radius_A, dtype=float)
        return _evaluate_cached_pchip(
            self.radius_profile_s,
            self.radius_profile_coefficients,
            normalized_s,
        )

    def _segment_field(self, index: int, points_A: np.ndarray) -> np.ndarray:
        start = self.segment_starts_A[index]
        frame = self.local_segment_frames[index]
        segment_length = float(self.segment_lengths_A[index])
        cumulative_start = float(self.segment_cumulative_starts_A[index])
        offset = points_A - start
        local = frame.inv().apply(offset)
        axial_raw = offset @ self.segment_tangents_A[index]
        axial = np.clip(axial_raw, 0.0, segment_length)
        normalized_s = (cumulative_start + axial) / max(self.arc_length_A, 1e-12)
        local_radius = self._radius_at_normalized_arclength(normalized_s)
        closest = start + axial[:, np.newaxis] * self.segment_tangents_A[index]
        radial = np.linalg.norm(points_A - closest, axis=1) - local_radius
        start_signed = -(offset @ self.segment_start_plane_normals_A[index])
        end_signed = (
            (points_A - self.segment_ends_A[index])
            @ self.segment_end_plane_normals_A[index]
        )
        inside = (start_signed <= 0.0) & (end_signed <= 0.0)
        base = radial.copy()
        base[~inside] = np.maximum(
            radial[~inside],
            np.maximum(start_signed[~inside], end_signed[~inside]),
        )
        roughness_coordinates = np.column_stack(
            [
                normalized_s,
                local[:, 1] / self.cross_radius_A,
                local[:, 2] / self.cross_radius_A,
            ]
        )
        return base - _roughness_perturbation(
            local_coordinates=roughness_coordinates,
            unit_id=self.unit_id,
            roughness=self.roughness,
            length_scale_A=self.cross_radius_A,
        )

    def _endpoint_cap_fields(
        self,
        points_A: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        first_frame = self.local_segment_frames[0]
        first_local = first_frame.inv().apply(points_A - self.segment_starts_A[0])
        first_radius = float(self.segment_start_radii_A[0])
        first = np.linalg.norm(first_local, axis=1) - first_radius
        first = np.where(first_local[:, 0] <= 0.0, first, np.inf)
        first_coordinates = np.column_stack(
            [
                np.zeros(points_A.shape[0], dtype=float),
                first_local[:, 1] / self.cross_radius_A,
                first_local[:, 2] / self.cross_radius_A,
            ]
        )
        first -= _roughness_perturbation(
            local_coordinates=first_coordinates,
            unit_id=self.unit_id,
            roughness=self.roughness,
            length_scale_A=self.cross_radius_A,
        )

        last_frame = self.local_segment_frames[-1]
        last_local = last_frame.inv().apply(points_A - self.segment_ends_A[-1])
        last_radius = float(self.segment_end_radii_A[-1])
        last = np.linalg.norm(last_local, axis=1) - last_radius
        last = np.where(last_local[:, 0] >= 0.0, last, np.inf)
        last_coordinates = np.column_stack(
            [
                np.ones(points_A.shape[0], dtype=float),
                last_local[:, 1] / self.cross_radius_A,
                last_local[:, 2] / self.cross_radius_A,
            ]
        )
        last -= _roughness_perturbation(
            local_coordinates=last_coordinates,
            unit_id=self.unit_id,
            roughness=self.roughness,
            length_scale_A=self.cross_radius_A,
        )
        return first, last

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": 2 if self.is_variable_radius else 1,
            "unit_id": self.unit_id,
            "kind": "channel",
            "latent_parameters": {
                "control_points_unwrapped_A": self.control_points_unwrapped_A.tolist(),
                "cross_radius_A": float(self.cross_radius_A),
                "roughness": float(self.roughness),
                "target_volume_A3": None
                if self.latent_target_volume_A3 is None
                else float(self.latent_target_volume_A3),
                "eta": float(self.latent_eta if self.latent_eta is not None else self.eta),
                "tau": float(
                    self.latent_tau if self.latent_tau is not None else self.tortuosity
                ),
                "orientation": _latent_orientation_record(
                    self.latent_theta_rad,
                    self.latent_phi_rad,
                    theta_xz_deg=self.latent_theta_xz_deg,
                    theta_xy_deg=self.latent_theta_xy_deg,
                    paired_component_index=self.latent_orientation_component_index,
                ),
                "equivalent_diameter_A": None
                if self.latent_equivalent_diameter_A is None
                else float(self.latent_equivalent_diameter_A),
                "curvature_fluctuation_target": None
                if self.latent_curvature_fluctuation_target is None
                else float(self.latent_curvature_fluctuation_target),
            },
            "realized_geometry": {
                "anchor_A": self.anchor_A.tolist(),
                "arc_length_A": float(self.arc_length_A),
                "end_distance_A": float(self.end_distance_A),
                "eta": float(self.eta),
                "equivalent_diameter_A": float(
                    _channel_equivalent_diameter_A(
                        self.radius_profile_s,
                        self.radius_profile_A,
                        self.cross_radius_A,
                    )
                ),
                "tortuosity": float(self.tortuosity),
                "orientation": _realized_orientation_record(
                    self.orientation
                    or _segment_frame(
                        self.centerline_samples_A[-1] - self.centerline_samples_A[0]
                    )
                ),
                "segment_frame_quaternions_xyzw": [
                    frame.as_quat().tolist() for frame in self.local_segment_frames
                ],
                "roughness_parameters": self.roughness_parameters,
            },
        }
        if self.is_variable_radius:
            dense_radii = np.asarray(
                PchipInterpolator(self.radius_profile_s, self.radius_profile_A)(
                    np.linspace(0.0, 1.0, 1025)
                ),
                dtype=float,
            )
            record["shape_model"] = self.shape_model
            record["shape_seed"] = int(self.shape_seed if self.shape_seed is not None else 0)
            record["realized_geometry"].update(
                {
                    "control_points_unwrapped_A": self.control_points_unwrapped_A.tolist(),
                    "equivalent_radius_A": float(self.cross_radius_A),
                    "radius_profile_s": self.radius_profile_s.tolist(),
                    "radius_profile_A": self.radius_profile_A.tolist(),
                    "radius_cv": float(np.std(dense_radii) / np.mean(dense_radii)),
                    "minimum_to_maximum_radius_ratio": float(
                        np.min(dense_radii) / np.max(dense_radii)
                    ),
                    "bend_count": int(self.bend_count or 0),
                    "nonplanarity": float(self.nonplanarity or 0.0),
                    "minimum_self_clearance_A": float(
                        self.minimum_self_clearance_A
                        if self.minimum_self_clearance_A is not None
                        else 0.0
                    ),
                }
            )
        return record


@dataclass(frozen=True)
class PoreGeometry:
    units: list[PoreUnit]
    target_box_A: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_box_A", _as_box(self.target_box_A).copy())

    @property
    def sdf_lipschitz_bound(self) -> float:
        if not self.units:
            return 0.0
        return max(float(unit.sdf_lipschitz_bound) for unit in self.units)

    def sdf(self, points_A: np.ndarray) -> np.ndarray:
        points = _as_points(points_A, "points_A")
        if not self.units:
            return np.full(points.shape[0], np.inf, dtype=float)
        shifted_values = []
        for unit in self.units:
            if isinstance(unit, CompactUnit) and _compact_minimum_image_safe(
                unit, self.target_box_A
            ):
                shifted_values.append(
                    _compact_periodic_sdf(unit, points, self.target_box_A)
                )
                continue
            x_shifts = _periodic_shifts_for_axis(
                points[:, 0], unit, axis=0, box=self.target_box_A
            )
            y_shifts = _periodic_shifts_for_axis(
                points[:, 1], unit, axis=1, box=self.target_box_A
            )
            for x_shift in x_shifts:
                for y_shift in y_shifts:
                    shifted = points - np.array([x_shift, y_shift, 0.0], dtype=float)
                    shifted_values.append(unit.sdf(shifted))
        values = np.vstack(shifted_values)
        return _smooth_min(values, axis=0)


@dataclass(frozen=True)
class BuiltGeometry:
    geometry: PoreGeometry
    units: list[PoreUnit]
    realized_anchors_A: np.ndarray
    latent_to_realized_ids: dict[str, str]


def build_units(
    config: GeneratorConfig,
    center_plan: CenterSeedPlan,
    rng: np.random.Generator,
) -> BuiltGeometry:
    latent_points = _as_points(center_plan.intended_points_A, "center_plan.intended_points_A")
    target_box = _target_box_from_config(config)
    count = latent_points.shape[0]
    if count == 0:
        units: list[PoreUnit] = []
        return BuiltGeometry(
            geometry=PoreGeometry(units, target_box),
            units=units,
            realized_anchors_A=np.empty((0, 3), dtype=float),
            latent_to_realized_ids={},
        )

    labels = _unit_labels(config, count, rng)
    volumes = _unit_volumes_A3(config, labels, rng)
    compact_eta = _sample_compact_eta(config, int(np.count_nonzero(labels == "compact")), rng)
    compact_roughness = stratified_sample(
        config.compact.roughness, int(np.count_nonzero(labels == "compact")), rng
    )
    channel_eta = _sample_channel_eta(config, int(np.count_nonzero(labels == "channel")), rng)
    channel_tau = _sample_channel_tau(config, int(np.count_nonzero(labels == "channel")), rng)
    channel_roughness = stratified_sample(
        config.channel.roughness, int(np.count_nonzero(labels == "channel")), rng
    )
    formal_shape = config.formal_targets.shape
    if config.source_schema_version == 3:
        channel_diameters = stratified_sample(
            formal_shape.equivalent_diameter_A,
            int(np.count_nonzero(labels == "channel")),
            rng,
        )
        curvature_targets = stratified_sample(
            formal_shape.curvature_fluctuation,
            count,
            rng,
        )
    else:
        channel_diameters = np.full(
            int(np.count_nonzero(labels == "channel")),
            np.nan,
            dtype=float,
        )
        curvature_targets = np.full(count, np.nan, dtype=float)
    orientations = _sample_orientations(config, count, rng)

    units = []
    realized_anchors = []
    latent_to_realized_ids = {}
    compact_index = 0
    channel_index = 0
    for latent_index, (
        latent_anchor,
        label,
        volume,
        orientation_sample,
        curvature_target,
    ) in enumerate(
        zip(
            latent_points,
            labels,
            volumes,
            orientations,
            curvature_targets,
            strict=True,
        )
    ):
        if label == "compact":
            eta = float(compact_eta[compact_index])
            shape_seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
            shape_profile = generate_multilobe_profile(
                target_volume_A3=float(volume),
                aspect_ratio=eta,
                shape_seed=shape_seed,
            )
            unit_id = f"compact-{compact_index:04d}"
            unit = CompactUnit(
                unit_id=unit_id,
                center_A=latent_anchor.copy(),
                radii_A=shape_profile.envelope_radii_A.copy(),
                orientation=_compact_major_axis_frame(
                    orientation_sample.rotation.apply(np.array([1.0, 0.0, 0.0]))
                ),
                exponent=_SUPERELLIPSOID_EXPONENT,
                roughness=float(
                    _initial_roughness_from_curvature_target(curvature_target)
                    if np.isfinite(curvature_target)
                    else compact_roughness[compact_index]
                ),
                latent_theta_rad=orientation_sample.theta_rad,
                latent_phi_rad=orientation_sample.phi_rad,
                latent_theta_xz_deg=orientation_sample.theta_xz_deg,
                latent_theta_xy_deg=orientation_sample.theta_xy_deg,
                latent_orientation_component_index=orientation_sample.paired_component_index,
                latent_curvature_fluctuation_target=None
                if not np.isfinite(curvature_target)
                else float(curvature_target),
                latent_target_volume_A3=float(volume),
                latent_eta=eta,
                shape_model="multilobe-v1",
                shape_seed=shape_seed,
                lobe_centers_local_A=shape_profile.lobe_centers_local_A.copy(),
                lobe_radii_A=shape_profile.lobe_radii_A.copy(),
                smooth_length_A=float(shape_profile.smooth_length_A),
                envelope_fill_fraction=float(shape_profile.envelope_fill_fraction),
                centroid_offset_A=float(shape_profile.centroid_offset_A),
                lobes_connected=bool(shape_profile.connected),
            )
            compact_index += 1
        else:
            eta = float(channel_eta[channel_index])
            tau = float(channel_tau[channel_index])
            shape_seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
            equivalent_diameter = float(channel_diameters[channel_index])
            shape_profile = generate_variable_radius_channel_profile(
                target_volume_A3=float(volume),
                eta=eta,
                tau=tau,
                shape_seed=shape_seed,
                target_equivalent_diameter_A=None
                if not np.isfinite(equivalent_diameter)
                else equivalent_diameter,
            )
            unit_id = f"channel-{channel_index:04d}"
            control_points = (
                orientation_sample.rotation.apply(shape_profile.control_points_local_A)
                + latent_anchor
            )
            unit = ChannelUnit.from_polyline(
                unit_id=unit_id,
                control_points_unwrapped_A=control_points,
                cross_radius_A=shape_profile.equivalent_radius_A,
                roughness=float(
                    _initial_roughness_from_curvature_target(curvature_target)
                    if np.isfinite(curvature_target)
                    else channel_roughness[channel_index]
                ),
                latent_eta=eta,
                latent_tau=tau,
                latent_theta_rad=orientation_sample.theta_rad,
                latent_phi_rad=orientation_sample.phi_rad,
                latent_theta_xz_deg=orientation_sample.theta_xz_deg,
                latent_theta_xy_deg=orientation_sample.theta_xy_deg,
                latent_orientation_component_index=orientation_sample.paired_component_index,
                latent_equivalent_diameter_A=None
                if not np.isfinite(equivalent_diameter)
                else equivalent_diameter,
                latent_curvature_fluctuation_target=None
                if not np.isfinite(curvature_target)
                else float(curvature_target),
                orientation=orientation_sample.rotation,
                latent_target_volume_A3=None
                if config.source_schema_version == 3
                else float(volume),
                shape_model="variable-radius-spline-v1",
                shape_seed=shape_seed,
                radius_profile_s=shape_profile.radius_profile_s,
                radius_profile_A=shape_profile.radius_profile_A,
                bend_count=shape_profile.bend_count,
                nonplanarity=shape_profile.nonplanarity,
                minimum_self_clearance_A=shape_profile.minimum_self_clearance_A,
            )
            channel_index += 1

        units.append(unit)
        realized_anchors.append(unit.anchor_A)
        latent_to_realized_ids[f"latent-{latent_index:04d}"] = unit.unit_id

    realized_anchors_array = np.vstack(realized_anchors)
    return BuiltGeometry(
        geometry=PoreGeometry(units, target_box),
        units=units,
        realized_anchors_A=realized_anchors_array,
        latent_to_realized_ids=latent_to_realized_ids,
    )


def _as_points(points_A: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(points_A, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be finite")
    return points


def _as_vector(vector_A: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(vector_A, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def _as_box(box_A: np.ndarray) -> np.ndarray:
    box = _as_vector(box_A, "target_box_A")
    if np.any(box <= 0.0):
        raise ValueError("target_box_A must contain positive lengths")
    return box


def _positive_float(value: float, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return parsed


def _nonnegative_float(value: float, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite value")
    return parsed


def _validate_channel_radius_profile(
    radius_profile_s: np.ndarray | None,
    radius_profile_A: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if radius_profile_s is None and radius_profile_A is None:
        return None, None
    if radius_profile_s is None or radius_profile_A is None:
        raise ValueError("radius_profile_s and radius_profile_A must be provided together")
    nodes = np.asarray(radius_profile_s, dtype=float)
    radii = np.asarray(radius_profile_A, dtype=float)
    if nodes.ndim != 1 or radii.ndim != 1 or nodes.shape != radii.shape:
        raise ValueError("channel radius profile arrays must be matching one-dimensional arrays")
    if nodes.size < 2:
        raise ValueError("channel radius profile must contain at least two nodes")
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(radii)):
        raise ValueError("channel radius profile values must be finite")
    if not np.isclose(nodes[0], 0.0) or not np.isclose(nodes[-1], 1.0):
        raise ValueError("channel radius profile nodes must start at 0 and end at 1")
    if np.any(np.diff(nodes) <= 0.0):
        raise ValueError("channel radius profile nodes must be strictly increasing")
    if np.any(radii <= 0.0):
        raise ValueError("channel radius profile radii must be positive")
    return nodes.copy(), radii.copy()


def _evaluate_cached_pchip(
    nodes: np.ndarray,
    coefficients: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    samples = np.asarray(values, dtype=float)
    indices = np.searchsorted(nodes, samples, side="right") - 1
    indices = np.clip(indices, 0, nodes.size - 2)
    delta = samples - nodes[indices]
    return (
        (
            coefficients[0, indices] * delta
            + coefficients[1, indices]
        )
        * delta
        + coefficients[2, indices]
    ) * delta + coefficients[3, indices]


def _insert_profile_arclength_samples(
    centerline_samples_A: np.ndarray,
    profile_s: np.ndarray,
) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(centerline_samples_A, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length <= 0.0:
        raise ValueError("sampled channel centerline has zero length")
    normalized = cumulative / total_length
    requested = np.unique(np.concatenate((normalized, np.asarray(profile_s, dtype=float))))
    result = np.empty((requested.size, 3), dtype=float)
    for index, target in enumerate(requested):
        if target <= 0.0:
            result[index] = centerline_samples_A[0]
            continue
        if target >= 1.0:
            result[index] = centerline_samples_A[-1]
            continue
        right = int(np.searchsorted(normalized, target, side="right"))
        left = right - 1
        span = normalized[right] - normalized[left]
        fraction = (target - normalized[left]) / max(float(span), 1.0e-15)
        result[index] = (
            (1.0 - fraction) * centerline_samples_A[left]
            + fraction * centerline_samples_A[right]
        )
    return result


def _channel_segment_max_radii(
    interpolator: PchipInterpolator,
    profile_s: np.ndarray,
    segment_starts_s: np.ndarray,
    segment_ends_s: np.ndarray,
) -> np.ndarray:
    maxima = np.empty(segment_starts_s.shape, dtype=float)
    for index, (start, end) in enumerate(
        zip(segment_starts_s, segment_ends_s, strict=True)
    ):
        interior = profile_s[(profile_s > start) & (profile_s < end)]
        sample_s = np.concatenate(([start], interior, [end]))
        maxima[index] = float(np.max(interpolator(sample_s)))
    return maxima


def _channel_equivalent_diameter_A(
    profile_s: np.ndarray | None,
    profile_radii_A: np.ndarray | None,
    cross_radius_A: float,
) -> float:
    if profile_s is None or profile_radii_A is None:
        return 2.0 * float(cross_radius_A)
    dense_s = np.linspace(0.0, 1.0, 4097)
    dense_radii = np.asarray(
        PchipInterpolator(profile_s, profile_radii_A)(dense_s),
        dtype=float,
    )
    mean_radius_squared = float(np.trapezoid(dense_radii**2, dense_s))
    return 2.0 * float(np.sqrt(mean_radius_squared))


def _initial_roughness_from_curvature_target(target: float) -> float:
    value = float(target)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("curvature fluctuation target must be finite and nonnegative")
    return min(0.25, 0.05 * value)


def _roughness_amplitude_support_A(
    *,
    unit_id: str,
    roughness: float,
    length_scale_A: float,
) -> float:
    if roughness == 0.0:
        return 0.0
    parameters = _roughness_parameters(unit_id, roughness)
    mean_absolute_amplitude = sum(
        abs(float(value)) for value in parameters["amplitudes"]
    ) / max(float(parameters["mode_count"]), 1.0)
    return float(roughness) * float(length_scale_A) * mean_absolute_amplitude


def _segment_aabb_field_lower_bound_A(
    points_A: np.ndarray,
    *,
    expanded_min_A: np.ndarray,
    expanded_max_A: np.ndarray,
    support_radius_A: float,
) -> np.ndarray:
    minimum = np.asarray(expanded_min_A, dtype=float)
    maximum = np.asarray(expanded_max_A, dtype=float)
    below = np.maximum(minimum - points_A, 0.0)
    above = np.maximum(points_A - maximum, 0.0)
    outside_distance = np.linalg.norm(below + above, axis=1)
    inside = np.all((points_A >= minimum) & (points_A <= maximum), axis=1)
    return np.where(
        inside,
        -float(support_radius_A),
        outside_distance / np.sqrt(2.0),
    )


def _smooth_min_skip_margin_A(segment_count: int, tolerance_A: float = 1.0e-10) -> float:
    if segment_count <= 1:
        return 0.0
    total_tail = np.expm1(_SMOOTH_UNION_SHARPNESS * float(tolerance_A))
    per_segment_tail = total_tail / float(segment_count)
    return float(-np.log(max(per_segment_tail, np.finfo(float).tiny)) / _SMOOTH_UNION_SHARPNESS)


def _smooth_min_pair(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return -np.logaddexp(
        -_SMOOTH_UNION_SHARPNESS * first,
        -_SMOOTH_UNION_SHARPNESS * second,
    ) / _SMOOTH_UNION_SHARPNESS


def _sample_adaptive_spline(control_points: np.ndarray, radius_A: float) -> np.ndarray:
    parameters, curve, tangent = _fit_centerline_spline(control_points)
    tolerance = max(_SPLINE_CHORD_TOLERANCE_FRACTION * radius_A, 1e-4)
    samples = [curve(parameters[0])]

    def refine(left: float, right: float, depth: int) -> None:
        left_point = curve(left)
        right_point = curve(right)
        middle = 0.5 * (left + right)
        middle_point = curve(middle)
        chord_middle = 0.5 * (left_point + right_point)
        chord_error = float(np.linalg.norm(middle_point - chord_middle))
        turn = _angle_between(tangent(left), tangent(right))
        if (
            depth < _SPLINE_MAX_REFINEMENT_DEPTH
            and (chord_error > tolerance or turn > _SPLINE_MAX_TURN_RAD)
        ):
            refine(left, middle, depth + 1)
            refine(middle, right, depth + 1)
        else:
            samples.append(right_point)

    for left, right in pairwise(parameters):
        refine(float(left), float(right), 0)
    return np.asarray(samples, dtype=float)


def _fit_centerline_spline(
    control_points: np.ndarray,
) -> tuple[np.ndarray, Any, Any]:
    chord_lengths = np.linalg.norm(np.diff(control_points, axis=0), axis=1)
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(chord_lengths)))
    parameters = cumulative_lengths / float(cumulative_lengths[-1])
    if control_points.shape[0] == 2:

        def curve(t: float) -> np.ndarray:
            return (1.0 - t) * control_points[0] + t * control_points[1]

        def tangent(_t: float) -> np.ndarray:
            return _unit_vector(control_points[1] - control_points[0])

        return parameters, curve, tangent

    splines = tuple(CubicSpline(parameters, control_points[:, axis]) for axis in range(3))
    derivative_splines = tuple(spline.derivative() for spline in splines)

    def curve(t: float) -> np.ndarray:
        return np.array([spline(t) for spline in splines], dtype=float)

    def tangent(t: float) -> np.ndarray:
        return _unit_vector(np.array([spline(t) for spline in derivative_splines], dtype=float))

    return parameters, curve, tangent


def _angle_between(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.arccos(np.clip(float(np.dot(first, second)), -1.0, 1.0)))


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("vector must have positive length")
    return np.asarray(vector, dtype=float) / norm


def _arclength_centroid(
    segment_starts_A: np.ndarray,
    segment_ends_A: np.ndarray,
    segment_lengths_A: np.ndarray,
) -> np.ndarray:
    midpoints = 0.5 * (segment_starts_A + segment_ends_A)
    return np.average(midpoints, axis=0, weights=segment_lengths_A)


def _safe_bisector(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    combined = np.asarray(first, dtype=float) + np.asarray(second, dtype=float)
    norm = float(np.linalg.norm(combined))
    if norm <= 1.0e-12:
        return _unit_vector(second)
    return combined / norm


def _segment_frame(segment_vector_A: np.ndarray) -> Rotation:
    return _frame_mapping_local_axis(segment_vector_A, np.array([1.0, 0.0, 0.0]))


def _compact_major_axis_frame(major_axis_A: np.ndarray) -> Rotation:
    return _frame_mapping_local_axis(major_axis_A, np.array([0.0, 0.0, 1.0]))


def _frame_mapping_local_axis(world_axis_A: np.ndarray, local_axis: np.ndarray) -> Rotation:
    direction = _unit_vector(world_axis_A)
    rotation, _ = Rotation.align_vectors(
        direction[np.newaxis, :],
        np.asarray(local_axis, dtype=float).reshape(1, 3),
    )
    return rotation


def _latent_orientation_record(
    theta_rad: float | None,
    phi_rad: float | None,
    *,
    theta_xz_deg: float | None = None,
    theta_xy_deg: float | None = None,
    paired_component_index: int | None = None,
) -> dict[str, Any]:
    return {
        "theta_rad": None if theta_rad is None else float(theta_rad),
        "phi_rad": None if phi_rad is None else float(phi_rad),
        "theta_xz_deg": None if theta_xz_deg is None else float(theta_xz_deg),
        "theta_xy_deg": None if theta_xy_deg is None else float(theta_xy_deg),
        "paired_component_index": paired_component_index,
    }


def _realized_orientation_record(
    rotation: Rotation,
    *,
    local_axis: np.ndarray | None = None,
) -> dict[str, Any]:
    direction = rotation.apply(
        np.array([1.0, 0.0, 0.0]) if local_axis is None else _unit_vector(local_axis)
    )
    theta = float(np.arccos(np.clip(direction[0], -1.0, 1.0)))
    phi = float(np.arctan2(direction[2], direction[1]))
    return {
        "theta_rad": theta,
        "phi_rad": phi,
        "direction": direction.tolist(),
        "quaternion_xyzw": rotation.as_quat().tolist(),
    }


def _roughness_parameters(unit_id: str, roughness: float) -> dict[str, Any]:
    if roughness == 0.0:
        return {"mode_count": 0, "frequencies": [], "phases_rad": [], "amplitudes": []}
    seed_bytes = blake2b(unit_id.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(seed_bytes, byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    amplitudes = (rng.uniform(0.35, 1.0, _ROUGHNESS_MODE_COUNT)).tolist()
    phases = (rng.uniform(0.0, 2.0 * np.pi, _ROUGHNESS_MODE_COUNT)).tolist()
    frequencies = np.arange(1, _ROUGHNESS_MODE_COUNT + 1, dtype=float).tolist()
    return {
        "mode_count": _ROUGHNESS_MODE_COUNT,
        "frequencies": frequencies,
        "phases_rad": phases,
        "amplitudes": amplitudes,
    }


def _roughness_perturbation(
    *,
    local_coordinates: np.ndarray,
    unit_id: str,
    roughness: float,
    length_scale_A: float,
) -> np.ndarray:
    if roughness == 0.0:
        return np.zeros(local_coordinates.shape[0], dtype=float)
    parameters = _roughness_parameters(unit_id, roughness)
    perturbation = np.zeros(local_coordinates.shape[0], dtype=float)
    coordinate = np.sum(local_coordinates, axis=1)
    for frequency, phase, amplitude in zip(
        parameters["frequencies"],
        parameters["phases_rad"],
        parameters["amplitudes"],
        strict=True,
    ):
        perturbation += amplitude * np.sin(2.0 * np.pi * frequency * coordinate + phase)
    perturbation /= max(float(parameters["mode_count"]), 1.0)
    return float(roughness) * float(length_scale_A) * perturbation


def _compact_base_lipschitz_bound(radii_A: np.ndarray, exponent: float) -> float:
    if not np.isfinite(exponent) or exponent < 1.0:
        return np.inf
    if np.isclose(exponent, 2.0) and np.allclose(radii_A, radii_A[0]):
        return 1.0
    minimum_axis = float(np.min(radii_A))
    axis_scale = float(np.max(1.0 / radii_A))
    norm_gradient_factor = float(2.0 ** max(1.0 / exponent - 0.5, 0.0))
    return minimum_axis * axis_scale * norm_gradient_factor


def _roughness_lipschitz_bound(
    *,
    unit_id: str,
    roughness: float,
    length_scale_A: float,
    coordinate_derivative_norm: float,
) -> float:
    if roughness == 0.0:
        return 0.0
    if (
        not np.isfinite(roughness)
        or roughness < 0.0
        or not np.isfinite(length_scale_A)
        or length_scale_A <= 0.0
        or not np.isfinite(coordinate_derivative_norm)
        or coordinate_derivative_norm <= 0.0
    ):
        return np.inf
    parameters = _roughness_parameters(unit_id, roughness)
    mode_count = max(float(parameters["mode_count"]), 1.0)
    derivative_sum = sum(
        abs(float(amplitude)) * 2.0 * np.pi * abs(float(frequency))
        for amplitude, frequency in zip(
            parameters["amplitudes"],
            parameters["frequencies"],
            strict=True,
        )
    )
    return (
        float(roughness)
        * float(length_scale_A)
        * derivative_sum
        * float(coordinate_derivative_norm)
        / mode_count
    )


def _smooth_min(values: np.ndarray, axis: int) -> np.ndarray:
    if values.shape[axis] == 1:
        return np.squeeze(values, axis=axis)
    return -logsumexp(-_SMOOTH_UNION_SHARPNESS * values, axis=axis) / _SMOOTH_UNION_SHARPNESS


def _target_box_from_config(config: GeneratorConfig) -> np.ndarray:
    target = config.film.target_box_A
    return np.array([target.x, target.y, target.z], dtype=float)


def _compact_minimum_image_safe(
    unit: CompactUnit,
    box_A: np.ndarray,
) -> bool:
    support = float(np.max(unit.radii_A)) * (1.0 + float(unit.roughness))
    return 2.0 * support < min(float(box_A[0]), float(box_A[1]))


def _compact_periodic_sdf(
    unit: CompactUnit,
    points_A: np.ndarray,
    box_A: np.ndarray,
) -> np.ndarray:
    delta = points_A - unit.center_A
    delta = delta.copy()
    delta[:, 0] -= float(box_A[0]) * np.rint(delta[:, 0] / float(box_A[0]))
    delta[:, 1] -= float(box_A[1]) * np.rint(delta[:, 1] / float(box_A[1]))
    wrapped = unit.center_A + delta
    result = unit.sdf(wrapped)
    support = float(np.max(unit.radii_A)) * (1.0 + float(unit.roughness))
    skip_margin = _smooth_min_skip_margin_A(9)
    for x_image in (-1.0, 0.0, 1.0):
        for y_image in (-1.0, 0.0, 1.0):
            if x_image == 0.0 and y_image == 0.0:
                continue
            offset = np.array(
                [x_image * box_A[0], y_image * box_A[1], 0.0],
                dtype=float,
            )
            alternative = wrapped + offset
            lower_bound = np.linalg.norm(alternative - unit.center_A, axis=1) - support
            active = lower_bound < result + skip_margin
            if not np.any(active):
                continue
            result[active] = _smooth_min_pair(
                result[active],
                unit.sdf(alternative[active]),
            )
    return result


def _periodic_shifts_for_axis(
    query_values: np.ndarray,
    unit: PoreUnit,
    *,
    axis: int,
    box: np.ndarray,
) -> np.ndarray:
    unit_min, unit_max = _unit_axis_extent(unit, axis)
    box_length = float(box[axis])
    query_min = float(np.min(query_values))
    query_max = float(np.max(query_values))
    first = int(np.floor((query_min - unit_max) / box_length))
    last = int(np.ceil((query_max - unit_min) / box_length))
    return np.arange(first, last + 1, dtype=float) * box_length


def _unit_axis_extent(unit: PoreUnit, axis: int) -> tuple[float, float]:
    if isinstance(unit, CompactUnit):
        support = float(np.max(unit.radii_A)) * (1.0 + float(unit.roughness))
        center = float(unit.center_A[axis])
        return center - support, center + support
    if isinstance(unit, ChannelUnit):
        support = float(unit.cross_radius_A) * (1.0 + float(unit.roughness))
        values = unit.centerline_samples_A[:, axis]
        return float(np.min(values) - support), float(np.max(values) + support)
    raise TypeError(f"unsupported pore unit type: {type(unit)!r}")


def _unit_labels(config: GeneratorConfig, count: int, rng: np.random.Generator) -> np.ndarray:
    compact_count, channel_count = allocate_largest_remainder(
        [
            1.0 - config.pores.channel_fraction_by_count,
            config.pores.channel_fraction_by_count,
        ],
        count,
    )
    labels = np.array(["compact"] * int(compact_count) + ["channel"] * int(channel_count))
    return labels[rng.permutation(labels.size)]


def _unit_volumes_A3(
    config: GeneratorConfig,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    total_volume = config.pores.target_porosity * config.film.target_volume_A3
    compact_count = int(np.count_nonzero(labels == "compact"))
    channel_count = int(np.count_nonzero(labels == "channel"))
    compact_relative = stratified_sample(
        config.compact.relative_volume,
        compact_count,
        rng,
    )
    channel_relative = stratified_sample(
        config.channel.relative_volume,
        channel_count,
        rng,
    )
    channel_ratio = config.pores.channel_to_compact_mean_volume_ratio
    denominator = float(compact_relative.sum() + channel_ratio * channel_relative.sum())
    if denominator <= 0.0:
        raise ValueError("unit relative volumes must include positive mass")

    compact_volumes = iter(total_volume * compact_relative / denominator)
    channel_volumes = iter(total_volume * channel_ratio * channel_relative / denominator)
    volumes = []
    for label in labels:
        volumes.append(float(next(channel_volumes if label == "channel" else compact_volumes)))
    return np.asarray(volumes, dtype=float)


def _sample_compact_eta(
    config: GeneratorConfig,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return stratified_sample(config.compact.aspect_ratio, count, rng)


def _sample_channel_eta(
    config: GeneratorConfig,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    eta_spec = config.channel.eta if config.channel.eta is not None else config.channel.aspect_ratio
    if eta_spec is None:
        eta_spec = config.compact.aspect_ratio
    return stratified_sample(eta_spec, count, rng)


def _sample_channel_tau(
    config: GeneratorConfig,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    tau_spec = config.channel.tau if config.channel.tau is not None else config.channel.tortuosity
    if tau_spec is None:
        return np.ones(count, dtype=float)
    return stratified_sample(tau_spec, count, rng)


def _sample_orientations(
    config: GeneratorConfig,
    count: int,
    rng: np.random.Generator,
) -> tuple[OrientationSample, ...]:
    paired = config.formal_targets.shape.orientation
    if config.source_schema_version == 3 and paired is not None:
        counts = allocate_largest_remainder(
            [component.weight for component in paired.components],
            count,
        )
        samples: list[OrientationSample] = []
        for component_index, (component, component_count) in enumerate(
            zip(paired.components, counts, strict=True)
        ):
            if not component_count:
                continue
            theta_xz_deg = stratified_sample(
                component.theta_xz_deg,
                int(component_count),
                rng,
            )
            theta_xy_deg = stratified_sample(
                component.theta_xy_deg,
                int(component_count),
                rng,
            )
            y_signs = rng.choice(np.array([-1.0, 1.0]), size=int(component_count))
            z_signs = rng.choice(np.array([-1.0, 1.0]), size=int(component_count))
            for xz_deg, xy_deg, y_sign, z_sign in zip(
                theta_xz_deg,
                theta_xy_deg,
                y_signs,
                z_signs,
                strict=True,
            ):
                xz_rad = np.deg2rad(float(xz_deg))
                xy_rad = np.deg2rad(float(xy_deg))
                direction = _unit_vector(
                    np.array(
                        [
                            1.0,
                            float(y_sign) * np.tan(xy_rad),
                            float(z_sign) * np.tan(xz_rad),
                        ]
                    )
                )
                samples.append(
                    OrientationSample(
                        rotation=_segment_frame(direction),
                        theta_rad=float(np.arccos(np.clip(direction[0], -1.0, 1.0))),
                        phi_rad=float(np.arctan2(direction[2], direction[1])),
                        theta_xz_deg=float(xz_deg),
                        theta_xy_deg=float(xy_deg),
                        paired_component_index=component_index,
                    )
                )
        permutation = rng.permutation(len(samples))
        return tuple(samples[int(index)] for index in permutation)

    polar_fraction = stratified_sample(config.orientation.distribution, count, rng)
    theta = np.pi * np.clip(polar_fraction, 0.0, 1.0)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, count)
    directions = np.column_stack(
        [
            np.cos(theta),
            np.sin(theta) * np.cos(azimuth),
            np.sin(theta) * np.sin(azimuth),
        ]
    )
    return tuple(
        OrientationSample(
            rotation=_segment_frame(direction),
            theta_rad=float(sampled_theta),
            phi_rad=float(sampled_phi),
        )
        for direction, sampled_theta, sampled_phi in zip(
            directions, theta, azimuth, strict=True
        )
    )


def _channel_control_points(
    *,
    start_A: np.ndarray,
    target_arc_length_A: float,
    tau: float,
    orientation: Rotation,
    radius_A: float,
) -> np.ndarray:
    if tau < 1.0:
        raise ValueError("channel tau must be at least 1")
    end_distance_A = target_arc_length_A / tau
    amplitude_A = _solve_symmetric_bend_amplitude(
        end_distance_A=end_distance_A,
        target_arc_length_A=target_arc_length_A,
        radius_A=radius_A,
    )
    local_end = np.array([end_distance_A, 0.0, 0.0], dtype=float)
    if amplitude_A <= 1e-12:
        local_points = np.array([[0.0, 0.0, 0.0], local_end], dtype=float)
    else:
        local_points = np.array(
            [[0.0, 0.0, 0.0], [0.5 * end_distance_A, amplitude_A, 0.0], local_end],
            dtype=float,
        )
    return start_A[np.newaxis, :] + orientation.apply(local_points)


def _solve_symmetric_bend_amplitude(
    *,
    end_distance_A: float,
    target_arc_length_A: float,
    radius_A: float,
) -> float:
    if target_arc_length_A <= end_distance_A * (1.0 + _CHANNEL_REALIZATION_TOLERANCE):
        return 0.0

    def measured_arc(amplitude_A: float) -> float:
        control_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5 * end_distance_A, amplitude_A, 0.0],
                [end_distance_A, 0.0, 0.0],
            ],
            dtype=float,
        )
        samples = _sample_adaptive_spline(control_points, radius_A)
        return float(np.linalg.norm(np.diff(samples, axis=0), axis=1).sum())

    low = 0.0
    high = max(end_distance_A, radius_A)
    while measured_arc(high) < target_arc_length_A:
        high *= 2.0

    for _ in range(64):
        middle = 0.5 * (low + high)
        if measured_arc(middle) < target_arc_length_A:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)
