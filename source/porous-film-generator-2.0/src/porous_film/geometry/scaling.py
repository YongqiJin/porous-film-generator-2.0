from __future__ import annotations

from typing import Any

import numpy as np

from porous_film.config import GeneratorConfig
from porous_film.geometry.complex_shapes import (
    _bend_count,
    _curve_nonplanarity,
    sample_channel_centerline,
)
from porous_film.geometry.sdf import (
    BuiltGeometry,
    ChannelUnit,
    CompactUnit,
    PoreGeometry,
    _position_control_points_through_z,
)


def scale_built_geometry(
    built: BuiltGeometry,
    scale: float,
    *,
    require_channels_through_z: bool = False,
) -> BuiltGeometry:
    target_box_z_A = float(built.geometry.target_box_A[2]) if require_channels_through_z else None
    units = [scale_unit(unit, float(scale), target_box_z_A=target_box_z_A) for unit in built.units]
    geometry = PoreGeometry(units, built.geometry.target_box_A)
    anchors = (
        np.vstack([unit.anchor_A for unit in units]) if units else np.empty((0, 3), dtype=float)
    )
    return BuiltGeometry(
        geometry=geometry,
        units=units,
        realized_anchors_A=anchors,
        latent_to_realized_ids=built.latent_to_realized_ids.copy(),
    )


def minimum_scale_for_channels_through_z(built: BuiltGeometry) -> float:
    """Return the smallest uniform scale that keeps every channel z-spanning."""
    channels = [unit for unit in built.units if isinstance(unit, ChannelUnit)]
    if not channels:
        raise ValueError("z-through generation requires at least one channel")
    target_box_z_A = float(built.geometry.target_box_A[2])
    spans = np.asarray(
        [
            float(np.ptp(np.asarray(unit.centerline_samples_A, dtype=float)[:, 2]))
            for unit in channels
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(spans)) or np.any(spans <= 0.0):
        raise ValueError("z-through channel centerlines must have positive finite z span")
    return float(np.max(target_box_z_A / spans))


def separate_channel_footprints_xy(
    built: BuiltGeometry,
    *,
    clearance_A: float = 0.0,
    maximum_iterations: int = 2048,
) -> BuiltGeometry:
    """Translate channels in periodic x/y to reduce avoidable pore merging."""
    channels = [unit for unit in built.units if isinstance(unit, ChannelUnit)]
    if len(channels) < 2:
        return built
    clearance = float(clearance_A)
    if not np.isfinite(clearance) or clearance < 0.0:
        raise ValueError("clearance_A must be a nonnegative finite value")
    iterations = int(maximum_iterations)
    if iterations <= 0:
        raise ValueError("maximum_iterations must be positive")

    box_xy = np.asarray(built.geometry.target_box_A[:2], dtype=float)
    positions = np.vstack([np.mod(unit.anchor_A[:2], box_xy) for unit in channels])
    footprint_radii = np.asarray(
        [
            float(
                np.max(
                    np.linalg.norm(
                        np.asarray(unit.centerline_samples_A[:, :2], dtype=float)
                        - unit.anchor_A[np.newaxis, :2],
                        axis=1,
                    )
                )
                + np.max(unit.segment_support_radii_A)
                + 0.5 * clearance
            )
            for unit in channels
        ],
        dtype=float,
    )

    for _ in range(iterations):
        displacements = np.zeros_like(positions)
        maximum_overlap = 0.0
        for first in range(len(channels)):
            for second in range(first + 1, len(channels)):
                delta = positions[second] - positions[first]
                delta -= box_xy * np.round(delta / box_xy)
                distance = float(np.linalg.norm(delta))
                overlap = float(footprint_radii[first] + footprint_radii[second] - distance)
                if overlap <= 0.0:
                    continue
                maximum_overlap = max(maximum_overlap, overlap)
                if distance <= 1.0e-12:
                    angle = (first * 37 + second * 101) * 0.6180339887498949
                    direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
                else:
                    direction = delta / distance
                move = 0.2 * overlap * direction
                displacements[first] -= move
                displacements[second] += move
        positions = np.mod(positions + displacements, box_xy)
        if maximum_overlap <= 1.0e-6:
            break

    translated_by_id = {}
    for unit, position in zip(channels, positions, strict=True):
        shift_xy = position - np.mod(unit.anchor_A[:2], box_xy)
        shift_xy -= box_xy * np.round(shift_xy / box_xy)
        translated_by_id[unit.unit_id] = _translate_channel_xy(unit, shift_xy)
    units = [translated_by_id.get(unit.unit_id, unit) for unit in built.units]
    return BuiltGeometry(
        geometry=PoreGeometry(units, built.geometry.target_box_A),
        units=units,
        realized_anchors_A=np.vstack([unit.anchor_A for unit in units]),
        latent_to_realized_ids=built.latent_to_realized_ids.copy(),
    )


def adjust_channel_lateral_deviations_xy(
    built: BuiltGeometry,
    factors_by_unit_id: dict[str, float],
) -> BuiltGeometry:
    """Scale each channel's xy departure from its endpoint chord while preserving z."""
    units = []
    for unit in built.units:
        if not isinstance(unit, ChannelUnit) or unit.unit_id not in factors_by_unit_id:
            units.append(unit)
            continue
        factor = float(factors_by_unit_id[unit.unit_id])
        if not np.isfinite(factor) or factor < 0.0:
            raise ValueError("channel lateral-deviation factors must be nonnegative and finite")
        controls = np.asarray(unit.control_points_unwrapped_A, dtype=float).copy()
        z_delta = float(controls[-1, 2] - controls[0, 2])
        if abs(z_delta) <= 1.0e-12:
            units.append(unit)
            continue
        relative_z = (controls[:, 2] - controls[0, 2]) / z_delta
        chord_xy = (1.0 - relative_z[:, np.newaxis]) * controls[0, :2] + relative_z[
            :, np.newaxis
        ] * controls[-1, :2]
        controls[:, :2] = chord_xy + factor * (controls[:, :2] - chord_xy)
        adjusted = _replace_channel_control_points(unit, controls)
        anchor_shift = unit.anchor_A - adjusted.anchor_A
        units.append(
            _replace_channel_control_points(
                adjusted,
                adjusted.control_points_unwrapped_A + anchor_shift,
            )
        )
    geometry = PoreGeometry(units, built.geometry.target_box_A)
    anchors = (
        np.vstack([unit.anchor_A for unit in units]) if units else np.empty((0, 3), dtype=float)
    )
    return BuiltGeometry(
        geometry=geometry,
        units=units,
        realized_anchors_A=anchors,
        latent_to_realized_ids=built.latent_to_realized_ids.copy(),
    )


def _translate_channel_xy(unit: ChannelUnit, shift_xy_A: np.ndarray) -> ChannelUnit:
    shift = np.array([float(shift_xy_A[0]), float(shift_xy_A[1]), 0.0], dtype=float)
    return _replace_channel_control_points(unit, unit.control_points_unwrapped_A + shift)


def _replace_channel_control_points(
    unit: ChannelUnit,
    control_points_unwrapped_A: np.ndarray,
) -> ChannelUnit:
    return ChannelUnit.from_polyline(
        unit_id=unit.unit_id,
        control_points_unwrapped_A=control_points_unwrapped_A,
        cross_radius_A=unit.cross_radius_A,
        roughness=unit.roughness,
        latent_eta=unit.latent_eta,
        latent_tau=unit.latent_tau,
        latent_theta_rad=unit.latent_theta_rad,
        latent_phi_rad=unit.latent_phi_rad,
        latent_theta_xz_deg=unit.latent_theta_xz_deg,
        latent_theta_xy_deg=unit.latent_theta_xy_deg,
        latent_orientation_component_index=unit.latent_orientation_component_index,
        latent_equivalent_diameter_A=unit.latent_equivalent_diameter_A,
        latent_curvature_fluctuation_target=unit.latent_curvature_fluctuation_target,
        orientation=unit.orientation,
        latent_target_volume_A3=unit.latent_target_volume_A3,
        shape_model=unit.shape_model,
        shape_seed=unit.shape_seed,
        radius_profile_s=unit.radius_profile_s,
        radius_profile_A=unit.radius_profile_A,
        bend_count=_bend_count(control_points_unwrapped_A),
        nonplanarity=_curve_nonplanarity(
            sample_channel_centerline(control_points_unwrapped_A, sample_count=513)
        ),
        minimum_self_clearance_A=None,
    )


def scale_unit(
    unit: Any,
    scale: float,
    *,
    target_box_z_A: float | None = None,
) -> Any:
    if isinstance(unit, CompactUnit):
        return CompactUnit(
            unit_id=unit.unit_id,
            center_A=unit.center_A.copy(),
            radii_A=unit.radii_A * scale,
            orientation=unit.orientation,
            exponent=unit.exponent,
            roughness=unit.roughness,
            latent_theta_rad=unit.latent_theta_rad,
            latent_phi_rad=unit.latent_phi_rad,
            latent_theta_xz_deg=unit.latent_theta_xz_deg,
            latent_theta_xy_deg=unit.latent_theta_xy_deg,
            latent_orientation_component_index=unit.latent_orientation_component_index,
            latent_equivalent_diameter_A=None
            if unit.latent_equivalent_diameter_A is None
            else unit.latent_equivalent_diameter_A * scale,
            latent_curvature_fluctuation_target=unit.latent_curvature_fluctuation_target,
            latent_target_volume_A3=None
            if unit.latent_target_volume_A3 is None
            else unit.latent_target_volume_A3 * scale**3,
            latent_eta=unit.latent_eta,
            shape_model=unit.shape_model,
            shape_seed=unit.shape_seed,
            lobe_centers_local_A=None
            if unit.lobe_centers_local_A is None
            else unit.lobe_centers_local_A * scale,
            lobe_radii_A=None if unit.lobe_radii_A is None else unit.lobe_radii_A * scale,
            smooth_length_A=None if unit.smooth_length_A is None else unit.smooth_length_A * scale,
            envelope_fill_fraction=unit.envelope_fill_fraction,
            centroid_offset_A=None
            if unit.centroid_offset_A is None
            else unit.centroid_offset_A * scale,
            lobes_connected=unit.lobes_connected,
        )
    if isinstance(unit, ChannelUnit):
        anchor = unit.anchor_A
        control = anchor[np.newaxis, :] + (unit.control_points_unwrapped_A - anchor) * scale
        scaled_radius_A = unit.cross_radius_A * scale
        if target_box_z_A is not None:
            control = _position_control_points_through_z(
                control,
                target_box_z_A=target_box_z_A,
                radius_A=scaled_radius_A,
            )
        return ChannelUnit.from_polyline(
            unit_id=unit.unit_id,
            control_points_unwrapped_A=control,
            cross_radius_A=scaled_radius_A,
            roughness=unit.roughness,
            latent_eta=unit.latent_eta,
            latent_tau=unit.latent_tau,
            latent_theta_rad=unit.latent_theta_rad,
            latent_phi_rad=unit.latent_phi_rad,
            latent_theta_xz_deg=unit.latent_theta_xz_deg,
            latent_theta_xy_deg=unit.latent_theta_xy_deg,
            latent_orientation_component_index=unit.latent_orientation_component_index,
            latent_equivalent_diameter_A=None
            if unit.latent_equivalent_diameter_A is None
            else unit.latent_equivalent_diameter_A * scale,
            latent_curvature_fluctuation_target=unit.latent_curvature_fluctuation_target,
            orientation=unit.orientation,
            latent_target_volume_A3=None
            if unit.latent_target_volume_A3 is None
            else unit.latent_target_volume_A3 * scale**3,
            shape_model=unit.shape_model,
            shape_seed=unit.shape_seed,
            radius_profile_s=None
            if unit.radius_profile_s is None
            else unit.radius_profile_s.copy(),
            radius_profile_A=None
            if unit.radius_profile_A is None
            else unit.radius_profile_A * scale,
            bend_count=unit.bend_count,
            nonplanarity=unit.nonplanarity,
            minimum_self_clearance_A=None
            if unit.minimum_self_clearance_A is None
            else unit.minimum_self_clearance_A * scale,
        )
    raise TypeError(f"unsupported unit type: {type(unit)!r}")


def scale_solver_tolerance(config: GeneratorConfig, spacing: float) -> float:
    return max(porosity_tolerance(config, spacing), 0.01)


def porosity_tolerance(config: GeneratorConfig, spacing: float) -> float:
    nvox = _estimated_voxel_count(_target_box(config), spacing)
    return max(1.0 / max(nvox, 1), 1e-4)


def _target_box(config: GeneratorConfig) -> np.ndarray:
    target = config.film.target_box_A
    return np.array([target.x, target.y, target.z], dtype=float)


def _estimated_voxel_count(target_box: np.ndarray, spacing: float) -> int:
    counts = np.ceil(target_box / float(spacing)).astype(np.int64)
    return int(np.prod(counts))
