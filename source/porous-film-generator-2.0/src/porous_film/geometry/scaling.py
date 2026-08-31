from __future__ import annotations

from typing import Any

import numpy as np

from porous_film.config import GeneratorConfig
from porous_film.geometry.sdf import BuiltGeometry, ChannelUnit, CompactUnit, PoreGeometry


def scale_built_geometry(built: BuiltGeometry, scale: float) -> BuiltGeometry:
    units = [scale_unit(unit, float(scale)) for unit in built.units]
    geometry = PoreGeometry(units, built.geometry.target_box_A)
    anchors = (
        np.vstack([unit.anchor_A for unit in units])
        if units
        else np.empty((0, 3), dtype=float)
    )
    return BuiltGeometry(
        geometry=geometry,
        units=units,
        realized_anchors_A=anchors,
        latent_to_realized_ids=built.latent_to_realized_ids.copy(),
    )


def scale_unit(unit: Any, scale: float) -> Any:
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
            lobe_radii_A=None
            if unit.lobe_radii_A is None
            else unit.lobe_radii_A * scale,
            smooth_length_A=None
            if unit.smooth_length_A is None
            else unit.smooth_length_A * scale,
            envelope_fill_fraction=unit.envelope_fill_fraction,
            centroid_offset_A=None
            if unit.centroid_offset_A is None
            else unit.centroid_offset_A * scale,
            lobes_connected=unit.lobes_connected,
        )
    if isinstance(unit, ChannelUnit):
        anchor = unit.anchor_A
        control = anchor[np.newaxis, :] + (unit.control_points_unwrapped_A - anchor) * scale
        return ChannelUnit.from_polyline(
            unit_id=unit.unit_id,
            control_points_unwrapped_A=control,
            cross_radius_A=unit.cross_radius_A * scale,
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
