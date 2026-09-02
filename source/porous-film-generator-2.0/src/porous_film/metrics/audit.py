from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel
from scipy import ndimage, stats

from porous_film.centers import CenterSeedPlan, pair_distances_periodic_xy
from porous_film.config import GeneratorConfig
from porous_film.distributions import allocate_largest_remainder, mixture_cdf
from porous_film.geometry import BuiltGeometry, ChannelUnit, CompactUnit, PoreGeometry
from porous_film.geometry.complex_shapes import (
    CompactShapeProfile,
    VariableRadiusChannelProfile,
    estimate_multilobe_volume_A3,
    estimate_variable_channel_volume_A3,
)
from porous_film.metrics.connectivity import (
    minimum_cross_section_fraction,
    minimum_cross_section_index,
    periodic_percolates_x,
    pore_component_summary,
)
from porous_film.metrics.final_geometry import (
    FinalGeometryMeasurements,
    measure_final_geometry,
)
from porous_film.metrics.local_thickness import (
    ThicknessStabilityResult,
    compare_local_thickness_coarse_fine,
)
from porous_film.performance import profile_stage
from porous_film.voxel import PhaseGrid
from porous_film.voxel.grid import voxelize_geometry

_DISTRIBUTION_KS_LIMIT = 0.20
_DISTRIBUTION_WASSERSTEIN_LIMIT = 0.20
_OVERLAP_CHUNK_SIZE = 250_000
_RDF_LOSS_LIMIT = 1.0
_MEAN_VOLUME_RATIO_RELATIVE_LIMIT = 0.20
_MIXTURE_WEIGHT_ABSOLUTE_TOLERANCE = 0.05
_COMPACT_ETA_CONSTANT_RELATIVE_TOLERANCE = 0.05
_CHANNEL_GEOMETRY_CONSTANT_RELATIVE_TOLERANCE = 0.01


@dataclass(frozen=True)
class DistributionComparison:
    passed: bool
    ks: float
    normalized_wasserstein: float


@dataclass(frozen=True)
class AuditResult:
    passed: bool
    scalar_errors: dict[str, float]
    distribution_results: dict[str, DistributionComparison]
    rdf_result: dict[str, Any]
    theta_result: DistributionComparison | None
    compact_eta_result: DistributionComparison | None
    channel_eta_result: DistributionComparison | None
    compact_relative_volume_result: DistributionComparison | None
    channel_relative_volume_result: DistributionComparison | None
    roughness_result: DistributionComparison | None
    tau_result: DistributionComparison | None
    channel_fraction_error: float
    realized_mean_volume_ratio: float | None
    mean_volume_ratio_relative_error: float | None
    unit_volume_summary: dict[str, Any]
    mixture_weight_errors: dict[str, dict[str, float]]
    overlap_fraction: float
    connected_pore_domains: int
    largest_pore_fraction: float
    x_surface_openings: tuple[float, float]
    y_surface_openings: tuple[float, float]
    z_lower_opening_fraction: float
    z_upper_opening_fraction: float
    minimum_cross_section_fraction: float
    minimum_cross_section_index: int
    local_thickness_stability_result: ThicknessStabilityResult
    warnings: tuple[str, ...]
    formal_measurements: FinalGeometryMeasurements | None = None
    center_distance_xy_result: dict[str, Any] | None = None
    equivalent_diameter_result: DistributionComparison | None = None
    theta_xz_result: DistributionComparison | None = None
    theta_xy_result: DistributionComparison | None = None
    curvature_fluctuation_result: DistributionComparison | None = None


def compare_samples_to_distribution(
    samples: np.ndarray,
    target: dict[str, Any],
    ks_limit: float,
    normalized_wasserstein_limit: float,
) -> DistributionComparison:
    sample_values = _finite_1d(samples, "samples")
    if sample_values.size == 0:
        return DistributionComparison(passed=False, ks=np.inf, normalized_wasserstein=np.inf)

    target_dict = _as_distribution_dict(target)
    ks = _ks_distance(sample_values, target_dict)
    target_values = _target_quantile_samples(target_dict, max(sample_values.size, 512))
    wasserstein = float(stats.wasserstein_distance(sample_values, target_values))
    normalized = wasserstein / _normalization_scale(target_values, target_dict)
    return DistributionComparison(
        passed=bool(ks <= ks_limit and normalized <= normalized_wasserstein_limit),
        ks=ks,
        normalized_wasserstein=normalized,
    )


def audit_target_distributions(
    config: GeneratorConfig,
    built: BuiltGeometry,
    center_plan: CenterSeedPlan,
    grid: PhaseGrid,
) -> AuditResult:
    with profile_stage("validation"):
        return _audit_target_distributions(config, built, center_plan, grid)


def _audit_target_distributions(
    config: GeneratorConfig,
    built: BuiltGeometry,
    center_plan: CenterSeedPlan,
    grid: PhaseGrid,
) -> AuditResult:
    if config.source_schema_version == 3:
        return _audit_final_geometry_targets(config, built, grid)

    warnings: list[str] = []
    pore_mask = np.asarray(grid.pore_mask, dtype=bool)
    semiconductor_mask = ~pore_mask

    scalar_errors = _scalar_errors(config, grid, built)
    distribution_results: dict[str, DistributionComparison] = {}
    mixture_weight_errors: dict[str, dict[str, float]] = {}
    mixture_sample_counts: dict[str, int] = {}
    expected_compact_count, expected_channel_count = _expected_type_counts(config)
    expected_seed_count = int(config.seed_count)

    theta_samples = _theta_samples(config, built)
    expected_theta_count = theta_samples.size if built.units else expected_seed_count
    _record_missing_required_samples(warnings, "theta", expected_theta_count, theta_samples)
    theta_result = _optional_distribution_comparison(
        theta_samples,
        config.orientation.distribution,
        distribution_results,
        "theta",
    )
    _record_distribution_failure(warnings, "theta", theta_result)
    compact_eta_samples = _compact_eta_samples(built)
    _record_missing_required_samples(
        warnings, "compact_eta", expected_compact_count, compact_eta_samples
    )
    compact_eta_result = _optional_distribution_comparison(
        compact_eta_samples,
        config.compact.aspect_ratio,
        distribution_results,
        "compact_eta",
        constant_relative_tolerance=_COMPACT_ETA_CONSTANT_RELATIVE_TOLERANCE,
    )
    _record_distribution_failure(warnings, "compact_eta", compact_eta_result)
    channel_eta_target = config.channel.eta or config.channel.aspect_ratio or config.compact.aspect_ratio
    channel_eta_samples = _channel_eta_samples(built)
    _record_missing_required_samples(
        warnings, "channel_eta", expected_channel_count, channel_eta_samples
    )
    channel_eta_result = _optional_distribution_comparison(
        channel_eta_samples,
        channel_eta_target,
        distribution_results,
        "channel_eta",
        constant_relative_tolerance=_CHANNEL_GEOMETRY_CONSTANT_RELATIVE_TOLERANCE,
    )
    _record_distribution_failure(warnings, "channel_eta", channel_eta_result)
    tau_target = config.channel.tau or config.channel.tortuosity or {"family": "constant", "value": 1.0}
    tau_samples = _channel_tau_samples(built)
    _record_missing_required_samples(warnings, "tau", expected_channel_count, tau_samples)
    tau_result = _optional_distribution_comparison(
        tau_samples,
        tau_target,
        distribution_results,
        "tau",
        constant_relative_tolerance=_CHANNEL_GEOMETRY_CONSTANT_RELATIVE_TOLERANCE,
    )
    _record_distribution_failure(warnings, "tau", tau_result)
    roughness_samples, roughness_target = _roughness_samples_and_target(config, built)
    _record_missing_required_samples(warnings, "roughness", expected_seed_count, roughness_samples)
    roughness_result = _optional_distribution_comparison(
        roughness_samples,
        roughness_target,
        distribution_results,
        "roughness",
    )
    _record_distribution_failure(warnings, "roughness", roughness_result)

    unit_volume_summary = _unit_volume_summary(built, grid)
    compact_relative_volume_samples = np.asarray(
        unit_volume_summary["compact"]["normalized_relative_volumes"],
        dtype=float,
    )
    channel_relative_volume_samples = np.asarray(
        unit_volume_summary["channel"]["normalized_relative_volumes"],
        dtype=float,
    )
    _record_missing_required_samples(
        warnings,
        "compact_relative_volume",
        expected_compact_count,
        compact_relative_volume_samples,
    )
    compact_relative_volume_result = _optional_distribution_comparison(
        compact_relative_volume_samples,
        config.compact.relative_volume,
        distribution_results,
        "compact_relative_volume",
    )
    _record_distribution_failure(
        warnings,
        "compact_relative_volume",
        compact_relative_volume_result,
    )
    _record_missing_required_samples(
        warnings,
        "channel_relative_volume",
        expected_channel_count,
        channel_relative_volume_samples,
    )
    channel_relative_volume_result = _optional_distribution_comparison(
        channel_relative_volume_samples,
        config.channel.relative_volume,
        distribution_results,
        "channel_relative_volume",
    )
    _record_distribution_failure(
        warnings,
        "channel_relative_volume",
        channel_relative_volume_result,
    )
    realized_mean_volume_ratio = unit_volume_summary["realized_channel_to_compact_mean_volume_ratio"]
    mean_volume_ratio_relative_error = _mean_volume_ratio_relative_error(
        realized_mean_volume_ratio,
        config.pores.channel_to_compact_mean_volume_ratio,
    )
    if (
        mean_volume_ratio_relative_error is not None
        and mean_volume_ratio_relative_error > _MEAN_VOLUME_RATIO_RELATIVE_LIMIT
    ):
        warnings.append("channel/compact mean volume ratio exceeds audit tolerance")

    _record_mixture_errors(
        mixture_weight_errors,
        mixture_sample_counts,
        "theta",
        theta_samples,
        config.orientation.distribution,
    )
    _record_mixture_errors(
        mixture_weight_errors,
        mixture_sample_counts,
        "compact_eta",
        compact_eta_samples,
        config.compact.aspect_ratio,
    )
    _record_mixture_errors(
        mixture_weight_errors,
        mixture_sample_counts,
        "channel_eta",
        channel_eta_samples,
        channel_eta_target,
    )
    _record_mixture_errors(
        mixture_weight_errors,
        mixture_sample_counts,
        "tau",
        _channel_tau_samples(built),
        tau_target,
    )
    _record_mixture_errors(
        mixture_weight_errors,
        mixture_sample_counts,
        "roughness",
        roughness_samples,
        roughness_target,
    )
    _record_mixture_errors(
        mixture_weight_errors,
        mixture_sample_counts,
        "compact_relative_volume",
        compact_relative_volume_samples,
        config.compact.relative_volume,
    )
    _record_mixture_errors(
        mixture_weight_errors,
        mixture_sample_counts,
        "channel_relative_volume",
        channel_relative_volume_samples,
        config.channel.relative_volume,
    )

    channel_fraction_error = _channel_fraction_error(config, built)
    overlap_fraction = _overlap_fraction(built, grid, warnings)
    connected_pore_domains, largest_pore_fraction = pore_component_summary(pore_mask)
    x_open, y_open, z_lower, z_upper = _surface_openings(pore_mask)
    minimum_fraction = minimum_cross_section_fraction(semiconductor_mask)
    minimum_index = minimum_cross_section_index(semiconductor_mask)

    matrix_mask_for_percolation = _matrix_mask_for_percolation(
        semiconductor_mask,
        grid.spacing_A,
        config.matrix_constraints.minimum_skeleton_thickness_A,
    )
    x_percolates = periodic_percolates_x(matrix_mask_for_percolation)
    constraints_passed = True
    if config.matrix_constraints.enabled:
        if config.matrix_constraints.require_x_percolation and not x_percolates:
            warnings.append("semiconductor matrix does not percolate along periodic x")
            constraints_passed = False
        if minimum_fraction < config.matrix_constraints.minimum_cross_section_fraction:
            warnings.append("minimum semiconductor cross-section fraction is below configured limit")
            constraints_passed = False
        if overlap_fraction > config.matrix_constraints.maximum_overlap_fraction:
            warnings.append("pore overlap fraction is above configured limit")
            constraints_passed = False

    rdf_result = _rdf_result(config, built, center_plan)
    if rdf_result["weighted_loss"] > _RDF_LOSS_LIMIT:
        warnings.append("rdf weighted loss exceeds configured audit limit")
    local_thickness_stability_result = _local_thickness_stability_result(config, built, warnings)
    _record_primary_gate_warnings(
        warnings,
        scalar_errors=scalar_errors,
        grid=grid,
        channel_fraction_error=channel_fraction_error,
        expected_seed_count=expected_seed_count,
        mixture_weight_errors=mixture_weight_errors,
        mixture_sample_counts=mixture_sample_counts,
    )
    distribution_passed = all(result.passed for result in distribution_results.values())
    primary_targets_passed = _primary_targets_passed(
        warnings=warnings,
        scalar_errors=scalar_errors,
        grid=grid,
        channel_fraction_error=channel_fraction_error,
        expected_seed_count=expected_seed_count,
        mixture_weight_errors=mixture_weight_errors,
        mixture_sample_counts=mixture_sample_counts,
        rdf_result=rdf_result,
        local_thickness_stability_result=local_thickness_stability_result,
        mean_volume_ratio_relative_error=mean_volume_ratio_relative_error,
    )
    return AuditResult(
        passed=bool(distribution_passed and constraints_passed and primary_targets_passed),
        scalar_errors=scalar_errors,
        distribution_results=distribution_results,
        rdf_result=rdf_result,
        theta_result=theta_result,
        compact_eta_result=compact_eta_result,
        channel_eta_result=channel_eta_result,
        compact_relative_volume_result=compact_relative_volume_result,
        channel_relative_volume_result=channel_relative_volume_result,
        roughness_result=roughness_result,
        tau_result=tau_result,
        channel_fraction_error=channel_fraction_error,
        realized_mean_volume_ratio=realized_mean_volume_ratio,
        mean_volume_ratio_relative_error=mean_volume_ratio_relative_error,
        unit_volume_summary=unit_volume_summary,
        mixture_weight_errors=mixture_weight_errors,
        overlap_fraction=overlap_fraction,
        connected_pore_domains=connected_pore_domains,
        largest_pore_fraction=largest_pore_fraction,
        x_surface_openings=x_open,
        y_surface_openings=y_open,
        z_lower_opening_fraction=z_lower,
        z_upper_opening_fraction=z_upper,
        minimum_cross_section_fraction=minimum_fraction,
        minimum_cross_section_index=minimum_index,
        local_thickness_stability_result=local_thickness_stability_result,
        warnings=tuple(warnings),
    )


def _audit_final_geometry_targets(
    config: GeneratorConfig,
    built: BuiltGeometry,
    grid: PhaseGrid,
) -> AuditResult:
    warnings: list[str] = []
    measurements = measure_final_geometry(grid, config.measurement)
    through_ids = {
        track.track_id for track in measurements.centerlines if track.is_through
    }
    distribution_results: dict[str, DistributionComparison] = {}
    shape = config.formal_targets.shape

    equivalent_diameter_samples = np.asarray(
        [
            float(section.equivalent_diameter_A)
            for section in measurements.cross_sections
            if section.valid
            and section.track_id in through_ids
            and section.equivalent_diameter_A is not None
        ],
        dtype=float,
    )
    equivalent_diameter_result = _formal_distribution_result(
        equivalent_diameter_samples,
        shape.equivalent_diameter_A,
        distribution_results,
        "equivalent_diameter",
        warnings,
    )

    theta_xz_samples = np.asarray(
        [
            float(value.theta_xz_deg)
            for value in measurements.projected_orientations
            if value.track_id in through_ids
            and value.theta_xz_identifiable
            and value.theta_xz_deg is not None
        ],
        dtype=float,
    )
    theta_xy_samples = np.asarray(
        [
            float(value.theta_xy_deg)
            for value in measurements.projected_orientations
            if value.track_id in through_ids
            and value.theta_xy_identifiable
            and value.theta_xy_deg is not None
        ],
        dtype=float,
    )
    theta_xz_target = _paired_orientation_marginal(shape.orientation, "theta_xz_deg")
    theta_xy_target = _paired_orientation_marginal(shape.orientation, "theta_xy_deg")
    theta_xz_result = _formal_distribution_result(
        theta_xz_samples,
        theta_xz_target,
        distribution_results,
        "theta_xz",
        warnings,
    )
    theta_xy_result = _formal_distribution_result(
        theta_xy_samples,
        theta_xy_target,
        distribution_results,
        "theta_xy",
        warnings,
    )

    channel_eta_samples = np.asarray(
        [
            float(value.eta)
            for value in measurements.channel_geometries
            if value.track_id in through_ids and value.valid and value.eta is not None
        ],
        dtype=float,
    )
    channel_tau_samples = np.asarray(
        [
            float(value.tortuosity)
            for value in measurements.channel_geometries
            if value.track_id in through_ids
            and value.valid
            and value.tortuosity is not None
        ],
        dtype=float,
    )
    channel_eta_result = _formal_distribution_result(
        channel_eta_samples,
        shape.channel_aspect_ratio,
        distribution_results,
        "channel_eta",
        warnings,
        constant_relative_tolerance=_CHANNEL_GEOMETRY_CONSTANT_RELATIVE_TOLERANCE,
    )
    tau_result = _formal_distribution_result(
        channel_tau_samples,
        shape.channel_tortuosity,
        distribution_results,
        "channel_tau",
        warnings,
        constant_relative_tolerance=_CHANNEL_GEOMETRY_CONSTANT_RELATIVE_TOLERANCE,
    )

    curvature_samples = np.asarray(
        [
            float(section.curvature_fluctuation)
            for section in measurements.cross_sections
            if section.valid
            and section.track_id in through_ids
            and section.curvature_fluctuation is not None
            and np.isfinite(section.curvature_fluctuation)
        ],
        dtype=float,
    )
    curvature_result = _formal_distribution_result(
        curvature_samples,
        shape.curvature_fluctuation,
        distribution_results,
        "curvature_fluctuation",
        warnings,
    )

    center_distance_result = _final_center_distance_result(config, measurements)
    if not center_distance_result["passed"]:
        warnings.append("center_distance_xy comparison exceeds audit limit")

    pore_mask = np.asarray(grid.pore_mask, dtype=bool)
    semiconductor_mask = ~pore_mask
    target_porosity = float(config.formal_targets.proportion.porosity)
    porosity_error = float(grid.porosity - target_porosity)
    scalar_errors = {
        "porosity_absolute": abs(porosity_error),
        "porosity_signed": porosity_error,
        "porosity_relative": porosity_error / target_porosity,
        "generation_seed_count_absolute": abs(float(len(built.units) - config.seed_count)),
    }

    overlap_fraction = _overlap_fraction(built, grid, warnings)
    connected_pore_domains, largest_pore_fraction = pore_component_summary(pore_mask)
    x_open, y_open, z_lower, z_upper = _surface_openings(pore_mask)
    minimum_fraction = minimum_cross_section_fraction(semiconductor_mask)
    minimum_index = minimum_cross_section_index(semiconductor_mask)
    matrix_mask = _matrix_mask_for_percolation(
        semiconductor_mask,
        grid.spacing_A,
        config.matrix_constraints.minimum_skeleton_thickness_A,
    )
    x_percolates = periodic_percolates_x(matrix_mask)
    constraints_passed = True
    if config.matrix_constraints.enabled:
        if config.matrix_constraints.require_x_percolation and not x_percolates:
            warnings.append("semiconductor matrix does not percolate along periodic x")
            constraints_passed = False
        if minimum_fraction < config.matrix_constraints.minimum_cross_section_fraction:
            warnings.append("minimum semiconductor cross-section fraction is below configured limit")
            constraints_passed = False

    porosity_ok = abs(porosity_error) <= _porosity_tolerance(grid)
    if not porosity_ok:
        warnings.append("porosity scalar error exceeds voxel-resolution audit tolerance")
    local_thickness_stability_result = _local_thickness_stability_result(config, built, warnings)
    required_results = [
        equivalent_diameter_result,
        theta_xz_result,
        theta_xy_result,
        channel_eta_result,
        tau_result,
        curvature_result,
    ]
    distributions_passed = all(
        result is not None and result.passed for result in required_results
    )
    formal_passed = (
        porosity_ok
        and center_distance_result["passed"]
        and distributions_passed
        and constraints_passed
        and local_thickness_stability_result.passed
    )

    unit_volume_summary = _unit_volume_summary(built, grid)
    realized_mean_volume_ratio = unit_volume_summary[
        "realized_channel_to_compact_mean_volume_ratio"
    ]
    return AuditResult(
        passed=bool(formal_passed),
        scalar_errors=scalar_errors,
        distribution_results=distribution_results,
        rdf_result=center_distance_result,
        theta_result=None,
        compact_eta_result=None,
        channel_eta_result=channel_eta_result,
        compact_relative_volume_result=None,
        channel_relative_volume_result=None,
        roughness_result=None,
        tau_result=tau_result,
        channel_fraction_error=_channel_fraction_error(config, built),
        realized_mean_volume_ratio=realized_mean_volume_ratio,
        mean_volume_ratio_relative_error=_mean_volume_ratio_relative_error(
            realized_mean_volume_ratio,
            config.pores.channel_to_compact_mean_volume_ratio,
        ),
        unit_volume_summary=unit_volume_summary,
        mixture_weight_errors={},
        overlap_fraction=overlap_fraction,
        connected_pore_domains=connected_pore_domains,
        largest_pore_fraction=largest_pore_fraction,
        x_surface_openings=x_open,
        y_surface_openings=y_open,
        z_lower_opening_fraction=z_lower,
        z_upper_opening_fraction=z_upper,
        minimum_cross_section_fraction=minimum_fraction,
        minimum_cross_section_index=minimum_index,
        local_thickness_stability_result=local_thickness_stability_result,
        warnings=tuple(warnings),
        formal_measurements=measurements,
        center_distance_xy_result=center_distance_result,
        equivalent_diameter_result=equivalent_diameter_result,
        theta_xz_result=theta_xz_result,
        theta_xy_result=theta_xy_result,
        curvature_fluctuation_result=curvature_result,
    )


def _formal_distribution_result(
    samples: np.ndarray,
    target: Any | None,
    results: dict[str, DistributionComparison],
    name: str,
    warnings: list[str],
    *,
    constant_relative_tolerance: float | None = None,
) -> DistributionComparison | None:
    if target is None:
        return None
    if samples.size == 0:
        warnings.append(f"{name} final-geometry measurement has no valid samples")
        return None
    result = _optional_distribution_comparison(
        samples,
        target,
        results,
        name,
        constant_relative_tolerance=constant_relative_tolerance,
    )
    _record_distribution_failure(warnings, name, result)
    return result


def _paired_orientation_marginal(
    target: Any | None,
    field_name: str,
) -> dict[str, Any] | None:
    if target is None:
        return None
    components = []
    for component in target.components:
        distribution = getattr(component, field_name).model_dump(exclude_none=True)
        components.append({"weight": float(component.weight), **distribution})
    return {"family": "mixture", "components": components}


def _final_center_distance_result(
    config: GeneratorConfig,
    measurements: FinalGeometryMeasurements,
) -> dict[str, Any]:
    measured = measurements.center_distance_xy
    target_spec = config.formal_targets.position_quantity.center_distance_xy
    if target_spec is None:
        return {
            "passed": True,
            "weighted_loss": 0.0,
            "bin_centers_A": measured.bin_centers_A.tolist(),
            "target": [],
            "observed": measured.g_xy.tolist(),
            "pair_count": measured.pair_count,
        }
    target = _evaluate_absolute_distance_target(
        measured.bin_centers_A,
        target_spec.components,
    )
    valid = measured.reference_pair_counts > 0.0
    if measured.pair_count == 0 or not np.any(valid):
        loss = np.inf
    else:
        weights = np.maximum(target[valid], 1.0e-12)
        loss = float(np.average((measured.g_xy[valid] - target[valid]) ** 2, weights=weights))
    return {
        "passed": bool(loss <= _RDF_LOSS_LIMIT),
        "weighted_loss": loss,
        "bin_centers_A": measured.bin_centers_A.tolist(),
        "target": target.tolist(),
        "observed": measured.g_xy.tolist(),
        "observed_pair_counts": measured.observed_pair_counts.tolist(),
        "reference_pair_counts": measured.reference_pair_counts.tolist(),
        "pair_count": measured.pair_count,
        "valid_slice_count": measured.valid_slice_count,
    }


def _evaluate_absolute_distance_target(
    distance_A: np.ndarray,
    components: Any,
) -> np.ndarray:
    values = np.ones_like(np.asarray(distance_A, dtype=float))
    for component in components:
        center = float(component.center_A)
        width = float(component.width_A)
        amplitude = float(component.amplitude)
        gaussian = np.exp(-0.5 * ((distance_A - center) / width) ** 2)
        if component.kind == "peak":
            values += amplitude * gaussian
        elif component.kind in {"dip", "exclusion"}:
            values -= amplitude * gaussian
        elif component.kind == "oscillation":
            phase = 2.0 * np.pi * (distance_A - center) / width
            values += amplitude * gaussian * np.cos(phase)
    return np.maximum(values, 0.0)


def _optional_distribution_comparison(
    samples: np.ndarray,
    target: Any,
    results: dict[str, DistributionComparison],
    name: str,
    *,
    constant_relative_tolerance: float | None = None,
) -> DistributionComparison | None:
    sample_values = _finite_1d(samples, name)
    if sample_values.size == 0:
        return None
    target_dict = _as_distribution_dict(target)
    comparison = compare_samples_to_distribution(
        sample_values,
        target_dict,
        ks_limit=_DISTRIBUTION_KS_LIMIT,
        normalized_wasserstein_limit=_DISTRIBUTION_WASSERSTEIN_LIMIT,
    )
    if (
        constant_relative_tolerance is not None
        and target_dict.get("family") == "constant"
    ):
        target_value = float(target_dict["value"])
        relative_errors = np.abs(sample_values - target_value) / max(
            abs(target_value), 1.0e-12
        )
        if np.all(relative_errors <= constant_relative_tolerance):
            comparison = DistributionComparison(
                passed=(
                    comparison.normalized_wasserstein
                    <= _DISTRIBUTION_WASSERSTEIN_LIMIT
                ),
                ks=0.0,
                normalized_wasserstein=comparison.normalized_wasserstein,
            )
    results[name] = comparison
    return comparison


def _expected_type_counts(config: GeneratorConfig) -> tuple[int, int]:
    seed_count = int(config.seed_count)
    compact_count, channel_count = allocate_largest_remainder(
        [
            1.0 - config.pores.channel_fraction_by_count,
            config.pores.channel_fraction_by_count,
        ],
        seed_count,
    )
    return int(compact_count), int(channel_count)


def _record_missing_required_samples(
    warnings: list[str],
    name: str,
    expected_count: int,
    samples: np.ndarray,
) -> None:
    if expected_count > 0 and samples.size == 0:
        warnings.append(f"{name} target has expected population {expected_count} but no samples")


def _record_distribution_failure(
    warnings: list[str],
    name: str,
    result: DistributionComparison | None,
) -> None:
    if result is not None and not result.passed:
        warnings.append(f"{name} distribution comparison exceeds audit limits")


def _record_primary_gate_warnings(
    warnings: list[str],
    *,
    scalar_errors: dict[str, float],
    grid: PhaseGrid,
    channel_fraction_error: float,
    expected_seed_count: int,
    mixture_weight_errors: dict[str, dict[str, float]],
    mixture_sample_counts: dict[str, int] | None = None,
) -> None:
    porosity_limit = _porosity_tolerance(grid)
    if scalar_errors["porosity_absolute"] > porosity_limit:
        warnings.append("porosity scalar error exceeds voxel-resolution audit tolerance")
    if scalar_errors["seed_count_absolute"] > 0.0:
        warnings.append("seed_count scalar error exceeds exact-count audit tolerance")
    channel_fraction_limit = 0.5 / max(expected_seed_count, 1)
    if abs(channel_fraction_error) > channel_fraction_limit:
        warnings.append("channel fraction error exceeds count-resolution audit tolerance")
    for name, errors in mixture_weight_errors.items():
        mixture_weight_limit = _mixture_weight_tolerance(
            _mixture_sample_count(name, mixture_sample_counts, expected_seed_count)
        )
        for component, error in errors.items():
            if abs(error) > mixture_weight_limit:
                warnings.append(f"{name} mixture {component} count error exceeds audit tolerance")


def _primary_targets_passed(
    *,
    warnings: list[str],
    scalar_errors: dict[str, float],
    grid: PhaseGrid,
    channel_fraction_error: float,
    expected_seed_count: int,
    mixture_weight_errors: dict[str, dict[str, float]],
    mixture_sample_counts: dict[str, int] | None = None,
    rdf_result: dict[str, Any],
    local_thickness_stability_result: ThicknessStabilityResult,
    mean_volume_ratio_relative_error: float | None,
) -> bool:
    missing_required_ok = not any("expected population" in warning for warning in warnings)
    porosity_ok = scalar_errors["porosity_absolute"] <= _porosity_tolerance(grid)
    seed_ok = scalar_errors["seed_count_absolute"] <= 0.0
    channel_fraction_ok = abs(channel_fraction_error) <= 0.5 / max(expected_seed_count, 1)
    mixtures_ok = all(
        abs(error)
        <= _mixture_weight_tolerance(
            _mixture_sample_count(name, mixture_sample_counts, expected_seed_count)
        )
        for name, errors in mixture_weight_errors.items()
        for error in errors.values()
    )
    rdf_ok = float(rdf_result["weighted_loss"]) <= _RDF_LOSS_LIMIT
    mean_volume_ratio_ok = (
        mean_volume_ratio_relative_error is None
        or mean_volume_ratio_relative_error <= _MEAN_VOLUME_RATIO_RELATIVE_LIMIT
    )
    return bool(
        porosity_ok
        and seed_ok
        and missing_required_ok
        and channel_fraction_ok
        and mixtures_ok
        and rdf_ok
        and mean_volume_ratio_ok
        and local_thickness_stability_result.passed
    )


def _porosity_tolerance(grid: PhaseGrid) -> float:
    return max(1.0 / float(grid.pore_mask.size), 0.01)


def _mixture_weight_tolerance(expected_seed_count: int) -> float:
    return _MIXTURE_WEIGHT_ABSOLUTE_TOLERANCE + 0.5 / max(expected_seed_count, 1)


def _mixture_sample_count(
    name: str,
    mixture_sample_counts: dict[str, int] | None,
    fallback_count: int,
) -> int:
    if mixture_sample_counts is None:
        return fallback_count
    return mixture_sample_counts.get(name, fallback_count)


def _scalar_errors(config: GeneratorConfig, grid: PhaseGrid, built: BuiltGeometry) -> dict[str, float]:
    target_porosity = float(config.pores.target_porosity)
    seed_target = float(config.seed_count)
    seed_actual = float(len(built.units))
    porosity_error = float(grid.porosity - target_porosity)
    seed_error = seed_actual - seed_target
    return {
        "porosity_absolute": abs(porosity_error),
        "porosity_signed": porosity_error,
        "porosity_relative": porosity_error / target_porosity,
        "seed_count_absolute": abs(seed_error),
        "seed_count_signed": seed_error,
        "seed_count_relative": seed_error / max(seed_target, 1.0),
    }


def _unit_volume_summary(built: BuiltGeometry, grid: PhaseGrid) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for unit in built.units:
        kind = "compact" if isinstance(unit, CompactUnit) else "channel"
        records.append(
            {
                "unit_id": unit.unit_id,
                "kind": kind,
                "latent_target_volume_A3": None
                if getattr(unit, "latent_target_volume_A3", None) is None
                else float(unit.latent_target_volume_A3),
                "realized_volume_A3": _realized_continuous_volume_A3(unit),
                "realized_clipped_volume_A3": _realized_clipped_volume_A3(unit, built, grid),
            }
        )

    for kind in ("compact", "channel"):
        kind_records = [record for record in records if record["kind"] == kind]
        volumes = np.asarray(
            [record["realized_volume_A3"] for record in kind_records],
            dtype=float,
        )
        mean_volume = float(np.mean(volumes)) if volumes.size else None
        normalized = (
            (volumes / mean_volume).tolist()
            if mean_volume is not None and mean_volume > 0.0
            else []
        )
        for record, normalized_value in zip(kind_records, normalized, strict=True):
            record["realized_normalized_relative_volume"] = float(normalized_value)

    compact_volumes = np.asarray(
        [
            record["realized_volume_A3"]
            for record in records
            if record["kind"] == "compact"
        ],
        dtype=float,
    )
    channel_volumes = np.asarray(
        [
            record["realized_volume_A3"]
            for record in records
            if record["kind"] == "channel"
        ],
        dtype=float,
    )
    compact_summary = _volume_kind_summary(compact_volumes, records, "compact")
    channel_summary = _volume_kind_summary(channel_volumes, records, "channel")
    compact_mean = compact_summary["mean_volume_A3"]
    channel_mean = channel_summary["mean_volume_A3"]
    ratio = (
        float(channel_mean / compact_mean)
        if compact_mean is not None
        and channel_mean is not None
        and compact_mean > 0.0
        else None
    )
    return {
        "per_unit": records,
        "compact": compact_summary,
        "channel": channel_summary,
        "realized_channel_to_compact_mean_volume_ratio": ratio,
    }


def _volume_kind_summary(
    volumes: np.ndarray,
    records: list[dict[str, Any]],
    kind: str,
) -> dict[str, Any]:
    kind_records = [record for record in records if record["kind"] == kind]
    normalized = [
        float(record["realized_normalized_relative_volume"])
        for record in kind_records
        if "realized_normalized_relative_volume" in record
    ]
    return {
        "count": int(volumes.size),
        "mean_volume_A3": float(np.mean(volumes)) if volumes.size else None,
        "normalized_relative_volumes": normalized,
    }


def _realized_continuous_volume_A3(
    unit: CompactUnit | ChannelUnit,
) -> float:
    if isinstance(unit, CompactUnit):
        if unit.is_multilobe:
            profile = CompactShapeProfile(
                shape_seed=int(unit.shape_seed if unit.shape_seed is not None else 0),
                lobe_centers_local_A=np.asarray(
                    unit.lobe_centers_local_A, dtype=float
                ),
                lobe_radii_A=np.asarray(unit.lobe_radii_A, dtype=float),
                smooth_length_A=float(unit.smooth_length_A),
                envelope_radii_A=np.asarray(unit.radii_A, dtype=float),
                envelope_fill_fraction=float(unit.envelope_fill_fraction or 0.0),
                centroid_offset_A=float(unit.centroid_offset_A or 0.0),
                connected=bool(unit.lobes_connected),
            )
            return estimate_multilobe_volume_A3(profile)
        return float(4.0 * np.pi * np.prod(unit.radii_A) / 3.0)

    if unit.is_variable_radius:
        profile = VariableRadiusChannelProfile(
            shape_seed=int(unit.shape_seed if unit.shape_seed is not None else 0),
            control_points_local_A=np.asarray(
                unit.control_points_unwrapped_A, dtype=float
            ),
            radius_profile_s=np.asarray(unit.radius_profile_s, dtype=float),
            radius_profile_A=np.asarray(unit.radius_profile_A, dtype=float),
            equivalent_radius_A=float(unit.cross_radius_A),
            bend_count=int(unit.bend_count or 0),
            nonplanarity=float(unit.nonplanarity or 0.0),
            minimum_self_clearance_A=float(unit.minimum_self_clearance_A or 0.0),
        )
        return estimate_variable_channel_volume_A3(profile)
    return float(
        np.pi * unit.cross_radius_A**2 * unit.arc_length_A
        + 4.0 * np.pi * unit.cross_radius_A**3 / 3.0
    )


def _realized_clipped_volume_A3(
    unit: CompactUnit | ChannelUnit,
    built: BuiltGeometry,
    grid: PhaseGrid,
) -> float:
    shape = grid.pore_mask.shape
    total = int(np.prod(shape))
    flat_indices = np.arange(total, dtype=np.int64)
    geometry = PoreGeometry([unit], built.geometry.target_box_A)
    occupied = 0
    for start in range(0, total, _OVERLAP_CHUNK_SIZE):
        stop = min(start + _OVERLAP_CHUNK_SIZE, total)
        points = _grid_points(flat_indices[start:stop], shape, grid.spacing_A, grid.origin_A)
        occupied += int(np.count_nonzero(geometry.sdf(points) < 0.0))
    return float(occupied * grid.spacing_A**3)


def _mean_volume_ratio_relative_error(
    realized_ratio: float | None,
    target_ratio: float,
) -> float | None:
    if realized_ratio is None:
        return None
    return abs(float(realized_ratio) - float(target_ratio)) / max(abs(float(target_ratio)), 1e-12)


def _theta_samples(config: GeneratorConfig, built: BuiltGeometry) -> np.ndarray:
    values = []
    near_spherical_tolerance = float(config.audit.orientation_aspect_ratio_tolerance)
    for unit in built.units:
        if isinstance(unit, CompactUnit):
            radii = np.asarray(unit.radii_A, dtype=float)
            if radii.size != 3 or np.min(radii) <= 0.0:
                continue
            aspect_ratio = float(np.max(radii) / np.min(radii))
            if aspect_ratio <= 1.0 + near_spherical_tolerance:
                continue
        record = unit.to_record()
        theta = record["realized_geometry"].get("orientation", {}).get("theta_rad")
        if theta is not None:
            values.append(float(theta) / np.pi)
    return np.asarray(values, dtype=float)


def _compact_eta_samples(built: BuiltGeometry) -> np.ndarray:
    values = []
    for unit in built.units:
        if isinstance(unit, CompactUnit):
            values.append(float(unit.radii_A[2] / unit.radii_A[0]))
    return np.asarray(values, dtype=float)


def _channel_eta_samples(built: BuiltGeometry) -> np.ndarray:
    return np.asarray(
        [float(unit.eta) for unit in built.units if isinstance(unit, ChannelUnit)],
        dtype=float,
    )


def _channel_tau_samples(built: BuiltGeometry) -> np.ndarray:
    return np.asarray(
        [float(unit.tortuosity) for unit in built.units if isinstance(unit, ChannelUnit)],
        dtype=float,
    )


def _roughness_samples_and_target(
    config: GeneratorConfig,
    built: BuiltGeometry,
) -> tuple[np.ndarray, dict[str, Any]]:
    compact_values = [float(unit.roughness) for unit in built.units if isinstance(unit, CompactUnit)]
    channel_values = [float(unit.roughness) for unit in built.units if isinstance(unit, ChannelUnit)]
    values = np.asarray(compact_values + channel_values, dtype=float)
    compact_target = _as_distribution_dict(config.compact.roughness)
    channel_target = _as_distribution_dict(config.channel.roughness)
    if compact_values and channel_values and compact_target == channel_target:
        target = compact_target
    elif compact_values and channel_values:
        total = len(compact_values) + len(channel_values)
        target = {
            "family": "mixture",
            "components": [
                {"weight": len(compact_values) / total, **compact_target},
                {"weight": len(channel_values) / total, **channel_target},
            ],
        }
    elif channel_values:
        target = _as_distribution_dict(config.channel.roughness)
    else:
        target = _as_distribution_dict(config.compact.roughness)
    return values, target


def _channel_fraction_error(config: GeneratorConfig, built: BuiltGeometry) -> float:
    if not built.units:
        actual = 0.0
    else:
        actual = sum(isinstance(unit, ChannelUnit) for unit in built.units) / len(built.units)
    return float(actual - config.pores.channel_fraction_by_count)


def _overlap_fraction(built: BuiltGeometry, grid: PhaseGrid, warnings: list[str]) -> float:
    if len(built.units) < 2:
        return 0.0
    pore_voxels = int(np.count_nonzero(grid.pore_mask))
    if pore_voxels == 0:
        return 0.0

    shape = grid.pore_mask.shape
    total = int(np.prod(shape))
    coverage = np.zeros(total, dtype=np.uint16)
    flat_indices = np.arange(total, dtype=np.int64)
    for unit in built.units:
        unit_geometry = PoreGeometry([unit], built.geometry.target_box_A)
        for start in range(0, total, _OVERLAP_CHUNK_SIZE):
            stop = min(start + _OVERLAP_CHUNK_SIZE, total)
            points = _grid_points(flat_indices[start:stop], shape, grid.spacing_A, grid.origin_A)
            coverage[start:stop] += (unit_geometry.sdf(points) < 0.0).astype(np.uint16)

    overlap_voxels = int(np.count_nonzero((coverage > 1) & grid.pore_mask.ravel()))
    if overlap_voxels and np.max(coverage) == np.iinfo(coverage.dtype).max:
        warnings.append("overlap coverage counter saturated")
    return float(overlap_voxels / pore_voxels)


def _grid_points(
    flat_indices: np.ndarray,
    shape_zyx: tuple[int, int, int],
    spacing_A: float,
    origin_A: np.ndarray,
) -> np.ndarray:
    _nz, ny, nx = shape_zyx
    x_indices = flat_indices % nx
    y_indices = (flat_indices // nx) % ny
    z_indices = flat_indices // (nx * ny)
    return np.column_stack(
        [
            origin_A[0] + (x_indices.astype(float) + 0.5) * spacing_A,
            origin_A[1] + (y_indices.astype(float) + 0.5) * spacing_A,
            origin_A[2] + (z_indices.astype(float) + 0.5) * spacing_A,
        ]
    )


def _surface_openings(
    pore_mask: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], float, float]:
    if pore_mask.size == 0:
        return (0.0, 0.0), (0.0, 0.0), 0.0, 0.0
    x_open = (float(np.mean(pore_mask[:, :, 0])), float(np.mean(pore_mask[:, :, -1])))
    y_open = (float(np.mean(pore_mask[:, 0, :])), float(np.mean(pore_mask[:, -1, :])))
    z_lower = float(np.mean(pore_mask[0, :, :]))
    z_upper = float(np.mean(pore_mask[-1, :, :]))
    return x_open, y_open, z_lower, z_upper


def _matrix_mask_for_percolation(
    mask: np.ndarray,
    spacing_A: float,
    minimum_skeleton_thickness_A: float | None,
) -> np.ndarray:
    if minimum_skeleton_thickness_A is None:
        return mask.copy()
    return _erode_by_half_thickness(mask, spacing_A, minimum_skeleton_thickness_A)


def _erode_by_half_thickness(
    mask: np.ndarray,
    spacing_A: float,
    minimum_skeleton_thickness_A: float,
) -> np.ndarray:
    if not np.any(mask):
        return mask.copy()
    distances = ndimage.distance_transform_edt(mask, sampling=spacing_A)
    return distances >= 0.5 * float(minimum_skeleton_thickness_A)


def _local_thickness_stability_result(
    config: GeneratorConfig,
    built: BuiltGeometry,
    warnings: list[str],
) -> ThicknessStabilityResult:
    if not config.audit.enabled:
        return ThicknessStabilityResult(True, 0.0, 0.0, 0.0, 0.0, None)
    coarse_spacing = float(config.audit.coarse_spacing_A)
    fine_spacing = float(config.audit.fine_spacing_A)
    if np.isclose(coarse_spacing, fine_spacing):
        return ThicknessStabilityResult(True, 0.0, 0.0, 0.0, 2.0 * fine_spacing, None)
    try:
        coarse_grid = voxelize_geometry(
            built.geometry,
            built.geometry.target_box_A,
            coarse_spacing,
        )
        fine_grid = voxelize_geometry(
            built.geometry,
            built.geometry.target_box_A,
            fine_spacing,
        )
        result = compare_local_thickness_coarse_fine(
            coarse_grid.pore_mask,
            fine_grid.pore_mask,
            coarse_spacing_A=coarse_spacing,
            fine_spacing_A=fine_spacing,
            periodic_xy=True,
        )
    except ValueError as exc:
        warning = f"coarse/fine local-thickness comparison could not be evaluated: {exc}"
        warnings.append(warning)
        return ThicknessStabilityResult(False, np.inf, np.inf, np.inf, 2.0 * fine_spacing, warning)
    if result.warning is not None:
        warnings.append(result.warning)
    return result


def _rdf_result(
    config: GeneratorConfig,
    built: BuiltGeometry,
    center_plan: CenterSeedPlan,
) -> dict[str, Any]:
    xi = np.asarray(center_plan.target_rdf_xi, dtype=float)
    target = np.asarray(center_plan.target_rdf_values, dtype=float)
    observed = np.ones_like(target)
    if xi.size and built.realized_anchors_A.shape[0] >= 2:
        box = np.array(
            [
                config.film.target_box_A.x,
                config.film.target_box_A.y,
                config.film.target_box_A.z,
            ],
            dtype=float,
        )
        distances = pair_distances_periodic_xy(built.realized_anchors_A, box)
        edges = _rdf_edges_from_xi(xi, built.realized_anchors_A.shape[0], box)
        hist, _ = np.histogram(distances, bins=edges)
        if np.any(hist):
            observed = hist.astype(float)
            observed /= float(np.mean(observed[observed > 0.0]))
    loss = float(np.average((observed - target) ** 2, weights=np.maximum(target, 1e-12)))
    return {
        "weighted_loss": loss,
        "initialization_loss": float(center_plan.initialization_loss),
        "starting_loss": float(center_plan.starting_loss),
        "target_peak_xi": _peak_locations(xi, target),
        "observed_peak_xi": _peak_locations(xi, observed),
    }


def _rdf_edges_from_xi(xi: np.ndarray, count: int, box: np.ndarray) -> np.ndarray:
    density_scale = (count / float(np.prod(box))) ** (1.0 / 3.0) if count else 1.0
    if xi.size <= 1:
        return np.array([0.0, np.inf])
    spacing = float(np.median(np.diff(xi)))
    xi_edges = np.concatenate(([max(0.0, xi[0] - 0.5 * spacing)], xi + 0.5 * spacing))
    return xi_edges / density_scale


def _peak_locations(xi: np.ndarray, values: np.ndarray) -> list[float]:
    if xi.size == 0:
        return []
    if xi.size < 3:
        return [float(xi[int(np.argmax(values))])]
    peak_mask = (values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:])
    return [float(value) for value in xi[1:-1][peak_mask]]


def _record_mixture_errors(
    output: dict[str, dict[str, float]],
    sample_counts: dict[str, int],
    name: str,
    samples: np.ndarray,
    target: Any,
) -> None:
    target_dict = _as_distribution_dict(target)
    if target_dict.get("family") != "mixture" or samples.size == 0:
        return
    components = [_as_distribution_dict(component) for component in target_dict["components"]]
    weights = np.asarray([component["weight"] for component in components], dtype=float)
    expected_counts = allocate_largest_remainder(weights, int(samples.size))
    assigned_counts = _assign_mixture_components(samples, components)
    sample_counts[name] = int(samples.size)
    output[name] = {
        f"component_{index}": float((assigned_counts[index] - expected_counts[index]) / samples.size)
        for index in range(len(components))
    }


def _assign_mixture_components(samples: np.ndarray, components: list[dict[str, Any]]) -> np.ndarray:
    prototypes = np.asarray(
        [
            float(component["value"])
            if component.get("family") == "constant"
            else float(np.median(_target_quantile_samples(component_without_weight, 129)))
            for component in components
            for component_without_weight in [{key: value for key, value in component.items() if key != "weight"}]
        ],
        dtype=float,
    )
    distances = np.abs(samples[:, np.newaxis] - prototypes[np.newaxis, :])
    return np.bincount(np.argmin(distances, axis=1), minlength=len(components))


def _ks_distance(samples: np.ndarray, target: dict[str, Any]) -> float:
    if target.get("family") == "constant":
        value = float(target["value"])
        return 0.0 if np.allclose(samples, value) else 1.0
    sorted_samples = np.sort(samples)
    n = sorted_samples.size
    cdf = mixture_cdf(target, sorted_samples)
    empirical_right = np.arange(1, n + 1, dtype=float) / n
    empirical_left = np.arange(0, n, dtype=float) / n
    return float(max(np.max(np.abs(empirical_right - cdf)), np.max(np.abs(cdf - empirical_left))))


def _target_quantile_samples(target: dict[str, Any], count: int) -> np.ndarray:
    quantiles = (np.arange(count, dtype=float) + 0.5) / count
    if target.get("family") == "mixture":
        components = [_as_distribution_dict(component) for component in target["components"]]
        weights = np.asarray([component["weight"] for component in components], dtype=float)
        counts = allocate_largest_remainder(weights, count)
        values = []
        for component, component_count in zip(components, counts, strict=True):
            if component_count:
                component_target = {key: value for key, value in component.items() if key != "weight"}
                values.append(_target_quantile_samples(component_target, int(component_count)))
        return np.sort(np.concatenate(values)) if values else np.array([], dtype=float)
    return _distribution_ppf(target, quantiles)


def _distribution_ppf(target: dict[str, Any], quantiles: np.ndarray) -> np.ndarray:
    family = target["family"]
    if family == "constant":
        return np.full_like(quantiles, float(target["value"]), dtype=float)
    if family == "lognormal":
        sigma = _required_float(target, "sigma", "s")
        loc = float(target.get("loc", 0.0))
        scale = float(target.get("scale", np.exp(float(target.get("mean", target.get("mu", 0.0))))))
        return stats.lognorm(s=sigma, loc=loc, scale=scale).ppf(quantiles)
    if family == "gamma":
        shape = _required_float(target, "alpha", "shape", "k")
        scale = float(target.get("scale", target.get("theta", 1.0)))
        loc = float(target.get("loc", 0.0))
        return stats.gamma(a=shape, loc=loc, scale=scale).ppf(quantiles)
    if family in {"weibull", "weibull_min"}:
        shape = _required_float(target, "shape", "k", "alpha")
        scale = float(target.get("scale", 1.0))
        loc = float(target.get("loc", 0.0))
        return stats.weibull_min(c=shape, loc=loc, scale=scale).ppf(quantiles)
    if family in {"truncated_normal", "truncnorm"}:
        mean = float(target.get("mean", target.get("loc", 0.0)))
        sigma = _required_float(target, "sigma", "s")
        lower = float(target["lower"])
        upper = float(target["upper"])
        return stats.truncnorm(
            a=(lower - mean) / sigma,
            b=(upper - mean) / sigma,
            loc=mean,
            scale=sigma,
        ).ppf(quantiles)
    if family == "beta":
        alpha = _required_float(target, "alpha")
        beta = _required_float(target, "beta")
        lower = float(target.get("lower", target.get("minimum", 0.0)))
        upper = float(target.get("upper", target.get("maximum", lower + target.get("scale", 1.0))))
        return stats.beta(a=alpha, b=beta, loc=lower, scale=upper - lower).ppf(quantiles)
    raise ValueError(f"unsupported distribution family: {family}")


def _normalization_scale(target_values: np.ndarray, target: dict[str, Any]) -> float:
    span = float(np.max(target_values) - np.min(target_values)) if target_values.size else 0.0
    if span > 0.0:
        return span
    if target.get("family") == "constant":
        return max(abs(float(target["value"])), 1.0)
    return max(abs(float(np.mean(target_values))) if target_values.size else 0.0, 1.0)


def _as_distribution_dict(target: Any) -> dict[str, Any]:
    if isinstance(target, BaseModel):
        return target.model_dump(exclude_none=True)
    if isinstance(target, dict):
        return {key: value for key, value in target.items() if value is not None}
    if hasattr(target, "model_dump"):
        return target.model_dump(exclude_none=True)
    raise TypeError("distribution target must be a mapping or model")


def _finite_1d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _required_float(target: dict[str, Any], *names: str) -> float:
    for name in names:
        if name in target:
            return float(target[name])
    joined = ", ".join(names)
    raise ValueError(f"distribution requires one of: {joined}")
