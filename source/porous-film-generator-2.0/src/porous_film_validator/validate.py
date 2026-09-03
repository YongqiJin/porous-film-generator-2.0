"""Independent export validation entry points."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from hashlib import blake2b
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import trimesh
import yaml
from scipy import ndimage, stats
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq, linear_sum_assignment
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp
from scipy.stats import qmc
from skimage.feature import peak_local_max
from skimage.measure import find_contours

_AVOGADRO_PER_MOL = 6.022_140_76e23
_ANGSTROM3_TO_CM3 = 1.0e-24
_POROSITY_TARGET_TOLERANCE = 0.01
_GLB_OCCUPANCY_WARNING_TOLERANCE = 0.05
_GLB_OCCUPANCY_ERROR_TOLERANCE = 0.25
_GLB_OCCUPANCY_MAX_SAMPLES = 8192
_RAY_POINT_CHUNK = 128
_RAY_TRIANGLE_CHUNK = 512
_DENSITY_TARGET_RELATIVE_TOLERANCE = 0.02
_V3_COORD_ABS_TOLERANCE_A = 1.0e-6
_V3_SCALAR_ABS_TOLERANCE = 1.0e-6
_V3_SCALAR_REL_TOLERANCE = 1.0e-6
_V3_GXY_ABS_TOLERANCE = 1.0e-8
_V3_GXY_REL_TOLERANCE = 1.0e-6
_V3_DISTRIBUTION_KS_LIMIT = 0.20
_V3_DISTRIBUTION_WASSERSTEIN_LIMIT = 0.20
_V3_RDF_LOSS_LIMIT = 1.0
_V3_MIXTURE_WEIGHT_ABSOLUTE_TOLERANCE = 0.05
_V3_COMPACT_ETA_CONSTANT_RELATIVE_TOLERANCE = 0.05
_V3_CHANNEL_GEOMETRY_CONSTANT_RELATIVE_TOLERANCE = 0.01
_OVERLAP_CHUNK_SIZE = 250_000
_SMOOTH_UNION_SHARPNESS = 32.0
_ROUGHNESS_MODE_COUNT = 4

_CHECKSUM_REQUIRED_FILES = (
    "contract.json",
    "normalized_config.yaml",
    "unit_candidates.jsonl",
    "unit_geometry.jsonl",
    "channel_curves.h5",
    "final_phase.h5",
    "final_surface.ply",
    "semiconductor_solid_target.glb",
    "main_unit_metrics.csv",
    "main_metrics.json",
    "molecules/instances.csv",
    "molecules/placed_atoms.h5",
    "molecules/placed_structure.cif",
)

_MANDATORY_EXPORT_ENTRIES = (
    "contract.json",
    "normalized_config.yaml",
    "unit_candidates.jsonl",
    "unit_geometry.jsonl",
    "channel_curves.h5",
    "final_phase.h5",
    "final_surface.ply",
    "semiconductor_solid_target.glb",
    "main_unit_metrics.csv",
    "main_metrics.json",
    "molecules/source",
    "molecules/instances.csv",
    "molecules/placed_atoms.h5",
    "molecules/placed_structure.cif",
    "checksums.sha256",
)


_V3_CHECKSUM_REQUIRED_FILES = (
    "contract.json",
    "normalized_config.yaml",
    "unit_candidates.jsonl",
    "unit_geometry.jsonl",
    "channel_curves.h5",
    "final_phase.h5",
    "final_centerlines.h5",
    "final_cross_sections.csv",
    "final_measurements.json",
    "final_surface.ply",
    "semiconductor_solid_target.glb",
    "main_unit_metrics.csv",
    "main_metrics.json",
)

_V3_MANDATORY_EXPORT_ENTRIES = (*_V3_CHECKSUM_REQUIRED_FILES, "checksums.sha256")


@dataclass(frozen=True)
class _ChannelCurveData:
    centerline_A: np.ndarray
    radius_A: np.ndarray | None
    schema_version: int
    shape_model: str | None = None
    shape_seed: int | None = None
    equivalent_radius_A: float | None = None


@dataclass(frozen=True)
class ValidationReport:
    status: str
    independent_metrics: dict[str, Any] = field(default_factory=dict)
    report_consistency: dict[str, Any] = field(default_factory=dict)
    target_compliance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def validate_export(qa_export: Path) -> ValidationReport:
    qa = Path(qa_export)
    errors: list[str] = []
    warnings: list[str] = []

    if not qa.is_dir():
        errors.append(f"mandatory QA export directory is missing: {qa}")
        report = ValidationReport("NOT_EVALUABLE", errors=errors)
        _write_reports(qa, report)
        return report

    config = _read_yaml(qa / "normalized_config.yaml", errors)
    schema_version = _config_schema_version(config)
    mandatory_entries = (
        _V3_MANDATORY_EXPORT_ENTRIES if schema_version >= 3 else _MANDATORY_EXPORT_ENTRIES
    )
    missing = [entry for entry in mandatory_entries if not (qa / entry).exists()]
    if missing:
        errors.append("mandatory QA export entries are missing: " + ", ".join(missing))

    contract = _read_json(qa / "contract.json", errors)
    phase_metrics, phase_data = _read_phase_metrics(
        qa / "final_phase.h5",
        contract,
        config,
        errors,
    )
    unit_metrics = _read_unit_metrics(
        qa / "unit_geometry.jsonl",
        qa / "channel_curves.h5",
        phase_data,
        errors,
        warnings,
    )
    molecule_metrics = (
        _read_molecule_metrics(qa / "molecules", errors, warnings)
        if schema_version < 3 or isinstance(config.get("pore_material"), dict)
        else {}
    )
    final_geometry_metrics = (
        _read_v3_final_geometry_metrics(qa, phase_data, config, errors, warnings)
        if schema_version >= 3
        else {}
    )
    glb_metrics = _read_glb_metrics(qa, phase_data, errors, warnings)
    checksum_metrics = _verify_checksums(
        qa,
        errors,
        schema_version=schema_version,
    )
    independent_metrics = {
        "phase": phase_metrics,
        "units": unit_metrics,
        "molecules": molecule_metrics,
        "final_geometry": final_geometry_metrics,
        "glb": glb_metrics,
        "checksums": checksum_metrics,
    }
    main_metrics = _read_json(qa / "main_metrics.json", errors)
    consistency = _compare_main_metrics(main_metrics, independent_metrics, errors, warnings)
    compliance = _target_compliance(
        config,
        contract,
        phase_metrics,
        unit_metrics,
        molecule_metrics,
        final_geometry_metrics,
        errors,
    )

    status = _status(errors, warnings, missing)
    report = ValidationReport(
        status=status,
        independent_metrics=independent_metrics,
        report_consistency=consistency,
        target_compliance=compliance,
        warnings=warnings,
        errors=errors,
    )
    _write_reports(qa, report)
    return report


@dataclass(frozen=True)
class _PhaseData:
    mask: np.ndarray
    spacing_A: float
    target_box_A: np.ndarray
    origin_A: np.ndarray


@dataclass(frozen=True)
class _V3MeasurementContract:
    z_slice_spacing_A: float = 1.0
    center_min_separation_A: float = 2.0
    center_tracking_max_displacement_A: float = 4.0
    center_distance_bin_width_A: float = 1.0
    center_distance_max_A: float | None = None
    center_distance_reference_samples: int = 4096
    centerline_sample_spacing_A: float = 2.0
    cross_section_spacing_A: float = 2.0
    boundary_resample_spacing_A: float = 0.5
    curvature_smoothing_length_A: float = 1.0
    branch_exclusion_length_A: float = 2.0
    surface_exclusion_length_A: float = 2.0
    orientation_projection_min_fraction: float = 0.05
    orientation_aspect_ratio_tolerance: float = 1.0e-6


@dataclass(frozen=True)
class _V3SliceCenter:
    xy_A: np.ndarray
    wall_distance_A: float


@dataclass(frozen=True)
class _V3SliceCenterRecord:
    z_index: int
    z_A: float
    centers: tuple[_V3SliceCenter, ...]


@dataclass(frozen=True)
class _V3CenterlineTrack:
    track_id: int
    slice_indices: np.ndarray
    points_wrapped_A: np.ndarray
    points_unwrapped_A: np.ndarray
    wall_distances_A: np.ndarray
    touches_z_lower: bool
    touches_z_upper: bool
    is_through: bool
    has_branch_neighborhood: bool


@dataclass(frozen=True)
class _V3XYCenterDistanceDistribution:
    bin_edges_A: np.ndarray
    bin_centers_A: np.ndarray
    observed_pair_counts: np.ndarray
    reference_pair_counts: np.ndarray
    g_xy: np.ndarray
    pair_count: int
    valid_slice_count: int


@dataclass(frozen=True)
class _V3CrossSectionMeasurement:
    track_id: int
    arc_position_A: float
    center_A: np.ndarray
    tangent: np.ndarray
    area_A2: float | None
    equivalent_diameter_A: float | None
    curvature_fluctuation: float | None
    valid: bool
    invalid_reason: str | None


@dataclass(frozen=True)
class _V3ProjectedOrientationMeasurement:
    track_id: int
    axis: np.ndarray
    theta_xz_deg: float | None
    theta_xy_deg: float | None
    theta_xz_identifiable: bool
    theta_xy_identifiable: bool


@dataclass(frozen=True)
class _V3ChannelGeometryMeasurement:
    track_id: int
    arc_length_A: float | None
    end_distance_A: float | None
    equivalent_diameter_A: float | None
    eta: float | None
    tortuosity: float | None
    valid: bool
    invalid_reason: str | None


@dataclass(frozen=True)
class _V3CompactGeometryMeasurement:
    component_id: int
    voxel_count: int
    eta: float | None
    valid: bool
    invalid_reason: str | None


@dataclass(frozen=True)
class _V3FinalGeometryMeasurements:
    porosity: float
    slice_centers: tuple[_V3SliceCenterRecord, ...]
    centerlines: tuple[_V3CenterlineTrack, ...]
    through_centerline_count: int
    branch_event_count: int
    center_distance_xy: _V3XYCenterDistanceDistribution
    cross_sections: tuple[_V3CrossSectionMeasurement, ...]
    projected_orientations: tuple[_V3ProjectedOrientationMeasurement, ...]
    channel_geometries: tuple[_V3ChannelGeometryMeasurement, ...]
    compact_geometries: tuple[_V3CompactGeometryMeasurement, ...]


@dataclass
class _V3MutableTrack:
    track_id: int
    slice_indices: list[int]
    wrapped_points_A: list[np.ndarray]
    unwrapped_points_A: list[np.ndarray]
    wall_distances_A: list[float]
    branch_z_A: list[float]


def _read_phase_metrics(
    path: Path,
    contract: dict[str, Any],
    config: dict[str, Any],
    errors: list[str],
) -> tuple[dict[str, Any], _PhaseData | None]:
    if not path.is_file():
        return {}, None
    try:
        with h5py.File(path, "r") as handle:
            if "pore_mask" not in handle:
                errors.append("mandatory final_phase.h5 dataset is missing: pore_mask")
                return {}, None
            mask = np.asarray(handle["pore_mask"], dtype=bool)
            spacing = float(handle.attrs.get("spacing_A", np.nan))
            target_box = np.asarray(handle.attrs.get("target_box_A", []), dtype=float)
            origin = np.asarray(handle.attrs.get("origin_A", [0.0, 0.0, 0.0]), dtype=float)
            axis_order = _text_attr(handle.attrs.get("axis_order", ""))
    except (OSError, ValueError) as exc:
        errors.append(f"mandatory final_phase.h5 could not be read: {exc}")
        return {}, None

    if mask.ndim != 3:
        errors.append("mandatory final_phase.h5 pore_mask must be a 3-D array")
        return {}, None
    if axis_order != "zyx":
        errors.append("mandatory final_phase.h5 axis_order must be zyx")
    if target_box.shape != (3,) or not np.all(np.isfinite(target_box)):
        errors.append("mandatory final_phase.h5 target_box_A must contain three finite values")
        target_box = np.zeros(3, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        errors.append("mandatory final_phase.h5 origin_A must contain three finite values")
        origin = np.zeros(3, dtype=float)
    if not np.isfinite(spacing) or spacing <= 0.0:
        errors.append("mandatory final_phase.h5 spacing_A must be positive")
        spacing = math.nan
    if (
        np.isfinite(spacing)
        and spacing > 0.0
        and target_box.shape == (3,)
        and np.all(np.isfinite(target_box))
    ):
        expected_shape = tuple(round(float(length) / spacing) for length in target_box[::-1])
        if tuple(mask.shape) != expected_shape:
            errors.append(
                "final_phase.h5 mask.shape is inconsistent with "
                "target_box_A and spacing_A: "
                f"mask.shape={tuple(mask.shape)}, expected={expected_shape}"
            )

    _validate_contract(contract, target_box, errors)
    pore_components = _periodic_xy_component_count(mask)
    semiconductor = ~mask
    min_cross_section = _minimum_cross_section_fraction(semiconductor)
    minimum_skeleton_thickness_A = _configured_minimum_skeleton_thickness_A(config)
    matrix_for_percolation = _matrix_mask_for_percolation(
        semiconductor,
        spacing,
        minimum_skeleton_thickness_A,
    )
    phase_data = _PhaseData(
        mask=mask,
        spacing_A=spacing,
        target_box_A=target_box,
        origin_A=origin,
    )

    return {
        "porosity": float(np.mean(mask)),
        "phase_dimensions_zyx": [int(value) for value in mask.shape],
        "spacing_A": spacing,
        "target_box_A": target_box.tolist() if target_box.shape == (3,) else [],
        "origin_A": origin.tolist(),
        "connected_pore_components": pore_components,
        "semiconductor_x_percolates": _percolates_x(semiconductor),
        "matrix_x_percolates": _percolates_x(matrix_for_percolation),
        "matrix_minimum_skeleton_thickness_A": minimum_skeleton_thickness_A,
        "minimum_semiconductor_cross_section_fraction": min_cross_section,
        "pore_voxels": int(np.count_nonzero(mask)),
        "semiconductor_voxels": int(np.count_nonzero(semiconductor)),
    }, phase_data


def _read_unit_metrics(
    path: Path,
    channel_curve_path: Path,
    phase: _PhaseData | None,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    records = _read_jsonl(path, errors)
    channel_curves = _read_channel_curves(channel_curve_path, errors)
    kinds = [str(record.get("kind", "")).lower() for record in records]
    compact_count = sum(kind == "compact" for kind in kinds)
    channel_count = sum(kind == "channel" for kind in kinds)
    anchors = np.asarray(
        [
            _anchor_from_record(record)
            for record in records
            if _anchor_from_record(record) is not None
        ],
        dtype=float,
    ).reshape((-1, 3))
    volume_samples = [
        _unit_volume_A3(record, channel_curves, errors, warnings) for record in records
    ]
    finite_volumes = np.asarray(
        [volume for volume in volume_samples if volume is not None and np.isfinite(volume)],
        dtype=float,
    )
    distances = (
        _pair_distances_periodic_xy(anchors, phase.target_box_A)
        if phase is not None and anchors.shape[0] >= 2
        else np.empty(0, dtype=float)
    )
    overlap_fraction = (
        _independent_overlap_fraction(records, channel_curves, phase, errors)
        if phase is not None
        else math.nan
    )
    return {
        "total_count": len(records),
        "compact_count": int(compact_count),
        "channel_count": int(channel_count),
        "channel_fraction": float(channel_count / len(records)) if records else 0.0,
        "volume_sample_count": int(finite_volumes.size),
        "volume_mean_A3": float(np.mean(finite_volumes)) if finite_volumes.size else None,
        "volume_std_A3": float(np.std(finite_volumes)) if finite_volumes.size else None,
        "rdf_pair_count": int(distances.size),
        "rdf_distance_mean_A": float(np.mean(distances)) if distances.size else None,
        "overlap_fraction": overlap_fraction if np.isfinite(overlap_fraction) else None,
    }


def _read_channel_curves(
    path: Path,
    errors: list[str],
) -> dict[str, _ChannelCurveData]:
    if not path.is_file():
        return {}
    curves: dict[str, _ChannelCurveData] = {}
    try:
        with h5py.File(path, "r") as handle:
            schema_version = int(handle.attrs.get("schema_version", 1))
            for unit_id, item in handle.items():
                if isinstance(item, h5py.Dataset):
                    samples = np.asarray(item, dtype=float)
                    radii = None
                    item_schema = 1
                elif isinstance(item, h5py.Group):
                    try:
                        samples = np.asarray(item["centerline_A"], dtype=float)
                        radii = np.asarray(item["radius_A"], dtype=float)
                    except (KeyError, ValueError, TypeError) as exc:
                        errors.append(f"channel_curves.h5 group {unit_id} is invalid: {exc}")
                        continue
                    item_schema = max(schema_version, 2)
                else:
                    errors.append(
                        f"channel_curves.h5 entry {unit_id} is neither a dataset nor group"
                    )
                    continue
                if samples.ndim != 2 or samples.shape[1] != 3 or samples.shape[0] < 2:
                    errors.append(
                        f"channel_curves.h5 centerline {unit_id} must have shape (n>=2, 3)"
                    )
                    continue
                if not np.all(np.isfinite(samples)):
                    errors.append(
                        f"channel_curves.h5 centerline {unit_id} contains non-finite samples"
                    )
                    continue
                if radii is not None:
                    if radii.shape != (samples.shape[0],):
                        errors.append(
                            f"channel_curves.h5 radius_A {unit_id} must match centerline length"
                        )
                        continue
                    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
                        errors.append(
                            f"channel_curves.h5 radius_A {unit_id} must contain positive finite values"
                        )
                        continue
                shape_model = None
                shape_seed = None
                equivalent_radius = None
                if isinstance(item, h5py.Group):
                    raw_model = item.attrs.get("shape_model")
                    if raw_model is not None:
                        shape_model = (
                            raw_model.decode("utf-8")
                            if isinstance(raw_model, bytes)
                            else str(raw_model)
                        )
                    if "shape_seed" in item.attrs:
                        shape_seed = int(item.attrs["shape_seed"])
                    if "equivalent_radius_A" in item.attrs:
                        equivalent_radius = float(item.attrs["equivalent_radius_A"])
                curves[str(unit_id)] = _ChannelCurveData(
                    centerline_A=samples,
                    radius_A=radii,
                    schema_version=item_schema,
                    shape_model=shape_model,
                    shape_seed=shape_seed,
                    equivalent_radius_A=equivalent_radius,
                )
    except (OSError, ValueError, TypeError) as exc:
        errors.append(f"mandatory channel_curves.h5 could not be read: {exc}")
    return curves


def _config_schema_version(config: dict[str, Any]) -> int:
    try:
        return int(config.get("source_schema_version", config.get("schema_version", 1)))
    except (TypeError, ValueError):
        return 1


def _read_v3_final_geometry_metrics(
    qa: Path,
    phase: _PhaseData | None,
    config: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    if phase is None:
        return {}
    contract = _v3_measurement_contract(config, errors)
    rebuilt = _v3_measure_final_geometry(phase, contract)
    through_network_count = _through_z_network_count(phase.mask)
    centerline_path = qa / "final_centerlines.h5"
    cross_section_path = qa / "final_cross_sections.csv"
    measurement_path = qa / "final_measurements.json"

    reported_centerlines, centerline_point_count, centerline_points_inside = (
        _compare_v3_centerline_h5(centerline_path, rebuilt, phase, errors)
    )
    csv_valid_count, csv_invalid_count = _compare_v3_cross_section_csv(
        cross_section_path,
        rebuilt,
        errors,
    )
    _compare_v3_final_measurements_json(
        _read_json(measurement_path, errors),
        rebuilt,
        errors,
    )

    inside_fraction = (
        centerline_points_inside / centerline_point_count if centerline_point_count else math.nan
    )
    if np.isfinite(inside_fraction) and inside_fraction < 0.95:
        errors.append("reported final centerline points are not contained in final pore phase")
    if through_network_count == 0 and reported_centerlines:
        warnings.append("reported centerlines exist but no z-through pore network was recomputed")
    formal_samples = _v3_formal_measurement_samples(rebuilt)
    return {
        "through_network_count": through_network_count,
        "connected_pore_component_count": _periodic_xy_component_count(phase.mask),
        "rebuilt_centerline_count": len(rebuilt.centerlines),
        "rebuilt_through_centerline_count": rebuilt.through_centerline_count,
        "rebuilt_branch_event_count": rebuilt.branch_event_count,
        "rebuilt_slice_count": len(rebuilt.slice_centers),
        "rebuilt_slice_center_count": int(
            sum(len(slice_record.centers) for slice_record in rebuilt.slice_centers)
        ),
        "rebuilt_g_xy_pair_count": rebuilt.center_distance_xy.pair_count,
        "reported_centerline_count": reported_centerlines,
        "centerline_point_count": centerline_point_count,
        "centerline_points_inside_phase": centerline_points_inside,
        "centerline_point_inside_fraction": inside_fraction
        if np.isfinite(inside_fraction)
        else None,
        "valid_cross_section_count": csv_valid_count,
        "invalid_cross_section_count": csv_invalid_count,
        "rebuilt_valid_cross_section_count": int(
            sum(section.valid for section in rebuilt.cross_sections)
        ),
        "rebuilt_invalid_cross_section_count": int(
            sum(not section.valid for section in rebuilt.cross_sections)
        ),
        "rebuilt_projected_orientation_count": len(rebuilt.projected_orientations),
        "rebuilt_valid_channel_geometry_count": int(
            sum(channel.valid for channel in rebuilt.channel_geometries)
        ),
        "rebuilt_valid_compact_geometry_count": int(
            sum(compact.valid for compact in rebuilt.compact_geometries)
        ),
        "rebuilt_center_distance_xy": {
            "bin_centers_A": rebuilt.center_distance_xy.bin_centers_A.tolist(),
            "g_xy": rebuilt.center_distance_xy.g_xy.tolist(),
            "reference_pair_counts": rebuilt.center_distance_xy.reference_pair_counts.tolist(),
            "observed_pair_counts": rebuilt.center_distance_xy.observed_pair_counts.tolist(),
            "pair_count": rebuilt.center_distance_xy.pair_count,
            "valid_slice_count": rebuilt.center_distance_xy.valid_slice_count,
        },
        "rebuilt_formal_samples": {key: values.tolist() for key, values in formal_samples.items()},
    }


def _v3_formal_measurement_samples(
    rebuilt: _V3FinalGeometryMeasurements,
) -> dict[str, np.ndarray]:
    through_ids = {track.track_id for track in rebuilt.centerlines if track.is_through}
    return {
        "equivalent_diameter_A": np.asarray(
            [
                float(section.equivalent_diameter_A)
                for section in rebuilt.cross_sections
                if section.valid
                and section.track_id in through_ids
                and section.equivalent_diameter_A is not None
            ],
            dtype=float,
        ),
        "theta_xz_deg": np.asarray(
            [
                float(value.theta_xz_deg)
                for value in rebuilt.projected_orientations
                if value.track_id in through_ids
                and value.theta_xz_identifiable
                and value.theta_xz_deg is not None
            ],
            dtype=float,
        ),
        "theta_xy_deg": np.asarray(
            [
                float(value.theta_xy_deg)
                for value in rebuilt.projected_orientations
                if value.track_id in through_ids
                and value.theta_xy_identifiable
                and value.theta_xy_deg is not None
            ],
            dtype=float,
        ),
        "channel_eta": np.asarray(
            [
                float(value.eta)
                for value in rebuilt.channel_geometries
                if value.track_id in through_ids and value.valid and value.eta is not None
            ],
            dtype=float,
        ),
        "channel_tau": np.asarray(
            [
                float(value.tortuosity)
                for value in rebuilt.channel_geometries
                if value.track_id in through_ids and value.valid and value.tortuosity is not None
            ],
            dtype=float,
        ),
        "compact_eta": np.asarray(
            [
                float(value.eta)
                for value in rebuilt.compact_geometries
                if value.valid and value.eta is not None
            ],
            dtype=float,
        ),
        "paired_orientation": np.asarray(
            [
                [float(value.theta_xz_deg), float(value.theta_xy_deg)]
                for value in rebuilt.projected_orientations
                if value.track_id in through_ids
                and value.theta_xz_identifiable
                and value.theta_xy_identifiable
                and value.theta_xz_deg is not None
                and value.theta_xy_deg is not None
            ],
            dtype=float,
        ).reshape((-1, 2)),
        "curvature_fluctuation": np.asarray(
            [
                float(section.curvature_fluctuation)
                for section in rebuilt.cross_sections
                if section.valid
                and section.track_id in through_ids
                and section.curvature_fluctuation is not None
                and np.isfinite(section.curvature_fluctuation)
            ],
            dtype=float,
        ),
    }


def _v3_measurement_contract(
    config: dict[str, Any],
    errors: list[str],
) -> _V3MeasurementContract:
    configured_measurement = config.get("measurement", {})
    raw = dict(configured_measurement) if isinstance(configured_measurement, dict) else {}
    if "orientation_aspect_ratio_tolerance" not in raw:
        for legacy_key in ("audit", "geometry_audit"):
            legacy = config.get(legacy_key)
            if isinstance(legacy, dict) and "orientation_aspect_ratio_tolerance" in legacy:
                raw["orientation_aspect_ratio_tolerance"] = legacy[
                    "orientation_aspect_ratio_tolerance"
                ]
                break
    return _V3MeasurementContract(
        z_slice_spacing_A=_v3_positive_float(raw, "z_slice_spacing_A", 1.0, errors),
        center_min_separation_A=_v3_positive_float(
            raw,
            "center_min_separation_A",
            2.0,
            errors,
        ),
        center_tracking_max_displacement_A=_v3_positive_float(
            raw,
            "center_tracking_max_displacement_A",
            4.0,
            errors,
        ),
        center_distance_bin_width_A=_v3_positive_float(
            raw,
            "center_distance_bin_width_A",
            1.0,
            errors,
        ),
        center_distance_max_A=_v3_optional_positive_float(
            raw,
            "center_distance_max_A",
            errors,
        ),
        center_distance_reference_samples=_v3_positive_int(
            raw,
            "center_distance_reference_samples",
            4096,
            errors,
        ),
        centerline_sample_spacing_A=_v3_positive_float(
            raw,
            "centerline_sample_spacing_A",
            2.0,
            errors,
        ),
        cross_section_spacing_A=_v3_positive_float(
            raw,
            "cross_section_spacing_A",
            2.0,
            errors,
        ),
        boundary_resample_spacing_A=_v3_positive_float(
            raw,
            "boundary_resample_spacing_A",
            0.5,
            errors,
        ),
        curvature_smoothing_length_A=_v3_nonnegative_float(
            raw,
            "curvature_smoothing_length_A",
            1.0,
            errors,
        ),
        branch_exclusion_length_A=_v3_nonnegative_float(
            raw,
            "branch_exclusion_length_A",
            2.0,
            errors,
        ),
        surface_exclusion_length_A=_v3_nonnegative_float(
            raw,
            "surface_exclusion_length_A",
            2.0,
            errors,
        ),
        orientation_projection_min_fraction=_v3_fraction_float(
            raw,
            "orientation_projection_min_fraction",
            0.05,
            errors,
        ),
        orientation_aspect_ratio_tolerance=_v3_nonnegative_float(
            raw,
            "orientation_aspect_ratio_tolerance",
            1.0e-6,
            errors,
        ),
    )


def _v3_positive_float(
    data: dict[str, Any],
    key: str,
    default: float,
    errors: list[str],
) -> float:
    value = _optional_float(data.get(key, default))
    if np.isfinite(value) and value > 0.0:
        return float(value)
    if key in data:
        errors.append(f"measurement.{key} must be positive")
    return default


def _v3_nonnegative_float(
    data: dict[str, Any],
    key: str,
    default: float,
    errors: list[str],
) -> float:
    value = _optional_float(data.get(key, default))
    if np.isfinite(value) and value >= 0.0:
        return float(value)
    if key in data:
        errors.append(f"measurement.{key} must be nonnegative")
    return default


def _v3_optional_positive_float(
    data: dict[str, Any],
    key: str,
    errors: list[str],
) -> float | None:
    if data.get(key) is None:
        return None
    value = _optional_float(data.get(key))
    if np.isfinite(value) and value > 0.0:
        return float(value)
    errors.append(f"measurement.{key} must be positive when provided")
    return None


def _v3_positive_int(
    data: dict[str, Any],
    key: str,
    default: int,
    errors: list[str],
) -> int:
    try:
        value = int(data.get(key, default))
    except (TypeError, ValueError):
        if key in data:
            errors.append(f"measurement.{key} must be a positive integer")
        return default
    if value > 0:
        return value
    if key in data:
        errors.append(f"measurement.{key} must be a positive integer")
    return default


def _v3_fraction_float(
    data: dict[str, Any],
    key: str,
    default: float,
    errors: list[str],
) -> float:
    value = _optional_float(data.get(key, default))
    if np.isfinite(value) and 0.0 < value < 1.0:
        return float(value)
    if key in data:
        errors.append(f"measurement.{key} must be between 0 and 1")
    return default


def _compare_v3_centerline_h5(
    path: Path,
    rebuilt: _V3FinalGeometryMeasurements,
    phase: _PhaseData,
    errors: list[str],
) -> tuple[int, int, int]:
    reported_centerlines = 0
    point_count = 0
    points_inside = 0
    if not path.is_file():
        return reported_centerlines, point_count, points_inside
    try:
        with h5py.File(path, "r") as handle:
            if int(handle.attrs.get("schema_version", 0)) != 3:
                errors.append("final_centerlines.h5 schema_version must be 3")
            group = handle.get("centerlines")
            if not isinstance(group, h5py.Group):
                errors.append("final_centerlines.h5 centerlines group is missing")
                return reported_centerlines, point_count, points_inside
            reported_centerlines = len(group)
            if reported_centerlines != len(rebuilt.centerlines):
                errors.append(
                    "final_centerlines.h5 centerline count does not match "
                    "independent final_phase reconstruction: "
                    f"reported={reported_centerlines}, rebuilt={len(rebuilt.centerlines)}"
                )
            for track in rebuilt.centerlines:
                track_name = str(track.track_id)
                if track_name not in group:
                    errors.append(f"final_centerlines.h5 missing rebuilt track {track.track_id}")
                    continue
                track_group = group[track_name]
                points = np.asarray(
                    track_group.get("points_wrapped_A", []),
                    dtype=float,
                )
                if points.ndim != 2 or points.shape[1:] != (3,):
                    errors.append(
                        f"final_centerlines.h5 track {track.track_id} points must have shape (n, 3)"
                    )
                    continue
                point_count += int(points.shape[0])
                points_inside += _count_points_inside_phase(points, phase)
                _v3_compare_array(
                    errors,
                    f"final_centerlines.h5 track {track.track_id} slice_indices",
                    np.asarray(track_group.get("slice_indices", []), dtype=int),
                    track.slice_indices,
                    atol=0.0,
                    rtol=0.0,
                )
                _v3_compare_array(
                    errors,
                    f"final_centerlines.h5 track {track.track_id} points_wrapped_A",
                    points,
                    track.points_wrapped_A,
                    atol=_V3_COORD_ABS_TOLERANCE_A,
                    rtol=0.0,
                )
                _v3_compare_array(
                    errors,
                    f"final_centerlines.h5 track {track.track_id} points_unwrapped_A",
                    np.asarray(track_group.get("points_unwrapped_A", []), dtype=float),
                    track.points_unwrapped_A,
                    atol=_V3_COORD_ABS_TOLERANCE_A,
                    rtol=0.0,
                )
                _v3_compare_array(
                    errors,
                    f"final_centerlines.h5 track {track.track_id} wall_distances_A",
                    np.asarray(track_group.get("wall_distances_A", []), dtype=float),
                    track.wall_distances_A,
                    atol=_V3_SCALAR_ABS_TOLERANCE,
                    rtol=_V3_SCALAR_REL_TOLERANCE,
                )
                for name, expected in (
                    ("touches_z_lower", track.touches_z_lower),
                    ("touches_z_upper", track.touches_z_upper),
                    ("is_through", track.is_through),
                    ("has_branch_neighborhood", track.has_branch_neighborhood),
                ):
                    _v3_compare_bool(
                        errors,
                        f"final_centerlines.h5 track {track.track_id} {name}",
                        track_group.attrs.get(name),
                        expected,
                    )
    except (OSError, ValueError, TypeError) as exc:
        errors.append(f"final_centerlines.h5 could not be independently read: {exc}")
    return reported_centerlines, point_count, points_inside


def _compare_v3_cross_section_csv(
    path: Path,
    rebuilt: _V3FinalGeometryMeasurements,
    errors: list[str],
) -> tuple[int, int]:
    valid_count = 0
    invalid_count = 0
    if not path.is_file():
        return valid_count, invalid_count
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        errors.append(f"final_cross_sections.csv could not be read: {exc}")
        return valid_count, invalid_count
    expected = rebuilt.cross_sections
    if len(rows) != len(expected):
        errors.append(
            "final_cross_sections.csv row count does not match independent "
            f"final_phase reconstruction: reported={len(rows)}, rebuilt={len(expected)}"
        )
    for index, (row, section) in enumerate(zip(rows, expected, strict=False)):
        row_name = f"final_cross_sections.csv row {index}"
        is_valid = _v3_parse_bool(row.get("valid"))
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
        _v3_compare_int(errors, f"{row_name} track_id", row.get("track_id"), section.track_id)
        _v3_compare_scalar(
            errors,
            f"{row_name} arc_position_A",
            _optional_float(row.get("arc_position_A")),
            section.arc_position_A,
        )
        _v3_compare_array(
            errors,
            f"{row_name} center_A",
            np.asarray(
                [
                    _optional_float(row.get("center_x_A")),
                    _optional_float(row.get("center_y_A")),
                    _optional_float(row.get("center_z_A")),
                ],
                dtype=float,
            ),
            section.center_A,
            atol=_V3_COORD_ABS_TOLERANCE_A,
            rtol=0.0,
        )
        _v3_compare_array(
            errors,
            f"{row_name} tangent",
            np.asarray(
                [
                    _optional_float(row.get("tangent_x")),
                    _optional_float(row.get("tangent_y")),
                    _optional_float(row.get("tangent_z")),
                ],
                dtype=float,
            ),
            section.tangent,
            atol=_V3_SCALAR_ABS_TOLERANCE,
            rtol=_V3_SCALAR_REL_TOLERANCE,
        )
        _v3_compare_bool(errors, f"{row_name} valid", is_valid, section.valid)
        reported_reason = row.get("invalid_reason") or None
        if reported_reason != section.invalid_reason:
            errors.append(
                f"{row_name} invalid_reason does not match independent final_phase "
                f"reconstruction: reported={reported_reason!r}, rebuilt={section.invalid_reason!r}"
            )
        _v3_compare_optional_scalar(
            errors,
            f"{row_name} area_A2",
            _optional_float(row.get("area_A2")),
            section.area_A2,
        )
        _v3_compare_optional_scalar(
            errors,
            f"{row_name} equivalent_diameter_A",
            _optional_float(row.get("equivalent_diameter_A")),
            section.equivalent_diameter_A,
        )
        _v3_compare_optional_scalar(
            errors,
            f"{row_name} curvature_fluctuation",
            _optional_float(row.get("curvature_fluctuation")),
            section.curvature_fluctuation,
        )
    return valid_count, invalid_count


def _compare_v3_final_measurements_json(
    reported: dict[str, Any],
    rebuilt: _V3FinalGeometryMeasurements,
    errors: list[str],
) -> None:
    required_keys = {
        "porosity",
        "slice_centers",
        "centerlines",
        "through_centerline_count",
        "branch_event_count",
        "center_distance_xy",
        "cross_sections",
        "projected_orientations",
        "channel_geometries",
        "compact_geometries",
    }
    if not reported:
        errors.append("final_measurements.json is empty or missing required schema-v3 measurements")
        return
    missing = sorted(required_keys - set(reported))
    if missing:
        errors.append(
            "final_measurements.json is missing required schema-v3 measurements: "
            + ", ".join(missing)
        )
    _v3_compare_scalar(
        errors,
        "final_measurements.json porosity",
        _optional_float(reported.get("porosity")),
        rebuilt.porosity,
    )
    _v3_compare_int(
        errors,
        "final_measurements.json through_centerline_count",
        reported.get("through_centerline_count"),
        rebuilt.through_centerline_count,
    )
    _v3_compare_int(
        errors,
        "final_measurements.json branch_event_count",
        reported.get("branch_event_count"),
        rebuilt.branch_event_count,
    )
    _compare_v3_json_slice_centers(reported.get("slice_centers"), rebuilt, errors)
    _compare_v3_json_centerlines(reported.get("centerlines"), rebuilt.centerlines, errors)
    _compare_v3_json_center_distance(reported.get("center_distance_xy"), rebuilt, errors)
    _compare_v3_json_cross_sections(reported.get("cross_sections"), rebuilt, errors)
    _compare_v3_json_orientations(
        reported.get("projected_orientations"),
        rebuilt.projected_orientations,
        errors,
    )
    _compare_v3_json_channels(
        reported.get("channel_geometries"),
        rebuilt.channel_geometries,
        errors,
    )
    _compare_v3_json_compacts(
        reported.get("compact_geometries"),
        rebuilt.compact_geometries,
        errors,
    )


def _compare_v3_json_compacts(
    reported: Any,
    rebuilt: tuple[_V3CompactGeometryMeasurement, ...],
    errors: list[str],
) -> None:
    records = reported if isinstance(reported, list) else []
    if len(records) != len(rebuilt):
        errors.append(
            "final_measurements.json compact geometry count does not match independent "
            f"final_phase reconstruction: reported={len(records)}, rebuilt={len(rebuilt)}"
        )
    for index, (record, expected) in enumerate(zip(records, rebuilt, strict=False)):
        if not isinstance(record, dict):
            errors.append(f"final_measurements.json compact_geometries[{index}] is invalid")
            continue
        _v3_compare_int(
            errors,
            f"final_measurements.json compact_geometries[{index}] component_id",
            record.get("component_id"),
            expected.component_id,
        )
        _v3_compare_int(
            errors,
            f"final_measurements.json compact_geometries[{index}] voxel_count",
            record.get("voxel_count"),
            expected.voxel_count,
        )
        _v3_compare_optional_scalar(
            errors,
            f"final_measurements.json compact_geometries[{index}] eta",
            _optional_float(record.get("eta")),
            expected.eta,
        )
        _v3_compare_bool(
            errors,
            f"final_measurements.json compact_geometries[{index}] valid",
            record.get("valid"),
            expected.valid,
        )
        if record.get("invalid_reason") != expected.invalid_reason:
            errors.append(
                "final_measurements.json "
                f"compact_geometries[{index}] invalid_reason does not match independent "
                f"final_phase reconstruction: reported={record.get('invalid_reason')!r}, "
                f"rebuilt={expected.invalid_reason!r}"
            )


def _compare_v3_json_slice_centers(
    reported: Any,
    rebuilt: _V3FinalGeometryMeasurements,
    errors: list[str],
) -> None:
    records = reported if isinstance(reported, list) else []
    if len(records) != len(rebuilt.slice_centers):
        errors.append(
            "final_measurements.json slice_centers count does not match independent "
            f"final_phase reconstruction: reported={len(records)}, rebuilt={len(rebuilt.slice_centers)}"
        )
    for index, (record, expected) in enumerate(zip(records, rebuilt.slice_centers, strict=False)):
        name = f"final_measurements.json slice_centers[{index}]"
        _v3_compare_int(errors, f"{name} z_index", record.get("z_index"), expected.z_index)
        _v3_compare_scalar(
            errors,
            f"{name} z_A",
            _optional_float(record.get("z_A")),
            expected.z_A,
        )
        centers = record.get("centers")
        centers = centers if isinstance(centers, list) else []
        if len(centers) != len(expected.centers):
            errors.append(
                f"{name} center count does not match independent final_phase "
                f"reconstruction: reported={len(centers)}, rebuilt={len(expected.centers)}"
            )
        for center_index, (center, expected_center) in enumerate(
            zip(centers, expected.centers, strict=False)
        ):
            center_name = f"{name}.centers[{center_index}]"
            _v3_compare_array(
                errors,
                f"{center_name} xy_A",
                _v3_array(center.get("xy_A")),
                expected_center.xy_A,
                atol=_V3_COORD_ABS_TOLERANCE_A,
                rtol=0.0,
            )
            _v3_compare_scalar(
                errors,
                f"{center_name} wall_distance_A",
                _optional_float(center.get("wall_distance_A")),
                expected_center.wall_distance_A,
            )


def _compare_v3_json_centerlines(
    reported: Any,
    expected: tuple[_V3CenterlineTrack, ...],
    errors: list[str],
) -> None:
    records = reported if isinstance(reported, list) else []
    if len(records) != len(expected):
        errors.append(
            "final_measurements.json centerlines count does not match independent "
            f"final_phase reconstruction: reported={len(records)}, rebuilt={len(expected)}"
        )
    for index, (record, track) in enumerate(zip(records, expected, strict=False)):
        name = f"final_measurements.json centerlines[{index}]"
        _v3_compare_int(errors, f"{name} track_id", record.get("track_id"), track.track_id)
        _v3_compare_array(
            errors,
            f"{name} slice_indices",
            _v3_array(record.get("slice_indices"), dtype=int),
            track.slice_indices,
            atol=0.0,
            rtol=0.0,
        )
        _v3_compare_array(
            errors,
            f"{name} points_wrapped_A",
            _v3_array(record.get("points_wrapped_A")),
            track.points_wrapped_A,
            atol=_V3_COORD_ABS_TOLERANCE_A,
            rtol=0.0,
        )
        _v3_compare_array(
            errors,
            f"{name} points_unwrapped_A",
            _v3_array(record.get("points_unwrapped_A")),
            track.points_unwrapped_A,
            atol=_V3_COORD_ABS_TOLERANCE_A,
            rtol=0.0,
        )
        _v3_compare_array(
            errors,
            f"{name} wall_distances_A",
            _v3_array(record.get("wall_distances_A")),
            track.wall_distances_A,
            atol=_V3_SCALAR_ABS_TOLERANCE,
            rtol=_V3_SCALAR_REL_TOLERANCE,
        )
        for key, expected_bool in (
            ("touches_z_lower", track.touches_z_lower),
            ("touches_z_upper", track.touches_z_upper),
            ("is_through", track.is_through),
            ("has_branch_neighborhood", track.has_branch_neighborhood),
        ):
            _v3_compare_bool(errors, f"{name} {key}", record.get(key), expected_bool)


def _compare_v3_json_center_distance(
    reported: Any,
    rebuilt: _V3FinalGeometryMeasurements,
    errors: list[str],
) -> None:
    record = reported if isinstance(reported, dict) else {}
    expected = rebuilt.center_distance_xy
    for key, array, atol, rtol in (
        ("bin_edges_A", expected.bin_edges_A, _V3_SCALAR_ABS_TOLERANCE, 0.0),
        ("bin_centers_A", expected.bin_centers_A, _V3_SCALAR_ABS_TOLERANCE, 0.0),
        ("observed_pair_counts", expected.observed_pair_counts, 0.0, 0.0),
        (
            "reference_pair_counts",
            expected.reference_pair_counts,
            _V3_GXY_ABS_TOLERANCE,
            _V3_GXY_REL_TOLERANCE,
        ),
        ("g_xy", expected.g_xy, _V3_GXY_ABS_TOLERANCE, _V3_GXY_REL_TOLERANCE),
    ):
        _v3_compare_array(
            errors,
            f"final_measurements.json center_distance_xy.{key}",
            _v3_array(record.get(key)),
            array,
            atol=atol,
            rtol=rtol,
        )
    _v3_compare_int(
        errors,
        "final_measurements.json center_distance_xy.pair_count",
        record.get("pair_count"),
        expected.pair_count,
    )
    _v3_compare_int(
        errors,
        "final_measurements.json center_distance_xy.valid_slice_count",
        record.get("valid_slice_count"),
        expected.valid_slice_count,
    )


def _compare_v3_json_cross_sections(
    reported: Any,
    rebuilt: _V3FinalGeometryMeasurements,
    errors: list[str],
) -> None:
    records = reported if isinstance(reported, list) else []
    if len(records) != len(rebuilt.cross_sections):
        errors.append(
            "final_measurements.json cross_sections count does not match independent "
            f"final_phase reconstruction: reported={len(records)}, rebuilt={len(rebuilt.cross_sections)}"
        )
    for index, (record, section) in enumerate(zip(records, rebuilt.cross_sections, strict=False)):
        name = f"final_measurements.json cross_sections[{index}]"
        _v3_compare_int(errors, f"{name} track_id", record.get("track_id"), section.track_id)
        _v3_compare_scalar(
            errors,
            f"{name} arc_position_A",
            _optional_float(record.get("arc_position_A")),
            section.arc_position_A,
        )
        _v3_compare_array(
            errors,
            f"{name} center_A",
            _v3_array(record.get("center_A")),
            section.center_A,
            atol=_V3_COORD_ABS_TOLERANCE_A,
            rtol=0.0,
        )
        _v3_compare_array(
            errors,
            f"{name} tangent",
            _v3_array(record.get("tangent")),
            section.tangent,
            atol=_V3_SCALAR_ABS_TOLERANCE,
            rtol=_V3_SCALAR_REL_TOLERANCE,
        )
        _v3_compare_bool(errors, f"{name} valid", record.get("valid"), section.valid)
        if record.get("invalid_reason") != section.invalid_reason:
            errors.append(
                f"{name} invalid_reason does not match independent final_phase "
                f"reconstruction: reported={record.get('invalid_reason')!r}, "
                f"rebuilt={section.invalid_reason!r}"
            )
        _v3_compare_optional_scalar(
            errors,
            f"{name} area_A2",
            _optional_float(record.get("area_A2")),
            section.area_A2,
        )
        _v3_compare_optional_scalar(
            errors,
            f"{name} equivalent_diameter_A",
            _optional_float(record.get("equivalent_diameter_A")),
            section.equivalent_diameter_A,
        )
        _v3_compare_optional_scalar(
            errors,
            f"{name} curvature_fluctuation",
            _optional_float(record.get("curvature_fluctuation")),
            section.curvature_fluctuation,
        )


def _compare_v3_json_orientations(
    reported: Any,
    expected: tuple[_V3ProjectedOrientationMeasurement, ...],
    errors: list[str],
) -> None:
    records = reported if isinstance(reported, list) else []
    if len(records) != len(expected):
        errors.append(
            "final_measurements.json projected_orientations count does not match independent "
            f"final_phase reconstruction: reported={len(records)}, rebuilt={len(expected)}"
        )
    for index, (record, orientation) in enumerate(zip(records, expected, strict=False)):
        name = f"final_measurements.json projected_orientations[{index}]"
        _v3_compare_int(errors, f"{name} track_id", record.get("track_id"), orientation.track_id)
        _v3_compare_array(
            errors,
            f"{name} axis",
            _v3_array(record.get("axis")),
            orientation.axis,
            atol=_V3_SCALAR_ABS_TOLERANCE,
            rtol=_V3_SCALAR_REL_TOLERANCE,
        )
        _v3_compare_optional_scalar(
            errors,
            f"{name} theta_xz_deg",
            _optional_float(record.get("theta_xz_deg")),
            orientation.theta_xz_deg,
        )
        _v3_compare_optional_scalar(
            errors,
            f"{name} theta_xy_deg",
            _optional_float(record.get("theta_xy_deg")),
            orientation.theta_xy_deg,
        )
        _v3_compare_bool(
            errors,
            f"{name} theta_xz_identifiable",
            record.get("theta_xz_identifiable"),
            orientation.theta_xz_identifiable,
        )
        _v3_compare_bool(
            errors,
            f"{name} theta_xy_identifiable",
            record.get("theta_xy_identifiable"),
            orientation.theta_xy_identifiable,
        )


def _compare_v3_json_channels(
    reported: Any,
    expected: tuple[_V3ChannelGeometryMeasurement, ...],
    errors: list[str],
) -> None:
    records = reported if isinstance(reported, list) else []
    if len(records) != len(expected):
        errors.append(
            "final_measurements.json channel_geometries count does not match independent "
            f"final_phase reconstruction: reported={len(records)}, rebuilt={len(expected)}"
        )
    for index, (record, channel) in enumerate(zip(records, expected, strict=False)):
        name = f"final_measurements.json channel_geometries[{index}]"
        _v3_compare_int(errors, f"{name} track_id", record.get("track_id"), channel.track_id)
        for key, expected_value in (
            ("arc_length_A", channel.arc_length_A),
            ("end_distance_A", channel.end_distance_A),
            ("equivalent_diameter_A", channel.equivalent_diameter_A),
            ("eta", channel.eta),
            ("tortuosity", channel.tortuosity),
        ):
            _v3_compare_optional_scalar(
                errors,
                f"{name} {key}",
                _optional_float(record.get(key)),
                expected_value,
            )
        _v3_compare_bool(errors, f"{name} valid", record.get("valid"), channel.valid)
        if record.get("invalid_reason") != channel.invalid_reason:
            errors.append(
                f"{name} invalid_reason does not match independent final_phase "
                f"reconstruction: reported={record.get('invalid_reason')!r}, "
                f"rebuilt={channel.invalid_reason!r}"
            )


def _v3_compare_int(
    errors: list[str],
    name: str,
    reported: Any,
    expected: int,
) -> None:
    try:
        value = int(reported)
    except (TypeError, ValueError):
        errors.append(
            f"{name} does not match independent final_phase reconstruction: "
            f"reported={reported!r}, rebuilt={expected}"
        )
        return
    if value != int(expected):
        errors.append(
            f"{name} does not match independent final_phase reconstruction: "
            f"reported={value}, rebuilt={int(expected)}"
        )


def _v3_compare_bool(
    errors: list[str],
    name: str,
    reported: Any,
    expected: bool,
) -> None:
    value = _v3_parse_bool(reported)
    if value is None or value != bool(expected):
        errors.append(
            f"{name} does not match independent final_phase reconstruction: "
            f"reported={reported!r}, rebuilt={bool(expected)}"
        )


def _v3_parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    if isinstance(value, int | np.integer):
        return bool(value)
    return None


def _v3_compare_scalar(
    errors: list[str],
    name: str,
    reported: float,
    expected: float,
    *,
    atol: float = _V3_SCALAR_ABS_TOLERANCE,
    rtol: float = _V3_SCALAR_REL_TOLERANCE,
) -> None:
    if not (
        np.isfinite(reported)
        and np.isfinite(expected)
        and np.isclose(reported, expected, atol=atol, rtol=rtol)
    ):
        errors.append(
            f"{name} does not match independent final_phase reconstruction: "
            f"reported={reported}, rebuilt={expected}, atol={atol}, rtol={rtol}"
        )


def _v3_compare_optional_scalar(
    errors: list[str],
    name: str,
    reported: float,
    expected: float | None,
) -> None:
    if expected is None:
        if np.isfinite(reported):
            errors.append(
                f"{name} does not match independent final_phase reconstruction: "
                f"reported={reported}, rebuilt=None"
            )
        return
    _v3_compare_scalar(errors, name, reported, expected)


def _v3_compare_array(
    errors: list[str],
    name: str,
    reported: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> None:
    reported = np.asarray(reported)
    expected = np.asarray(expected)
    if reported.shape != expected.shape:
        errors.append(
            f"{name} shape does not match independent final_phase reconstruction: "
            f"reported={reported.shape}, rebuilt={expected.shape}"
        )
        return
    if not np.allclose(reported, expected, atol=atol, rtol=rtol, equal_nan=True):
        delta = float(np.nanmax(np.abs(reported.astype(float) - expected.astype(float))))
        errors.append(
            f"{name} does not match independent final_phase reconstruction: "
            f"max_abs_delta={delta}, atol={atol}, rtol={rtol}"
        )


def _v3_array(value: Any, *, dtype: type = float) -> np.ndarray:
    try:
        return np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        return np.asarray([], dtype=dtype)


def _v3_measure_final_geometry(
    phase: _PhaseData,
    contract: _V3MeasurementContract,
) -> _V3FinalGeometryMeasurements:
    sampled_indices = _v3_sampled_z_indices(
        phase.mask.shape[0],
        phase.spacing_A,
        contract.z_slice_spacing_A,
    )
    slices = tuple(
        _v3_measure_slice_centers(phase, int(z_index), contract) for z_index in sampled_indices
    )
    centerlines, branch_count, branch_z_by_track = _v3_track_slice_centers(
        slices,
        target_box_A=phase.target_box_A,
        maximum_displacement_A=contract.center_tracking_max_displacement_A,
        lower_index=int(sampled_indices[0]) if sampled_indices.size else 0,
        upper_index=int(sampled_indices[-1]) if sampled_indices.size else 0,
    )
    center_distance = _v3_measure_xy_center_distance_distribution_from_tracks(
        centerlines,
        box_xy_A=np.asarray(phase.target_box_A[:2], dtype=float),
        bin_width_A=contract.center_distance_bin_width_A,
        maximum_distance_A=contract.center_distance_max_A,
        reference_samples=contract.center_distance_reference_samples,
    )
    sampled_centerlines = {
        track.track_id: _v3_resample_centerline(
            _v3_smoothed_track_points(track.points_unwrapped_A),
            track.wall_distances_A,
            sample_spacing_A=contract.centerline_sample_spacing_A,
        )
        for track in centerlines
    }
    cross_sections = _v3_measure_normal_cross_sections(
        phase,
        centerlines,
        contract,
        branch_z_by_track=branch_z_by_track,
        sampled_centerlines=sampled_centerlines,
    )
    channel_geometries = tuple(
        _v3_channel_geometry(
            track,
            cross_sections,
            centerline_points_A=sampled_centerlines[track.track_id][0],
        )
        for track in centerlines
    )
    channel_geometry_by_track = {value.track_id: value for value in channel_geometries}
    projected_orientations = tuple(
        _v3_projected_orientation(
            track,
            minimum_projection_fraction=contract.orientation_projection_min_fraction,
            channel_aspect_ratio=(
                channel_geometry_by_track[track.track_id].eta
                if channel_geometry_by_track[track.track_id].valid
                else None
            ),
            aspect_ratio_tolerance=contract.orientation_aspect_ratio_tolerance,
        )
        for track in centerlines
    )
    compact_geometries = _v3_measure_compact_geometries(phase)
    return _V3FinalGeometryMeasurements(
        porosity=float(np.mean(phase.mask)),
        slice_centers=slices,
        centerlines=centerlines,
        through_centerline_count=sum(track.is_through for track in centerlines),
        branch_event_count=branch_count,
        center_distance_xy=center_distance,
        cross_sections=cross_sections,
        projected_orientations=projected_orientations,
        channel_geometries=channel_geometries,
        compact_geometries=compact_geometries,
    )


def _v3_measure_compact_geometries(
    phase: _PhaseData,
) -> tuple[_V3CompactGeometryMeasurement, ...]:
    labels, label_count = ndimage.label(
        phase.mask,
        structure=_six_connected_structure(),
    )
    if label_count == 0:
        return ()
    parents = np.arange(label_count + 1, dtype=int)
    for first, second in zip(labels[:, :, 0].ravel(), labels[:, :, -1].ravel(), strict=True):
        _union_labels(int(first), int(second), parents)
    for first, second in zip(labels[:, 0, :].ravel(), labels[:, -1, :].ravel(), strict=True):
        _union_labels(int(first), int(second), parents)
    roots = np.zeros(label_count + 1, dtype=int)
    for label in range(1, label_count + 1):
        roots[label] = _find(label, parents)
    rooted = roots[labels]
    lower_roots = {int(value) for value in np.unique(rooted[0]) if int(value) != 0}
    upper_roots = {int(value) for value in np.unique(rooted[-1]) if int(value) != 0}

    output: list[_V3CompactGeometryMeasurement] = []
    for component_id, root in enumerate(sorted(set(roots[1:])), start=1):
        if root in lower_roots or root in upper_roots:
            continue
        indices_zyx = np.argwhere(rooted == root)
        if indices_zyx.shape[0] < 4:
            output.append(
                _V3CompactGeometryMeasurement(
                    component_id=component_id,
                    voxel_count=int(indices_zyx.shape[0]),
                    eta=None,
                    valid=False,
                    invalid_reason="insufficient_component_voxels",
                )
            )
            continue
        indices_xyz = indices_zyx[:, [2, 1, 0]].astype(float)
        indices_xyz[:, 0] = _v3_unwrap_periodic_indices(indices_xyz[:, 0], phase.mask.shape[2])
        indices_xyz[:, 1] = _v3_unwrap_periodic_indices(indices_xyz[:, 1], phase.mask.shape[1])
        points_A = phase.origin_A + (indices_xyz + 0.5) * phase.spacing_A
        centered = points_A - np.mean(points_A, axis=0)
        covariance = centered.T @ centered / float(points_A.shape[0])
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        projected = centered @ eigenvectors[:, np.argsort(eigenvalues)[::-1]]
        extents = np.ptp(projected, axis=0) + float(phase.spacing_A)
        if np.any(extents <= 0.0) or not np.all(np.isfinite(extents)):
            output.append(
                _V3CompactGeometryMeasurement(
                    component_id=component_id,
                    voxel_count=int(indices_zyx.shape[0]),
                    eta=None,
                    valid=False,
                    invalid_reason="invalid_principal_extents",
                )
            )
            continue
        eta = float(extents[0] / np.sqrt(extents[1] * extents[2]))
        output.append(
            _V3CompactGeometryMeasurement(
                component_id=component_id,
                voxel_count=int(indices_zyx.shape[0]),
                eta=max(eta, 1.0),
                valid=True,
                invalid_reason=None,
            )
        )
    return tuple(output)


def _v3_unwrap_periodic_indices(values: np.ndarray, size: int) -> np.ndarray:
    wrapped = np.asarray(values, dtype=float)
    unique = np.unique(wrapped.astype(int))
    if unique.size < 2:
        return wrapped.copy()
    cyclic = np.concatenate((unique, unique[:1] + int(size)))
    cut = int(unique[(int(np.argmax(np.diff(cyclic))) + 1) % unique.size])
    return np.where(wrapped < cut, wrapped + int(size), wrapped)


def _v3_projected_orientation(
    track: _V3CenterlineTrack,
    *,
    minimum_projection_fraction: float,
    channel_aspect_ratio: float | None = None,
    aspect_ratio_tolerance: float = 0.0,
) -> _V3ProjectedOrientationMeasurement:
    invalid_axis = np.array([np.nan, np.nan, np.nan])
    if track.points_unwrapped_A.shape[0] < 2:
        return _V3ProjectedOrientationMeasurement(
            track_id=track.track_id,
            axis=invalid_axis,
            theta_xz_deg=None,
            theta_xy_deg=None,
            theta_xz_identifiable=False,
            theta_xy_identifiable=False,
        )
    axis = track.points_unwrapped_A[-1] - track.points_unwrapped_A[0]
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        return _V3ProjectedOrientationMeasurement(
            track_id=track.track_id,
            axis=invalid_axis,
            theta_xz_deg=None,
            theta_xy_deg=None,
            theta_xz_identifiable=False,
            theta_xy_identifiable=False,
        )
    axis = axis / norm
    if (
        channel_aspect_ratio is not None
        and channel_aspect_ratio <= 1.0 + float(aspect_ratio_tolerance)
    ):
        return _V3ProjectedOrientationMeasurement(
            track_id=track.track_id,
            axis=axis,
            theta_xz_deg=None,
            theta_xy_deg=None,
            theta_xz_identifiable=False,
            theta_xy_identifiable=False,
        )
    xz_projection = float(np.hypot(axis[0], axis[2]))
    xy_projection = float(np.hypot(axis[0], axis[1]))
    xz_identifiable = xz_projection >= float(minimum_projection_fraction)
    xy_identifiable = xy_projection >= float(minimum_projection_fraction)
    theta_xz = (
        float(np.degrees(np.arctan2(abs(axis[2]), abs(axis[0])))) if xz_identifiable else None
    )
    theta_xy = (
        float(np.degrees(np.arctan2(abs(axis[1]), abs(axis[0])))) if xy_identifiable else None
    )
    return _V3ProjectedOrientationMeasurement(
        track_id=track.track_id,
        axis=axis,
        theta_xz_deg=theta_xz,
        theta_xy_deg=theta_xy,
        theta_xz_identifiable=xz_identifiable,
        theta_xy_identifiable=xy_identifiable,
    )


def _v3_channel_geometry(
    track: _V3CenterlineTrack,
    cross_sections: tuple[_V3CrossSectionMeasurement, ...],
    *,
    centerline_points_A: np.ndarray,
) -> _V3ChannelGeometryMeasurement:
    if track.has_branch_neighborhood:
        return _v3_invalid_channel_geometry(track.track_id, "branch_neighborhood")
    if centerline_points_A.shape[0] < 2:
        return _v3_invalid_channel_geometry(track.track_id, "insufficient_centerline_points")
    segment_lengths = np.linalg.norm(np.diff(centerline_points_A, axis=0), axis=1)
    arc_length = float(np.sum(segment_lengths))
    end_distance = float(np.linalg.norm(centerline_points_A[-1] - centerline_points_A[0]))
    valid_sections = [
        section
        for section in cross_sections
        if section.track_id == track.track_id and section.valid and section.area_A2 is not None
    ]
    if not valid_sections:
        return _v3_invalid_channel_geometry(track.track_id, "no_valid_cross_sections")
    mean_area = float(np.mean([float(section.area_A2) for section in valid_sections]))
    equivalent_diameter = 2.0 * float(np.sqrt(mean_area / np.pi))
    if arc_length <= 0.0 or end_distance <= 0.0 or equivalent_diameter <= 0.0:
        return _v3_invalid_channel_geometry(track.track_id, "nonpositive_channel_measure")
    return _V3ChannelGeometryMeasurement(
        track_id=track.track_id,
        arc_length_A=arc_length,
        end_distance_A=end_distance,
        equivalent_diameter_A=equivalent_diameter,
        eta=arc_length / equivalent_diameter,
        tortuosity=arc_length / end_distance,
        valid=True,
        invalid_reason=None,
    )


def _v3_invalid_channel_geometry(
    track_id: int,
    reason: str,
) -> _V3ChannelGeometryMeasurement:
    return _V3ChannelGeometryMeasurement(
        track_id=track_id,
        arc_length_A=None,
        end_distance_A=None,
        equivalent_diameter_A=None,
        eta=None,
        tortuosity=None,
        valid=False,
        invalid_reason=reason,
    )


def _v3_measure_normal_cross_sections(
    phase: _PhaseData,
    tracks: tuple[_V3CenterlineTrack, ...],
    contract: _V3MeasurementContract,
    *,
    branch_z_by_track: dict[int, np.ndarray],
    sampled_centerlines: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[_V3CrossSectionMeasurement, ...]:
    output: list[_V3CrossSectionMeasurement] = []
    for track in tracks:
        points, wall_distances = sampled_centerlines[track.track_id]
        if points.shape[0] < 2:
            continue
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total_length = float(cumulative[-1])
        if total_length <= 0.0:
            continue
        branch_z = branch_z_by_track.get(track.track_id, np.empty(0, dtype=float))
        branch_arc_positions = (
            np.interp(branch_z, points[:, 2], cumulative)
            if branch_z.size
            else np.empty(0, dtype=float)
        )
        lower = float(contract.surface_exclusion_length_A)
        upper = total_length - float(contract.surface_exclusion_length_A)
        if upper < lower:
            positions = np.array([0.5 * total_length], dtype=float)
        else:
            positions = np.arange(
                lower,
                upper + 0.5 * contract.cross_section_spacing_A,
                contract.cross_section_spacing_A,
                dtype=float,
            )
            positions = positions[positions <= upper + 1.0e-9]
        for arc_position in positions:
            center, tangent, wall_distance = _v3_interpolate_track(
                points,
                wall_distances,
                cumulative,
                float(arc_position),
            )
            if (
                branch_arc_positions.size
                and contract.branch_exclusion_length_A > 0.0
                and np.min(np.abs(branch_arc_positions - arc_position))
                <= contract.branch_exclusion_length_A + 1.0e-9
            ):
                output.append(
                    _v3_invalid_cross_section(
                        track.track_id,
                        float(arc_position),
                        center,
                        tangent,
                        "branch_neighborhood",
                    )
                )
                continue
            output.append(
                _v3_measure_one_cross_section(
                    phase,
                    track_id=track.track_id,
                    arc_position_A=float(arc_position),
                    center_A=center,
                    tangent=tangent,
                    wall_distance_A=wall_distance,
                    contract=contract,
                )
            )
    return tuple(output)


def _v3_smoothed_track_points(points_A: np.ndarray) -> np.ndarray:
    points = np.asarray(points_A, dtype=float)
    if points.shape[0] < 4:
        return points.copy()
    smoothed = np.column_stack(
        [ndimage.gaussian_filter1d(points[:, axis], sigma=0.5, mode="nearest") for axis in range(3)]
    )
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return smoothed


def _v3_resample_centerline(
    points_A: np.ndarray,
    wall_distances_A: np.ndarray,
    *,
    sample_spacing_A: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_A, dtype=float)
    walls = np.asarray(wall_distances_A, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or walls.shape != (points.shape[0],):
        raise ValueError("centerline points and wall distances must have matching lengths")
    if points.shape[0] < 2:
        return points.copy(), walls.copy()
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 1.0e-12))
    points = points[keep]
    walls = walls[keep]
    if points.shape[0] < 2:
        return points.copy(), walls.copy()
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    total_length = float(cumulative[-1])
    positions = np.arange(0.0, total_length, float(sample_spacing_A), dtype=float)
    if positions.size == 0 or total_length - positions[-1] > 1.0e-12:
        positions = np.append(positions, total_length)
    else:
        positions[-1] = total_length
    sampled_points = np.column_stack(
        [np.interp(positions, cumulative, points[:, axis]) for axis in range(3)]
    )
    sampled_walls = np.interp(positions, cumulative, walls)
    return sampled_points, sampled_walls


def _v3_interpolate_track(
    points_A: np.ndarray,
    wall_distances_A: np.ndarray,
    cumulative_A: np.ndarray,
    arc_position_A: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if arc_position_A <= 0.0:
        index = 0
        fraction = 0.0
    elif arc_position_A >= cumulative_A[-1]:
        index = points_A.shape[0] - 2
        fraction = 1.0
    else:
        index = int(np.searchsorted(cumulative_A, arc_position_A, side="right") - 1)
        span = float(cumulative_A[index + 1] - cumulative_A[index])
        fraction = (arc_position_A - float(cumulative_A[index])) / max(span, 1.0e-12)
    center = (1.0 - fraction) * points_A[index] + fraction * points_A[index + 1]
    tangent = points_A[index + 1] - points_A[index]
    tangent /= np.linalg.norm(tangent)
    wall = (1.0 - fraction) * wall_distances_A[index] + fraction * wall_distances_A[
        min(index + 1, wall_distances_A.size - 1)
    ]
    return center, tangent, float(wall)


def _v3_measure_one_cross_section(
    phase: _PhaseData,
    *,
    track_id: int,
    arc_position_A: float,
    center_A: np.ndarray,
    tangent: np.ndarray,
    wall_distance_A: float,
    contract: _V3MeasurementContract,
) -> _V3CrossSectionMeasurement:
    first_axis, second_axis = _v3_normal_plane_axes(tangent)
    plane_spacing = min(
        0.5 * float(phase.spacing_A),
        float(contract.boundary_resample_spacing_A),
    )
    plane_spacing = max(plane_spacing, 0.25 * float(phase.spacing_A))
    base_extent = max(
        3.0 * float(wall_distance_A),
        4.0 * float(phase.spacing_A),
        2.0 * float(contract.center_min_separation_A),
    )
    maximum_extent = 0.75 * float(np.linalg.norm(phase.target_box_A))
    extent = min(base_extent, maximum_extent)
    for _attempt in range(4):
        component, coordinates = _v3_sample_normal_component(
            phase,
            center_A=center_A,
            first_axis=first_axis,
            second_axis=second_axis,
            half_extent_A=extent,
            plane_spacing_A=plane_spacing,
        )
        if component is None:
            return _v3_invalid_cross_section(
                track_id,
                arc_position_A,
                center_A,
                tangent,
                "center_not_in_pore",
            )
        if not _v3_component_touches_border(component):
            return _v3_cross_section_from_component(
                component,
                coordinates,
                track_id=track_id,
                arc_position_A=arc_position_A,
                center_A=center_A,
                tangent=tangent,
                plane_spacing_A=plane_spacing,
                contract=contract,
            )
        if extent >= maximum_extent:
            break
        extent = min(1.75 * extent, maximum_extent)
    return _v3_invalid_cross_section(
        track_id,
        arc_position_A,
        center_A,
        tangent,
        "cross_section_not_bounded",
    )


def _v3_normal_plane_axes(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(tangent, dtype=float)
    direction /= np.linalg.norm(direction)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(direction, reference))) > 0.85:
        reference = np.array([0.0, 1.0, 0.0])
    first = np.cross(direction, reference)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    second /= np.linalg.norm(second)
    return first, second


def _v3_sample_normal_component(
    phase: _PhaseData,
    *,
    center_A: np.ndarray,
    first_axis: np.ndarray,
    second_axis: np.ndarray,
    half_extent_A: float,
    plane_spacing_A: float,
) -> tuple[np.ndarray | None, np.ndarray]:
    coordinate = np.arange(
        -half_extent_A,
        half_extent_A + 0.5 * plane_spacing_A,
        plane_spacing_A,
        dtype=float,
    )
    vv, uu = np.meshgrid(coordinate, coordinate, indexing="ij")
    world = (
        center_A[np.newaxis, np.newaxis, :]
        + uu[:, :, np.newaxis] * first_axis
        + vv[:, :, np.newaxis] * second_axis
    )
    nx = phase.mask.shape[2]
    ny = phase.mask.shape[1]
    x_index = (world[:, :, 0] - phase.origin_A[0]) / phase.spacing_A - 0.5
    y_index = (world[:, :, 1] - phase.origin_A[1]) / phase.spacing_A - 0.5
    z_index = (world[:, :, 2] - phase.origin_A[2]) / phase.spacing_A - 0.5
    x_index = np.mod(x_index, nx)
    y_index = np.mod(y_index, ny)
    sampled = ndimage.map_coordinates(
        phase.mask.astype(float),
        [z_index, y_index, x_index],
        order=1,
        mode="constant",
        cval=0.0,
    )
    plane_mask = sampled >= 0.5
    labels, _ = ndimage.label(plane_mask, structure=np.ones((3, 3), dtype=int))
    middle = coordinate.size // 2
    label = int(labels[middle, middle])
    if label == 0:
        window = labels[
            max(0, middle - 1) : min(labels.shape[0], middle + 2),
            max(0, middle - 1) : min(labels.shape[1], middle + 2),
        ]
        candidates = window[window > 0]
        if candidates.size:
            label = int(candidates[0])
    if label == 0:
        return None, coordinate
    return labels == label, coordinate


def _v3_component_touches_border(component: np.ndarray) -> bool:
    return bool(
        np.any(component[0])
        or np.any(component[-1])
        or np.any(component[:, 0])
        or np.any(component[:, -1])
    )


def _v3_cross_section_from_component(
    component: np.ndarray,
    coordinates_A: np.ndarray,
    *,
    track_id: int,
    arc_position_A: float,
    center_A: np.ndarray,
    tangent: np.ndarray,
    plane_spacing_A: float,
    contract: _V3MeasurementContract,
) -> _V3CrossSectionMeasurement:
    contours = find_contours(component.astype(float), 0.5)
    closed = [contour for contour in contours if contour.shape[0] >= 8]
    if not closed:
        return _v3_invalid_cross_section(
            track_id,
            arc_position_A,
            center_A,
            tangent,
            "boundary_not_found",
        )
    contour = max(closed, key=lambda item: item.shape[0])
    plane_xy = np.column_stack(
        [
            coordinates_A[0] + contour[:, 1] * plane_spacing_A,
            coordinates_A[0] + contour[:, 0] * plane_spacing_A,
        ]
    )
    if np.linalg.norm(plane_xy[0] - plane_xy[-1]) > 1.5 * plane_spacing_A:
        return _v3_invalid_cross_section(
            track_id,
            arc_position_A,
            center_A,
            tangent,
            "open_boundary",
        )
    area = abs(_v3_polygon_area_A2(plane_xy))
    if area <= 0.0:
        return _v3_invalid_cross_section(
            track_id,
            arc_position_A,
            center_A,
            tangent,
            "nonpositive_area",
        )
    equivalent_diameter = 2.0 * float(np.sqrt(area / np.pi))
    curvature_fluctuation = _v3_closed_curve_curvature_fluctuation(
        plane_xy,
        resample_spacing_A=contract.boundary_resample_spacing_A,
        smoothing_length_A=contract.curvature_smoothing_length_A,
    )
    return _V3CrossSectionMeasurement(
        track_id=track_id,
        arc_position_A=arc_position_A,
        center_A=np.asarray(center_A, dtype=float),
        tangent=np.asarray(tangent, dtype=float),
        area_A2=area,
        equivalent_diameter_A=equivalent_diameter,
        curvature_fluctuation=curvature_fluctuation,
        valid=True,
        invalid_reason=None,
    )


def _v3_polygon_area_A2(points_xy_A: np.ndarray) -> float:
    x = points_xy_A[:, 0]
    y = points_xy_A[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _v3_closed_curve_curvature_fluctuation(
    points_xy_A: np.ndarray,
    *,
    resample_spacing_A: float,
    smoothing_length_A: float,
) -> float:
    points = np.asarray(points_xy_A, dtype=float)
    if np.linalg.norm(points[0] - points[-1]) <= 1.0e-12:
        points = points[:-1]
    closed = np.vstack([points, points[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(np.sum(lengths))
    if perimeter <= 0.0:
        return float("nan")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    sample_count = max(16, int(np.ceil(perimeter / float(resample_spacing_A))))
    sample_s = np.linspace(0.0, perimeter, sample_count, endpoint=False)
    sampled = np.column_stack(
        [np.interp(sample_s, cumulative, closed[:, axis]) for axis in range(2)]
    )
    sigma = float(smoothing_length_A) / max(float(resample_spacing_A), 1.0e-12)
    if sigma > 0.0:
        sampled = np.column_stack(
            [
                ndimage.gaussian_filter1d(sampled[:, axis], sigma=sigma, mode="wrap")
                for axis in range(2)
            ]
        )
    step = perimeter / sample_count
    first = (np.roll(sampled, -1, axis=0) - np.roll(sampled, 1, axis=0)) / (2.0 * step)
    second = (np.roll(sampled, -1, axis=0) - 2.0 * sampled + np.roll(sampled, 1, axis=0)) / (
        step**2
    )
    denominator = np.maximum(np.sum(first**2, axis=1) ** 1.5, 1.0e-12)
    curvature = (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / denominator
    mean_curvature = float(np.mean(curvature))
    return float(np.std(curvature) / max(abs(mean_curvature), 1.0e-12))


def _v3_invalid_cross_section(
    track_id: int,
    arc_position_A: float,
    center_A: np.ndarray,
    tangent: np.ndarray,
    reason: str,
) -> _V3CrossSectionMeasurement:
    return _V3CrossSectionMeasurement(
        track_id=track_id,
        arc_position_A=arc_position_A,
        center_A=np.asarray(center_A, dtype=float),
        tangent=np.asarray(tangent, dtype=float),
        area_A2=None,
        equivalent_diameter_A=None,
        curvature_fluctuation=None,
        valid=False,
        invalid_reason=reason,
    )


def _v3_measure_xy_center_distance_distribution(
    slices: tuple[_V3SliceCenterRecord, ...],
    *,
    box_xy_A: np.ndarray,
    bin_width_A: float,
    maximum_distance_A: float | None,
    reference_samples: int,
) -> _V3XYCenterDistanceDistribution:
    maximum = (
        float(maximum_distance_A)
        if maximum_distance_A is not None
        else float(np.linalg.norm(0.5 * np.asarray(box_xy_A, dtype=float)))
    )
    bin_width = float(bin_width_A)
    bin_count = max(1, int(np.ceil(maximum / bin_width)))
    edges = np.linspace(0.0, bin_count * bin_width, bin_count + 1)
    observed = np.zeros(bin_count, dtype=np.int64)
    pair_count = 0
    valid_slice_count = 0
    for slice_record in slices:
        if len(slice_record.centers) < 2:
            continue
        xy = np.vstack([center.xy_A for center in slice_record.centers])
        distances = _v3_unique_periodic_xy_distances(xy, box_xy_A)
        observed += np.histogram(distances, bins=edges)[0]
        pair_count += int(distances.size)
        valid_slice_count += 1
    reference = np.zeros(bin_count, dtype=float)
    if pair_count:
        reference_probability = _v3_uniform_periodic_xy_reference(
            box_xy_A,
            edges,
            sample_count=reference_samples,
        )
        reference = reference_probability * float(pair_count)
    g_xy = np.zeros(bin_count, dtype=float)
    valid = reference > 0.0
    g_xy[valid] = observed[valid] / reference[valid]
    return _V3XYCenterDistanceDistribution(
        bin_edges_A=edges,
        bin_centers_A=0.5 * (edges[:-1] + edges[1:]),
        observed_pair_counts=observed,
        reference_pair_counts=reference,
        g_xy=g_xy,
        pair_count=pair_count,
        valid_slice_count=valid_slice_count,
    )


def _v3_measure_xy_center_distance_distribution_from_tracks(
    tracks: tuple[_V3CenterlineTrack, ...],
    *,
    box_xy_A: np.ndarray,
    bin_width_A: float,
    maximum_distance_A: float | None,
    reference_samples: int,
) -> _V3XYCenterDistanceDistribution:
    through_tracks = [track for track in tracks if track.is_through]
    by_slice: dict[int, list[tuple[float, _V3SliceCenter]]] = {}
    for track in through_tracks:
        for row, z_index in enumerate(track.slice_indices):
            point = track.points_wrapped_A[row]
            wall = float(track.wall_distances_A[row])
            by_slice.setdefault(int(z_index), []).append(
                (
                    float(point[2]),
                    _V3SliceCenter(xy_A=np.asarray(point[:2], dtype=float), wall_distance_A=wall),
                )
            )
    slices = tuple(
        _V3SliceCenterRecord(
            z_index=z_index,
            z_A=float(np.mean([item[0] for item in entries])),
            centers=tuple(item[1] for item in entries),
        )
        for z_index, entries in sorted(by_slice.items())
    )
    return _v3_measure_xy_center_distance_distribution(
        slices,
        box_xy_A=box_xy_A,
        bin_width_A=bin_width_A,
        maximum_distance_A=maximum_distance_A,
        reference_samples=reference_samples,
    )


def _v3_unique_periodic_xy_distances(
    points_xy_A: np.ndarray,
    box_xy_A: np.ndarray,
) -> np.ndarray:
    count = points_xy_A.shape[0]
    row, col = np.triu_indices(count, k=1)
    delta = points_xy_A[col] - points_xy_A[row]
    delta -= box_xy_A * np.round(delta / box_xy_A)
    return np.linalg.norm(delta, axis=1)


def _v3_uniform_periodic_xy_reference(
    box_xy_A: np.ndarray,
    edges_A: np.ndarray,
    *,
    sample_count: int,
) -> np.ndarray:
    power = max(1, int(np.ceil(np.log2(max(int(sample_count), 2)))))
    samples = qmc.Sobol(d=4, scramble=False).random_base2(power)
    first = samples[:, :2] * box_xy_A
    second = samples[:, 2:] * box_xy_A
    delta = second - first
    delta -= box_xy_A * np.round(delta / box_xy_A)
    distances = np.linalg.norm(delta, axis=1)
    counts = np.histogram(distances, bins=edges_A)[0].astype(float)
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.zeros(edges_A.size - 1, dtype=float)
    return counts / total


def _v3_sampled_z_indices(
    count: int,
    spacing_A: float,
    requested_spacing_A: float,
) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=int)
    stride = max(1, round(float(requested_spacing_A) / float(spacing_A)))
    indices = np.arange(0, count, stride, dtype=int)
    if indices[-1] != count - 1:
        indices = np.append(indices, count - 1)
    return indices


def _v3_measure_slice_centers(
    phase: _PhaseData,
    z_index: int,
    contract: _V3MeasurementContract,
) -> _V3SliceCenterRecord:
    mask = np.asarray(phase.mask[z_index], dtype=bool)
    centers = _v3_periodic_slice_centers(
        mask,
        spacing_A=phase.spacing_A,
        origin_xy_A=phase.origin_A[:2],
        minimum_separation_A=contract.center_min_separation_A,
    )
    z_A = float(phase.origin_A[2] + (z_index + 0.5) * phase.spacing_A)
    return _V3SliceCenterRecord(z_index=z_index, z_A=z_A, centers=centers)


def _v3_periodic_slice_centers(
    mask_yx: np.ndarray,
    *,
    spacing_A: float,
    origin_xy_A: np.ndarray,
    minimum_separation_A: float,
) -> tuple[_V3SliceCenter, ...]:
    mask = np.asarray(mask_yx, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("slice mask must have shape (y, x)")
    if not np.any(mask):
        return ()
    ny, nx = mask.shape
    tiled = np.tile(mask, (3, 3))
    distance = ndimage.distance_transform_edt(tiled, sampling=float(spacing_A))
    min_distance_voxels = max(1, int(np.ceil(minimum_separation_A / spacing_A)))
    coordinates = peak_local_max(
        distance,
        min_distance=min_distance_voxels,
        threshold_abs=0.5 * float(spacing_A),
        exclude_border=False,
    )
    in_center = coordinates[
        (coordinates[:, 0] >= ny)
        & (coordinates[:, 0] < 2 * ny)
        & (coordinates[:, 1] >= nx)
        & (coordinates[:, 1] < 2 * nx)
    ]
    if in_center.size == 0:
        return ()
    candidates: list[tuple[float, np.ndarray]] = []
    for tiled_y, tiled_x in in_center:
        local_y = int(tiled_y - ny)
        local_x = int(tiled_x - nx)
        xy = np.array(
            [
                float(origin_xy_A[0] + (local_x + 0.5) * spacing_A),
                float(origin_xy_A[1] + (local_y + 0.5) * spacing_A),
            ]
        )
        candidates.append((float(distance[tiled_y, tiled_x]), xy))
    candidates.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
    box_xy = np.array([nx * spacing_A, ny * spacing_A], dtype=float)
    candidates = _v3_merge_plateau_candidates(
        candidates,
        box_xy_A=box_xy,
        spacing_A=spacing_A,
    )
    accepted: list[_V3SliceCenter] = []
    for wall_distance, xy in candidates:
        if any(
            _v3_periodic_xy_distance(xy, center.xy_A, box_xy) <= minimum_separation_A + 1.0e-12
            for center in accepted
        ):
            continue
        accepted.append(_V3SliceCenter(xy_A=xy, wall_distance_A=wall_distance))
    accepted = _v3_refine_slice_centers(
        mask,
        distance[ny : 2 * ny, nx : 2 * nx],
        accepted,
        spacing_A=spacing_A,
        origin_xy_A=origin_xy_A,
        box_xy_A=box_xy,
    )
    accepted.sort(key=lambda center: (float(center.xy_A[0]), float(center.xy_A[1])))
    return tuple(accepted)


def _v3_merge_plateau_candidates(
    candidates: list[tuple[float, np.ndarray]],
    *,
    box_xy_A: np.ndarray,
    spacing_A: float,
) -> list[tuple[float, np.ndarray]]:
    count = len(candidates)
    if count < 2:
        return candidates
    parents = np.arange(count, dtype=int)

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first in range(count):
        first_distance, first_xy = candidates[first]
        for second in range(first + 1, count):
            second_distance, second_xy = candidates[second]
            if not np.isclose(first_distance, second_distance, atol=1.0e-12):
                continue
            delta = second_xy - first_xy
            delta -= box_xy_A * np.round(delta / box_xy_A)
            if np.max(np.abs(delta)) <= float(spacing_A) + 1.0e-12:
                union(first, second)
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    merged: list[tuple[float, np.ndarray]] = []
    for members in groups.values():
        reference = candidates[members[0]][1]
        unwrapped = []
        for member in members:
            xy = candidates[member][1]
            delta = xy - reference
            delta -= box_xy_A * np.round(delta / box_xy_A)
            unwrapped.append(reference + delta)
        merged_xy = np.mod(np.mean(np.vstack(unwrapped), axis=0), box_xy_A)
        merged.append((max(float(candidates[member][0]) for member in members), merged_xy))
    merged.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
    return merged


def _v3_refine_slice_centers(
    mask_yx: np.ndarray,
    distance_A: np.ndarray,
    centers: list[_V3SliceCenter],
    *,
    spacing_A: float,
    origin_xy_A: np.ndarray,
    box_xy_A: np.ndarray,
) -> list[_V3SliceCenter]:
    if not centers:
        return centers
    pixel_y, pixel_x = np.nonzero(mask_yx)
    pixel_xy = np.column_stack(
        [
            origin_xy_A[0] + (pixel_x.astype(float) + 0.5) * spacing_A,
            origin_xy_A[1] + (pixel_y.astype(float) + 0.5) * spacing_A,
        ]
    )
    center_xy = np.vstack([center.xy_A for center in centers])
    distances = _v3_periodic_xy_distance_matrix(center_xy, pixel_xy, box_xy_A).T
    assignments = np.argmin(distances, axis=1)
    refined: list[_V3SliceCenter] = []
    for index, center in enumerate(centers):
        selected = assignments == index
        if not np.any(selected):
            refined.append(center)
            continue
        deltas = pixel_xy[selected] - center.xy_A
        deltas -= box_xy_A * np.round(deltas / box_xy_A)
        weights = np.asarray(distance_A[pixel_y[selected], pixel_x[selected]], dtype=float) ** 2
        if not np.any(weights > 0.0):
            weights = np.ones_like(weights)
        xy = np.mod(center.xy_A + np.average(deltas, axis=0, weights=weights), box_xy_A)
        refined.append(_V3SliceCenter(xy_A=xy, wall_distance_A=center.wall_distance_A))
    return refined


def _v3_track_slice_centers(
    slices: tuple[_V3SliceCenterRecord, ...],
    *,
    target_box_A: np.ndarray,
    maximum_displacement_A: float,
    lower_index: int,
    upper_index: int,
) -> tuple[tuple[_V3CenterlineTrack, ...], int, dict[int, np.ndarray]]:
    box_xy = np.asarray(target_box_A[:2], dtype=float)
    tracks: dict[int, _V3MutableTrack] = {}
    active: dict[int, int] = {}
    branch_tracks: set[int] = set()
    branch_events = 0
    next_track_id = 0
    for slice_record in slices:
        centers = slice_record.centers
        if not centers:
            active = {}
            continue
        current_xy = np.vstack([center.xy_A for center in centers])
        previous_ids = sorted(active)
        current_to_track: dict[int, int] = {}
        if previous_ids:
            previous_xy = np.vstack(
                [tracks[track_id].wrapped_points_A[-1][:2] for track_id in previous_ids]
            )
            costs = _v3_periodic_xy_distance_matrix(previous_xy, current_xy, box_xy)
            neighbor_mask = costs <= float(maximum_displacement_A)
            row_branch = np.sum(neighbor_mask, axis=1) > 1
            col_branch = np.sum(neighbor_mask, axis=0) > 1
            branch_events += int(np.count_nonzero(row_branch))
            branch_events += int(np.count_nonzero(col_branch))
            for row in np.flatnonzero(row_branch):
                track_id = previous_ids[int(row)]
                branch_tracks.add(track_id)
                tracks[track_id].branch_z_A.append(float(slice_record.z_A))
            for col in np.flatnonzero(col_branch):
                for row in np.flatnonzero(neighbor_mask[:, int(col)]):
                    track_id = previous_ids[int(row)]
                    branch_tracks.add(track_id)
                    tracks[track_id].branch_z_A.append(float(slice_record.z_A))
            row_indices, col_indices = linear_sum_assignment(costs)
            for row, col in zip(row_indices, col_indices, strict=True):
                if costs[row, col] > maximum_displacement_A:
                    continue
                track_id = previous_ids[int(row)]
                current_to_track[int(col)] = track_id
                if np.sum(neighbor_mask[row]) > 1 or np.sum(neighbor_mask[:, col]) > 1:
                    branch_tracks.add(track_id)
        for center_index, center in enumerate(centers):
            track_id = current_to_track.get(center_index)
            point_wrapped = np.array(
                [center.xy_A[0], center.xy_A[1], slice_record.z_A],
                dtype=float,
            )
            if track_id is None:
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = _V3MutableTrack(
                    track_id=track_id,
                    slice_indices=[],
                    wrapped_points_A=[],
                    unwrapped_points_A=[],
                    wall_distances_A=[],
                    branch_z_A=[],
                )
                if previous_ids:
                    distances = _v3_periodic_xy_distance_matrix(
                        np.vstack(
                            [
                                tracks[previous_id].wrapped_points_A[-1][:2]
                                for previous_id in previous_ids
                            ]
                        ),
                        center.xy_A[np.newaxis, :],
                        box_xy,
                    )
                    if np.any(distances[:, 0] <= maximum_displacement_A):
                        branch_tracks.add(track_id)
                        tracks[track_id].branch_z_A.append(float(slice_record.z_A))
            track = tracks[track_id]
            if track.unwrapped_points_A:
                previous_unwrapped = track.unwrapped_points_A[-1]
                delta_xy = center.xy_A - track.wrapped_points_A[-1][:2]
                delta_xy -= box_xy * np.round(delta_xy / box_xy)
                point_unwrapped = np.array(
                    [
                        previous_unwrapped[0] + delta_xy[0],
                        previous_unwrapped[1] + delta_xy[1],
                        slice_record.z_A,
                    ],
                    dtype=float,
                )
            else:
                point_unwrapped = point_wrapped.copy()
            track.slice_indices.append(slice_record.z_index)
            track.wrapped_points_A.append(point_wrapped)
            track.unwrapped_points_A.append(point_unwrapped)
            track.wall_distances_A.append(center.wall_distance_A)
            current_to_track[center_index] = track_id
        active = {track_id: track_id for track_id in current_to_track.values()}
    output = []
    for track_id in sorted(tracks):
        track = tracks[track_id]
        indices = np.asarray(track.slice_indices, dtype=int)
        wrapped = np.vstack(track.wrapped_points_A)
        unwrapped = np.vstack(track.unwrapped_points_A)
        touches_lower = bool(indices.size and indices[0] == lower_index)
        touches_upper = bool(indices.size and indices[-1] == upper_index)
        output.append(
            _V3CenterlineTrack(
                track_id=track_id,
                slice_indices=indices,
                points_wrapped_A=wrapped,
                points_unwrapped_A=unwrapped,
                wall_distances_A=np.asarray(track.wall_distances_A, dtype=float),
                touches_z_lower=touches_lower,
                touches_z_upper=touches_upper,
                is_through=touches_lower and touches_upper,
                has_branch_neighborhood=track_id in branch_tracks,
            )
        )
    branch_z_by_track = {
        track_id: np.unique(np.asarray(track.branch_z_A, dtype=float))
        for track_id, track in tracks.items()
        if track.branch_z_A
    }
    return tuple(output), branch_events, branch_z_by_track


def _v3_periodic_xy_distance(
    first_xy_A: np.ndarray,
    second_xy_A: np.ndarray,
    box_xy_A: np.ndarray,
) -> float:
    delta = np.asarray(second_xy_A, dtype=float) - np.asarray(first_xy_A, dtype=float)
    delta -= box_xy_A * np.round(delta / box_xy_A)
    return float(np.linalg.norm(delta))


def _v3_periodic_xy_distance_matrix(
    first_xy_A: np.ndarray,
    second_xy_A: np.ndarray,
    box_xy_A: np.ndarray,
) -> np.ndarray:
    delta = second_xy_A[np.newaxis, :, :] - first_xy_A[:, np.newaxis, :]
    delta -= box_xy_A[np.newaxis, np.newaxis, :] * np.round(
        delta / box_xy_A[np.newaxis, np.newaxis, :]
    )
    return np.linalg.norm(delta, axis=2)


def _count_points_inside_phase(points_A: np.ndarray, phase: _PhaseData) -> int:
    points = np.asarray(points_A, dtype=float)
    x = np.mod(points[:, 0] - phase.origin_A[0], phase.target_box_A[0])
    y = np.mod(points[:, 1] - phase.origin_A[1], phase.target_box_A[1])
    z = points[:, 2] - phase.origin_A[2]
    ix = np.floor(x / phase.spacing_A).astype(int)
    iy = np.floor(y / phase.spacing_A).astype(int)
    iz = np.floor(z / phase.spacing_A).astype(int)
    valid = (
        (ix >= 0)
        & (ix < phase.mask.shape[2])
        & (iy >= 0)
        & (iy < phase.mask.shape[1])
        & (iz >= 0)
        & (iz < phase.mask.shape[0])
    )
    inside = np.zeros(points.shape[0], dtype=bool)
    inside[valid] = phase.mask[iz[valid], iy[valid], ix[valid]]
    return int(np.count_nonzero(inside))


def _through_z_network_count(mask: np.ndarray) -> int:
    labels, count = ndimage.label(mask, structure=_six_connected_structure())
    if count == 0:
        return 0
    parent = np.arange(count + 1, dtype=int)
    for z_index in range(mask.shape[0]):
        for y_index in range(mask.shape[1]):
            _union_labels(labels[z_index, y_index, 0], labels[z_index, y_index, -1], parent)
        for x_index in range(mask.shape[2]):
            _union_labels(labels[z_index, 0, x_index], labels[z_index, -1, x_index], parent)
    lower = {_find(int(label), parent) for label in np.unique(labels[0]) if int(label) != 0}
    upper = {_find(int(label), parent) for label in np.unique(labels[-1]) if int(label) != 0}
    return len(lower & upper)


def _read_molecule_metrics(
    molecule_dir: Path,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    h5_path = molecule_dir / "placed_atoms.h5"
    cif_path = molecule_dir / "placed_structure.cif"
    if not h5_path.is_file():
        return {}
    try:
        with h5py.File(h5_path, "r") as handle:
            claimed_count = int(handle.attrs.get("count", 0))
            pore_volume = float(handle.attrs.get("pore_volume_A3", math.nan))
            target_box = np.asarray(handle.attrs.get("target_box_A", []), dtype=float)
            density_attr = float(handle.attrs.get("actual_density_g_cm3", math.nan))
            min_distance_attr = float(handle.attrs.get("minimum_interatomic_distance_A", math.inf))
            atom_positions = np.asarray(handle["atom_positions_A"], dtype=float)
            template_positions = np.asarray(handle["template_positions_A"], dtype=float)
            template_masses = np.asarray(handle["template_masses_g_mol"], dtype=float)
            atom_instance_index = np.asarray(handle["atom_instance_index"], dtype=int)
            transforms = handle.get("instance_transforms")
            transform_count = _transform_count(transforms, errors)
    except (OSError, KeyError, ValueError) as exc:
        errors.append(f"mandatory placed_atoms.h5 could not be read: {exc}")
        return {}

    atoms_per_instance = int(template_positions.shape[0])
    csv_count = _instances_csv_count(molecule_dir / "instances.csv", errors)
    independent_count, count_sources = _independent_molecule_count(
        atom_positions=atom_positions,
        atom_instance_index=atom_instance_index,
        atoms_per_instance=atoms_per_instance,
        transform_count=transform_count,
        csv_count=csv_count,
        errors=errors,
    )
    if claimed_count != independent_count:
        errors.append(
            "claimed molecule count in placed_atoms.h5 does not match independent sources "
            f"({claimed_count} != {independent_count})"
        )
    cif_atom_count = _cif_atom_count(cif_path, errors)
    if cif_atom_count != atom_positions.shape[0]:
        errors.append("placed_structure.cif atom count does not match placed_atoms.h5")
    total_mass_g = independent_count * float(np.sum(template_masses)) / _AVOGADRO_PER_MOL
    density = (
        total_mass_g / (pore_volume * _ANGSTROM3_TO_CM3)
        if np.isfinite(pore_volume) and pore_volume > 0.0
        else math.nan
    )
    min_distance = _minimum_interinstance_distance(atom_positions, atom_instance_index, target_box)
    rmsd = _rigid_template_rmsd(
        atom_positions,
        template_positions,
        independent_count,
        atoms_per_instance,
    )
    density_delta = _finite_delta(density, density_attr)
    min_distance_delta = _finite_delta(min_distance, min_distance_attr)
    if density_delta > 1e-9:
        errors.append("placed_atoms.h5 density does not match independent mass/volume calculation")
    if min_distance_delta > 1e-9:
        warnings.append("placed_atoms.h5 minimum-distance metadata differs from independent value")
    return {
        "instance_count": independent_count,
        "claimed_count": claimed_count,
        "count_sources": count_sources,
        "atom_count": int(atom_positions.shape[0]),
        "cif_atom_count": cif_atom_count,
        "total_mass_g": total_mass_g,
        "density_g_cm3": density,
        "density_metadata_delta": density_delta,
        "minimum_interatomic_distance_A": min_distance,
        "minimum_distance_metadata_delta": min_distance_delta,
        "rigid_template_rmsd_A": rmsd,
    }


def _read_glb_metrics(
    qa: Path,
    phase: _PhaseData | None,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    path = qa / "semiconductor_solid_target.glb"
    if not path.is_file():
        warnings.append("semiconductor_solid_target.glb is unavailable for mesh validation")
        return {"available": False, "bounds_match": False}
    try:
        scene = trimesh.load(path, force="scene")
    except (OSError, ValueError) as exc:
        errors.append(f"semiconductor_solid_target.glb could not be loaded: {exc}")
        return {"available": False, "bounds_match": False}
    geometry_count = len(scene.geometry)
    if phase is None or scene.bounds is None:
        return {"available": True, "geometry_count": geometry_count, "bounds_match": False}
    bounds = np.asarray(scene.bounds, dtype=float)
    expected = np.vstack([phase.origin_A, phase.origin_A + phase.target_box_A])
    bounds_match = bool(np.allclose(bounds, expected, atol=max(phase.spacing_A, 1.0) * 1e-6))
    if not bounds_match:
        errors.append("semiconductor_solid_target.glb bounds do not match final_phase target box")
    solid_volume = float(sum(abs(mesh.volume) for mesh in scene.geometry.values()))
    expected_solid_volume = float((1.0 - np.mean(phase.mask)) * np.prod(phase.target_box_A))
    volume_relative_error = (
        abs(solid_volume - expected_solid_volume) / expected_solid_volume
        if expected_solid_volume > 0.0
        else 0.0
    )
    if volume_relative_error > 0.15:
        warnings.append("GLB sampled occupancy proxy differs from final_phase by more than 15%")
    occupancy = _glb_occupancy_metrics(scene, phase, errors)
    mismatch = _optional_float(occupancy.get("occupancy_mismatch_fraction"))
    if np.isfinite(mismatch) and mismatch > _GLB_OCCUPANCY_ERROR_TOLERANCE:
        errors.append(
            "GLB occupancy differs from final_phase.h5 by more than "
            f"{_GLB_OCCUPANCY_ERROR_TOLERANCE:g}"
        )
    elif np.isfinite(mismatch) and mismatch > _GLB_OCCUPANCY_WARNING_TOLERANCE:
        warnings.append(
            "GLB occupancy differs from final_phase.h5 by more than "
            f"{_GLB_OCCUPANCY_WARNING_TOLERANCE:g}"
        )
    metadata = getattr(scene, "metadata", {}) or {}
    return {
        "available": True,
        "geometry_count": int(geometry_count),
        "bounds_A": bounds.tolist(),
        "bounds_match": bounds_match,
        "solid_mesh_volume_A3": solid_volume,
        "expected_solid_volume_A3": expected_solid_volume,
        "solid_volume_relative_error": volume_relative_error,
        "metadata_target_box_match": metadata.get("target_box_A") == phase.target_box_A.tolist(),
        **occupancy,
    }


def _glb_occupancy_metrics(
    scene: trimesh.Scene,
    phase: _PhaseData,
    errors: list[str],
) -> dict[str, Any]:
    mesh = _scene_mesh(scene)
    if mesh is None or mesh.faces.size == 0:
        errors.append("semiconductor_solid_target.glb does not contain triangle mesh faces")
        return {
            "occupancy_sample_count": 0,
            "occupancy_mismatch_fraction": math.inf,
        }
    indices = _occupancy_sample_indices(phase.mask.size)
    points = _voxel_centers_for_indices(phase, indices)
    expected_semiconductor = np.logical_not(phase.mask.ravel()[indices])
    observed_semiconductor = _points_inside_triangle_mesh(mesh, points)
    mismatch = float(np.mean(observed_semiconductor != expected_semiconductor))
    return {
        "occupancy_sample_count": int(indices.size),
        "occupancy_mismatch_fraction": mismatch,
        "occupancy_match_fraction": float(1.0 - mismatch),
    }


def _scene_mesh(scene: trimesh.Scene) -> trimesh.Trimesh | None:
    if hasattr(scene, "to_mesh"):
        mesh = scene.to_mesh()
        if isinstance(mesh, trimesh.Trimesh):
            return mesh
    meshes = [mesh for mesh in scene.geometry.values() if isinstance(mesh, trimesh.Trimesh)]
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0].copy()


def _occupancy_sample_indices(total: int) -> np.ndarray:
    if total <= _GLB_OCCUPANCY_MAX_SAMPLES:
        return np.arange(total, dtype=np.int64)
    return np.unique(np.linspace(0, total - 1, _GLB_OCCUPANCY_MAX_SAMPLES, dtype=np.int64))


def _voxel_centers_for_indices(phase: _PhaseData, indices: np.ndarray) -> np.ndarray:
    _nz, ny, nx = phase.mask.shape
    x_index = indices % nx
    y_index = (indices // nx) % ny
    z_index = indices // (nx * ny)
    return np.column_stack(
        [
            phase.origin_A[0] + (x_index.astype(float) + 0.5) * phase.spacing_A,
            phase.origin_A[1] + (y_index.astype(float) + 0.5) * phase.spacing_A,
            phase.origin_A[2] + (z_index.astype(float) + 0.5) * phase.spacing_A,
        ]
    )


def _points_inside_triangle_mesh(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    triangles = np.asarray(mesh.triangles, dtype=float)
    direction = np.asarray([1.0, 0.3713906763541037, 0.13736056394868904], dtype=float)
    direction /= np.linalg.norm(direction)
    inside = np.zeros(points.shape[0], dtype=bool)
    eps = 1.0e-9
    for point_start in range(0, points.shape[0], _RAY_POINT_CHUNK):
        point_stop = min(point_start + _RAY_POINT_CHUNK, points.shape[0])
        chunk = points[point_start:point_stop]
        hit_counts = np.zeros(chunk.shape[0], dtype=np.int64)
        for tri_start in range(0, triangles.shape[0], _RAY_TRIANGLE_CHUNK):
            tri_stop = min(tri_start + _RAY_TRIANGLE_CHUNK, triangles.shape[0])
            tri = triangles[tri_start:tri_stop]
            edge1 = tri[:, 1] - tri[:, 0]
            edge2 = tri[:, 2] - tri[:, 0]
            h_vec = np.cross(direction, edge2)
            det = np.einsum("ij,ij->i", edge1, h_vec)
            valid = np.abs(det) > eps
            inv_det = np.zeros_like(det)
            inv_det[valid] = 1.0 / det[valid]
            s_vec = chunk[:, np.newaxis, :] - tri[np.newaxis, :, 0]
            u_coord = inv_det[np.newaxis, :] * np.einsum("ptj,tj->pt", s_vec, h_vec)
            q_vec = np.cross(s_vec, edge1[np.newaxis, :, :])
            v_coord = inv_det[np.newaxis, :] * np.einsum("j,ptj->pt", direction, q_vec)
            distance = inv_det[np.newaxis, :] * np.einsum("tj,ptj->pt", edge2, q_vec)
            hits = (
                valid[np.newaxis, :]
                & (u_coord >= -eps)
                & (v_coord >= -eps)
                & (u_coord + v_coord <= 1.0 + eps)
                & (distance > eps)
            )
            hit_counts += np.count_nonzero(hits, axis=1)
        inside[point_start:point_stop] = (hit_counts % 2) == 1
    return inside


def _verify_checksums(
    qa: Path,
    errors: list[str],
    *,
    schema_version: int = 1,
) -> dict[str, Any]:
    path = qa / "checksums.sha256"
    if not path.is_file():
        return {"verified_count": 0, "mismatches": []}
    qa_root = qa.resolve()
    mismatches: list[str] = []
    unsafe_paths: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    covered: set[str] = set()
    verified = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            mismatches.append(line)
            continue
        relative = relative.strip()
        if relative.startswith("independent-validation"):
            continue
        relative_path = Path(relative)
        if relative_path.is_absolute() or relative_path.drive:
            unsafe_paths.append(relative)
            continue
        if relative in seen:
            duplicates.append(relative)
        seen.add(relative)
        candidate = (qa / relative).resolve()
        try:
            candidate.relative_to(qa_root)
        except ValueError:
            unsafe_paths.append(relative)
            continue
        if not candidate.is_file():
            mismatches.append(relative)
            continue
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append(relative)
        covered.add(relative)
        verified += 1
    required = _required_checksum_files(qa, schema_version=schema_version)
    missing_required = sorted(required - covered)
    if duplicates:
        errors.append("checksums.sha256 has duplicate entries: " + ", ".join(sorted(duplicates)))
    if unsafe_paths:
        errors.append(
            "checksums.sha256 contains paths outside qa_export or absolute paths: "
            + ", ".join(sorted(unsafe_paths))
        )
    if missing_required:
        errors.append(
            "checksums.sha256 is missing required entries: " + ", ".join(missing_required)
        )
    if mismatches:
        errors.append("checksums.sha256 has mismatches: " + ", ".join(mismatches))
    return {
        "verified_count": verified,
        "mismatches": mismatches,
        "duplicates": sorted(duplicates),
        "unsafe_paths": sorted(unsafe_paths),
        "missing_required": missing_required,
    }


def _required_checksum_files(qa: Path, *, schema_version: int = 1) -> set[str]:
    required = set(_V3_CHECKSUM_REQUIRED_FILES if schema_version >= 3 else _CHECKSUM_REQUIRED_FILES)
    source_dir = qa / "molecules" / "source"
    if source_dir.is_dir():
        required.update(
            path.relative_to(qa).as_posix() for path in source_dir.rglob("*") if path.is_file()
        )
    return required


def _target_compliance(
    config: dict[str, Any],
    contract: dict[str, Any],
    phase: dict[str, Any],
    units: dict[str, Any],
    molecules: dict[str, Any],
    final_geometry: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    target_box = _box_from_config(config)
    phase_box = np.asarray(phase.get("target_box_A", []), dtype=float)
    contract_box = np.asarray(contract.get("target_box_A", []), dtype=float)
    target_box_match = (
        target_box.shape == (3,)
        and phase_box.shape == (3,)
        and contract_box.shape == (3,)
        and np.allclose(target_box, phase_box)
        and np.allclose(target_box, contract_box)
    )
    target_porosity = (
        _nested_float(config, ["formal_targets", "proportion", "porosity"])
        if _config_schema_version(config) >= 3
        else _nested_float(config, ["pores", "target_porosity"])
    )
    realized_porosity = float(phase.get("porosity", math.nan))
    porosity_error = abs(realized_porosity - target_porosity)
    voxel_count = int(phase.get("pore_voxels", 0)) + int(phase.get("semiconductor_voxels", 0))
    porosity_tolerance = max(
        _POROSITY_TARGET_TOLERANCE,
        1.0 / voxel_count if voxel_count > 0 else 0.0,
    )
    porosity_ok = bool(np.isfinite(porosity_error) and porosity_error <= porosity_tolerance)
    molecule_required = isinstance(config.get("pore_material"), dict)
    expected_count = _nested_int(config, ["pore_material", "molecule_count"])
    molecule_count_match = (
        True
        if not molecule_required
        else expected_count is None or expected_count == molecules.get("instance_count")
    )
    target_density = _nested_float(config, ["pore_material", "target_density_g_cm3"])
    realized_density = _optional_float(molecules.get("density_g_cm3"))
    density_relative_error = (
        abs(realized_density - target_density) / target_density
        if np.isfinite(target_density) and target_density > 0.0 and np.isfinite(realized_density)
        else math.nan
    )
    density_ok = bool(
        not molecule_required
        or not np.isfinite(target_density)
        or (
            np.isfinite(density_relative_error)
            and density_relative_error <= _DENSITY_TARGET_RELATIVE_TOLERANCE
        )
    )
    if not target_box_match:
        errors.append("target_box_A is inconsistent between config, contract, and final_phase")
    if not porosity_ok:
        errors.append(
            "target porosity is outside tolerance: "
            f"target={target_porosity}, realized={realized_porosity}, "
            f"absolute_error={porosity_error}"
        )
    if not molecule_count_match:
        errors.append(
            "target molecule count does not match independent molecule count: "
            f"target={expected_count}, realized={molecules.get('instance_count')}"
        )
    if not density_ok:
        errors.append(
            "target density does not match independent molecule density: "
            f"target={target_density}, realized={realized_density}, "
            f"relative_error={density_relative_error}"
        )
    compliance = {
        "target_box_match": bool(target_box_match),
        "target_porosity": target_porosity,
        "realized_porosity": realized_porosity,
        "porosity_absolute_error": porosity_error,
        "porosity_tolerance": porosity_tolerance,
        "porosity_within_tolerance": porosity_ok,
        "molecule_count_match": bool(molecule_count_match),
        "target_density_g_cm3": target_density if np.isfinite(target_density) else None,
        "realized_density_g_cm3": realized_density if np.isfinite(realized_density) else None,
        "density_relative_error": density_relative_error
        if np.isfinite(density_relative_error)
        else None,
        "density_relative_tolerance": _DENSITY_TARGET_RELATIVE_TOLERANCE,
        "density_within_tolerance": density_ok,
    }
    if _config_schema_version(config) >= 3:
        compliance.update(
            _v3_target_compliance(
                config,
                phase,
                units,
                final_geometry,
                errors,
            )
        )
    return compliance


def _v3_target_compliance(
    config: dict[str, Any],
    phase: dict[str, Any],
    units: dict[str, Any],
    final_geometry: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    samples = final_geometry.get("rebuilt_formal_samples", {})
    samples = samples if isinstance(samples, dict) else {}
    shape = config.get("formal_targets", {}).get("shape", {})
    shape = shape if isinstance(shape, dict) else {}
    pore = config.get("pore_constraints", {})
    pore = pore if isinstance(pore, dict) else {}
    evaluate_through_targets = str(pore.get("z_connectivity", "unrestricted")) == "all_components"
    output: dict[str, Any] = {}
    distribution_specs = {
        "equivalent_diameter_A": shape.get("equivalent_diameter_A"),
        "theta_xz_deg": _v3_paired_orientation_marginal(
            shape.get("orientation"),
            "theta_xz_deg",
        ),
        "theta_xy_deg": _v3_paired_orientation_marginal(
            shape.get("orientation"),
            "theta_xy_deg",
        ),
        "compact_eta": shape.get("compact_aspect_ratio"),
        "channel_eta": shape.get("channel_aspect_ratio"),
        "channel_tau": shape.get("channel_tortuosity"),
        "curvature_fluctuation": shape.get("curvature_fluctuation"),
    }
    sample_keys = {
        "equivalent_diameter_A": "equivalent_diameter_A",
        "theta_xz_deg": "theta_xz_deg",
        "theta_xy_deg": "theta_xy_deg",
        "compact_eta": "compact_eta",
        "channel_eta": "channel_eta",
        "channel_tau": "channel_tau",
        "curvature_fluctuation": "curvature_fluctuation",
    }
    through_dependent_names = {
        "equivalent_diameter_A",
        "theta_xz_deg",
        "theta_xy_deg",
        "channel_eta",
        "channel_tau",
        "curvature_fluctuation",
    }
    for name, target in distribution_specs.items():
        sample_values = np.asarray(samples.get(sample_keys[name], []), dtype=float)
        evaluated = bool(
            target is not None
            and (name not in through_dependent_names or evaluate_through_targets)
        )
        result = (
            _v3_distribution_target_result(
                sample_values,
                target,
                constant_relative_tolerance=(
                    _V3_COMPACT_ETA_CONSTANT_RELATIVE_TOLERANCE
                    if name == "compact_eta"
                    else _V3_CHANNEL_GEOMETRY_CONSTANT_RELATIVE_TOLERANCE
                    if name in {"channel_eta", "channel_tau"}
                    else None
                ),
            )
            if evaluated
            else {
                "passed": None,
                "sample_count": int(_v3_finite_1d(sample_values).size),
                "ks": None,
                "normalized_wasserstein": None,
            }
        )
        output[f"{name}_evaluated"] = evaluated
        output[f"{name}_within_tolerance"] = result["passed"]
        output[f"{name}_sample_count"] = result["sample_count"]
        output[f"{name}_ks"] = result["ks"]
        output[f"{name}_normalized_wasserstein"] = result["normalized_wasserstein"]
        if evaluated and not result["passed"]:
            errors.append(
                f"schema-v3 target {name} does not match independent final_phase "
                f"measurements: sample_count={result['sample_count']}, "
                f"ks={result['ks']}, "
                f"normalized_wasserstein={result['normalized_wasserstein']}"
            )
    orientation_target = shape.get("orientation")
    paired_evaluated = bool(orientation_target is not None and evaluate_through_targets)
    paired_result = (
        _v3_compare_paired_orientation_pairs(
            np.asarray(samples.get("paired_orientation", []), dtype=float).reshape((-1, 2)),
            orientation_target,
        )
        if paired_evaluated
        else None
    )
    output["paired_orientation_evaluated"] = paired_evaluated
    if not paired_evaluated or paired_result is None:
        output["paired_orientation_within_tolerance"] = None
        output["paired_orientation_sample_count"] = 0
        output["paired_orientation_unassigned_pair_count"] = 0
        output["paired_orientation_component_count_errors"] = {}
    else:
        output["paired_orientation_within_tolerance"] = paired_result["passed"]
        output["paired_orientation_sample_count"] = paired_result["sample_count"]
        output["paired_orientation_unassigned_pair_count"] = paired_result["unassigned_pair_count"]
        output["paired_orientation_component_count_errors"] = paired_result[
            "component_count_errors"
        ]
        if not paired_result["passed"]:
            errors.append(
                "schema-v3 target paired orientation does not match independent "
                "final_phase measurements"
            )
    center_result = _v3_center_distance_target_compliance(
        config,
        final_geometry,
    )
    center_target = (
        config.get("formal_targets", {}).get("position_quantity", {}).get("center_distance_xy")
    )
    center_evaluated = bool(isinstance(center_target, dict) and evaluate_through_targets)
    output["g_xy_evaluated"] = center_evaluated
    output["g_xy_within_tolerance"] = (
        bool(center_result["passed"]) if center_evaluated else None
    )
    output["g_xy_weighted_loss"] = (
        center_result["weighted_loss"] if center_evaluated else None
    )
    output["g_xy_pair_count"] = center_result["pair_count"]
    if center_evaluated and not center_result["passed"]:
        errors.append(
            "schema-v3 target g_xy does not match independent final_phase "
            f"measurements: weighted_loss={center_result['weighted_loss']}, "
            f"pair_count={center_result['pair_count']}"
        )
    output.update(
        _v3_constraint_compliance(
            config,
            phase,
            units,
            final_geometry,
            errors,
        )
    )
    return output


def _v3_distribution_target_result(
    samples: np.ndarray,
    target: Any,
    *,
    constant_relative_tolerance: float | None = None,
) -> dict[str, Any]:
    sample_values = _v3_finite_1d(samples)
    if target is None:
        return {
            "passed": True,
            "sample_count": int(sample_values.size),
            "ks": 0.0,
            "normalized_wasserstein": 0.0,
        }
    if sample_values.size == 0:
        return {
            "passed": False,
            "sample_count": 0,
            "ks": math.inf,
            "normalized_wasserstein": math.inf,
        }
    target_dict = _v3_distribution_dict(target)
    ks = _v3_ks_distance(sample_values, target_dict)
    target_values = _v3_target_quantile_samples(target_dict, max(sample_values.size, 512))
    wasserstein = float(stats.wasserstein_distance(sample_values, target_values))
    normalized = wasserstein / _v3_distribution_scale(target_values, target_dict)
    passed = bool(
        ks <= _V3_DISTRIBUTION_KS_LIMIT and normalized <= _V3_DISTRIBUTION_WASSERSTEIN_LIMIT
    )
    if constant_relative_tolerance is not None and target_dict.get("family") == "constant":
        target_value = float(target_dict["value"])
        relative_errors = np.abs(sample_values - target_value) / max(
            abs(target_value),
            1.0e-12,
        )
        if np.all(relative_errors <= constant_relative_tolerance):
            passed = bool(normalized <= _V3_DISTRIBUTION_WASSERSTEIN_LIMIT)
            ks = 0.0
    return {
        "passed": passed,
        "sample_count": int(sample_values.size),
        "ks": float(ks),
        "normalized_wasserstein": float(normalized),
    }


def _v3_finite_1d(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def _v3_distribution_dict(target: Any) -> dict[str, Any]:
    if isinstance(target, dict):
        return {key: value for key, value in target.items() if value is not None}
    return {}


def _v3_ks_distance(samples: np.ndarray, target: dict[str, Any]) -> float:
    if target.get("family") == "constant":
        value = float(target.get("value", 0.0))
        return 0.0 if np.allclose(samples, value) else 1.0
    values = np.sort(samples)
    n = values.size
    cdf = _v3_distribution_cdf(values, target)
    empirical_upper = np.arange(1, n + 1, dtype=float) / n
    empirical_lower = np.arange(0, n, dtype=float) / n
    return float(np.max(np.maximum(np.abs(empirical_upper - cdf), np.abs(cdf - empirical_lower))))


def _v3_distribution_cdf(values: np.ndarray, target: dict[str, Any]) -> np.ndarray:
    family = str(target.get("family", "constant")).lower()
    if family == "constant":
        return (values >= float(target.get("value", 0.0))).astype(float)
    if family == "mixture":
        components = _v3_mixture_components(target)
        if not components:
            return np.zeros_like(values, dtype=float)
        output = np.zeros_like(values, dtype=float)
        for weight, component in components:
            output += weight * _v3_distribution_cdf(values, component)
        return np.clip(output, 0.0, 1.0)
    return np.asarray(_v3_scipy_distribution(target).cdf(values), dtype=float)


def _v3_target_quantile_samples(target: dict[str, Any], count: int) -> np.ndarray:
    count = max(int(count), 1)
    probabilities = (np.arange(count, dtype=float) + 0.5) / count
    family = str(target.get("family", "constant")).lower()
    if family == "constant":
        return np.full(count, float(target.get("value", 0.0)))
    if family == "mixture":
        components = _v3_mixture_components(target)
        if not components:
            return np.zeros(count, dtype=float)
        weights = np.asarray([weight for weight, _component in components], dtype=float)
        counts = _v3_allocate_largest_remainder(weights, count)
        values = [
            _v3_target_quantile_samples(component, int(component_count))
            for component_count, (_weight, component) in zip(counts, components, strict=True)
            if component_count > 0
        ]
        return np.sort(np.concatenate(values)) if values else np.zeros(count, dtype=float)
    return np.asarray(_v3_scipy_distribution(target).ppf(probabilities), dtype=float)


def _v3_scipy_distribution(target: dict[str, Any]) -> Any:
    family = str(target["family"]).lower()
    if family == "lognormal":
        sigma = _v3_required_float(target, "sigma", "s")
        loc = float(target.get("loc", 0.0))
        scale = float(
            target.get(
                "scale",
                math.exp(float(target.get("mean", target.get("mu", 0.0)))),
            )
        )
        return stats.lognorm(s=sigma, loc=loc, scale=scale)
    if family == "gamma":
        shape = _v3_required_float(target, "alpha", "shape", "k")
        scale = float(target.get("scale", target.get("theta", 1.0)))
        loc = float(target.get("loc", 0.0))
        return stats.gamma(a=shape, loc=loc, scale=scale)
    if family in {"weibull", "weibull_min"}:
        shape = _v3_required_float(target, "shape", "k", "alpha")
        scale = float(target.get("scale", 1.0))
        loc = float(target.get("loc", 0.0))
        return stats.weibull_min(c=shape, loc=loc, scale=scale)
    if family in {"truncated_normal", "truncnorm"}:
        mean = float(target.get("mean", target.get("loc", 0.0)))
        sigma = _v3_required_float(target, "sigma", "s")
        lower = float(target["lower"])
        upper = float(target["upper"])
        return stats.truncnorm(
            a=(lower - mean) / sigma,
            b=(upper - mean) / sigma,
            loc=mean,
            scale=sigma,
        )
    if family == "beta":
        alpha = _v3_required_float(target, "alpha")
        beta = _v3_required_float(target, "beta")
        lower = float(target.get("lower", target.get("minimum", 0.0)))
        if "upper" in target:
            scale = float(target["upper"]) - lower
        elif "maximum" in target:
            scale = float(target["maximum"]) - lower
        else:
            scale = float(target.get("scale", 1.0))
        return stats.beta(a=alpha, b=beta, loc=lower, scale=scale)
    raise ValueError(f"unsupported distribution family: {family}")


def _v3_required_float(target: dict[str, Any], *names: str) -> float:
    for name in names:
        if name in target:
            return float(target[name])
    raise ValueError("distribution requires one of: " + ", ".join(names))


def _v3_allocate_largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    exact = values / float(np.sum(values)) * int(total)
    counts = np.floor(exact).astype(int)
    remainder = int(total) - int(np.sum(counts))
    if remainder:
        order = np.lexsort((np.arange(values.size), -(exact - counts)))
        counts[order[:remainder]] += 1
    return counts


def _v3_mixture_components(target: dict[str, Any]) -> list[tuple[float, dict[str, Any]]]:
    components = target.get("components", [])
    if not isinstance(components, list | tuple):
        return []
    parsed: list[tuple[float, dict[str, Any]]] = []
    for component in components:
        if not isinstance(component, dict):
            continue
        weight = _optional_float(component.get("weight", 0.0))
        if not np.isfinite(weight) or weight <= 0.0:
            continue
        distribution = {
            key: value for key, value in component.items() if key != "weight" and value is not None
        }
        parsed.append((float(weight), distribution))
    total = sum(weight for weight, _component in parsed)
    if total <= 0.0:
        return []
    return [(weight / total, component) for weight, component in parsed]


def _v3_distribution_scale(target_values: np.ndarray, target: dict[str, Any]) -> float:
    finite = _v3_finite_1d(target_values)
    if finite.size:
        spread = float(np.max(finite) - np.min(finite))
        if spread > 1.0e-12:
            return spread
    if target.get("family") == "constant":
        return max(abs(float(target.get("value", 0.0))), 1.0)
    return max(abs(float(np.mean(finite))) if finite.size else 0.0, 1.0)


def _v3_paired_orientation_marginal(
    target: Any,
    field_name: str,
) -> dict[str, Any] | None:
    if not isinstance(target, dict):
        return None
    components = target.get("components")
    if not isinstance(components, list | tuple):
        return None
    output = []
    for component in components:
        if not isinstance(component, dict) or not isinstance(component.get(field_name), dict):
            continue
        output.append(
            {
                "weight": float(component.get("weight", 0.0)),
                **{key: value for key, value in component[field_name].items() if value is not None},
            }
        )
    return {"family": "mixture", "components": output} if output else None


def _v3_compare_paired_orientation_pairs(
    samples: np.ndarray,
    target: Any,
) -> dict[str, Any] | None:
    if target is None:
        return None
    values = np.asarray(samples, dtype=float).reshape((-1, 2))
    target_dict = _v3_distribution_dict(target)
    components = target_dict.get("components") or ()
    if values.size == 0 or not components:
        return {
            "passed": False,
            "sample_count": int(values.shape[0]),
            "unassigned_pair_count": int(values.shape[0]),
            "component_count_errors": {},
        }

    scores = np.full((values.shape[0], len(components)), -np.inf, dtype=float)
    weights = np.asarray([float(component["weight"]) for component in components])
    for component_index, component in enumerate(components):
        score = np.full(
            values.shape[0],
            np.log(max(weights[component_index], 1.0e-300)),
        )
        supported = np.ones(values.shape[0], dtype=bool)
        for value_index, field_name in enumerate(("theta_xz_deg", "theta_xy_deg")):
            distribution = _v3_distribution_dict(component[field_name])
            lower = float(distribution.get("lower", distribution.get("minimum", 0.0)))
            upper = float(distribution.get("upper", distribution.get("maximum", 90.0)))
            in_support = (values[:, value_index] >= lower) & (values[:, value_index] <= upper)
            supported &= in_support
            normalized = np.clip(
                (values[:, value_index] - lower) / max(upper - lower, 1.0e-12),
                1.0e-12,
                1.0 - 1.0e-12,
            )
            score += stats.beta.logpdf(
                normalized,
                a=_v3_required_float(distribution, "alpha"),
                b=_v3_required_float(distribution, "beta"),
            ) - np.log(max(upper - lower, 1.0e-12))
        scores[supported, component_index] = score[supported]

    assigned = np.argmax(scores, axis=1)
    unassigned = ~np.any(np.isfinite(scores), axis=1)
    actual_counts = np.bincount(assigned[~unassigned], minlength=len(components))
    expected_counts = _v3_allocate_largest_remainder(weights, values.shape[0])
    tolerance = _V3_MIXTURE_WEIGHT_ABSOLUTE_TOLERANCE + 0.5 / max(
        values.shape[0],
        1,
    )
    count_errors = {
        f"component_{index}": float((actual - expected) / max(values.shape[0], 1))
        for index, (actual, expected) in enumerate(zip(actual_counts, expected_counts, strict=True))
    }
    return {
        "passed": bool(
            not np.any(unassigned)
            and all(abs(error) <= tolerance for error in count_errors.values())
        ),
        "sample_count": int(values.shape[0]),
        "unassigned_pair_count": int(np.count_nonzero(unassigned)),
        "component_count_errors": count_errors,
    }


def _v3_constraint_compliance(
    config: dict[str, Any],
    phase: dict[str, Any],
    units: dict[str, Any],
    final_geometry: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    matrix = config.get("matrix_constraints", {})
    matrix = matrix if isinstance(matrix, dict) else {}
    pore = config.get("pore_constraints", {})
    pore = pore if isinstance(pore, dict) else {}

    matrix_enabled = bool(matrix.get("enabled", True))
    require_x_percolation = bool(matrix.get("require_x_percolation", True))
    x_percolates = bool(phase.get("matrix_x_percolates", False))
    minimum_cross_section = _optional_float(
        phase.get("minimum_semiconductor_cross_section_fraction")
    )
    required_cross_section = _optional_float(matrix.get("minimum_cross_section_fraction", 0.0))
    overlap_fraction = _optional_float(units.get("overlap_fraction"))
    maximum_overlap = _optional_float(matrix.get("maximum_overlap_fraction", 0.0))

    x_ok = bool(not matrix_enabled or not require_x_percolation or x_percolates)
    cross_section_ok = bool(
        not matrix_enabled
        or (
            np.isfinite(minimum_cross_section)
            and np.isfinite(required_cross_section)
            and minimum_cross_section >= required_cross_section
        )
    )
    overlap_ok = bool(
        not matrix_enabled
        or (
            np.isfinite(overlap_fraction)
            and np.isfinite(maximum_overlap)
            and overlap_fraction <= maximum_overlap
        )
    )
    if not x_ok:
        errors.append("semiconductor matrix does not percolate along periodic x")
    if not cross_section_ok:
        errors.append("minimum semiconductor cross-section fraction is below configured limit")
    if not overlap_ok:
        errors.append(
            "pore overlap fraction is above configured limit or not independently evaluable"
        )

    connected_components = int(final_geometry.get("connected_pore_component_count", 0))
    through_components = int(final_geometry.get("through_network_count", 0))
    through_centerlines = int(final_geometry.get("rebuilt_through_centerline_count", 0))
    valid_through_cross_sections = len(
        final_geometry.get("rebuilt_formal_samples", {}).get("equivalent_diameter_A", [])
    )
    z_mode = str(pore.get("z_connectivity", "unrestricted"))
    all_components_through = bool(
        connected_components > 0 and through_components == connected_components
    )
    z_ok = z_mode == "unrestricted" or (z_mode == "all_components" and all_components_through)
    minimum_through = int(pore.get("minimum_through_centerlines", 0))
    minimum_valid_sections = int(pore.get("minimum_valid_cross_sections", 0))
    through_ok = through_centerlines >= minimum_through
    valid_sections_ok = valid_through_cross_sections >= minimum_valid_sections
    if not z_ok:
        errors.append("not all pore components connect both finite z surfaces")
    if not through_ok:
        errors.append("minimum through centerlines requirement is not met")
    if not valid_sections_ok:
        errors.append("minimum valid cross-sections requirement is not met")

    return {
        "matrix_constraints_enabled": matrix_enabled,
        "matrix_x_percolates": x_percolates,
        "matrix_x_percolation_within_constraint": (
            x_ok if matrix_enabled and require_x_percolation else None
        ),
        "minimum_semiconductor_cross_section_fraction": minimum_cross_section,
        "minimum_semiconductor_cross_section_within_constraint": (
            cross_section_ok if matrix_enabled else None
        ),
        "pore_overlap_fraction": overlap_fraction,
        "pore_overlap_within_constraint": overlap_ok if matrix_enabled else None,
        "pore_z_connectivity_mode": z_mode,
        "connected_pore_component_count": connected_components,
        "through_pore_component_count": through_components,
        "all_pore_components_through_z": all_components_through,
        "pore_z_connectivity_within_constraint": z_ok if z_mode == "all_components" else None,
        "through_centerline_count": through_centerlines,
        "minimum_through_centerlines": minimum_through,
        "minimum_through_centerlines_within_constraint": (
            through_ok if minimum_through > 0 else None
        ),
        "valid_through_cross_section_count": valid_through_cross_sections,
        "minimum_valid_cross_sections": minimum_valid_sections,
        "minimum_valid_cross_sections_within_constraint": (
            valid_sections_ok if minimum_valid_sections > 0 else None
        ),
    }


def _v3_center_distance_target_compliance(
    config: dict[str, Any],
    final_geometry: dict[str, Any],
) -> dict[str, Any]:
    measured = final_geometry.get("rebuilt_center_distance_xy", {})
    measured = measured if isinstance(measured, dict) else {}
    target_spec = (
        config.get("formal_targets", {}).get("position_quantity", {}).get("center_distance_xy")
    )
    bin_centers = np.asarray(measured.get("bin_centers_A", []), dtype=float)
    observed = np.asarray(measured.get("g_xy", []), dtype=float)
    reference = np.asarray(measured.get("reference_pair_counts", []), dtype=float)
    pair_count = int(measured.get("pair_count", 0) or 0)
    if not isinstance(target_spec, dict):
        return {"passed": True, "weighted_loss": 0.0, "pair_count": pair_count}
    target = _v3_evaluate_absolute_distance_target(
        bin_centers,
        target_spec.get("components", []),
    )
    valid = reference > 0.0
    if pair_count == 0 or not np.any(valid) or observed.shape != target.shape:
        loss = math.inf
    else:
        weights = np.maximum(target[valid], 1.0e-12)
        loss = float(np.average((observed[valid] - target[valid]) ** 2, weights=weights))
    return {
        "passed": bool(loss <= _V3_RDF_LOSS_LIMIT),
        "weighted_loss": loss,
        "pair_count": pair_count,
    }


def _v3_evaluate_absolute_distance_target(
    distance_A: np.ndarray,
    components: Any,
) -> np.ndarray:
    values = np.ones_like(np.asarray(distance_A, dtype=float))
    if not isinstance(components, list | tuple):
        return values
    for component in components:
        if not isinstance(component, dict):
            continue
        center = _optional_float(component.get("center_A"))
        width = _optional_float(component.get("width_A"))
        amplitude = _optional_float(component.get("amplitude"))
        if not (
            np.isfinite(center) and np.isfinite(width) and width > 0.0 and np.isfinite(amplitude)
        ):
            continue
        gaussian = np.exp(-0.5 * ((distance_A - center) / width) ** 2)
        kind = str(component.get("kind", "peak"))
        if kind == "peak":
            values += amplitude * gaussian
        elif kind in {"dip", "exclusion"}:
            values -= amplitude * gaussian
        elif kind == "oscillation":
            phase = 2.0 * np.pi * (distance_A - center) / width
            values += amplitude * gaussian * np.cos(phase)
    return np.maximum(values, 0.0)


def _compare_main_metrics(
    main_metrics: dict[str, Any],
    independent_metrics: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    consistency: dict[str, Any] = {"checked": False, "main_metrics_match": False}
    phase = independent_metrics.get("phase", {})
    molecules = independent_metrics.get("molecules", {})
    if not main_metrics or "porosity" not in phase:
        return consistency
    main_porosity = main_metrics.get("realized_porosity", main_metrics.get("porosity"))
    if main_porosity is None:
        warnings.append("main_metrics.json does not expose realized_porosity or porosity")
        return consistency
    porosity_delta = abs(float(main_porosity) - float(phase["porosity"]))
    has_molecule_metrics = bool(molecules)
    packing_count_match = (
        main_metrics.get("packing_count") is None
        if not has_molecule_metrics
        else main_metrics.get("packing_count") == molecules.get("instance_count")
    )
    density_delta = (
        0.0
        if not has_molecule_metrics
        else _finite_delta(
            _optional_float(main_metrics.get("actual_density_g_cm3")),
            _optional_float(molecules.get("density_g_cm3")),
        )
    )
    distance_delta = (
        0.0
        if not has_molecule_metrics
        else _finite_delta(
            _optional_float(main_metrics.get("minimum_interatomic_distance_A")),
            _optional_float(molecules.get("minimum_interatomic_distance_A")),
        )
    )
    main_metrics_match = (
        porosity_delta <= 1e-12
        and packing_count_match
        and density_delta <= 1e-9
        and distance_delta <= 1e-9
    )
    consistency.update(
        {
            "checked": True,
            "main_metrics_match": bool(main_metrics_match),
            "porosity_delta": porosity_delta,
            "packing_count_match": bool(packing_count_match),
            "density_delta_g_cm3": density_delta,
            "minimum_distance_delta_A": distance_delta,
        }
    )
    if not main_metrics_match:
        errors.append(
            "main_metrics.json porosity does not match independent final_phase.h5 porosity"
        )
    return consistency


def _validate_contract(contract: dict[str, Any], target_box: np.ndarray, errors: list[str]) -> None:
    if contract.get("format_version") != 1:
        errors.append("contract.json format_version must be 1")
    if contract.get("final_phase_file") != "final_phase.h5":
        errors.append("contract.json final_phase_file must be final_phase.h5")
    required = {
        "length_unit": "angstrom",
        "axis_order": "zyx",
        "phase_encoding": {"semiconductor": 0, "pore": 1},
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            errors.append(f"contract.json {key} does not match expected value")
    if contract.get("periodic_axes") != ["x", "y"]:
        errors.append("contract.json periodic_axes must be ['x', 'y']")
    contract_box = np.asarray(contract.get("target_box_A", []), dtype=float)
    if contract_box.shape != (3,) or not np.allclose(contract_box, target_box):
        errors.append("contract.json target_box_A does not match final_phase.h5")


def _read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"mandatory JSON could not be read: {path.name}: {exc}")
        return {}
    if not isinstance(data, dict):
        errors.append(f"mandatory JSON must contain an object: {path.name}")
        return {}
    return data


def _read_yaml(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        errors.append(f"mandatory YAML could not be read: {path.name}: {exc}")
        return {}
    if not isinstance(data, dict):
        errors.append(f"mandatory YAML must contain a mapping: {path.name}")
        return {}
    return data


def _read_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{index} could not be parsed as JSON: {exc}")
            continue
        if isinstance(record, dict):
            records.append(record)
        else:
            errors.append(f"{path.name}:{index} must contain a JSON object")
    return records


def _periodic_xy_component_count(mask: np.ndarray) -> int:
    labels, count = ndimage.label(mask, structure=_six_connected_structure())
    if count <= 1:
        return int(count)
    parent = np.arange(count + 1, dtype=int)
    for z_index in range(mask.shape[0]):
        for y_index in range(mask.shape[1]):
            _union_labels(labels[z_index, y_index, 0], labels[z_index, y_index, -1], parent)
        for x_index in range(mask.shape[2]):
            _union_labels(labels[z_index, 0, x_index], labels[z_index, -1, x_index], parent)
    roots = {_find(label, parent) for label in range(1, count + 1)}
    return len(roots)


def _six_connected_structure() -> np.ndarray:
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, 1, 1] = True
    structure[0, 1, 1] = True
    structure[2, 1, 1] = True
    structure[1, 0, 1] = True
    structure[1, 2, 1] = True
    structure[1, 1, 0] = True
    structure[1, 1, 2] = True
    return structure


def _union_labels(first: int, second: int, parent: np.ndarray) -> None:
    if first == 0 or second == 0:
        return
    root_first = _find(int(first), parent)
    root_second = _find(int(second), parent)
    if root_first != root_second:
        parent[root_second] = root_first


def _find(label: int, parent: np.ndarray) -> int:
    while parent[label] != label:
        parent[label] = parent[parent[label]]
        label = int(parent[label])
    return label


def _percolates_x(mask: np.ndarray) -> bool:
    domain = np.asarray(mask, dtype=bool)
    if domain.ndim != 3:
        raise ValueError("mask must have shape (z, y, x)")
    if not np.any(domain):
        return False

    ny = domain.shape[1]
    nx = domain.shape[2]
    tiled = np.tile(domain, (1, 3, 3))
    labels, label_count = ndimage.label(tiled, structure=_six_connected_structure())
    if label_count == 0:
        return False
    central_labels = np.unique(labels[:, ny : 2 * ny, nx : 2 * nx][domain])
    central_labels = central_labels[central_labels != 0]
    for label in central_labels:
        _z, _y, x = np.nonzero(labels == label)
        if x.size and int(x.min()) < nx and int(x.max()) >= 2 * nx:
            return True
    return False


def _minimum_cross_section_fraction(semiconductor: np.ndarray) -> float:
    if semiconductor.ndim != 3 or semiconductor.shape[2] == 0:
        return 0.0
    fractions = np.mean(semiconductor, axis=(0, 1))
    return float(np.min(fractions))


def _configured_minimum_skeleton_thickness_A(config: dict[str, Any]) -> float | None:
    matrix = config.get("matrix_constraints", {})
    if not isinstance(matrix, dict):
        return None
    value = _optional_float(matrix.get("minimum_skeleton_thickness_A"))
    return float(value) if np.isfinite(value) and value > 0.0 else None


def _matrix_mask_for_percolation(
    semiconductor: np.ndarray,
    spacing_A: float,
    minimum_skeleton_thickness_A: float | None,
) -> np.ndarray:
    mask = np.asarray(semiconductor, dtype=bool)
    if minimum_skeleton_thickness_A is None:
        return mask.copy()
    if not np.any(mask):
        return mask.copy()
    distances = ndimage.distance_transform_edt(mask, sampling=float(spacing_A))
    return distances >= 0.5 * float(minimum_skeleton_thickness_A)


def _independent_overlap_fraction(
    records: list[dict[str, Any]],
    channel_curves: dict[str, _ChannelCurveData],
    phase: _PhaseData,
    errors: list[str],
) -> float:
    pore_indices = np.flatnonzero(phase.mask.ravel())
    if pore_indices.size == 0 or len(records) < 2:
        return 0.0
    overlap_count = 0
    for start in range(0, pore_indices.size, _OVERLAP_CHUNK_SIZE):
        stop = min(start + _OVERLAP_CHUNK_SIZE, pore_indices.size)
        points = _voxel_centers_for_indices(phase, pore_indices[start:stop])
        coverage = np.zeros(points.shape[0], dtype=np.uint16)
        for record in records:
            try:
                occupied = (
                    _independent_periodic_unit_field(
                        record,
                        channel_curves,
                        points,
                        phase.target_box_A,
                    )
                    < 0.0
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(
                    "pore overlap fraction could not be independently reconstructed for "
                    f"unit {record.get('unit_id', '')}: {exc}"
                )
                return math.nan
            coverage += occupied.astype(np.uint16)
        overlap_count += int(np.count_nonzero(coverage > 1))
    return float(overlap_count / pore_indices.size)


def _independent_periodic_unit_field(
    record: dict[str, Any],
    channel_curves: dict[str, _ChannelCurveData],
    points_A: np.ndarray,
    target_box_A: np.ndarray,
) -> np.ndarray:
    kind = str(record.get("kind", "")).lower()
    geometry = record.get("realized_geometry", {})
    latent = record.get("latent_parameters", {})
    if not isinstance(geometry, dict) or not isinstance(latent, dict):
        raise TypeError("invalid realized_geometry or latent_parameters")
    unit_id = str(record.get("unit_id", ""))
    if kind == "compact":
        center = np.asarray(geometry["center_A"], dtype=float)
        radii = np.asarray(
            geometry.get("envelope_radii_A", geometry.get("radii_A", [])),
            dtype=float,
        )
        support = float(np.max(radii)) * (1.0 + float(latent.get("roughness", 0.0)))
        extent_min = center - support
        extent_max = center + support
    elif kind == "channel":
        curve = channel_curves.get(unit_id)
        if curve is None:
            raise ValueError("channel_curves entry is missing")
        cross_radius = _independent_channel_cross_radius_A(geometry, latent, curve)
        support = cross_radius * (1.0 + float(latent.get("roughness", 0.0)))
        extent_min = np.min(curve.centerline_A, axis=0) - support
        extent_max = np.max(curve.centerline_A, axis=0) + support
    else:
        raise ValueError(f"unsupported unit kind {kind!r}")

    x_shifts = _independent_periodic_shifts(
        points_A[:, 0],
        float(extent_min[0]),
        float(extent_max[0]),
        float(target_box_A[0]),
    )
    y_shifts = _independent_periodic_shifts(
        points_A[:, 1],
        float(extent_min[1]),
        float(extent_max[1]),
        float(target_box_A[1]),
    )
    result = np.full(points_A.shape[0], np.inf, dtype=float)
    for x_shift in x_shifts:
        for y_shift in y_shifts:
            shifted = points_A - np.array([x_shift, y_shift, 0.0], dtype=float)
            if kind == "compact":
                field = _independent_compact_field(record, shifted)
            else:
                field = _independent_channel_field(
                    record,
                    channel_curves[unit_id],
                    shifted,
                )
            result = _independent_smooth_min_pair(result, field)
    return result


def _independent_periodic_shifts(
    query_values: np.ndarray,
    unit_minimum_A: float,
    unit_maximum_A: float,
    box_length_A: float,
) -> np.ndarray:
    first = int(np.floor((float(np.min(query_values)) - unit_maximum_A) / box_length_A))
    last = int(np.ceil((float(np.max(query_values)) - unit_minimum_A) / box_length_A))
    return np.arange(first, last + 1, dtype=float) * box_length_A


def _independent_compact_field(
    record: dict[str, Any],
    points_A: np.ndarray,
) -> np.ndarray:
    geometry = record["realized_geometry"]
    latent = record["latent_parameters"]
    center = np.asarray(geometry["center_A"], dtype=float)
    radii = np.asarray(geometry.get("radii_A", []), dtype=float)
    quaternion = np.asarray(geometry.get("orientation_quaternion_xyzw", []), dtype=float)
    if center.shape != (3,) or radii.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("invalid compact center, radii, or orientation")
    local = Rotation.from_quat(quaternion).inv().apply(points_A - center)
    if record.get("shape_model") == "multilobe-v1":
        lobe_centers = np.asarray(geometry.get("lobe_centers_local_A", []), dtype=float)
        lobe_radii = np.asarray(geometry.get("lobe_radii_A", []), dtype=float)
        smooth = float(geometry.get("smooth_length_A", 0.0))
        if lobe_centers.ndim != 2 or lobe_radii.shape != lobe_centers.shape:
            raise ValueError("invalid multilobe arrays")
        fields = []
        for lobe_center, axes in zip(lobe_centers, lobe_radii, strict=True):
            implicit = np.sqrt(np.sum(((local - lobe_center) / axes) ** 2, axis=1))
            fields.append((implicit - 1.0) * float(np.min(axes)))
        base = -smooth * logsumexp(-np.vstack(fields) / smooth, axis=0)
    else:
        exponent = float(latent.get("superellipsoid_exponent", 2.0))
        if exponent == 2.0 and np.allclose(radii, radii[0]):
            base = np.linalg.norm(local, axis=1) - float(radii[0])
        else:
            radial_xy = np.sqrt((local[:, 0] / radii[0]) ** 2 + (local[:, 1] / radii[1]) ** 2)
            axial_z = np.abs(local[:, 2] / radii[2])
            implicit = (radial_xy**exponent + axial_z**exponent) ** (1.0 / exponent)
            base = (implicit - 1.0) * float(np.min(radii))
    roughness = float(latent.get("roughness", 0.0))
    return base - _independent_roughness_perturbation(
        local / np.maximum(radii, 1.0e-12),
        unit_id=str(record.get("unit_id", "")),
        roughness=roughness,
        length_scale_A=float(np.min(radii)),
    )


def _independent_channel_cross_radius_A(
    geometry: dict[str, Any],
    latent: dict[str, Any],
    curve: _ChannelCurveData,
) -> float:
    value = _optional_float(
        geometry.get(
            "equivalent_radius_A",
            latent.get("cross_radius_A", curve.equivalent_radius_A),
        )
    )
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("missing positive channel equivalent radius")
    return float(value)


def _independent_channel_field(
    record: dict[str, Any],
    curve: _ChannelCurveData,
    points_A: np.ndarray,
) -> np.ndarray:
    geometry = record["realized_geometry"]
    latent = record["latent_parameters"]
    centerline = np.asarray(curve.centerline_A, dtype=float)
    starts = centerline[:-1]
    ends = centerline[1:]
    segment_vectors = ends - starts
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    if np.any(segment_lengths <= 0.0):
        raise ValueError("channel centerline contains zero-length segments")
    tangents = segment_vectors / segment_lengths[:, np.newaxis]
    start_normals = tangents.copy()
    end_normals = tangents.copy()
    for index in range(1, tangents.shape[0]):
        start_normals[index] = _independent_safe_bisector(tangents[index - 1], tangents[index])
    for index in range(tangents.shape[0] - 1):
        end_normals[index] = _independent_safe_bisector(tangents[index], tangents[index + 1])
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    arc_length = float(cumulative[-1])
    cross_radius = _independent_channel_cross_radius_A(geometry, latent, curve)
    roughness = float(latent.get("roughness", 0.0))
    profile_s = np.asarray(geometry.get("radius_profile_s", []), dtype=float)
    profile_A = np.asarray(geometry.get("radius_profile_A", []), dtype=float)
    interpolator = (
        PchipInterpolator(profile_s, profile_A)
        if profile_s.size >= 2 and profile_A.shape == profile_s.shape
        else None
    )

    frames = [
        Rotation.align_vectors(
            tangent[np.newaxis, :],
            np.array([[1.0, 0.0, 0.0]]),
        )[0]
        for tangent in tangents
    ]
    result = np.full(points_A.shape[0], np.inf, dtype=float)
    for index, (start, end, tangent, segment_length, frame) in enumerate(
        zip(starts, ends, tangents, segment_lengths, frames, strict=True)
    ):
        offset = points_A - start
        local = frame.inv().apply(offset)
        axial = np.clip(offset @ tangent, 0.0, float(segment_length))
        normalized_s = (float(cumulative[index]) + axial) / arc_length
        local_radius = (
            np.asarray(interpolator(normalized_s), dtype=float)
            if interpolator is not None
            else np.full(points_A.shape[0], cross_radius, dtype=float)
        )
        closest = start + axial[:, np.newaxis] * tangent
        radial = np.linalg.norm(points_A - closest, axis=1) - local_radius
        start_signed = -(offset @ start_normals[index])
        end_signed = (points_A - end) @ end_normals[index]
        inside = (start_signed <= 0.0) & (end_signed <= 0.0)
        field = radial.copy()
        field[~inside] = np.maximum(
            radial[~inside],
            np.maximum(start_signed[~inside], end_signed[~inside]),
        )
        coordinates = np.column_stack(
            [normalized_s, local[:, 1] / cross_radius, local[:, 2] / cross_radius]
        )
        field -= _independent_roughness_perturbation(
            coordinates,
            unit_id=str(record.get("unit_id", "")),
            roughness=roughness,
            length_scale_A=cross_radius,
        )
        result = _independent_smooth_min_pair(result, field)

    first_local = frames[0].inv().apply(points_A - starts[0])
    first_radius = float(interpolator(0.0)) if interpolator is not None else cross_radius
    first = np.linalg.norm(first_local, axis=1) - first_radius
    first = np.where(first_local[:, 0] <= 0.0, first, np.inf)
    first -= _independent_roughness_perturbation(
        np.column_stack(
            [
                np.zeros(points_A.shape[0]),
                first_local[:, 1] / cross_radius,
                first_local[:, 2] / cross_radius,
            ]
        ),
        unit_id=str(record.get("unit_id", "")),
        roughness=roughness,
        length_scale_A=cross_radius,
    )
    result = _independent_smooth_min_pair(result, first)

    last_local = frames[-1].inv().apply(points_A - ends[-1])
    last_radius = float(interpolator(1.0)) if interpolator is not None else cross_radius
    last = np.linalg.norm(last_local, axis=1) - last_radius
    last = np.where(last_local[:, 0] >= 0.0, last, np.inf)
    last -= _independent_roughness_perturbation(
        np.column_stack(
            [
                np.ones(points_A.shape[0]),
                last_local[:, 1] / cross_radius,
                last_local[:, 2] / cross_radius,
            ]
        ),
        unit_id=str(record.get("unit_id", "")),
        roughness=roughness,
        length_scale_A=cross_radius,
    )
    return _independent_smooth_min_pair(result, last)


def _independent_safe_bisector(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    combined = np.asarray(first, dtype=float) + np.asarray(second, dtype=float)
    norm = float(np.linalg.norm(combined))
    return np.asarray(second, dtype=float) if norm <= 1.0e-12 else combined / norm


def _independent_roughness_perturbation(
    local_coordinates: np.ndarray,
    *,
    unit_id: str,
    roughness: float,
    length_scale_A: float,
) -> np.ndarray:
    if roughness == 0.0:
        return np.zeros(local_coordinates.shape[0], dtype=float)
    seed_bytes = blake2b(unit_id.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(seed_bytes, byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    amplitudes = rng.uniform(0.35, 1.0, _ROUGHNESS_MODE_COUNT)
    phases = rng.uniform(0.0, 2.0 * np.pi, _ROUGHNESS_MODE_COUNT)
    coordinate = np.sum(local_coordinates, axis=1)
    perturbation = np.zeros(local_coordinates.shape[0], dtype=float)
    for frequency, phase, amplitude in zip(
        np.arange(1, _ROUGHNESS_MODE_COUNT + 1, dtype=float),
        phases,
        amplitudes,
        strict=True,
    ):
        perturbation += amplitude * np.sin(2.0 * np.pi * frequency * coordinate + phase)
    perturbation /= float(_ROUGHNESS_MODE_COUNT)
    return float(roughness) * float(length_scale_A) * perturbation


def _independent_smooth_min_pair(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (
        -np.logaddexp(
            -_SMOOTH_UNION_SHARPNESS * first,
            -_SMOOTH_UNION_SHARPNESS * second,
        )
        / _SMOOTH_UNION_SHARPNESS
    )


def _anchor_from_record(record: dict[str, Any]) -> np.ndarray | None:
    geometry = record.get("realized_geometry", {})
    if not isinstance(geometry, dict):
        return None
    anchor = geometry.get("anchor_A", geometry.get("center_A"))
    if anchor is None:
        return None
    array = np.asarray(anchor, dtype=float)
    return array if array.shape == (3,) and np.all(np.isfinite(array)) else None


def _unit_volume_A3(
    record: dict[str, Any],
    channel_curves: dict[str, _ChannelCurveData],
    errors: list[str],
    warnings: list[str],
) -> float | None:
    geometry = record.get("realized_geometry", {})
    if not isinstance(geometry, dict):
        return None
    kind = str(record.get("kind", "")).lower()
    schema_version = _schema_version(record)
    if kind == "compact":
        if schema_version >= 2 or record.get("shape_model") == "multilobe-v1":
            return _v2_multilobe_volume_A3(record, geometry, errors)
        radii = np.asarray(geometry.get("radii_A", []), dtype=float)
        if radii.shape == (3,) and np.all(np.isfinite(radii)):
            volume = float(4.0 * math.pi * np.prod(radii) / 3.0)
            _check_target_volume(
                record,
                str(record.get("unit_id", "")),
                volume,
                errors,
            )
            return volume
    if kind == "channel":
        unit_id = str(record.get("unit_id", ""))
        latent = record.get("latent_parameters", {})
        curve = channel_curves.get(unit_id)
        if curve is None:
            errors.append(f"channel_curves.h5 is missing centerline samples for channel {unit_id}")
            return None
        if schema_version >= 2 or curve.schema_version >= 2:
            return _v2_channel_volume_A3(record, geometry, curve, errors)
        radius = _optional_float(
            geometry.get(
                "cross_radius_A",
                latent.get("cross_radius_A") if isinstance(latent, dict) else math.nan,
            )
        )
        if not np.isfinite(radius) or radius <= 0.0:
            errors.append(f"channel {unit_id} is missing a positive cross_radius_A")
            return None
        length = float(np.sum(np.linalg.norm(np.diff(curve.centerline_A, axis=0), axis=1)))
        volume = math.pi * radius * radius * length + 4.0 * math.pi * radius**3 / 3.0
        _check_target_volume(record, unit_id, volume, errors)
        return volume
    if kind:
        warnings.append(f"unit volume was not independently evaluable for {record.get('unit_id')}")
    return None


def _schema_version(record: dict[str, Any]) -> int:
    try:
        return int(record.get("schema_version", 1))
    except (TypeError, ValueError):
        return 1


def _v2_multilobe_volume_A3(
    record: dict[str, Any],
    geometry: dict[str, Any],
    errors: list[str],
) -> float | None:
    unit_id = str(record.get("unit_id", ""))
    if record.get("shape_model") != "multilobe-v1":
        errors.append(f"compact {unit_id} has unsupported v2 shape_model")
    if not isinstance(record.get("shape_seed"), int):
        errors.append(f"compact {unit_id} is missing integer shape_seed")

    centers = np.asarray(geometry.get("lobe_centers_local_A", []), dtype=float)
    radii = np.asarray(geometry.get("lobe_radii_A", []), dtype=float)
    envelope = np.asarray(
        geometry.get("envelope_radii_A", geometry.get("radii_A", [])),
        dtype=float,
    )
    smooth = _optional_float(geometry.get("smooth_length_A"))
    if (
        centers.ndim != 2
        or centers.shape[1:] != (3,)
        or radii.shape != centers.shape
        or centers.shape[0] == 0
        or not np.all(np.isfinite(centers))
        or not np.all(np.isfinite(radii))
        or np.any(radii <= 0.0)
    ):
        errors.append(f"compact {unit_id} has invalid multilobe center/radius arrays")
        return None
    lobe_count = int(centers.shape[0])
    if not 2 <= lobe_count <= 4:
        errors.append(f"compact {unit_id} must contain 2 to 4 lobes")
    if envelope.shape != (3,) or not np.all(np.isfinite(envelope)) or np.any(envelope <= 0.0):
        errors.append(f"compact {unit_id} has invalid envelope radii")
        return None
    if not np.isfinite(smooth) or smooth <= 0.0:
        errors.append(f"compact {unit_id} has invalid smooth_length_A")
        return None

    expected_envelope = _independent_multilobe_envelope_radii_A(
        centers,
        radii,
        smooth,
        envelope,
    )
    if not np.allclose(envelope, expected_envelope, rtol=0.01, atol=1.0e-8):
        errors.append(f"compact {unit_id} envelope radii do not match its lobes")
    record_radii = np.asarray(geometry.get("radii_A", envelope), dtype=float)
    if record_radii.shape != (3,) or not np.allclose(
        record_radii, envelope, rtol=1.0e-8, atol=1.0e-8
    ):
        errors.append(f"compact {unit_id} radii_A must equal the realized envelope")
    if not np.isclose(envelope[0], envelope[1], rtol=0.02):
        errors.append(f"compact {unit_id} transverse envelope axes are not equal")

    volume = _independent_multilobe_volume_A3(centers, radii, smooth, envelope)
    fill_fraction = volume / (4.0 * math.pi * float(np.prod(envelope)) / 3.0)
    if not 0.50 <= fill_fraction <= 0.85:
        errors.append(f"compact {unit_id} envelope fill fraction is outside 0.50-0.85")
    claimed_fill = _optional_float(geometry.get("envelope_fill_fraction"))
    if np.isfinite(claimed_fill) and not np.isclose(
        claimed_fill, fill_fraction, rtol=0.03, atol=0.01
    ):
        errors.append(f"compact {unit_id} envelope_fill_fraction is inconsistent")

    centroid = np.average(centers, axis=0, weights=np.prod(radii, axis=1))
    centroid_offset = float(np.linalg.norm(centroid))
    if centroid_offset > 1.0e-8 * float(np.max(envelope)):
        errors.append(f"compact {unit_id} volume-weighted centroid is not at the origin")
    claimed_centroid = _optional_float(geometry.get("centroid_offset_A"))
    if np.isfinite(claimed_centroid) and not np.isclose(
        claimed_centroid, centroid_offset, rtol=0.05, atol=1.0e-8
    ):
        errors.append(f"compact {unit_id} centroid_offset_A is inconsistent")

    connected = _multilobe_connected(centers, radii)
    if not connected:
        errors.append(f"compact {unit_id} lobe graph is disconnected")
    if (
        geometry.get("lobes_connected") is not None
        and bool(geometry.get("lobes_connected")) != connected
    ):
        errors.append(f"compact {unit_id} lobes_connected flag is inconsistent")

    latent = record.get("latent_parameters", {})
    if isinstance(latent, dict):
        target_eta = _optional_float(latent.get("eta"))
        if np.isfinite(target_eta) and target_eta > 0.0:
            realized_eta = float(envelope[2] / envelope[0])
            if not np.isclose(realized_eta, target_eta, rtol=0.05):
                errors.append(f"compact {unit_id} realized eta is inconsistent")
    _check_target_volume(record, unit_id, volume, errors)
    return volume


def _independent_multilobe_envelope_radii_A(
    centers_A: np.ndarray,
    radii_A: np.ndarray,
    smooth_length_A: float,
    envelope_hint_A: np.ndarray,
) -> np.ndarray:
    extents = np.zeros(3, dtype=float)
    maximum_radius = float(np.max(radii_A))
    for axis in range(3):
        for center, radii in zip(centers_A, radii_A, strict=True):
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
                    values = []
                    for lobe_center, lobe_radii in zip(
                        centers_A,
                        radii_A,
                        strict=True,
                    ):
                        normalized = math.sqrt(
                            float(np.sum(((point - lobe_center) / lobe_radii) ** 2))
                        )
                        values.append((normalized - 1.0) * float(np.min(lobe_radii)))
                    return float(
                        -smooth_length_A * logsumexp(-np.asarray(values) / smooth_length_A)
                    )

                upper = max(
                    maximum_radius + smooth_length_A,
                    float(envelope_hint_A[axis]),
                )
                while field(upper) < 0.0:
                    upper *= 2.0
                distance = brentq(field, 0.0, upper)
                extents[axis] = max(
                    extents[axis],
                    abs(float(boundary[axis] + sign * distance)),
                )
    return extents


def _check_target_volume(
    record: dict[str, Any],
    unit_id: str,
    realized_volume_A3: float,
    errors: list[str],
) -> None:
    latent = record.get("latent_parameters", {})
    if not isinstance(latent, dict):
        return
    target = _optional_float(latent.get("target_volume_A3"))
    if not np.isfinite(target):
        return
    if target <= 0.0:
        errors.append(f"unit {unit_id} target_volume_A3 must be positive")
        return
    if not np.isclose(realized_volume_A3, target, rtol=0.03, atol=1.0e-8):
        errors.append(
            f"unit {unit_id} target_volume_A3 differs from independently recomputed volume"
        )


def _independent_multilobe_volume_A3(
    centers_A: np.ndarray,
    radii_A: np.ndarray,
    smooth_length_A: float,
    envelope_A: np.ndarray,
) -> float:
    sampler = qmc.Sobol(d=3, scramble=False)
    samples = sampler.random_base2(15)
    points = (2.0 * samples - 1.0) * envelope_A
    fields = []
    for center, radii in zip(centers_A, radii_A, strict=True):
        normalized = np.sqrt(np.sum(((points - center) / radii) ** 2, axis=1))
        fields.append((normalized - 1.0) * float(np.min(radii)))
    values = np.vstack(fields)
    sdf = -smooth_length_A * logsumexp(-values / smooth_length_A, axis=0)
    return float(np.mean(sdf < 0.0) * 8.0 * np.prod(envelope_A))


def _multilobe_connected(centers_A: np.ndarray, radii_A: np.ndarray) -> bool:
    count = centers_A.shape[0]
    adjacency = [set() for _ in range(count)]
    for first in range(count):
        for second in range(first + 1, count):
            delta = centers_A[second] - centers_A[first]
            distance = float(np.linalg.norm(delta))
            if distance <= 1.0e-12:
                overlaps = True
            else:
                direction = delta / distance
                support_first = 1.0 / math.sqrt(float(np.sum((direction / radii_A[first]) ** 2)))
                support_second = 1.0 / math.sqrt(float(np.sum((direction / radii_A[second]) ** 2)))
                overlaps = distance <= support_first + support_second + 1.0e-10
            if overlaps:
                adjacency[first].add(second)
                adjacency[second].add(first)
    reached = {0}
    pending = [0]
    while pending:
        current = pending.pop()
        for neighbor in adjacency[current] - reached:
            reached.add(neighbor)
            pending.append(neighbor)
    return len(reached) == count


def _v2_channel_volume_A3(
    record: dict[str, Any],
    geometry: dict[str, Any],
    curve: _ChannelCurveData,
    errors: list[str],
) -> float | None:
    unit_id = str(record.get("unit_id", ""))
    shape_model = record.get("shape_model")
    if shape_model not in {"variable-radius-spline-v1", "variable-radius-polyline-v1"}:
        errors.append(f"channel {unit_id} has unsupported v2 shape_model")
    if not isinstance(record.get("shape_seed"), int):
        errors.append(f"channel {unit_id} is missing integer shape_seed")
    if curve.shape_model is not None and curve.shape_model != shape_model:
        errors.append(f"channel {unit_id} HDF5 shape_model is inconsistent")
    if curve.shape_seed is not None and curve.shape_seed != record.get("shape_seed"):
        errors.append(f"channel {unit_id} HDF5 shape_seed is inconsistent")
    if curve.radius_A is None:
        errors.append(f"channel {unit_id} v2 curve is missing radius_A")
        return None

    centerline = curve.centerline_A
    radius_samples = curve.radius_A
    segment_lengths = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    if np.any(segment_lengths <= 0.0):
        errors.append(f"channel {unit_id} centerline contains zero-length segments")
        return None
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    arc_length = float(cumulative[-1])
    end_distance = float(np.linalg.norm(centerline[-1] - centerline[0]))
    equivalent_radius = _optional_float(
        geometry.get(
            "equivalent_radius_A",
            record.get("latent_parameters", {}).get("cross_radius_A")
            if isinstance(record.get("latent_parameters"), dict)
            else math.nan,
        )
    )
    if not np.isfinite(equivalent_radius) or equivalent_radius <= 0.0:
        errors.append(f"channel {unit_id} is missing positive equivalent_radius_A")
        return None
    if curve.equivalent_radius_A is not None and not np.isclose(
        curve.equivalent_radius_A,
        equivalent_radius,
        rtol=1.0e-8,
        atol=1.0e-8,
    ):
        errors.append(f"channel {unit_id} HDF5 equivalent_radius_A is inconsistent")
    if end_distance <= 0.0:
        errors.append(f"channel {unit_id} end distance must be positive")
        return None

    eta = arc_length / (2.0 * equivalent_radius)
    tau = arc_length / end_distance
    _check_close_metric(
        unit_id,
        "arc_length_A",
        _optional_float(geometry.get("arc_length_A")),
        arc_length,
        errors,
    )
    _check_close_metric(
        unit_id,
        "end_distance_A",
        _optional_float(geometry.get("end_distance_A")),
        end_distance,
        errors,
    )
    _check_close_metric(
        unit_id,
        "eta",
        _optional_float(geometry.get("eta")),
        eta,
        errors,
        rtol=0.01,
    )
    _check_close_metric(
        unit_id,
        "tortuosity",
        _optional_float(geometry.get("tortuosity")),
        tau,
        errors,
        rtol=0.01,
    )

    profile_s = np.asarray(geometry.get("radius_profile_s", []), dtype=float)
    profile_A = np.asarray(geometry.get("radius_profile_A", []), dtype=float)
    profile_invalid = (
        profile_s.ndim != 1
        or profile_A.shape != profile_s.shape
        or profile_s.size < 2
        or not np.all(np.isfinite(profile_s))
        or not np.all(np.isfinite(profile_A))
        or np.any(profile_A <= 0.0)
        or not np.isclose(profile_s[0], 0.0)
        or not np.isclose(profile_s[-1], 1.0)
        or np.any(np.diff(profile_s) <= 0.0)
    )
    if profile_invalid:
        errors.append(f"channel {unit_id} has invalid radius profile nodes")
        return None
    if shape_model == "variable-radius-spline-v1" and profile_s.size != 7:
        errors.append(f"channel {unit_id} generated radius profile must have 7 nodes")
    interpolator = PchipInterpolator(profile_s, profile_A)
    expected_radii = np.asarray(
        interpolator(cumulative / arc_length),
        dtype=float,
    )
    if not np.allclose(radius_samples, expected_radii, rtol=1.0e-5, atol=1.0e-7):
        errors.append(f"channel {unit_id} HDF5 radius_A is inconsistent with its profile")

    dense_s = np.linspace(0.0, 1.0, 4097)
    dense_radii = np.asarray(interpolator(dense_s), dtype=float)
    radius_cv = float(np.std(dense_radii) / np.mean(dense_radii))
    minimum_ratio = float(np.min(dense_radii) / equivalent_radius)
    maximum_ratio = float(np.max(dense_radii) / equivalent_radius)
    if not 0.15 <= radius_cv <= 0.30:
        errors.append(f"channel {unit_id} radius CV is outside 0.15-0.30")
    if minimum_ratio < 0.60 - 1.0e-6 or maximum_ratio > 1.45 + 1.0e-6:
        errors.append(f"channel {unit_id} radius profile exceeds 0.60-1.45 bounds")
    if minimum_ratio >= 0.85 or maximum_ratio <= 1.15:
        errors.append(f"channel {unit_id} lacks a required neck or bulge")
    claimed_cv = _optional_float(geometry.get("radius_cv"))
    if np.isfinite(claimed_cv) and not np.isclose(claimed_cv, radius_cv, rtol=0.05, atol=0.01):
        errors.append(f"channel {unit_id} radius_cv is inconsistent")
    claimed_minmax = _optional_float(geometry.get("minimum_to_maximum_radius_ratio"))
    realized_minmax = float(np.min(dense_radii) / np.max(dense_radii))
    if np.isfinite(claimed_minmax) and not np.isclose(
        claimed_minmax, realized_minmax, rtol=0.03, atol=0.01
    ):
        errors.append(f"channel {unit_id} minimum_to_maximum_radius_ratio is inconsistent")

    controls = np.asarray(geometry.get("control_points_unwrapped_A", []), dtype=float)
    if (
        controls.ndim != 2
        or controls.shape[1:] != (3,)
        or controls.shape[0] < 2
        or not np.all(np.isfinite(controls))
    ):
        errors.append(f"channel {unit_id} has invalid control points")
    else:
        if shape_model == "variable-radius-spline-v1" and controls.shape[0] != 7:
            errors.append(f"channel {unit_id} generated centerline must have 7 control points")
        bend_count = _independent_bend_count(controls)
        sampled_controls = _independent_control_spline(controls, 1025)
        nonplanarity = _independent_nonplanarity(sampled_controls)
        claimed_bends = geometry.get("bend_count")
        if claimed_bends is not None and int(claimed_bends) != bend_count:
            errors.append(f"channel {unit_id} bend_count is inconsistent")
        claimed_nonplanarity = _optional_float(geometry.get("nonplanarity"))
        if np.isfinite(claimed_nonplanarity) and not np.isclose(
            claimed_nonplanarity, nonplanarity, rtol=0.05, atol=1.0e-5
        ):
            errors.append(f"channel {unit_id} nonplanarity is inconsistent")
        if tau > 1.05 and (bend_count < 2 or nonplanarity <= 1.0e-3):
            errors.append(f"channel {unit_id} must be multibend and nonplanar when tau > 1.05")

    clearance = _independent_channel_self_clearance_A(
        centerline,
        radius_samples,
    )
    if clearance < -1.0e-8:
        errors.append(f"channel {unit_id} contains a non-adjacent tube self-intersection")
    claimed_clearance = _optional_float(geometry.get("minimum_self_clearance_A"))
    if np.isfinite(claimed_clearance) and claimed_clearance < -1.0e-8:
        errors.append(f"channel {unit_id} minimum_self_clearance_A is invalid")

    body = math.pi * arc_length * float(np.trapezoid(dense_radii**2, dense_s))
    caps = 2.0 * math.pi * (float(profile_A[0]) ** 3 + float(profile_A[-1]) ** 3) / 3.0
    volume = body + caps
    _check_target_volume(record, unit_id, volume, errors)
    return volume


def _check_close_metric(
    unit_id: str,
    name: str,
    claimed: float,
    realized: float,
    errors: list[str],
    *,
    rtol: float = 0.01,
) -> None:
    if not np.isfinite(claimed) or not np.isclose(claimed, realized, rtol=rtol, atol=1.0e-8):
        errors.append(f"channel {unit_id} {name} is inconsistent with exported geometry")


def _independent_bend_count(control_points_A: np.ndarray) -> int:
    vectors = np.diff(control_points_A, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 0.0):
        return 0
    directions = vectors / lengths[:, np.newaxis]
    angles = np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1.0, 1.0))
    return int(np.count_nonzero(angles > 0.08))


def _independent_control_spline(
    control_points_A: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    chord_lengths = np.linalg.norm(np.diff(control_points_A, axis=0), axis=1)
    parameters = np.concatenate(([0.0], np.cumsum(chord_lengths)))
    parameters /= float(parameters[-1])
    target = np.linspace(0.0, 1.0, sample_count)
    if control_points_A.shape[0] == 2:
        return (1.0 - target[:, np.newaxis]) * control_points_A[0] + target[
            :, np.newaxis
        ] * control_points_A[1]
    return np.column_stack(
        [CubicSpline(parameters, control_points_A[:, axis])(target) for axis in range(3)]
    )


def _independent_nonplanarity(centerline_A: np.ndarray) -> float:
    centered = centerline_A - np.mean(centerline_A, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    return float(singular[-1] / max(float(singular[0]), 1.0e-12))


def _independent_channel_self_clearance_A(
    centerline_A: np.ndarray,
    radius_A: np.ndarray,
) -> float:
    if centerline_A.shape[0] > 129:
        indices = np.unique(np.linspace(0, centerline_A.shape[0] - 1, 129).round().astype(int))
        centerline_A = centerline_A[indices]
        radius_A = radius_A[indices]
    segment_lengths = np.linalg.norm(np.diff(centerline_A, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    minimum = math.inf
    for first in range(centerline_A.shape[0] - 1):
        for second in range(first + 2, centerline_A.shape[0] - 1):
            local_support = 3.0 * max(
                radius_A[first],
                radius_A[first + 1],
                radius_A[second],
                radius_A[second + 1],
            )
            if cumulative[second] - cumulative[first + 1] <= local_support:
                continue
            distance = _independent_segment_distance_A(
                centerline_A[first],
                centerline_A[first + 1],
                centerline_A[second],
                centerline_A[second + 1],
            )
            clearance = distance - max(radius_A[first], radius_A[first + 1])
            clearance -= max(radius_A[second], radius_A[second + 1])
            minimum = min(minimum, float(clearance))
    return float(minimum if np.isfinite(minimum) else 0.0)


def _independent_segment_distance_A(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    first_vector = first_end - first_start
    second_vector = second_end - second_start
    offset = first_start - second_start
    aa = float(np.dot(first_vector, first_vector))
    bb = float(np.dot(first_vector, second_vector))
    cc = float(np.dot(second_vector, second_vector))
    dd = float(np.dot(first_vector, offset))
    ee = float(np.dot(second_vector, offset))
    denominator = aa * cc - bb * bb
    if denominator <= 1.0e-15:
        first_parameter = 0.0
        second_parameter = np.clip(ee / max(cc, 1.0e-15), 0.0, 1.0)
    else:
        first_parameter = np.clip(
            (bb * ee - cc * dd) / denominator,
            0.0,
            1.0,
        )
        second_parameter = np.clip(
            (aa * ee - bb * dd) / denominator,
            0.0,
            1.0,
        )
    first_parameter = np.clip(
        (bb * second_parameter - dd) / max(aa, 1.0e-15),
        0.0,
        1.0,
    )
    second_parameter = np.clip(
        (bb * first_parameter + ee) / max(cc, 1.0e-15),
        0.0,
        1.0,
    )
    first_point = first_start + first_parameter * first_vector
    second_point = second_start + second_parameter * second_vector
    return float(np.linalg.norm(first_point - second_point))


def _pair_distances_periodic_xy(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    distances: list[float] = []
    for index in range(points.shape[0]):
        delta = points[index + 1 :] - points[index]
        if delta.size == 0:
            continue
        delta[:, 0] -= box[0] * np.rint(delta[:, 0] / box[0])
        delta[:, 1] -= box[1] * np.rint(delta[:, 1] / box[1])
        distances.extend(float(value) for value in np.linalg.norm(delta, axis=1))
    return np.asarray(distances, dtype=float)


def _cif_atom_count(path: Path, errors: list[str]) -> int:
    if not path.is_file():
        errors.append("mandatory placed_structure.cif is missing")
        return 0
    try:
        return sum(
            1
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("HETATM ")
        )
    except OSError as exc:
        errors.append(f"mandatory placed_structure.cif could not be read: {exc}")
        return 0


def _instances_csv_count(path: Path, errors: list[str]) -> int:
    if not path.is_file():
        errors.append("mandatory molecules/instances.csv is missing")
        return 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return sum(1 for _row in csv.DictReader(handle))
    except OSError as exc:
        errors.append(f"mandatory molecules/instances.csv could not be read: {exc}")
        return 0


def _transform_count(group: h5py.Group | None, errors: list[str]) -> int:
    if group is None:
        errors.append("mandatory placed_atoms.h5 instance_transforms group is missing")
        return 0
    try:
        translations = np.asarray(group["translations_A"], dtype=float)
        quaternions = np.asarray(group["quaternions_xyzw"], dtype=float)
    except (KeyError, ValueError) as exc:
        errors.append(f"mandatory placed_atoms.h5 instance transform datasets are invalid: {exc}")
        return 0
    if translations.ndim != 2 or translations.shape[1] != 3:
        errors.append("placed_atoms.h5 translations_A must have shape (n, 3)")
        return 0
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        errors.append("placed_atoms.h5 quaternions_xyzw must have shape (n, 4)")
        return 0
    if translations.shape[0] != quaternions.shape[0]:
        errors.append("placed_atoms.h5 instance transform row counts do not match")
    return int(translations.shape[0])


def _independent_molecule_count(
    *,
    atom_positions: np.ndarray,
    atom_instance_index: np.ndarray,
    atoms_per_instance: int,
    transform_count: int,
    csv_count: int,
    errors: list[str],
) -> tuple[int, dict[str, int]]:
    if atom_positions.ndim != 2 or atom_positions.shape[1] != 3:
        errors.append("placed_atoms.h5 atom_positions_A must have shape (n, 3)")
        atom_count = 0
    else:
        atom_count = int(atom_positions.shape[0])
    if atom_instance_index.ndim != 1 or atom_instance_index.shape[0] != atom_count:
        errors.append("placed_atoms.h5 atom_instance_index must have one row per atom")
        atom_index_count = 0
    elif atom_count == 0:
        atom_index_count = 0
    else:
        unique_indices = np.unique(atom_instance_index)
        atom_index_count = int(unique_indices.size)
        if not np.array_equal(unique_indices, np.arange(atom_index_count)):
            errors.append("placed_atoms.h5 atom_instance_index values must be contiguous from zero")
    if atoms_per_instance < 0:
        errors.append("placed_atoms.h5 template atom count cannot be negative")
        atom_shape_count = 0
    elif atoms_per_instance == 0:
        atom_shape_count = 0 if atom_count == 0 else -1
    elif atom_count % atoms_per_instance == 0:
        atom_shape_count = atom_count // atoms_per_instance
    else:
        atom_shape_count = -1
    if atom_shape_count < 0:
        errors.append("placed_atoms.h5 atom count is not divisible by template atom count")
        atom_shape_count = 0
    sources = {
        "atom_instance_index": atom_index_count,
        "atom_template_shape": int(atom_shape_count),
        "instance_transforms": int(transform_count),
        "instances_csv": int(csv_count),
    }
    if len(set(sources.values())) != 1:
        errors.append("independent molecule count sources disagree: " + json.dumps(sources))
    return int(atom_index_count), sources


def _minimum_interinstance_distance(
    atom_positions: np.ndarray,
    instance_indices: np.ndarray,
    target_box: np.ndarray,
) -> float:
    if atom_positions.shape[0] < 2:
        return math.inf
    best = math.inf
    for index in range(atom_positions.shape[0]):
        other = instance_indices[index + 1 :] != instance_indices[index]
        if not np.any(other):
            continue
        delta = atom_positions[index + 1 :][other] - atom_positions[index]
        if target_box.shape == (3,) and np.all(target_box[:2] > 0.0):
            delta[:, 0] -= target_box[0] * np.rint(delta[:, 0] / target_box[0])
            delta[:, 1] -= target_box[1] * np.rint(delta[:, 1] / target_box[1])
        best = min(best, float(np.min(np.linalg.norm(delta, axis=1))))
    return best


def _rigid_template_rmsd(
    atom_positions: np.ndarray,
    template_positions: np.ndarray,
    count: int,
    atoms_per_instance: int,
) -> float:
    if count == 0 or atoms_per_instance == 0:
        return 0.0
    values = []
    for index in range(count):
        start = index * atoms_per_instance
        stop = start + atoms_per_instance
        values.append(_kabsch_rmsd(template_positions, atom_positions[start:stop]))
    return float(max(values)) if values else 0.0


def _kabsch_rmsd(reference: np.ndarray, observed: np.ndarray) -> float:
    if reference.shape != observed.shape:
        return math.inf
    if reference.shape[0] <= 1:
        return 0.0
    ref = reference - np.mean(reference, axis=0)
    obs = observed - np.mean(observed, axis=0)
    covariance = ref.T @ obs
    u_matrix, _singular, vt_matrix = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt_matrix.T @ u_matrix.T))
    rotation = vt_matrix.T @ np.diag([1.0, 1.0, sign]) @ u_matrix.T
    aligned = ref @ rotation.T
    return float(np.sqrt(np.mean(np.sum((aligned - obs) ** 2, axis=1))))


def _box_from_config(config: dict[str, Any]) -> np.ndarray:
    target = config.get("film", {}).get("target_box_A", {})
    if not isinstance(target, dict):
        return np.asarray([], dtype=float)
    return np.asarray([target.get("x"), target.get("y"), target.get("z")], dtype=float)


def _nested_float(data: dict[str, Any], keys: list[str]) -> float:
    value: Any = data
    for key in keys:
        value = value.get(key, {}) if isinstance(value, dict) else {}
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _nested_int(data: dict[str, Any], keys: list[str]) -> int | None:
    value: Any = data
    for key in keys:
        value = value.get(key, {}) if isinstance(value, dict) else {}
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _finite_delta(first: float, second: float) -> float:
    if math.isinf(first) and math.isinf(second):
        return 0.0
    if not np.isfinite(first) or not np.isfinite(second):
        return math.inf
    return abs(float(first) - float(second))


def _status(errors: list[str], warnings: list[str], missing: list[str]) -> str:
    if missing or any("mandatory" in error for error in errors):
        return "NOT_EVALUABLE"
    if errors:
        return "FAIL"
    if warnings:
        return "WARNING"
    return "PASS"


def _write_reports(qa: Path, report: ValidationReport) -> None:
    try:
        qa.mkdir(parents=True, exist_ok=True)
        (qa / "independent-validation.json").write_text(
            json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (qa / "independent-validation-report.md").write_text(
            _markdown_report(report),
            encoding="utf-8",
        )
    except OSError:
        return


def _markdown_report(report: ValidationReport) -> str:
    lines = [
        "# Independent validation report",
        "",
        f"Status: {report.status}",
        "",
        "## Errors",
    ]
    lines.extend(f"- {error}" for error in report.errors) if report.errors else lines.append(
        "- None"
    )
    lines.extend(["", "## Warnings"])
    lines.extend(
        f"- {warning}" for warning in report.warnings
    ) if report.warnings else lines.append("- None")
    lines.extend(["", "## Independent metrics", ""])
    lines.append("```json")
    lines.append(json.dumps(report.independent_metrics, indent=2, sort_keys=True))
    lines.append("```")
    lines.extend(["", "## Report consistency", ""])
    lines.append("```json")
    lines.append(json.dumps(report.report_consistency, indent=2, sort_keys=True))
    lines.append("```")
    lines.extend(["", "## Target compliance", ""])
    lines.append("```json")
    lines.append(json.dumps(report.target_compliance, indent=2, sort_keys=True))
    lines.append("```")
    return "\n".join(lines) + "\n"


def _json_ready(value: Any) -> Any:
    if isinstance(value, ValidationReport):
        return {
            "status": value.status,
            "independent_metrics": _json_ready(value.independent_metrics),
            "report_consistency": _json_ready(value.report_consistency),
            "target_compliance": _json_ready(value.target_compliance),
            "warnings": list(value.warnings),
            "errors": list(value.errors),
        }
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value


def _text_attr(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
