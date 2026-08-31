from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

TABULATED_FIELDS = frozenset(
    {
        "table",
        "tabulated",
        "pdf",
        "cdf",
        "values",
        "probabilities",
        "histogram",
        "bins",
    }
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParallelSourceSpec(StrictModel):
    enabled: str = "config"
    strategy: str = "config"
    max_workers: str = "config"


class ParallelSpec(StrictModel):
    enabled: bool = True
    strategy: Literal["auto", "seeds", "candidates", "serial"] = "auto"
    max_workers: int | None = Field(default=None, gt=0)
    cpu_fraction: float = Field(default=0.80, gt=0, le=1)
    memory_fraction: float = Field(default=0.75, gt=0, le=1)
    worker_threads: Literal[1] = 1
    start_method: Literal["spawn"] = "spawn"
    sources: ParallelSourceSpec = Field(default_factory=ParallelSourceSpec)


class TaskSpec(StrictModel):
    name: str
    random_seed: int


class Box3D(StrictModel):
    x: float = Field(gt=0)
    y: float = Field(gt=0)
    z: float = Field(gt=0)

    @property
    def volume_A3(self) -> float:
        return self.x * self.y * self.z


class ZPaddingSpec(StrictModel):
    lower: float = Field(ge=0)
    upper: float = Field(ge=0)


class FilmSpec(StrictModel):
    target_box_A: Box3D
    packing_box_A: Box3D | None = None
    z_padding_A: ZPaddingSpec | None = None

    @model_validator(mode="after")
    def validate_and_normalize_packing(self) -> Self:
        has_packing = self.packing_box_A is not None
        has_padding = self.z_padding_A is not None
        if has_packing and has_padding:
            raise ValueError("provide at most one of packing_box_A or z_padding_A")

        if self.z_padding_A is not None:
            self.packing_box_A = Box3D(
                x=self.target_box_A.x,
                y=self.target_box_A.y,
                z=self.target_box_A.z + self.z_padding_A.lower + self.z_padding_A.upper,
            )
        elif self.packing_box_A is None:
            self.packing_box_A = self.target_box_A.model_copy()

        if self.packing_box_A.x != self.target_box_A.x:
            raise ValueError("packing x must equal target x")
        if self.packing_box_A.y != self.target_box_A.y:
            raise ValueError("packing y must equal target y")
        if self.packing_box_A.z < self.target_box_A.z:
            raise ValueError("packing z must be at least target z")
        return self

    @property
    def target_origin_in_packing_A(self) -> np.ndarray:
        if self.packing_box_A is None:
            return np.zeros(3, dtype=float)
        lower_padding = (
            (self.packing_box_A.z - self.target_box_A.z) / 2.0
            if self.z_padding_A is None
            else self.z_padding_A.lower
        )
        return np.array([0.0, 0.0, lower_padding], dtype=float)

    @property
    def target_volume_A3(self) -> float:
        return self.target_box_A.volume_A3


class MixtureComponent(StrictModel):
    weight: float = Field(ge=0)
    family: str
    value: float | None = None
    components: tuple[MixtureComponent, ...] | None = None
    alpha: float | None = None
    beta: float | None = None
    mean: float | None = None
    mu: float | None = None
    sigma: float | None = None
    s: float | None = None
    scale: float | None = None
    loc: float | None = None
    shape: float | None = None
    k: float | None = None
    theta: float | None = None
    lower: float | None = None
    upper: float | None = None
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_tabulated_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            illegal = TABULATED_FIELDS.intersection(data)
            if data.get("family") == "tabulated" or illegal:
                raise ValueError("tabulated distribution fields are not supported")
        return data

    @model_validator(mode="after")
    def validate_mixture(self) -> Self:
        if self.family == "mixture":
            if not self.components:
                raise ValueError("mixture distribution requires at least one component")
            weight_sum = sum(component.weight for component in self.components)
            if abs(weight_sum - 1.0) > 1e-9:
                raise ValueError("mixture weights must sum to 1")
        return self

    def as_distribution_dict(self) -> dict[str, Any]:
        data = self.model_dump(exclude_none=True)
        data.pop("weight", None)
        return data


class DistributionSpec(StrictModel):
    family: str
    value: float | None = None
    components: tuple[MixtureComponent, ...] | None = None
    alpha: float | None = None
    beta: float | None = None
    mean: float | None = None
    mu: float | None = None
    sigma: float | None = None
    s: float | None = None
    scale: float | None = None
    loc: float | None = None
    shape: float | None = None
    k: float | None = None
    theta: float | None = None
    lower: float | None = None
    upper: float | None = None
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_tabulated_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            illegal = TABULATED_FIELDS.intersection(data)
            if data.get("family") == "tabulated" or illegal:
                raise ValueError("tabulated distribution fields are not supported")
        return data

    @model_validator(mode="after")
    def validate_distribution(self) -> Self:
        if self.family == "mixture":
            if not self.components:
                raise ValueError("mixture distribution requires at least one component")
            weight_sum = sum(component.weight for component in self.components)
            if abs(weight_sum - 1.0) > 1e-9:
                raise ValueError("mixture weights must sum to 1")
        return self

    @staticmethod
    def constant(value: float) -> DistributionSpec:
        return DistributionSpec(family="constant", value=value)


def _distribution_lower_support(spec: DistributionSpec | MixtureComponent) -> float:
    family = spec.family
    if family == "constant":
        if spec.value is None:
            raise ValueError("constant distribution requires value")
        return spec.value
    if family == "mixture":
        if not spec.components:
            raise ValueError("mixture distribution requires at least one component")
        return min(_distribution_lower_support(component) for component in spec.components)
    if family in {"lognormal", "gamma", "weibull", "weibull_min"}:
        return 0.0 if spec.loc is None else spec.loc
    if family in {"truncated_normal", "truncnorm"}:
        if spec.lower is None:
            raise ValueError("truncated normal distribution requires lower support")
        return spec.lower
    if family == "beta":
        return 0.0 if spec.lower is None else spec.lower
    if spec.minimum is not None:
        return spec.minimum
    if spec.lower is not None:
        return spec.lower
    raise ValueError(f"unsupported distribution family: {family}")


def _distribution_upper_support(spec: DistributionSpec | MixtureComponent) -> float:
    family = spec.family
    if family == "constant":
        if spec.value is None:
            raise ValueError("constant distribution requires value")
        return spec.value
    if family == "mixture":
        if not spec.components:
            raise ValueError("mixture distribution requires at least one component")
        return max(_distribution_upper_support(component) for component in spec.components)
    if family in {"truncated_normal", "truncnorm"}:
        if spec.upper is None:
            raise ValueError("truncated normal distribution requires upper support")
        return spec.upper
    if family == "beta":
        if spec.upper is not None:
            return spec.upper
        if spec.maximum is not None:
            return spec.maximum
        lower = 0.0 if spec.lower is None else spec.lower
        return lower + (1.0 if spec.scale is None else spec.scale)
    return float("inf")


def _ensure_support_starts_at_one(
    spec: DistributionSpec | None,
    field_name: str,
) -> None:
    if spec is None:
        return
    lower = _distribution_lower_support(spec)
    if lower < 1.0:
        raise ValueError(f"{field_name} support must start at 1")


class RDFComponent(StrictModel):
    kind: Literal["peak", "dip", "exclusion", "oscillation"] = "peak"
    amplitude: float = Field(ge=0, default=1.0, allow_inf_nan=False)
    center_xi: float = Field(ge=0, allow_inf_nan=False)
    width_xi: float = Field(gt=0, allow_inf_nan=False)


class CenterDistributionSpec(StrictModel):
    mode: str
    lattice: str | None = None
    position_jitter: float = Field(ge=0, default=0.0)
    rdf: tuple[RDFComponent, ...] = ()

    @field_validator("lattice")
    @classmethod
    def validate_lattice(cls, value: str | None) -> str | None:
        if value is not None and value not in {"simple_cubic", "bcc", "fcc"}:
            raise ValueError("lattice must be simple_cubic, bcc, or fcc")
        return value


class GlobalPoreSpec(StrictModel):
    seed_number_density_A3: float = Field(gt=0)
    target_porosity: float = Field(gt=0, lt=1)
    channel_fraction_by_count: float = Field(ge=0, le=1)
    channel_to_compact_mean_volume_ratio: float = Field(gt=0, default=1.0)


class CompactPoreSpec(StrictModel):
    relative_volume: DistributionSpec
    aspect_ratio: DistributionSpec
    roughness: DistributionSpec

    @model_validator(mode="after")
    def validate_eta_support(self) -> Self:
        _ensure_support_starts_at_one(self.aspect_ratio, "compact eta")
        return self


class ChannelPoreSpec(StrictModel):
    relative_volume: DistributionSpec = Field(
        default_factory=lambda: DistributionSpec.constant(1.0)
    )
    aspect_ratio: DistributionSpec | None = None
    eta: DistributionSpec | None = None
    tortuosity: DistributionSpec | None = None
    tau: DistributionSpec | None = None
    roughness: DistributionSpec = Field(default_factory=lambda: DistributionSpec.constant(0.0))

    @model_validator(mode="after")
    def validate_eta_and_tau_support(self) -> Self:
        eta_spec = self.eta if self.eta is not None else self.aspect_ratio
        tau_spec = self.tau if self.tau is not None else self.tortuosity
        _ensure_support_starts_at_one(eta_spec, "channel eta")
        _ensure_support_starts_at_one(tau_spec, "channel tau")
        return self


class XYDistanceComponent(StrictModel):
    kind: Literal["peak", "dip", "exclusion", "oscillation"] = "peak"
    amplitude: float = Field(ge=0, default=1.0, allow_inf_nan=False)
    center_A: float = Field(ge=0, allow_inf_nan=False)
    width_A: float = Field(gt=0, allow_inf_nan=False)


class XYCenterDistanceTarget(StrictModel):
    components: tuple[XYDistanceComponent, ...] = ()


class PositionQuantityTargets(StrictModel):
    center_distance_xy: XYCenterDistanceTarget | None = None


class PairedOrientationComponent(StrictModel):
    weight: float = Field(ge=0)
    theta_xz_deg: DistributionSpec
    theta_xy_deg: DistributionSpec

    @model_validator(mode="after")
    def validate_beta_angles(self) -> Self:
        for name, distribution in (
            ("theta_xz_deg", self.theta_xz_deg),
            ("theta_xy_deg", self.theta_xy_deg),
        ):
            if distribution.family != "beta":
                raise ValueError("paired orientation components must use Beta distributions")
            lower = _distribution_lower_support(distribution)
            upper = _distribution_upper_support(distribution)
            if lower < 0.0 or upper > 90.0 or upper <= lower:
                raise ValueError(f"{name} support must lie within 0 to 90 degrees")
        return self


class PairedOrientationSpec(StrictModel):
    model: Literal["paired_projected_planes"] = "paired_projected_planes"
    components: tuple[PairedOrientationComponent, ...]

    @model_validator(mode="after")
    def validate_components(self) -> Self:
        if not self.components:
            raise ValueError("paired orientation requires at least one component")
        if abs(sum(component.weight for component in self.components) - 1.0) > 1e-9:
            raise ValueError("paired orientation component weights must sum to 1")
        return self


class ShapeTargets(StrictModel):
    equivalent_diameter_A: DistributionSpec | None = None
    orientation: PairedOrientationSpec | None = None
    compact_aspect_ratio: DistributionSpec | None = None
    channel_aspect_ratio: DistributionSpec | None = None
    channel_tortuosity: DistributionSpec | None = None
    curvature_fluctuation: DistributionSpec | None = None

    @model_validator(mode="after")
    def validate_shape_support(self) -> Self:
        if (
            self.equivalent_diameter_A is not None
            and _distribution_lower_support(self.equivalent_diameter_A) <= 0.0
        ):
            raise ValueError("equivalent diameter support must be positive")
        if (
            self.curvature_fluctuation is not None
            and _distribution_lower_support(self.curvature_fluctuation) < 0.0
        ):
            raise ValueError("curvature fluctuation support must be nonnegative")
        _ensure_support_starts_at_one(self.compact_aspect_ratio, "compact eta")
        _ensure_support_starts_at_one(self.channel_aspect_ratio, "channel eta")
        _ensure_support_starts_at_one(self.channel_tortuosity, "channel tau")
        return self


class ProportionTargets(StrictModel):
    porosity: float = Field(gt=0, lt=1)


class FormalTargets(StrictModel):
    position_quantity: PositionQuantityTargets
    shape: ShapeTargets
    proportion: ProportionTargets


class GenerationControlSpec(StrictModel):
    seed_number_density_A3: float = Field(gt=0)
    channel_fraction_by_count: float = Field(ge=0, le=1, default=1.0)
    channel_to_compact_mean_volume_ratio: float = Field(gt=0, default=1.0)
    compact_relative_volume: DistributionSpec = Field(
        default_factory=lambda: DistributionSpec.constant(1.0)
    )
    channel_relative_volume: DistributionSpec = Field(
        default_factory=lambda: DistributionSpec.constant(1.0)
    )
    compact_roughness: DistributionSpec = Field(
        default_factory=lambda: DistributionSpec.constant(0.0)
    )
    channel_roughness: DistributionSpec = Field(
        default_factory=lambda: DistributionSpec.constant(0.0)
    )


class MeasurementSpec(StrictModel):
    z_slice_spacing_A: float = Field(gt=0, default=1.0)
    center_min_separation_A: float = Field(gt=0, default=2.0)
    center_tracking_max_displacement_A: float = Field(gt=0, default=4.0)
    center_distance_bin_width_A: float = Field(gt=0, default=1.0)
    center_distance_max_A: float | None = Field(default=None, gt=0)
    center_distance_reference_samples: int = Field(gt=0, default=4096)
    centerline_sample_spacing_A: float = Field(gt=0, default=2.0)
    cross_section_spacing_A: float = Field(gt=0, default=2.0)
    boundary_resample_spacing_A: float = Field(gt=0, default=0.5)
    curvature_smoothing_length_A: float = Field(ge=0, default=1.0)
    branch_exclusion_length_A: float = Field(ge=0, default=2.0)
    surface_exclusion_length_A: float = Field(ge=0, default=2.0)
    orientation_projection_min_fraction: float = Field(gt=0, lt=1, default=0.05)


class MatrixConstraintSpec(StrictModel):
    enabled: bool = True
    require_x_percolation: bool = True
    minimum_cross_section_fraction: float = Field(ge=0, le=1, default=0.0)
    maximum_overlap_fraction: float = Field(ge=0, le=1, default=0.0)
    minimum_skeleton_thickness_A: float | None = Field(default=None, gt=0)


class PoreMaterialSpec(StrictModel):
    pdb: Path
    target_density_g_cm3: float | None = Field(default=None, gt=0)
    molecule_count: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_material_amount(self) -> Self:
        has_density = self.target_density_g_cm3 is not None
        has_count = self.molecule_count is not None
        if has_density == has_count:
            raise ValueError("provide exactly one of target_density_g_cm3 or molecule_count")
        return self


class AuditSpec(StrictModel):
    enabled: bool = True
    candidate_count_per_round: int = Field(gt=0, default=1)
    maximum_rounds: int = Field(gt=0, default=1)
    coarse_spacing_A: float = Field(gt=0, default=2.0)
    fine_spacing_A: float = Field(gt=0, default=1.0)
    orientation_aspect_ratio_tolerance: float = Field(ge=0, default=1.0e-6)
    interface_mixing_layer_fraction: float = Field(ge=0, le=1, default=0.0)
    available_memory_cap_bytes: int | None = Field(default=None, gt=0)


class OutputSpec(StrictModel):
    root: Path = Path("runs")
    write_plots: bool = True


class OptimizationSpec(StrictModel):
    seed_panel: tuple[int, ...] | None = None

    @field_validator("seed_panel")
    @classmethod
    def validate_seed_panel(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if value is None:
            return value
        if not value:
            raise ValueError("OptimizationSpec.seed_panel must be non-empty")
        if len(set(value)) != len(value):
            raise ValueError("OptimizationSpec.seed_panel must contain unique integers")
        return value


class OrientationSpec(StrictModel):
    distribution: DistributionSpec = Field(
        default_factory=lambda: DistributionSpec(family="beta", alpha=2.0, beta=2.0)
    )
    azimuth: str = "uniform"

    @model_validator(mode="after")
    def validate_orientation_contract(self) -> Self:
        if self.distribution.family != "beta":
            raise ValueError("orientation distribution must use beta family")
        if self.azimuth != "uniform":
            raise ValueError("orientation azimuth must be uniform")
        return self


class GeneratorConfig(StrictModel):
    schema_version: Literal[3] = 3
    source_schema_version: Literal[2, 3] = 3
    task: TaskSpec
    film: FilmSpec
    formal_targets: FormalTargets
    generation_controls: GenerationControlSpec
    measurement: MeasurementSpec = Field(default_factory=MeasurementSpec)

    # Compatibility views used by the v2 generation implementation while the
    # schema-v3 measurement pipeline is introduced task by task.
    pores: GlobalPoreSpec
    center_distribution: CenterDistributionSpec
    compact: CompactPoreSpec
    pore_material: PoreMaterialSpec | None = None
    orientation: OrientationSpec = Field(default_factory=OrientationSpec)
    channel: ChannelPoreSpec = Field(default_factory=ChannelPoreSpec)
    matrix_constraints: MatrixConstraintSpec = Field(default_factory=MatrixConstraintSpec)
    audit: AuditSpec = Field(
        default_factory=AuditSpec,
        validation_alias=AliasChoices("audit", "geometry_audit"),
    )
    output: OutputSpec = Field(default_factory=OutputSpec)
    optimization: OptimizationSpec = Field(default_factory=OptimizationSpec)
    parallel: ParallelSpec = Field(default_factory=ParallelSpec)

    @model_validator(mode="before")
    @classmethod
    def translate_schema_contract(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        data = dict(raw)
        formal = data.get("formal_targets")
        generation = data.get("generation_controls")
        is_v3_input = formal is not None
        data.setdefault("schema_version", 3)
        data.setdefault("source_schema_version", 3 if is_v3_input else 2)

        if is_v3_input:
            if not isinstance(formal, dict):
                raise TypeError("formal_targets must be a mapping")
            if not isinstance(generation, dict):
                raise ValueError("schema v3 requires generation_controls")
            proportion = formal.get("proportion") or {}
            shape = formal.get("shape") or {}
            position = formal.get("position_quantity") or {}
            porosity = proportion.get("porosity")
            if porosity is None:
                raise ValueError("schema v3 requires formal_targets.proportion.porosity")

            data.setdefault(
                "pores",
                {
                    "seed_number_density_A3": generation.get("seed_number_density_A3"),
                    "target_porosity": porosity,
                    "channel_fraction_by_count": generation.get(
                        "channel_fraction_by_count", 1.0
                    ),
                    "channel_to_compact_mean_volume_ratio": generation.get(
                        "channel_to_compact_mean_volume_ratio", 1.0
                    ),
                },
            )
            density = float(generation.get("seed_number_density_A3", 0.0))
            center_target = position.get("center_distance_xy") or {}
            physical_components = center_target.get("components") or ()
            density_scale = density ** (1.0 / 3.0) if density > 0.0 else 1.0
            data.setdefault(
                "center_distribution",
                {
                    "mode": "rdf",
                    "rdf": [
                        {
                            "kind": component.get("kind", "peak"),
                            "amplitude": component.get("amplitude", 1.0),
                            "center_xi": float(component["center_A"]) * density_scale,
                            "width_xi": float(component["width_A"]) * density_scale,
                        }
                        for component in physical_components
                    ],
                },
            )

            compact_eta = shape.get("compact_aspect_ratio") or {
                "family": "constant",
                "value": 1.5,
            }
            channel_eta = shape.get("channel_aspect_ratio") or {
                "family": "constant",
                "value": 4.0,
            }
            channel_tau = shape.get("channel_tortuosity") or {
                "family": "constant",
                "value": 1.0,
            }
            data.setdefault(
                "compact",
                {
                    "relative_volume": generation.get(
                        "compact_relative_volume", {"family": "constant", "value": 1.0}
                    ),
                    "aspect_ratio": compact_eta,
                    "roughness": generation.get(
                        "compact_roughness", {"family": "constant", "value": 0.0}
                    ),
                },
            )
            data.setdefault(
                "channel",
                {
                    "relative_volume": generation.get(
                        "channel_relative_volume", {"family": "constant", "value": 1.0}
                    ),
                    "eta": channel_eta,
                    "tau": channel_tau,
                    "roughness": generation.get(
                        "channel_roughness", {"family": "constant", "value": 0.0}
                    ),
                },
            )
            paired_orientation = shape.get("orientation") or {}
            components = paired_orientation.get("components") or ()
            if components:
                first_xz = dict(components[0].get("theta_xz_deg") or {})
                if first_xz.get("family") == "beta":
                    for key in ("lower", "upper", "minimum", "maximum"):
                        if key in first_xz and first_xz[key] is not None:
                            first_xz[key] = float(first_xz[key]) / 180.0
                    data.setdefault(
                        "orientation",
                        {"distribution": first_xz, "azimuth": "uniform"},
                    )
            data.setdefault(
                "orientation",
                {
                    "distribution": {"family": "beta", "alpha": 2.0, "beta": 2.0},
                    "azimuth": "uniform",
                },
            )
        else:
            pores = data.get("pores")
            compact = data.get("compact")
            channel = data.get("channel") or {}
            if not isinstance(pores, dict) or not isinstance(compact, dict):
                return data
            data.setdefault(
                "generation_controls",
                {
                    "seed_number_density_A3": pores.get("seed_number_density_A3"),
                    "channel_fraction_by_count": pores.get(
                        "channel_fraction_by_count", 0.0
                    ),
                    "channel_to_compact_mean_volume_ratio": pores.get(
                        "channel_to_compact_mean_volume_ratio", 1.0
                    ),
                    "compact_relative_volume": compact.get(
                        "relative_volume", {"family": "constant", "value": 1.0}
                    ),
                    "channel_relative_volume": channel.get(
                        "relative_volume", {"family": "constant", "value": 1.0}
                    ),
                    "compact_roughness": compact.get(
                        "roughness", {"family": "constant", "value": 0.0}
                    ),
                    "channel_roughness": channel.get(
                        "roughness", {"family": "constant", "value": 0.0}
                    ),
                },
            )
            data.setdefault(
                "formal_targets",
                {
                    "position_quantity": {"center_distance_xy": None},
                    "shape": {
                        "equivalent_diameter_A": None,
                        "orientation": None,
                        "compact_aspect_ratio": compact.get("aspect_ratio"),
                        "channel_aspect_ratio": channel.get("eta")
                        or channel.get("aspect_ratio")
                        or compact.get("aspect_ratio"),
                        "channel_tortuosity": channel.get("tau")
                        or channel.get("tortuosity")
                        or {"family": "constant", "value": 1.0},
                        "curvature_fluctuation": None,
                    },
                    "proportion": {"porosity": pores.get("target_porosity")},
                },
            )
        return data

    @model_validator(mode="after")
    def validate_schema_and_seed_panel(self) -> Self:
        if self.source_schema_version == 3:
            if self.formal_targets.position_quantity.center_distance_xy is None:
                raise ValueError("schema v3 requires center_distance_xy")
            shape = self.formal_targets.shape
            missing = [
                name
                for name, value in (
                    ("equivalent_diameter_A", shape.equivalent_diameter_A),
                    ("orientation", shape.orientation),
                    ("channel_aspect_ratio", shape.channel_aspect_ratio),
                    ("channel_tortuosity", shape.channel_tortuosity),
                    ("curvature_fluctuation", shape.curvature_fluctuation),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "schema v3 missing formal shape targets: " + ", ".join(missing)
                )
        if self.optimization.seed_panel is None:
            self.optimization.seed_panel = (self.task.random_seed,)
        return self

    @property
    def seed_count(self) -> int:
        density = (
            self.generation_controls.seed_number_density_A3
            if self.source_schema_version == 3
            else self.pores.seed_number_density_A3
        )
        return round(density * self.film.target_volume_A3)

    @property
    def geometry_audit(self) -> AuditSpec:
        return self.audit


def load_config(path: Path) -> GeneratorConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("configuration file must contain a YAML mapping")

    pore_material = data.get("pore_material")
    if isinstance(pore_material, dict) and "pdb" in pore_material:
        pdb_path = Path(pore_material["pdb"])
        if not pdb_path.is_absolute():
            pore_material["pdb"] = config_path.parent / pdb_path

    return GeneratorConfig.model_validate(data)
