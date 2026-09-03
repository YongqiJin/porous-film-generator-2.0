from __future__ import annotations

import base64
import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from porous_film.config import GeneratorConfig
from porous_film.distributions import stratified_sample
from porous_film.geometry import BuiltGeometry
from porous_film.metrics import AuditResult
from porous_film.parallel import CandidateResult
from porous_film.voxel import PhaseGrid

_MAX_POINT_COUNT = 10_000
_MAX_SLICE_COUNT = 48
_MAX_SLICE_PIXELS_PER_AXIS = 160
_TARGET_SAMPLE_COUNT = 512
_DISTRIBUTION_KS_LIMIT = 0.20
_DISTRIBUTION_WASSERSTEIN_LIMIT = 0.20
_RDF_LOSS_LIMIT = 1.0

_INPUT_SECTION_LABELS = {
    "contract": "合同与版本",
    "task": "任务与随机数",
    "film": "计算盒",
    "formal_targets": "正式目标",
    "generation_controls": "生成控制",
    "measurement": "测量定义",
    "pore_constraints": "孔相拓扑与样本约束",
    "matrix_constraints": "基体约束",
    "pore_material": "孔内材料",
    "audit": "候选与审核设置",
    "output": "输出设置",
    "optimization": "优化设置",
    "parallel": "并行设置",
    "pores": "孔参数（兼容输入）",
    "center_distribution": "中心分布（兼容输入）",
    "compact": "紧凑孔参数（兼容输入）",
    "orientation": "方向参数（兼容输入）",
    "channel": "通道参数（兼容输入）",
}


def write_visual_report(
    path: Path,
    *,
    config: GeneratorConfig,
    built: BuiltGeometry,
    grid: PhaseGrid,
    audit: AuditResult,
    candidates: Sequence[CandidateResult],
    selected_sequence_index: int,
    performance: Mapping[str, Any] | None = None,
) -> Path:
    """Write a self-contained, read-only HTML report for one geometry run."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _report_payload(
        config=config,
        built=built,
        grid=grid,
        audit=audit,
        candidates=candidates,
        selected_sequence_index=selected_sequence_index,
        performance=performance,
    )
    encoded = json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")
    title = html.escape(f"{config.task.name} · Porous Film Visual Report")
    output_path.write_text(_html_document(title, encoded), encoding="utf-8")
    return output_path


def _report_payload(
    *,
    config: GeneratorConfig,
    built: BuiltGeometry,
    grid: PhaseGrid,
    audit: AuditResult,
    candidates: Sequence[CandidateResult],
    selected_sequence_index: int,
    performance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    measurements = audit.formal_measurements
    target_porosity = float(config.formal_targets.proportion.porosity)
    through_ids = (
        {track.track_id for track in measurements.centerlines if track.is_through}
        if measurements is not None
        else set()
    )
    ordered_candidates = sorted(candidates, key=lambda value: value.sequence_index)
    distributions = _distribution_payload(config, audit, through_ids)
    input_parameters = _input_parameter_payload(config)
    input_checks = _input_check_payload(config, grid)
    observations = _observation_payload(audit, measurements, grid, distributions)
    observation_checks = _observation_check_payload(
        config,
        audit,
        measurements,
        grid,
        distributions,
    )
    candidate_rows = [
        {
            "sequence": int(value.sequence_index),
            "round": int(value.identity.round_index),
            "candidate": int(value.identity.candidate_index),
            "seed": int(value.identity.derived_random_seed),
            "succeeded": bool(value.succeeded),
            "passed": bool(value.audit_passed),
            "selected": int(value.sequence_index) == int(selected_sequence_index),
            "porosity": None if value.porosity is None else float(value.porosity),
            "porosity_error": None
            if value.porosity is None
            else abs(float(value.porosity) - target_porosity),
            "wall_time_seconds": float(value.wall_time_seconds),
            "warning_count": len(value.warnings),
            "failure": value.failure_message,
        }
        for value in ordered_candidates
    ]
    performance_payload = _performance_payload(
        performance,
        candidates=ordered_candidates,
        selected_sequence_index=selected_sequence_index,
        grid=grid,
    )
    return {
        "meta": {
            "task": config.task.name,
            "seed": int(config.task.random_seed),
            "schema_version": int(config.schema_version),
            "audit_passed": bool(audit.passed),
            "spacing_A": float(grid.spacing_A),
            "box_A": np.asarray(grid.target_box_A, dtype=float).tolist(),
            "unit_count": len(built.units),
        },
        "geometry": {
            "pore_points": _sample_pore_points(grid),
            "centerlines": _centerline_payload(measurements),
            "slices": _slice_payload(grid),
        },
        "validation": {
            "porosity": {
                "target": target_porosity,
                "actual": float(grid.porosity),
                "absolute_error": abs(float(grid.porosity) - target_porosity),
            },
            "through_centerline_count": 0
            if measurements is None
            else int(measurements.through_centerline_count),
            "branch_event_count": 0
            if measurements is None
            else int(measurements.branch_event_count),
            "constraints": _constraint_payload(config, audit, measurements),
            "distributions": distributions,
            "rdf": audit.center_distance_xy_result,
            "warnings": list(audit.warnings),
            "input_parameters": input_parameters,
            "input_checks": input_checks,
            "observations": observations,
            "observation_checks": observation_checks,
            "summary": _validation_summary(
                input_checks,
                observation_checks,
                audit_passed=bool(audit.passed),
            ),
        },
        "optimization": {
            "candidates": candidate_rows,
            "configured_seeds": []
            if config.optimization.seed_panel is None
            else [int(value) for value in config.optimization.seed_panel],
            "generation_process": _generation_process_payload(
                performance_payload,
                candidate_rows,
                audit_passed=bool(audit.passed),
            ),
        },
        "performance": performance_payload,
    }


def _input_parameter_payload(config: GeneratorConfig) -> list[dict[str, Any]]:
    normalized = config.model_dump(mode="json", exclude_none=True)
    normalized["contract"] = {
        "schema_version": int(config.schema_version),
        "source_schema_version": int(config.source_schema_version),
        "periodic_axes": ["x", "y"],
        "finite_axes": ["z"],
        "derived_seed_count": int(config.seed_count),
    }
    if config.source_schema_version == 3:
        section_order = (
            "contract",
            "task",
            "film",
            "formal_targets",
            "generation_controls",
            "measurement",
            "pore_constraints",
            "matrix_constraints",
            "pore_material",
            "audit",
            "output",
            "optimization",
            "parallel",
        )
    else:
        section_order = (
            "contract",
            "task",
            "film",
            "pores",
            "center_distribution",
            "compact",
            "orientation",
            "channel",
            "matrix_constraints",
            "pore_material",
            "audit",
            "output",
            "optimization",
            "parallel",
        )
    sections = []
    for key in section_order:
        if key not in normalized:
            continue
        rows = _flatten_parameter_rows(normalized[key], key)
        sections.append(
            {
                "key": key,
                "label": _INPUT_SECTION_LABELS.get(key, key),
                "rows": rows,
            }
        )
    return sections


def _flatten_parameter_rows(value: Any, path: str) -> list[dict[str, str]]:
    if isinstance(value, Mapping):
        rows: list[dict[str, str]] = []
        for key, item in value.items():
            rows.extend(_flatten_parameter_rows(item, f"{path}.{key}"))
        return rows or [{"path": path, "value": "{}"}]
    if isinstance(value, (list, tuple)):
        if not value:
            return [{"path": path, "value": "[]"}]
        if all(not isinstance(item, (Mapping, list, tuple)) for item in value):
            return [{"path": path, "value": _display_value(value)}]
        rows = []
        for index, item in enumerate(value):
            rows.extend(_flatten_parameter_rows(item, f"{path}[{index}]"))
        return rows
    return [{"path": path, "value": _display_value(value)}]


def _input_check_payload(
    config: GeneratorConfig,
    grid: PhaseGrid,
) -> list[dict[str, Any]]:
    target_box = np.asarray(grid.target_box_A, dtype=float)
    fine_spacing = float(config.audit.fine_spacing_A)
    coarse_spacing = float(config.audit.coarse_spacing_A)
    shape = config.formal_targets.shape
    configured_shape_targets = (
        shape.equivalent_diameter_A,
        shape.orientation,
        shape.compact_aspect_ratio,
        shape.channel_aspect_ratio,
        shape.channel_tortuosity,
        shape.curvature_fluctuation,
    )
    configured_target_count = sum(
        value is not None
        for value in (
            config.formal_targets.position_quantity.center_distance_xy,
            *configured_shape_targets,
        )
    )
    orientation_weights = (
        []
        if shape.orientation is None
        else [float(component.weight) for component in shape.orientation.components]
    )
    seed_count = int(config.seed_count)
    audit_enabled = bool(config.audit.enabled)
    return [
        _check_row(
            "Schema 与正式目标合同",
            "PASS" if config.source_schema_version in {2, 3} else "FAIL",
            f"source_schema_version={config.source_schema_version}",
            "必须可按受支持合同解析",
            "配置已通过严格 Pydantic 模型解析。",
        ),
        _check_row(
            "Schema v3 正式目标",
            "PASS" if config.source_schema_version == 3 else "N/A",
            f"孔隙率 + {configured_target_count} 项可选目标",
            "孔隙率必填；其余目标按需配置、未配置即 N/A",
            "只对明确配置的正式目标进行最终孔相审核。"
            if config.source_schema_version == 3
            else "当前是兼容输入。",
            gate=config.source_schema_version == 3,
        ),
        _check_row(
            "方向混合权重",
            "N/A"
            if not orientation_weights
            else "PASS"
            if np.isclose(sum(orientation_weights), 1.0)
            else "FAIL",
            _display_value(orientation_weights),
            "权重和 = 1",
            f"权重和为 {_format_number(sum(orientation_weights))}。"
            if orientation_weights
            else "本输入没有成对方向混合组件。",
            gate=bool(orientation_weights),
        ),
        _check_row(
            "派生中心数量",
            "FAIL" if seed_count < 1 else "WARN" if seed_count < 5 else "PASS",
            str(seed_count),
            ">= 1；分布审核建议 >= 5",
            "样本数很少，分布与贯通审核容易失败。"
            if 0 < seed_count < 5
            else "中心数量可用于生成。",
        ),
        _check_row(
            "细网格整除计算盒",
            "PASS" if _box_divisible(target_box, fine_spacing) else "FAIL",
            f"box={_display_value(target_box.tolist())}, spacing={fine_spacing} Å",
            "每个轴长度必须整除 spacing",
            "体素网格可精确构造。"
            if _box_divisible(target_box, fine_spacing)
            else "至少一个轴无法被细网格间距整除。",
        ),
        _check_row(
            "粗网格整除计算盒",
            "PASS" if _box_divisible(target_box, coarse_spacing) else "FAIL",
            f"box={_display_value(target_box.tolist())}, spacing={coarse_spacing} Å",
            "每个轴长度必须整除 spacing",
            "粗网格可精确构造。"
            if _box_divisible(target_box, coarse_spacing)
            else "至少一个轴无法被粗网格间距整除。",
        ),
        _check_row(
            "候选搜索规模",
            "PASS",
            (
                f"{config.audit.maximum_rounds} 轮 × "
                f"{config.audit.candidate_count_per_round} 候选"
            ),
            ">= 1 个候选",
            "候选计划有效。",
        ),
        _check_row(
            "额外几何审核开关",
            "PASS" if audit_enabled else "WARN",
            f"audit.enabled={str(audit_enabled).lower()}",
            "正式目标始终从最终孔相比较",
            "已启用粗细网格稳定性审核。"
            if audit_enabled
            else "关闭的是额外粗细网格稳定性审核；schema-v3 最终目标比较仍会执行。",
        ),
        _check_row(
            "可视化输出",
            "PASS" if config.output.write_plots else "N/A",
            f"output.write_plots={str(config.output.write_plots).lower()}",
            "true 才生成 HTML",
            "本次运行会生成报告。"
            if config.output.write_plots
            else "本次配置不会生成可视化报告。",
        ),
    ]


def _check_row(
    name: str,
    status: str,
    observed: str,
    requirement: str,
    reason: str,
    *,
    category: str = "输入配置",
    gate: bool = True,
) -> dict[str, Any]:
    return {
        "category": category,
        "name": name,
        "status": status,
        "observed": observed,
        "requirement": requirement,
        "reason": reason,
        "gate": bool(gate),
    }


def _box_divisible(box_A: np.ndarray, spacing_A: float) -> bool:
    ratios = np.asarray(box_A, dtype=float) / float(spacing_A)
    return bool(np.all(np.isclose(ratios, np.rint(ratios), rtol=0.0, atol=1.0e-9)))


def _observation_payload(
    audit: AuditResult,
    measurements: Any | None,
    grid: PhaseGrid,
    distributions: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    pore_voxels = int(np.count_nonzero(grid.pore_mask))
    total_centerlines = 0 if measurements is None else len(measurements.centerlines)
    through_count = 0 if measurements is None else int(measurements.through_centerline_count)
    branch_count = 0 if measurements is None else int(measurements.branch_event_count)
    rows = [
        _observation_row("最终孔隙率", _format_number(grid.porosity), "final_phase.h5"),
        _observation_row(
            "孔体素数量",
            f"{pore_voxels} / {grid.pore_mask.size}",
            "final_phase.h5",
        ),
        _observation_row("中心线总数", str(total_centerlines), "最终孔相同义测量"),
        _observation_row("z 贯通中心线", str(through_count), "最终孔相同义测量"),
        _observation_row("分支事件", str(branch_count), "最终孔相同义测量"),
        _observation_row("连通孔域数量", str(audit.connected_pore_domains), "最终孔体素"),
        _observation_row(
            "最大孔域占比",
            _format_number(audit.largest_pore_fraction),
            "最终孔体素",
        ),
        _observation_row("孔重叠比例", _format_number(audit.overlap_fraction), "最终几何覆盖"),
        _observation_row(
            "最小半导体 yz 截面",
            _format_number(audit.minimum_cross_section_fraction),
            "最终半导体体素",
        ),
        _observation_row(
            "x 表面开口",
            _display_value(list(audit.x_surface_openings)),
            "最终孔体素",
        ),
        _observation_row(
            "y 表面开口",
            _display_value(list(audit.y_surface_openings)),
            "最终孔体素",
        ),
        _observation_row(
            "z 下/上表面开口",
            _display_value([audit.z_lower_opening_fraction, audit.z_upper_opening_fraction]),
            "最终孔体素",
        ),
    ]
    for distribution in distributions:
        summary = distribution["observed_summary"]
        rows.append(
            _observation_row(
                f"{distribution['name']} 有效样本",
                _sample_summary_text(summary),
                "z 贯通中心线及其截面",
            )
        )
    return rows


def _observation_row(name: str, value: str, source: str) -> dict[str, str]:
    return {"name": name, "value": value, "source": source}


def _observation_check_payload(
    config: GeneratorConfig,
    audit: AuditResult,
    measurements: Any | None,
    grid: PhaseGrid,
    distributions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    target_porosity = float(config.formal_targets.proportion.porosity)
    porosity_error = abs(float(grid.porosity) - target_porosity)
    porosity_tolerance = max(1.0 / float(grid.pore_mask.size), 0.01)
    through_count = 0 if measurements is None else int(measurements.through_centerline_count)
    pore_constraints = config.pore_constraints
    z_topology_required = pore_constraints.z_connectivity == "all_components"
    connected_count = int(audit.connected_pore_domains)
    through_component_count = int(audit.through_pore_domain_count)
    rows = [
        _check_row(
            "孔隙率",
            "PASS" if porosity_error <= porosity_tolerance else "FAIL",
            f"{_format_number(grid.porosity)}；|Δ|={_format_number(porosity_error)}",
            (
                f"输入 {_format_number(target_porosity)}；"
                f"|Δ| <= {_format_number(porosity_tolerance)}"
            ),
            "最终孔相孔隙率满足容差。"
            if porosity_error <= porosity_tolerance
            else "最终孔相孔隙率偏差超过体素分辨率容差。",
            category="比例",
        ),
        _check_row(
            "z 孔拓扑",
            "N/A"
            if not z_topology_required
            else "PASS"
            if connected_count == through_component_count
            else "FAIL",
            f"{through_component_count}/{connected_count} 个组件贯通",
            "所有孔组件贯通 z" if z_topology_required else "unrestricted：不审核",
            "该约束未启用。"
            if not z_topology_required
            else "所有最终孔组件均贯通 z。"
            if connected_count == through_component_count
            else "至少一个最终孔组件未贯通 z。",
            category="贯通性",
            gate=z_topology_required,
        ),
        _check_row(
            "最少贯通中心线",
            "N/A"
            if pore_constraints.minimum_through_centerlines <= 0
            else "PASS"
            if through_count >= pore_constraints.minimum_through_centerlines
            else "FAIL",
            str(through_count),
            (
                f">= {pore_constraints.minimum_through_centerlines}"
                if pore_constraints.minimum_through_centerlines > 0
                else "未配置正数门槛"
            ),
            "未启用该样本数门槛。"
            if pore_constraints.minimum_through_centerlines <= 0
            else "贯通中心线数量满足门槛。"
            if through_count >= pore_constraints.minimum_through_centerlines
            else "贯通中心线数量低于门槛。",
            category="贯通性",
            gate=pore_constraints.minimum_through_centerlines > 0,
        ),
        _check_row(
            "最少有效贯通截面",
            "N/A"
            if pore_constraints.minimum_valid_cross_sections <= 0
            else "PASS"
            if audit.valid_through_cross_section_count
            >= pore_constraints.minimum_valid_cross_sections
            else "FAIL",
            str(audit.valid_through_cross_section_count),
            (
                f">= {pore_constraints.minimum_valid_cross_sections}"
                if pore_constraints.minimum_valid_cross_sections > 0
                else "未配置正数门槛"
            ),
            "未启用该样本数门槛。"
            if pore_constraints.minimum_valid_cross_sections <= 0
            else "有效贯通截面数量满足门槛。"
            if audit.valid_through_cross_section_count
            >= pore_constraints.minimum_valid_cross_sections
            else "有效贯通截面数量低于门槛。",
            category="贯通性",
            gate=pore_constraints.minimum_valid_cross_sections > 0,
        ),
    ]
    rdf = audit.center_distance_xy_result or audit.rdf_result
    rdf_loss = None if not isinstance(rdf, Mapping) else _optional_float(rdf.get("weighted_loss"))
    pair_count = 0 if not isinstance(rdf, Mapping) else int(rdf.get("pair_count", 0))
    rdf_passed = bool(isinstance(rdf, Mapping) and rdf.get("passed"))
    rdf_required = bool(
        z_topology_required
        and config.formal_targets.position_quantity.center_distance_xy is not None
    )
    rows.append(
        _check_row(
            "中心距离 g_xy",
            "N/A" if not rdf_required else "PASS" if rdf_passed else "FAIL",
            f"weighted_loss={_display_value(rdf_loss)}；pair_count={pair_count}",
            f"输入中心距离模型；weighted_loss <= {_RDF_LOSS_LIMIT}",
            "该正式目标未配置或当前为 unrestricted。"
            if not rdf_required
            else "中心距离分布满足目标。"
            if rdf_passed
            else (
                "没有有效中心对，无法审核输入的中心距离目标。"
                if pair_count == 0
                else "中心距离加权损失超过门槛。"
            ),
            category="位置",
            gate=rdf_required,
        )
    )
    for distribution in distributions:
        sample_count = int(distribution["observed_summary"]["count"])
        comparison_passed = distribution.get("passed") is True
        required = bool(distribution.get("required"))
        rows.append(
            _check_row(
                str(distribution["name"]),
                "N/A" if not required else "PASS" if comparison_passed else "FAIL",
                (
                    f"{_sample_summary_text(distribution['observed_summary'])}；"
                    f"KS={_display_value(distribution.get('ks'))}；"
                    f"W={_display_value(distribution.get('wasserstein'))}"
                ),
                (
                    f"输入 {distribution['target_description']}；"
                    f"KS <= {_DISTRIBUTION_KS_LIMIT} 且 W <= "
                    f"{_DISTRIBUTION_WASSERSTEIN_LIMIT}"
                ),
                "该正式目标没有配置。"
                if not required
                else "最终孔相分布满足输入目标。"
                if comparison_passed
                else (
                    "最终孔相没有可用于该正式目标的有效贯通样本。"
                    if sample_count == 0
                    else "KS 或归一化 Wasserstein 距离超过门槛。"
                ),
                category="形状分布",
                gate=required,
            )
        )
    matrix_enabled = bool(config.matrix_constraints.enabled)
    x_warning = any("does not percolate along periodic x" in item for item in audit.warnings)
    x_required = bool(config.matrix_constraints.require_x_percolation)
    x_status = (
        "N/A"
        if not matrix_enabled or not x_required
        else "FAIL"
        if x_warning
        else "PASS"
    )
    rows.extend(
        [
            _check_row(
                "半导体 x 贯通",
                x_status,
                "失败" if x_warning else "未发现不贯通警告",
                "必须贯通" if matrix_enabled and x_required else "当前输入未启用此门槛",
                "该约束未启用。"
                if x_status == "SKIP"
                else "半导体保持 x 贯通。"
                if x_status == "PASS"
                else "半导体未保持 x 贯通。",
                category="基体约束",
                gate=matrix_enabled and x_required,
            ),
            _check_row(
                "最小半导体 yz 截面",
                (
                    "N/A"
                    if not matrix_enabled
                    else "PASS"
                    if audit.minimum_cross_section_fraction
                    >= config.matrix_constraints.minimum_cross_section_fraction
                    else "FAIL"
                ),
                _format_number(audit.minimum_cross_section_fraction),
                (
                    f">= {_format_number(config.matrix_constraints.minimum_cross_section_fraction)}"
                    if matrix_enabled
                    else "当前输入未启用基体约束"
                ),
                "该约束未启用。"
                if not matrix_enabled
                else "最终半导体截面满足门槛。"
                if audit.minimum_cross_section_fraction
                >= config.matrix_constraints.minimum_cross_section_fraction
                else "最终半导体截面低于门槛。",
                category="基体约束",
                gate=matrix_enabled,
            ),
            _check_row(
                "最大孔重叠比例",
                (
                    "N/A"
                    if not matrix_enabled
                    else "PASS"
                    if audit.overlap_fraction
                    <= config.matrix_constraints.maximum_overlap_fraction
                    else "FAIL"
                ),
                _format_number(audit.overlap_fraction),
                (
                    f"<= {_format_number(config.matrix_constraints.maximum_overlap_fraction)}"
                    if matrix_enabled
                    else "当前输入未启用基体约束"
                ),
                "该约束未启用。"
                if not matrix_enabled
                else "孔重叠比例满足门槛。"
                if audit.overlap_fraction
                <= config.matrix_constraints.maximum_overlap_fraction
                else "孔重叠比例超过门槛。",
                category="基体约束",
                gate=matrix_enabled,
            ),
            _check_row(
                "粗细网格局部厚度稳定性",
                (
                    "N/A"
                    if not config.audit.enabled
                    else "PASS"
                    if audit.local_thickness_stability_result.passed
                    else "FAIL"
                ),
                (
                    f"max error={_format_number(audit.local_thickness_stability_result.max_quantile_error_A)} Å"
                ),
                (
                    f"<= {_format_number(audit.local_thickness_stability_result.tolerance_A)} Å"
                    if config.audit.enabled
                    else "audit.enabled=false"
                ),
                "该审核被输入配置关闭。"
                if not config.audit.enabled
                else "粗细网格测量一致。"
                if audit.local_thickness_stability_result.passed
                else "粗细网格局部厚度差异超过门槛。",
                category="数值稳定性",
                gate=bool(config.audit.enabled),
            ),
        ]
    )
    return rows


def _validation_summary(
    input_checks: Sequence[Mapping[str, Any]],
    observation_checks: Sequence[Mapping[str, Any]],
    *,
    audit_passed: bool,
) -> dict[str, Any]:
    input_failures = sum(row["status"] == "FAIL" for row in input_checks)
    observed_failures = sum(row["status"] == "FAIL" for row in observation_checks)
    gated_failures = sum(
        row["status"] == "FAIL" and bool(row.get("gate")) for row in observation_checks
    )
    return {
        "status": "PASS" if audit_passed else "FAIL",
        "input_failure_count": int(input_failures),
        "observation_failure_count": int(observed_failures),
        "gated_failure_count": int(gated_failures),
        "input_check_count": len(input_checks),
        "observation_check_count": len(observation_checks),
    }


def _generation_process_payload(
    performance: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    audit_passed: bool,
) -> list[dict[str, Any]]:
    stage_seconds = dict(performance.get("stage_timings_seconds") or {})
    stage_calls = dict(performance.get("stage_call_counts") or {})
    successful_candidates = sum(bool(candidate.get("succeeded")) for candidate in candidates)
    passed_candidates = sum(bool(candidate.get("passed")) for candidate in candidates)

    def stage_total(*names: str) -> float | None:
        values = [float(stage_seconds[name]) for name in names if name in stage_seconds]
        return sum(values) if values else None

    def stage_calls_total(*names: str) -> int | None:
        values = [int(stage_calls[name]) for name in names if name in stage_calls]
        return sum(values) if values else None

    candidate_status = "PASS" if passed_candidates else "WARN" if successful_candidates else "FAIL"
    return [
        {
            "name": "输入与执行计划",
            "status": "PASS",
            "seconds": None,
            "calls": 1,
            "detail": f"配置解析完成；候选数 {len(candidates)}",
        },
        {
            "name": "候选搜索",
            "status": candidate_status,
            "seconds": _optional_float(performance.get("candidate_search_wall_time_seconds")),
            "calls": len(candidates),
            "detail": f"成功 {successful_candidates}；审核通过 {passed_candidates}",
        },
        {
            "name": "中心种子生成",
            "status": "PASS",
            "seconds": stage_total("center_seed_generation"),
            "calls": stage_calls_total("center_seed_generation"),
            "detail": "按输入密度和中心距离控制生成中心",
        },
        {
            "name": "形状与尺度迭代",
            "status": "PASS",
            "seconds": stage_total("shape_generation", "scale_optimization"),
            "calls": stage_calls_total("shape_generation", "scale_optimization"),
            "detail": "构建孔形状并搜索满足目标孔隙率的尺度",
        },
        {
            "name": "体素化",
            "status": "PASS",
            "seconds": stage_total("voxelization"),
            "calls": stage_calls_total("voxelization"),
            "detail": "尺度求解、粗网格和最终网格的累计体素化",
        },
        {
            "name": "中心线与最终测量",
            "status": "PASS",
            "seconds": stage_total("centerline_generation", "final_measurement"),
            "calls": stage_calls_total("centerline_generation", "final_measurement"),
            "detail": "只从最终孔相提取中心线、截面、方向和通道指标",
        },
        {
            "name": "正式目标审核",
            "status": "PASS" if audit_passed else "FAIL",
            "seconds": stage_total("validation"),
            "calls": stage_calls_total("validation"),
            "detail": "全部输入目标与最终观测逐项比较",
        },
        {
            "name": "结果导出",
            "status": "PASS",
            "seconds": stage_total("export"),
            "calls": stage_calls_total("export"),
            "detail": "写入 QA、几何、报告和可视化文件",
        },
    ]


def _display_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return _format_number(value)
    if isinstance(value, (list, tuple, Mapping)):
        return json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _sample_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "minimum": None, "maximum": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def _sample_summary_text(summary: Mapping[str, Any]) -> str:
    count = int(summary.get("count", 0))
    if count == 0:
        return "n=0"
    return (
        f"n={count}；mean={_format_number(summary['mean'])}；"
        f"range=[{_format_number(summary['minimum'])}, {_format_number(summary['maximum'])}]"
    )


def _describe_distribution(target: Any | None) -> str:
    if target is None:
        return "未配置"
    data = target.model_dump(exclude_none=True) if hasattr(target, "model_dump") else target
    if not isinstance(data, Mapping):
        return _display_value(data)
    family = str(data.get("family", "distribution"))
    if family == "constant":
        return f"constant({_display_value(data.get('value'))})"
    if family == "mixture":
        components = data.get("components") or []
        return f"mixture({len(components)} components)"
    parameters = [
        f"{key}={_display_value(value)}"
        for key, value in data.items()
        if key not in {"family", "components"}
    ]
    return f"{family}({', '.join(parameters)})" if parameters else family


def _performance_payload(
    performance: Mapping[str, Any] | None,
    *,
    candidates: Sequence[CandidateResult],
    selected_sequence_index: int,
    grid: PhaseGrid,
) -> dict[str, Any]:
    supplied = dict(performance or {})
    stage_seconds = {
        str(name): float(value)
        for name, value in dict(supplied.get("stage_timings_seconds") or {}).items()
    }
    stage_calls = {
        str(name): int(value)
        for name, value in dict(supplied.get("stage_call_counts") or {}).items()
    }
    labels = {
        "center_seed_generation": "中心种子生成",
        "centerline_generation": "中心线生成",
        "shape_generation": "形状生成与缩放",
        "scale_optimization": "尺度搜索调度",
        "voxelization": "体素化",
        "final_measurement": "最终孔相测量",
        "validation": "约束与分布验证",
        "export": "结果导出",
    }
    stage_order = tuple(labels) + tuple(sorted(set(stage_seconds) - set(labels)))
    stage_rows = [
        {
            "key": name,
            "label": labels.get(name, name),
            "seconds": float(stage_seconds[name]),
            "calls": int(stage_calls.get(name, 0)),
        }
        for name in stage_order
        if name in stage_seconds
    ]
    selected_candidate_seconds = next(
        (
            float(value.wall_time_seconds)
            for value in candidates
            if int(value.sequence_index) == int(selected_sequence_index)
        ),
        None,
    )
    return {
        **supplied,
        "candidate_wall_time_sum_seconds": float(
            supplied.get(
                "candidate_wall_time_sum_seconds",
                sum(value.wall_time_seconds for value in candidates),
            )
        ),
        "selected_candidate_seconds": selected_candidate_seconds,
        "candidate_count": len(candidates),
        "voxel_shape_zyx": list(grid.pore_mask.shape),
        "voxel_count": int(grid.pore_mask.size),
        "peak_rss_mib": _optional_float(supplied.get("peak_rss_mib")),
        "worker_peak_rss_mib": _optional_float(supplied.get("worker_peak_rss_mib")),
        "stage_timings_available": bool(stage_rows),
        "stages": stage_rows,
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sample_pore_points(grid: PhaseGrid) -> list[list[float]]:
    indices = np.argwhere(np.asarray(grid.pore_mask, dtype=bool))
    if indices.size == 0:
        return []
    if indices.shape[0] > _MAX_POINT_COUNT:
        selection = np.linspace(0, indices.shape[0] - 1, _MAX_POINT_COUNT, dtype=int)
        indices = indices[selection]
    zyx = indices.astype(float) + 0.5
    xyz = zyx[:, [2, 1, 0]] * float(grid.spacing_A) + grid.origin_A
    return np.round(xyz, 5).tolist()


def _centerline_payload(measurements: Any | None) -> list[dict[str, Any]]:
    if measurements is None:
        return []
    return [
        {
            "id": int(track.track_id),
            "through": bool(track.is_through),
            "branch": bool(track.has_branch_neighborhood),
            "points": np.round(np.asarray(track.points_unwrapped_A, dtype=float), 5).tolist(),
        }
        for track in measurements.centerlines
    ]


def _slice_payload(grid: PhaseGrid) -> dict[str, Any]:
    mask = np.asarray(grid.pore_mask, dtype=bool)
    nz, ny, nx = mask.shape
    slice_count = min(nz, _MAX_SLICE_COUNT)
    indices = np.unique(np.linspace(0, nz - 1, slice_count, dtype=int))
    step_y = max(1, int(np.ceil(ny / _MAX_SLICE_PIXELS_PER_AXIS)))
    step_x = max(1, int(np.ceil(nx / _MAX_SLICE_PIXELS_PER_AXIS)))
    packed = []
    sampled_shape = None
    for index in indices:
        sampled = mask[int(index), ::step_y, ::step_x]
        sampled_shape = sampled.shape
        encoded = base64.b64encode(np.packbits(sampled.ravel()).tobytes()).decode("ascii")
        packed.append(encoded)
    height, width = sampled_shape or (0, 0)
    return {
        "indices": indices.astype(int).tolist(),
        "width": int(width),
        "height": int(height),
        "packed": packed,
    }


def _constraint_payload(
    config: GeneratorConfig,
    audit: AuditResult,
    measurements: Any | None,
) -> list[dict[str, Any]]:
    through_count = 0 if measurements is None else int(measurements.through_centerline_count)
    valid_cross_section_count = int(audit.valid_through_cross_section_count)
    connected_component_count = int(audit.connected_pore_domains)
    through_component_count = int(audit.through_pore_domain_count)
    pore_constraints = config.pore_constraints
    all_components_required = pore_constraints.z_connectivity == "all_components"
    z_connectivity_passed = (
        connected_component_count == through_component_count
        if all_components_required
        else None
    )
    x_warning = any("does not percolate along periodic x" in item for item in audit.warnings)
    thickness = audit.local_thickness_stability_result
    return [
        {
            "name": "总体审核",
            "passed": bool(audit.passed),
            "value": "PASS" if audit.passed else "FAIL",
            "limit": "全部正式目标与约束",
        },
        {
            "name": "z 孔拓扑",
            "passed": z_connectivity_passed,
            "value": f"{through_component_count}/{connected_component_count} 个组件贯通",
            "limit": "所有组件必须贯通" if all_components_required else "不限制",
        },
        {
            "name": "最少贯通中心线",
            "passed": (
                through_count >= pore_constraints.minimum_through_centerlines
                if pore_constraints.minimum_through_centerlines > 0
                else None
            ),
            "value": str(through_count),
            "limit": f">= {pore_constraints.minimum_through_centerlines}",
        },
        {
            "name": "最少有效贯通截面",
            "passed": (
                valid_cross_section_count >= pore_constraints.minimum_valid_cross_sections
                if pore_constraints.minimum_valid_cross_sections > 0
                else None
            ),
            "value": str(valid_cross_section_count),
            "limit": f">= {pore_constraints.minimum_valid_cross_sections}",
        },
        {
            "name": "半导体 x 贯通",
            "passed": (
                None
                if not config.matrix_constraints.enabled
                else not x_warning
                if config.matrix_constraints.require_x_percolation
                else None
            ),
            "value": "通过" if not x_warning else "失败",
            "limit": "必须通过" if config.matrix_constraints.require_x_percolation else "未强制",
        },
        {
            "name": "最小 yz 截面",
            "passed": (
                None
                if not config.matrix_constraints.enabled
                else float(audit.minimum_cross_section_fraction)
                >= float(config.matrix_constraints.minimum_cross_section_fraction)
            ),
            "value": _format_number(audit.minimum_cross_section_fraction),
            "limit": f">= {_format_number(config.matrix_constraints.minimum_cross_section_fraction)}",
        },
        {
            "name": "最大重叠比例",
            "passed": (
                None
                if not config.matrix_constraints.enabled
                else float(audit.overlap_fraction)
                <= float(config.matrix_constraints.maximum_overlap_fraction)
            ),
            "value": _format_number(audit.overlap_fraction),
            "limit": f"<= {_format_number(config.matrix_constraints.maximum_overlap_fraction)}",
        },
        {
            "name": "粗细网格稳定性",
            "passed": bool(thickness.passed) if config.audit.enabled else None,
            "value": _format_number(thickness.max_quantile_error_A) + " Å",
            "limit": (
                f"<= {_format_number(thickness.tolerance_A)} Å"
                if config.audit.enabled
                else "未启用"
            ),
        },
    ]


def _distribution_payload(
    config: GeneratorConfig,
    audit: AuditResult,
    through_ids: set[int],
) -> list[dict[str, Any]]:
    measurements = audit.formal_measurements
    if measurements is None:
        return []
    shape = config.formal_targets.shape
    evaluate_through_targets = config.pore_constraints.z_connectivity == "all_components"
    observed: dict[str, np.ndarray] = {
        "紧凑孔 eta": np.asarray(
            [
                item.eta
                for item in measurements.compact_geometries
                if item.valid and item.eta is not None
            ],
            dtype=float,
        ),
        "等效直径": np.asarray(
            [
                item.equivalent_diameter_A
                for item in measurements.cross_sections
                if item.valid
                and item.track_id in through_ids
                and item.equivalent_diameter_A is not None
            ],
            dtype=float,
        ),
        "theta_xz": np.asarray(
            [
                item.theta_xz_deg
                for item in measurements.projected_orientations
                if item.track_id in through_ids
                and item.theta_xz_identifiable
                and item.theta_xz_deg is not None
            ],
            dtype=float,
        ),
        "theta_xy": np.asarray(
            [
                item.theta_xy_deg
                for item in measurements.projected_orientations
                if item.track_id in through_ids
                and item.theta_xy_identifiable
                and item.theta_xy_deg is not None
            ],
            dtype=float,
        ),
        "通道 eta": np.asarray(
            [
                item.eta
                for item in measurements.channel_geometries
                if item.track_id in through_ids and item.valid and item.eta is not None
            ],
            dtype=float,
        ),
        "通道 tau": np.asarray(
            [
                item.tortuosity
                for item in measurements.channel_geometries
                if item.track_id in through_ids and item.valid and item.tortuosity is not None
            ],
            dtype=float,
        ),
        "曲率波动": np.asarray(
            [
                item.curvature_fluctuation
                for item in measurements.cross_sections
                if item.valid
                and item.track_id in through_ids
                and item.curvature_fluctuation is not None
                and np.isfinite(item.curvature_fluctuation)
            ],
            dtype=float,
        ),
    }
    targets: dict[str, Any] = {
        "紧凑孔 eta": shape.compact_aspect_ratio,
        "等效直径": shape.equivalent_diameter_A,
        "theta_xz": _orientation_marginal(shape.orientation, "theta_xz_deg"),
        "theta_xy": _orientation_marginal(shape.orientation, "theta_xy_deg"),
        "通道 eta": shape.channel_aspect_ratio,
        "通道 tau": shape.channel_tortuosity,
        "曲率波动": shape.curvature_fluctuation,
    }
    result_names = {
        "紧凑孔 eta": "compact_eta",
        "等效直径": "equivalent_diameter",
        "theta_xz": "theta_xz",
        "theta_xy": "theta_xy",
        "通道 eta": "channel_eta",
        "通道 tau": "channel_tau",
        "曲率波动": "curvature_fluctuation",
    }
    rng = np.random.default_rng(0)
    rows = []
    for label, values in observed.items():
        target = targets[label]
        through_dependent = label != "紧凑孔 eta"
        required = bool(
            target is not None and (not through_dependent or evaluate_through_targets)
        )
        target_values = (
            np.array([], dtype=float)
            if target is None
            else np.asarray(stratified_sample(target, _TARGET_SAMPLE_COUNT, rng), dtype=float)
        )
        comparison = audit.distribution_results.get(result_names[label])
        rows.append(
            {
                "name": label,
                "observed": _finite_list(values),
                "observed_summary": _sample_summary(values),
                "target": _finite_list(target_values),
                "target_description": _describe_distribution(target),
                "required": required,
                "passed": None if comparison is None else bool(comparison.passed),
                "ks": None if comparison is None else float(comparison.ks),
                "wasserstein": None
                if comparison is None
                else float(comparison.normalized_wasserstein),
            }
        )
    return rows


def _orientation_marginal(target: Any | None, field_name: str) -> dict[str, Any] | None:
    if target is None:
        return None
    return {
        "family": "mixture",
        "components": [
            {
                "weight": float(component.weight),
                **getattr(component, field_name).model_dump(exclude_none=True),
            }
            for component in target.components
        ],
    }


def _finite_list(values: np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=float)
    return np.round(array[np.isfinite(array)], 8).tolist()


def _format_number(value: Any) -> str:
    number = float(value)
    return "∞" if not np.isfinite(number) else f"{number:.4g}"


def _html_document(title: str, payload: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--bg:#f6f8fb;--panel:#fff;--text:#172033;--muted:#667085;--border:#d8dee9;--accent:#2457d6;--good:#168653;--bad:#c33b3b;--warn:#b26a00;--pore:#3976e8;--through:#16a36a;--other:#e18b2d}}
@media(prefers-color-scheme:dark){{:root{{--bg:#111722;--panel:#182131;--text:#eef2f8;--muted:#aab4c4;--border:#334155;--accent:#79a2ff;--good:#51cf93;--bad:#ff7777;--warn:#ffc267;--pore:#72a2ff;--through:#52d89a;--other:#ffb45e}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,sans-serif}}main{{max-width:1280px;margin:auto;padding:22px}}h1{{font-size:24px;margin:0 0 4px}}h2{{font-size:18px;margin:0 0 14px}}.muted{{color:var(--muted)}}nav{{display:flex;gap:8px;flex-wrap:wrap;margin:20px 0}}button{{font:inherit;color:var(--text);background:transparent;border:1px solid var(--border);border-radius:7px;padding:7px 12px;cursor:pointer}}button[aria-selected=true]{{background:var(--accent);color:#fff;border-color:var(--accent)}}section[hidden]{{display:none}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.stats{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:16px}}.panel,.stat{{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px}}.stat strong{{display:block;font-size:20px}}canvas{{display:block;width:100%;background:var(--panel);border:1px solid var(--border);border-radius:8px}}#scene{{height:470px}}#slice{{height:470px;image-rendering:pixelated}}.controls{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:10px 0}}input[type=range]{{min-width:220px}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;border-bottom:1px solid var(--border);padding:8px;vertical-align:top}}th{{color:var(--muted);font-weight:500}}.pass{{color:var(--good);font-weight:600}}.fail{{color:var(--bad);font-weight:600}}.na{{color:var(--muted)}}.chart{{height:220px}}.warning{{border-left:3px solid var(--warn);padding:6px 10px;margin:6px 0;background:color-mix(in srgb,var(--warn) 8%,transparent)}}@media(max-width:760px){{main{{padding:14px}}.grid,.stats{{grid-template-columns:1fr}}#scene,#slice{{height:340px}}}}
.audit-banner{{display:flex;gap:12px;align-items:center;border:1px solid var(--border);border-left-width:5px;border-radius:9px;padding:12px 14px;margin-bottom:16px;background:var(--panel)}}.audit-banner.pass{{border-left-color:var(--good)}}.audit-banner.fail{{border-left-color:var(--bad)}}.audit-status{{font-weight:700;white-space:nowrap}}.audit-status.pass{{color:var(--good)}}.audit-status.fail{{color:var(--bad)}}.audit-status.warn{{color:var(--warn)}}.audit-status.skip{{color:var(--muted)}}.parameter-groups{{display:grid;gap:8px}}details{{border:1px solid var(--border);border-radius:8px;padding:0 10px}}summary{{cursor:pointer;padding:9px 0;font-weight:600}}details table{{margin-bottom:8px}}.path{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}}.process-flow{{display:flex;align-items:stretch;gap:8px;overflow-x:auto;padding:4px 0 8px}}.process-step{{min-width:150px;flex:1;border:1px solid var(--border);border-top-width:4px;border-radius:8px;padding:10px;background:var(--panel)}}.process-step.pass{{border-top-color:var(--good)}}.process-step.fail{{border-top-color:var(--bad)}}.process-step.warn{{border-top-color:var(--warn)}}.process-step.skip{{border-top-color:var(--muted)}}.process-step strong{{display:block;margin:3px 0}}.process-arrow{{align-self:center;color:var(--muted);font-size:20px}}.table-scroll{{overflow:auto}}.audit-table{{min-width:920px}}.parameter-table{{min-width:680px}}@media(max-width:760px){{.process-flow{{display:grid;grid-template-columns:1fr}}.process-arrow{{display:none}}}}
</style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <div class="muted" id="subtitle"></div>
  <nav aria-label="报告页面">
    <button data-tab="geometry" aria-selected="true">Geometry</button>
    <button data-tab="validation" aria-selected="false">Validation</button>
    <button data-tab="optimization" aria-selected="false">Optimization</button>
    <button data-tab="performance" aria-selected="false">Performance</button>
  </nav>
  <section id="geometry">
    <div class="grid">
      <div class="panel"><h2>三维孔相与中心线</h2><div class="controls"><label><input id="showPores" type="checkbox" checked> 孔体素</label><label><input id="showThrough" type="checkbox" checked> 贯通中心线</label><label><input id="showOther" type="checkbox" checked> 其他中心线</label><span class="muted">拖动旋转，滚轮缩放</span></div><canvas id="scene"></canvas></div>
      <div class="panel"><h2>z 截面</h2><div class="controls"><label for="sliceRange">截面 <span id="sliceLabel"></span></label><input id="sliceRange" type="range" min="0" value="0"></div><canvas id="slice"></canvas></div>
    </div>
  </section>
  <section id="validation" hidden>
    <div class="audit-banner" id="validationBanner"></div>
    <div class="stats" id="validationStats"></div>
    <div class="panel"><h2>输入参数（归一化）</h2><p class="muted">展示本次实际进入生成器的参数及默认值；正式目标与生成控制分开列出。</p><div class="parameter-groups" id="inputParameters"></div></div>
    <div class="grid" style="margin-top:16px">
      <div class="panel"><h2>输入配置审核</h2><div class="table-scroll"><table class="audit-table"><thead><tr><th>结果</th><th>检查项</th><th>输入值</th><th>要求</th><th>说明</th></tr></thead><tbody id="inputAuditRows"></tbody></table></div></div>
      <div class="panel"><h2>最终观测</h2><div class="table-scroll"><table><thead><tr><th>观测量</th><th>数值</th><th>来源</th></tr></thead><tbody id="observationRows"></tbody></table></div></div>
    </div>
    <div class="panel" style="margin-top:16px"><h2>输入 → 观测逐项审核</h2><div class="table-scroll"><table class="audit-table"><thead><tr><th>结果</th><th>类别</th><th>审核项</th><th>最终观测</th><th>输入目标 / 判据</th><th>门槛</th><th>解释</th></tr></thead><tbody id="observationAuditRows"></tbody></table></div></div>
    <div class="panel" style="margin-top:16px"><h2>运行警告与失败原因</h2><div id="warnings"></div></div>
  </section>
  <section id="optimization" hidden><div class="panel"><h2>生成过程</h2><p class="muted">复用本次运行已有的候选结果与计时，不保存额外三维中间快照。</p><div class="process-flow" id="generationProcess"></div></div><div class="panel" style="margin-top:16px"><h2>候选搜索</h2><canvas id="candidateChart" class="chart"></canvas><div style="overflow:auto"><table><thead><tr><th>序号</th><th>轮次/候选</th><th>seed</th><th>孔隙率误差</th><th>耗时</th><th>结果</th></tr></thead><tbody id="candidateRows"></tbody></table></div><p class="muted" id="seedPanel"></p></div></section>
  <section id="performance" hidden><div class="stats" id="performanceStats"></div><div class="grid"><div class="panel"><h2>阶段耗时</h2><canvas id="stageChart" class="chart"></canvas><div style="overflow:auto"><table><thead><tr><th>阶段</th><th>耗时</th><th>调用次数</th></tr></thead><tbody id="stageRows"></tbody></table></div><p class="muted" id="performanceScope"></p></div><div class="panel"><h2>候选耗时</h2><canvas id="performanceChart" class="chart"></canvas><p class="muted" id="candidateTimingNote"></p></div></div></section>
</main>
<script id="reportData" type="application/json">{payload}</script>
<script>
const data=JSON.parse(document.getElementById('reportData').textContent);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
document.getElementById('subtitle').textContent=`seed ${{data.meta.seed}} · ${{data.meta.voxel_shape_zyx?.join('×')||''}} voxels · spacing ${{data.meta.spacing_A}} Å`;
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{{document.querySelectorAll('nav button').forEach(x=>x.setAttribute('aria-selected',String(x===b)));document.querySelectorAll('main>section').forEach(s=>s.hidden=s.id!==b.dataset.tab);if(b.dataset.tab==='optimization')drawCandidate();if(b.dataset.tab==='performance')drawPerformance();}});
function fit(c){{const r=c.getBoundingClientRect(),d=devicePixelRatio||1;c.width=Math.max(1,Math.floor(r.width*d));c.height=Math.max(1,Math.floor(r.height*d));const x=c.getContext('2d');x.setTransform(d,0,0,d,0,0);return [x,r.width,r.height]}}
let yaw=.65,pitch=-.35,zoom=1,drag=null;
function rotate(p){{const b=data.meta.box_A,c=[p[0]/b[0]-.5,p[1]/b[1]-.5,p[2]/b[2]-.5];let x=c[0]*Math.cos(yaw)-c[2]*Math.sin(yaw),z=c[0]*Math.sin(yaw)+c[2]*Math.cos(yaw),y=c[1]*Math.cos(pitch)-z*Math.sin(pitch);z=c[1]*Math.sin(pitch)+z*Math.cos(pitch);return[x,y,z]}}
function drawScene(){{const c=document.getElementById('scene'),[x,w,h]=fit(c);x.clearRect(0,0,w,h);const scale=Math.min(w,h)*.68*zoom,project=p=>{{const q=rotate(p);return[w/2+q[0]*scale,h/2-q[1]*scale,q[2]]}};if(document.getElementById('showPores').checked){{const pts=data.geometry.pore_points.map(p=>project(p)).sort((a,b)=>a[2]-b[2]);x.fillStyle=css('--pore');x.globalAlpha=.28;for(const p of pts)x.fillRect(p[0],p[1],1.8,1.8);x.globalAlpha=1}}for(const line of data.geometry.centerlines){{if(line.through&&!document.getElementById('showThrough').checked)continue;if(!line.through&&!document.getElementById('showOther').checked)continue;x.strokeStyle=line.through?css('--through'):css('--other');x.lineWidth=line.through?3:1.5;x.beginPath();line.points.forEach((p,i)=>{{const q=project(p);i?x.lineTo(q[0],q[1]):x.moveTo(q[0],q[1])}});x.stroke()}}}}
const scene=document.getElementById('scene');scene.onpointerdown=e=>{{drag=[e.clientX,e.clientY];scene.setPointerCapture(e.pointerId)}};scene.onpointermove=e=>{{if(!drag)return;yaw+=(e.clientX-drag[0])*.01;pitch+=(e.clientY-drag[1])*.01;drag=[e.clientX,e.clientY];drawScene()}};scene.onpointerup=()=>drag=null;scene.onwheel=e=>{{e.preventDefault();zoom=Math.max(.35,Math.min(3,zoom*Math.exp(-e.deltaY*.001)));drawScene()}};['showPores','showThrough','showOther'].forEach(id=>document.getElementById(id).onchange=drawScene);
function unpack(s,n){{const raw=atob(s),a=new Uint8Array(n);for(let i=0;i<n;i++)a[i]=(raw.charCodeAt(i>>3)>>(7-(i&7)))&1;return a}}
function drawSlice(){{const s=data.geometry.slices,idx=+document.getElementById('sliceRange').value,c=document.getElementById('slice'),[x,w,h]=fit(c);x.clearRect(0,0,w,h);if(!s.packed.length)return;const bits=unpack(s.packed[idx],s.width*s.height),sx=w/s.width,sy=h/s.height;x.fillStyle=css('--pore');for(let y=0;y<s.height;y++)for(let xx=0;xx<s.width;xx++)if(bits[y*s.width+xx])x.fillRect(xx*sx,y*sy,Math.ceil(sx),Math.ceil(sy));document.getElementById('sliceLabel').textContent=`z-index ${{s.indices[idx]}}`}}
const range=document.getElementById('sliceRange');range.max=Math.max(0,data.geometry.slices.packed.length-1);range.oninput=drawSlice;
function status(v){{return v===null?'<span class="na">N/A</span>':v?'<span class="pass">PASS</span>':'<span class="fail">FAIL</span>'}}
function stat(label,value,detail=''){{return `<div class="stat"><span class="muted">${{label}}</span><strong>${{value}}</strong><span class="muted">${{detail}}</span></div>`}}
function escapeHtml(s){{return String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}
function auditStatus(s){{const label=String(s||'N/A'),key=label==='N/A'?'skip':label.toLowerCase();return `<span class="audit-status ${{key}}">${{escapeHtml(label)}}</span>`}}
function auditRows(rows){{const rank={{FAIL:0,WARN:1,PASS:2,'N/A':3}},ordered=[...rows].sort((a,b)=>(rank[a.status]??9)-(rank[b.status]??9));return ordered.map(q=>`<tr><td>${{auditStatus(q.status)}}</td><td>${{escapeHtml(q.category)}}</td><td>${{escapeHtml(q.name)}}</td><td>${{escapeHtml(q.observed)}}</td><td>${{escapeHtml(q.requirement)}}</td><td>${{q.gate?'正式门槛':'诊断'}}</td><td>${{escapeHtml(q.reason)}}</td></tr>`).join('')}}
const v=data.validation,summary=v.summary,banner=document.getElementById('validationBanner');banner.classList.add(summary.status.toLowerCase());banner.innerHTML=`${{auditStatus(summary.status)}}<span>${{summary.status==='PASS'?'全部正式目标与启用约束均通过。':`有 ${{summary.gated_failure_count}} 个正式门槛失败；先查看下方 FAIL 行。`}}</span>`;
document.getElementById('validationStats').innerHTML=stat('总体审核',auditStatus(summary.status))+stat('输入配置',summary.input_failure_count===0?'<span class="pass">可解析</span>':`<span class="fail">${{summary.input_failure_count}} FAIL</span>`,`${{summary.input_check_count}} 项检查`)+stat('观测审核失败',summary.observation_failure_count,`${{summary.observation_check_count}} 项审核`)+stat('z 贯通中心线',v.through_centerline_count,`全部中心线 ${{v.observations.find(x=>x.name==='中心线总数')?.value||0}}`);
document.getElementById('inputParameters').innerHTML=v.input_parameters.map((section,index)=>`<details ${{index<5?'open':''}}><summary>${{escapeHtml(section.label)}} <span class="muted">(${{section.rows.length}})</span></summary><div class="table-scroll"><table class="parameter-table"><thead><tr><th>参数路径</th><th>本次输入值</th></tr></thead><tbody>${{section.rows.map(row=>`<tr><td class="path">${{escapeHtml(row.path)}}</td><td>${{escapeHtml(row.value)}}</td></tr>`).join('')}}</tbody></table></div></details>`).join('');
document.getElementById('inputAuditRows').innerHTML=v.input_checks.map(q=>`<tr><td>${{auditStatus(q.status)}}</td><td>${{escapeHtml(q.name)}}</td><td>${{escapeHtml(q.observed)}}</td><td>${{escapeHtml(q.requirement)}}</td><td>${{escapeHtml(q.reason)}}</td></tr>`).join('');
document.getElementById('observationRows').innerHTML=v.observations.map(row=>`<tr><td>${{escapeHtml(row.name)}}</td><td>${{escapeHtml(row.value)}}</td><td class="muted">${{escapeHtml(row.source)}}</td></tr>`).join('');
document.getElementById('observationAuditRows').innerHTML=auditRows(v.observation_checks);
document.getElementById('warnings').innerHTML=v.warnings.length?v.warnings.map(w=>`<div class="warning">${{escapeHtml(w)}}</div>`).join(''):'<p class="pass">无审核警告</p>';
const candidates=data.optimization.candidates;document.getElementById('candidateRows').innerHTML=candidates.map(c=>`<tr><td>${{c.sequence}}${{c.selected?' ★':''}}</td><td>${{c.round}}/${{c.candidate}}</td><td>${{c.seed}}</td><td>${{c.porosity_error==null?'N/A':c.porosity_error.toFixed(5)}}</td><td>${{c.wall_time_seconds.toFixed(2)}} s</td><td>${{status(c.passed)}}</td></tr>`).join('');document.getElementById('seedPanel').textContent=data.optimization.configured_seeds.length?`配置的 seed panel：${{data.optimization.configured_seeds.join(', ')}}`:'本次未配置 seed panel。';
const generationProcess=data.optimization.generation_process;document.getElementById('generationProcess').innerHTML=generationProcess.map((step,index)=>`${{index?'<div class="process-arrow">→</div>':''}}<div class="process-step ${{step.status.toLowerCase()}}"><span class="muted">步骤 ${{index+1}}</span><strong>${{escapeHtml(step.name)}}</strong><div>${{auditStatus(step.status)}} · ${{step.seconds==null?'N/A':step.seconds.toFixed(3)+' s'}}</div><div class="muted">${{escapeHtml(step.detail)}}${{step.calls==null?'':`；调用 ${{step.calls}} 次`}}</div></div>`).join('');
function drawBars(canvas,values,labelFn,colorFn){{const [x,w,h]=fit(canvas);x.clearRect(0,0,w,h);if(!values.length)return;const pad=38,max=Math.max(...values.map(v=>v.value),1e-9),row=(h-2*pad)/values.length;values.forEach((v,i)=>{{const y=pad+i*row+row*.2,bw=(w-2*pad)*v.value/max;x.fillStyle=colorFn(v);x.fillRect(pad,y,bw,row*.55);x.fillStyle=css('--text');x.fillText(labelFn(v),pad+4,y+row*.4)}})}}
function drawCandidate(){{drawBars(document.getElementById('candidateChart'),candidates.map(c=>({{...c,value:c.porosity_error||0}})),c=>`#${{c.sequence}}  Δφ=${{c.porosity_error==null?'N/A':c.porosity_error.toFixed(5)}}`,c=>c.passed?css('--good'):css('--bad'))}}
const p=data.performance,seconds=v=>v==null?'N/A':Number(v).toFixed(2)+' s',memory=v=>v==null?'N/A':Number(v).toFixed(1)+' MiB';document.getElementById('performanceStats').innerHTML=stat('端到端耗时',seconds(p.total_wall_time_seconds),'截至报告写入前')+stat('候选搜索',seconds(p.candidate_search_wall_time_seconds),`${{p.candidate_count}} 个候选`)+stat('选中候选重放',seconds(p.selected_replay_wall_time_seconds),`worker 原始耗时 ${{seconds(p.selected_candidate_seconds)}}`)+stat('峰值 RSS',memory(p.peak_rss_mib),p.worker_peak_rss_mib==null?'主进程':`主进程；单 worker 最大 ${{memory(p.worker_peak_rss_mib)}}`);
document.getElementById('stageRows').innerHTML=p.stages.length?p.stages.map(s=>`<tr><td>${{escapeHtml(s.label)}}</td><td>${{seconds(s.seconds)}}</td><td>${{s.calls}}</td></tr>`).join(''):'<tr><td colspan="3" class="na">本次运行没有阶段计时</td></tr>';document.getElementById('performanceScope').textContent=p.measurement_scope||'阶段耗时来自选中候选的确定性重放及结果导出。';document.getElementById('candidateTimingNote').textContent=`候选累计 CPU-like 耗时 ${{seconds(p.candidate_wall_time_sum_seconds)}}；并行时不等于墙钟时间。体素规模 ${{p.voxel_shape_zyx.join('×')}}（${{p.voxel_count.toLocaleString()}}）。`;
function drawPerformance(){{drawBars(document.getElementById('stageChart'),p.stages.map(s=>({{...s,value:s.seconds}})),s=>`${{s.label}}  ${{seconds(s.seconds)}}`,()=>css('--accent'));drawBars(document.getElementById('performanceChart'),candidates.map(c=>({{...c,value:c.wall_time_seconds}})),c=>`候选 #${{c.sequence}}  ${{c.wall_time_seconds.toFixed(2)}} s`,c=>c.selected?css('--accent'):css('--pore'))}}
function redraw(){{drawScene();drawSlice();const active=document.querySelector('nav button[aria-selected=true]').dataset.tab;if(active==='optimization')drawCandidate();if(active==='performance')drawPerformance()}}window.addEventListener('resize',redraw);drawScene();drawSlice();
</script>
</body>
</html>
"""


__all__ = ["write_visual_report"]
