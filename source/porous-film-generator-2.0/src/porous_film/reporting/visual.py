from __future__ import annotations

import base64
import html
import json
from collections.abc import Sequence
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


def write_visual_report(
    path: Path,
    *,
    config: GeneratorConfig,
    built: BuiltGeometry,
    grid: PhaseGrid,
    audit: AuditResult,
    candidates: Sequence[CandidateResult],
    selected_sequence_index: int,
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
    )
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
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
) -> dict[str, Any]:
    measurements = audit.formal_measurements
    target_porosity = float(config.formal_targets.proportion.porosity)
    through_ids = (
        {track.track_id for track in measurements.centerlines if track.is_through}
        if measurements is not None
        else set()
    )
    ordered_candidates = sorted(candidates, key=lambda value: value.sequence_index)
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
            "distributions": _distribution_payload(config, audit, through_ids),
            "rdf": audit.center_distance_xy_result,
            "warnings": list(audit.warnings),
        },
        "optimization": {
            "candidates": candidate_rows,
            "configured_seeds": []
            if config.optimization.seed_panel is None
            else [int(value) for value in config.optimization.seed_panel],
        },
        "performance": {
            "candidate_wall_time_sum_seconds": float(
                sum(value.wall_time_seconds for value in ordered_candidates)
            ),
            "selected_candidate_seconds": next(
                (
                    float(value.wall_time_seconds)
                    for value in ordered_candidates
                    if int(value.sequence_index) == int(selected_sequence_index)
                ),
                None,
            ),
            "candidate_count": len(ordered_candidates),
            "voxel_shape_zyx": list(grid.pore_mask.shape),
            "voxel_count": int(grid.pore_mask.size),
            "peak_rss_mib": None,
            "stage_timings_available": False,
        },
    }


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
            "name": "z 贯通轨迹",
            "passed": through_count > 0,
            "value": str(through_count),
            "limit": "> 0（诊断门）",
        },
        {
            "name": "半导体 x 贯通",
            "passed": not x_warning,
            "value": "通过" if not x_warning else "失败",
            "limit": "必须通过" if config.matrix_constraints.require_x_percolation else "未强制",
        },
        {
            "name": "最小 yz 截面",
            "passed": float(audit.minimum_cross_section_fraction)
            >= float(config.matrix_constraints.minimum_cross_section_fraction),
            "value": _format_number(audit.minimum_cross_section_fraction),
            "limit": f">= {_format_number(config.matrix_constraints.minimum_cross_section_fraction)}",
        },
        {
            "name": "最大重叠比例",
            "passed": float(audit.overlap_fraction)
            <= float(config.matrix_constraints.maximum_overlap_fraction),
            "value": _format_number(audit.overlap_fraction),
            "limit": f"<= {_format_number(config.matrix_constraints.maximum_overlap_fraction)}",
        },
        {
            "name": "粗细网格稳定性",
            "passed": bool(thickness.passed),
            "value": _format_number(thickness.max_quantile_error_A) + " Å",
            "limit": f"<= {_format_number(thickness.tolerance_A)} Å",
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
    observed: dict[str, np.ndarray] = {
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
        "等效直径": shape.equivalent_diameter_A,
        "theta_xz": _orientation_marginal(shape.orientation, "theta_xz_deg"),
        "theta_xy": _orientation_marginal(shape.orientation, "theta_xy_deg"),
        "通道 eta": shape.channel_aspect_ratio,
        "通道 tau": shape.channel_tortuosity,
        "曲率波动": shape.curvature_fluctuation,
    }
    result_names = {
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
                "target": _finite_list(target_values),
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
  <section id="validation" hidden><div class="stats" id="validationStats"></div><div class="grid" id="validationCharts"></div><div class="panel" style="margin-top:16px"><h2>约束与失败原因</h2><div id="constraints"></div><div id="warnings"></div></div></section>
  <section id="optimization" hidden><div class="panel"><h2>候选搜索</h2><canvas id="candidateChart" class="chart"></canvas><div style="overflow:auto"><table><thead><tr><th>序号</th><th>轮次/候选</th><th>seed</th><th>孔隙率误差</th><th>耗时</th><th>结果</th></tr></thead><tbody id="candidateRows"></tbody></table></div><p class="muted" id="seedPanel"></p></div></section>
  <section id="performance" hidden><div class="stats" id="performanceStats"></div><div class="panel"><h2>候选耗时</h2><canvas id="performanceChart" class="chart"></canvas><p class="muted">阶段级耗时和峰值 RSS 尚未由当前运行时记录；此页会明确标为不可用，不使用估算值代替实测值。</p></div></section>
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
const v=data.validation;document.getElementById('validationStats').innerHTML=stat('总体审核',status(data.meta.audit_passed))+stat('孔隙率',v.porosity.actual.toFixed(4),`目标 ${{v.porosity.target.toFixed(4)}}`)+stat('贯通中心线',v.through_centerline_count)+stat('分支事件',v.branch_event_count);
document.getElementById('constraints').innerHTML=`<table><thead><tr><th>约束</th><th>结果</th><th>实测</th><th>门槛</th></tr></thead><tbody>${{v.constraints.map(q=>`<tr><td>${{q.name}}</td><td>${{status(q.passed)}}</td><td>${{q.value}}</td><td>${{q.limit}}</td></tr>`).join('')}}</tbody></table>`;
document.getElementById('warnings').innerHTML=v.warnings.length?v.warnings.map(w=>`<div class="warning">${{escapeHtml(w)}}</div>`).join(''):'<p class="pass">无审核警告</p>';
function escapeHtml(s){{return String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}
function hist(values,min,max,bins=20){{const out=Array(bins).fill(0),span=Math.max(max-min,1e-12);for(const v of values)out[Math.min(bins-1,Math.max(0,Math.floor((v-min)/span*bins)))]++;const sum=out.reduce((a,b)=>a+b,0)||1;return out.map(x=>x/sum)}}
function drawDistribution(container,row){{const panel=document.createElement('div');panel.className='panel';panel.innerHTML=`<h2>${{row.name}} · ${{status(row.passed)}}</h2><div class="muted">n=${{row.observed.length}} · KS=${{row.ks==null?'N/A':row.ks.toFixed(3)}} · W=${{row.wasserstein==null?'N/A':row.wasserstein.toFixed(3)}}</div><canvas class="chart"></canvas>`;container.appendChild(panel);const c=panel.querySelector('canvas'),[x,w,h]=fit(c),all=row.target.concat(row.observed);if(!all.length)return;let min=Math.min(...all),max=Math.max(...all);if(min===max){{min-=.5;max+=.5}}const a=hist(row.target,min,max),b=hist(row.observed,min,max),pad=34,bw=(w-2*pad)/a.length,scale=(h-2*pad)/Math.max(...a,...b,1e-9);x.strokeStyle=css('--border');x.strokeRect(pad,pad,w-2*pad,h-2*pad);for(let i=0;i<a.length;i++){{x.fillStyle=css('--accent');x.globalAlpha=.35;x.fillRect(pad+i*bw,h-pad-a[i]*scale,bw*.9,a[i]*scale);x.fillStyle=css('--other');x.globalAlpha=.65;x.fillRect(pad+i*bw+bw*.2,h-pad-b[i]*scale,bw*.55,b[i]*scale)}}x.globalAlpha=1;x.fillStyle=css('--muted');x.fillText(min.toPrecision(4),pad,h-10);x.textAlign='right';x.fillText(max.toPrecision(4),w-pad,h-10);x.textAlign='left'}}
const chartRoot=document.getElementById('validationCharts');v.distributions.forEach(row=>drawDistribution(chartRoot,row));
const candidates=data.optimization.candidates;document.getElementById('candidateRows').innerHTML=candidates.map(c=>`<tr><td>${{c.sequence}}${{c.selected?' ★':''}}</td><td>${{c.round}}/${{c.candidate}}</td><td>${{c.seed}}</td><td>${{c.porosity_error==null?'N/A':c.porosity_error.toFixed(5)}}</td><td>${{c.wall_time_seconds.toFixed(2)}} s</td><td>${{status(c.passed)}}</td></tr>`).join('');document.getElementById('seedPanel').textContent=data.optimization.configured_seeds.length?`配置的 seed panel：${{data.optimization.configured_seeds.join(', ')}}`:'本次未配置 seed panel。';
function drawBars(canvas,values,labelFn,colorFn){{const [x,w,h]=fit(canvas);x.clearRect(0,0,w,h);if(!values.length)return;const pad=38,max=Math.max(...values.map(v=>v.value),1e-9),row=(h-2*pad)/values.length;values.forEach((v,i)=>{{const y=pad+i*row+row*.2,bw=(w-2*pad)*v.value/max;x.fillStyle=colorFn(v);x.fillRect(pad,y,bw,row*.55);x.fillStyle=css('--text');x.fillText(labelFn(v),pad+4,y+row*.4)}})}}
function drawCandidate(){{drawBars(document.getElementById('candidateChart'),candidates.map(c=>({{...c,value:c.porosity_error||0}})),c=>`#${{c.sequence}}  Δφ=${{c.porosity_error==null?'N/A':c.porosity_error.toFixed(5)}}`,c=>c.passed?css('--good'):css('--bad'))}}
const p=data.performance;document.getElementById('performanceStats').innerHTML=stat('候选总耗时',p.candidate_wall_time_sum_seconds.toFixed(2)+' s')+stat('选中候选耗时',p.selected_candidate_seconds==null?'N/A':p.selected_candidate_seconds.toFixed(2)+' s')+stat('体素数',p.voxel_count.toLocaleString())+stat('峰值内存','N/A','等待阶段计时接入');
function drawPerformance(){{drawBars(document.getElementById('performanceChart'),candidates.map(c=>({{...c,value:c.wall_time_seconds}})),c=>`候选 #${{c.sequence}}  ${{c.wall_time_seconds.toFixed(2)}} s`,c=>c.selected?css('--accent'):css('--pore'))}}
function redraw(){{drawScene();drawSlice();const active=document.querySelector('nav button[aria-selected=true]').dataset.tab;if(active==='optimization')drawCandidate();if(active==='performance')drawPerformance()}}window.addEventListener('resize',redraw);drawScene();drawSlice();
</script>
</body>
</html>
"""


__all__ = ["write_visual_report"]
