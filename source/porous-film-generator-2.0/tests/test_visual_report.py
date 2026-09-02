from __future__ import annotations

from pathlib import Path


def test_visual_report_contains_all_four_pages(sample_config_path: Path, tmp_path: Path) -> None:
    from porous_film.config import load_config
    from porous_film.parallel import (
        build_candidate_tasks,
        evaluate_candidate_task,
        replay_candidate,
    )
    from porous_film.reporting.visual import write_visual_report

    config = load_config(sample_config_path)
    task = build_candidate_tasks(config)[0]
    candidate = evaluate_candidate_task(task)
    artifacts = replay_candidate(config, task.identity)
    assert candidate.performance is not None
    assert {
        "center_seed_generation",
        "shape_generation",
        "voxelization",
        "validation",
    } <= candidate.performance.stage_seconds.keys()
    output = write_visual_report(
        tmp_path / "outputs" / "visual-report" / "index.html",
        config=config,
        built=artifacts.built,
        grid=artifacts.phase_grid,
        audit=artifacts.audit,
        candidates=(candidate,),
        selected_sequence_index=task.sequence_index,
        performance={
            "total_wall_time_seconds": 12.5,
            "candidate_search_wall_time_seconds": 7.0,
            "selected_replay_wall_time_seconds": 4.0,
            "peak_rss_mib": 128.0,
            "worker_peak_rss_mib": 96.0,
            "stage_timings_seconds": {
                "center_seed_generation": 0.2,
                "centerline_generation": 0.4,
                "shape_generation": 0.8,
                "voxelization": 1.2,
                "final_measurement": 0.7,
                "validation": 0.5,
                "export": 0.9,
            },
            "stage_call_counts": {
                "center_seed_generation": 1,
                "centerline_generation": 1,
                "shape_generation": 8,
                "voxelization": 9,
                "final_measurement": 1,
                "validation": 1,
                "export": 1,
            },
        },
    )

    report = output.read_text(encoding="utf-8")
    assert output.exists()
    assert "Geometry" in report
    assert "Validation" in report
    assert "Optimization" in report
    assert "Performance" in report
    assert "reportData" in report
    assert "pore_points" in report
    assert "centerlines" in report
    assert '\"total_wall_time_seconds\":12.5' in report
    assert '\"centerline_generation\":0.4' in report
    assert "stageChart" in report
    assert "峰值 RSS" in report
    assert "Infinity" not in report
    assert '"input_parameters"' in report
    assert '"input_checks"' in report
    assert '"observations"' in report
    assert '"observation_checks"' in report
    assert '"generation_process"' in report
    assert "输入参数（归一化）" in report
    assert "输入配置审核" in report
    assert "最终观测" in report
    assert "输入 → 观测逐项审核" in report
    assert "生成过程" in report
    assert "generationProcess" in report
