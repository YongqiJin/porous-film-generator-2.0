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
    output = write_visual_report(
        tmp_path / "outputs" / "visual-report" / "index.html",
        config=config,
        built=artifacts.built,
        grid=artifacts.phase_grid,
        audit=artifacts.audit,
        candidates=(candidate,),
        selected_sequence_index=task.sequence_index,
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
