from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from porous_film.storage import (
    create_task_directory,
    create_task_paths_at_root,
    task_paths_from_existing_root,
)


def test_task_directory_uses_shanghai_date_and_unique_suffix(tmp_path: Path) -> None:
    now = datetime(2026, 8, 12, 23, 30, tzinfo=ZoneInfo("Asia/Shanghai"))

    first = create_task_directory(tmp_path, "demo", now)
    second = create_task_directory(tmp_path, "demo", now)

    assert first.root == tmp_path / "2026-08-12" / "python_results" / "demo"
    assert second.root == tmp_path / "2026-08-12" / "python_results" / "demo-02"
    assert first.outputs.is_dir()
    assert first.analysis.is_dir()
    assert first.reports.is_dir()


def test_create_task_paths_at_root_uses_exact_requested_root(tmp_path: Path) -> None:
    root = tmp_path / "exact-root"

    paths = create_task_paths_at_root(root)

    assert paths.root == root
    assert paths.inputs == root / "inputs"
    assert paths.work == root / "work"
    assert paths.outputs == root / "outputs"
    assert paths.analysis == root / "analysis"
    assert paths.reports == root / "reports"
    assert paths.logs == root / "logs"
    assert paths.qa_export == root / "qa_export"


def test_task_paths_from_existing_root_requires_complete_layout(
    tmp_path: Path,
) -> None:
    root = tmp_path / "partial"
    root.mkdir()
    (root / "inputs").mkdir()

    with pytest.raises(FileNotFoundError, match="outputs"):
        task_paths_from_existing_root(root)

    assert not (root / "outputs").exists()


def test_seed_panel_paths_keep_optimizer_exchange_at_seed_root(
    tmp_path: Path,
) -> None:
    from porous_film.pipeline import _seed_panel_task_paths

    root = tmp_path / "seed-panel" / "11"

    paths, optimizer_output_dir = _seed_panel_task_paths(root)

    assert paths.root == root
    assert paths.outputs == root / "outputs"
    assert paths.qa_export == root / "qa_export"
    assert optimizer_output_dir == root
    assert paths.outputs.is_dir()
    assert paths.qa_export.is_dir()
