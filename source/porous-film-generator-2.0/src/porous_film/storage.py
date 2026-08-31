from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_RESULT_TYPE = "python_results"
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_TASK_SUBDIRS = ("inputs", "work", "outputs", "analysis", "reports", "logs", "qa_export")


@dataclass(frozen=True)
class TaskPaths:
    root: Path
    inputs: Path
    work: Path
    outputs: Path
    analysis: Path
    reports: Path
    logs: Path
    qa_export: Path


def create_task_directory(result_root: Path, task_name: str, now: datetime) -> TaskPaths:
    """Create a dated, non-overwriting porous-film task directory."""
    if not isinstance(now, datetime):
        raise TypeError("now must be a datetime")

    local_now = now if now.tzinfo is not None else now.replace(tzinfo=_SHANGHAI)
    date = local_now.astimezone(_SHANGHAI).date().isoformat()
    base = Path(result_root) / date / _RESULT_TYPE
    root = _unique_task_root(base, _slug(task_name))
    paths = {name: root / name for name in _TASK_SUBDIRS}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=False)
    return TaskPaths(
        root=root,
        inputs=paths["inputs"],
        work=paths["work"],
        outputs=paths["outputs"],
        analysis=paths["analysis"],
        reports=paths["reports"],
        logs=paths["logs"],
        qa_export=paths["qa_export"],
    )


def create_task_paths_at_root(root: Path) -> TaskPaths:
    """Create a complete task directory at the exact requested root."""
    task_root = Path(root)
    task_root.mkdir(parents=True, exist_ok=False)
    children = {name: task_root / name for name in _TASK_SUBDIRS}
    for child in children.values():
        child.mkdir(parents=True, exist_ok=False)
    return TaskPaths(root=task_root, **children)


def task_paths_from_existing_root(root: Path) -> TaskPaths:
    """Return task paths for an already-created complete task layout."""
    task_root = Path(root)
    missing = [name for name in _TASK_SUBDIRS if not (task_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"missing task directories under {task_root}: {', '.join(missing)}"
        )
    children = {name: task_root / name for name in _TASK_SUBDIRS}
    return TaskPaths(root=task_root, **children)


def _unique_task_root(base: Path, task_slug: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    first = base / task_slug
    if not first.exists():
        return first
    suffix = 2
    while True:
        candidate = base / f"{task_slug}-{suffix:02d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _slug(name: str) -> str:
    slug = "".join(
        character.lower() if character.isalnum() else "-"
        for character in str(name).strip()
    ).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    if not slug:
        raise ValueError("task_name must contain at least one alphanumeric character")
    return slug
