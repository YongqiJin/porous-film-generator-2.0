from pathlib import Path

from porous_film.config import load_config
from porous_film.parallel import ResourceSnapshot, build_execution_plan, discover_resources


def test_linux_affinity_counts_sibling_groups(tmp_path: Path) -> None:
    root = tmp_path / "cpu"
    for cpu, siblings in {0: "0,8", 2: "2,10"}.items():
        path = root / f"cpu{cpu}" / "topology"
        path.mkdir(parents=True)
        (path / "thread_siblings_list").write_text(siblings, encoding="utf-8")
    snapshot = discover_resources(
        affinity={0, 2}, available_memory_bytes=8 * 1024**3,
        platform_name="linux", sysfs_cpu_root=root,
    )
    assert snapshot.allowed_logical_cpus == (0, 2)
    assert snapshot.available_physical_cores == 2
    assert snapshot.physical_core_source == "linux-thread-siblings"


def test_memory_is_clamped_by_audit_cap() -> None:
    snapshot = discover_resources(
        affinity={0, 1}, available_memory_bytes=16 * 1024**3,
        audit_memory_cap_bytes=2 * 1024**3,
        platform_name="windows", sysfs_cpu_root=Path("unused"),
    )
    assert snapshot.available_memory_bytes == 16 * 1024**3
    assert snapshot.effective_memory_bytes == 2 * 1024**3


def test_auto_prefers_seeds(sample_config_path: Path) -> None:
    config = load_config(sample_config_path)
    plan = build_execution_plan(
        config, command="generate", seed_task_count=4, candidate_task_count=8,
        resources=ResourceSnapshot.for_test(16, 64 * 1024**3),
        estimated_worker_memory_bytes=512 * 1024**2,
    )
    assert plan.effective_strategy == "seeds"
    assert plan.worker_count == 4


def test_tightest_worker_limit_wins(sample_config_path: Path) -> None:
    config = load_config(sample_config_path)
    parallel = config.parallel.model_copy(
        update={"max_workers": 7, "cpu_fraction": 0.5, "memory_fraction": 0.5}
    )
    plan = build_execution_plan(
        config.model_copy(update={"parallel": parallel}),
        command="generate", seed_task_count=20, candidate_task_count=4,
        resources=ResourceSnapshot.for_test(12, 3 * 1024**3),
        estimated_worker_memory_bytes=512 * 1024**2,
    )
    assert plan.worker_limits == {"tasks": 20, "user": 7, "cpu": 6, "memory": 3}
    assert plan.worker_count == 3
    assert plan.limiting_factors == ("memory",)
