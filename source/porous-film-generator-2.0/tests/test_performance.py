from __future__ import annotations


def test_runtime_profiler_records_nested_exclusive_stage_time() -> None:
    from porous_film.performance import RuntimeProfiler, activate_runtime_profiler, profile_stage

    profiler = RuntimeProfiler()
    with activate_runtime_profiler(profiler), profile_stage("validation"):
        sum(range(1_000))
        with profile_stage("final_measurement"):
            sum(range(1_000))

    snapshot = profiler.snapshot()

    assert snapshot.wall_time_seconds >= 0.0
    assert snapshot.peak_rss_mib > 0.0
    assert snapshot.stage_call_counts == {"final_measurement": 1, "validation": 1}
    assert snapshot.stage_inclusive_seconds["validation"] >= snapshot.stage_seconds["validation"]
    assert snapshot.stage_seconds["final_measurement"] >= 0.0
