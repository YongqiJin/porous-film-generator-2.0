from porous_film.parallel.candidates import (
    CandidateArtifacts,
    CandidateIdentity,
    CandidateResult,
    CandidateTask,
    GeometryAcceptanceError,
    build_candidate_tasks,
    evaluate_candidate_task,
    evaluate_candidate_task_with_artifacts,
    replay_candidate,
    select_candidate,
)
from porous_film.parallel.overrides import apply_parallel_cli_overrides
from porous_film.parallel.planning import ParallelExecutionPlan, build_execution_plan
from porous_film.parallel.reporting import write_parallel_plan, write_parallel_summary
from porous_film.parallel.resources import (
    ResourceSnapshot,
    discover_resources,
    estimate_generation_memory_bytes,
    estimate_worker_memory_bytes,
)
from porous_film.parallel.runtime import (
    THREAD_ENVIRONMENT,
    ParallelCancelled,
    ParallelPoolError,
    ensure_pool_allowed,
    initialize_worker,
    numeric_thread_limits,
    run_spawn_tasks,
    spawn_pool,
)
from porous_film.parallel.seeds import (
    SeedIdentity,
    SeedTask,
    SeedTaskResult,
    build_seed_tasks,
    execute_seed_task,
)

__all__ = [
    "THREAD_ENVIRONMENT",
    "CandidateArtifacts",
    "CandidateIdentity",
    "CandidateResult",
    "CandidateTask",
    "GeometryAcceptanceError",
    "ParallelCancelled",
    "ParallelExecutionPlan",
    "ParallelPoolError",
    "ResourceSnapshot",
    "SeedIdentity",
    "SeedTask",
    "SeedTaskResult",
    "apply_parallel_cli_overrides",
    "build_candidate_tasks",
    "build_execution_plan",
    "build_seed_tasks",
    "discover_resources",
    "ensure_pool_allowed",
    "estimate_generation_memory_bytes",
    "estimate_worker_memory_bytes",
    "evaluate_candidate_task",
    "evaluate_candidate_task_with_artifacts",
    "execute_seed_task",
    "initialize_worker",
    "numeric_thread_limits",
    "replay_candidate",
    "run_spawn_tasks",
    "select_candidate",
    "spawn_pool",
    "write_parallel_plan",
    "write_parallel_summary",
]
