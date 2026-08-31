# Final Pore Parameter Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace generation-unit-facing formal targets with the approved position-quantity, shape,
and proportion contract, and audit those targets from the final phase geometry.

**Architecture:** Add a schema-v3 target layer and a separate internal-generation layer. Introduce a
standalone final-phase measurement module that extracts per-slice pore centers, centerline tracks,
normal cross sections, equivalent diameters, paired projected orientations, channel eta/tau, and
curvature fluctuations. The existing audit becomes a comparator over immutable measurements, while
the validator independently recomputes the same physical quantities from exported phase data.

**Tech Stack:** Python 3.12+, NumPy, SciPy, scikit-image, Pydantic v2, h5py, pytest, Ruff.

## Global Constraints

- Work only in `C:\Calculation_assist\porous-film-generator\.worktrees\complex-shape-generation-v1`.
- Preserve `original-v0.2.0`, `main`, and existing deployments.
- Standard geometry uses only `target_box_A`; no padding or redundant packing box.
- Formal parameters are grouped as position-quantity, shape, and proportion.
- Audit measurements are computed only from the final phase geometry.
- Legacy generation metadata remains diagnostic and cannot satisfy a formal target.
- Use TDD for every production behavior change.
- Store generated calculation examples under `C:\Calculation_results`, not in the repository.

---

### Task 1: Add the schema-v3 formal target contract

**Files:**
- Modify: `src/porous_film/config/models.py`
- Modify: `src/porous_film/config/__init__.py`
- Modify: `tests/test_config.py`
- Modify: `tests/conftest.py`
- Modify: `tests/fixtures/minimal_config.yaml`

**Interfaces:**
- Produces `PositionQuantityTargets`, `ShapeTargets`, `ProportionTargets`,
  `GenerationControls`, and `MeasurementSpec`.
- `GeneratorConfig.formal_targets` exposes the three approved dimensions.
- Legacy fields are translated only through a compatibility path.

- [ ] Write failing tests for target-only `film.target_box_A`, grouped target validation, paired
  Beta orientation components, bounded equivalent-diameter/curvature distributions, and legacy
  compatibility metadata.
- [ ] Run the focused config tests and confirm failures are caused by missing schema-v3 behavior.
- [ ] Implement the minimal schema and compatibility translation.
- [ ] Run config tests and the existing config/payload tests.
- [ ] Refactor names and serialization while keeping tests green.

### Task 2: Sample paired orientations and physical diameter/curvature targets

**Files:**
- Modify: `src/porous_film/distributions/core.py`
- Modify: `src/porous_film/geometry/sdf.py`
- Modify: `src/porous_film/geometry/complex_shapes.py`
- Modify: `src/porous_film/geometry/scaling.py`
- Modify: `tests/test_distributions.py`
- Modify: `tests/test_geometry.py`
- Modify: `tests/test_complex_shapes.py`

**Interfaces:**
- Produces deterministic paired `(theta_xz, theta_xy)` samples.
- Constructs one normalized three-dimensional axis from each pair.
- Uses sampled absolute equivalent diameter to size channel cross sections.
- Maps target curvature fluctuation to an internal roughness starting value and preserves the target
  value as diagnostic provenance only.

- [ ] Write failing deterministic sampling and axis-reconstruction tests.
- [ ] Verify RED.
- [ ] Implement exact mixture-count allocation and paired sampling.
- [ ] Write failing tests for equivalent-diameter-controlled channel construction and scaling.
- [ ] Implement physical-diameter construction and update scale semantics.
- [ ] Run geometry/distribution tests and refactor.

### Task 3: Extract pore centers and centerline tracks from the final phase

**Files:**
- Create: `src/porous_film/metrics/final_geometry.py`
- Modify: `src/porous_film/metrics/__init__.py`
- Create: `tests/test_final_geometry_measurement.py`

**Interfaces:**
- `measure_final_geometry(grid: PhaseGrid, contract: MeasurementSpec) -> FinalGeometryMeasurements`
- Returns per-slice centers, centerline tracks, through-connectivity, invalid-sample diagnostics,
  and measurement-resolution metadata.

- [ ] Write synthetic-mask tests for one straight through pore, two periodic-edge pores, tilted
  tracks, a branch/merge neighborhood, and disconnected cavities.
- [ ] Verify RED.
- [ ] Implement periodic 2-D distance transforms and deterministic local-center extraction.
- [ ] Implement adjacent-slice assignment, periodic unwrapping, track filtering, and branch flags.
- [ ] Run focused tests and refactor.

### Task 4: Measure g_xy(r) from final slice centers

**Files:**
- Modify: `src/porous_film/metrics/final_geometry.py`
- Modify: `src/porous_film/centers/generation.py`
- Modify: `tests/test_final_geometry_measurement.py`
- Modify: `tests/test_centers.py`

**Interfaces:**
- Produces absolute-distance bins, observed pair counts, deterministic same-count random reference,
  and normalized `g_xy(r)`.
- Generation may use the same target as a heuristic, but only the final-phase measurement is audited.

- [ ] Write failing periodic-distance and uniform-reference tests.
- [ ] Verify RED.
- [ ] Implement deterministic same-slice pair aggregation and reference normalization.
- [ ] Update center-generation target evaluation from reduced 3-D distance to absolute periodic xy
  distance without changing the measurement implementation.
- [ ] Run center and measurement tests.

### Task 5: Measure normal cross sections, equivalent diameter, and curvature fluctuation

**Files:**
- Modify: `src/porous_film/metrics/final_geometry.py`
- Modify: `tests/test_final_geometry_measurement.py`

**Interfaces:**
- Produces `CrossSectionMeasurement` records containing track id, arc position, area,
  `D_eq`, curvature fluctuation, and validity reason.
- Uses one shared set of valid normal sections for diameter and curvature distributions.

- [ ] Write failing tests using a straight cylinder, tilted cylinder, elliptical cylinder, and a
  deliberately corrugated tube.
- [ ] Verify RED.
- [ ] Implement tangent frames, periodic trilinear phase sampling, component selection, contour
  resampling, fixed-scale smoothing, area integration, and curvature calculation.
- [ ] Add branch/surface exclusion and invalid-sample reporting.
- [ ] Run focused tests and refactor.

### Task 6: Measure paired orientation and channel eta/tau

**Files:**
- Modify: `src/porous_film/metrics/final_geometry.py`
- Modify: `tests/test_final_geometry_measurement.py`

**Interfaces:**
- Produces `(theta_xz, theta_xy)` with per-plane identifiability flags.
- Produces `D_channel = 2 * sqrt(mean(A_perp) / pi)`, `eta_ch`, and `tau` for each valid track
  segment.

- [ ] Write failing tests for vertical-x-biased, vertical-y-biased, tilted, and nearly pure-z tracks.
- [ ] Verify RED.
- [ ] Implement projected-angle and identifiability calculations.
- [ ] Implement channel-level diameter, arc length, endpoint distance, eta, and tau.
- [ ] Run focused tests and refactor.

### Task 7: Separate final measurement from target comparison

**Files:**
- Modify: `src/porous_film/metrics/audit.py`
- Modify: `src/porous_film/parallel/candidates.py`
- Modify: `src/porous_film/pipeline.py`
- Modify: `tests/test_metrics.py`
- Modify: `tests/test_parallel_candidates.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- `audit_target_distributions` compares formal targets to `FinalGeometryMeasurements`.
- Generation-unit comparisons move to a clearly named diagnostics section and cannot set formal
  distribution pass/fail.

- [ ] Write failing tests showing that changing generator-written unit metrics cannot change final
  measurements or formal audit results.
- [ ] Verify RED.
- [ ] Compare final porosity, `g_xy`, diameter, paired orientation, eta/tau, and curvature targets.
- [ ] Preserve overlap and matrix hard constraints as separate gates.
- [ ] Update candidate selection and deterministic replay checks.
- [ ] Run metric, candidate, and pipeline tests.

### Task 8: Export schema-v3 evidence and independently validate it

**Files:**
- Modify: `src/porous_film/pipeline.py`
- Modify: `src/porous_film_validator/validate.py`
- Modify: `tests/test_validator.py`
- Modify: `tests/test_end_to_end.py`

**Interfaces:**
- Adds final-centerline and normal-cross-section HDF5/CSV evidence.
- Keeps v1/v2 unit-geometry input readable but labels it generation provenance.
- Independent validator recomputes principal final-phase measurements without importing
  `porous_film`.

- [ ] Write failing schema-v3 export and validator-independence tests.
- [ ] Verify RED.
- [ ] Implement artifacts and independent recomputation.
- [ ] Run validator/end-to-end tests and refactor.

### Task 9: Remove padding workflow and obsolete molecule-facing defaults

**Files:**
- Modify: `src/porous_film/pipeline.py`
- Modify: `src/porous_film/molecules/packing.py`
- Modify: `src/porous_film/cli.py`
- Modify: `tests/test_packing.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_cli_smoke.py`

**Interfaces:**
- Geometry-only phase-field runs require no PDB and no redundant box.
- Legacy molecule-placement flow is either explicit compatibility mode or rejected with a clear
  migration message; it cannot silently reintroduce padding.

- [ ] Write failing geometry-only configuration and output tests.
- [ ] Verify RED.
- [ ] Implement target-box-only paths and compatibility behavior.
- [ ] Run packing, pipeline, and CLI tests.

### Task 10: Version, documentation, and release verification

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `skills/porous-film-generator/SKILL.md`
- Modify: `skills/porous-film-generator/references/input-schema.md`
- Modify: `skills/porous-film-generator/references/generation-workflow.md`
- Modify: `skills/porous-film-generator/references/output-artifacts.md`
- Modify: relevant tests under `tests/`

**Interfaces:**
- Version becomes `0.4.0.dev1`.
- Documentation distinguishes formal targets, measurement-contract settings, and internal controls.

- [ ] Update contract tests first and verify RED.
- [ ] Update version and documentation.
- [ ] Run focused documentation/deployment tests.
- [ ] Run Ruff and the complete test suite with a task-owned `--basetemp`.
- [ ] Generate one low-resolution through-pore example under `C:\Calculation_results` and run the
  independent validator.
- [ ] Review `git diff`, confirm no calculation artifacts entered the repository, and commit the
  experimental branch.

## Known baseline issues

Before this plan, Ruff passed. Six unrelated tests failed locally: five deployment-release tests
because nested Windows PowerShell could not resolve `Get-FileHash`, and one console-script test
because `porous-film-validate` was not installed on PATH. Feature verification must report these
environmental failures separately unless they are resolved by the normal editable-install setup.

