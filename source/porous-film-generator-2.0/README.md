# Porous Film Generator

Auditable 3D porous-film geometry generator for phase-field and structure-property workflows.

## Current experimental branch

Version 0.4.0.dev1 introduces schema v3. Formal parameters are grouped into:

- position-quantity: through-pore centerline xy-distance distribution;
- shape: local equivalent diameter, paired xz/xy orientation, compact/channel eta, channel tau, and
  local normal-section curvature fluctuation;
- proportion: final three-dimensional porosity.

Formal audit values are measured from the final phase. Generation-unit labels, relative volumes,
roughness amplitudes, and random shape seeds are diagnostics and cannot substitute for final
measurements.

The stable original behavior remains available from the immutable original-v0.2.0 tag.

## Knowledge skill

A server-neutral Codex knowledge skill is stored at
skills/porous-film-generator/SKILL.md.

## Install and run

    uv sync
    uv run porous-film version
    uv run porous-film preflight --config config.yaml --result-root C:\Calculation_results
    uv run porous-film generate-geometry --config config.yaml --result-root C:\Calculation_results
    uv run porous-film-validate <run>\qa_export

Schema-v3 phase-field generation needs only target_box_A. It does not require a PDB, packing_box_A,
or z_padding_A. Legacy molecule-packing inputs remain readable for compatibility.

## Formal schema-v3 outline

    schema_version: 3
    task:
      name: through-pore-study
      random_seed: 123
    film:
      target_box_A: {x: 10000, y: 10000, z: 1500}
    formal_targets:
      position_quantity:
        center_distance_xy:
          components:
            - {kind: exclusion, amplitude: 0.9, center_A: 0, width_A: 300}
            - {kind: peak, amplitude: 0.3, center_A: 900, width_A: 180}
      shape:
        equivalent_diameter_A:
          {family: beta, alpha: 2.5, beta: 3.5, lower: 300, upper: 900}
        orientation:
          model: paired_projected_planes
          components:
            - weight: 1.0
              theta_xz_deg:
                {family: beta, alpha: 4, beta: 2, lower: 55, upper: 88}
              theta_xy_deg:
                {family: beta, alpha: 2, beta: 5, lower: 0, upper: 30}
        channel_aspect_ratio:
          {family: beta, alpha: 2, beta: 2, lower: 3, upper: 12}
        channel_tortuosity:
          {family: beta, alpha: 2, beta: 3, lower: 1, upper: 1.8}
        curvature_fluctuation:
          {family: beta, alpha: 2, beta: 5, lower: 0, upper: 0.8}
      proportion:
        porosity: 0.15
    generation_controls:
      seed_number_density_A3: 8.0e-10
      channel_fraction_by_count: 1.0
      channel_to_compact_mean_volume_ratio: 1.0

The numerical values are syntax examples, not approved physical parameters.

## Output evidence

Geometry runs write the authoritative final phase, extracted centerlines, normal-section
measurements, GLB/PLY surfaces, normalized input, audit summaries, and checksums. Unit geometry and
channel curves remain available as generation provenance. See
skills/porous-film-generator/references/output-artifacts.md for the complete inventory.

When `output.write_plots` is true, each geometry run also writes a self-contained interactive report
to `outputs/visual-report/index.html`. It contains Geometry, Validation, Optimization, and
Performance pages and does not require a web service.

## Parallel execution

Candidate and seed multiprocessing are CPU-only. The generator itself does not use GPUs. Use
--workers to cap processes or --no-parallel for deterministic serial execution. Scientific results
for the same candidate identity must be identical between serial and parallel execution.

## Result storage

Use C:\Calculation_results\YYYY-MM-DD\python_results\<task-name>\ for calculation inputs,
outputs, analyses, reports, and logs. Do not store formal calculation outputs in the source tree.
