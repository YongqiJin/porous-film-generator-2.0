# Troubleshooting

## Preflight and configuration

### Zero pore seeds

Cause: `round(number_density * target volume)` is zero. Increase number density or target volume; do not bypass the check.

### Grid divisibility error

Choose spacing that divides x, y, and z exactly. For 150 nm thickness, 20 nm does not divide the thickness; use a compatible spacing such as 10 nm or change the box intentionally.

### Invalid mixture or distribution

Mixture weights must sum to 1, required family parameters must exist, support must be valid, and eta/tau/aspect ratio must start at 1. Tabulated inputs are unsupported.

## Performance

### Memory warning or worker reduction

Preflight estimates grid memory; worker planning also considers CPU affinity and effective memory. Smaller spacing increases voxel count cubically for an isotropic box. Reduce workers, use a coarser preview, reduce candidate concurrency, or use sufficient memory.

### Slow channel voxelization

Curved channels are slower than compact pores because every voxel query evaluates distances to many centerline segments and periodic images. Cost grows with voxel count, channel count, eta/length, roughness, scale-search iterations, candidates, and replay.

Use a labeled coarse preview first, keep production resolution for final acceptance, and inspect Python spawn workers rather than searching only for `porous_film`. Candidate progress may appear only when a candidate finishes.

## Audit results

### `completed_infeasible`

Geometry exists, but strict acceptance failed. Read `feasibility.json`, `realized_geometry_parameters.json`, candidate JSONL, and reports. Process success is not scientific feasibility.

### Distribution mismatch

Small samples, boundary clipping, overlap, and candidate selection shift realized volume, eta, tau, orientation, or roughness distributions. Inspect KS and normalized Wasserstein results.

### Excessive overlap

Reduce porosity or upper relative-volume tail, increase seed count while controlling total volume, strengthen center exclusion, or reduce channel volume ratio. RDF exclusion is not an absolute no-overlap constraint.

### Matrix does not percolate

A “does not percolate” warning or failed cross-section gate requires reduced merging/porosity, changed center statistics, reduced channel dominance, increased matrix thickness, or another seed. Do not silently disable production constraints.

### RDF loss is high

Increase seed support, simplify contradictory RDF terms, generate more candidates, or treat RDF as diagnostic when pair distances are correlated or finite-z effects dominate.

## Missing outputs

### Missing GLB

Possible causes: only preflight ran; a candidate is still running; strict geometry was infeasible; preview and full audit were confused; or remote outputs were not retrieved. Check process trees, `work/parallel`, reports, status, submission state, and transfer logs.

### Preview versus full audit

A preview can have a GLB while a separate full audit is still running. Record distinct task names, resolution, timing, and roots. Preview completion requires declared outputs and verification; full audit completion requires status, feasibility, reports, QA export, and independent validation.

## Molecule placement and Packmol

Packing failures usually indicate inaccessible pore volume, oversized molecules, excessive requested amount, or insufficient attempts. External Packmol success does not prove confinement; run `porous-film audit-packmol-output`.

## Reproducibility failures

If replay or serial/parallel comparisons differ, compare normalized config, identities, package version, platform, numeric thread limits, GLB occupancy, and scientific JSON/HDF5. Timing, process IDs, run IDs, and raw binary encoding may differ without scientific change.

## Complex shape diagnostics

### Shape generation candidate failure

A v2 **shape generation** failure means all 16 internal attempts violated connectivity, envelope fill, radius CV, bend/nonplanarity, or tube self-intersection constraints. The generator intentionally does not replace that unit with a regular fallback; another candidate/seed or a feasible requested `eta/tau/volume` regime is required.

### Radius CV, neck, or bulge failure

Inspect `main_unit_metrics.csv`, `unit_geometry.jsonl`, and `shape_complexity_summary`. A generated channel must keep radius CV in 0.15-0.30, local radius within 0.60-1.45 of the equivalent radius, and include both a neck and a bulge.

### Self-intersection or nonplanarity failure

For `tau > 1.05`, channels require at least two bends and nonzero nonplanarity. Negative `minimum_self_clearance_A` indicates non-adjacent tube self-intersection and invalidates the candidate.

### Independent validator rejects v2

Confirm the validator reports `0.3.0.dev1`, `channel_curves.h5` contains per-channel groups with `centerline_A` and `radius_A`, and `unit_geometry.jsonl` contains `shape_model` and `shape_seed`. Do not repair exported metrics manually; regenerate the artifact set and rerun the **independent validator**.
