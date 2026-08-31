# Parameter recipes and design heuristics

These are starting patterns, not universal physical constants. Record every change and audit the realized geometry.

## Dispersed compact pores

Goal: separated irregular cavities without a dominant connected pore cluster.

- Set `channel_fraction_by_count: 0`.
- Use enough seeds that desired porosity is distributed across many units.
- Use an RDF `exclusion` component with high amplitude and adequate width.
- Keep RDF peaks weak if clustering is undesirable.
- Narrow `compact.relative_volume` to suppress extreme large pores.
- Lower porosity or increase seed count to reduce mean unit size.
- Audit overlap, largest pore-domain fraction, nearest centers, and matrix cross-section.

## Bicontinuous channels

Goal: a dominant curved pore network while preserving a connected semiconductor matrix.

- Use a nonzero channel fraction and channel/compact volume ratio above 1 when channels should dominate volume.
- Use an `eta` distribution spanning long channels and `tau > 1` for curvature.
- Bias orientation toward the desired periodic direction when directional paths matter.
- Generate multiple candidates because topology depends strongly on realization.
- Require pore connectivity in post-analysis and retain matrix percolation/cross-section gates.

## Thin near-through film

Goal: pores that approach or cross finite z thickness.

- Reduce `film.target_box_A.z` while keeping x/y unchanged.
- Fixed number density produces fewer units in a thinner film.
- Use broad size/aspect-ratio support if only some pores should cross z.
- Evaluate both z-face openings and whether the same component touches both faces.
- Openings on both faces do not alone prove through-connectivity.

## Preview resolution

Goal: rapid morphology inspection.

- Use spacing that divides every target-box axis exactly.
- Include `preview` in task name and metadata.
- Coarsen scale search and final voxelization only as needed.
- Recompute porosity/connectivity from preview HDF5/GLB.
- Do not present a preview as a production audit; narrow necks and roughness can be lost.

## Production audit

Goal: quantitative acceptance and optimizer data.

- Enable strict audit.
- Choose coarse/fine spacings after convergence and memory checks.
- Use multiple candidates and a seed panel when realization variance matters.
- Preserve distribution, overlap, RDF, local-thickness, connectivity, and cross-section metrics.
- Run the independent validator and retain hashes, logs, and exact inputs.

## Smaller or more separated pores

```text
mean pore volume ≈ porosity * film volume / unit count
linear size ∝ mean pore volume^(1/3)
```

Reduce porosity, increase unit count moderately, strengthen RDF exclusion, weaken aggregation peaks, and narrow the upper relative-volume tail. Compare realized metrics because boundary clipping and overlap change the final in-box distribution.

## Surface smoothness

- Generation roughness changes scientific implicit geometry.
- Blender Shade Smooth changes only appearance.
- Mesh smoothing changes geometry; fix outer boundaries, use volume-preserving smoothing, and rerun porosity, connectivity, cross-section, opening, and phase/mesh checks.