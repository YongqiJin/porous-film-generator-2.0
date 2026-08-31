# Generation workflow and parameter determination

## Summary

One YAML supplies a complete design point, grouped as position-quantity, shape, and proportion.
Geometry is produced in dependency order. The generator does not trust its sampled unit parameters
as final measurements.

## Ordered schema-v3 workflow

1. Validate the real target box, formal target distributions, internal generation controls,
   measurement contract, hard constraints, and parallel settings.
2. Preflight grid divisibility, candidate count, memory, and measurement resolution. A geometry-only
   phase-field run needs no PDB and no padding.
3. Derive an internal generation-unit count from seed_number_density_A3 and target-box volume.
4. Generate internal anchors using the center-distance target as a construction heuristic.
5. Allocate internal compact/channel labels.
6. Sample internal relative sizes and the formal shape targets.
7. Sample paired projected orientation: choose one mixture component, then sample its xz and xy Beta
   curves together and construct one three-dimensional unoriented axis.
8. Construct multilobe compact pores and variable-radius multibend channels.
9. Voxelize the smooth union in the real target box with periodic x/y and finite z.
10. Extract final-phase slice centers and track them through z into centerlines.
11. Measure final g_xy(r), centerline-normal equivalent diameters, paired projected orientations,
    channel eta/tau, curvature fluctuation, porosity, through-connectivity, and matrix constraints.
12. Compare only identically defined final measurements with formal targets.
13. Select the first passing candidate; otherwise retain the closest successful candidate as an
    explicitly infeasible diagnostic.
14. Replay the selected candidate deterministically.
15. Export final phase, centerlines, normal-section evidence, GLB, reports, and optimizer exchange.
16. Run the import-isolated validator.

## Complex shape mapping retained from v1

YAML-driven compact shapes use 2-4 lobes, shape_seed, Sobol volume estimation, smooth-min joining,
at most 16 attempts, and does not fall back to an ellipsoid. Channels use 7 control points, PCHIP
radii, radius CV constraints, shared bisector planes, multibend and nonplanar checks, at most
16 attempts, and does not fall back to a constant-radius channel.

## Internal versus formal variables

Internal controls include generation-unit count, type labels, relative volume, roughness amplitude,
lobe count, control points, and shape seeds. They can be changed by the construction algorithm but
cannot satisfy a formal audit target.

Formal targets are:

- position-quantity: g_xy(r);
- shape: local equivalent diameter, paired projected orientation, compact/channel eta, channel tau,
  and curvature fluctuation;
- proportion: final volume porosity.

## Porosity and absolute diameter

Schema v3 specifies an absolute local equivalent-diameter distribution and a porosity target. A
global scale change affects both. Therefore a candidate is accepted only after both are remeasured
from the final phase and pass together. Future internal search may change unit count, channel length,
placement, connection, or starting scale; it must not declare success from porosity alone.

## Candidate and seed parallelism

Candidate and seed tasks remain CPU multiprocessing jobs. Random identities are prepared by the
parent, workers use one numerical thread, and the selected identity is replayed before publication.
Serial and parallel execution must produce the same scientific geometry for the same identity.

## Preview versus production

A coarse preview can hide small pores, necks, roughness, branches, and surface openings. Production
acceptance requires the fixed measurement contract and independent validation at the chosen
resolution.


## External optimization

The generator is not a joint multi-parameter optimizer. External optimization, including Bayesian
optimization, proposes complete formal target sets between runs. Requested design parameters and
realized geometry parameters remain separate outputs.
