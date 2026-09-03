# Input schema v3 and parameter interactions

The input is a strict YAML mapping. Unknown fields are rejected. Lengths use angstrom (Å), angles in
the public schema use degrees, fractions are dimensionless, and the standard phase-field workflow
does not require a molecule PDB.

## Formal parameter dimensions

Formal targets are grouped into exactly three dimensions.

### Position-quantity

formal_targets.position_quantity.center_distance_xy describes g_xy(r), the periodic xy distance
distribution between different through-pore centerlines observed in the same z slice and aggregated
over film thickness. Its components use kind, amplitude, center_A, and width_A. Supported kinds are
exclusion, dip, peak, and oscillation.

The generator may retain a seed count and seed-number density internally, but final center count and
center-number density are diagnostics rather than strict targets.

### Shape

formal_targets.shape contains:

- equivalent_diameter_A: the length-weighted local equivalent-diameter distribution measured from
  centerline-normal pore cross sections;
- orientation: paired projected xz/xy orientation components;
- compact_aspect_ratio: optional final compact-pore eta target;
- channel_aspect_ratio: channel eta = arc length / channel equivalent diameter;
- channel_tortuosity: tau = arc length / endpoint distance;
- curvature_fluctuation: the local normal-section boundary-curvature fluctuation distribution.

The channel equivalent diameter is 2 sqrt(mean(A_perp)/pi), using the same valid normal sections as
the local diameter distribution.

Paired orientation uses model: paired_projected_planes. Every component contains one bounded Beta
distribution named theta_xz_deg and one named theta_xy_deg. Weights are nonnegative and sum to one.
Both supports must lie within 0 to 90 degrees.

### Proportion

formal_targets.proportion.porosity is the final three-dimensional pore volume fraction in the real
target box.

## Geometry

film.target_box_A is the only required physical box. Schema v3 does not require packing_box_A or
z_padding_A. Legacy schema-v2 inputs remain readable through an explicit compatibility translation,
but their extra packing frame is not part of the schema-v3 phase-field contract.

## Internal generation controls

generation_controls contains variables used to construct candidates:

- seed_number_density_A3;
- channel_fraction_by_count;
- channel_to_compact_mean_volume_ratio;
- optional compact/channel relative-volume distributions;
- optional compact/channel roughness starting distributions.

These values are diagnostics and internal search controls. They are not accepted as evidence that a
final structure matches a formal target.

## Pore topology and sample constraints

`pore_constraints` contains final-pore hard gates:

- `z_connectivity` is `unrestricted` by default. `all_components` requires every final pore
  component, after periodic x/y merging, to touch both finite z surfaces.
- `minimum_through_centerlines` is the minimum number of reconstructed centerline tracks that touch
  the first and last sampled z slices.
- `minimum_valid_cross_sections` is the minimum number of valid normal sections belonging to those
  through tracks.

When `z_connectivity` is `unrestricted`, formal comparisons that require through tracks are marked
`N/A` (not evaluated) and cannot fail validation: `g_xy`, local equivalent diameter, paired orientation,
channel eta/tau, and curvature fluctuation. Porosity, compact eta, matrix constraints, and any
explicit nonzero minimum counts remain active. The corresponding formal-target fields are optional
in this mode. If retained for compatibility, they may guide candidate generation but are still
reported as `N/A`; omit them when no such generation preference is intended.

In `all_components` mode, generation actively translates every channel across both z surfaces and
the porosity scale solver preserves that span. This mode requires
`generation_controls.channel_fraction_by_count: 1`, at least one planned channel, and no
`formal_targets.shape.compact_aspect_ratio`. A channel whose configured diameter, orientation,
eta, and tau cannot span the film is rejected as infeasible instead of silently relaxing a target.
The requested minimum centerline count is preflighted against the largest-remainder planned channel
count; a positive valid-section minimum also requires at least one planned channel.

## Measurement contract

measurement fixes the algorithms' physical resolution:

- z_slice_spacing_A;
- center_min_separation_A;
- center_tracking_max_displacement_A;
- center_distance_bin_width_A and optional center_distance_max_A;
- center_distance_reference_samples;
- centerline_sample_spacing_A: physical arc-length spacing used to resample each smoothed
  centerline before tangent, arc-length, eta/tau, and normal-section measurements;
- cross_section_spacing_A: physical arc-length spacing between normal-section measurements;
- boundary_resample_spacing_A;
- curvature_smoothing_length_A;
- branch_exclusion_length_A and surface_exclusion_length_A;
- orientation_projection_min_fraction;
- orientation_aspect_ratio_tolerance: final channels with eta less than or equal to
  `1 + tolerance` have no identifiable major-axis orientation and are excluded from the paired
  orientation samples.

`orientation_aspect_ratio_tolerance` belongs to `measurement`, so the main audit and independent
validator use the same final-phase rule. The former `audit` location remains input-compatible and
is normalized to `measurement` when a configuration is loaded.

These are fixed for a study and are not Bayesian optimization variables.

## Supported one-dimensional distributions

- constant
- lognormal
- gamma
- weibull / weibull_min
- truncated_normal / truncnorm
- beta
- mixture

A mixture contains weighted analytic child distributions, with weights summing to one. Tabulated
PDFs, CDFs, histograms, arbitrary value tables, and probability arrays remain unsupported.

## Minimal schema-v3 outline

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
          family: beta
          alpha: 2.5
          beta: 3.5
          lower: 300
          upper: 900
        orientation:
          model: paired_projected_planes
          components:
            - weight: 1.0
              theta_xz_deg: {family: beta, alpha: 4, beta: 2, lower: 55, upper: 88}
              theta_xy_deg: {family: beta, alpha: 2, beta: 5, lower: 0, upper: 30}
        channel_aspect_ratio: {family: beta, alpha: 2, beta: 2, lower: 3, upper: 12}
        channel_tortuosity: {family: beta, alpha: 2, beta: 3, lower: 1, upper: 1.8}
        curvature_fluctuation: {family: beta, alpha: 2, beta: 5, lower: 0, upper: 0.8}
      proportion:
        porosity: 0.15
    generation_controls:
      seed_number_density_A3: 8.0e-10
      channel_fraction_by_count: 1.0
      channel_to_compact_mean_volume_ratio: 1.0

The numerical values are syntax examples, not approved physical parameters.


## Other top-level groups

- matrix_constraints: final semiconductor connectivity and cross-section gates.
- pore_constraints: final pore z topology and minimum usable-sample gates.
- audit: candidate count, coarse/fine voxel resolution, memory cap, and comparison settings.
- output: output behavior and plot requests.
- optimization: replicate seed panel for external uncertainty estimation.
- parallel: CPU process planning and worker limits.
- pore_material: optional legacy molecule-packing configuration; not required by schema-v3 geometry-only runs.
