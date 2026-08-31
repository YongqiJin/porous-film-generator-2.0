# Final Pore Parameter Contract Design

## Status

Approved through the August 21-24, 2026 design review recorded in the Feishu documents:

- `多孔有机半导体薄膜定量建模记录`
- `孔结构生成器参数定义与最终结构审核改造方案（逐项确认稿）`
- `多孔半导体薄膜：孔统计生成参数与解析优化参数`

This document is the repository-side implementation contract. It does not change the immutable
`original-v0.2.0` tag or the deployed stable release.

## Goal

Make user-facing pore targets and final-geometry audit measurements mean exactly the same thing.
Generation-unit metadata remains available for construction and diagnostics, but it is never used
as a substitute for a measurement made from the final phase geometry.

## Formal parameter dimensions

Formal pore targets remain grouped into exactly three dimensions.

### Position-quantity

- `g_xy(r)`: at each physical z slice, identify distinct through-pore centerlines and measure
  periodic minimum-image xy distances between different centerlines; aggregate over film depth.
- Center count and center-number density are diagnostics, not strict target gates.

### Shape

- Local equivalent-diameter distribution `p_D(D_eq)` from centerline-normal cross sections.
- Paired projected orientation mixture `p(theta_xz, theta_xy)`; every mixture component contains
  one bounded Beta distribution for each projected angle.
- Compact-pore aspect ratio `eta_c` when a final-geometry classification contract is available.
- Channel aspect ratio `eta_ch = L_arc / D_channel`.
- Channel tortuosity `tau = L_arc / L_end`.
- Local cross-section curvature-fluctuation distribution `p_kappa(w)`.

The channel-level equivalent diameter is

`D_channel = 2 * sqrt(mean_s(A_perp(s)) / pi)`.

### Proportion

- Final three-dimensional pore volume fraction `Phi_V` in the real target box.

## Geometry and boundary contract

- `target_box_A` is the only physical box.
- No packing box or z padding is part of the standard phase-field workflow.
- x and y are periodic; z has finite lower and upper surfaces.
- Required through pores must connect the lower and upper z surfaces through the same pore network.
- The semiconductor phase must preserve x percolation and the configured minimum yz cross section.

## Measurement isolation

The final-geometry measurer consumes only the final phase field, target box, grid coordinates, and
the fixed measurement contract. It must not consume requested target values, generation-unit labels,
sampled values, shape seeds, or generator-written per-unit metrics.

The comparator consumes the immutable measurement result plus requested targets. Changing a target
without changing the phase file must not change the measurement result.

## Through-pore centerline contract

1. Detect local pore centers independently in every z slice from the final pore phase.
2. Track compatible centers between adjacent slices to form through-pore centerlines.
3. Use the periodic minimum-image xy distance only between different centerlines at the same z.
4. Aggregate those distances over all valid z slices.
5. Build the random RDF reference with the same per-slice center counts and xy box.

The exact slice spacing, center-resolution scale, tracking tolerance, branch exclusion length, and
minimum valid track length are measurement-contract settings. They are fixed for a study and are not
optimization variables.

## Local normal cross sections

At fixed physical arc-length spacing along each valid centerline segment:

1. construct a plane normal to the local tangent;
2. intersect the final pore phase with the plane;
3. select the connected cross-section containing the centerline point;
4. measure `A_perp(s)` and compute `D_eq(s) = 2 * sqrt(A_perp(s) / pi)`;
5. resample the boundary at a fixed physical interval and compute curvature fluctuation
   `w = std(kappa) / abs(mean(kappa))`.

Branch/merge neighborhoods and surface openings that do not yield one unambiguous local section are
reported as invalid samples and excluded using fixed physical exclusion distances.

## Paired orientation contract

For an unoriented unit vector `n = (nx, ny, nz)`, with `n` equivalent to `-n`:

- `theta_xz = atan2(abs(nz), abs(nx))`
- `theta_xy = atan2(abs(ny), abs(nx))`

Both are in `[0, pi/2]`. A mixture component owns both Beta marginals, and the component is chosen
before either angle is sampled. Cross-component pairing is forbidden. A projection below the fixed
identifiability threshold is reported as unidentifiable and excluded from the corresponding angle
comparison.

## Internal generation controls

The following are internal controls, not formal final-geometry targets:

- generation-unit density and count;
- compact/channel generation labels and count ratio;
- generation-unit relative volumes and channel/compact mean-volume ratio;
- roughness amplitude;
- lobe count, shape seed, control points, and radius knots.

They may be logged and optimized internally, but they must not be presented as final-geometry audit
measurements.

## Compatibility strategy

- Introduce the new grouped target schema and measurement contract as schema v3.
- Keep v2 geometry records readable by the independent validator.
- Accept legacy YAML only through an explicit compatibility translator; normalized and requested
  outputs identify legacy-derived values as internal controls rather than formal targets.
- Bump the development package version to `0.4.0.dev1`.

## Acceptance principles

- Final porosity, `g_xy(r)`, local equivalent diameter, paired orientation, channel `eta/tau`, and
  curvature fluctuation are measured from the final phase.
- A target and a measurement are compared only if their object, units, boundary handling, sampling
  weights, and segmentation rules are identical.
- No approximate proxy may silently replace a formal parameter.
- Measurement-resolution instability is a measurement failure, not a distribution failure.
- Main and independent validators must agree within explicit numerical tolerances.

