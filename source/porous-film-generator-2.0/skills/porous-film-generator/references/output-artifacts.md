# Output tree and artifact consumers

A run is stored under the configured result root using the required dated task hierarchy.


## Directory layout

- inputs/: exact and normalized inputs.
- work/: intermediate and parallel state.
- outputs/: primary user-facing structures and optimizer exchange.
- analysis/: processed measurements and replicate summaries.
- reports/: human-readable run reports.
- logs/: failures and diagnostic traces.
- qa_export/: neutral evidence for the independent validator.

Legacy optional molecule workflows may additionally write pore_reference_coordinates.cif and
packmol_handoff.inp.

## Schema-v3 geometry evidence

The geometry-only phase-field workflow writes:

- qa_export/final_phase.h5: authoritative binary final pore phase;
- qa_export/final_centerlines.h5: extracted final-phase centerline tracks;
- qa_export/final_cross_sections.csv: each valid or rejected normal section, including area,
  equivalent_diameter_A, curvature_fluctuation, and rejection reason;
- qa_export/final_measurements.json: machine-readable final-phase measurements;
- qa_export/semiconductor_solid_target.glb: Blender-readable semiconductor solid with pore cavities;
- qa_export/final_surface.ply: neutral surface mesh;
- qa_export/main_metrics.json: porosity and high-level audit status;
- qa_export/contract.json and checksums.sha256;
- qa_export/normalized_config.yaml;
- qa_export/unit_geometry.jsonl, main_unit_metrics.csv, and channel_curves.h5 as generation
  provenance, not formal final-geometry evidence.

The output GLB is also written to outputs/semiconductor_solid_target.glb.
The combined geometry/provenance container is written to outputs/pore_geometry.h5.

## Final measurement evidence

final_centerlines.h5 schema v3 stores one group per extracted track, including centerline_A data,
wrapped/unwrapped coordinates, wall-distance support, through-surface flags, and branch diagnostics.

final_cross_sections.csv stores track id, arc position, center, tangent, validity, rejection reason,
area, local equivalent diameter, and curvature fluctuation. Equivalent diameter and curvature use
the same accepted normal sections.

final_measurements.json contains the complete final-phase measurement summary, including g_xy(r),
through tracks, projected orientation, channel eta/tau, and valid/invalid section counts.

## Legacy geometry provenance

unit_geometry.jsonl keeps schema v1 and schema v2 records readable. Schema v2 records include
shape_model, shape_seed, lobe_centers_local_A, radius_profile_s, target_volume_A3 where meaningful,
and complex-shape diagnostics.

channel_curves.h5 retains centerline_A and radius_A for generation replay. The independent validator
continues to read v1 and v2 provenance while treating schema-v3 final-phase evidence as authoritative
for formal targets. Existing per-unit volume checks retain the 3% tolerance only where a meaningful
target_volume_A3 is present. shape_complexity_summary and main_unit_metrics.csv remain diagnostic
outputs.

## Consumers

- Blender: outputs/semiconductor_solid_target.glb.
- Phase-field calculation: qa_export/final_phase.h5.
- Final-geometry audit: final_centerlines.h5, final_cross_sections.csv, and final_measurements.json.
- Generator debugging: unit_geometry.jsonl, main_unit_metrics.csv, and channel_curves.h5.
- Bayesian or other external optimization: requested_design_parameters.json, realized_geometry_parameters.json,
  feasibility.json, calculation_status.json, objectives.json, and uncertainty.json.
- Independent validation: porous-film-validate followed by the qa_export directory.
  The validator writes independent-validation.json and independent-validation-report.md.

Molecule and Packmol artifacts are legacy optional outputs. They are not mandatory in a schema-v3
geometry-only phase-field run.
