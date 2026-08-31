# Porous-film generation report

- Software version: porous-film 0.4.0.dev1
- Task: phase-field-only
- Status: accepted
- Target box (A): {'x': 10.0, 'y': 10.0, 'z': 10.0}
- Packing box (A): {'x': 10.0, 'y': 10.0, 'z': 10.0}
- Target porosity: 0.12
- Seed count: 1

## Parameters

```yaml
{
  "schema_version": 3,
  "source_schema_version": 3,
  "task": {
    "name": "phase-field-only",
    "random_seed": 29
  },
  "film": {
    "target_box_A": {
      "x": 10.0,
      "y": 10.0,
      "z": 10.0
    },
    "packing_box_A": {
      "x": 10.0,
      "y": 10.0,
      "z": 10.0
    },
    "z_padding_A": null
  },
  "formal_targets": {
    "position_quantity": {
      "center_distance_xy": {
        "components": []
      }
    },
    "shape": {
      "equivalent_diameter_A": {
        "family": "constant",
        "value": 4.0,
        "components": null,
        "alpha": null,
        "beta": null,
        "mean": null,
        "mu": null,
        "sigma": null,
        "s": null,
        "scale": null,
        "loc": null,
        "shape": null,
        "k": null,
        "theta": null,
        "lower": null,
        "upper": null,
        "minimum": null,
        "maximum": null
      },
      "orientation": {
        "model": "paired_projected_planes",
        "components": [
          {
            "weight": 1.0,
            "theta_xz_deg": {
              "family": "beta",
              "value": null,
              "components": null,
              "alpha": 2.0,
              "beta": 2.0,
              "mean": null,
              "mu": null,
              "sigma": null,
              "s": null,
              "scale": null,
              "loc": null,
              "shape": null,
              "k": null,
              "theta": null,
              "lower": 60.0,
              "upper": 90.0,
              "minimum": null,
              "maximum": null
            },
            "theta_xy_deg": {
              "family": "beta",
              "value": null,
              "components": null,
              "alpha": 2.0,
              "beta": 2.0,
              "mean": null,
              "mu": null,
              "sigma": null,
              "s": null,
              "scale": null,
              "loc": null,
              "shape": null,
              "k": null,
              "theta": null,
              "lower": 0.0,
              "upper": 30.0,
              "minimum": null,
              "maximum": null
            }
          }
        ]
      },
      "compact_aspect_ratio": null,
      "channel_aspect_ratio": {
        "family": "constant",
        "value": 4.0,
        "components": null,
        "alpha": null,
        "beta": null,
        "mean": null,
        "mu": null,
        "sigma": null,
        "s": null,
        "scale": null,
        "loc": null,
        "shape": null,
        "k": null,
        "theta": null,
        "lower": null,
        "upper": null,
        "minimum": null,
        "maximum": null
      },
      "channel_tortuosity": {
        "family": "constant",
        "value": 1.1,
        "components": null,
        "alpha": null,
        "beta": null,
        "mean": null,
        "mu": null,
        "sigma": null,
        "s": null,
        "scale": null,
        "loc": null,
        "shape": null,
        "k": null,
        "theta": null,
        "lower": null,
        "upper": null,
        "minimum": null,
        "maximum": null
      },
      "curvature_fluctuation": {
        "family": "constant",
        "value": 0.2,
        "components": null,
        "alpha": null,
        "beta": null,
        "mean": null,
        "mu": null,
        "sigma": null,
        "s": null,
        "scale": null,
        "loc": null,
        "shape": null,
        "k": null,
        "theta": null,
        "lower": null,
        "upper": null,
        "minimum": null,
        "maximum": null
      }
    },
    "proportion": {
      "porosity": 0.12
    }
  },
  "generation_controls": {
    "seed_number_density_A3": 0.001,
    "channel_fraction_by_count": 1.0,
    "channel_to_compact_mean_volume_ratio": 1.0,
    "compact_relative_volume": {
      "family": "constant",
      "value": 1.0,
      "components": null,
      "alpha": null,
      "beta": null,
      "mean": null,
      "mu": null,
      "sigma": null,
      "s": null,
      "scale": null,
      "loc": null,
      "shape": null,
      "k": null,
      "theta": null,
      "lower": null,
      "upper": null,
      "minimum": null,
      "maximum": null
    },
    "channel_relative_volume": {
      "family": "constant",
      "value": 1.0,
      "components": null,
      "alpha": null,
      "beta": null,
      "mean": null,
      "mu": null,
      "sigma": null,
      "s": null,
      "scale": null,
      "loc": null,
      "shape": null,
      "k": null,
      "theta": null,
      "lower": null,
      "upper": null,
      "minimum": null,
      "maximum": null
    },
    "compact_roughness": {
      "family": "constant",
      "value": 0.0,
      "components": null,
      "alpha": null,
      "beta": null,
      "mean": null,
      "mu": null,
      "sigma": null,
      "s": null,
      "scale": null,
      "loc": null,
      "shape": null,
      "k": null,
      "theta": null,
      "lower": null,
      "upper": null,
      "minimum": null,
      "maximum": null
    },
    "channel_roughness": {
      "family": "constant",
      "value": 0.0,
      "components": null,
      "alpha": null,
      "beta": null,
      "mean": null,
      "mu": null,
      "sigma": null,
      "s": null,
      "scale": null,
      "loc": null,
      "shape": null,
      "k": null,
      "theta": null,
      "lower": null,
      "upper": null,
      "minimum": null,
      "maximum": null
    }
  },
  "measurement": {
    "z_slice_spacing_A": 1.0,
    "center_min_separation_A": 1.0,
    "center_tracking_max_displacement_A": 2.0,
    "center_distance_bin_width_A": 1.0,
    "center_distance_max_A": null,
    "center_distance_reference_samples": 1024,
    "centerline_sample_spacing_A": 1.0,
    "cross_section_spacing_A": 1.0,
    "boundary_resample_spacing_A": 0.5,
    "curvature_smoothing_length_A": 0.5,
    "branch_exclusion_length_A": 1.0,
    "surface_exclusion_length_A": 1.0,
    "orientation_projection_min_fraction": 0.05
  },
  "pores": {
    "seed_number_density_A3": 0.001,
    "target_porosity": 0.12,
    "channel_fraction_by_count": 1.0,
    "channel_to_compact_mean_volume_ratio": 1.0
  },
  "center_distribution": {
    "mode": "rdf",
    "lattice": null,
    "position_jitter": 0.0,
    "rdf": []
  },
  "compact": {
    "relative_volume": {
      "family": "constant",
      "value": 1.0,
      "components": null,
      "alpha": null,
      "beta": null,
      "mean": null,
      "mu": null,
      "sigma": null,
      "s": null,
      "scale": null,
      "loc": null,
      "shape": null,
      "k": null,
      "theta": null,
      "lower": null,
      "upper": null,
      "minimum": null,
      "maximum": null
    },
    "aspect_ratio": {
      "family": "constant",
      "value": 1.5,
      "components": null,
      "alpha": null,
      "beta": null,
      "mean": null,
      "mu": null,
      "sigma": null,
      "s": null,
      "scale": null,
      "loc": null,
      "shape": null,
      "k": null,
      "theta": null,
      "lower": null,
      "upper": null,
      "minimum": null,
      "maximum": null
    },
    "roughness": {
      "family": "constant",
      "value": 0.0,
      "components": null,
      "alpha": null,
      "beta": null,
      "mean": null,
      "mu": null,
      "sigma": null,
      "s": null,
      "scale": null,
      "loc": null,
      "shape": null,
      "k": null,
      "theta": null,
      "lower": null,
      "upper": null,
      "minimum": null,
      "maximum": null
    }
  },
  "pore_material": null,
  "orientation": {
    "distribution": {
      "family": "beta",
      "value": null,
      "components": null,
      "alpha": 2.0,
      "beta": 2.0,
      "mean": null,
      "mu": null,
      "sigma": null,
      "s": null,
      "scale": null,
      "loc": null,
      "shape": null,
      "k": null,
      "theta": null,
      "lower": 0.3333333333333333,
      "upper": 0.5,
      "minimum": null,
      "maximum": null
    },
    "azimuth": "uniform"
  },
  "channel": {
    "relative_volume": {
      "family": "constant",
      "value": 1.0,
      "components": null,
      "alpha": null,
      "beta": null,
      "mean": null,
      "mu": null,
      "sigma": null,
      "s": null,
      "scale": null,
      "loc": null,
      "shape": null,
      "k": null,
      "theta": null,
      "lower": null,
      "upper": null,
      "minimum": null,
      "maximum": null
    },
    "aspect_ratio": null,
    "eta": {
      "family": "constant",
      "value": 4.0,
      "components": null,
      "alpha": null,
      "beta": null,
      "mean": null,
      "mu": null,
      "sigma": null,
      "s": null,
      "scale": null,
      "loc": null,
      "shape": null,
      "k": null,
      "theta": null,
      "lower": null,
      "upper": null,
      "minimum": null,
      "maximum": null
    },
    "tortuosity": null,
    "tau": {
      "family": "constant",
      "value": 1.1,
      "components": null,
      "alpha": null,
      "beta": null,
      "mean": null,
      "mu": null,
      "sigma": null,
      "s": null,
      "scale": null,
      "loc": null,
      "shape": null,
      "k": null,
      "theta": null,
      "lower": null,
      "upper": null,
      "minimum": null,
      "maximum": null
    },
    "roughness": {
      "family": "constant",
      "value": 0.0,
      "components": null,
      "alpha": null,
      "beta": null,
      "mean": null,
      "mu": null,
      "sigma": null,
      "s": null,
      "scale": null,
      "loc": null,
      "shape": null,
      "k": null,
      "theta": null,
      "lower": null,
      "upper": null,
      "minimum": null,
      "maximum": null
    }
  },
  "matrix_constraints": {
    "enabled": false,
    "require_x_percolation": false,
    "minimum_cross_section_fraction": 0.0,
    "maximum_overlap_fraction": 1.0,
    "minimum_skeleton_thickness_A": null
  },
  "audit": {
    "enabled": false,
    "candidate_count_per_round": 1,
    "maximum_rounds": 1,
    "coarse_spacing_A": 2.0,
    "fine_spacing_A": 1.0,
    "orientation_aspect_ratio_tolerance": 1e-6,
    "interface_mixing_layer_fraction": 0.0,
    "available_memory_cap_bytes": null
  },
  "output": {
    "root": "runs",
    "write_plots": true
  },
  "optimization": {
    "seed_panel": [
      29
    ]
  },
  "parallel": {
    "enabled": false,
    "strategy": "serial",
    "max_workers": null,
    "cpu_fraction": 0.8,
    "memory_fraction": 0.75,
    "worker_threads": 1,
    "start_method": "spawn",
    "sources": {
      "enabled": "config",
      "strategy": "config",
      "max_workers": "config"
    }
  }
}
```

## Warnings

- equivalent_diameter distribution comparison exceeds audit limits
- theta_xz distribution comparison exceeds audit limits
- theta_xy final-geometry measurement has no valid samples
- channel_eta distribution comparison exceeds audit limits
- channel_tau distribution comparison exceeds audit limits
- curvature_fluctuation distribution comparison exceeds audit limits
- center_distance_xy comparison exceeds audit limit

## Convergence

- audit_passed: False
- porosity: 0.12
- target_porosity: 0.12

## Output paths

- C:\Calculation_results\2026-08-31\python_results\porous-film-generator-v2-external-handoff\work\pytest-temp\test_validator_full_schema_v3_0\v3\qa_export\unit_geometry.jsonl
