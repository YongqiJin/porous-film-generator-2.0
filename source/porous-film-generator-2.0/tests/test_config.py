from pathlib import Path

import pytest

from porous_film.config import GeneratorConfig, load_config


def _minimal_config(overrides: dict | None = None) -> dict:
    config = {
        "task": {"name": "minimal", "random_seed": 1},
        "film": {
            "target_box_A": {"x": 10, "y": 20, "z": 30},
            "packing_box_A": {"x": 10, "y": 20, "z": 50},
        },
        "pores": {
            "seed_number_density_A3": 0.001,
            "target_porosity": 0.2,
            "channel_fraction_by_count": 0.0,
            "channel_to_compact_mean_volume_ratio": 1.0,
        },
        "center_distribution": {
            "mode": "lattice_jitter",
            "lattice": "simple_cubic",
            "position_jitter": 0.0,
        },
        "orientation": {
            "distribution": {"family": "beta", "alpha": 2.0, "beta": 2.0},
            "azimuth": "uniform",
        },
        "compact": {
            "relative_volume": {"family": "constant", "value": 1.0},
            "aspect_ratio": {"family": "constant", "value": 1.5},
            "roughness": {"family": "constant", "value": 0.0},
        },
        "pore_material": {"pdb": "molecule.pdb", "molecule_count": 1},
    }
    if overrides is not None:
        config.update(overrides)
    return config


def test_packing_box_must_cover_target_box(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text(
        """
task:
  name: bad-box
  random_seed: 7
film:
  target_box_A: {x: 40, y: 50, z: 30}
  packing_box_A: {x: 40, y: 50, z: 20}
pores:
  seed_number_density_A3: 0.0001
  target_porosity: 0.2
  channel_fraction_by_count: 0.0
center_distribution:
  mode: lattice_jitter
  lattice: simple_cubic
  position_jitter: 0.0
compact:
  relative_volume: {family: constant, value: 1.0}
  aspect_ratio: {family: constant, value: 1.5}
  roughness: {family: constant, value: 0.0}
pore_material:
  pdb: molecule.pdb
  target_density_g_cm3: 1.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="packing z"):
        load_config(config_file)


def test_asymmetric_padding_sets_target_origin() -> None:
    config = GeneratorConfig.model_validate(
        {
            "task": {"name": "padding", "random_seed": 1},
            "film": {
                "target_box_A": {"x": 10, "y": 20, "z": 30},
                "z_padding_A": {"lower": 7, "upper": 13},
            },
            "pores": {
                "seed_number_density_A3": 0.001,
                "target_porosity": 0.2,
                "channel_fraction_by_count": 0.0,
                "channel_to_compact_mean_volume_ratio": 1.0,
            },
            "center_distribution": {
                "mode": "lattice_jitter",
                "lattice": "simple_cubic",
                "position_jitter": 0.0,
            },
            "orientation": {
                "distribution": {"family": "beta", "alpha": 2.0, "beta": 2.0},
                "azimuth": "uniform",
            },
            "compact": {
                "relative_volume": {"family": "constant", "value": 1.0},
                "aspect_ratio": {"family": "constant", "value": 1.5},
                "roughness": {"family": "constant", "value": 0.0},
            },
            "pore_material": {"pdb": "molecule.pdb", "molecule_count": 1},
        }
    )

    assert config.film.packing_box_A.z == 50.0
    assert config.film.target_origin_in_packing_A.tolist() == [0.0, 0.0, 7.0]


def test_target_volume_uses_target_box() -> None:
    config = GeneratorConfig.model_validate(
        {
            "task": {"name": "box", "random_seed": 1},
            "film": {
                "target_box_A": {"x": 10, "y": 20, "z": 30},
                "packing_box_A": {"x": 10, "y": 20, "z": 50},
            },
            "pores": {
                "seed_number_density_A3": 0.001,
                "target_porosity": 0.2,
                "channel_fraction_by_count": 0.25,
                "channel_to_compact_mean_volume_ratio": 2.0,
            },
            "center_distribution": {
                "mode": "lattice_jitter",
                "lattice": "simple_cubic",
                "position_jitter": 0.0,
            },
            "compact": {
                "relative_volume": {"family": "constant", "value": 1.0},
                "aspect_ratio": {"family": "constant", "value": 1.5},
                "roughness": {"family": "constant", "value": 0.0},
            },
            "pore_material": {
                "pdb": "molecule.pdb",
                "target_density_g_cm3": 1.0,
            },
        }
    )

    assert config.film.target_volume_A3 == 6000.0
    assert config.seed_count == 6
    assert config.optimization.seed_panel == (1,)


def test_explicit_packing_box_centers_target_by_default() -> None:
    config = GeneratorConfig.model_validate(_minimal_config())

    assert config.film.target_origin_in_packing_A.tolist() == [0.0, 0.0, 10.0]


def test_identical_explicit_packing_and_target_boxes_keep_zero_origin() -> None:
    config = GeneratorConfig.model_validate(
        _minimal_config(
            {
                "film": {
                    "target_box_A": {"x": 10, "y": 20, "z": 30},
                    "packing_box_A": {"x": 10, "y": 20, "z": 30},
                }
            }
        )
    )

    assert config.film.target_origin_in_packing_A.tolist() == [0.0, 0.0, 0.0]


def test_orientation_rejects_non_beta_distribution_family() -> None:
    config = _minimal_config(
        {
            "orientation": {
                "distribution": {"family": "constant", "value": 0.5},
                "azimuth": "uniform",
            }
        }
    )

    with pytest.raises(ValueError, match="orientation distribution"):
        GeneratorConfig.model_validate(config)


def test_orientation_rejects_non_uniform_azimuth() -> None:
    config = _minimal_config(
        {
            "orientation": {
                "distribution": {"family": "beta", "alpha": 2.0, "beta": 2.0},
                "azimuth": "fixed",
            }
        }
    )

    with pytest.raises(ValueError, match="azimuth"):
        GeneratorConfig.model_validate(config)


def test_downstream_geometry_audit_and_matrix_constraint_keys_are_accepted() -> None:
    config = GeneratorConfig.model_validate(
        {
            "task": {"name": "sample", "random_seed": 11},
            "film": {
                "target_box_A": {"x": 20, "y": 20, "z": 20},
                "packing_box_A": {"x": 20, "y": 20, "z": 30},
            },
            "pores": {
                "seed_number_density_A3": 0.000125,
                "target_porosity": 0.10,
                "channel_fraction_by_count": 0.0,
                "channel_to_compact_mean_volume_ratio": 1.0,
            },
            "center_distribution": {
                "mode": "lattice_jitter",
                "lattice": "simple_cubic",
                "position_jitter": 0.0,
            },
            "compact": {
                "relative_volume": {"family": "constant", "value": 1.0},
                "aspect_ratio": {"family": "constant", "value": 1.0},
                "roughness": {"family": "constant", "value": 0.0},
            },
            "matrix_constraints": {
                "require_x_percolation": True,
                "minimum_cross_section_fraction": 0.05,
            },
            "pore_material": {"pdb": "molecule.pdb", "molecule_count": 1},
            "geometry_audit": {
                "candidate_count_per_round": 1,
                "maximum_rounds": 1,
                "coarse_spacing_A": 2.0,
                "fine_spacing_A": 1.0,
            },
        }
    )

    assert config.matrix_constraints.require_x_percolation is True
    assert config.geometry_audit.fine_spacing_A == 1.0


def test_rdf_components_use_dimensionless_xi_fields() -> None:
    config = _minimal_config(
        {
            "center_distribution": {
                "mode": "rdf",
                "position_jitter": 0.0,
                "rdf": [
                    {
                        "kind": "oscillation",
                        "amplitude": 0.25,
                        "center_xi": 0.8,
                        "width_xi": 0.15,
                    }
                ],
            }
        }
    )

    parsed = GeneratorConfig.model_validate(config)

    assert parsed.center_distribution.rdf[0].kind == "oscillation"
    assert parsed.center_distribution.rdf[0].amplitude == 0.25
    assert parsed.center_distribution.rdf[0].center_xi == 0.8
    assert parsed.center_distribution.rdf[0].width_xi == 0.15


def test_rdf_components_reject_physical_angstrom_fields() -> None:
    config = _minimal_config(
        {
            "center_distribution": {
                "mode": "rdf",
                "position_jitter": 0.0,
                "rdf": [{"amplitude": 0.25, "center_A": 5.0, "width_A": 1.0}],
            }
        }
    )

    with pytest.raises(ValueError, match="center_xi|width_xi|extra"):
        GeneratorConfig.model_validate(config)


def test_rdf_components_reject_bad_kind_and_negative_amplitude() -> None:
    bad_kind = _minimal_config(
        {
            "center_distribution": {
                "mode": "rdf",
                "position_jitter": 0.0,
                "rdf": [
                    {"kind": "valley", "amplitude": 0.25, "center_xi": 0.8, "width_xi": 0.15}
                ],
            }
        }
    )
    negative_amplitude = _minimal_config(
        {
            "center_distribution": {
                "mode": "rdf",
                "position_jitter": 0.0,
                "rdf": [
                    {"kind": "peak", "amplitude": -0.25, "center_xi": 0.8, "width_xi": 0.15}
                ],
            }
        }
    )

    with pytest.raises(ValueError, match="kind"):
        GeneratorConfig.model_validate(bad_kind)
    with pytest.raises(ValueError, match="amplitude"):
        GeneratorConfig.model_validate(negative_amplitude)


def _schema_v3_config() -> dict:
    return {
        "schema_version": 3,
        "task": {"name": "schema-v3", "random_seed": 17},
        "film": {"target_box_A": {"x": 100.0, "y": 120.0, "z": 40.0}},
        "formal_targets": {
            "position_quantity": {
                "center_distance_xy": {
                    "components": [
                        {
                            "kind": "exclusion",
                            "amplitude": 0.8,
                            "center_A": 0.0,
                            "width_A": 12.0,
                        },
                        {
                            "kind": "peak",
                            "amplitude": 0.25,
                            "center_A": 28.0,
                            "width_A": 5.0,
                        },
                    ]
                }
            },
            "shape": {
                "equivalent_diameter_A": {
                    "family": "mixture",
                    "components": [
                        {
                            "weight": 0.65,
                            "family": "beta",
                            "alpha": 2.5,
                            "beta": 3.5,
                            "lower": 8.0,
                            "upper": 18.0,
                        },
                        {
                            "weight": 0.35,
                            "family": "beta",
                            "alpha": 2.5,
                            "beta": 2.5,
                            "lower": 18.0,
                            "upper": 30.0,
                        },
                    ],
                },
                "orientation": {
                    "model": "paired_projected_planes",
                    "components": [
                        {
                            "weight": 0.7,
                            "theta_xz_deg": {
                                "family": "beta",
                                "alpha": 4.0,
                                "beta": 2.0,
                                "lower": 55.0,
                                "upper": 88.0,
                            },
                            "theta_xy_deg": {
                                "family": "beta",
                                "alpha": 2.0,
                                "beta": 5.0,
                                "lower": 0.0,
                                "upper": 30.0,
                            },
                        },
                        {
                            "weight": 0.3,
                            "theta_xz_deg": {
                                "family": "beta",
                                "alpha": 2.5,
                                "beta": 2.5,
                                "lower": 20.0,
                                "upper": 65.0,
                            },
                            "theta_xy_deg": {
                                "family": "beta",
                                "alpha": 2.0,
                                "beta": 3.0,
                                "lower": 0.0,
                                "upper": 60.0,
                            },
                        },
                    ],
                },
                "compact_aspect_ratio": {
                    "family": "beta",
                    "alpha": 2.0,
                    "beta": 3.0,
                    "lower": 1.0,
                    "upper": 3.0,
                },
                "channel_aspect_ratio": {
                    "family": "beta",
                    "alpha": 2.0,
                    "beta": 2.0,
                    "lower": 3.0,
                    "upper": 12.0,
                },
                "channel_tortuosity": {
                    "family": "beta",
                    "alpha": 2.0,
                    "beta": 3.0,
                    "lower": 1.0,
                    "upper": 1.8,
                },
                "curvature_fluctuation": {
                    "family": "mixture",
                    "components": [
                        {
                            "weight": 0.7,
                            "family": "beta",
                            "alpha": 2.0,
                            "beta": 5.0,
                            "lower": 0.0,
                            "upper": 0.3,
                        },
                        {
                            "weight": 0.3,
                            "family": "beta",
                            "alpha": 3.0,
                            "beta": 3.0,
                            "lower": 0.25,
                            "upper": 0.8,
                        },
                    ],
                },
            },
            "proportion": {"porosity": 0.18},
        },
        "generation_controls": {
            "seed_number_density_A3": 0.00025,
            "channel_fraction_by_count": 0.75,
            "channel_to_compact_mean_volume_ratio": 1.5,
        },
        "measurement": {
            "z_slice_spacing_A": 2.0,
            "center_min_separation_A": 4.0,
            "center_tracking_max_displacement_A": 6.0,
            "centerline_sample_spacing_A": 2.0,
            "cross_section_spacing_A": 2.0,
            "boundary_resample_spacing_A": 0.5,
            "curvature_smoothing_length_A": 1.0,
            "branch_exclusion_length_A": 4.0,
            "surface_exclusion_length_A": 2.0,
            "orientation_projection_min_fraction": 0.05,
        },
        "matrix_constraints": {
            "require_x_percolation": True,
            "minimum_cross_section_fraction": 0.05,
        },
        "audit": {
            "candidate_count_per_round": 1,
            "maximum_rounds": 1,
            "coarse_spacing_A": 2.0,
            "fine_spacing_A": 1.0,
        },
    }


def test_schema_v3_accepts_target_box_without_padding_or_pore_material() -> None:
    parsed = GeneratorConfig.model_validate(_schema_v3_config())

    assert parsed.schema_version == 3
    assert parsed.source_schema_version == 3
    assert parsed.film.packing_box_A == parsed.film.target_box_A
    assert parsed.film.target_origin_in_packing_A.tolist() == [0.0, 0.0, 0.0]
    assert parsed.pore_material is None


def test_schema_v3_groups_formal_targets_into_three_dimensions() -> None:
    parsed = GeneratorConfig.model_validate(_schema_v3_config())

    assert parsed.formal_targets.position_quantity.center_distance_xy.components[1].center_A == 28.0
    assert parsed.formal_targets.shape.equivalent_diameter_A.components[0].lower == 8.0
    assert parsed.formal_targets.shape.channel_aspect_ratio.lower == 3.0
    assert parsed.formal_targets.shape.curvature_fluctuation.components[1].upper == 0.8
    assert parsed.formal_targets.proportion.porosity == 0.18
    assert parsed.generation_controls.seed_number_density_A3 == 0.00025


def test_schema_v3_orientation_requires_paired_beta_components() -> None:
    config = _schema_v3_config()
    config["formal_targets"]["shape"]["orientation"]["components"][0]["theta_xy_deg"] = {
        "family": "truncated_normal",
        "mean": 15.0,
        "sigma": 3.0,
        "lower": 0.0,
        "upper": 30.0,
    }

    with pytest.raises(ValueError, match="paired orientation.*Beta"):
        GeneratorConfig.model_validate(config)


def test_schema_v3_rejects_negative_equivalent_diameter_support() -> None:
    config = _schema_v3_config()
    config["formal_targets"]["shape"]["equivalent_diameter_A"] = {
        "family": "beta",
        "alpha": 2.0,
        "beta": 2.0,
        "lower": -1.0,
        "upper": 10.0,
    }

    with pytest.raises(ValueError, match="equivalent diameter"):
        GeneratorConfig.model_validate(config)


def test_legacy_config_derives_formal_targets_and_marks_source_schema() -> None:
    parsed = GeneratorConfig.model_validate(_minimal_config())

    assert parsed.schema_version == 3
    assert parsed.source_schema_version == 2
    assert parsed.formal_targets.proportion.porosity == 0.2
    assert parsed.generation_controls.seed_number_density_A3 == 0.001
