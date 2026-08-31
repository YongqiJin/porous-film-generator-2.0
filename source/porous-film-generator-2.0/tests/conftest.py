from pathlib import Path

import h5py
import numpy as np
import pytest


def write_config(tmp_path: Path, *, name: str = "sample", target_porosity: float = 0.10,
                 seed_density: float = 0.000125, random_seed: int = 11,
                 molecule_count: int = 1, lower_padding: float | None = None,
                 audit_enabled: bool = True, minimum_cross_section_fraction: float = 0.05,
                 maximum_overlap_fraction: float = 0.02,
                 pdb_path: Path | None = None, seed_panel: list[int] | None = None) -> Path:
    config = tmp_path / f"{name}.yaml"
    pore_pdb = (pdb_path or Path("tests/fixtures/argon.pdb").resolve())
    if lower_padding is None:
        film_block = """
  target_box_A: {x: 20, y: 20, z: 20}
  packing_box_A: {x: 20, y: 20, z: 30}"""
    else:
        upper_padding = 10.0 - lower_padding
        film_block = f"""
  target_box_A: {{x: 20, y: 20, z: 20}}
  z_padding_A: {{lower: {lower_padding}, upper: {upper_padding}}}"""
    optimization_block = ""
    if seed_panel is not None:
        optimization_block = "optimization:\n  seed_panel: [" + ", ".join(str(seed) for seed in seed_panel) + "]\n"
    config.write_text(
        f"""
task:
  name: {name}
  random_seed: {random_seed}
film:{film_block}
pores:
  seed_number_density_A3: {seed_density}
  target_porosity: {target_porosity}
  channel_fraction_by_count: 0.0
  channel_to_compact_mean_volume_ratio: 1.0
center_distribution:
  mode: lattice_jitter
  lattice: simple_cubic
  position_jitter: 0.0
compact:
  relative_volume: {{family: constant, value: 1.0}}
  aspect_ratio: {{family: constant, value: 1.0}}
  roughness: {{family: constant, value: 0.0}}
matrix_constraints:
  require_x_percolation: true
  minimum_cross_section_fraction: {minimum_cross_section_fraction}
  maximum_overlap_fraction: {maximum_overlap_fraction}
pore_material:
  pdb: "{pore_pdb.as_posix()}"
  molecule_count: {molecule_count}
geometry_audit:
  enabled: {str(audit_enabled).lower()}
  candidate_count_per_round: 1
  maximum_rounds: 1
  coarse_spacing_A: 2.0
  fine_spacing_A: 1.0
{optimization_block}""",
        encoding="utf-8",
    )
    return config


@pytest.fixture
def sample_config_path(tmp_path: Path) -> Path:
    return write_config(tmp_path)


@pytest.fixture
def feasible_config_path(tmp_path: Path) -> Path:
    return write_config(
        tmp_path,
        name="feasible",
        target_porosity=0.30,
        seed_density=0.001,
        audit_enabled=False,
        minimum_cross_section_fraction=0.0,
    )


@pytest.fixture
def incomplete_qa_export(tmp_path: Path) -> Path:
    qa = tmp_path / "qa_export"
    qa.mkdir()
    with h5py.File(qa / "final_phase.h5", "w") as handle:
        mask = np.zeros((2, 2, 2), dtype=np.uint8)
        mask[0, 0, 0] = 1
        handle.create_dataset("pore_mask", data=mask)
        handle.attrs["spacing_A"] = 1.0
        handle.attrs["target_box_A"] = [2.0, 2.0, 2.0]
        handle.attrs["axis_order"] = "zyx"
    (qa / "contract.json").write_text(
        '{"schema_version": 1, "phase_encoding": {"semiconductor": 0, "pore": 1}}',
        encoding="utf-8",
    )
    (qa / "main_metrics.json").write_text(
        '{"porosity": 0.125}',
        encoding="utf-8",
    )
    return qa


@pytest.fixture
def temporary_result_root(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    root.mkdir()
    return root


@pytest.fixture
def minimal_config_path() -> Path:
    return Path("tests/fixtures/minimal_config.yaml").resolve()
