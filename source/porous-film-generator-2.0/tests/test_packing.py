from pathlib import Path

import h5py
import numpy as np
import pytest

from porous_film.geometry import CompactUnit, PoreGeometry
from porous_film.molecules import MoleculeTemplate, PackingConfig, PackingError, pack_molecules
from porous_film.molecules import packing as packing_module
from porous_film.molecules.packing import (
    _bond_envelopes_respect_walls_and_open_z,
    _no_collision,
    _template_envelopes_respect_walls_and_open_z,
)
from porous_film.voxel import voxelize_geometry


def test_argon_atoms_pack_inside_spherical_pore() -> None:
    geometry = PoreGeometry(
        [
            CompactUnit.sphere(
                unit_id="sphere",
                center_A=np.array([10.0, 10.0, 10.0]),
                radius_A=6.0,
            )
        ],
        target_box_A=np.array([20.0, 20.0, 20.0]),
    )
    template = MoleculeTemplate.from_pdb(Path("tests/fixtures/argon.pdb"))

    result = pack_molecules(
        template,
        geometry,
        PackingConfig(exact_count=4, minimum_distance_A=2.5),
        np.random.default_rng(9),
    )

    assert result.count == 4
    assert np.all(geometry.sdf(result.atom_positions_A) < 0.0)
    assert result.minimum_interatomic_distance_A >= 2.5


def test_packing_result_writers_emit_handoff_files(tmp_path: Path) -> None:
    geometry = PoreGeometry(
        [
            CompactUnit.sphere(
                unit_id="sphere",
                center_A=np.array([5.0, 5.0, 5.0]),
                radius_A=3.0,
            )
        ],
        target_box_A=np.array([10.0, 10.0, 10.0]),
    )
    template = MoleculeTemplate.from_pdb(Path("tests/fixtures/argon.pdb"))
    result = pack_molecules(
        template,
        geometry,
        PackingConfig(exact_count=1, minimum_distance_A=2.5),
        np.random.default_rng(3),
    )

    pdb_path = result.write_pdb(tmp_path / "packed.pdb")
    mmcif_path = result.write_mmcif(tmp_path / "packed.cif")
    csv_path = result.write_instances_csv(tmp_path / "instances.csv")
    h5_path = result.write_hdf5(tmp_path / "packed.h5")
    metrics_path = result.write_metrics_json(tmp_path / "metrics.json")

    assert pdb_path.read_text(encoding="utf-8").startswith("HETATM")
    assert "_atom_site.Cartn_x" in mmcif_path.read_text(encoding="utf-8")
    assert "quaternion_x" in csv_path.read_text(encoding="utf-8")
    with h5py.File(h5_path, "r") as handle:
        assert handle["atom_positions_A"].shape == (1, 3)
        assert handle["instance_transforms/translations_A"].shape == (1, 3)
    assert '"count": 1' in metrics_path.read_text(encoding="utf-8")


def test_packing_config_requires_exactly_one_count_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PackingConfig()

    with pytest.raises(ValueError, match="exactly one"):
        PackingConfig(exact_count=1, target_density_g_cm3=0.1)


def test_density_driven_packing_uses_density_count() -> None:
    geometry = PoreGeometry(
        [CompactUnit.sphere("sphere", np.array([10.0, 10.0, 10.0]), 6.0)],
        target_box_A=np.array([20.0, 20.0, 20.0]),
    )
    template = MoleculeTemplate.from_pdb(Path("tests/fixtures/argon.pdb"))
    grid = voxelize_geometry(geometry, geometry.target_box_A, 0.625)
    pore_volume_A3 = float(np.count_nonzero(grid.pore_mask) * grid.spacing_A**3)
    density_for_two_argon = 2.0 * template.molar_mass_g_mol / 6.022_140_76e23
    density_for_two_argon /= pore_volume_A3 * 1.0e-24

    result = pack_molecules(
        template,
        geometry,
        PackingConfig(target_density_g_cm3=density_for_two_argon, minimum_distance_A=2.5),
        np.random.default_rng(8),
    )

    assert result.count == 2


def test_atom_vdw_envelope_must_fit_inside_target_z_pore() -> None:
    geometry = PoreGeometry(
        [CompactUnit.sphere("too-small", np.array([5.0, 5.0, 5.0]), 1.0)],
        target_box_A=np.array([10.0, 10.0, 10.0]),
    )
    template = MoleculeTemplate.from_pdb(Path("tests/fixtures/argon.pdb"))

    with pytest.raises(PackingError) as error:
        pack_molecules(
            template,
            geometry,
            PackingConfig(exact_count=1, max_attempts=50),
            np.random.default_rng(1),
        )

    assert error.value.reason == "algorithm_not_converged"


def test_bond_capsule_envelope_rejects_segment_through_semiconductor() -> None:
    geometry = PoreGeometry(
        [
            CompactUnit.sphere("left", np.array([3.0, 5.0, 5.0]), 0.75),
            CompactUnit.sphere("center", np.array([5.0, 5.0, 5.0]), 0.75),
            CompactUnit.sphere("right", np.array([7.0, 5.0, 5.0]), 0.75),
        ],
        target_box_A=np.array([10.0, 10.0, 10.0]),
    )
    template = _two_atom_template(
        positions_A=np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        radii_A=np.array([0.1, 0.1]),
        conect_pairs=((0, 1),),
    )
    positions = template.positions_A + np.array([5.0, 5.0, 5.0])

    assert not _bond_envelopes_respect_walls_and_open_z(
        positions,
        template,
        translation_A=np.array([5.0, 5.0, 5.0]),
        geometry=geometry,
        target_box=np.array([10.0, 10.0, 10.0]),
        wall_clearance_A=0.0,
    )


def test_z_open_protrusion_allowed_only_for_com_open_to_face() -> None:
    open_geometry = PoreGeometry(
        [CompactUnit.sphere("open", np.array([5.0, 5.0, 0.5]), 2.0)],
        target_box_A=np.array([10.0, 10.0, 10.0]),
    )
    closed_geometry = PoreGeometry(
        [CompactUnit.sphere("closed", np.array([5.0, 5.0, 4.0]), 2.0)],
        target_box_A=np.array([10.0, 10.0, 10.0]),
    )
    template = _two_atom_template(
        positions_A=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.25]]),
        radii_A=np.array([0.05, 0.05]),
    )
    open_translation = np.array([5.0, 5.0, 0.5])
    open_positions = template.positions_A + open_translation
    closed_translation = np.array([5.0, 5.0, 4.0])
    closed_positions = template.positions_A + closed_translation - np.array([0.0, 0.0, 4.5])

    assert _template_envelopes_respect_walls_and_open_z(
        open_positions,
        template,
        open_translation,
        open_geometry,
        np.array([10.0, 10.0, 10.0]),
        0.0,
    )
    assert not _template_envelopes_respect_walls_and_open_z(
        closed_positions,
        template,
        closed_translation,
        closed_geometry,
        np.array([10.0, 10.0, 10.0]),
        0.0,
    )


def test_xy_minimum_image_collision_rejects_boundary_overlap() -> None:
    assert not _no_collision(
        candidate_positions_A=np.array([[0.25, 5.0, 5.0]]),
        existing_positions_A=np.array([[9.75, 5.0, 5.0]]),
        target_box=np.array([10.0, 10.0, 10.0]),
        minimum_distance_A=1.0,
    )


def test_error_reason_geometrically_infeasible_for_capacity_bound() -> None:
    geometry = PoreGeometry(
        [CompactUnit.sphere("small", np.array([5.0, 5.0, 5.0]), 3.0)],
        target_box_A=np.array([10.0, 10.0, 10.0]),
    )
    template = MoleculeTemplate.from_pdb(Path("tests/fixtures/argon.pdb"))

    with pytest.raises(PackingError) as error:
        pack_molecules(
            template,
            geometry,
            PackingConfig(exact_count=100, minimum_distance_A=3.0),
            np.random.default_rng(2),
        )

    assert error.value.reason == "algorithm_not_converged"


def test_error_reason_algorithm_not_converged_for_low_attempt_feasible_case() -> None:
    geometry = PoreGeometry(
        [CompactUnit.sphere("feasible", np.array([10.0, 10.0, 10.0]), 6.0)],
        target_box_A=np.array([20.0, 20.0, 20.0]),
    )
    template = MoleculeTemplate.from_pdb(Path("tests/fixtures/argon.pdb"))

    with pytest.raises(PackingError) as error:
        pack_molecules(
            template,
            geometry,
            PackingConfig(exact_count=2, minimum_distance_A=2.5, max_attempts=1),
            np.random.default_rng(9),
        )

    assert error.value.reason == "algorithm_not_converged"


def test_hdf5_and_mmcif_preserve_identifiers_and_bond_metadata(tmp_path: Path) -> None:
    geometry = PoreGeometry(
        [CompactUnit.sphere("sphere", np.array([5.0, 5.0, 5.0]), 4.0)],
        target_box_A=np.array([10.0, 10.0, 10.0]),
    )
    template = MoleculeTemplate.from_pdb(Path("tests/fixtures/chlorine-bond.pdb"))
    result = pack_molecules(
        template,
        geometry,
        PackingConfig(exact_count=1, minimum_distance_A=2.5),
        np.random.default_rng(5),
    )

    h5_path = result.write_hdf5(tmp_path / "packed.h5")
    cif_text = result.write_mmcif(tmp_path / "packed.cif").read_text(encoding="utf-8")
    pdb_text = result.write_pdb(tmp_path / "packed.pdb").read_text(encoding="utf-8")

    with h5py.File(h5_path, "r") as handle:
        assert handle["template_atom_names"].asstr()[:].tolist() == ["CL1", "CL2"]
        assert handle["template_residue_names"].asstr()[:].tolist() == ["MOL", "MOL"]
        assert handle["template_chain_ids"].asstr()[:].tolist() == ["B", "B"]
        assert handle["template_residue_numbers"][:].tolist() == [7, 7]
        assert handle["template_serial_numbers"][:].tolist() == [11, 12]
        assert handle["template_masses_g_mol"].shape == (2,)
        assert handle["template_radii_A"].shape == (2,)
        assert handle["template_source_atom_indices"][:].tolist() == [0, 1]
        assert handle["template_conect_pairs"][:].tolist() == [[0, 1]]
        assert handle["atom_instance_index"][:].tolist() == [0, 0]
        assert handle["atom_template_index"][:].tolist() == [0, 1]

    assert "_struct_conn.ptnr1_label_atom_id" in cif_text
    assert "CL1" in cif_text and "CL2" in cif_text
    assert "B" in cif_text and "MOL" in cif_text
    assert "CONECT" in pdb_text


def test_pdb_writer_uses_written_serials_for_conect_round_trip(tmp_path: Path) -> None:
    geometry = PoreGeometry(
        [CompactUnit.sphere("sphere", np.array([5.0, 5.0, 5.0]), 4.0)],
        target_box_A=np.array([10.0, 10.0, 10.0]),
    )
    template = MoleculeTemplate.from_pdb(Path("tests/fixtures/chlorine-bond.pdb"))
    result = pack_molecules(
        template,
        geometry,
        PackingConfig(exact_count=1, minimum_distance_A=2.5),
        np.random.default_rng(5),
    )

    pdb_lines = result.write_pdb(tmp_path / "packed.pdb").read_text(encoding="utf-8").splitlines()

    assert pdb_lines[0][6:11] == "   11"
    assert pdb_lines[1][6:11] == "   12"
    assert "CONECT   11   12" in pdb_lines


def test_pack_allows_protrusion_when_com_component_is_open_to_face(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(packing_module.Rotation, "random", lambda random_state=None: packing_module.Rotation.identity())
    geometry = _BoxPoreGeometry(
        target_box_A=np.array([10.0, 10.0, 10.0]),
        boxes=[
            (
                np.array([3.0, 3.0, -2.1]),
                np.array([7.0, 7.0, 1.25]),
            )
        ],
    )
    template = _protruding_two_atom_template()

    result = pack_molecules(
        template,
        geometry,
        PackingConfig(exact_count=1, minimum_distance_A=0.0, max_attempts=200),
        np.random.default_rng(4),
    )

    assert result.count == 1
    assert result.protrusion_metrics["atoms_below_z"] == 2


def test_pack_rejects_protrusion_when_same_xy_face_pore_is_disconnected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(packing_module.Rotation, "random", lambda random_state=None: packing_module.Rotation.identity())
    geometry = _BoxPoreGeometry(
        target_box_A=np.array([10.0, 10.0, 10.0]),
        boxes=[
            (
                np.array([3.0, 3.0, 0.85]),
                np.array([7.0, 7.0, 1.25]),
            ),
            (
                np.array([4.8, 4.8, 0.0]),
                np.array([5.2, 5.2, 0.2]),
            ),
        ],
    )
    template = _protruding_two_atom_template()

    with pytest.raises(PackingError) as error:
        pack_molecules(
            template,
            geometry,
            PackingConfig(exact_count=1, minimum_distance_A=0.0, max_attempts=200),
            np.random.default_rng(4),
        )

    assert error.value.reason == "algorithm_not_converged"


def test_distance_overpacking_exhaustion_is_not_geometrically_infeasible() -> None:
    geometry = PoreGeometry(
        [CompactUnit.sphere("small", np.array([5.0, 5.0, 5.0]), 3.0)],
        target_box_A=np.array([10.0, 10.0, 10.0]),
    )
    template = _two_atom_template(
        positions_A=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        radii_A=np.array([0.1, 0.1]),
    )

    with pytest.raises(PackingError) as error:
        pack_molecules(
            template,
            geometry,
            PackingConfig(exact_count=100, minimum_distance_A=3.0, max_attempts=20),
            np.random.default_rng(2),
        )

    assert error.value.reason == "algorithm_not_converged"


def test_arbitrary_nondivisible_target_box_does_not_leak_phasegrid_error() -> None:
    geometry = _BoxPoreGeometry(
        target_box_A=np.array([10.3, 9.7, 10.1]),
        boxes=[
            (
                np.array([0.0, 0.0, 0.0]),
                np.array([10.3, 9.7, 10.1]),
            )
        ],
    )
    template = MoleculeTemplate.from_pdb(Path("tests/fixtures/argon.pdb"))

    result = pack_molecules(
        template,
        geometry,
        PackingConfig(exact_count=1, minimum_distance_A=0.0, max_attempts=1000),
        np.random.default_rng(11),
    )

    assert result.count == 1


def test_largest_atom_volume_overrequest_exhausts_as_algorithm_not_converged() -> None:
    geometry = PoreGeometry(
        [CompactUnit.sphere("small", np.array([5.0, 5.0, 5.0]), 3.0)],
        target_box_A=np.array([10.0, 10.0, 10.0]),
    )
    template = MoleculeTemplate.from_pdb(Path("tests/fixtures/argon.pdb"))

    with pytest.raises(PackingError) as error:
        pack_molecules(
            template,
            geometry,
            PackingConfig(exact_count=100, minimum_distance_A=0.0, max_attempts=20),
            np.random.default_rng(2),
        )

    assert error.value.reason == "algorithm_not_converged"


def test_tiny_offgrid_positive_pore_exhausts_as_algorithm_not_converged() -> None:
    geometry = _TinySphereGeometry(
        target_box_A=np.array([10.0, 10.0, 10.0]),
        center_A=np.array([0.05, 0.05, 0.05]),
        radius_A=0.03,
    )
    template = MoleculeTemplate.from_pdb(Path("tests/fixtures/argon.pdb"))

    with pytest.raises(PackingError) as error:
        pack_molecules(
            template,
            geometry,
            PackingConfig(exact_count=1, minimum_distance_A=0.0, max_attempts=3),
            np.random.default_rng(12),
        )

    assert error.value.reason == "algorithm_not_converged"


def test_single_atom_candidate_sampler_does_not_call_local_thickness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = PoreGeometry(
        [CompactUnit.sphere("sphere", np.array([5.0, 5.0, 5.0]), 4.0)],
        target_box_A=np.array([10.0, 10.0, 10.0]),
    )
    template = _single_atom_template(radius_A=0.5)

    def fail_local_thickness(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("single-atom candidate sampling must not compute local thickness")

    monkeypatch.setattr(packing_module, "local_thickness_field", fail_local_thickness)

    sampler = packing_module._candidate_sampler(
        template,
        geometry,
        PackingConfig(exact_count=1, minimum_distance_A=0.0),
        np.random.default_rng(42),
    )

    candidate = sampler()

    assert candidate.shape == (3,)
    assert np.all(candidate >= 0.0)
    assert np.all(candidate <= geometry.target_box_A)


def test_single_atom_candidate_sampler_zero_weights_insufficient_sdf_clearance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _XDependentClearanceGeometry(
        target_box_A=np.array([4.0, 4.0, 4.0]),
        split_x_A=2.0,
        shallow_clearance_A=0.2,
        deep_clearance_A=2.0,
    )
    template = _single_atom_template(radius_A=0.5)
    rng = _RecordingChoiceRng()

    monkeypatch.setattr(
        packing_module,
        "local_thickness_field",
        lambda mask_zyx, spacing_A, periodic_xy: np.ones_like(mask_zyx, dtype=float),
    )

    sampler = packing_module._candidate_sampler(
        template,
        geometry,
        PackingConfig(exact_count=1, minimum_distance_A=0.0),
        rng,
    )
    sampler()

    pore_centers, spacing_A = _sampler_pore_centers(geometry)
    required_radius_A = 0.5
    sdf_clearance_A = -geometry.sdf(pore_centers)
    sufficient = (sdf_clearance_A - required_radius_A + spacing_A) > 0.0

    assert np.any(sufficient)
    assert np.any(~sufficient)
    assert rng.choice_probabilities is not None
    assert np.all(rng.choice_probabilities[~sufficient] == 0.0)
    assert np.all(rng.choice_probabilities[sufficient] > 0.0)
    assert rng.choice_probabilities.sum() == pytest.approx(1.0)


def test_multi_atom_candidate_sampler_still_uses_local_thickness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = PoreGeometry(
        [CompactUnit.sphere("sphere", np.array([5.0, 5.0, 5.0]), 4.0)],
        target_box_A=np.array([10.0, 10.0, 10.0]),
    )
    template = _two_atom_template(
        positions_A=np.array([[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        radii_A=np.array([0.1, 0.1]),
    )
    calls = 0

    def count_local_thickness(mask_zyx: np.ndarray, spacing_A: float, periodic_xy: bool) -> np.ndarray:
        nonlocal calls
        calls += 1
        assert periodic_xy
        return np.full(mask_zyx.shape, 10.0 * spacing_A, dtype=float)

    monkeypatch.setattr(packing_module, "local_thickness_field", count_local_thickness)

    sampler = packing_module._candidate_sampler(
        template,
        geometry,
        PackingConfig(exact_count=1, minimum_distance_A=0.0),
        np.random.default_rng(7),
    )
    candidate = sampler()

    assert calls == 1
    assert candidate.shape == (3,)
    assert np.all(candidate >= 0.0)
    assert np.all(candidate <= geometry.target_box_A)


def test_empty_units_report_geometrically_infeasible() -> None:
    geometry = PoreGeometry([], np.array([10.0, 10.0, 10.0]))
    template = MoleculeTemplate.from_pdb(Path("tests/fixtures/argon.pdb"))

    with pytest.raises(PackingError) as error:
        pack_molecules(
            template,
            geometry,
            PackingConfig(exact_count=1, minimum_distance_A=0.0, max_attempts=3),
            np.random.default_rng(12),
        )

    assert error.value.reason == "geometrically_infeasible"


def test_certified_grid_does_not_authorize_subcell_lower_opening() -> None:
    target_box = np.array([10.0, 10.0, 10.0])
    geometry = _BoxPoreGeometry(
        target_box_A=target_box,
        boxes=[
            (
                np.array([4.8, 4.8, 0.0]),
                np.array([5.2, 5.2, 1.25]),
            )
        ],
    )

    grid = packing_module._PoreAccessibilityGrid.build(geometry, target_box)

    assert not grid.is_open_to_face(np.array([5.0, 5.0, 0.75]), lower=True)


def test_certified_grid_authorizes_resolved_lower_open_channel() -> None:
    target_box = np.array([10.0, 10.0, 10.0])
    geometry = _BoxPoreGeometry(
        target_box_A=target_box,
        boxes=[
            (
                np.array([3.0, 3.0, -1.0]),
                np.array([7.0, 7.0, 1.25]),
            )
        ],
    )

    grid = packing_module._PoreAccessibilityGrid.build(geometry, target_box)

    assert grid.is_open_to_face(np.array([5.0, 5.0, 0.75]), lower=True)


def test_atom_envelope_clearance_uses_declared_sdf_lipschitz_bound() -> None:
    geometry = _ConstantSdfGeometry(
        target_box_A=np.array([10.0, 10.0, 10.0]),
        value=-1.5,
        sdf_lipschitz_bound=2.0,
    )
    template = _two_atom_template(
        positions_A=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        radii_A=np.array([1.0, 1.0]),
    )
    positions = template.positions_A + np.array([5.0, 5.0, 5.0])

    assert not _template_envelopes_respect_walls_and_open_z(
        positions,
        template,
        np.array([5.0, 5.0, 5.0]),
        geometry,
        np.array([10.0, 10.0, 10.0]),
        0.0,
    )


def test_bond_envelope_clearance_uses_declared_sdf_lipschitz_bound() -> None:
    geometry = _ConstantSdfGeometry(
        target_box_A=np.array([10.0, 10.0, 10.0]),
        value=-0.3,
        sdf_lipschitz_bound=2.0,
    )
    template = _two_atom_template(
        positions_A=np.array([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        radii_A=np.array([0.1, 0.1]),
        conect_pairs=((0, 1),),
    )
    positions = template.positions_A + np.array([5.0, 5.0, 5.0])

    assert not _bond_envelopes_respect_walls_and_open_z(
        positions,
        template,
        translation_A=np.array([5.0, 5.0, 5.0]),
        geometry=geometry,
        target_box=np.array([10.0, 10.0, 10.0]),
        wall_clearance_A=0.0,
    )


def test_pack_rejects_missing_or_invalid_lipschitz_bound() -> None:
    template = MoleculeTemplate.from_pdb(Path("tests/fixtures/argon.pdb"))
    geometries = [
        _MissingLipschitzGeometry(np.array([10.0, 10.0, 10.0])),
        _ConstantSdfGeometry(
            np.array([10.0, 10.0, 10.0]),
            value=-10.0,
            sdf_lipschitz_bound=0.0,
        ),
        _ConstantSdfGeometry(
            np.array([10.0, 10.0, 10.0]),
            value=-10.0,
            sdf_lipschitz_bound=np.inf,
        ),
    ]

    for geometry in geometries:
        with pytest.raises(ValueError, match="sdf_lipschitz_bound"):
            pack_molecules(
                template,
                geometry,
                PackingConfig(exact_count=1, minimum_distance_A=0.0, max_attempts=5),
                np.random.default_rng(21),
            )


def test_protruding_atom_center_must_remain_inside_open_pore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        packing_module.Rotation,
        "random",
        lambda random_state=None: packing_module.Rotation.identity(),
    )
    geometry = _BoxPoreGeometry(
        target_box_A=np.array([10.0, 10.0, 10.0]),
        boxes=[
            (
                np.array([3.0, 3.0, 0.0]),
                np.array([7.0, 7.0, 1.25]),
            )
        ],
    )
    template = _protruding_two_atom_template()

    with pytest.raises(PackingError) as error:
        pack_molecules(
            template,
            geometry,
            PackingConfig(exact_count=1, minimum_distance_A=0.0, max_attempts=200),
            np.random.default_rng(4),
        )

    assert error.value.reason == "algorithm_not_converged"


def test_pack_rejects_bond_crossing_semiconductor_between_valid_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        packing_module.Rotation,
        "random",
        lambda random_state=None: packing_module.Rotation.identity(),
    )
    geometry = _BoxPoreGeometry(
        target_box_A=np.array([10.0, 10.0, 10.0]),
        boxes=[
            (np.array([3.4, 4.0, 4.0]), np.array([4.6, 6.0, 6.0])),
            (np.array([4.9, 4.0, 4.0]), np.array([5.1, 6.0, 6.0])),
            (np.array([5.4, 4.0, 4.0]), np.array([6.6, 6.0, 6.0])),
        ],
    )
    template = _two_atom_template(
        positions_A=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        radii_A=np.array([0.05, 0.05]),
        conect_pairs=((0, 1),),
    )

    with pytest.raises(PackingError) as error:
        pack_molecules(
            template,
            geometry,
            PackingConfig(exact_count=1, minimum_distance_A=0.0, max_attempts=200),
            np.random.default_rng(3),
        )

    assert error.value.reason == "algorithm_not_converged"


@pytest.mark.parametrize(
    ("face", "translation_z", "radius_A", "closed_box_z", "open_box_z"),
    [
        ("lower", 0.2, 0.3, (-0.11, 0.51), (-1.0, 1.0)),
        ("upper", 9.8, 0.3, (9.49, 10.11), (9.0, 11.0)),
    ],
)
@pytest.mark.parametrize("open_to_face", [False, True])
def test_pack_authorizes_atom_complete_z_envelope_by_certified_component(
    monkeypatch: pytest.MonkeyPatch,
    face: str,
    translation_z: float,
    radius_A: float,
    closed_box_z: tuple[float, float],
    open_box_z: tuple[float, float],
    open_to_face: bool,
) -> None:
    target_box = np.array([10.0, 10.0, 10.0])
    translation = np.array([5.0, 5.0, translation_z])
    _force_single_candidate(monkeypatch, translation)
    box_z = open_box_z if open_to_face else closed_box_z
    geometry = _BoxPoreGeometry(
        target_box_A=target_box,
        boxes=[(np.array([3.0, 3.0, box_z[0]]), np.array([7.0, 7.0, box_z[1]]))],
    )
    template = _single_atom_template(radius_A=radius_A)
    config = PackingConfig(exact_count=1, minimum_distance_A=0.0, max_attempts=1)

    if not open_to_face:
        with pytest.raises(PackingError) as error:
            pack_molecules(template, geometry, config, np.random.default_rng(31))
        assert error.value.reason == "algorithm_not_converged"
        return

    result = pack_molecules(template, geometry, config, np.random.default_rng(31))

    assert result.count == 1
    if face == "lower":
        assert result.atom_positions_A[0, 2] - radius_A < 0.0
    else:
        assert result.atom_positions_A[0, 2] + radius_A > target_box[2]


@pytest.mark.parametrize(
    ("face", "translation_z", "closed_box_z", "open_box_z"),
    [
        ("lower", 0.25, (-0.06, 0.56), (-1.0, 1.0)),
        ("upper", 9.75, (9.44, 10.06), (9.0, 11.0)),
    ],
)
@pytest.mark.parametrize("open_to_face", [False, True])
def test_pack_authorizes_bond_complete_z_envelope_by_certified_component(
    monkeypatch: pytest.MonkeyPatch,
    face: str,
    translation_z: float,
    closed_box_z: tuple[float, float],
    open_box_z: tuple[float, float],
    open_to_face: bool,
) -> None:
    target_box = np.array([10.0, 10.0, 10.0])
    translation = np.array([5.0, 5.0, translation_z])
    _force_single_candidate(monkeypatch, translation)
    box_z = open_box_z if open_to_face else closed_box_z
    geometry = _BoxPoreGeometry(
        target_box_A=target_box,
        boxes=[(np.array([4.0, 4.0, box_z[0]]), np.array([6.0, 6.0, box_z[1]]))],
    )
    template = _two_atom_template(
        positions_A=np.array([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        radii_A=np.array([0.2, 0.2]),
        conect_pairs=((0, 1),),
    )
    config = PackingConfig(exact_count=1, minimum_distance_A=0.0, max_attempts=1)

    if not open_to_face:
        with pytest.raises(PackingError) as error:
            pack_molecules(template, geometry, config, np.random.default_rng(37))
        assert error.value.reason == "algorithm_not_converged"
        return

    result = pack_molecules(template, geometry, config, np.random.default_rng(37))

    assert result.count == 1
    sample_spacing_A = 0.2
    capsule_radius_A = 0.2 + 0.5 * sample_spacing_A
    if face == "lower":
        assert translation_z - capsule_radius_A < 0.0
        assert np.all(result.atom_positions_A[:, 2] - template.radii_A >= 0.0)
    else:
        assert translation_z + capsule_radius_A > target_box[2]
        assert np.all(result.atom_positions_A[:, 2] + template.radii_A <= target_box[2])


def _two_atom_template(
    *,
    positions_A: np.ndarray,
    radii_A: np.ndarray,
    conect_pairs: tuple[tuple[int, int], ...] = (),
) -> MoleculeTemplate:
    masses = np.array([12.0, 12.0], dtype=float)
    return MoleculeTemplate(
        elements=("C", "C"),
        molar_mass_g_mol=24.0,
        positions_A=positions_A,
        atom_names=("C1", "C2"),
        residue_names=("MOL", "MOL"),
        chain_ids=("A", "A"),
        residue_numbers=(1, 1),
        serial_numbers=(1, 2),
        original_positions_A=positions_A,
        center_of_mass_A=np.zeros(3),
        masses_g_mol=masses,
        radii_A=radii_A,
        conect_pairs=conect_pairs,
    )


def _single_atom_template(*, radius_A: float) -> MoleculeTemplate:
    position = np.array([[0.0, 0.0, 0.0]], dtype=float)
    masses = np.array([12.0], dtype=float)
    return MoleculeTemplate(
        elements=("C",),
        molar_mass_g_mol=12.0,
        positions_A=position,
        atom_names=("C1",),
        residue_names=("MOL",),
        chain_ids=("A",),
        residue_numbers=(1,),
        serial_numbers=(1,),
        original_positions_A=position,
        center_of_mass_A=np.zeros(3),
        masses_g_mol=masses,
        radii_A=np.array([radius_A], dtype=float),
        conect_pairs=(),
    )


def _protruding_two_atom_template() -> MoleculeTemplate:
    return _two_atom_template(
        positions_A=np.array([[0.0, 0.0, -1.0], [0.5, 0.0, -1.0]]),
        radii_A=np.array([1.0, 1.0]),
        conect_pairs=((0, 1),),
    )


def _force_single_candidate(
    monkeypatch: pytest.MonkeyPatch,
    translation_A: np.ndarray,
) -> None:
    monkeypatch.setattr(
        packing_module.Rotation,
        "random",
        lambda random_state=None: packing_module.Rotation.identity(),
    )
    monkeypatch.setattr(
        packing_module,
        "_candidate_sampler",
        lambda template, geometry, config, rng: lambda: translation_A.copy(),
    )


class _BoxPoreGeometry:
    def __init__(
        self,
        *,
        target_box_A: np.ndarray,
        boxes: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        self.target_box_A = target_box_A
        self._boxes = boxes
        self.units = (object(),)
        self.sdf_lipschitz_bound = 1.0

    def sdf(self, points_A: np.ndarray) -> np.ndarray:
        points = np.asarray(points_A, dtype=float)
        values = []
        for lower, upper in self._boxes:
            center = 0.5 * (lower + upper)
            half = 0.5 * (upper - lower)
            q = np.abs(points - center) - half
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            inside = np.minimum(np.max(q, axis=1), 0.0)
            values.append(outside + inside)
        return np.min(np.vstack(values), axis=0)


class _TinySphereGeometry:
    def __init__(self, *, target_box_A: np.ndarray, center_A: np.ndarray, radius_A: float) -> None:
        self.target_box_A = target_box_A
        self.center_A = center_A
        self.radius_A = radius_A
        self.units = (object(),)
        self.sdf_lipschitz_bound = 1.0

    def sdf(self, points_A: np.ndarray) -> np.ndarray:
        points = np.asarray(points_A, dtype=float)
        return np.linalg.norm(points - self.center_A, axis=1) - self.radius_A


class _ConstantSdfGeometry:
    def __init__(
        self,
        target_box_A: np.ndarray,
        *,
        value: float,
        sdf_lipschitz_bound: float,
    ) -> None:
        self.target_box_A = target_box_A
        self.value = value
        self.sdf_lipschitz_bound = sdf_lipschitz_bound
        self.units = (object(),)

    def sdf(self, points_A: np.ndarray) -> np.ndarray:
        points = np.asarray(points_A, dtype=float)
        return np.full(points.shape[0], self.value, dtype=float)


class _XDependentClearanceGeometry:
    def __init__(
        self,
        *,
        target_box_A: np.ndarray,
        split_x_A: float,
        shallow_clearance_A: float,
        deep_clearance_A: float,
    ) -> None:
        self.target_box_A = target_box_A
        self.split_x_A = split_x_A
        self.shallow_clearance_A = shallow_clearance_A
        self.deep_clearance_A = deep_clearance_A
        self.units = (object(),)
        self.sdf_lipschitz_bound = 1.0

    def sdf(self, points_A: np.ndarray) -> np.ndarray:
        points = np.asarray(points_A, dtype=float)
        clearance = np.where(
            points[:, 0] < self.split_x_A,
            self.shallow_clearance_A,
            self.deep_clearance_A,
        )
        return -clearance


class _RecordingChoiceRng:
    def __init__(self) -> None:
        self.choice_probabilities: np.ndarray | None = None

    def choice(self, stop: int, *, p: np.ndarray) -> int:
        self.choice_probabilities = np.asarray(p, dtype=float).copy()
        return int(np.argmax(self.choice_probabilities))

    def uniform(
        self,
        low: float | np.ndarray,
        high: float | np.ndarray,
        size: int | tuple[int, ...] | None = None,
    ) -> np.ndarray:
        if size is None:
            return 0.5 * (np.asarray(low, dtype=float) + np.asarray(high, dtype=float))
        return np.zeros(size, dtype=float)


def _sampler_pore_centers(geometry: _XDependentClearanceGeometry) -> tuple[np.ndarray, float]:
    target_box = np.asarray(geometry.target_box_A, dtype=float)
    spacing_A = float(np.min(target_box) / 32)
    grid = voxelize_geometry(geometry, target_box, spacing_A)
    nz, ny, nx = grid.pore_mask.shape
    z, y, x = np.indices((nz, ny, nx), dtype=float)
    centers = np.column_stack(
        [
            (x.ravel() + 0.5) * spacing_A,
            (y.ravel() + 0.5) * spacing_A,
            (z.ravel() + 0.5) * spacing_A,
        ]
    )
    return centers[grid.pore_mask.ravel()], spacing_A


class _MissingLipschitzGeometry:
    def __init__(self, target_box_A: np.ndarray) -> None:
        self.target_box_A = target_box_A
        self.units = (object(),)

    def sdf(self, points_A: np.ndarray) -> np.ndarray:
        points = np.asarray(points_A, dtype=float)
        return np.full(points.shape[0], -10.0, dtype=float)


def test_pack_molecules_reuses_one_large_sdf_grid_for_packing_precomputation() -> None:
    delegate = PoreGeometry(
        [CompactUnit.sphere("sphere", np.array([5.0, 5.0, 5.0]), 4.0)],
        target_box_A=np.array([10.0, 10.0, 10.0]),
    )

    class CountingGeometry:
        def __init__(self) -> None:
            self.target_box_A = delegate.target_box_A
            self.units = delegate.units
            self.sdf_lipschitz_bound = delegate.sdf_lipschitz_bound
            self.large_sdf_calls = 0

        def sdf(self, points_A: np.ndarray) -> np.ndarray:
            points = np.asarray(points_A, dtype=float)
            if points.shape[0] >= 1_000:
                self.large_sdf_calls += 1
            return delegate.sdf(points)

    geometry = CountingGeometry()
    template = _single_atom_template(radius_A=0.5)

    result = pack_molecules(
        template,
        geometry,
        PackingConfig(exact_count=1, minimum_distance_A=0.0, max_attempts=100),
        np.random.default_rng(1201),
    )

    assert result.count == 1
    assert geometry.large_sdf_calls == 1
