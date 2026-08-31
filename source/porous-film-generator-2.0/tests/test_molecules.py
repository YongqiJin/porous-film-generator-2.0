from pathlib import Path

import numpy as np

from porous_film.molecules import (
    MoleculeTemplate,
    molecule_count_for_density,
)


def test_argon_template_parses_element_and_mass() -> None:
    template = MoleculeTemplate.from_pdb(Path("tests/fixtures/argon.pdb"))

    assert template.elements == ("Ar",)
    assert 39.9 < template.molar_mass_g_mol < 40.0


def test_density_count_rounds_to_nearest_integer() -> None:
    template = MoleculeTemplate.from_pdb(Path("tests/fixtures/argon.pdb"))

    count = molecule_count_for_density(
        template=template,
        pore_volume_A3=100000.0,
        density_g_cm3=1.0,
    )

    assert count == 1508


def test_template_parses_supported_heavy_elements_and_canonical_symbols() -> None:
    template = MoleculeTemplate.from_pdb(Path("tests/fixtures/supported-elements.pdb"))

    assert template.elements == ("Se", "F", "Cl", "Br", "I", "Si")
    assert np.all(template.masses_g_mol > 18.0)
    assert np.all(template.radii_A > 1.0)


def test_template_preserves_rigid_distances_and_conect_pairs() -> None:
    template = MoleculeTemplate.from_pdb(Path("tests/fixtures/chlorine-bond.pdb"))

    distance = np.linalg.norm(template.positions_A[1] - template.positions_A[0])

    assert np.isclose(distance, 2.0)
    assert template.conect_pairs == ((0, 1),)
    assert template.serial_numbers == (11, 12)
    assert template.atom_names == ("CL1", "CL2")
    assert template.residue_numbers == (7, 7)
