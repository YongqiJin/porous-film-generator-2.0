from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np

_AVOGADRO_PER_MOL = 6.022_140_76e23
_ANGSTROM3_TO_CM3 = 1.0e-24


@dataclass(frozen=True)
class ElementData:
    symbol: str
    mass_g_mol: float
    radius_A: float


_ELEMENTS: dict[str, ElementData] = {
    "H": ElementData("H", 1.008, 1.20),
    "B": ElementData("B", 10.81, 1.92),
    "C": ElementData("C", 12.011, 1.70),
    "N": ElementData("N", 14.007, 1.55),
    "O": ElementData("O", 15.999, 1.52),
    "F": ElementData("F", 18.998_403_163, 1.47),
    "Si": ElementData("Si", 28.085, 2.10),
    "P": ElementData("P", 30.973_761_998, 1.80),
    "S": ElementData("S", 32.06, 1.80),
    "Cl": ElementData("Cl", 35.45, 1.75),
    "Ar": ElementData("Ar", 39.93, 1.88),
    "Se": ElementData("Se", 78.971, 1.90),
    "Br": ElementData("Br", 79.904, 1.85),
    "I": ElementData("I", 126.904_47, 1.98),
}


@dataclass(frozen=True)
class MoleculeTemplate:
    elements: tuple[str, ...]
    molar_mass_g_mol: float
    positions_A: np.ndarray
    atom_names: tuple[str, ...]
    residue_names: tuple[str, ...]
    chain_ids: tuple[str, ...]
    residue_numbers: tuple[int, ...]
    serial_numbers: tuple[int, ...]
    original_positions_A: np.ndarray
    center_of_mass_A: np.ndarray
    masses_g_mol: np.ndarray
    radii_A: np.ndarray
    source_atom_indices: tuple[int, ...] | None = None
    conect_pairs: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        positions = _as_positions(self.positions_A, "positions_A")
        original = _as_positions(self.original_positions_A, "original_positions_A")
        center = _as_vector(self.center_of_mass_A, "center_of_mass_A")
        masses = np.asarray(self.masses_g_mol, dtype=float)
        radii = np.asarray(self.radii_A, dtype=float)
        atom_count = positions.shape[0]
        if original.shape != positions.shape:
            raise ValueError("original_positions_A must match positions_A shape")
        if masses.shape != (atom_count,) or not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
            raise ValueError("masses_g_mol must contain one positive finite mass per atom")
        if radii.shape != (atom_count,) or not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
            raise ValueError("radii_A must contain one positive finite radius per atom")
        for name, values in (
            ("elements", self.elements),
            ("atom_names", self.atom_names),
            ("residue_names", self.residue_names),
            ("chain_ids", self.chain_ids),
            ("residue_numbers", self.residue_numbers),
            ("serial_numbers", self.serial_numbers),
        ):
            if len(values) != atom_count:
                raise ValueError(f"{name} must contain one value per atom")
        source_atom_indices = (
            tuple(range(atom_count))
            if self.source_atom_indices is None
            else tuple(int(index) for index in self.source_atom_indices)
        )
        if len(source_atom_indices) != atom_count:
            raise ValueError("source_atom_indices must contain one value per atom")
        object.__setattr__(self, "positions_A", positions.copy())
        object.__setattr__(self, "original_positions_A", original.copy())
        object.__setattr__(self, "center_of_mass_A", center.copy())
        object.__setattr__(self, "masses_g_mol", masses.copy())
        object.__setattr__(self, "radii_A", radii.copy())
        object.__setattr__(self, "molar_mass_g_mol", float(np.sum(masses)))
        object.__setattr__(self, "source_atom_indices", source_atom_indices)
        object.__setattr__(
            self,
            "conect_pairs",
            tuple(
                sorted(
                    {
                        (int(min(first, second)), int(max(first, second)))
                        for first, second in self.conect_pairs
                    }
                )
            ),
        )

    @staticmethod
    def from_pdb(path: Path) -> MoleculeTemplate:
        input_path = Path(path)
        structure = gemmi.read_structure(str(input_path))
        atoms: list[tuple[str, str, str, int, int, str, np.ndarray]] = []
        for model in structure:
            for chain in model:
                chain_id = chain.name or "A"
                for residue in chain:
                    residue_name = residue.name or "MOL"
                    residue_number = int(residue.seqid.num) if residue.seqid.num is not None else 1
                    for atom in residue:
                        element = _canonical_element(atom.element.name, atom.name)
                        atoms.append(
                            (
                                atom.name.strip() or element,
                                residue_name,
                                chain_id,
                                residue_number,
                                int(atom.serial) if atom.serial > 0 else len(atoms) + 1,
                                element,
                                np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float),
                            )
                        )
            break
        if not atoms:
            raise ValueError(f"PDB contains no atoms: {input_path}")

        positions = np.vstack([atom[-1] for atom in atoms])
        elements = tuple(atom[5] for atom in atoms)
        masses = np.asarray([_ELEMENTS[element].mass_g_mol for element in elements], dtype=float)
        radii = np.asarray([_ELEMENTS[element].radius_A for element in elements], dtype=float)
        center_of_mass = np.average(positions, axis=0, weights=masses)
        serial_numbers = tuple(atom[4] for atom in atoms)

        return MoleculeTemplate(
            elements=elements,
            molar_mass_g_mol=float(np.sum(masses)),
            positions_A=positions - center_of_mass,
            atom_names=tuple(atom[0] for atom in atoms),
            residue_names=tuple(atom[1] for atom in atoms),
            chain_ids=tuple(atom[2] for atom in atoms),
            residue_numbers=tuple(atom[3] for atom in atoms),
            serial_numbers=serial_numbers,
            original_positions_A=positions,
            center_of_mass_A=center_of_mass,
            masses_g_mol=masses,
            radii_A=radii,
            source_atom_indices=tuple(range(len(atoms))),
            conect_pairs=_read_conect_pairs(input_path, serial_numbers),
        )


def molecule_count_for_density(
    template: MoleculeTemplate,
    pore_volume_A3: float,
    density_g_cm3: float,
) -> int:
    volume_A3 = _positive_float(pore_volume_A3, "pore_volume_A3")
    density = _nonnegative_float(density_g_cm3, "density_g_cm3")
    molecule_mass_g = float(template.molar_mass_g_mol) / _AVOGADRO_PER_MOL
    expected = density * volume_A3 * _ANGSTROM3_TO_CM3 / molecule_mass_g
    return int(np.floor(expected + 0.5))


def density_for_count(template: MoleculeTemplate, pore_volume_A3: float, count: int) -> float:
    volume_A3 = _positive_float(pore_volume_A3, "pore_volume_A3")
    molecule_count = int(count)
    if molecule_count < 0:
        raise ValueError("count must be nonnegative")
    mass_g = molecule_count * float(template.molar_mass_g_mol) / _AVOGADRO_PER_MOL
    return mass_g / (volume_A3 * _ANGSTROM3_TO_CM3)


def _canonical_element(raw_symbol: str, atom_name: str) -> str:
    candidates: list[str] = []
    stripped_symbol = raw_symbol.strip()
    if stripped_symbol:
        candidates.append(stripped_symbol)
    letters = "".join(character for character in atom_name.strip() if character.isalpha())
    if letters:
        candidates.append(letters[:2])
        candidates.append(letters[:1])
    for candidate in candidates:
        normalized = candidate[:1].upper() + candidate[1:].lower()
        if normalized in _ELEMENTS:
            return normalized
    raise ValueError(f"unsupported element for atom {atom_name!r}: {raw_symbol!r}")


def _read_conect_pairs(path: Path, serial_numbers: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    serial_to_index = {serial: index for index, serial in enumerate(serial_numbers)}
    pairs: set[tuple[int, int]] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("CONECT"):
            continue
        serials = [int(field) for field in line[6:].split() if field.isdigit()]
        if len(serials) < 2 or serials[0] not in serial_to_index:
            continue
        source = serial_to_index[serials[0]]
        for target_serial in serials[1:]:
            if target_serial not in serial_to_index:
                continue
            target = serial_to_index[target_serial]
            if source != target:
                pairs.add((min(source, target), max(source, target)))
    return tuple(sorted(pairs))


def _as_positions(positions_A: np.ndarray, name: str) -> np.ndarray:
    positions = np.asarray(positions_A, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    if positions.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one atom")
    if not np.all(np.isfinite(positions)):
        raise ValueError(f"{name} must be finite")
    return positions


def _as_vector(vector_A: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(vector_A, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def _positive_float(value: float, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return parsed


def _nonnegative_float(value: float, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite value")
    return parsed
