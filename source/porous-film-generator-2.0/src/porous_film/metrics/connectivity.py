from __future__ import annotations

import numpy as np
from scipy import ndimage

_CONNECTIVITY_6 = ndimage.generate_binary_structure(3, 1)


def periodic_percolates_x(mask_zyx: np.ndarray) -> bool:
    """Return whether a 6-connected domain reaches its periodic x image.

    The film is periodic in both x and y, and finite in z. A winding path
    along x may therefore depend on y-face adjacency.
    """

    mask = _as_bool_zyx(mask_zyx)
    if not np.any(mask):
        return False

    ny = mask.shape[1]
    nx = mask.shape[2]
    tiled = np.tile(mask, (1, 3, 3))
    labels, label_count = ndimage.label(tiled, structure=_CONNECTIVITY_6)
    if label_count == 0:
        return False

    central_labels = np.unique(labels[:, ny : 2 * ny, nx : 2 * nx][mask])
    central_labels = central_labels[central_labels != 0]
    for label in central_labels:
        _z, _y, x = np.nonzero(labels == label)
        if x.size and int(x.min()) < nx and int(x.max()) >= 2 * nx:
            return True
    return False


def minimum_cross_section_fraction(mask_zyx: np.ndarray) -> float:
    """Return the minimum true-fraction in any yz cross-section along x."""

    fractions = _cross_section_fractions(mask_zyx)
    if fractions.size == 0:
        return 0.0
    return float(np.min(fractions))


def minimum_cross_section_index(mask_zyx: np.ndarray) -> int:
    """Return the x index of the minimum true-fraction yz cross-section."""

    fractions = _cross_section_fractions(mask_zyx)
    if fractions.size == 0:
        return 0
    return int(np.argmin(fractions))


def pore_component_summary(mask_zyx: np.ndarray) -> tuple[int, float]:
    """Return x/y-periodic, z-finite pore-domain count and largest fraction."""

    mask = _as_bool_zyx(mask_zyx)
    phase_count = int(np.count_nonzero(mask))
    if phase_count == 0:
        return 0, 0.0
    labels, label_count = ndimage.label(mask, structure=_CONNECTIVITY_6)
    if label_count == 0:
        return 0, 0.0
    parents = np.arange(label_count + 1, dtype=np.int32)
    _merge_periodic_face_labels(parents, labels[:, :, 0], labels[:, :, -1])
    _merge_periodic_face_labels(parents, labels[:, 0, :], labels[:, -1, :])
    phase_roots = np.array([_find_root(parents, label) for label in labels[mask]], dtype=np.int32)
    unique_roots, counts = np.unique(phase_roots[phase_roots != 0], return_counts=True)
    return int(unique_roots.size), float(np.max(counts) / phase_count)


def _merge_periodic_face_labels(parents: np.ndarray, first: np.ndarray, second: np.ndarray) -> None:
    for first_label, second_label in zip(first.ravel(), second.ravel(), strict=True):
        if first_label != 0 and second_label != 0:
            _union_roots(parents, int(first_label), int(second_label))


def _find_root(parents: np.ndarray, label: int) -> int:
    root = int(label)
    while int(parents[root]) != root:
        root = int(parents[root])
    while int(parents[label]) != root:
        parent = int(parents[label])
        parents[label] = root
        label = parent
    return root


def _union_roots(parents: np.ndarray, first: int, second: int) -> None:
    first_root = _find_root(parents, first)
    second_root = _find_root(parents, second)
    if first_root != second_root:
        parents[second_root] = first_root


def _cross_section_fractions(mask_zyx: np.ndarray) -> np.ndarray:
    mask = _as_bool_zyx(mask_zyx)
    if 0 in mask.shape:
        return np.array([], dtype=float)
    return np.mean(mask, axis=(0, 1))


def _as_bool_zyx(mask_zyx: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask_zyx, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("mask_zyx must have shape (z, y, x)")
    return mask
