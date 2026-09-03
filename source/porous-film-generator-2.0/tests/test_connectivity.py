from __future__ import annotations

import numpy as np

from porous_film.metrics import (
    minimum_cross_section_fraction,
    periodic_percolates_x,
    pore_z_connectivity_summary,
)


def test_periodic_x_percolation_detects_wrapped_path() -> None:
    semiconductor = np.zeros((3, 3, 4), dtype=bool)
    semiconductor[1, 1, :] = True

    assert periodic_percolates_x(semiconductor)


def test_periodic_x_percolation_uses_y_periodic_adjacency() -> None:
    semiconductor = np.zeros((1, 3, 4), dtype=bool)
    semiconductor[0, 0, 0] = True
    semiconductor[0, 2, :] = True

    assert periodic_percolates_x(semiconductor)


def test_minimum_cross_section_uses_yz_area() -> None:
    semiconductor = np.ones((2, 4, 5), dtype=bool)
    semiconductor[:, :, 2] = False

    assert np.isclose(minimum_cross_section_fraction(semiconductor), 0.0)


def test_pore_z_connectivity_counts_all_and_through_periodic_components() -> None:
    pore = np.zeros((5, 5, 5), dtype=bool)
    pore[:, 2, 2] = True
    pore[2, 0, 0] = True
    pore[2, -1, 0] = True

    summary = pore_z_connectivity_summary(pore)

    assert summary.component_count == 2
    assert summary.through_component_count == 1
    assert not summary.all_components_through


def test_pore_z_connectivity_merges_periodic_xy_faces() -> None:
    pore = np.zeros((4, 3, 4), dtype=bool)
    pore[:2, 1, 0] = True
    pore[1:, 1, -1] = True

    summary = pore_z_connectivity_summary(pore)

    assert summary.component_count == 1
    assert summary.through_component_count == 1
    assert summary.all_components_through
