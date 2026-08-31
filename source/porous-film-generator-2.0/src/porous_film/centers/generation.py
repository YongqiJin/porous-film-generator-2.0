from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import qmc

from porous_film.config import GeneratorConfig

_RDF_BIN_COUNT = 64
_RDF_REFERENCE_MIN_POWER = 7
_RDF_MONTE_CARLO_STEPS_PER_POINT = 24
DEFAULT_MAX_REFERENCE_PAIRS = 65_536
_RDF_TARGET_NEGATIVE_TOLERANCE = 1e-10


@dataclass(frozen=True)
class CenterSeedPlan:
    """Latent center-anchor seed plan.

    `target_rdf_xi` is the reduced dimensionless RDF coordinate. The
    `starting_loss` records the pre-Monte-Carlo RDF-mode loss; `initialization_loss`
    is the final seed loss after initialization/optimization.
    """

    intended_points_A: np.ndarray
    target_rdf_xi: np.ndarray
    target_rdf_values: np.ndarray
    starting_loss: float
    initialization_loss: float
    distance_coordinate: str = "dimensionless_3d"


def pair_distances_periodic_xy(points_A: np.ndarray, target_box_A: np.ndarray) -> np.ndarray:
    points = _as_points(points_A)
    box = _as_box(target_box_A)
    point_count = points.shape[0]
    if point_count < 2:
        return np.array([], dtype=float)

    row, col = np.triu_indices(point_count, k=1)
    deltas = points[col] - points[row]
    deltas[:, 0] -= box[0] * np.round(deltas[:, 0] / box[0])
    deltas[:, 1] -= box[1] * np.round(deltas[:, 1] / box[1])
    return np.linalg.norm(deltas, axis=1)


def pair_distances_periodic_xy_plane(
    points_A: np.ndarray,
    box_xy_A: np.ndarray,
) -> np.ndarray:
    points = _as_points(points_A)
    box_xy = np.asarray(box_xy_A, dtype=float)
    if box_xy.shape != (2,) or np.any(box_xy <= 0.0):
        raise ValueError("box_xy_A must contain two positive lengths")
    if points.shape[0] < 2:
        return np.array([], dtype=float)
    row, col = np.triu_indices(points.shape[0], k=1)
    delta = points[col, :2] - points[row, :2]
    delta -= box_xy * np.round(delta / box_xy)
    return np.linalg.norm(delta, axis=1)


def generate_lattice_jitter(
    count: int,
    box_A: np.ndarray,
    lattice: str,
    jitter_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if count < 0:
        raise ValueError("count must be nonnegative")
    if jitter_fraction < 0:
        raise ValueError("jitter_fraction must be nonnegative")
    if count == 0:
        return np.empty((0, 3), dtype=float)

    box = _as_box(box_A)
    basis = _lattice_basis(lattice)
    cells_per_axis = int(np.ceil((count / len(basis)) ** (1.0 / 3.0)))
    cells_per_axis = max(cells_per_axis, 1)
    cell = box / cells_per_axis

    fractions = []
    for ix in range(cells_per_axis):
        for iy in range(cells_per_axis):
            for iz in range(cells_per_axis):
                cell_origin = np.array([ix, iy, iz], dtype=float)
                for basis_fraction in basis:
                    fractions.append((cell_origin + basis_fraction) / cells_per_axis)
    points = np.asarray(fractions, dtype=float)[:count] * box

    if jitter_fraction:
        half_width = 0.5 * jitter_fraction * float(np.min(cell))
        points = points + rng.uniform(-half_width, half_width, size=points.shape)
        points[:, 0] = np.mod(points[:, 0], box[0])
        points[:, 1] = np.mod(points[:, 1], box[1])
        points[:, 2] = np.clip(points[:, 2], 0.0, box[2])
    return points


def evaluate_rdf_target(xi: np.ndarray, components: list[dict]) -> np.ndarray:
    xi_values = np.asarray(xi, dtype=float)
    if not np.all(np.isfinite(xi_values)):
        raise ValueError("RDF xi values must be finite")

    values = np.ones_like(xi_values, dtype=float)
    for component in components:
        kind, amplitude, center, width = _rdf_component_parameters(component)
        gaussian = np.exp(-0.5 * ((xi_values - center) / width) ** 2)
        if kind == "peak":
            values += amplitude * gaussian
        elif kind in {"dip", "exclusion"}:
            values -= amplitude * gaussian
        elif kind == "oscillation":
            phase = 2.0 * np.pi * (xi_values - center) / width
            values += amplitude * gaussian * np.cos(phase)
        else:
            raise ValueError(f"unsupported RDF component kind: {kind}")

    if not np.all(np.isfinite(values)):
        raise ValueError("RDF target values must be finite")
    if np.any(values < -_RDF_TARGET_NEGATIVE_TOLERANCE):
        raise ValueError("RDF target must be nonnegative")
    return np.maximum(values, 0.0)


def generate_center_seeds(config: GeneratorConfig, rng: np.random.Generator) -> CenterSeedPlan:
    count = config.seed_count
    box = _box_from_config(config)
    if config.source_schema_version == 3:
        xi = _xy_distance_grid_A(config, box)
        target = evaluate_rdf_target(xi, _formal_xy_components(config))
        xy_only = True
        coordinate_name = "angstrom_xy"
    else:
        xi = _rdf_xi_grid(count, box)
        target = evaluate_rdf_target(xi, _config_rdf_components(config))
        xy_only = False
        coordinate_name = "dimensionless_3d"

    mode = config.center_distribution.mode
    if mode == "lattice_jitter":
        lattice = config.center_distribution.lattice or "simple_cubic"
        points = generate_lattice_jitter(
            count=count,
            box_A=box,
            lattice=lattice,
            jitter_fraction=config.center_distribution.position_jitter,
            rng=rng,
        )
        starting_loss = _rdf_loss(points, box, xi, target, xy_only=xy_only)
        final_loss = starting_loss
    elif mode == "rdf":
        points, starting_loss, final_loss = _generate_rdf_optimized_points(
            count,
            box,
            xi,
            target,
            rng,
            xy_only=xy_only,
        )
    else:
        raise ValueError(f"unsupported center distribution mode: {mode}")

    return CenterSeedPlan(
        intended_points_A=points,
        target_rdf_xi=xi,
        target_rdf_values=target,
        starting_loss=starting_loss,
        initialization_loss=final_loss,
        distance_coordinate=coordinate_name,
    )


def generate_centers(config: GeneratorConfig, rng: np.random.Generator) -> CenterSeedPlan:
    return generate_center_seeds(config, rng)


def _as_points(points_A: np.ndarray) -> np.ndarray:
    points = np.asarray(points_A, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_A must have shape (n, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_A must be finite")
    return points


def _as_box(box_A: np.ndarray) -> np.ndarray:
    box = np.asarray(box_A, dtype=float)
    if box.shape != (3,):
        raise ValueError("box_A must have shape (3,)")
    if not np.all(np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError("box_A must contain positive finite lengths")
    return box


def _lattice_basis(lattice: str) -> np.ndarray:
    if lattice == "simple_cubic":
        return np.array([[0.5, 0.5, 0.5]], dtype=float)
    if lattice == "bcc":
        return np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]], dtype=float)
    if lattice == "fcc":
        return np.array(
            [
                [0.25, 0.25, 0.25],
                [0.25, 0.75, 0.75],
                [0.75, 0.25, 0.75],
                [0.75, 0.75, 0.25],
            ],
            dtype=float,
        )
    raise ValueError("lattice must be simple_cubic, bcc, or fcc")


def _rdf_component_parameters(component: Any) -> tuple[str, float, float, float]:
    if hasattr(component, "model_dump"):
        data = component.model_dump(exclude_none=True)
    else:
        data = dict(component)

    kind = str(data.get("kind", "peak"))
    amplitude = float(data.get("amplitude", data.get("weight", 1.0)))
    center = float(data.get("center", data.get("center_xi", 0.0)))
    width = float(data.get("width", data.get("width_xi", 0.0)))
    if kind not in {"peak", "dip", "exclusion", "oscillation"}:
        raise ValueError(f"unsupported RDF component kind: {kind}")
    if not np.all(np.isfinite([amplitude, center, width])):
        raise ValueError("RDF component parameters must be finite")
    if amplitude < 0.0:
        raise ValueError("RDF component amplitude must be nonnegative")
    if width <= 0.0:
        raise ValueError("RDF component width must be positive")
    return kind, amplitude, center, width


def _box_from_config(config: GeneratorConfig) -> np.ndarray:
    target = config.film.target_box_A
    return np.array([target.x, target.y, target.z], dtype=float)


def _config_rdf_components(config: GeneratorConfig) -> list[Any]:
    return list(config.center_distribution.rdf)


def _rdf_xi_grid(count: int, box: np.ndarray) -> np.ndarray:
    if count <= 0:
        return np.linspace(0.0, 1.0, _RDF_BIN_COUNT, dtype=float)
    number_density = count / float(np.prod(box))
    density_scale = number_density ** (1.0 / 3.0)
    max_distance = np.sqrt((0.5 * box[0]) ** 2 + (0.5 * box[1]) ** 2 + box[2] ** 2)
    edges = np.linspace(0.0, max_distance * density_scale, _RDF_BIN_COUNT + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def _xy_distance_grid_A(config: GeneratorConfig, box: np.ndarray) -> np.ndarray:
    maximum = config.measurement.center_distance_max_A
    if maximum is None:
        maximum = float(np.linalg.norm(0.5 * box[:2]))
    width = float(config.measurement.center_distance_bin_width_A)
    count = max(1, int(np.ceil(float(maximum) / width)))
    edges = np.linspace(0.0, count * width, count + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def _formal_xy_components(config: GeneratorConfig) -> list[dict[str, float | str]]:
    target = config.formal_targets.position_quantity.center_distance_xy
    if target is None:
        return []
    return [
        {
            "kind": component.kind,
            "amplitude": component.amplitude,
            "center": component.center_A,
            "width": component.width_A,
        }
        for component in target.components
    ]


def _generate_rdf_optimized_points(
    count: int,
    box: np.ndarray,
    xi: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
    *,
    xy_only: bool = False,
) -> tuple[np.ndarray, float, float]:
    if count <= 0:
        return np.empty((0, 3), dtype=float), 0.0, 0.0

    points = rng.random((count, 3)) * box
    starting_loss = _rdf_loss(points, box, xi, target, xy_only=xy_only)
    current_loss = starting_loss
    best_points = points.copy()
    best_loss = current_loss
    steps = max(_RDF_MONTE_CARLO_STEPS_PER_POINT * count, 1)
    base_step = 0.15 * float(np.min(box))

    for step in range(steps):
        progress = step / max(steps - 1, 1)
        temperature = 0.05 * (1.0 - progress) + 1e-4 * progress
        step_scale = base_step * (1.0 - 0.75 * progress)
        index = int(rng.integers(0, count))
        trial = points.copy()
        trial[index] += rng.normal(0.0, step_scale, size=3)
        trial[index, 0] = trial[index, 0] % box[0]
        trial[index, 1] = trial[index, 1] % box[1]
        trial[index, 2] = np.clip(trial[index, 2], 0.0, box[2])

        trial_loss = _rdf_loss(trial, box, xi, target, xy_only=xy_only)
        if trial_loss <= current_loss or rng.random() < np.exp((current_loss - trial_loss) / temperature):
            points = trial
            current_loss = trial_loss
            if trial_loss <= best_loss:
                best_points = trial.copy()
                best_loss = trial_loss

    return best_points, starting_loss, best_loss


def _rdf_loss(
    points: np.ndarray,
    box: np.ndarray,
    xi: np.ndarray,
    target: np.ndarray,
    *,
    xy_only: bool = False,
) -> float:
    if points.shape[0] < 2:
        return 0.0
    observed = (
        _periodic_xy_rdf(points, box[:2], xi)
        if xy_only
        else _finite_z_corrected_rdf(points, box, xi)
    )
    return float(np.mean((observed - target) ** 2))


def _periodic_xy_rdf(
    points: np.ndarray,
    box_xy_A: np.ndarray,
    distance_centers_A: np.ndarray,
) -> np.ndarray:
    distances = pair_distances_periodic_xy_plane(points, box_xy_A)
    if distances.size == 0:
        return np.ones_like(distance_centers_A, dtype=float)
    edges = _edges_from_centers(distance_centers_A)
    observed_hist = np.histogram(distances, bins=edges)[0].astype(float)
    reference_hist = _sobol_xy_reference_histogram(
        points.shape[0],
        box_xy_A,
        edges,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = observed_hist / reference_hist
    corrected[~np.isfinite(corrected)] = 0.0
    positive = corrected[reference_hist > 0.0]
    if positive.size and float(np.mean(positive)) > 0.0:
        corrected /= float(np.mean(positive))
    return corrected


def _edges_from_centers(centers: np.ndarray) -> np.ndarray:
    if centers.size <= 1:
        width = 1.0
    else:
        width = float(np.median(np.diff(centers)))
    return np.concatenate(([max(0.0, centers[0] - 0.5 * width)], centers + 0.5 * width))


def _sobol_xy_reference_histogram(
    point_count: int,
    box_xy_A: np.ndarray,
    distance_edges_A: np.ndarray,
    max_reference_pairs: int = DEFAULT_MAX_REFERENCE_PAIRS,
) -> np.ndarray:
    if point_count < 2:
        return np.zeros(distance_edges_A.size - 1, dtype=float)
    total_pairs = point_count * (point_count - 1) // 2
    sampled_pairs = min(max_reference_pairs, total_pairs)
    power = max(_RDF_REFERENCE_MIN_POWER, int(np.ceil(np.log2(sampled_pairs))))
    samples = qmc.Sobol(d=4, scramble=False).random_base2(power)[:sampled_pairs]
    first = samples[:, :2] * box_xy_A
    second = samples[:, 2:] * box_xy_A
    delta = second - first
    delta -= box_xy_A * np.round(delta / box_xy_A)
    reference_hist = np.histogram(np.linalg.norm(delta, axis=1), bins=distance_edges_A)[0]
    return reference_hist.astype(float) * (total_pairs / sampled_pairs)


def _finite_z_corrected_rdf(points: np.ndarray, box: np.ndarray, xi: np.ndarray) -> np.ndarray:
    distances = pair_distances_periodic_xy(points, box)
    if distances.size == 0:
        return np.ones_like(xi, dtype=float)

    distance_edges = _distance_edges_from_xi(xi, points.shape[0], box)
    observed_hist, _ = np.histogram(distances, bins=distance_edges)
    reference_hist = _sobol_reference_histogram(points.shape[0], box, distance_edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = observed_hist / reference_hist
    corrected[~np.isfinite(corrected)] = 0.0
    positive = corrected[reference_hist > 0.0]
    if positive.size and float(np.mean(positive)) > 0.0:
        corrected = corrected / float(np.mean(positive))
    return corrected.astype(float)


def _distance_edges_from_xi(xi: np.ndarray, count: int, box: np.ndarray) -> np.ndarray:
    if count <= 0:
        density_scale = 1.0
    else:
        density_scale = (count / float(np.prod(box))) ** (1.0 / 3.0)
    spacing = float(np.median(np.diff(xi))) if xi.size > 1 else 1.0
    xi_edges = np.concatenate(([max(0.0, xi[0] - 0.5 * spacing)], xi + 0.5 * spacing))
    return xi_edges / density_scale


def _sobol_reference_histogram(
    point_count: int,
    box: np.ndarray,
    distance_edges: np.ndarray,
    max_reference_pairs: int = DEFAULT_MAX_REFERENCE_PAIRS,
) -> np.ndarray:
    if max_reference_pairs <= 0:
        raise ValueError("max_reference_pairs must be positive")
    if point_count < 2:
        return np.zeros(distance_edges.size - 1, dtype=float)

    total_pairs = point_count * (point_count - 1) // 2
    sampled_pairs = min(max_reference_pairs, total_pairs)
    sobol_power = max(_RDF_REFERENCE_MIN_POWER, int(np.ceil(np.log2(sampled_pairs))))
    sobol = qmc.Sobol(d=6, scramble=False)
    unit_pairs = sobol.random_base2(sobol_power)[:sampled_pairs]
    first = unit_pairs[:, :3] * box
    second = unit_pairs[:, 3:] * box
    reference_distances = _paired_distances_periodic_xy(first, second, box)
    reference_hist, _ = np.histogram(reference_distances, bins=distance_edges)
    pair_scale = total_pairs / sampled_pairs
    return reference_hist.astype(float) * pair_scale


def _paired_distances_periodic_xy(
    first_points_A: np.ndarray,
    second_points_A: np.ndarray,
    box_A: np.ndarray,
) -> np.ndarray:
    first = _as_points(first_points_A)
    second = _as_points(second_points_A)
    if first.shape != second.shape:
        raise ValueError("paired point arrays must have matching shape")
    box = _as_box(box_A)
    deltas = second - first
    deltas[:, 0] -= box[0] * np.round(deltas[:, 0] / box[0])
    deltas[:, 1] -= box[1] * np.round(deltas[:, 1] / box[1])
    return np.linalg.norm(deltas, axis=1)
