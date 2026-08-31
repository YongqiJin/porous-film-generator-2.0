# 3D Porous Film Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build a tested Python CLI that generates auditable 3D porous-film geometry, fills the pore phase with rigid PDB molecules, exports a Blender-readable semiconductor solid, and emits stable files for downstream Packmol and Bayesian optimization.

**Architecture:** The generator uses finite-dimensional configuration models, analytic/approximate signed-distance fields for compact and channel pore units, chunked voxelization for phase analysis, and a staged candidate/audit pipeline. Geometry generation and reporting live in the porous_film package; an independent porous_film_validator package reads only neutral exported files and must never import porous_film.

**Tech Stack:** Python 3.12 managed by uv; NumPy, SciPy, Pydantic v2, PyYAML, h5py, scikit-image, trimesh, gemmi, Typer, Rich, pytest, pytest-cov, and Ruff.

## Global Constraints

- Source root: C:\Calculation_assist\porous-film-generator.
- Calculation artifacts: C:\Calculation_results\YYYY-MM-DD\python_results\<task-name>\ with inputs, work, outputs, analysis, reports, and logs.
- Internal geometry units are Å, Å², and Å³; density is g/cm³.
- x/y are periodic; z is open.
- Target-box dimensions define morphology statistics. Packing-box x/y must match target x/y; packing z must be at least target z.
- Pore units may overlap, merge, and percolate. Semiconductor must percolate in x and satisfy the configured minimum cross-section.
- Supported pore units in v1: axisymmetric compact superellipsoids and unbranched non-closed spline channels.
- Supported statistical inputs are parametric distributions and finite convex mixtures. Tabulated PDF/CDF input is forbidden.
- Pore material molecules are rigid copies of one PDB template. Their centers of mass stay in target pores; atoms may extend into z padding.
- Required visualization output: semiconductor_solid_target.glb, clipped to the target box, with semiconductor solid and pore void.
- Main code and validator code may share third-party dependencies and file schemas, but validator code must not import porous_film.
- All new production behavior follows strict red-green-refactor TDD.
- Every task ends with focused tests and a commit.
- No calculation output is written into the Git repository.

---

### Task 1: Project scaffold and executable test harness

**Files:**
- Create: pyproject.toml
- Create: README.md
- Create: .gitignore
- Create: src/porous_film/__init__.py
- Create: src/porous_film/cli.py
- Create: tests/test_cli_smoke.py

**Interfaces:**
- Produces console command: porous-film.
- Produces Python constant: porous_film.__version__.
- Later tasks may add subcommands to cli.app.

- [ ] **Step 1: Write the failing CLI smoke test**

~~~python
from typer.testing import CliRunner

from porous_film.cli import app


runner = CliRunner()


def test_version_command_reports_package_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "porous-film 0.1.0"
~~~

- [ ] **Step 2: Run the test and verify RED**

Run:

~~~powershell
uv run --python 3.12 pytest tests/test_cli_smoke.py -v
~~~

Expected: collection fails because porous_film does not exist.

- [ ] **Step 3: Create pyproject.toml and minimal package**

pyproject.toml must contain:

~~~toml
[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[project]
name = "porous-film-generator"
version = "0.1.0"
description = "Auditable 3D porous-film geometry and pore-phase molecule generator"
requires-python = ">=3.12"
dependencies = [
  "gemmi>=0.7.3",
  "h5py>=3.12",
  "numpy>=2.1",
  "pydantic>=2.10",
  "pyyaml>=6.0.2",
  "rich>=13.9",
  "scikit-image>=0.25",
  "scipy>=1.15",
  "trimesh>=4.6",
  "typer>=0.15",
]

[project.scripts]
porous-film = "porous_film.cli:app"

[dependency-groups]
dev = [
  "pytest>=8.3",
  "pytest-cov>=6.0",
  "ruff>=0.9",
]

[tool.hatch.build.targets.wheel]
packages = ["src/porous_film"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 100
target-version = "py312"
~~~

src/porous_film/__init__.py:

~~~python
__version__ = "0.1.0"
~~~

src/porous_film/cli.py:

~~~python
import typer

from porous_film import __version__


app = typer.Typer(no_args_is_help=True)


@app.command()
def version() -> None:
    typer.echo(f"porous-film {__version__}")
~~~

.gitignore must include:

~~~gitignore
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
.worktrees/
.superpowers/
~~~

- [ ] **Step 4: Sync dependencies and verify GREEN**

Run:

~~~powershell
uv sync --python 3.12
uv run pytest tests/test_cli_smoke.py -v
uv run ruff check src tests
~~~

Expected: one test passes and Ruff exits 0.

- [ ] **Step 5: Commit**

~~~powershell
git add pyproject.toml README.md .gitignore src tests
git commit -m "build: scaffold porous film generator"
~~~

---

### Task 2: Configuration models and parametric mixture distributions

**Files:**
- Create: src/porous_film/config/__init__.py
- Create: src/porous_film/config/models.py
- Create: src/porous_film/distributions/__init__.py
- Create: src/porous_film/distributions/core.py
- Create: tests/test_config.py
- Create: tests/test_distributions.py

**Interfaces:**
- Produces: load_config(path: Path) -> GeneratorConfig.
- Produces: allocate_largest_remainder(weights, total) -> ndarray[int].
- Produces: stratified_sample(spec, count, rng) -> ndarray[float].
- Produces: mixture_cdf(spec, values) -> ndarray[float].
- GeneratorConfig is consumed by all later tasks.

- [ ] **Step 1: Write failing configuration tests**

~~~python
from pathlib import Path

import pytest

from porous_film.config import GeneratorConfig, load_config


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
~~~

- [ ] **Step 2: Write failing distribution tests**

~~~python
import numpy as np

from porous_film.distributions import (
    allocate_largest_remainder,
    stratified_sample,
)


def test_largest_remainder_preserves_total_and_weights() -> None:
    counts = allocate_largest_remainder(np.array([0.6, 0.3, 0.1]), 17)

    assert counts.tolist() == [10, 5, 2]
    assert int(counts.sum()) == 17


def test_stratified_mixture_is_reproducible_and_covers_both_modes() -> None:
    spec = {
        "family": "mixture",
        "components": [
            {"weight": 0.5, "family": "constant", "value": 0.5},
            {"weight": 0.5, "family": "constant", "value": 2.0},
        ],
    }

    first = stratified_sample(spec, 8, np.random.default_rng(12))
    second = stratified_sample(spec, 8, np.random.default_rng(12))

    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == [0.5] * 4 + [2.0] * 4
~~~

- [ ] **Step 3: Run focused tests and verify RED**

~~~powershell
uv run pytest tests/test_config.py tests/test_distributions.py -v
~~~

Expected: imports fail because modules are missing.

- [ ] **Step 4: Implement Pydantic models and distributions**

GeneratorConfig must include exact nested models for:

- TaskSpec
- Box3D and FilmSpec
- GlobalPoreSpec
- RDFComponent and CenterDistributionSpec
- DistributionSpec and MixtureComponent
- CompactPoreSpec and ChannelPoreSpec
- MatrixConstraintSpec
- PoreMaterialSpec
- AuditSpec
- OutputSpec

Validation must enforce:

- positive box lengths;
- packing x/y equal target x/y;
- packing z at least target z;
- exactly one of explicit packing_box_A or z_padding_A;
- explicit packing x/y equal target x/y;
- normalized target_origin_in_packing_A equals [0, 0, lower_padding];
- porosity strictly between 0 and 1;
- channel fraction in [0,1];
- positive density or exact molecule count, but not both;
- mixture weights nonnegative and summing to 1 within 1e-9;
- no tabulated-distribution fields;
- eta and tau supports starting at 1;
- deterministic seed_count property;
- OptimizationSpec.seed_panel is a non-empty tuple of unique integers and defaults to the task random seed.
- load_config resolves relative PDB paths against the YAML parent directory.

Distribution implementation must use scipy.stats for lognormal, gamma, Weibull, truncated normal, and beta; constant distributions are handled directly. Stratified samples use one random quantile inside each equal-probability stratum, followed by a random permutation.

- [ ] **Step 5: Verify GREEN**

~~~powershell
uv run pytest tests/test_config.py tests/test_distributions.py -v
uv run ruff check src tests
~~~

- [ ] **Step 6: Commit**

~~~powershell
git add src/porous_film/config src/porous_film/distributions tests/test_config.py tests/test_distributions.py
git commit -m "feat: add configuration and mixture distributions"
~~~

---

### Task 3: Periodic center generation and RDF optimization

**Files:**
- Create: src/porous_film/centers/__init__.py
- Create: src/porous_film/centers/generation.py
- Create: tests/test_centers.py

**Interfaces:**
- CenterSeedPlan(intended_points_A: ndarray, target_rdf_xi: ndarray, target_rdf_values: ndarray, initialization_loss: float).
- pair_distances_periodic_xy(points_A: ndarray, target_box_A: ndarray) -> ndarray.
- generate_lattice_jitter(count: int, box_A: ndarray, lattice: str, jitter_fraction: float, rng: numpy.random.Generator) -> ndarray.
- evaluate_rdf_target(xi: ndarray, components: list[dict]) -> ndarray.
- generate_center_seeds(config: GeneratorConfig, rng: numpy.random.Generator) -> CenterSeedPlan.
- Intended center seeds are latent anchors only. Final RDF acceptance uses realized anchors returned by Task 4.

- [ ] **Step 1: Write failing minimum-image and lattice tests**

~~~python
import numpy as np

from porous_film.centers import (
    generate_lattice_jitter,
    pair_distances_periodic_xy,
)


def test_pair_distance_wraps_x_and_y_but_not_z() -> None:
    points = np.array([[0.5, 0.5, 0.5], [9.5, 9.5, 9.5]])

    distances = pair_distances_periodic_xy(points, np.array([10.0, 10.0, 10.0]))

    assert np.allclose(distances, [np.sqrt(83.0)])


def test_simple_cubic_lattice_is_reproducible() -> None:
    first = generate_lattice_jitter(
        count=8,
        box_A=np.array([20.0, 20.0, 20.0]),
        lattice="simple_cubic",
        jitter_fraction=0.0,
        rng=np.random.default_rng(3),
    )
    second = generate_lattice_jitter(
        count=8,
        box_A=np.array([20.0, 20.0, 20.0]),
        lattice="simple_cubic",
        jitter_fraction=0.0,
        rng=np.random.default_rng(3),
    )

    assert np.array_equal(first, second)
    assert first.shape == (8, 3)
~~~

- [ ] **Step 2: Write failing RDF target test**

~~~python
import numpy as np

from porous_film.centers import evaluate_rdf_target


def test_rdf_linear_components_return_one_at_long_range() -> None:
    xi = np.array([0.0, 1.0, 10.0])
    components = [
        {"kind": "exclusion", "amplitude": -1.0, "center": 0.0, "width": 0.3},
        {"kind": "peak", "amplitude": 1.2, "center": 1.0, "width": 0.1},
    ]

    values = evaluate_rdf_target(xi, components)

    assert values[0] >= 0.0
    assert values[1] > 1.0
    assert np.isclose(values[-1], 1.0, atol=1e-6)
~~~

- [ ] **Step 3: Verify RED**

~~~powershell
uv run pytest tests/test_centers.py -v
~~~

- [ ] **Step 4: Implement center generation**

Implementation requirements:

- periodic minimum-image correction on x/y only;
- finite-z RDF correction by dividing observed pair histograms by a deterministic Sobol uniform-reference histogram in the same box;
- compact anchor is its center;
- channel anchor is supplied later as unfolded arclength centroid;
- RDF target components are analytic Gaussian-like peaks/dips and smooth exclusion terms;
- target preflight rejects negative values below -1e-10 and clips roundoff to zero;
- RDF optimizer uses Monte Carlo point moves with a deterministic temperature schedule and keeps z in [0,Lz];
- lattice mode supports simple_cubic, bcc, and fcc, selecting or truncating sites deterministically before jitter;
- generate_centers dispatches by config mode.

- [ ] **Step 5: Verify GREEN**

~~~powershell
uv run pytest tests/test_centers.py -v
uv run ruff check src tests
~~~

- [ ] **Step 6: Commit**

~~~powershell
git add src/porous_film/centers tests/test_centers.py
git commit -m "feat: generate periodic pore centers"
~~~

---

### Task 4: Compact and channel SDF geometry

**Files:**
- Create: src/porous_film/geometry/__init__.py
- Create: src/porous_film/geometry/sdf.py
- Create: tests/test_geometry.py

**Interfaces:**
- CompactUnit.sphere(unit_id: str, center_A: ndarray, radius_A: float) -> CompactUnit.
- ChannelUnit.from_polyline(unit_id: str, control_points_unwrapped_A: ndarray, cross_radius_A: float, roughness: float) -> ChannelUnit.
- PoreUnit.sdf(points_A: ndarray) -> ndarray and PoreUnit.to_record() -> dict.
- PoreGeometry(units: list[PoreUnit], target_box_A: ndarray).
- PoreGeometry.sdf(points_A: ndarray) -> ndarray returns the smooth periodic union.
- BuiltGeometry(geometry: PoreGeometry, units: list[PoreUnit], realized_anchors_A: ndarray, latent_to_realized_ids: dict[str, str]).
- build_units(config: GeneratorConfig, center_plan: CenterSeedPlan, rng: numpy.random.Generator) -> BuiltGeometry.
- Compact realized anchor is its geometric center. Channel realized anchor is the arclength-weighted centroid of the unfolded centerline.

- [ ] **Step 1: Write failing compact-unit tests**

~~~python
import numpy as np

from porous_film.geometry import CompactUnit


def test_spherical_compact_sdf_matches_radius() -> None:
    unit = CompactUnit.sphere(
        unit_id="compact-0001",
        center_A=np.array([0.0, 0.0, 0.0]),
        radius_A=2.0,
    )

    values = unit.sdf(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
    )

    assert np.allclose(values, [-2.0, 0.0, 1.0], atol=1e-7)
~~~

- [ ] **Step 2: Write failing channel tests**

~~~python
import numpy as np

from porous_film.geometry import ChannelUnit


def test_straight_channel_has_tau_one_and_continuous_negative_sdf() -> None:
    channel = ChannelUnit.from_polyline(
        unit_id="channel-0001",
        control_points_unwrapped_A=np.array(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
        ),
        cross_radius_A=1.0,
        roughness=0.0,
    )

    sample = np.column_stack(
        [np.linspace(0.0, 10.0, 21), np.zeros(21), np.zeros(21)]
    )

    assert np.isclose(channel.tortuosity, 1.0)
    assert np.all(channel.sdf(sample) < 0.0)
~~~

- [ ] **Step 3: Verify RED**

~~~powershell
uv run pytest tests/test_geometry.py -v
~~~

- [ ] **Step 4: Implement geometry**

Implementation requirements:

- CompactUnit uses an axisymmetric superellipsoid implicit norm; exponent 2 gives ellipsoid and sphere tests are exact.
- Orientations use scipy.spatial.transform.Rotation.
- Roughness uses a reproducible finite sum of sinusoidal modes evaluated in normalized local coordinates and subtracted from the base SDF.
- ChannelUnit fits a cubic spline through unfolded control points, samples it adaptively by curvature, places oriented local superellipsoid/capsule segments, and returns their smooth minimum.
- ChannelUnit stores arc length, end distance, eta, tau, anchor, control points, local segment frames, and roughness parameters.
- PoreGeometry applies x/y periodic images by evaluating relevant neighbor images and takes the smooth union.
- Unit records distinguish latent parameters from realized values.

- [ ] **Step 5: Verify GREEN**

~~~powershell
uv run pytest tests/test_geometry.py -v
uv run ruff check src tests
~~~

- [ ] **Step 6: Commit**

~~~powershell
git add src/porous_film/geometry tests/test_geometry.py
git commit -m "feat: add compact and channel pore geometry"
~~~

---

### Task 5: Chunked voxelization and porosity scaling

**Files:**
- Create: src/porous_film/voxel/__init__.py
- Create: src/porous_film/voxel/grid.py
- Create: tests/test_voxel.py

**Interfaces:**
- PhaseGrid(pore_mask: ndarray, origin_A: ndarray, spacing_A: float, target_box_A: ndarray).
- voxelize_geometry(geometry: PoreGeometry, box_A: ndarray, spacing_A: float, max_points_per_chunk: int = 1_000_000) -> PhaseGrid.
- solve_scale_for_porosity(build_at_linear_scale: Callable[[float], BuiltGeometry], target_phi: float, tolerance: float, lower: float = 0.1, upper: float = 10.0) -> tuple[float, BuiltGeometry, PhaseGrid].
- PhaseGrid.write_hdf5(path: Path) and PhaseGrid.read_hdf5(path: Path).
- The porosity scale multiplies every unit linear dimension, rebuilds every SDF, recomputes x/y-periodic target-box-clipped unit volumes, and preserves latent versus realized records.

- [ ] **Step 1: Write failing voxelization test**

~~~python
import numpy as np

from porous_film.geometry import CompactUnit, PoreGeometry
from porous_film.voxel import voxelize_geometry


def test_sphere_voxel_porosity_converges_with_resolution() -> None:
    unit = CompactUnit.sphere(
        unit_id="sphere",
        center_A=np.array([5.0, 5.0, 5.0]),
        radius_A=2.0,
    )
    geometry = PoreGeometry([unit], np.array([10.0, 10.0, 10.0]))

    coarse = voxelize_geometry(geometry, np.array([10.0, 10.0, 10.0]), 0.5)
    fine = voxelize_geometry(geometry, np.array([10.0, 10.0, 10.0]), 0.25)

    exact = 4.0 * np.pi * 2.0**3 / 3.0 / 1000.0
    assert abs(fine.porosity - exact) < abs(coarse.porosity - exact)
~~~

- [ ] **Step 2: Write failing HDF5 round-trip test**

~~~python
from pathlib import Path

import numpy as np

from porous_film.voxel import PhaseGrid


def test_phase_grid_hdf5_round_trip(tmp_path: Path) -> None:
    grid = PhaseGrid(
        pore_mask=np.array([[[False, True], [True, False]]], dtype=bool),
        origin_A=np.array([0.0, 0.0, 0.0]),
        spacing_A=0.5,
        target_box_A=np.array([1.0, 1.0, 0.5]),
    )
    path = tmp_path / "phase.h5"

    grid.write_hdf5(path)
    restored = PhaseGrid.read_hdf5(path)

    assert np.array_equal(restored.pore_mask, grid.pore_mask)
    assert np.array_equal(restored.target_box_A, grid.target_box_A)
~~~

- [ ] **Step 3: Verify RED**

~~~powershell
uv run pytest tests/test_voxel.py -v
~~~

- [ ] **Step 4: Implement chunked voxelization and scaling**

Implementation requirements:

- use voxel-center coordinates;
- evaluate SDF in chunks bounded by max_points_per_chunk;
- store mask axes in z,y,x order and declare that order in HDF5 attributes;
- porosity is pore-mask mean in target box only;
- HDF5 stores spacing, origin, target box, periodic axes, phase encoding, schema version, and compressed mask;
- porosity scaling uses monotonic bisection over one common linear scale applied to all unit volumes;
- reject target porosity if a bracket cannot be found without invalid dimensions.

- [ ] **Step 5: Verify GREEN**

~~~powershell
uv run pytest tests/test_voxel.py -v
uv run ruff check src tests
~~~

- [ ] **Step 6: Commit**

~~~powershell
git add src/porous_film/voxel tests/test_voxel.py
git commit -m "feat: voxelize pores and solve target porosity"
~~~

---

### Task 6: Distribution audit, topology, and local thickness

**Files:**
- Create: src/porous_film/metrics/__init__.py
- Create: src/porous_film/metrics/audit.py
- Create: src/porous_film/metrics/connectivity.py
- Create: src/porous_film/metrics/local_thickness.py
- Create: tests/test_metrics.py
- Create: tests/test_connectivity.py
- Create: tests/test_local_thickness.py

**Interfaces:**
- DistributionComparison(passed: bool, ks: float, normalized_wasserstein: float).
- AuditResult fields: passed, scalar_errors, distribution_results, rdf_result, theta_result, compact_eta_result, channel_eta_result, roughness_result, tau_result, channel_fraction_error, mixture_weight_errors, overlap_fraction, connected_pore_domains, largest_pore_fraction, x_surface_openings, y_surface_openings, z_lower_opening_fraction, z_upper_opening_fraction, minimum_cross_section_fraction, minimum_cross_section_index, warnings.
- compare_samples_to_distribution(samples: ndarray, target: dict, ks_limit: float, normalized_wasserstein_limit: float) -> DistributionComparison.
- audit_target_distributions(config: GeneratorConfig, built: BuiltGeometry, center_plan: CenterSeedPlan, grid: PhaseGrid) -> AuditResult.
- periodic_percolates_x(mask_zyx: ndarray) -> bool.
- minimum_cross_section_fraction(mask_zyx: ndarray) -> float.
- local_thickness_field(mask_zyx: ndarray, spacing_A: float, periodic_xy: bool, max_voxels: int = 64_000_000) -> ndarray.
- ThicknessResult(bin_edges_A: ndarray, probabilities: ndarray, field_A: ndarray, uncertainty_A: float).
- local_thickness_distribution(mask_zyx: ndarray, spacing_A: float, periodic_xy: bool) -> ThicknessResult.

- [ ] **Step 1: Write failing distribution-audit test**

~~~python
import numpy as np

from porous_film.metrics import compare_samples_to_distribution


def test_audit_result_exposes_every_required_metric() -> None:
    from porous_film.metrics import AuditResult

    fields = set(AuditResult.__dataclass_fields__)
    required = {
        "theta_result",
        "compact_eta_result",
        "channel_eta_result",
        "roughness_result",
        "tau_result",
        "channel_fraction_error",
        "mixture_weight_errors",
        "overlap_fraction",
        "connected_pore_domains",
        "largest_pore_fraction",
        "z_lower_opening_fraction",
        "z_upper_opening_fraction",
        "minimum_cross_section_index",
    }

    assert required <= fields


def test_distribution_audit_rejects_shifted_constant_samples() -> None:
    result = compare_samples_to_distribution(
        samples=np.array([2.0, 2.0, 2.0, 2.0]),
        target={"family": "constant", "value": 1.0},
        ks_limit=0.05,
        normalized_wasserstein_limit=0.03,
    )

    assert not result.passed
    assert result.normalized_wasserstein > 0.03
~~~

- [ ] **Step 2: Write failing periodic-connectivity tests**

~~~python
import numpy as np

from porous_film.metrics import (
    minimum_cross_section_fraction,
    periodic_percolates_x,
)


def test_periodic_x_percolation_detects_wrapped_path() -> None:
    semiconductor = np.zeros((3, 3, 4), dtype=bool)
    semiconductor[1, 1, :] = True

    assert periodic_percolates_x(semiconductor)


def test_minimum_cross_section_uses_yz_area() -> None:
    semiconductor = np.ones((2, 4, 5), dtype=bool)
    semiconductor[:, :, 2] = False

    assert np.isclose(minimum_cross_section_fraction(semiconductor), 0.0)
~~~

- [ ] **Step 3: Write failing local-thickness test**

~~~python
import numpy as np

from porous_film.metrics import local_thickness_field


def test_cylindrical_channel_reports_diameter_not_wall_distance() -> None:
    z, y, x = np.indices((9, 9, 9))
    mask = (y - 4) ** 2 + (z - 4) ** 2 <= 4

    field = local_thickness_field(mask, spacing_A=1.0, periodic_xy=False)

    assert field[4, 4, 4] >= 4.0
    assert field[4, 4, 1] >= 4.0
~~~

- [ ] **Step 4: Verify RED**

~~~powershell
uv run pytest tests/test_metrics.py tests/test_connectivity.py tests/test_local_thickness.py -v
~~~

- [ ] **Step 5: Implement audits and topology**

Implementation requirements:

- scalar relative/absolute errors;
- KS and normalized first Wasserstein distance against analytic target CDF;
- mixture component-count errors;
- RDF weighted loss plus peak locations;
- 6-neighbor connectivity;
- x percolation by labeling a three-cell x tiling and proving a central component reaches its periodic image;
- minimum yz semiconductor fraction for every x index;
- optional erosion by h_min/2 using EDT before percolation;
- local thickness uses EDT local maxima, quantizes radii to integer voxel bins, processes bins from largest to smallest, and applies precomputed spherical binary-dilation footprints; each voxel is updated only when the current diameter exceeds its stored value;
- x/y periodic thickness uses a 3x3 tile but processes radius bins in z slabs to bound peak memory;
- preflight rejects grids above max_voxels (default 64,000,000) unless the user explicitly raises the limit;
- reported thickness uncertainty is one voxel spacing, and coarse/fine comparisons must agree within two fine voxels;- x/y periodic thickness through 3x3 tiling, returning only the central tile;
- histogram values and probabilities normalized by phase voxel count;
- coarse/fine grid comparison result.

- [ ] **Step 6: Verify GREEN**

~~~powershell
uv run pytest tests/test_metrics.py tests/test_connectivity.py tests/test_local_thickness.py -v
uv run ruff check src tests
~~~

- [ ] **Step 7: Commit**

~~~powershell
git add src/porous_film/metrics tests
git commit -m "feat: audit pore statistics and topology"
~~~

---

### Task 7: Blender GLB and neutral geometry export

**Files:**
- Create: src/porous_film/io/__init__.py
- Create: src/porous_film/io/exporters.py
- Create: tests/test_exporters.py

**Interfaces:**
- export_semiconductor_glb(grid: PhaseGrid, output_path: Path, metadata: dict) -> Path.
- export_surface_ply(grid: PhaseGrid, output_path: Path) -> Path.
- write_qa_contract(contract: dict, output_dir: Path) -> Path.
- voxelize_exported_glb(path: Path, grid: PhaseGrid) -> ndarray is a test/QA helper owned by the io package.

- [ ] **Step 1: Write failing GLB test**

~~~python
from pathlib import Path

import numpy as np
import trimesh

from porous_film.io import export_semiconductor_glb
from porous_film.voxel import PhaseGrid


def test_glb_uses_target_box_and_contains_pore_cavity(tmp_path: Path) -> None:
    pore = np.zeros((10, 10, 10), dtype=bool)
    pore[3:7, 3:7, 3:7] = True
    grid = PhaseGrid(
        pore_mask=pore,
        origin_A=np.zeros(3),
        spacing_A=1.0,
        target_box_A=np.array([10.0, 10.0, 10.0]),
    )
    path = tmp_path / "semiconductor_solid_target.glb"

    export_semiconductor_glb(grid, path, {"length_unit": "angstrom"})
    scene = trimesh.load(path, force="scene")
    bounds = scene.bounds

    assert path.exists()
    assert np.allclose(bounds[0], [0.0, 0.0, 0.0], atol=1.0)
    assert np.allclose(bounds[1], [10.0, 10.0, 10.0], atol=1.0)
    solid_volume = sum(abs(mesh.volume) for mesh in scene.geometry.values())
    assert 0.0 < solid_volume < 1000.0
    assert "SEMICONDUCTOR_SOLID_TARGET" in scene.geometry
    assert scene.metadata["length_unit"] == "angstrom"


def test_glb_preserves_open_z_and_periodic_x_pore_openings(tmp_path: Path) -> None:
    from porous_film.io import voxelize_exported_glb

    pore = np.zeros((10, 10, 10), dtype=bool)
    pore[:, 4:6, 4:6] = True
    pore[4:6, 4:6, 0] = True
    pore[4:6, 4:6, -1] = True
    grid = PhaseGrid(
        pore_mask=pore,
        origin_A=np.zeros(3),
        spacing_A=1.0,
        target_box_A=np.array([10.0, 10.0, 10.0]),
    )
    path = tmp_path / "open-periodic.glb"

    export_semiconductor_glb(
        grid,
        path,
        {
            "length_unit": "angstrom",
            "periodic_axes": ["x", "y"],
            "target_box_A": [10.0, 10.0, 10.0],
        },
    )
    reconstructed_semiconductor = voxelize_exported_glb(path, grid)
    expected_semiconductor = np.logical_not(pore)
    mismatch = np.mean(reconstructed_semiconductor != expected_semiconductor)

    assert mismatch <= 0.05
~~~

- [ ] **Step 2: Verify RED**

~~~powershell
uv run pytest tests/test_exporters.py -v
~~~

- [ ] **Step 3: Implement export**

Requirements:

- construct semiconductor mask as logical_not(pore_mask);
- pad mask for marching cubes while preserving target boundaries;
- use skimage.measure.marching_cubes;
- map z,y,x vertices to x,y,z Å coordinates;
- orient normals from semiconductor to pore/outside;
- create one Trimesh named SEMICONDUCTOR_SOLID_TARGET;
- preserve z-open and x/y-periodic pore openings without caps;
- remove duplicate faces, degenerate triangles, self-intersections detectable by trimesh, and nonphysical cracks;
- assign semi-transparent PBR material;
- export GLB with scene metadata containing target_box_A, periodic_axes, porosity, mesh_resolution_A, and task ID;
- export PLY for QA;
- write contract.json, unit JSONL files, final_phase.h5 references, and checksums.sha256.

- [ ] **Step 4: Verify GREEN**

~~~powershell
uv run pytest tests/test_exporters.py -v
uv run ruff check src tests
~~~

- [ ] **Step 5: Commit**

~~~powershell
git add src/porous_film/io tests/test_exporters.py
git commit -m "feat: export Blender semiconductor solid"
~~~

---

### Task 8: Rigid PDB molecule packing in pore SDF

**Files:**
- Create: src/porous_film/molecules/__init__.py
- Create: src/porous_film/molecules/template.py
- Create: src/porous_film/molecules/packing.py
- Create: tests/fixtures/argon.pdb
- Create: tests/test_molecules.py
- Create: tests/test_packing.py

**Interfaces:**
- MoleculeTemplate.from_pdb(path: Path) -> MoleculeTemplate.
- molecule_count_for_density(template: MoleculeTemplate, pore_volume_A3: float, density_g_cm3: float) -> int.
- PackingConfig(exact_count: int | None = None, target_density_g_cm3: float | None = None, minimum_distance_A: float = 2.0, wall_clearance_A: float = 0.0, max_attempts: int = 100_000).
- PackingResult fields: count, atom_positions_A, instance_transforms, minimum_interatomic_distance_A, actual_density_g_cm3, protrusion_metrics, status.
- pack_molecules(template: MoleculeTemplate, geometry: PoreGeometry, config: PackingConfig, rng: numpy.random.Generator) -> PackingResult.
- PackingResult.write_pdb, write_mmcif, write_instances_csv, write_hdf5, and write_metrics_json.

- [ ] **Step 1: Add a literal one-atom PDB fixture**

tests/fixtures/argon.pdb:

~~~text
HETATM    1 AR   ARG A   1       0.000   0.000   0.000  1.00  0.00          AR
END
~~~

- [ ] **Step 2: Write failing parser and density tests**

~~~python
from pathlib import Path

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
~~~

- [ ] **Step 3: Write failing packing test**

~~~python
from pathlib import Path

import numpy as np

from porous_film.geometry import CompactUnit, PoreGeometry
from porous_film.molecules import MoleculeTemplate
from porous_film.molecules import PackingConfig, pack_molecules


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
~~~

- [ ] **Step 4: Verify RED**

~~~powershell
uv run pytest tests/test_molecules.py tests/test_packing.py -v
~~~

- [ ] **Step 5: Implement molecule handling**

Requirements:

- parse PDB with gemmi and preserve atom/residue identifiers and CONECT when available;
- infer canonical element symbols and provide masses/radii for H, B, C, N, O, F, Si, P, S, Cl, Ar, Se, Br, and I;
- center template on mass center while retaining original transform;
- count from density using 1 Å³ = 1e-24 cm³ and Avogadro constant;
- random quaternion rotations;
- COM candidates sampled from pore voxels weighted toward adequate local thickness;
- atoms within target z must satisfy pore SDF clearance; atoms outside target z are permitted only when COM is in a pore open to that face;
- x/y collision distances use minimum image;
- random sequential insertion followed by bounded Monte Carlo translation/rotation repair;
- raise PackingError with reason algorithm_not_converged or geometrically_infeasible;
- write PDB for handoff and authoritative high-precision mmCIF/HDF5 transforms.

- [ ] **Step 6: Verify GREEN**

~~~powershell
uv run pytest tests/test_molecules.py tests/test_packing.py -v
uv run ruff check src tests
~~~

- [ ] **Step 7: Commit**

~~~powershell
git add src/porous_film/molecules tests/fixtures tests/test_molecules.py tests/test_packing.py
git commit -m "feat: pack rigid molecules into pore phase"
~~~

---

### Task 9: Task directories, pipeline, CLI, reports, and optimizer exchange

**Files:**
- Create: src/porous_film/storage.py
- Create: src/porous_film/pipeline.py
- Create: src/porous_film/reporting/__init__.py
- Create: src/porous_film/reporting/markdown.py
- Create: src/porous_film/optimization/__init__.py
- Create: src/porous_film/optimization/io.py
- Modify: src/porous_film/cli.py
- Create: tests/test_storage.py
- Create: tests/test_pipeline.py
- Create: tests/test_optimization_io.py
- Create: tests/conftest.py

**Interfaces:**
- TaskPaths(root, inputs, work, outputs, analysis, reports, logs, qa_export).
- create_task_directory(result_root: Path, task_name: str, now: datetime) -> TaskPaths.
- PreflightResult(passed: bool, errors: list[str], warnings: list[str], estimated_voxels: int, estimated_memory_bytes: int, report_path: Path | None).
- preflight(config: GeneratorConfig, result_root: Path | None = None) -> PreflightResult.
- GeometryRun(built: BuiltGeometry, phase_grid: PhaseGrid, audit: AuditResult, paths: TaskPaths).
- RunResult(paths: TaskPaths, geometry_run: GeometryRun, packing_result: PackingResult, status: str).
- generate_geometry(config: GeneratorConfig, paths: TaskPaths) -> GeometryRun.
- run_full(config: GeneratorConfig, result_root: Path) -> RunResult.
- write_optimizer_exchange(output_dir: Path, requested: dict, realized: dict, feasible: bool, constraints: dict, calculation_status: dict, objectives: dict, uncertainty: dict) -> None.
- aggregate_seed_results(seed_results: list[dict]) -> dict.
- Multi-seed runs use config.optimization.seed_panel and write one record per seed plus aggregate mean, variance, feasible fraction, and per-seed failure reasons.

- [ ] **Step 1: Create the shared CLI fixture and write the failing storage test**

tests/conftest.py must provide:

~~~python
from pathlib import Path

import pytest


@pytest.fixture
def sample_config_path(tmp_path: Path) -> Path:
    config = tmp_path / "sample.yaml"
    pore_pdb = Path("tests/fixtures/argon.pdb").resolve()
    config.write_text(
        f"""
task:
  name: sample
  random_seed: 11
film:
  target_box_A: {{x: 20, y: 20, z: 20}}
  packing_box_A: {{x: 20, y: 20, z: 30}}
pores:
  seed_number_density_A3: 0.000125
  target_porosity: 0.10
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
  minimum_cross_section_fraction: 0.05
pore_material:
  pdb: "{pore_pdb.as_posix()}"
  molecule_count: 1
geometry_audit:
  candidate_count_per_round: 1
  maximum_rounds: 1
  coarse_spacing_A: 2.0
  fine_spacing_A: 1.0
""",
        encoding="utf-8",
    )
    return config
~~~

Then write the storage test:

~~~python
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from porous_film.storage import create_task_directory


def test_task_directory_uses_shanghai_date_and_unique_suffix(tmp_path: Path) -> None:
    now = datetime(2026, 8, 12, 23, 30, tzinfo=ZoneInfo("Asia/Shanghai"))

    first = create_task_directory(tmp_path, "demo", now)
    second = create_task_directory(tmp_path, "demo", now)

    assert first.root == tmp_path / "2026-08-12" / "python_results" / "demo"
    assert second.root == tmp_path / "2026-08-12" / "python_results" / "demo-02"
    assert first.outputs.is_dir()
    assert first.analysis.is_dir()
    assert first.reports.is_dir()
~~~

- [ ] **Step 2: Write failing optimizer-schema test**

~~~python
import json
from pathlib import Path

from porous_film.optimization import write_optimizer_exchange


def test_seed_panel_aggregation_reports_noise_and_failures() -> None:
    from porous_film.optimization import aggregate_seed_results

    summary = aggregate_seed_results(
        [
            {"seed": 11, "feasible": True, "objective": 2.0},
            {"seed": 22, "feasible": True, "objective": 4.0},
            {"seed": 33, "feasible": False, "failure_reason": "x_not_percolating"},
        ]
    )

    assert summary["objective_mean"] == 3.0
    assert summary["objective_variance"] == 1.0
    assert summary["feasible_fraction"] == 2 / 3
    assert summary["failures"] == {"33": "x_not_percolating"}


def test_optimizer_exchange_separates_requested_and_realized(tmp_path: Path) -> None:
    write_optimizer_exchange(
        tmp_path,
        requested={"target_porosity": 0.25},
        realized={"porosity": 0.247},
        feasible=True,
        constraints={"x_percolates": True},
        objectives={},
        uncertainty={"geometry_replicate_variance": None},
    )

    requested = json.loads((tmp_path / "requested_design_parameters.json").read_text())
    realized = json.loads((tmp_path / "realized_geometry_parameters.json").read_text())

    assert requested["target_porosity"] == 0.25
    assert realized["porosity"] == 0.247
~~~

- [ ] **Step 3: Write failing CLI preflight and full-handoff tests in tests/test_pipeline.py**

~~~python
from pathlib import Path

from typer.testing import CliRunner

from porous_film.cli import app


runner = CliRunner()


def test_preflight_creates_report(sample_config_path: Path) -> None:
    result = runner.invoke(app, ["preflight", "--config", str(sample_config_path)])

    assert result.exit_code == 0
    assert "preflight-report.md" in result.stdout


def test_run_full_writes_every_required_handoff_file(
    sample_config_path: Path,
    tmp_path: Path,
) -> None:
    from porous_film.config import load_config
    from porous_film.pipeline import run_full

    result = run_full(load_config(sample_config_path), tmp_path)
    expected = [
        result.paths.outputs / "semiconductor_solid_target.glb",
        result.paths.outputs / "pore_material.pdb",
        result.paths.outputs / "pore_material_high_precision.cif",
        result.paths.outputs / "molecule_instances.csv",
        result.paths.outputs / "pore_geometry.h5",
        result.paths.outputs / "packing_metrics.json",
        result.paths.outputs / "packmol_handoff.inp",
        result.paths.outputs / "pore_reference_coordinates.cif",
        result.paths.outputs / "phase_mapping.json",
        result.paths.outputs / "pore_atom_indices.ndx",
        result.paths.outputs / "compression_metadata.json",
    ]

    assert all(path.exists() for path in expected)
~~~

- [ ] **Step 4: Verify RED**

~~~powershell
uv run pytest tests/test_storage.py tests/test_pipeline.py tests/test_optimization_io.py -v
~~~

- [ ] **Step 5: Implement orchestration**

Requirements:

- Asia/Shanghai dated result hierarchy;
- copied source inputs and normalized YAML;
- preflight validates box geometry, PDB, distributions, center target, estimated memory, and sample sufficiency;
- candidate loop builds centers, units, porosity scale, coarse grid, target audits, constraints, and fine grid;
- only accepted geometry is packed;
- Markdown reports contain software version, parameters, status, warnings, convergence, and absolute paths;
- CLI subcommands: version, preflight, generate-geometry, fill-pore, generate, audit, audit-packmol-output;
- requested, realized, feasibility, status, objectives, and uncertainty JSON outputs;
- optimization seed panels run deterministically, write per-seed requested/realized/feasibility records, aggregate objective means/variances, feasible fraction, and per-seed failure reasons;
- calculation failures retain logs and machine-readable reasons;
- no silent overwrite;
- qa_export contains contract.json, normalized_config.yaml, unit_candidates.jsonl, unit_geometry.jsonl, channel_curves.h5, final_phase.h5, final_surface.ply, main_unit_metrics.csv, main_metrics.json, molecules/source, molecules/instances.csv, molecules/placed_atoms.h5, molecules/placed_structure.cif, and checksums.sha256;
- molecule_instances.csv columns: instance_id, source_hash, residue_name, translation_x_A, translation_y_A, translation_z_A, quaternion_w, quaternion_x, quaternion_y, quaternion_z, periodic_image_x, periodic_image_y;
- pore_geometry.h5 stores target/packing boxes, target origin, schema version, unit records, and the authoritative final phase reference;
- packmol_handoff.inp references only real pore material and records target/packing boxes without temporary blocker atoms;
- pore_reference_coordinates.cif contains absolute pore coordinates;
- phase_mapping.json maps PORE and MATRIX phase identifiers;
- pore_atom_indices.ndx contains a PORE group;
- compression_metadata.json records target box, packing box, target origin, absolute lock mode, and pore coordinate hash.
- [ ] **Step 6: Verify GREEN**

~~~powershell
uv run pytest tests/test_storage.py tests/test_pipeline.py tests/test_optimization_io.py -v
uv run ruff check src tests
~~~

- [ ] **Step 7: Commit**

~~~powershell
git add src/porous_film tests
git commit -m "feat: orchestrate porous film generation"
~~~

---

### Task 10: Independent validator, end-to-end fixture, and user documentation

**Files:**
- Modify: pyproject.toml
- Create: src/porous_film_validator/__init__.py
- Create: src/porous_film_validator/cli.py
- Create: src/porous_film_validator/validate.py
- Create: tests/fixtures/minimal_config.yaml
- Modify: tests/conftest.py
- Create: tests/test_validator.py
- Create: tests/test_end_to_end.py
- Modify: README.md

**Interfaces:**
- ValidationReport(status: str, independent_metrics: dict, report_consistency: dict, target_compliance: dict, warnings: list[str], errors: list[str]).
- validate_export(qa_export: Path) -> ValidationReport.
- Console command porous-film-validate is registered only after a failing command-discovery test.
- Validator reads qa_export only and never imports porous_film.
- End-to-end test runs the real CLI with a small deterministic system, then runs validate_export on the produced complete QA export.

- [ ] **Step 1: Write a failing validator command-discovery test before editing pyproject.toml**

~~~python
import subprocess


def test_validator_console_command_is_installed() -> None:
    result = subprocess.run(
        ["porous-film-validate", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
~~~

Run this test first and verify it fails because the command is missing. Then add porous-film-validate = "porous_film_validator.cli:app" to the existing project.scripts table and change tool.hatch.build.targets.wheel.packages to ["src/porous_film", "src/porous_film_validator"].

- [ ] **Step 2: Write the failing behavioral import-isolation test**

~~~python
import subprocess
import sys


def test_validator_imports_when_main_package_is_blocked() -> None:
    script = """
import builtins

real_import = builtins.__import__


def guarded_import(name, *args, **kwargs):
    if name == "porous_film" or name.startswith("porous_film."):
        raise RuntimeError("validator imported main package")
    return real_import(name, *args, **kwargs)


builtins.__import__ = guarded_import
import porous_film_validator.validate
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
~~~

- [ ] **Step 3: Add independent QA and end-to-end fixtures, then write the failing validator test**

Append these fixtures to tests/conftest.py:

~~~python
import h5py
import numpy as np


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
~~~

tests/fixtures/minimal_config.yaml must include a Beta(2,2) theta orientation with uniform azimuth and be a deterministic one-compact-pore configuration with target box 20×20×20 Å, packing box 20×20×30 Å, seed density 0.000125 Å⁻³, target porosity 0.10, one exact argon molecule, lattice_jitter with zero jitter, one candidate round, coarse spacing 2 Å, and fine spacing 1 Å. The PDB path is argon.pdb relative to the YAML file.

Then write the validator test:

~~~python
from pathlib import Path

from porous_film_validator.validate import validate_export


def test_validator_rejects_incomplete_mandatory_export(
    incomplete_qa_export: Path,
) -> None:
    report = validate_export(incomplete_qa_export)

    assert report.status in {"FAIL", "NOT_EVALUABLE"}
    assert "mandatory" in " ".join(report.errors).lower()
~~~

- [ ] **Step 4: Write failing end-to-end test**

~~~python
from pathlib import Path

import trimesh
from typer.testing import CliRunner

from porous_film.cli import app


runner = CliRunner()


def test_minimal_generate_produces_required_outputs(
    minimal_config_path: Path,
    temporary_result_root: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "generate",
            "--config",
            str(minimal_config_path),
            "--result-root",
            str(temporary_result_root),
        ],
    )

    assert result.exit_code == 0
    run_root = Path(result.stdout.strip().splitlines()[-1])
    required = [
        run_root / "outputs" / "semiconductor_solid_target.glb",
        run_root / "outputs" / "pore_material.pdb",
        run_root / "outputs" / "pore_material_high_precision.cif",
        run_root / "outputs" / "molecule_instances.csv",
        run_root / "outputs" / "pore_geometry.h5",
        run_root / "outputs" / "packing_metrics.json",
        run_root / "outputs" / "packmol_handoff.inp",
        run_root / "outputs" / "pore_reference_coordinates.cif",
        run_root / "outputs" / "phase_mapping.json",
        run_root / "outputs" / "pore_atom_indices.ndx",
        run_root / "outputs" / "compression_metadata.json",
        run_root / "analysis" / "target_vs_actual_metrics.json",
        run_root / "reports" / "final-summary.md",
        run_root / "qa_export" / "contract.json",
        run_root / "qa_export" / "normalized_config.yaml",
        run_root / "qa_export" / "unit_candidates.jsonl",
        run_root / "qa_export" / "unit_geometry.jsonl",
        run_root / "qa_export" / "channel_curves.h5",
        run_root / "qa_export" / "final_phase.h5",
        run_root / "qa_export" / "final_surface.ply",
        run_root / "qa_export" / "main_unit_metrics.csv",
        run_root / "qa_export" / "main_metrics.json",
        run_root / "qa_export" / "molecules" / "source",
        run_root / "qa_export" / "molecules" / "instances.csv",
        run_root / "qa_export" / "molecules" / "placed_atoms.h5",
        run_root / "qa_export" / "molecules" / "placed_structure.cif",
        run_root / "qa_export" / "checksums.sha256",
    ]
    assert all(path.exists() for path in required)

    scene = trimesh.load(
        run_root / "outputs" / "semiconductor_solid_target.glb",
        force="scene",
    )
    assert scene.geometry

    from porous_film_validator.validate import validate_export

    validation = validate_export(run_root / "qa_export")
    assert validation.status == "PASS", validation.errors
~~~

- [ ] **Step 5: Verify RED**

~~~powershell
uv run pytest tests/test_validator.py tests/test_end_to_end.py -v
~~~

- [ ] **Step 6: Implement validator and documentation**

Validator requirements:

- read contract.json and verify schema/checksums;
- independently read final_phase.h5 and recompute porosity, connected-component counts, x percolation, minimum cross-section, and phase dimensions;
- independently read unit JSONL and recompute counts, channel fraction, volume sample statistics, and RDF using its own minimum-image function;
- independently read molecule HDF5/CIF and recompute counts, total mass, density, rigid-template RMSD, and minimum distances;
- load semiconductor_solid_target.glb with trimesh, compare target bounds, voxelize or sample occupancy, and compare with final_phase.h5;
- compare independent metrics to main_metrics.json;
- separately judge report consistency and target compliance;
- output independent-validation-report.md and independent-validation.json with PASS, WARNING, FAIL, or NOT_EVALUABLE.

README requirements:

- installation with uv;
- minimal YAML example;
- explanation of target versus packing box;
- all CLI commands;
- result directory tree;
- Blender import instructions for GLB;
- Packmol handoff explanation;
- Bayesian optimization exchange files;
- limitations and troubleshooting.

- [ ] **Step 7: Run full verification**

~~~powershell
uv run pytest -v
uv run pytest --cov=porous_film --cov=porous_film_validator --cov-report=term-missing
uv run ruff check src tests
uv build
~~~

Expected: all tests pass, Ruff exits 0, and wheel/sdist build succeeds.

- [ ] **Step 8: Commit**

~~~powershell
git add pyproject.toml README.md src/porous_film_validator tests
git commit -m "feat: add independent validator and end-to-end workflow"
~~~

---

## Plan self-review checklist

- [ ] Every design requirement maps to at least one task.
- [ ] No task asks an implementer to invent an undefined interface.
- [ ] All inter-task signatures are stated before use.
- [ ] Target and packing boxes are never conflated.
- [ ] Main generator and validator remain import-isolated.
- [ ] Blender GLB output is covered by real load tests.
- [ ] Bayesian optimization outputs separate requested, realized, feasibility, objectives, and uncertainty.
- [ ] Full test, lint, coverage, and build commands appear in Task 10.
