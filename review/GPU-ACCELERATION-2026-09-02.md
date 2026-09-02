# GPU acceleration record — 2026-09-02

## Scope

- Product line: Porous Film Generator 2.0 — Complex Shapes
- Source baseline: immutable tag `2.0`
- Source provenance recorded by the handoff: `1c9a10793e96437202482ec44e4263b87ef64882`
- Review-bundle commit carrying tag `2.0`: `9cd263b55d26fd7aa39db848125b5bb3a65b27fb`
- Development branch: `gpu-acceleration-20260902`
- Remote host alias: `merlin-worker`
- Remote working copy: `/home/tiger/project/porous-film-generator-2.0-gpu-20260902-102654`
- Preserved backup: `/home/tiger/project/porous-film-generator-2.0-backup-20260902-102654`

The CPU implementation remains the default reference. The optional backend uses CuPy float64 and
is selected with `POROUS_FILM_VOXEL_BACKEND=cuda`.

## Environment

- Python: 3.12.14
- CuPy: 13.6.0 (`cupy-cuda12x`)
- GPU: NVIDIA A800-SXM4-80GB
- GPU count: 8
- Driver: 535.261.03
- CUDA compiler: 12.9
- CUDA device used: 0

## A01 fixed-geometry equivalence benchmark

- YAML: `examples/configs/v2-A01-dispersed-worms.yaml`
- YAML SHA-256: `4a3698ffa746566d993559b7a135b5090bae828ea6687e2f78985404eb7cb6ff`
- Task seed: 21001
- Candidate derived seed: 277068599
- Unit count: 43
- Historical selected linear scale: 1.0087890625
- Fine spacing: 500 Å
- Grid: 60 × 60 × 12 = 43,200 voxels
- GPU point chunk: 65,536
- Numeric type: float64

Command shape:

```bash
POROUS_FILM_CUDA_DEVICE=0 \
POROUS_FILM_GPU_MAX_POINTS_PER_CHUNK=65536 \
uv run --extra gpu python <fixed-A01 CPU/GPU comparison script>
```

Results:

| Measurement | Result |
| --- | ---: |
| Shape/center construction | 5.494 s |
| GPU voxelization, cold | 2.513 s |
| GPU voxelization, warm | 1.988 s |
| CPU voxelization | 66.735 s |
| Warm GPU speedup for voxelization | 33.57× |
| CPU porosity | 0.27930555555555553 |
| GPU porosity | 0.27930555555555553 |
| CPU/GPU differing voxels | 0 |
| GPU cold/warm differing voxels | 0 |
| CPU phase SHA-256 | `8590fb4df229f49eac8de9cbeca03dbb49aeedb35ac778c7b74f4854f8d732dc` |
| GPU phase SHA-256 | `8590fb4df229f49eac8de9cbeca03dbb49aeedb35ac778c7b74f4854f8d732dc` |

The hashes are computed from the packed boolean pore mask, with little-endian bit order.

## Focused verification

- Ruff on GPU implementation and tests: passed.
- GPU equivalence plus existing voxel/scale tests: 34 passed.
- Pointwise mixed multilobe/channel SDF: absolute tolerance 2e-9 Å, identical sign classification.
- Mixed-geometry CUDA/CPU phase: exact boolean equality.
- Explicit CUDA request without an available backend: clear runtime failure.

## Full A01 GPU generation

Command:

```bash
CUDA_VISIBLE_DEVICES=0 \
POROUS_FILM_VOXEL_BACKEND=cuda \
POROUS_FILM_CUDA_DEVICE=0 \
POROUS_FILM_GPU_MAX_POINTS_PER_CHUNK=65536 \
uv run --extra gpu porous-film generate-geometry \
  --config ../../examples/configs/v2-A01-dispersed-worms.yaml \
  --result-root /home/tiger/project/gpu-validation-runs-20260902
```

Output directory:

```text
/home/tiger/project/gpu-validation-runs-20260902/2026-09-02/python_results/v2-a01-dispersed-worms
```

| Measurement | GPU result | Historical CPU result |
| --- | ---: | ---: |
| Candidate wall time | 175.249 s | 937.737 s |
| Candidate speedup | 5.35× | reference |
| Selected scale | 1.0087890625 | 1.0087890625 |
| Final porosity | 0.27930555555555553 | 0.27930555555555553 |
| Through centerlines | 0 | 0 |
| Audit result | FAIL | FAIL |
| Packed phase SHA-256 | `8590fb4df229f49eac8de9cbeca03dbb49aeedb35ac778c7b74f4854f8d732dc` | same CPU reference |

The warning set is unchanged: no valid through-track samples for equivalent diameter, projected
orientation, channel eta/tau, or curvature, and no valid `g_xy` pairs. Independent validation exits
with status 1 and the same scientific failures. This is the expected result for the known-infeasible
A01 configuration and demonstrates that the GPU backend does not turn a failed structure into a
passing one.

## Repository verification

- `uv lock --check`: passed with public PyPI sources only.
- `uv sync --frozen --all-groups --extra gpu`: passed.
- Full Ruff: passed.
- Full default pytest: 338 passed, 5 skipped in 540.34 s.
- `uv build`: passed for both wheel and source distribution.
- Heavy end-to-end/parallel selection with `POROUS_FILM_RUN_HEAVY=1` and CUDA enabled:
  17 passed, 1 failed in 51.32 s.
- The one heavy failure is reproducible with `POROUS_FILM_VOXEL_BACKEND=cpu`: the legacy
  `minimal_config.yaml` run omits schema-v3 final measurement artifacts and the independent
  validator returns `NOT_EVALUABLE`. It is a baseline contract/test issue, not a CPU/GPU mismatch.

## Acceleration boundary

The CUDA backend currently accelerates periodic SDF evaluation and voxelization. The following
remain CPU reference operations:

- center seed optimization and shape construction;
- final centerline tracking and normal-section measurement;
- connectivity and local-thickness analysis;
- overlap audit;
- mesh/file export;
- independent validator.

The selected candidate is still fully replayed by the baseline pipeline. Removing that duplicate
production replay is a separate optimization and is not included in this GPU change.

## Scientific equivalence policy

- CPU remains the default backend.
- CUDA must be explicitly selected or requested through `auto`.
- GPU SDF uses float64.
- A performance change must be compared with CPU pointwise SDF and exact phase masks.
- Any boundary-voxel difference, changed final measurement, or validator difference must be treated
  as a scientific behavior change and reviewed separately.
