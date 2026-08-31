# AGENTS.md — Porous Film Generator 2.0 External Review

## Scope

This package represents **Porous Film Generator 2.0 — Complex Shapes**.

- Git tag: `2.0`
- Commit: `1c9a10793e96437202482ec44e4263b87ef64882`
- Historical Python package version: `0.4.0.dev1`

Do not treat schema version `3` or package version `0.4.0.dev1` as the product-line number. Do not move or overwrite the `2.0` tag.

## Scientific invariants

1. Formal targets must be measured from the final pore phase using the same definition used by the input.
2. Generation-unit labels, sampled values, relative volumes, roughness amplitudes, control points, and shape seeds are provenance/controls, not final measurements.
3. x and y are periodic; z is finite.
4. The semiconductor phase must retain x percolation and the configured minimum yz cross-section.
5. A visual GLB or successful process exit is not evidence of scientific feasibility.
6. The independent validator must remain import-isolated from `porous_film`.
7. Any performance change must be checked against a brute-force or previous-reference result for scientific equivalence.

## Development workflow

- Create a new branch from tag `2.0`.
- Use Python 3.12+ and `uv sync --frozen --all-groups`.
- Add a failing test before changing behavior.
- Run focused tests, Ruff, the full suite, heavy tests when resources permit, one real generation, and `porous-film-validate`.
- Record exact YAML, seed, version, command, timings, status, warnings, and hashes.
- Do not commit generated calculation outputs.

## Primary references

- `docs/v2孔生成器说明.md`
- `review/KNOWN-ISSUES.md`
- `review/code-map.md`
- `source/porous-film-generator-2.0/docs/superpowers/specs/2026-08-20-complex-shape-generation-v1-design.md`
- `source/porous-film-generator-2.0/docs/superpowers/specs/2026-08-24-final-pore-parameter-contract-design.md`