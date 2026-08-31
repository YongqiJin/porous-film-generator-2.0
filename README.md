# Porous Film Generator 2.0

This repository contains the reviewed Porous Film Generator 2.0 handoff, source code,
reference documents, reproducible examples, validation evidence, and optimization notes.

Development happens on the `dev` branch. The `main` branch and immutable `2.0` tag preserve the
initial handoff baseline.

## Quick start on merlin-stem

```bash
cd ~/project/porous-film-generator-2.0/source/porous-film-generator-2.0
uv sync --frozen --all-groups
uv run porous-film generate-geometry \
  --config ../../examples/configs/v2-A01-dispersed-worms.yaml \
  --result-root ../../runs
```

The A01 example is a full 3 × 3 × 0.6 μm visual example and can take tens of minutes. It is a
known audit-failing visualization input, not a scientifically feasible reference case.

Each run with `output.write_plots: true` writes a self-contained report to:

```text
<run>/outputs/visual-report/index.html
```

The report contains Geometry, Validation, Optimization, and Performance pages. It has no external
JavaScript dependency and can be opened directly in a browser.

## Repository layout

- `source/porous-film-generator-2.0/`: Python source and tests.
- `examples/`: schema-v3 configurations and reference outputs.
- `docs/` and `reference-docs/`: project and parameter documentation.
- `review/`: known issues, verification evidence, code map, and optimization plan.
- `dist/`: baseline wheel and source distribution.
