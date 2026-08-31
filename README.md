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
  --config ../../examples/configs/quick-visual-demo.yaml \
  --result-root ../../runs
```

The quick visual demo is intentionally small and is intended to verify installation, geometry
generation, and report rendering. The A01–A03 examples are larger visual examples and can take tens
of minutes; they are known audit-failing inputs, not scientifically feasible reference cases.

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
