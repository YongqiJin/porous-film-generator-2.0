---
name: porous-film-generator
description: Use when configuring, running, auditing, troubleshooting, or integrating the porous-film generator in any local or remote environment, including YAML inputs, compact and channel pores, multiprocessing, Packmol handoff, Blender GLB output, independent validation, and optimizer exchange files.
---

# Porous Film Generator

## Core principle

Treat one YAML file as one complete pore-design point. Validate it before expensive work, generate one or more random realizations, audit requested distributions and hard constraints, and preserve both requested and realized parameters. The generator is not a joint multi-parameter optimizer: it internally solves a global pore-size scale and searches random candidates; an external optimizer changes the design variables between runs.

## Start here

1. Read the current workspace instructions (`AGENTS.md`, scheduler rules, storage policy, security policy) before choosing paths or remote tools.
2. Discover the current installation instead of assuming a host or install directory. See [execution-guide.md](references/execution-guide.md).
3. Read only the references needed for the request:

| Request | Read |
|---|---|
| Write or review a YAML configuration | [input-schema.md](references/input-schema.md) |
| Explain how parameters become geometry | [generation-workflow.md](references/generation-workflow.md) |
| Find a GLB, HDF5, Packmol, QA, or optimizer file | [output-artifacts.md](references/output-artifacts.md) |
| Run locally, on a server, in a container, or through a scheduler | [execution-guide.md](references/execution-guide.md) |
| Choose starting parameter patterns | [parameter-recipes.md](references/parameter-recipes.md) |
| Diagnose slow, failed, infeasible, or incomplete runs | [troubleshooting.md](references/troubleshooting.md) |

## Required operating sequence

1. Preserve original inputs and copy the exact run inputs into the calculation task directory.
2. Confirm the command and version with `porous-film version`.
3. Run `porous-film preflight` before production-resolution generation.
4. Use `generate-geometry` when only pore geometry is needed; use `generate` when molecule placement and the full export set are needed.
5. Inspect `calculation_status.json`, `feasibility.json`, `realized_geometry_parameters.json`, and the reports rather than inferring success from process exit alone.
6. Run `porous-film-validate` on `qa_export/` for an independent exported-artifact check.
7. Record command, version, input hash, timing, warnings, output root, and final artifact hashes.

## Current experimental capability boundary

This branch reports **0.4.0.dev1** and introduces **schema v3**. Formal targets are grouped as
**position-quantity**, **shape**, and **proportion**. The standard phase-field workflow uses only the
real target box, with no padding or redundant packing box.

- Position-quantity: final-phase through-pore centerline xy-distance target g_xy(r); generated and
  measured center counts are diagnostics rather than a strict target density.
- Shape: final-phase local equivalent diameter, paired projected xz/xy orientation,
  compact/channel eta, channel tau, and normal-section curvature fluctuation.
- Proportion: final three-dimensional porosity in the target box.
- Geometry: multilobe compact pores, variable-radius channels, periodic x/y boundaries, finite z,
  and smooth implicit unions.
- Sampling: bounded analytic distributions and linear mixtures. Paired orientation components own
  both xz and xy Beta distributions.
- Audit: formal measurements are derived from the final phase. Unit volumes, labels, sampled
  orientation, and roughness are generation diagnostics and cannot satisfy a formal target.
- Parallelism: CPU multiprocessing across independent seeds or candidates; GPUs do not accelerate
  the current generator itself.
- Compatibility: schema v1 and v2 unit records remain readable while schema v3 final-phase evidence
  is added.

## Safety and portability

Never embed credentials, endpoint secrets, private keys, or rendered private submission JSON in this skill or in reports. Generic command examples are subordinate to the current environment-specific security, scheduler, and result-storage instructions. Use only the connection and submission mechanism approved by the active environment.