# Porous-Film Generator Knowledge Skill Design

## Goal

Create a server-neutral Codex knowledge skill named `porous-film-generator` that teaches other agents and users how to configure, run, audit, troubleshoot, and integrate the current porous-film generator. Keep a distributable copy in this repository and install the same files into the local Codex skills directory.

## Scope

The skill documents the current generator at Git version `0.2.0` and covers:

- YAML input fields and their dependencies;
- compact superellipsoid pores and curved channel pores;
- center placement by lattice jitter or target RDF;
- mixture distributions without tabulated inputs;
- candidate/seed multiprocessing;
- porosity scaling, voxelization, geometry audits, molecule placement, and export;
- CLI commands and server-neutral execution patterns;
- output directories and the intended consumer for each artifact;
- Packmol handoff, independent validation, and Bayesian-optimization exchange files;
- the difference between a complete design point, sequential parameter derivation, candidate search, and external multi-parameter optimization.

The skill does not automate credentials, hard-code a host, or replace environment-specific scheduler and storage rules.

## Server-Neutral Boundary

The skill must not contain:

- IP addresses, hostnames, usernames, passwords, private-key paths, or host-key fingerprints;
- a fixed CPU/GPU server role;
- a fixed `/software`, `/data`, `/root/data`, or Windows result path;
- a fixed scheduler, queue, partition, or worker count;
- a credential-bearing DPDispatcher submission.

Instead, it instructs the caller to discover the current environment:

```bash
command -v porous-film
command -v porous-film-validate
porous-film version
porous-film --help
```

The current implementation is CPU multiprocessing code. The skill may state that GPUs are not used by the generator itself, but it must not prescribe a particular machine.

## Directory Structure

Repository copy:

```text
skills/porous-film-generator/
  SKILL.md
  agents/openai.yaml
  references/
    input-schema.md
    generation-workflow.md
    output-artifacts.md
    execution-guide.md
    parameter-recipes.md
    troubleshooting.md
```

Installed copy:

```text
%CODEX_HOME%/skills/porous-film-generator/
```

`SKILL.md` stays concise and routes the agent to one or more references based on the request. Detailed schemas and file inventories live only in `references/`.

## Trigger Contract

The description begins with `Use when...` and includes these triggers:

- configuring a porous-film YAML;
- explaining inputs, outputs, or workflow;
- generating compact, irregular, channel, through-pore, or bicontinuous films;
- choosing preview versus production voxel resolution;
- running or troubleshooting the CLI locally, remotely, or through a scheduler;
- interpreting audits, connectivity, overlap, distributions, or optimizer exchange files;
- preparing Packmol handoff or Blender-readable GLB output.

## Core Workflow Contract

The skill explains this exact workflow:

1. Discover the installed commands and environment rules.
2. Copy inputs into the calculation task directory.
3. Validate the complete YAML model.
4. Run `preflight` before expensive work.
5. Derive seed count from number density and target volume.
6. Generate pore centers.
7. Allocate compact/channel types and sample volume, shape, orientation, and roughness distributions.
8. Construct superellipsoids and curved centerline channels.
9. Solve one global linear scale to approach target porosity.
10. Voxelize and audit all distributions and hard constraints together.
11. Select and deterministically replay a candidate.
12. For `generate`, place molecules and write all exports.
13. Run the independent validator and inspect feasibility/status files.

The skill explicitly states that users provide a complete multi-parameter design point at once, but the generator derives geometry sequentially. Internally it optimizes only the global pore-size scale and random candidate realization. Joint tuning of porosity, number density, RDF, aspect ratio, channel fraction, eta, tau, and roughness belongs to an external optimizer.

## Reference Responsibilities

### `input-schema.md`

Document every top-level YAML group, supported distribution family, units, derived quantities, interactions, and validation rules.

### `generation-workflow.md`

Explain the internal data flow, parameter dependencies, candidate selection, deterministic replay, audit gates, and parameter-optimization boundary.

### `output-artifacts.md`

Map the task directory tree and explain which files are used by Blender, Packmol, independent validation, simulation setup, reproducibility checks, and external optimizers.

### `execution-guide.md`

Provide server-neutral command discovery and direct CLI examples. Explain how to adapt the commands to local shells, schedulers, containers, or an environment-approved remote submission mechanism without embedding credentials.

### `parameter-recipes.md`

Provide practical starting patterns for dispersed compact pores, curved channels/bicontinuous pores, thin near-through films, preview meshes, and production audits. Recipes explain parameter effects rather than promising universal physical values.

### `troubleshooting.md`

Cover zero seeds, grid divisibility, memory pressure, slow channel voxelization, infeasible audits, distribution deviations, excessive overlap, matrix disconnection, missing final exports, and the distinction between a completed preview and a completed full audit.

## Testing

The repository receives `tests/test_knowledge_skill.py`. Tests first fail because the skill does not exist, then verify:

- required files and frontmatter;
- server-neutral text with no known deployment secrets or fixed hosts;
- required workflow concepts and CLI commands;
- required input/output artifact coverage;
- the explicit non-joint-optimization explanation;
- valid `agents/openai.yaml` metadata;
- a README link to the skill.

After implementation, run:

```powershell
.venv\Scripts\pytest.exe tests\test_knowledge_skill.py -q
.venv\Scripts\pytest.exe -q
.venv\Scripts\ruff.exe check .
python C:\Users\Admin\.codex\skills\.system\skill-creator\scripts\quick_validate.py skills\porous-film-generator
```

Copy the repository skill to the local skills directory, compare hashes, and run validation again on the installed copy.

## Version Baseline

After the skill and tests are committed, create an annotated Git tag:

```text
original-v0.2.0
```

The tag marks the documented, deployed generator plus its knowledge skill as the original baseline for later experimental branches. Future modifications branch from this tag or its tagged commit; the tag is never moved.
