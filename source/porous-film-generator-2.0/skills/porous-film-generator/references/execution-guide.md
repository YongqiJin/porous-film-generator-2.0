# Server-neutral execution guide

Current environment-specific security, scheduler, filesystem, and result-retention instructions always override these generic examples.

## 1. Discover the installation

POSIX shell:

```bash
command -v porous-film
command -v porous-film-validate
porous-film version
porous-film --help
```

PowerShell:

```powershell
Get-Command porous-film
Get-Command porous-film-validate
porous-film version
porous-film --help
```

If commands are absent, inspect the project virtual environment, activated venv/Conda environment, modules, container image, or administrator-provided launcher. Do not guess a fixed installation path.

## 2. Prepare a task directory

Follow the active storage policy. Preserve originals and copy the YAML and source PDB into the task input area. Relative PDB paths resolve from the YAML directory.

```bash
CONFIG=/path/to/task/inputs/config.yaml
RESULT_ROOT=/path/to/calculation-results
```

## 3. Preflight

```bash
porous-film preflight \
  --config "$CONFIG" \
  --result-root "$RESULT_ROOT"
```

Stop on errors. Review warnings about sample support and memory before continuing.

## 4. Generate geometry or the full structure

Geometry only:

```bash
porous-film generate-geometry \
  --config "$CONFIG" \
  --result-root "$RESULT_ROOT" \
  --workers 8
```

Full geometry, molecule placement, and exports:

```bash
porous-film generate \
  --config "$CONFIG" \
  --result-root "$RESULT_ROOT" \
  --workers 8
```

Use `--no-parallel` for serial execution. Do not combine it with `--workers`.

The generator uses CPU multiprocessing. Choose workers from CPU affinity, physical cores, independent task count, and memory. GPU availability does not directly accelerate current geometry generation.

## 5. Continue an existing run

```bash
porous-film fill-pore --run /path/to/run
porous-film audit --run /path/to/run
porous-film audit-packmol-output \
  --run /path/to/run \
  --structure /path/to/packed-output.pdb
```

`fill-pore` requires a geometry-stage run with mandatory neutral artifacts. `audit-packmol-output` checks intrusion against the exported pore geometry and interface allowance.

## 6. Independent validation

```bash
porous-film-validate /path/to/run/qa_export
```

Treat `PASS`, `FAIL`, and `NOT_EVALUABLE` distinctly.

## Execution environments

### Local shell

Run commands in an activated environment. Use a persistent terminal/session manager when disconnects are possible.

### Scheduler

Wrap the same commands in the site's scheduler script. Request CPU and memory from preflight estimates. Do not hard-code Slurm, PBS, LSF, queue, or partition choices in reusable configs.

### Container

Mount YAML, PDB, and result root. Confirm paths inside the YAML are valid inside the container and output ownership is acceptable.

### DPDispatcher or another approved remote runner

Use the environment-approved schema and credential mechanism. Keep credentials outside committed files and reports. Never print credentials or load credential-bearing rendered submission JSON into chat context. Forward only required inputs and retrieve declared outputs/logs. Re-running an unchanged DPDispatcher submission is the normal idempotent recovery action after interruption.

## Completion checks

Do not infer completion from a missing process name. Verify:

1. the command/submission finished;
2. `calculation_status.json` exists for a full run;
3. expected GLB/HDF5/JSON outputs exist;
4. reports identify feasible versus infeasible status;
5. `porous-film-validate` produced a final status;
6. hashes match after transfer.

A preview helper can finish while a separate full audit remains running. Label them separately.