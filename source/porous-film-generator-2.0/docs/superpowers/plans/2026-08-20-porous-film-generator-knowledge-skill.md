# Porous-Film Generator Knowledge Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add, install, test, commit, and tag a server-neutral knowledge skill for the current porous-film generator.

**Architecture:** Keep `SKILL.md` as a concise router and place detailed input, workflow, output, execution, recipe, and troubleshooting knowledge in focused reference files. Validate the skill with repository tests and the official local skill validator, then install a byte-identical copy into the local Codex skill directory.

**Tech Stack:** Markdown, YAML, Python 3.12, pytest, Ruff, Git, Codex skill metadata.

## Global Constraints

- The skill name is `porous-film-generator`.
- The skill is server-neutral and contains no real endpoint, account, credential path, host fingerprint, scheduler, queue, or fixed deployment/result directory.
- The repository copy lives at `skills/porous-film-generator/`.
- The installed copy lives at `%CODEX_HOME%/skills/porous-film-generator/`.
- The skill documents current version `0.2.0`; it does not change generator runtime behavior.
- The final annotated immutable baseline tag is `original-v0.2.0`.
- Existing project tests must remain green.

---

### Task 1: Add the Failing Skill Contract Test

**Files:**
- Create: `tests/test_knowledge_skill.py`

**Interfaces:**
- Consumes: repository root and the intended `skills/porous-film-generator/` path.
- Produces: executable contract tests for skill structure, content, neutrality, and discovery metadata.

- [ ] **Step 1: Write a test that requires the complete skill tree**

```python
from pathlib import Path

SKILL = Path("skills/porous-film-generator")


def test_knowledge_skill_has_required_files() -> None:
    expected = {
        "SKILL.md",
        "agents/openai.yaml",
        "references/input-schema.md",
        "references/generation-workflow.md",
        "references/output-artifacts.md",
        "references/execution-guide.md",
        "references/parameter-recipes.md",
        "references/troubleshooting.md",
    }
    assert expected <= {
        path.relative_to(SKILL).as_posix()
        for path in SKILL.rglob("*")
        if path.is_file()
    }
```

- [ ] **Step 2: Add content and server-neutrality assertions**

Assert that the skill contains all CLI commands, workflow stages, optimizer exchange files, and the explanation that only the global scale is internally solved. Assert that it contains none of the project’s known real endpoint, host fingerprints, credential paths, or pinned installation directory.

- [ ] **Step 3: Run the focused test and verify RED**

Run:

```powershell
.venv\Scripts\pytest.exe tests\test_knowledge_skill.py -q
```

Expected: failure because `skills/porous-film-generator/` does not exist.

---

### Task 2: Create the Skill Router and References

**Files:**
- Create: `skills/porous-film-generator/SKILL.md`
- Create: `skills/porous-film-generator/references/input-schema.md`
- Create: `skills/porous-film-generator/references/generation-workflow.md`
- Create: `skills/porous-film-generator/references/output-artifacts.md`
- Create: `skills/porous-film-generator/references/execution-guide.md`
- Create: `skills/porous-film-generator/references/parameter-recipes.md`
- Create: `skills/porous-film-generator/references/troubleshooting.md`

**Interfaces:**
- Consumes: current CLI, Pydantic configuration models, pipeline, audit implementation, README, and deployment-independent behavior.
- Produces: a concise trigger/router plus six focused knowledge references.

- [ ] **Step 1: Initialize the skill skeleton**

```powershell
.venv\Scripts\python.exe C:\Users\Admin\.codex\skills\.system\skill-creator\scripts\init_skill.py porous-film-generator --path skills --resources references
```

+- [ ] **Step 2: Replace `SKILL.md` with the final routing contract**
+
+Use frontmatter:
+
+```yaml
+---
+name: porous-film-generator
+description: Use when configuring, running, auditing, troubleshooting, or integrating the porous-film generator in any local or remote environment, including YAML inputs, compact and channel pores, multiprocessing, Packmol handoff, Blender GLB output, independent validation, and optimizer exchange files.
+---
+```
+
+The body must tell the agent which reference to read, require `preflight` before expensive production work, and require environment-specific security/storage instructions to override generic examples.
+
+- [ ] **Step 3: Write the six references from the current code**
+
+Each reference owns one topic. Do not duplicate full schemas or file inventories in `SKILL.md`.
+
+- [ ] **Step 4: Run the focused test**
+
+```powershell
+.venv\Scripts\pytest.exe tests\test_knowledge_skill.py -q
+```
+
+Expected: remaining failure only for missing UI metadata or README discovery link.
+
+---
+
+### Task 3: Add UI Metadata and Repository Discovery
+
+**Files:**
+- Create: `skills/porous-film-generator/agents/openai.yaml`
+- Modify: `README.md`
+
+**Interfaces:**
+- Consumes: final skill name and description.
+- Produces: UI metadata and a repository entry point for human users.
+
+- [ ] **Step 1: Generate `agents/openai.yaml` deterministically**
+
+```powershell
+.venv\Scripts\python.exe C:\Users\Admin\.codex\skills\.system\skill-creator\scripts\generate_openai_yaml.py skills\porous-film-generator `
+  --interface 'display_name=Porous Film Generator' `
+  --interface 'short_description=Configure, run, and audit porous-film generation' `
+  --interface 'default_prompt=Use $porous-film-generator to explain or run a porous-film generation workflow.'
+```
+
+- [ ] **Step 2: Add a README section**
+
+Link to `skills/porous-film-generator/SKILL.md`, state that it is server-neutral, and list its six reference topics.
+
+- [ ] **Step 3: Run the focused test and local skill validator**
+
+```powershell
+.venv\Scripts\pytest.exe tests\test_knowledge_skill.py -q
+.venv\Scripts\python.exe C:\Users\Admin\.codex\skills\.system\skill-creator\scripts\quick_validate.py skills\porous-film-generator
+```
+
+Expected: both pass.
+
+---
+
+### Task 4: Validate the Whole Repository and Install the Skill
+
+**Files:**
+- Create outside repository: `%CODEX_HOME%/skills/porous-film-generator/`
+
+**Interfaces:**
+- Consumes: validated repository skill.
+- Produces: byte-identical installed skill available to Codex.
+
+- [ ] **Step 1: Run full verification**
+
+```powershell
+.venv\Scripts\ruff.exe check .
+.venv\Scripts\pytest.exe -q
+```
+
+Expected: Ruff passes and the existing full suite plus the new skill tests passes.
+
+- [ ] **Step 2: Install without merging stale files**
+
+Remove only an existing `porous-film-generator` skill directory, then copy the complete repository skill directory into `%CODEX_HOME%/skills/`.
+
+- [ ] **Step 3: Validate and compare the installed copy**
+
+Run `quick_validate.py` against the installed directory and compare SHA-256 manifests for every file in the repository and installed copies. Expected: validation passes and manifests match exactly.
+
+---
+
+### Task 5: Commit and Mark the Original Baseline
+
+**Files:**
+- Commit all new skill, test, README, spec, and plan files.
+- Create annotated Git tag: `original-v0.2.0`.
+
+**Interfaces:**
+- Consumes: clean, tested repository and installed skill.
+- Produces: immutable baseline commit/tag for future experimental branches.
+
+- [ ] **Step 1: Verify the final diff and repository status**
+
+```powershell
+git diff --check
+git status --short
+```
+
+- [ ] **Step 2: Commit the baseline**
+
+```powershell
+git add README.md docs/superpowers/specs/2026-08-20-porous-film-generator-knowledge-skill-design.md docs/superpowers/plans/2026-08-20-porous-film-generator-knowledge-skill.md tests/test_knowledge_skill.py skills/porous-film-generator
+git commit -m "docs: add porous film generator knowledge skill"
+```
+
+- [ ] **Step 3: Create the annotated original-version tag**
+
+```powershell
+git tag -a original-v0.2.0 -m "Original porous-film-generator v0.2.0 baseline with knowledge skill"
+```
+
+- [ ] **Step 4: Verify the immutable baseline**
+
+```powershell
+git status --short --branch
+git show --stat --oneline HEAD
+git show --no-patch --decorate original-v0.2.0
+```
+
+Expected: clean `main`, tag resolves to the new baseline commit, and no remote push is attempted because the repository has no configured remote.
+