# Security redactions and package boundary

The external package preserves all core generator, audit, validator, tests, configuration examples, design specifications, and build metadata needed for code review and secondary development.

The following tag-2.0 material is intentionally not included because it is environment-specific and contains fixed server endpoints, local credential-file paths, host-key fingerprints, or deployment-only instructions:

- `deployment/`
- `.superpowers/`
- `docs/superpowers/plans/2026-08-13-parallel-execution-and-dual-server-deployment.md`
- `docs/superpowers/specs/2026-08-13-parallel-execution-and-dual-server-deployment-design.md`
- `tests/test_deployment_assets.py`
- `tests/test_knowledge_skill.py` (contains known-secret strings only for negative assertions)

No core module under `src/porous_film/` or `src/porous_film_validator/` was modified or omitted. The wheel is built from the original tag-2.0 package code; the source folder is a review-safe snapshot with the exclusions above. No password, private key, access token, rendered submission JSON, or credential file is present.