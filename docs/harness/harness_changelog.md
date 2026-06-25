# Harness Changelog

## 2026-06-25 — v1 initial project-specific harness

- Added root-level agent harness files: `AGENTS.md`, `feature_list.json`, `progress.md`, `init.sh`, and `session-handoff.md`.
- Added project-specific harness policies for worktrees, verification, claim safety, result schemas, and subagent execution.
- Added lightweight harness scripts for startup checks, harness validation, Round-2 evidence validation, and comparison CSV schema validation.
- Added tests for required harness structure, feature status validity, result schema validation, and claim-safety language.
- Future lesson-by-lesson corrections should append entries here and update only the relevant harness subsystem.

## 2026-06-25 — v1.1 operational hardening

- Inspected the local Learn Harness Engineering templates and Project 05/06 solution patterns.
- Added startup readiness, artifact policy, clean-state checklist, evaluator rubric, quality document, decision log, and feature-list schema.
- Added feature-list validation, report-artifact verification, artifact bootstrap, clean-session check, and harness scoring scripts.
- Updated `AGENTS.md` to keep root instructions as a router and point agents to operational checks.

## 2026-06-25 — v1.2 feature lifecycle helper

- Added `scripts/update_feature_status.py` so agents can update `feature_list.json` safely.
- Added `docs/harness/feature-lifecycle.md` to document status transitions and evidence requirements.
- Added lifecycle tests for starting work, enforcing one active feature, and requiring evidence for done-like statuses.
