# Harness Decision Log

## 2026-06-25 — v1.1 operational hardening

Decision: extend the v1 harness with operational scripts and state validation rather than expanding `AGENTS.md` into a giant instruction file.

Rationale:

- The Learn Harness Engineering project solutions use root instructions as a router, with deeper state and reliability details in separate artifacts.
- The American option repo has a real hidden assumption: full report-link tests need ignored generated figures.
- Feature state needs validation so long-running and multi-session work does not drift.

Consequences:

- `feature_list.json` moves to schema version 2.
- Artifact verification and bootstrap become explicit scripts.
- Clean-session checks become part of handoff.
- Harness quality gets a machine-readable score.
