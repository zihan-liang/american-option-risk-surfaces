# Harness Quality Document

This document tracks whether the harness is making the project easier to restart, verify, and review.

## Current assessment

| Dimension | Status | Evidence |
|---|---|---|
| Instructions | Strong | `AGENTS.md` routes to detailed docs instead of becoming a giant manual. |
| State | Improving | `feature_list.json` now records priority, area, behavior, verification, evidence, and notes. |
| Verification | Strong | `init_harness.sh`, `verify_harness.sh`, `verify_round2_evidence.sh`, and schema validators exist. |
| Scope | Strong | Worktree policy, claim-safety rules, and subagent protocol protect branch ownership. |
| Lifecycle | Improving | Clean-state checklist and session handoff define restart/exit behavior. |
| Observability | Improving | `score_harness.py` writes a machine-readable harness score and markdown summary. |

## Known weaknesses

- Full report-link tests depend on ignored generated figures in fresh worktrees.
- Artifact bootstrap may rewrite tracked result summaries; agents must keep commits scoped.
- `feature_list.json` is still manually edited; future work could add a status-update helper.

## Next quality target

Make the harness self-auditing enough that a fresh agent can answer:

1. What is active?
2. What must not be changed?
3. What command proves the current claim?
4. What generated artifacts are needed?
5. What evidence was left for the next session?
