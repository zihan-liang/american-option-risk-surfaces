# Evaluator Rubric

Use this rubric when reviewing an agent branch. The evaluator should verify evidence directly rather than trusting the worker's narrative.

| Criterion | Score 1 | Score 3 | Score 5 |
|---|---|---|---|
| Functional completeness | Major requested behavior missing | Main behavior present with gaps | Requested behavior complete and scoped |
| Verification evidence | No fresh commands | Partial focused checks | Required commands ran with outputs recorded |
| State continuity | Handoff missing | Some state updated | `feature_list.json`, `progress.md`, and `session-handoff.md` updated |
| Scope control | Unrelated edits or dirty main checkout | Minor unrelated churn explained | Scoped worktree, no unrelated changes |
| Claim safety | Unsafe or broad claims | Mostly safe but vague | Claim language matches audit evidence |
| Artifact hygiene | Generated artifacts staged accidentally | Generated churn noted | Generated artifacts ignored or curated intentionally |
| Reproducibility | Hidden setup assumptions | Setup partially documented | Startup/artifact/verification path explicit |

## Minimum promotion rule

A branch should not be promoted unless:

- no criterion scores below 3;
- verification evidence is fresh;
- claim-safety language is checked;
- remaining risks are written down.

For research method branches, the evaluator must also check result schema and provenance manifests.
