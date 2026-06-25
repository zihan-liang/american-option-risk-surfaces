# Feature Lifecycle

`feature_list.json` is the project-level state file for agent work. Agents should update it with the lifecycle helper instead of editing status fields by hand.

## Status flow

```text
not-started -> in-progress -> done
                       \-> blocked
research branch -> failed_metric_gate | diagnostic_only | future_work_only
```

Use `failed_metric_gate`, `diagnostic_only`, and `future_work_only` for research outcomes that should remain visible but are not promoted as completed implementation features.

## Start one feature

```bash
$PY scripts/update_feature_status.py \
  --id harness-v1.2-feature-lifecycle \
  --status in-progress \
  --note "Started with TDD."
```

The command rejects a second `in-progress` feature. This protects long-running sessions from split attention.

## Mark a feature done

```bash
$PY scripts/update_feature_status.py \
  --id harness-v1.2-feature-lifecycle \
  --status done \
  --evidence "$PY -m pytest tests/test_feature_lifecycle_tools.py -q; bash scripts/verify_harness.sh" \
  --note "Committed in <sha>."
```

Done-like statuses require evidence. No evidence, no completion claim.

## Mark a research method failed

```bash
$PY scripts/update_feature_status.py \
  --id round2-structural-full \
  --status failed_metric_gate \
  --evidence "results/.../pilot_structural_boundary_metrics.csv" \
  --note "Boundary RMSE did not pass Round-2 gate."
```

This keeps negative evidence in the system of record instead of hiding it in chat history.
