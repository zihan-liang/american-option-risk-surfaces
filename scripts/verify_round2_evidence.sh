#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${PY:-}" ]]; then
  PYTHON="$PY"
elif [[ -x "/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/.venv-pilot/bin/python" ]]; then
  PYTHON="/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/.venv-pilot/bin/python"
else
  PYTHON="python3"
fi

WT_ROOT="/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces"
SUMMARY_DIR="$WT_ROOT/round2-evidence-summary/results/round2_ablation"
BOUNDARY_DIR="$WT_ROOT/round2-boundary-enriched-full/results/round2_boundary_enriched"
OPTION_DIR="$WT_ROOT/round2-option-specific-full/results/round2_option_specific_boundary"
STRUCTURAL_DIR="$WT_ROOT/round2-structural-full/results/round2_structural_boundary_loss"

comparison_files=(
  "$SUMMARY_DIR/round2_result_matrix.csv"
  "$BOUNDARY_DIR/pilot_comparison.csv"
  "$OPTION_DIR/option_specific_boundary_comparison.csv"
  "$STRUCTURAL_DIR/pilot_structural_boundary_metrics.csv"
)

manifest_files=(
  "$BOUNDARY_DIR/round2_run_manifest.json"
  "$OPTION_DIR/option_specific_boundary_config_manifest.json"
  "$STRUCTURAL_DIR/pilot_structural_boundary_config_manifest.json"
)

for path in "${comparison_files[@]}" "${manifest_files[@]}" \
  "$SUMMARY_DIR/round2_result_matrix.summary.json" \
  "$SUMMARY_DIR/round2_best_combo_decision.csv"; do
  if [[ ! -f "$path" ]]; then
    echo "missing Round-2 evidence file: $path" >&2
    exit 1
  fi
done

"$PYTHON" scripts/check_result_schema.py "${comparison_files[@]}"

"$PYTHON" - "$SUMMARY_DIR" "${manifest_files[@]}" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

summary_dir = Path(sys.argv[1])
manifest_paths = [Path(path) for path in sys.argv[2:]]

summary = json.loads((summary_dir / "round2_result_matrix.summary.json").read_text(encoding="utf-8"))
expected_summary = {
    "missing_tables": 0,
    "pilot_candidates": 0,
    "best_combo_decision": "no_methods_passed_pilot_gate",
    "recommended_next_step": "do_not_build_best_combo_yet",
}
for key, expected in expected_summary.items():
    actual = summary.get(key)
    if actual != expected:
        raise SystemExit(f"unexpected Round-2 summary {key}: expected {expected!r}, got {actual!r}")

with (summary_dir / "round2_best_combo_decision.csv").open(newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
if not rows:
    raise SystemExit("Round-2 best-combo decision CSV is empty")
decision = rows[0]
if decision.get("decision") != "no_methods_passed_pilot_gate":
    raise SystemExit(f"unexpected best-combo decision: {decision}")

full_caps = {
    "train_cap": 150000,
    "validation_cap": 75000,
    "test_cap": 150000,
    "hidden": 128,
    "depth": 3,
    "epochs": 30,
    "batch_size": 8192,
    "device": "auto",
    "run_mode": "round2_full",
}
allowed_override_keys = {"run_mode"}
disallowed_budget_overrides = {"train_cap", "validation_cap", "test_cap", "hidden", "depth", "epochs", "batch_size", "device"}

for path in manifest_paths:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    effective = manifest.get("effective_config") or {}
    missing_keys = [key for key in ("config_path", "config_sha256", "git_branch", "git_commit", "run_mode", "source_config", "effective_config") if key not in manifest]
    if missing_keys:
        raise SystemExit(f"{path} missing manifest keys: {missing_keys}")
    for key, expected in full_caps.items():
        actual = manifest.get(key, effective.get(key))
        if actual != expected:
            raise SystemExit(f"{path} has non-full {key}: expected {expected!r}, got {actual!r}")
    override_keys = set(manifest.get("override_keys") or [])
    if override_keys - allowed_override_keys:
        raise SystemExit(f"{path} has disallowed override keys: {sorted(override_keys)}")
    if override_keys & disallowed_budget_overrides:
        raise SystemExit(f"{path} has budget/model-size overrides: {sorted(override_keys & disallowed_budget_overrides)}")

    sample_groups = [
        ("train_samples", "validation_samples", "test_samples"),
        ("put_train_samples", "put_validation_samples", "put_test_samples"),
        ("call_train_samples", "call_validation_samples", "call_test_samples"),
    ]
    found_counts = False
    for train_key, validation_key, test_key in sample_groups:
        if train_key in manifest:
            found_counts = True
            counts = {
                train_key: manifest[train_key],
                validation_key: manifest.get(validation_key),
                test_key: manifest.get(test_key),
            }
            if counts[train_key] != 150000 or counts[validation_key] != 75000 or counts[test_key] != 150000:
                raise SystemExit(f"{path} has non-full sample counts: {counts}")
    if not found_counts:
        raise SystemExit(f"{path} does not record actual sample counts")

print("Round-2 evidence verification ok")
PY
