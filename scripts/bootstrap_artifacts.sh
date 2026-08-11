#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FORCE=false
CHECK_ONLY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=true
      shift
      ;;
    --check-only)
      CHECK_ONLY=true
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "${PY:-}" ]]; then
  PYTHON="$PY"
elif [[ -x "/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/.venv-pilot/bin/python" ]]; then
  PYTHON="/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/.venv-pilot/bin/python"
else
  PYTHON="python3"
fi

export PYTHONPATH="${PYTHONPATH:-src:experiments}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${TMPDIR:-/tmp}/american_option_pycache}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/codex_mplconfig}"
mkdir -p "$MPLCONFIGDIR"

if [[ "$FORCE" != true ]] && bash scripts/verify_report_artifacts.sh >/dev/null 2>&1; then
  echo "report artifacts already present"
  exit 0
fi

if [[ "$CHECK_ONLY" == true ]]; then
  bash scripts/verify_report_artifacts.sh
  exit $?
fi

echo "bootstrapping ignored report artifacts with $PYTHON"

"$PYTHON" experiments/03_convergence_domain_runtime.py --config configs/convergence.yaml --figures-only
"$PYTHON" experiments/04_application_risk_surfaces.py --config configs/application_sweep.yaml

if [[ "$FORCE" == true || ! -f data/processed/surrogate_dataset.npz ]]; then
  "$PYTHON" experiments/05_surrogate_dataset.py --config configs/surrogate_dataset.yaml
fi

"$PYTHON" - <<'PY'
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def rows(path: str) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        output = list(csv.DictReader(fh))
    for row in output:
        for key, value in list(row.items()):
            if value in ("", None):
                continue
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return output


baseline = load_module("baseline_figs", "experiments/06_surrogate_baselines.py")
baseline._save_figures(
    rows("results/experiment_06_surrogate_baselines/baseline_metrics.csv"),
    rows("results/experiment_06_surrogate_baselines/violation_table.csv"),
    Path("results/figures/stage07_baseline_surrogates"),
)

positive = load_module("positive_figs", "experiments/07_positive_premium_tuning.py")
positive._save_figures(
    rows("results/experiment_07_positive_premium_tuning/positive_premium_metrics.csv"),
    rows("results/experiment_07_positive_premium_tuning/positive_premium_violation_table.csv"),
    Path("results/figures/stage08_positive_premium_surrogate"),
)

boundary = load_module("boundary_figs", "experiments/09_boundary_head.py")
boundary._save_figures(
    rows("results/experiment_09_boundary_head/boundary_metrics.csv"),
    rows("results/experiment_09_boundary_head/boundary_by_regime.csv"),
    Path("results/figures/stage09_boundary_head"),
)

call_boundary = load_module("call_boundary_figs", "experiments/10_option_specific_call_boundary_refinement.py")
call_boundary._save_figures(
    rows("results/experiment_10_option_specific_call_boundary/call_boundary_metrics.csv"),
    Path("results/figures/stage09_boundary_head"),
)

delta = load_module("delta_figs", "experiments/12_supervised_delta_head.py")
delta._save_figures(
    rows("results/experiment_12_supervised_delta_head/delta_head_metrics.csv"),
    rows("results/experiment_12_supervised_delta_head/delta_head_violation_table.csv"),
    Path("results/figures/stage10_delta_head"),
)

integrated = load_module("integrated_figs", "experiments/13_integrated_application_workflow.py")
integrated._save_figures(
    rows("results/experiment_13_integrated_application_workflow/integrated_case_metrics.csv"),
    rows("results/experiment_13_integrated_application_workflow/integrated_timing.csv"),
    Path("results/figures/stage11_integrated_workflow"),
)
PY

"$PYTHON" experiments/18_publication_figures.py --config configs/publication_figures.yaml
"$PYTHON" experiments/19_report_visualization_enrichment.py --config configs/report_visualization_enrichment.yaml

bash scripts/verify_report_artifacts.sh
echo "artifact bootstrap complete"
