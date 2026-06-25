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

export PYTHONPATH="${PYTHONPATH:-src:experiments}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${TMPDIR:-/tmp}/american_option_pycache}"

required_files=(
  AGENTS.md
  feature_list.json
  progress.md
  init.sh
  session-handoff.md
  docs/harness/worktree-policy.md
  docs/harness/verification-matrix.md
  docs/harness/claim-safety-rules.md
  docs/harness/result-schema-policy.md
  docs/harness/subagent-protocol.md
  docs/harness/harness_changelog.md
  docs/harness/startup-readiness-checklist.md
  docs/harness/artifact-policy.md
  docs/harness/clean-state-checklist.md
  docs/harness/evaluator-rubric.md
  docs/harness/quality-document.md
  docs/harness/decision-log.md
  docs/harness/feature-list.schema.json
  scripts/init_harness.sh
  scripts/verify_harness.sh
  scripts/verify_round2_evidence.sh
  scripts/check_result_schema.py
  scripts/validate_feature_list.py
  scripts/verify_report_artifacts.sh
  scripts/bootstrap_artifacts.sh
  scripts/check_clean_session.sh
  scripts/score_harness.py
)

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing harness file: $path" >&2
    exit 1
  fi
done

"$PYTHON" -m json.tool feature_list.json >/dev/null
"$PYTHON" scripts/validate_feature_list.py feature_list.json >/dev/null
"$PYTHON" -m pytest tests/test_harness_contract.py tests/test_harness_v11_contract.py -q

echo "harness verification ok"
