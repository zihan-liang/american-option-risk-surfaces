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

branch="$(git branch --show-current 2>/dev/null || true)"
echo "repo: $ROOT"
echo "branch: ${branch:-unknown}"
echo "python: $PYTHON"

if [[ "$ROOT" == "/Users/liangzihan/Documents/SURF_0159/SURF/american-option-risk-surfaces" ]]; then
  echo "warning: this is the main checkout; implementation work should use an external worktree" >&2
fi

"$PYTHON" -m compileall -q src tests experiments
bash scripts/verify_harness.sh

echo "startup health check ok"
