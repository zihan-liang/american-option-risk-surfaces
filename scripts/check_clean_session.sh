#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ALLOW_DIRTY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow-dirty)
      ALLOW_DIRTY=true
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

"$PYTHON" scripts/validate_feature_list.py feature_list.json >/dev/null

status="$(git status --porcelain)"
if [[ -n "$status" && "$ALLOW_DIRTY" != true ]]; then
  echo "worktree is not clean:" >&2
  echo "$status" >&2
  echo "Use --allow-dirty only for active uncommitted work and document the reason in session-handoff.md." >&2
  exit 1
fi

if [[ "$ROOT" == "/Users/liangzihan/Documents/SURF_0159/SURF/american-option-risk-surfaces" ]]; then
  echo "warning: running clean-session check in main checkout; implementation work should use an external worktree" >&2
fi

echo "clean-session check ok"
