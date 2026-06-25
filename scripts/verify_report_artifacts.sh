#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"
TEX_FILES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --tex)
      TEX_FILES+=("$2")
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ${#TEX_FILES[@]} -eq 0 ]]; then
  while IFS= read -r path; do
    TEX_FILES+=("$path")
  done < <(
    {
      find "$ROOT/reports" -maxdepth 1 -name 'stage*.tex' -type f 2>/dev/null || true
      for path in "$ROOT/reports/final_report.tex" "$ROOT/paper/manuscript.tex" "$ROOT/paper/supplementary_material.tex"; do
        [[ -f "$path" ]] && echo "$path"
      done
    } | sort
  )
fi

PYTHON="${PY:-python3}"

"$PYTHON" - "$ROOT" "${TEX_FILES[@]}" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
tex_files = [Path(path).resolve() for path in sys.argv[2:]]

include_pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
input_pattern = re.compile(r"\\input\{([^}]+)\}")


def read_expanded(tex_path: Path, seen: set[Path] | None = None) -> str:
    seen = seen or set()
    if tex_path in seen or not tex_path.exists():
        return ""
    seen.add(tex_path)
    text = tex_path.read_text(encoding="utf-8")
    for include in input_pattern.findall(text):
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = root / include_path
        text += "\n" + read_expanded(include_path, seen)
    return text


missing: list[str] = []
checked = 0
for tex_path in tex_files:
    text = read_expanded(tex_path)
    for raw in include_pattern.findall(text):
        if raw.startswith(("http://", "https://")):
            continue
        checked += 1
        figure_path = Path(raw)
        if not figure_path.is_absolute():
            figure_path = root / figure_path
        if not figure_path.exists():
            try:
                display_tex = tex_path.relative_to(root)
            except ValueError:
                display_tex = tex_path
            missing.append(f"{display_tex}: {raw}")

if missing:
    print("missing report artifacts:", file=sys.stderr)
    for item in missing:
        print(f"  {item}", file=sys.stderr)
    raise SystemExit(1)

print(f"report artifacts ok: {checked} includegraphics paths checked")
PY
