"""Windows entry point for the final boundary-complete matched HD3 experiment."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "experiments"
for path in (SRC, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hd3_matched_american_put_boundary_complete import main

if __name__ == "__main__":
    raise SystemExit(main())
