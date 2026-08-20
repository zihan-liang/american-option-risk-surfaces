"""Windows-safe entry point for the matched HD3 experiment.

The scientific training/evaluation implementation lives in
``hd3_matched_american_put_comparison.py``.  This adapter only makes optional
CPU peak-memory telemetry nonfatal on Windows, where the legacy
GetProcessMemoryInfo probe can fail on some Python/Windows combinations.
Scientific metrics and CUDA memory accounting are unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "experiments"
for path in (SRC, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hd3_classical_vs_structured_high_dimensional_pinn as hd3_base
from hd1_windows_high_dimensional_pinn_48h import peak_cpu_memory_bytes as _raw_peak_cpu_memory_bytes


def safe_peak_cpu_memory_bytes(probe: Callable[[], int] | None = None) -> int | None:
    """Return peak CPU memory when available; otherwise record it as unavailable.

    CPU memory is optional engineering telemetry and must never invalidate a
    completed numerical experiment.  Only the known OS-level probe failure is
    suppressed; unexpected exceptions still propagate.
    """
    memory_probe = _raw_peak_cpu_memory_bytes if probe is None else probe
    try:
        value = int(memory_probe())
    except OSError:
        return None
    return value if value >= 0 else None


# ``evaluate_hd3_job`` resolves this name from the base module at call time, so
# replacing only this optional telemetry callback leaves the numerical code and
# all scientific metrics untouched.
hd3_base.peak_cpu_memory_bytes = safe_peak_cpu_memory_bytes

from hd3_matched_american_put_comparison import main  # noqa: E402  (patch first)


if __name__ == "__main__":
    raise SystemExit(main())
