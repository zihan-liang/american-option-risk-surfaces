"""Release entrypoint for the final HD2 -> HD3 Windows pipeline.

The implementation lives in ``hd_pipeline_windows_final``.  This thin wrapper
pins the release preflight test set and applies release-only compatibility
shims that must not change the scientific training objective.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import hd_pipeline_windows_final as pipeline

ROOT = Path(__file__).resolve().parents[1]


def release_focused_tests() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "experiments")])
    env["MPLBACKEND"] = "Agg"
    tests = [
        "tests/test_hd1_high_dimensional_pinn.py",
        "tests/test_hd2_hvi_objective_stabilization.py",
        "tests/test_hd2_conditional_methods.py",
        "tests/test_hd3_high_dimensional_pinn.py",
        "tests/test_hd3_parallelism_benchmark.py",
        "tests/test_hd_pipeline_resume.py",
        "tests/test_hd_pipeline_review_regressions.py",
        "tests/test_hd_objective_optimizers_release.py",
        "tests/test_hd_pipeline_final_contract.py",
        "tests/test_hd_pipeline_release_contract.py",
    ]
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
        cwd=ROOT,
        env=env,
        check=True,
    )


pipeline._run_focused_tests = release_focused_tests

# Windows process-memory telemetry is non-scientific bookkeeping.  The inherited
# HD1 ctypes helper can fail on some Windows builds even when CUDA training is
# healthy.  Do not abort price/PDE/gradient experiments because this optional
# telemetry query failed; record 0 bytes instead.
_original_peak_cpu_memory_bytes = pipeline.hd2.peak_cpu_memory_bytes


def safe_peak_cpu_memory_bytes() -> int:
    try:
        return int(_original_peak_cpu_memory_bytes())
    except OSError:
        return 0


pipeline.hd2.peak_cpu_memory_bytes = safe_peak_cpu_memory_bytes
pipeline.hd3.peak_cpu_memory_bytes = safe_peak_cpu_memory_bytes


def main(argv: list[str] | None = None) -> int:
    return pipeline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
