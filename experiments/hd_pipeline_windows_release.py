"""Release entrypoint for the final HD2 -> HD3 Windows pipeline.

The implementation lives in ``hd_pipeline_windows_final``.  This thin wrapper
only pins the release preflight test set, replacing one superseded development
CAGrad tolerance test with the release-grade check that matches the official
reference implementation's 1e-4 denominator stabilizer.
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


def main(argv: list[str] | None = None) -> int:
    return pipeline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
