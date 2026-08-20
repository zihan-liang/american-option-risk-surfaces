from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run(script_name: str) -> None:
    script = HERE / script_name
    print(f"\n=== Running {script.name} ===", flush=True)
    subprocess.run([sys.executable, str(script)], check=True)


def main() -> None:
    run("build_stage1_tables.py")
    run("plot_hd_scaling.py")
    run("plot_hd_boundary_and_compute.py")
    run("plot_binomial_convergence.py")
    print("\nAll Stage 1 plots completed successfully.", flush=True)


if __name__ == "__main__":
    main()
