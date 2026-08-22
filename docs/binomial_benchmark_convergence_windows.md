# Binomial benchmark convergence audit (Windows)

This is an independent CRR convergence check for the matched American put
contract used by the final HD3 comparison:

- option: American put
- K = 1
- T = 1
- r = 0.05
- q = 0.10
- sigma = 0.60

The audit is intentionally separate from PINN training. It does not use CUDA.
It writes a resumable point-price ledger and convergence summaries only.

## Frozen audit grid

Probe phase:

- S/K = 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20, 1.40
- tau/T = 0.10, 0.50, 1.00
- CRR steps = 2048, 4096, 8192, 16384

Confirmation phase:

- S/K = 0.80, 1.00, 1.20
- tau/T = 0.10, 0.50, 1.00
- CRR steps = 32768, 50000

The script does not automatically declare a benchmark step count. The output is
scientific evidence for choosing the final tree resolution after review.

## Windows commands

From the repository root after switching to branch
`exp/binomial-benchmark-convergence`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_binomial_benchmark_windows.ps1 -Phase preflight -Workers 4
```

If preflight passes:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_binomial_benchmark_windows.ps1 -Phase all -Workers 4
```

If the desired Python environment is not the default `python`, pass it explicitly:

```powershell
$PythonExe = "D:\path\to\python.exe"
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_binomial_benchmark_windows.ps1 -Phase all -Workers 4 -Python $PythonExe
```

If the run is interrupted, rerun the same command. Completed jobs are loaded
from `results/binomial_benchmark_convergence/point_results.csv` and are not
recomputed.

If four workers cause memory or responsiveness problems, rerun with `-Workers 2`.

## Outputs

The run writes:

- `results/binomial_benchmark_convergence/point_results.csv`
- `results/binomial_benchmark_convergence/convergence_summary.csv`
- `results/binomial_benchmark_convergence/config_manifest.json`
- `results/binomial_benchmark_convergence/environment_manifest.json`
- `results/binomial_benchmark_convergence/final_status.json`

After completion, commit the result directory on the experiment branch and push
it so the benchmark can be reviewed against CN--PSOR and the high-accuracy DIRK
reference.
