# HD1 Windows Execution Instructions

This is the execution guide for the frozen HD1 one-seed diagnostic pilot. The implementation was prepared without access to the target Windows machine, RTX 4060, or CUDA runtime. Canonical evidence exists only after these commands are run successfully on that machine.

## Prerequisites

Use the repository checkout on `exp/hd1-windows-high-dimensional-pinn-48h`. Use the existing project environment; the runner never installs or upgrades packages. The default interpreter is `.\.venv-pinn\Scripts\python.exe`; override it with `-Python` if the already-provisioned environment lives elsewhere. PyTorch must report CUDA available. Mixed precision is disabled.

Before execution, confirm the implementation SHA and worktree state:

```powershell
git switch exp/hd1-windows-high-dimensional-pinn-48h
git pull --ff-only
git rev-parse HEAD
git status --short
```

## Required commands

Run from the repository root in PowerShell. Do not run canonical jobs directly; the wrapper enforces phase gates, fixed sequential order, budgets, safe resume, and status capture.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hd1_windows_48h.ps1 -Phase preflight
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hd1_windows_48h.ps1 -Phase smoke
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hd1_windows_48h.ps1 -Phase all
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hd1_windows_48h.ps1 -Phase status
```

`all` intentionally re-runs preflight and smoke before canonical execution. If those gates already passed for the exact current Git commit, `-Phase canonical` runs only the canonical ladder. The runner rejects stale gate files from another commit.

## Phase behavior

`preflight` records Python, PyTorch, NumPy, SciPy, CUDA, driver, GPU identity, and memory; checks writable runtime paths; executes `nvidia-smi`; runs a CPU-float64 CN/PSOR smoke; runs the focused HD1 tests; builds the reducible one-dimensional oracle at `(256,256)` and `(512,512)` for d=1/2/5/10; and runs the d=2 full-Hessian versus two-probe fixed-seed Hutchinson <=5% gate. Any failure stops the pilot.

`smoke` runs only the documented 500-step d=1 HVI and d=2 HVI jobs. The slower measured profile is conservatively projected across all eight canonical jobs. If that projection exceeds 24 GPU-hours, the sole predeclared fallback changes every canonical job from 10,000 to 6,000 steps before canonical execution starts.

`canonical` executes exactly `d1 SUP, d1 HVI, d2 SUP, d2 HVI, d5 SUP, d5 HVI, d10 SUP, d10 HVI`, sequentially. OOM retries are limited to batch sizes `2048 -> 1024 -> 512`. A training run exceeding three hours is `resource_blocked`; peak CUDA allocation at or above 7 GiB is a hard failure; the ladder stops before a larger dimension when neither model at the previous dimension satisfies the correctness gate. A job is skipped on resume only when its status belongs to the current Git commit and every recorded artifact SHA-256 verifies.

`status` is read-only. It prints compact preflight, smoke, and per-job state without creating or modifying files.

## Runtime artifacts and claim boundary

Compact outputs are written under `results/hd1_windows_high_dimensional_pinn_48h/`: `environment_manifest.json`, `config_manifest.json`, `run_metrics.csv`, `dimension_summary.csv`, `training_history.csv`, `hessian_validation.csv`, `decision.json`, and `artifact_hashes.json`. Per-job status JSONs live in `status/`; full logs, `.pt` checkpoints, and oracle `.npz` arrays are local runtime artifacts.

The finalizer also writes `m02_bridge.json` as `not_comparable` unless the metric definitions and test distribution match exactly; no cross-panel numerical ranking is performed. The strongest allowed positive HD1 classification is `pilot_promising`. The code cannot emit `candidate_supported`.

Do not commit results until the target Windows/CUDA run and fixed-test evaluation actually complete. Any result commit must be separate from the implementation commit. Do not access the campaign blind holdout, add dimensions/models/seeds, tune after test evaluation, or describe this reducible geometric-basket control as a general high-dimensional American-option solver.
