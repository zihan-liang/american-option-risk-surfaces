# HD2 -> HD3 Windows RTX 4070 Laptop Execution Instructions

## Scope

This guide runs the approved one-seed high-dimensional diagnostic pipeline on a
single Windows laptop with an NVIDIA RTX 4070 Laptop GPU and 8 GB VRAM.

The run is **diagnostic only**. Seed `101` is used once; the experiment cannot
establish statistical superiority or a general replacement for CN/PSOR.

The implementation branch is:

```text
exp/hd2-hvi-objective-stabilization
```

Do not merge the draft implementation PR before the laptop evidence has been
inspected and accepted.

## What The One Command Does

The release runner executes, in order:

```text
preflight
HD2 d=2/5 historical-anchor gradient diagnosis
HD2 mandatory stabilization screen
conditional CAGrad / 8-probe / ReLoBRaLo only when frozen triggers permit
validation-only HD2 selection or the preregistered HVI-warm-low fallback
HD2 fresh diagnostic test evaluation
HD3 oracle/preflight
single-GPU parallelism benchmark
HD3 10D
HD3 20D
HD3 50D
HD3 100D stress only when the remaining budget permits
fresh HD3 fixed-test evaluation
compact evidence aggregation and SHA-256 verification
```

The HD3 architecture comparison is `FC_HVI` versus `DEEPSETS_HVI`. Both receive
the same seed, supervised stream, physics stream, and minibatch ordering at a
given dimension. The 10D/20D pair is concurrent only when the non-evidence
benchmark demonstrates at least a 10% wall-time benefit, both child jobs
succeed, and sampled total child-process GPU memory remains below 7.2 GiB.
50D and 100D are always sequential.

## Prerequisites

Use the existing project Python/CUDA environment. The runner does **not** install,
upgrade, or downgrade packages.

The default interpreter is:

```text
.\.venv-pinn\Scripts\python.exe
```

It must already provide PyTorch with CUDA, NumPy, SciPy, pandas, matplotlib,
PyYAML, pytest, scikit-learn, and tqdm. The preflight checks CUDA and
`nvidia-smi`, verifies at least approximately 7 GiB of GPU memory, requires at
least 5 GiB of free disk space, and runs the focused release test suite.

Keep the laptop plugged in and use the normal high-performance GPU/power mode.
Do not run another CUDA-heavy job concurrently with this experiment.

## Pull The Frozen Implementation

Run from the repository checkout:

```powershell
git fetch origin
git switch exp/hd2-hvi-objective-stabilization
git pull --ff-only origin exp/hd2-hvi-objective-stabilization
git rev-parse HEAD
git status --short
```

`git status --short` should show no tracked source modification before the first
preflight.

**Once the experiment has started, do not pull a newer implementation commit.**
Every checkpoint and phase status is bound to the implementation SHA and config
hash. A changed HEAD is intentionally rejected rather than silently resuming a
scientifically different run.

## Recommended First Check

Run the full preflight separately once:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hd2_hd3_release_windows.ps1 -Phase preflight
```

If preflight succeeds, the result is provenance-recorded. Running `all` next
skips that already-completed current-commit phase.

## Run Everything

The normal unattended command is:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hd2_hd3_release_windows.ps1 -Phase all
```

The command can remain running overnight. The intended active-compute ceiling
is 18 hours. That ceiling is a resource policy, not a promise that every laptop
TGP/thermal configuration will finish in the same time.

## Read-Only Status

At any point when the training process is not actively holding the terminal,
run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\check_hd2_hd3_release_status.ps1
```

The status reports phase state, model/dimension state, durable checkpoint step,
whether resume is available, the resolved HD2 objective, the parallelism
benchmark decision, and remaining budget estimate.

## Interrupt And Resume

A normal training checkpoint is atomically replaced every 250 optimizer steps.
It contains model state, optimizer state, training step, objective state,
batch-fallback state, Python/NumPy/Torch/CUDA RNG state, Sobol consumed-point
counter, elapsed training time, implementation SHA, config hash, and resolved
HD2-objective hash where applicable.

If you press `Ctrl+C`, close PowerShell, reboot Windows, or lose a CUDA process,
do **not** delete result/status/checkpoint files. Re-run exactly:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hd2_hd3_release_windows.ps1 -Phase all
```

Verified completed phases/jobs are skipped. An interrupted training job resumes
from its latest verified checkpoint. At most the work since the last 250-step
checkpoint is not guaranteed recoverable after an abrupt kill.

Do not use `git reset`, delete `results/.../runtime`, or manually edit status
JSON to force a phase to pass.

## OOM Handling

HD2 starts at batch 2048 and may use only the frozen fallbacks 1024 then 512.

HD3 sequential jobs start at batch 512 and may use only 256 then 128. If the
parallel benchmark enables concurrency, 10D/20D concurrent jobs start at batch
256 and may use only 128.

An OOM is recorded. If every frozen fallback is exhausted, the job remains a
resource failure. The runner does not silently shrink the network, remove a
Hutchinson probe, drop a dimension, alter the loss, or change the seed.

## HD2 Conditional Logic

The mandatory screen always runs the fixed candidates first. CAGrad is executed
only if the frozen persistent negative-gradient-cosine criterion is met. The
8-probe candidate is executed only if the d=5 exact-vs-Hutchinson diagnostic
crosses the frozen contraction-error or physics-gradient-cosine trigger.
ReLoBRaLo is executed only if no mandatory fixed candidate passes and enough
budget remains after reserving HD3 canonical and finalization time.

If no candidate passes the price-first HD2 gates, HD3 still continues with the
preregistered fallback:

```text
candidate = HVI-warm-low
warm_fraction = 0.5
final_physics_weight = 0.01
```

That fallback is recorded as `fallback_objective=true`; it is not presented as
an HD2 promotion.

## HD3 Dimensions And Interpretation

Canonical dimensions:

```text
10D
20D
50D
```

Stress dimension:

```text
100D
```

The 100D pair starts only if measured 50D time, remaining active budget, the
4000/6000 step ratio, a conservative 50D->100D dimension factor, and a one-hour
finalization reserve permit it. A budget-skipped 100D run is a legitimate
resource result, not a hidden failure.

HD3 remains an intentionally reducible equal-weight geometric-basket American
put. A DeepSets advantage is evidence for this permutation-symmetric controlled
problem, not universal evidence for arbitrary high-dimensional American
stopping boundaries.

## Detailed Local Runtime Evidence

Detailed runtime material remains local and ignored by Git:

```text
results/hd2_hvi_objective_stabilization/runtime/
results/hd3_classical_vs_structured_high_dimensional_pinn/runtime/
results/hd_pipeline/runtime/
```

These directories contain `.pt` checkpoints, per-job status JSON, JSONL
training histories, detailed gradient diagnostics, OOM records, non-evidence
smoke outputs, and PowerShell/Python logs.

Do not delete them until the experiment has been accepted and no further resume
is needed.

## Compact Git Evidence

Finalization regenerates compact evidence such as:

```text
results/hd2_hvi_objective_stabilization/
  environment_manifest.json
  config_manifest.json
  preflight.json
  conditional_methods.json
  selection_decision.json
  test_metrics.csv
  training_history.csv
  validation_history.csv
  gradient_diagnostics.csv
  layerwise_gradient_diagnostics.csv
  hutchinson_gradient_validation.csv
  final_status.json

results/hd3_classical_vs_structured_high_dimensional_pinn/
  environment_manifest.json
  config_manifest.json
  preflight.json
  resolved_objective.json
  parallelism_benchmark.json
  parallelism_benchmark.csv
  run_metrics.csv
  dimension_summary.csv
  structural_diagnostics.csv
  training_history.csv
  validation_history.csv
  final_status.json

results/hd_pipeline/
  environment_manifest.json
  config_manifest.json
  run_ledger.csv
  artifact_hashes.json
  final_status.json
```

Not every conditional file is expected when its trigger is false. The final
artifact hash manifest covers the generated compact evidence present at
finalization.

## After The Run

First inspect status:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\check_hd2_hd3_release_status.ps1
```

Then inspect Git staging candidates:

```powershell
git status --short
```

The ignored `runtime/` directories must not be staged. To stage only the compact
result trees after a successful/terminal run:

```powershell
git add results/hd2_hvi_objective_stabilization results/hd3_classical_vs_structured_high_dimensional_pinn results/hd_pipeline
git status --short
```

Before committing, review the staged list and confirm that it contains no `.pt`,
`.npz`, environment directory, token, credential, or full log.

If the compact evidence is correct:

```powershell
git commit -m "results(hd2-hd3): add RTX 4070 laptop diagnostic run"
git push origin exp/hd2-hvi-objective-stabilization
```

Keep the result commit separate from the implementation commits. Do not merge
the draft PR automatically.

## What To Send Back For Review

Return the output of the read-only status command plus the committed result SHA.
The compact `run_metrics.csv`, `selection_decision.json`,
`parallelism_benchmark.json`, `run_ledger.csv`, and `artifact_hashes.json` are
sufficient for the first scientific review; full `.pt` checkpoints and logs are
not needed unless a specific failure must be debugged.
