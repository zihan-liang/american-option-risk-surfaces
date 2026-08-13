# HD0 Windows RTX 4060 Calibration Packet

## Scope

HD0 is a hardware and precision feasibility check, not a high-dimensional PINN
experiment and not an M01--M24 campaign candidate. It asks whether an RTX 4060
8 GB can reproduce a fixed one-dimensional surrogate workload in GPU float32
closely enough to justify HD1 development.

HD0 must not access the blind holdout, claim CUDA superiority in general, or
reopen M02's unavailable boundary diagnostic.

## Frozen Comparison

Compare exactly two backends on the same Windows machine:

1. CPU float64 reference.
2. NVIDIA CUDA float32 candidate.

Both runs use the same dataset subset, complete regimes, architecture, initial
seed, minibatch order, epoch count, and evaluation points. Boundary RMSE is not
an HD0 gate. Required outputs are price, Delta, normalized `K*Gamma`, masked PDE
residual, obstacle/complementarity diagnostics, training time, inference time,
peak memory, and fixed-seed repeatability.

Use one seed for the implementation smoke. Use seeds `101`, `203`, and `307`
only after the smoke passes and the manager accepts the implementation. HD0
does not use the M02 five-seed screen or its canonical result directory.

## Go/No-Go Gates

CUDA float32 may advance to HD1 preparation only when all of these hold on the
predeclared validation regimes:

- finite outputs and metrics;
- maximum obstacle violation no worse than `1e-7`;
- price RMSE no worse than CPU float64 by more than 2%;
- Delta RMSE no worse by more than 5%;
- normalized `K*Gamma` RMSE no worse by more than 10%;
- masked PDE residual mean absolute value no worse by more than 10%;
- two repeated fixed-seed CUDA runs agree within `1e-5` in price RMSE;
- measured CUDA peak allocation remains below 7.0 GiB; and
- end-to-end CUDA training time is recorded, regardless of whether it is
  faster.

Failure produces `no_go` or `resource_blocked`; it never triggers automatic
precision changes, smaller evaluation masks, or a favorable claim.

## Required Windows Probe

Run in PowerShell and save the complete output under an ignored local log:

```powershell
nvidia-smi
python --version
python -c "import torch; print({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda_runtime': torch.version.cuda, 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
```

Stop if CUDA is unavailable, the device is not the expected RTX 4060, or the
dataset is missing. Do not install or upgrade packages from inside an active
experiment run.

## Git State

- Repository: `american-option-risk-surfaces`
- Integration branch: `campaign/independent-methods`
- HD0 branch: `exp/hd0-windows-gpu-calibration`
- Accepted M02 closure commit: `7e786d3`
- Starting state: the current pushed tip of `campaign/independent-methods` that
  contains this packet and has `7e786d3` as an ancestor
- Result directory: `results/hd0_windows_gpu_calibration/`

The Windows task creates the HD0 branch only after confirming that its local
`campaign/independent-methods` is up to date and contains the accepted M02
closure commit.

## Task Creation

Suggested task name: `HD0 Windows GPU Calibration`

Recommended model: GPT-5.6 Terra Medium. The work is bounded implementation and
hardware diagnosis. Sol is reserved for manager acceptance of the scientific
comparison.

### Stage 1 Initialization Prompt

```text
Initialization only.

Report:
git rev-parse --show-toplevel
git status --short --branch
git rev-parse HEAD
git branch --show-current

Do not read project files, modify files, install packages, run tests, or start
HD0. Wait for the full task prompt.
```

### Stage 2 Full Prompt

Send this only after the Project Manager verifies the Windows repository root,
branch, HEAD, and clean status:

```text
Implement HD0 only: Windows RTX 4060 CPU-float64 versus CUDA-float32 calibration.

Read first:
- AGENTS.md
- docs/campaign/token_budget_policy.md
- docs/campaign/hd0_windows_gpu_calibration.md
- configs/independent_m02_obstacle_pinn.yaml
- experiments/27_m02_obstacle_aware_pinn.py
- src/models/obstacle_pinn.py
- src/models/premium_surrogate.py
- tests/test_m02_obstacle_pinn_losses.py

Repository must end in /american-option-risk-surfaces.
Start from the current pushed campaign/independent-methods tip. It must contain
accepted M02 closure commit 7e786d3 as an ancestor and this HD0 packet.
Create and work only on branch exp/hd0-windows-gpu-calibration.
Do not create tasks or subagents. Do not access the blind holdout. Do not run
M02 canonical and do not modify M02 canonical artifacts.

Use TDD to add a separate HD0 config, experiment, focused tests, and compact
CSV/JSON outputs. Keep CPU float64 and CUDA float32 runs matched in data,
complete-regime selection, architecture, seed, minibatch order, epochs, and
evaluation points. Boundary RMSE is out of scope. Implement the gates exactly
as specified in docs/campaign/hd0_windows_gpu_calibration.md.

First run only the environment probe, focused tests, and a one-seed tiny smoke.
Do not launch the three-seed screen. Save full logs locally and return only
compact results. If CUDA is unavailable, memory exceeds 7.0 GiB, any metric is
non-finite, or matched-workload invariants fail, stop and report the blocker.

Commit only the focused HD0 implementation and tests. Do not push or merge.
When complete or blocked, send the compact report to Project Manager task
019f20d4-3a13-79a0-bab6-67370e11b43b if task messaging is available, and also
reply in this task. Do not send routine progress updates.

Return at most 25 lines:
Task and phase:
Windows repository / branch / commit:
GPU and PyTorch/CUDA versions:
Files changed:
Focused tests and exact counts:
Smoke workload:
CPU float64 metrics/time/memory:
CUDA float32 metrics/time/memory:
Gate status:
Artifacts:
Known limitations:
Decision requested from manager:
```

## Manager Acceptance

The manager reviews the code diff, test results, environment manifest, and
compact smoke artifacts. Only after acceptance does the user instruct the same
Windows task to launch the three-seed HD0 screen from a local PowerShell job
with log and exit-status files. No Codex polling is used during that run.
