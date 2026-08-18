# HD2 -> HD3 One-Command Windows Pipeline Design

## Goal

Provide one Windows PowerShell entry point that can execute the complete laptop experiment from preflight through HD2 stabilization, objective freeze, HD3 10D/20D/50D scaling, optional 100D stress, and final artifact generation.

The same command must be safe to rerun after Ctrl+C, shell closure, Windows restart, CUDA process failure, or laptop reboot. Resume is the default behavior.

## User Interface

Primary command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hd_pipeline_windows.ps1 -Phase all
```

Read-only status:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hd_pipeline_windows.ps1 -Phase status
```

The implementation may expose HD2- and HD3-specific runners for debugging, but normal laptop execution should require only the top-level command.

## Pipeline State Machine

`all` executes:

```text
1. preflight
2. hd2_phase_a
3. hd2_screen
4. hd2_conditional
5. hd2_select
6. hd2_finalize
7. hd3_preflight
8. hd3_10d
9. hd3_20d
10. hd3_50d
11. hd3_100d_stress_if_budget_allows
12. final_evaluation
13. final_hashes
14. final_status
```

Every phase has a machine-readable status artifact and explicit dependency list. The runner never infers completion solely from the presence of a file.

## Resume Semantics

Resume is provenance-checked.

Every status/checkpoint/result record contains:

```text
repository commit SHA
resolved config SHA-256
phase ID
job ID
model ID
dimension
seed
resolved HD2 objective hash when applicable
```

On rerun:

- verified completed jobs are skipped;
- interrupted jobs resume from their latest verified checkpoint;
- failed numerical jobs remain failures unless the frozen protocol explicitly allows an OOM batch fallback;
- stale-provenance jobs are not resumed;
- missing/corrupt artifacts downgrade the job to incomplete and require recovery from the latest valid checkpoint or rerun of that job only.

No completed current-provenance canonical job is rerun merely because another later job failed.

## Checkpointing

Write an atomic `latest_checkpoint.pt` every 250 optimizer steps and at every warm-start/physics/final transition.

Checkpoint contents:

- model state;
- optimizer state;
- step;
- training phase;
- lambda/epsilon schedule position;
- Python/NumPy/Torch/CUDA RNG states;
- deterministic Sobol stream IDs and consumed-point counters;
- current batch-size fallback state;
- accumulated elapsed training seconds;
- job/provenance identifiers.

Checkpoint writes use a temporary path plus atomic replacement.

The runner traps normal interruption signals where PowerShell/Python permit, writes an `interrupted` status, and exits without rewriting completed evidence.

Abrupt termination is recovered from the last atomic checkpoint and flushed JSONL record.

## Detailed Result Logging

Detailed results are required, not optional.

Each job writes append-and-flush JSONL during execution so an abrupt stop loses at most the current record interval.

HD2 detail streams:

```text
events.jsonl
training_history.jsonl
validation_history.jsonl
gradient_diagnostics.jsonl
```

HD3 detail streams:

```text
events.jsonl
training_history.jsonl
validation_history.jsonl
```

Basic training records include:

- step;
- phase;
- supervised loss;
- physics/FB loss;
- total loss;
- lambda;
- epsilon;
- learning rate;
- batch sizes;
- elapsed seconds;
- seconds/step moving average;
- current CUDA allocation/reserved memory when available.

Validation records include all frozen price/Greek/PDE/FB/boundary/structure metrics available at that phase.

HD2 gradient records include global and layerwise norms, ratios, cosines, descent projections, diagnostic batch size, and diagnostic wall time.

## Compact Result Generation

Runtime JSONL/checkpoints are the recovery source of truth. Compact CSV/JSON files are deterministic derived artifacts.

After each completed phase and at finalization, regenerate compact summaries from the durable runtime logs.

A partially written final CSV must never be treated as canonical evidence; canonical summaries are written atomically.

## Run Ledger

Maintain a single top-level ledger:

```text
results/hd_pipeline/run_ledger.csv
```

Columns include at least:

```text
pipeline_phase
experiment_id
job_id
dimension
model
candidate
run_class
seed
status
start_time
end_time
elapsed_seconds
last_completed_step
target_steps
batch_size
oom_count
checkpoint_path
implementation_sha
config_hash
objective_hash
result_hash_status
```

This ledger is regenerated from per-job statuses rather than manually edited.

## Global Environment Manifest

Preflight records:

- Windows version;
- hostname hash or non-identifying machine label;
- Python version/path;
- PyTorch version;
- CUDA runtime version;
- NVIDIA driver;
- GPU model;
- total VRAM;
- CPU model/logical-core count;
- RAM;
- relevant package versions;
- mixed precision setting;
- Git branch and implementation SHA;
- dirty-worktree state.

Canonical work refuses to start from a dirty implementation worktree unless the only changes are ignored runtime/result files explicitly allowed by the protocol.

## Preflight Tests

Before any canonical work, the one-command runner executes:

1. repository/provenance check;
2. CUDA availability and small tensor backward test;
3. focused HD2/HD3 tests;
4. oracle smoke/refinement checks;
5. d=2 exact-vs-Hutchinson correctness check;
6. checkpoint write/read test;
7. short interruption/resume equivalence test if not already verified for the exact implementation SHA;
8. free disk-space check;
9. writable result/runtime directories;
10. laptop power-state warning if GPU is not on AC/high-performance mode when detectable.

A failed hard preflight stops before canonical training.

## Parallelism Controller

The pipeline contains a small scheduler, not unrestricted background process launching.

HD2 diagnostic/screen jobs are sequential by default because objective-specific higher-order gradient diagnostics are memory-sensitive and cheap enough that concurrency is not required.

HD3 runs the 10D parallelism benchmark. If the frozen speed/memory rule passes, the scheduler may execute the FC/DeepSets pair concurrently for 10D and 20D. Otherwise they remain sequential.

50D and 100D are always scheduled sequentially.

The scheduler writes process IDs, start/end times, exit codes, log paths, and sampled GPU-memory records to status artifacts.

## Global 18-Hour Budget

The pipeline stores one `pipeline_start_time` and applies the same 18-hour wall-clock cap across HD2 and HD3.

Before every optional/conditional job, compute:

```text
elapsed_time + projected_required_remaining_time
```

Conditional HD2 methods and the 100D stress are skipped if starting them would threaten completion of mandatory work inside the cap.

Mandatory order priority:

```text
HD2 mandatory diagnosis/screen
HD2 deterministic selection/fallback
HD3 10D
HD3 20D
HD3 50D
finalization
```

Optional priority:

```text
HD2 conditional comparator(s)
HD3 100D stress
```

If mandatory projected time itself exceeds the cap, emit `resource_blocked` rather than silently shrinking the frozen scientific protocol.

## Logs

Each job has a plain-text stdout/stderr log under an ignored runtime log directory. The top-level runner writes a concise pipeline log containing only phase transitions, status, ETA, OOM fallback, interruption/resume, and final summary.

Detailed full logs are local artifacts and are not committed unless a small excerpt is explicitly curated for a documented software failure.

## Result Commit Boundary

Implementation is committed before laptop execution.

After the laptop run completes, only compact evidence is intended for the result commit:

```text
HD2 compact CSV/JSON artifacts
HD3 compact CSV/JSON artifacts
pipeline run ledger
environment manifest
selection/fallback decision
artifact hashes
closure note
```

Do not commit:

```text
.pt checkpoints
raw oracle arrays
full stdout logs
large tensors
large transient monitoring files
```

## Failure Semantics

Scientific failure and software interruption are distinct.

Examples:

- Ctrl+C/reboot -> `interrupted`, resumable;
- OOM with fallback remaining -> resumable same job with next batch size;
- OOM with no fallback remaining -> `oom_exhausted`, preserve result;
- NaN/non-finite metric -> `failed_numerical`, no silent retry with changed science;
- stale commit/config -> `stale_provenance`, refuse resume;
- 18-hour cap -> `resource_blocked` for unstarted/unfinished optional work;
- failed HD2 selection -> valid `no_stabilized_candidate`, continue HD3 with frozen fallback objective.

## Status Output

`-Phase status` is read-only and prints a compact table:

```text
phase | job | d | model | status | step/target | elapsed | ETA | batch | VRAM | checkpoint age
```

It also prints:

- current HD2 selection/fallback;
- whether HD3 objective is frozen;
- global elapsed time and remaining 18-hour budget;
- next runnable job;
- blockers requiring user action.

## Definition Of Done

The pipeline implementation is ready for the user's laptop only when tests demonstrate:

- one-command phase ordering;
- deterministic resume from a forced mid-run interruption;
- completed-job skipping;
- stale-provenance rejection;
- atomic checkpoint/status/summary writes;
- JSONL crash recovery;
- OOM fallback state persistence;
- HD2 selection/fallback handoff to HD3;
- HD3 objective immutability after canonical start;
- final artifact-hash verification;
- no access to the independent-campaign blind holdout.
