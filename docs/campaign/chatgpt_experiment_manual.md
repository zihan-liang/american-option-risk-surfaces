# ChatGPT Independent-Method Experiment Manual

## Purpose

This manual transfers implementation of M07--M24 to ChatGPT while preserving
the approved independent-method protocol. M01--M06 are already recorded and
must not be rerun or rewritten. The project manager will review all commits,
evidence, and claims after M24.

M04 currently has reviewed implementation but no accepted real-data screen;
its recorded data-freeze blocker remains valid campaign evidence. Do not invent
a dataset or reopen M04 during this handoff. The project manager will decide
whether to revisit it after M24.

This is an experiment campaign, not a competition that must produce winners.
`failed_metric_gate`, `resource_blocked`, `research_blocked`, and
`diagnostic_only` are valid outcomes. Never alter a gate after seeing a result.

## Required Reading

For each method, read only:

1. `AGENTS.md`.
2. `docs/campaign/token_budget_policy.md`.
3. This manual.
4. The corresponding section in
   `docs/campaign/chatgpt_method_packets_m07_m24.md`.
5. The one named Deep Research file, when it exists.
6. The smallest relevant existing solver/model files located with `rg`.

Do not read all Deep Research files, weekly reports, task transcripts, or old
branches. Do not modify M01--M06 evidence.

## Repository And Git Contract

- Preferred local integration worktree:
  `/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/campaign-independent-methods`
- GitHub repository: `zihan-liang/american-option-risk-surfaces`, branch
  `campaign/independent-methods`.
- Integration branch: `campaign/independent-methods`.
- Do not try to check out this branch in the main repository directory while
  the preferred worktree above exists; Git correctly permits one worktree per
  checked-out branch.
- Start every method from the latest pushed integration branch and a clean
  worktree. Methods run sequentially; do not edit the same branch concurrently.
- Use exactly two commits per method when a canonical run is possible:
  `feat: add standalone Mxx <name>` and `results: close Mxx <name>`.
- The first commit contains config, implementation, tests, and method plan. Run
  the canonical experiment only after that commit so the manifest records its
  exact SHA. The second commit contains compact evidence and closure notes.
- Push after every completed method. Never force-push, amend published commits,
  merge to `main`, or rewrite prior evidence.
- Do not commit virtual environments, caches, checkpoints, large tensors, raw
  datasets, temporary logs, or secrets.

## Standard File Layout

For Mxx create:

```text
configs/standalone/mxx_<slug>.json
experiments/standalone/mxx_<slug>.py
tests/test_mxx_<slug>.py
docs/superpowers/plans/YYYY-MM-DD-mxx-standalone-<slug>.md
docs/campaign/mxx_closure.md
results/standalone/Mxx/
```

The plan is the handoff specification. It must state the equation or algorithm,
data source, frozen configurations, seeds, baselines, metrics, gates, expected
but non-binding outcome, limits, run command, and stop conditions. A future
reviewer must understand the experiment without the original conversation.

## Prompt To Start One ChatGPT Method

Use one fresh ChatGPT coding task per method to prevent context accumulation:

```text
Implement Mxx only on branch campaign/independent-methods.

Read AGENTS.md, docs/campaign/token_budget_policy.md,
docs/campaign/chatgpt_experiment_manual.md, and only the Mxx section of
docs/campaign/chatgpt_method_packets_m07_m24.md. Follow them exactly.

Use TDD. Run the bounded smoke and canonical protocol. Keep negative and
blocked outcomes. Make the implementation and result commits, push the branch,
then return the two SHAs, status, primary metric, tests, and blockers in at most
20 lines. Do not start another method in this task.
```

After it finishes, start the next method from the newly pushed integration
branch. A normal chat without repository write access can review or propose a
patch but cannot complete this contract; use a ChatGPT coding environment with
GitHub or local repository access.

## Standard Outputs

Every completed or blocked method writes finite, machine-readable artifacts:

```text
config_manifest.json
metrics.csv
decision.json
artifact_hashes.json
```

Add `training_history.csv` for learned methods and concise method-specific CSVs
only when required by the packet. `config_manifest.json` records the full
resolved config, branch, implementation commit, environment, dependency
versions, data hashes, seeds, split identifiers, `blind_holdout_accessed=false`,
and an explicit method-scope limitation. `artifact_hashes.json` hashes every
other compact output with SHA-256.

`decision.json` contains at least:

```json
{
  "method_id": "Mxx",
  "status": "diagnostic_only",
  "claim_status": "diagnostic_only",
  "primary_metric": "...",
  "numerical_gate_passed": false,
  "reason": "...",
  "limitations": ["..."],
  "blind_holdout_accessed": false
}
```

Allowed statuses are `candidate_supported`, `diagnostic_only`,
`failed_metric_gate`, `resource_blocked`, `research_blocked`, and
`software_failed`. Smoke outputs are always `diagnostic_only` and are never
canonical evidence.

## Shared Scientific Protocol

- Common benchmark: normalized one-asset American put under Black--Scholes.
- CN/PSOR remains the oracle and label generator. M01 policy iteration may be a
  classical timing control but does not replace the oracle.
- CPU float64 is canonical except M14, where Apple MPS is the candidate and CPU
  float64 remains the correctness oracle.
- Learned methods use seeds `[101,203,307,409,503]`. Add
  `[607,709,811,907,1009]` only when the five-seed confidence interval is
  inconclusive or the packet explicitly promotes confirmation.
- Split complete parameter regimes before expanding grid points. No scenario,
  time slice, or neighboring nodes from one regime may cross train,
  validation, calibration, or test boundaries.
- Existing test data influenced earlier work and is `legacy_test`. It may be
  used for diagnostics but never described as blind evidence.
- Do not access the sealed blind holdout. M23 freezes selections but defers
  holdout evaluation to the final manager-controlled stage.
- Hyperparameter selection uses validation data only and at most eight
  predeclared configurations per method. Record every attempted configuration,
  including failures.
- Resample complete regimes for uncertainty. Report paired mean differences,
  standard deviations, scenario-bootstrap 95% confidence intervals, and Holm
  correction within a method family when multiple claims are tested.
- A claimed improvement needs at least 2% practical gain and a paired 95%
  interval excluding zero. Ordinary diagnostics may not worsen more than 2%;
  residual diagnostics may not worsen more than 5%; hard obstacle violation
  must remain at most `1e-10` where hard feasibility is promised.
- Time deterministic candidates with five warm-ups and 30 repetitions. Record
  preprocessing, training, label generation, inference, fallback, and solver
  time separately. Do not compare against dense linear algebra when an O(N)
  tridiagonal method is the correct baseline.
- Screen budget is one hour per method, except the quantum diagnostic M07 and
  diffusion experiment M12 may use four hours. A run over budget becomes
  `resource_blocked`. Do not silently reduce the protocol after the budget is
  exceeded.

## Implementation And Test Rules

1. Write focused failing tests first and observe the expected failure.
2. Implement the smallest standalone method; do not refactor unrelated code.
3. Keep method logic isolated from the blind-holdout module.
4. Validate dimensions, finite values, seed counts, split disjointness, config
   caps, artifact hashes, and no-promotion behavior for smoke mode.
5. Run focused tests, `compileall`, `git diff --check`, and the full test suite.
6. Run the canonical screen once from a clean implementation commit. Rerun only
   for a documented software failure, never because metrics are disappointing.
7. Inspect compact CSV/JSON evidence before writing the closure. The closure
   reports negative results and limitations as prominently as positive ones.

Use the existing campaign Python environment if available:

```bash
PYTHONPATH=src \
/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/.venv-campaign/bin/python \
-m pytest -q -p no:cacheprovider tests/test_mxx_<slug>.py
```

If a required dependency is absent, first look for a project-local ignored
environment. Do not install a large framework, download data, or change the
global machine without user approval. Record `resource_blocked` when the
approved environment cannot support the method.

## Claim Safety

- Never say a method is better because only one metric improved.
- Never call a classical proxy a quantum implementation.
- Never call MC Dropout variance a 95% confidence interval.
- Never call conformal coverage conditional coverage without evidence.
- Never claim CUDA behavior from Apple MPS.
- Never call feature attribution a causal explanation.
- Never call a simulator timing result a quantum speedup.
- Never hide unavailable boundaries, failed seeds, NaNs, fallback time, label
  cost, or full-vector readout cost.
- Expected outcomes in the method packet are hypotheses, not targets. Results
  must not be tuned to match them.

## Blocker Rule

Continue to the next independent method after recording a genuine blocker.
Do not invent substitute data or silently weaken scope. M22--M24 require a
primary-source audit because their original Deep Research reports were missing;
the packet defines the experiment. Each audit must cite at least three relevant
primary sources, distinguish established results from project inference, state
the nearest precedent and the unresolved gap, and justify the proposed metrics
and limitations. If web access is unavailable, record `research_blocked` and
continue.

## Final Handoff To Project Manager

After M24, return only:

1. Branch name and current commit.
2. Implementation and result commit SHA for each M07--M24.
3. One-line status and primary metric for each method.
4. Any blocked methods and exact blocker.
5. Full-suite command and outcome.
6. Confirmation that the worktree is clean and pushed.

Do not send full logs or reproduce the complete manual. The project manager
will inspect commits and artifacts directly, rerun selected checks, compare all
methods by category, and commission the final bilingual report.
