# Token-Budgeted Experiment And Model Policy

## Purpose

The independent-method campaign must conserve Codex usage without weakening
scientific standards. The principal controls are small task scope, staged
experiments, limited context, explicit stopping rules, and capability-aware
model selection. A cheaper model alone does not solve repeated repository
reading, duplicated reviews, invalid canonical runs, or oversized handoffs.

This policy supplements the repository-wide rules in `AGENTS.md`. If the two
documents appear to conflict, follow `AGENTS.md` and ask the user before
continuing.

## Ownership And Access

The user is the research owner and the only person who creates Codex tasks and
assigns models. The project manager decides what work is needed, prepares a
bounded prompt, verifies returned evidence, and recommends whether to accept,
revise, rerun, commit, push, or reject the work.

The manager must not create tasks or subagents. This restriction also applies
when a product tool could create them automatically. When a specialist is
needed, the manager provides the user with:

1. Task name.
2. Recommended model and reason.
3. Repository or worktree absolute path.
4. Required branch and accepted base commit.
5. Files allowed to be read or modified.
6. Complete prompt and expected artifacts.
7. Stop conditions and test commands.
8. Maximum response format and acceptance criteria.

The user-created task must begin with:

```bash
git rev-parse --show-toplevel
git status --short --branch
git rev-parse HEAD
```

It must stop if the root is not `american-option-risk-surfaces`, if the assigned
branch is not available, or if unrelated changes make the requested edit
unsafe. It must not recreate project files in an accessible but unrelated
directory.

## Work Unit

One task handles one method and one phase only:

```text
design -> implementation -> smoke review -> canonical run -> acceptance review
```

Each arrow is a checkpoint. A task returns after its assigned phase instead of
continuing automatically. The manager checks the result and the user decides
whether to create the next task.

Report writing, figure production, and cleanup are separate downstream phases.
They do not run concurrently with a numerical implementation unless the user
explicitly requests that overlap. No more than one implementation task and one
focused review task should be active for the same method.

## Method Packet And Context Budget

Every task prompt should contain a small method packet rather than the full
project history. The packet contains:

- scientific objective and primary decision gate;
- equations and notation needed for the phase;
- accepted baseline commit;
- permitted files and relevant tests;
- frozen seeds, splits, precision, budgets, and timing protocol;
- required output schema and artifact paths;
- known blockers and non-goals;
- stop conditions and compact completion format.

Tasks inspect named files first and use `rg` or a focused `git diff` to locate
additional dependencies. Reading all Deep Research reports, historical weekly
reports, old task transcripts, or the full repository requires a specific
justification. A review task normally reads only the accepted base, candidate
commits, focused tests, and compact evidence artifacts.

Do not paste long logs into task messages. Save them under the appropriate
ignored log directory and return only the failing command, exit status, and the
smallest diagnostic excerpt needed to understand the failure.

## Model Selection

Model selection is a recommendation made by the manager and a decision made by
the user.

| Work type | Recommended model | Reason |
|---|---|---|
| Scientific design, difficult derivation, final claim and acceptance review | GPT-5.6 Sol | Highest error cost and strongest cross-domain reasoning requirement |
| Normal implementation, focused debugging, LaTeX integration | GPT-5.6 Terra | Strong implementation quality with a better capability-cost balance |
| Inventory, status checks, schema checks, authorized cleanup | GPT-5.6 Luna | Narrow and rule-governed high-volume work |
| Independent mathematical second opinion | DeepSeek Reasoner | Useful independent derivation before an expensive run |
| Bounded low-risk implementation alternative | DeepSeek V4 Pro | Candidate replacement for focused Terra work after qualification |
| Log triage, short summaries, repetitive low-risk checks | DeepSeek V4 Flash or Chat | Efficient supporting work with limited scientific authority |

Use Sol only at scientific decision points. A Sol review should receive compact
evidence rather than repeat routine file inventory or full log parsing. Use a
lower-cost model only when the task has explicit inputs, outputs, and tests.

## DeepSeek Qualification And Safety

DeepSeek is connected through a local OpenCodex proxy and an external DeepSeek
API. This can reduce consumption of the Codex weekly allowance, but it transfers
usage to DeepSeek API billing and limits. It is not free computation.

Before a DeepSeek model modifies the repository, give it a qualification task:

```text
Read AGENTS.md and only the two named files. Do not modify files. Verify one
known formula or one small known bug. Return at most 20 lines containing the
repository path, evidence, uncertainty, and visible usage information.
```

The manager checks whether it obeyed file scope, used tools correctly, reached
the known answer, avoided secrets, and followed the response limit. A failed
qualification restricts that model to second-opinion text work.

DeepSeek output is always provisional. It may propose code or analysis, but a
trusted review must verify equations, focused tests, artifact provenance, and
claim language before acceptance. DeepSeek must not:

- approve a scientific or performance claim;
- decide that a method beats CN/PSOR;
- access or reveal the blind holdout;
- merge, push, or publish experimental evidence without manager review;
- receive credentials, API keys, authentication state, or unnecessarily broad
  private-repository context.

## Verification Ladder

Every stochastic or numerical method follows this order:

```text
static review
    -> analytic or theorem control
    -> focused unit and contract tests
    -> tiny deterministic smoke run
    -> manager review of formulas and artifacts
    -> canonical five-seed screen
    -> manager acceptance
    -> optional ten-seed confirmation
```

Canonical work is forbidden while an earlier stage is unresolved. Local scripts
perform training, bootstrap, repeated timing, artifact hashing, and aggregation.
The model should launch or monitor those programs without repeatedly reasoning
over unchanged state.

## Immediate Stop Conditions

Stop and report the blocker when any of these conditions appears:

- focused test, formula check, analytic control, or schema check fails;
- a required metric is non-finite, null, or silently unavailable;
- the primary metric cannot be computed;
- required seeds, complete-regime splits, precision, budget, warm-ups, or timing
  repetitions differ from the protocol;
- a broad exception hides a scientific diagnostic failure;
- provenance commit does not equal the code used for the run;
- the working tree contains unintended changes or staged virtual environments;
- a required promotion metric or gate was removed or weakened;
- the task would access the blind holdout before configuration freeze;
- the phase exceeds its declared time or context scope.

An unavailable metric may proceed only when the frozen protocol explicitly
defines it as optional and the result records a machine-readable reason. Missing
required diagnostics must never count as a pass.

## Git And Artifact Protocol

Each method uses a focused branch or worktree based on the accepted integration
commit. If a branch has an older base, review and cherry-pick the relevant
method commits instead of merging the whole branch.

Before commit, check:

```bash
git status --short --branch
git diff --check
git diff --cached --check
git diff --cached --name-only
```

Never stage `.venv*`, credentials, caches, large checkpoints, or unrelated
generated files. Commit compact CSV/JSON summaries, configuration manifests,
artifact hashes, focused tests, and concise implementation notes. The artifacts
must name the exact branch and commit used to generate them. If code changes
after a run, rerun the affected evidence rather than editing provenance by hand.

Push reviewed integration checkpoints after a meaningful wave. Experimental
branches without review do not need immediate pushing. Merging to `main` follows
the user-requested PR, CI, report-inspection, and acceptance workflow.

## Completion Message

A worker returns no more detail than needed for review:

```text
Task and phase:
Repository / branch / commit:
Files changed:
Commands run and exact pass/fail counts:
Primary metric and decision status:
Artifacts and provenance:
Known limitations or blockers:
Decision requested from manager:
```

Full logs remain in files. The manager independently opens the changed code and
artifacts, reruns proportionate checks, and either accepts the result or asks the
user to create a bounded refinement task. A completion message is never, by
itself, evidence of correctness.

## Manager Task Request Template

When specialist work is needed, the manager sends the following to the user:

```text
Suggested task name: <name>
Recommended model: <model and reason>
Repository/worktree: <absolute path>
Branch and accepted base: <branch> / <commit>
Permitted files: <small explicit list>

Prompt:
Read AGENTS.md first. Verify the repository root, branch, HEAD, and status before
working. Perform only <one method and one phase>. Do not create subagents or
expand scope. Stop on any AGENTS.md experimental gate. Run <focused commands>.
Return only the compact completion format from
docs/campaign/token_budget_policy.md and notify the manager/user when finished.

Acceptance criteria: <tests, metrics, artifacts, and scientific checks>
Stop conditions: <phase-specific conditions>
```

The manager does not send the prompt to a newly created task. The user creates
the task in the correct repository context, assigns the recommended or chosen
model, and pastes the prompt.
