# M23 Agentic Hyperparameter Search

## English

### 1. Problem definition

M23 tests a search policy, not whether an agent can produce persuasive prose. Three mutually blind Codex agents each receive eight validation-only configuration evaluations. Their aggregate best-so-far performance is compared with random search, a fixed grid, and Bayesian optimization under the same evaluator-call and runtime budgets. Frozen winners are evaluated once on the campaign holdout after the holdout seed is revealed by the manager.

Let `A` denote a search policy and `B=8` its per-replicate evaluation budget. At trial `t`, the policy proposes a configuration `lambda_t` using only its prompt, permitted repository state, and prior validation observations from the same replicate. The evaluator returns a predeclared validation objective plus diagnostics. The primary search outcome is the incumbent curve

```text
y_A(t) = min_{1 <= j <= t} validation_objective(lambda_j),  t=1,...,8.
```

The trial, not an agent message or shell command, is the unit of budget. Failed, duplicate, timed-out, and invalid configurations consume a trial. The method-level comparison uses three independent eight-trial replicates per policy, giving 24 evaluator calls for each of agentic, random, grid, and Bayesian search.

This interpretation creates 96 total evaluator calls. **Project inference:** equal aggregate budgets are the least ambiguous reading of “three agents, eight trials each” and “equal-budget” baselines. If the manager instead treats each agent as a separate policy, the protocol must be rewritten before any run; post-hoc budget reinterpretation invalidates the comparison.

### 2. Directly relevant prior work

1. Bergstra and Bengio show why random search is a necessary baseline and why a grid can waste evaluations when only a subset of hyperparameters matters [M23-S1]. Their empirical advantage is not guaranteed for every eight-trial space.
2. Snoek, Larochelle, and Adams formulate hyperparameter tuning as sequential Bayesian optimization and discuss cost-aware evaluations [M23-S2]. With only eight trials, startup design and priors can dominate, so these must be frozen.
3. Cawley and Talbot demonstrate that optimizing a finite-sample model-selection criterion can overfit that criterion and bias later performance estimates [M23-S3]. More intelligent adaptivity does not remove validation overfitting.
4. Dwork et al. analyze adaptive reuse of holdout data and propose mechanisms for limited safe reuse [M23-S4]. This campaign does not implement their reusable-holdout mechanism; therefore the simpler safe policy is zero holdout feedback until all configurations are frozen.
5. MLAgentBench evaluates language agents through executable ML experimentation and records action traces, final performance, time, and token cost [M23-S5]. It supports trace-based evaluation but does not establish Codex superiority in this option-pricing task.
6. Eggensperger, Lindauer, and Hutter document pitfalls in algorithm-configuration experiments, including unfair budgets and weak reporting [M23-S6]. Lindauer and Hutter give related empirical-evaluation guidance for search methods [M23-S7].
7. Optuna's primary paper describes study/trial abstractions, dynamic search spaces, and pruning [M23-S8]. M23 should use a static predeclared space and disable unequal-fidelity pruning in the primary comparison.
8. MLflow's current official tracking documentation specifies run records containing parameters, metrics, code versions, inputs, times, and artifacts [M23-S9]. M23 may use the repository's own JSON/CSV contracts instead of MLflow, but must preserve equivalent provenance.

### 3. Proposed experiment and fair-comparison design

#### 3.1 Freeze one search space

Before agents are launched, commit a typed search-space manifest. A feasible example for the existing width-128, depth-3 positive-premium MLP is:

| Hyperparameter | Domain |
|---|---|
| learning rate | log-uniform `[1e-4, 3e-3]` |
| weight decay | `{0, 1e-8, 1e-6, 1e-4}` |
| batch size | `{4096, 8192}` |
| patience | `{5, 8}` |
| premium output transform | predeclared categorical choices already implemented by the campaign |
| optimizer | predeclared categorical choices already implemented by the campaign |

Do not allow architecture depth, width, data volume, epochs, precision, target, or validation metric to vary in the primary search unless all policies receive exactly the same expanded domain. Unsupported proposals are invalid trials, not opportunities for an agent to edit the evaluator.

The fixed grid is an explicit ordered list of eight configurations from this same domain. Random search uses a committed random seed and the same distributions. Bayesian search uses the same domain, a committed sampler version/seed, two random startup trials, then six sequential proposals. Agent prompts include the domain schema and objective contract but not baseline proposals or outcomes.

#### 3.2 Freeze one evaluator

Every trial must call the same immutable evaluator. It owns:

- regime-level train/validation data access;
- campaign seed schedule;
- preprocessing and feature order;
- early stopping and maximum epochs;
- CPU float64 execution;
- objective and diagnostic calculation;
- timeout and memory limits;
- artifact hashing and trial-status validation.

Agents may propose configuration values and a short rationale. They may not edit training code, split logic, metrics, seeds, data files, or evaluator outputs. This narrows M23 to hyperparameter search; allowing arbitrary code changes would instead test open-ended ML engineering as in MLAgentBench [M23-S5].

The primary validation objective should be a predeclared scalar composed from price RMSE and hard feasibility gates, for example: reject non-finite metrics or obstacle violations above `1e-10`, then minimize validation price RMSE. Secondary metrics remain visible but cannot be used to change the objective after trials begin.

#### 3.3 Equalize all budgets

Each policy receives:

- three independent replicates;
- eight evaluator calls per replicate;
- the same per-trial timeout;
- the same aggregate CPU-time ceiling;
- the same initial baseline result, if any;
- the same allowed validation response schema;
- serial execution on the same machine;
- no replacement trials after failure or duplicate proposal.

For agentic search, each of the three blind agents is one replicate. For the other policies, use three committed seeds/orderings. Compare both evaluator-call count and measured wall time. If one policy exhausts its aggregate time before eight trials, remaining trials are recorded `resource_blocked`; another policy may not spend the unused time.

#### 3.4 Blindness and information barriers

Use separate workspaces for all replicates. Each workspace contains only:

- the frozen search-space schema;
- the evaluator client;
- the common task prompt;
- its own prior trial summaries;
- read-only public method documentation needed to interpret the fields.

It must not contain other policies' logs, holdout code or data, the secret holdout seed, final-evaluation artifacts, `legacy_test` metrics, or writable evaluator internals. Network access should be disabled or consistently allowed for every policy; the chosen rule and any cached documentation must be recorded.

### 4. Why the holdout remains inaccessible

Repeatedly reporting holdout results to a search policy converts the holdout into another adaptive validation set. Cawley and Talbot show how selection overfits noisy selection criteria [M23-S3], and Dwork et al. explicitly study validity loss under adaptive reuse [M23-S4]. The campaign has not adopted a reusable-holdout mechanism, so no holdout statistic, rank, pass/fail bit, prediction, gradient, exception detail, timing side channel, or file-presence clue may reach any policy before freeze.

Enforcement should be structural:

- keep the decimal seed outside Git until manager reveal;
- expose no holdout path in agent environments;
- deny the `final` phase unless all winner config hashes and commits are frozen;
- generate the 96 Sobol and 32 stress regimes only after reveal;
- permit exactly one evaluation of each frozen policy winner;
- write results to a manager-owned directory that agents cannot read during search;
- prohibit any post-holdout change, rerun, exception-based retry, or winner switch.

The holdout is for final estimation, not model selection. A poor holdout result remains the result.

### 5. Reproducible trial logging

Write one append-only row per evaluator call plus immutable JSON artifacts. Required fields are:

```text
campaign_id, method_id, policy, replicate_id, trial_index, trial_id,
agent_model, agent_version, prompt_hash, parent_trial_id, rationale,
proposal_timestamp, config_json, config_hash, search_space_hash,
code_commit, dirty_tree_hash, evaluator_hash, dataset_manifest_hash,
train_regime_hash, validation_regime_hash, seed_schedule,
start_time, end_time, wall_seconds, peak_rss_mb, status,
validation_objective, all_validation_metrics, gate_results,
stdout_hash, stderr_hash, artifact_hashes, cumulative_trials,
cumulative_wall_seconds, remaining_trial_budget, remaining_time_budget
```

Also preserve the agent action trace and messages as a hashed artifact, following the trace-oriented precedent of MLAgentBench [M23-S5]. Parameters, metrics, inputs, code versions, times, and artifacts correspond to the core run entities in current MLflow documentation [M23-S9]. Secrets and raw holdout data must never enter the trace.

Determinism is not assumed. Record package lock, Python version, macOS version, hardware identifier, thread counts, and any nondeterministic backend flags. Replaying a trial means rerunning the exact config and seed schedule; a replay is verification and must not feed another proposal unless it was preallocated as a trial.

### 6. Outcome analysis

Primary validation comparisons:

- final best validation objective after eight trials;
- best-so-far curve over trials one through eight;
- normalized area under the best-so-far curve;
- valid-trial, duplicate, failure, timeout, and resource-blocked counts;
- total evaluator wall time and agent token/tool cost.

Report all three replicates, not only the best agent. Pair replicates by committed seed/order index for descriptive differences. Three policy replicates are too few for a strong policy-level population claim. **Project inference:** use the campaign's scenario bootstrap only for the final frozen models' 128 holdout-regime errors; do not manufacture a narrow confidence interval by treating 24 adaptively selected trials as independent replicates.

After all winners are frozen, evaluate the four policy winners once on the same holdout regimes. Report paired scenario-level differences, 95% bootstrap intervals, and the campaign's 2% practical-effect rule. Apply Holm correction across the three predeclared contrasts of agentic search versus random, grid, and Bayesian search. Do not select a different agent after seeing holdout outcomes.

### 7. Falsification conditions

M23 is invalid if:

- policies receive different search domains, data, objectives, seed schedules, timeouts, or evaluator code;
- failed or duplicate agent trials are replaced but baseline failures are counted;
- the agent can change source code or metrics in what is labeled hyperparameter search;
- any policy accesses `legacy_test` or holdout information during search;
- baseline configurations or results leak into agent prompts;
- only the best of three agents is reported while all baseline replicates are retained;
- unequal trial counts are described as equal-budget;
- the winner, threshold, or contrast changes after holdout reveal;
- logs omit proposals, failures, provenance, timing, or action traces;
- the analysis treats adaptive trials as independent statistical replicates.

An agentic advantage is not supported unless the frozen agentic selection exceeds each predeclared baseline by at least the campaign's 2% practical threshold on holdout and the paired 95% interval excludes zero after correction. Otherwise report `no_advantage`, `inconclusive`, or `resource_blocked` as appropriate.

### 8. Compute feasibility on Apple M5 / 16 GB

The strict equal-budget design uses 96 evaluator calls. To finish within the one-hour M23 screening budget, the mean end-to-end allowance is only

```text
3600 seconds / 96 = 37.5 seconds per trial.
```

The canonical 150,000/75,000-point, 30-epoch CPU float64 workload may exceed that allowance. A timed calibration run must occur before launching any policy and cannot be used to tune search behavior.

Recommended decision rule:

1. Benchmark three predeclared representative configurations with all outputs discarded.
2. If the conservative P90 trial time is at most 37.5 seconds, run the full 96-call screen.
3. Otherwise either obtain manager approval for a larger M23 budget or freeze a lower-cost M23-specific benchmark before any agent sees validation feedback.
4. A lower-cost benchmark must keep identical data, epochs, and fidelity across all policies and limits claims to that benchmark. It must not be presented as canonical-surrogate optimization.
5. If neither option is approved, return `resource_blocked` rather than silently reducing baseline budgets.

Sequential CPU float64 trials keep memory within 16 GB; one width-128 MLP and batched arrays are modest. Parallel agent evaluation is prohibited because it changes contention and timing. The language agents may reason concurrently only while no evaluator job is running, but serial orchestration is simpler and more auditable.

### 9. Primary-source table

| ID | Primary source | DOI / arXiv / official URL | Direct support used here |
|---|---|---|---|
| M23-S1 | Bergstra, J. and Bengio, Y. (2012), *Random Search for Hyper-Parameter Optimization*, JMLR 13:281-305 | [JMLR](https://www.jmlr.org/papers/v13/bergstra12a.html) | Random search as a reproducible baseline and its contrast with grids. |
| M23-S2 | Snoek, J., Larochelle, H., and Adams, R. P. (2012), *Practical Bayesian Optimization of Machine Learning Algorithms*, NeurIPS 25 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2012/hash/05311655a15b75fab86956663e1819cd-Abstract.html) | Sequential Bayesian optimization and variable evaluation cost. |
| M23-S3 | Cawley, G. C. and Talbot, N. L. C. (2010), *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, JMLR 11:2079-2107 | [JMLR](https://jmlr.org/papers/v11/cawley10a.html) | Overfitting finite-sample selection criteria and biased evaluation. |
| M23-S4 | Dwork, C. et al. (2015), *The Reusable Holdout: Preserving Validity in Adaptive Data Analysis*, Science 349(6248):636-638 | [DOI 10.1126/science.aaa9375](https://doi.org/10.1126/science.aaa9375), [arXiv:1506.02629](https://arxiv.org/abs/1506.02629) | Adaptive holdout reuse threatens validity; special mechanisms are needed for reuse. |
| M23-S5 | Huang, Q., Vora, J., Liang, P., and Leskovec, J. (2024), *MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation*, ICML, PMLR 235:20271-20309 | [PMLR](https://proceedings.mlr.press/v235/huang24y.html) | Executable ML-agent evaluation, action traces, success, time, and token metrics. |
| M23-S6 | Eggensperger, K., Lindauer, M., and Hutter, F. (2019), *Pitfalls and Best Practices in Algorithm Configuration*, JAIR 64:861-893 | [DOI 10.1613/jair.1.11420](https://doi.org/10.1613/jair.1.11420), [arXiv:1705.06058](https://arxiv.org/abs/1705.06058) | Fair budgets, reproducibility, and algorithm-configuration pitfalls. |
| M23-S7 | Lindauer, M. and Hutter, F. (2020), *Best Practices for Scientific Research on Neural Architecture Search*, JMLR 21(243):1-18 | [arXiv:1909.02453](https://arxiv.org/abs/1909.02453) | Controlled, transparent empirical comparison of search procedures. |
| M23-S8 | Akiba, T. et al. (2019), *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD 2019:2623-2631 | [DOI 10.1145/3292500.3330701](https://doi.org/10.1145/3292500.3330701), [arXiv:1907.10902](https://arxiv.org/abs/1907.10902) | Study/trial abstraction, search, pruning, and reproducible HPO software design. |
| M23-S9 | MLflow Project, *MLflow Tracking* (current official documentation, accessed 2026-08-11) | [Official documentation](https://mlflow.org/docs/latest/ml/tracking/) | Logging parameters, metrics, code versions, inputs, times, runs, and artifacts. |

### 10. Claim safety

Allowed wording:

- "Under 8-trial validation-only budgets on the frozen M23 benchmark, the agentic policy achieved [result] relative to random, grid, and Bayesian search."
- "The selected configurations were frozen before one holdout evaluation."
- "The screen was resource-blocked under the one-hour local budget."

Disallowed wording:

- "Codex is a better hyperparameter optimizer in general."
- "Validation superiority proves generalization."
- "The holdout was blind" without access-control evidence.
- "Equal budget" if evaluator calls, fidelity, or aggregate time differ.
- Any claim that the agent replaces Bayesian optimization, human researchers, or the numerical solver.

**Explicit project inferences:** the 24-calls-per-policy interpretation, 96-call total, search space, stability/statistical treatment, and 37.5-second feasibility threshold are proposed protocol decisions, not findings from the cited sources.

---

## 中文

### 1. 问题定义

M23 检验的是搜索策略，而不是代理能否写出有说服力的说明。三个相互隔离的 Codex 代理各获得 8 次仅使用验证集的配置评估。代理策略的 best-so-far 曲线应与随机搜索、固定网格和贝叶斯优化在相同评估次数与时间预算下比较。所有优胜配置在 holdout 揭示前冻结，之后仅评估一次。

每个策略使用三个独立重复，每个重复 8 次，因此代理、随机、网格和贝叶斯策略各有 24 次 evaluator 调用，总计 96 次。失败、重复、超时和非法配置都消耗一次 trial。**项目推断：** 这是对“3 个代理各 8 次”和“等预算基线”最清晰的解释；若管理者采用别的预算单位，必须在运行前重写协议。

### 2. 直接相关的既有工作

- Bergstra 和 Bengio 说明了随机搜索作为必要基线的原因，以及高维网格可能浪费评估 [M23-S1]。
- Snoek 等人把调参表述为顺序贝叶斯优化，并讨论不同评估成本 [M23-S2]。
- Cawley 和 Talbot 证明有限样本模型选择准则可能被过拟合，从而污染后续性能估计 [M23-S3]。
- Dwork 等人研究自适应重复使用 holdout 的有效性问题 [M23-S4]。本项目没有实现 reusable holdout 机制，因此应采用零反馈策略。
- MLAgentBench 支持通过可执行任务、动作轨迹、性能、时间和 token 成本评估语言代理 [M23-S5]，但不能直接证明 Codex 在本任务上优于其他搜索方法。
- Algorithm configuration 与 NAS 的主要论文提供了公平预算和可复现实验的注意事项 [M23-S6, M23-S7]。
- Optuna 论文和 MLflow 官方文档分别支持 trial/study 抽象以及参数、指标、版本、输入、时间和产物记录 [M23-S8, M23-S9]。

### 3. 本项目建议实验与公平比较设计

在启动代理前提交固定的类型化搜索空间。所有策略必须共享相同的学习率、weight decay、batch size、patience、优化器和已实现的 premium 变换选项。主实验不得允许代理单独改变深度、宽度、数据量、epoch、精度、目标或验证指标。

固定网格必须是预先提交的 8 个配置；随机搜索使用已承诺的随机种子；贝叶斯优化使用固定版本、先验与种子，前 2 次随机启动，后 6 次顺序建议；代理只接收搜索空间、目标契约和自身历史，不接收其他策略的配置或结果。

所有 trial 通过同一个只读 evaluator 执行。evaluator 独占数据切分、种子、预处理、early stopping、CPU float64、指标、超时、内存限制和产物哈希。代理只提交配置与简短理由，不得修改训练代码或指标。每个策略均有三个重复、每重复 8 次、相同超时和总 CPU 上限。失败 trial 不补发。

### 4. holdout 必须保持不可访问

向搜索策略反复返回 holdout 结果会把 holdout 变成另一个自适应验证集 [M23-S3, M23-S4]。因此在所有配置冻结前，任何 holdout 指标、排名、通过位、预测、异常细节、时间侧信道或文件存在信息都不得返回给代理。

应在结构上执行隔离：种子保存在 Git 外；代理环境没有 holdout 路径；`final` 阶段只有在配置哈希与提交冻结后才可运行；揭示后才生成 96 个 Sobol 域内情景和 32 个压力情景；每个冻结优胜配置仅评估一次；结果写入代理不可读的管理者目录；禁止根据 holdout 改配置、换优胜者或重跑。

### 5. 可复现 trial 日志

每次 evaluator 调用必须写入不可追加修改的记录，包括策略、重复、trial 序号、代理模型与版本、prompt 哈希、父 trial、理由、配置 JSON 与哈希、搜索空间哈希、代码提交、evaluator 与数据清单哈希、训练/验证情景哈希、种子序列、起止时间、墙钟时间、峰值内存、状态、全部验证指标、gate、stdout/stderr 和产物哈希，以及剩余次数和时间预算。

同时保存去除秘密后的代理动作轨迹 [M23-S5]。包锁、Python、macOS、硬件、线程数和非确定性标志也必须记录。任何 replay 都是验证运行；除非预先占用一个 trial，否则不能把 replay 结果反馈给策略。

### 6. 结果分析

验证阶段报告第 8 次后的最佳目标、1 至 8 次 best-so-far 曲线、归一化曲线面积、有效/重复/失败/超时/resource-blocked 数量、总 evaluator 时间及代理 token/工具成本。必须报告三个代理全部结果，不能只挑最佳代理。

三个策略重复不足以支持强总体结论。**项目推断：** 24 个自适应 trial 不能当作 24 个独立统计重复。情景 bootstrap 仅用于冻结模型在 128 个最终 holdout 情景上的配对误差，并对代理相对随机、网格和贝叶斯的三个预设对比执行 Holm 修正。

### 7. 证伪条件

以下情况使 M23 无效：搜索空间、数据、目标、种子、超时或 evaluator 不一致；只给代理补发失败 trial；允许代理修改 evaluator 却仍称为调参；搜索期间访问 `legacy_test` 或 holdout；向代理泄露基线结果；只报告三个代理中的最佳者；不等次数或不等 fidelity 却称等预算；holdout 后改变优胜者或对比；日志缺少失败、轨迹或来源；把自适应 trial 当成独立重复。

只有在冻结的代理选择相对三个基线在 holdout 上均达到至少 2% 实际改进，且修正后的配对 95% 区间排除 0 时，才支持代理优势。否则应报告 `no_advantage`、`inconclusive` 或 `resource_blocked`。

### 8. Apple M5 / 16 GB 可行性

96 次 evaluator 调用若严格受一小时限制，平均每次只有 37.5 秒。标准的 150,000/75,000 样本、30 epoch、CPU float64 训练很可能超过这一额度。必须先做不反馈给策略的定时校准。若保守 P90 超过 37.5 秒，只能由管理者批准更大预算，或在任何代理看到验证反馈前冻结一个更低成本且对所有策略完全一致的 M23 专用 benchmark。否则应诚实返回 `resource_blocked`，不能偷偷减少基线预算。

串行运行单个宽度 128 的 MLP 在 16 GB 内存上可行。禁止并行 evaluator，因为资源争用会破坏时间公平性。

### 9. 主要来源对应关系

主要来源表见英文第 9 节。M23-S1 与 M23-S2 定义随机和贝叶斯基线；M23-S3 与 M23-S4 支持选择偏差和 holdout 隔离；M23-S5 至 M23-S9 支持代理轨迹、公平 benchmark、HPO trial 结构与实验追踪。

### 10. 声明安全

允许表述：“在冻结的 M23 benchmark 和每重复 8 次验证评估下，代理策略相对随机、网格与贝叶斯搜索取得了某结果”；“配置在一次 holdout 评估前冻结”；“一小时本地预算导致资源阻塞”。

禁止表述：“Codex 普遍优于其他调参方法”；“验证集最优证明泛化”；没有访问控制证据却称 holdout 盲测；预算或 fidelity 不同却称公平；以及任何替代贝叶斯优化、人类研究者或数值求解器的声明。
