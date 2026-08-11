# M22 Mechanistic Interaction Audit

## English

### 1. Problem definition

M22 asks a narrow question: for a frozen seven-input option-price surrogate, which input subsets are required to reproduce the model's output at selected points under one explicitly defined masking intervention, and are those subset effects stable across independently trained seeds and compatible with separately tested financial invariances?

The repository's input vector is

```text
x = (log_moneyness, tau_over_T, r, q, sigma, T, is_call).
```

For a point `x`, a baseline `b`, and feature index set `N={1,...,7}`, define the coalition value

```text
v_x(S) = f(x_S, b_{N\S}) - f(b),  S subseteq N.
```

The exact Mobius/Harsanyi coefficient for a nonempty subset `S` is

```text
I_x(S) = sum_{T subseteq S} (-1)^(|S|-|T|) v_x(T).
```

The inversion identity gives

```text
f(x) - f(b) = sum_{empty != S subseteq N} I_x(S).
```

This is an exact decomposition of the 128 coalition values induced by `(f,x,b,masker)` when all `2^7` model evaluations are made. Owen's multilinear extension and later Shapley-Taylor work give the relevant set-function basis; Kang et al. explicitly connect Mobius coefficients to unique scores for input subsets and higher-order interactions [M22-S1, M22-S2, M22-S3].

The target output must be fixed before evaluation. The recommended primary target is `price_over_K`; `premium_over_K` is a separately reported sensitivity analysis because payoff subtraction changes the interaction allocation. No coefficient may be described without naming the output, point, baseline, masking rule, and seed.

### 2. Directly relevant prior work

1. Owen's multilinear extension represents a cooperative set function on the unit hypercube and is the algebraic foundation for coalition-based decompositions [M22-S1].
2. Shapley-Taylor attributes effects to subsets up to a chosen order and relates those terms to derivatives of the multilinear extension. It is a different allocation rule from reporting every raw Mobius/Harsanyi coefficient [M22-S2]. M22 should compute the full Mobius table first and derive any Shapley-Taylor summaries second.
3. Kang et al. treat Mobius coefficients as unique importance scores for sets of variables and state that the full transform has `2^n` coefficients, while their sparse algorithms address larger `n` [M22-S3]. With `n=7`, exhaustive evaluation is simpler and avoids approximation error.
4. Herren and Hahn connect SHAP estimation to functional ANOVA and emphasize dependence on the feature distribution and the `2^p` family of terms [M22-S4]. This supports treating the baseline/masking distribution as part of the estimand, not as a neutral implementation detail.
5. Adebayo et al. show that explanation methods need model- and data-randomization checks; visual plausibility alone is inadequate [M22-S5]. Their tests motivate negative controls here, but do not prove this project's decomposition is mechanistic.
6. Merton and Ekstrom establish option-price restrictions and comparative properties relevant to independent metamorphic tests, including the no-dividend American-call result in the classical model and convexity/monotonicity properties under stated assumptions [M22-S6, M22-S7].

### 3. What exact `2^7` evaluation can and cannot establish

It can establish, up to recorded floating-point tolerance:

- every value of the finite coalition game induced by the chosen baseline and replacement masker;
- every Mobius coefficient from order one through order seven;
- exact reconstruction of `f(x)-f(b)` for that finite game;
- whether a feature is a dummy for that point and intervention, meaning all involving coefficients are numerically zero;
- how signed and absolute interaction mass is distributed by order for that point.

It cannot establish:

- a causal mechanism inside the network;
- a unique decomposition of the continuous model independent of baseline or masker;
- that hybrid masked inputs follow the training distribution or represent likely financial regimes;
- global behavior away from the audited points;
- stability across training seeds, unless separately measured;
- correctness of the model or compliance with financial invariances;
- a symbolic proof about the Black-Scholes obstacle problem;
- that a large coefficient is economically important rather than a consequence of output scale or baseline choice.

In particular, a put-only campaign keeps `is_call=0`. If both `x` and `b` are puts, `is_call` is a dummy and the nominal 128 evaluations contain duplicate pairs. This is a useful control, not evidence that option type is irrelevant. A matched put/call sensitivity analysis may flip `is_call`, but must be labeled outside the common put benchmark.

The representation already uses `log(S/K)` and normalized outputs such as `V/K`. Positive homogeneity in `(S,K,V)` is therefore largely built into the data representation. **Project inference:** observing no strike interaction cannot show that the network learned homogeneity because strike is not an independent input. Any scale test must reconstruct dimensional values for several synthetic `K` values and verify `V(lambda S, lambda K)=lambda V(S,K)` at the pipeline level.

### 4. Proposed experiment for this project

#### 4.1 Freeze the estimand

- Audit a frozen M10-style selected surrogate without retraining or changing weights.
- Use CPU float64 for canonical evaluation.
- Record model artifact hash, config hash, code commit, feature order, output name, scaler parameters, masker version, and all seeds.
- Define a domain-valid reference baseline from a predeclared validation regime medoid. Add two sensitivity baselines: a second medoid and a domain-valid financial reference point. Never use an all-zero vector unless every transformed coordinate is documented as valid.
- Apply replacement masking only in transformed feature space, then validate every hybrid against training-domain bounds and physical constraints. Mark invalid hybrids and do not silently clip them.

#### 4.2 Select representative points without holdout access

Select points from validation regimes only, before inspecting interaction values. Use a balanced panel of 48 points:

- 12 exercise-region points;
- 12 continuation-region points;
- 12 points near the PSOR free boundary;
- 12 validation stress-edge points within the declared domain.

Balance `tau_over_T`, `sigma`, and moneyness strata. Keep complete regime identifiers so uncertainty is summarized by regime rather than by correlated grid node. Use the same points and baselines for campaign seeds `101, 203, 307, 409, 503`.

#### 4.3 Compute the exact transform

For each `(seed, point, baseline, output)`:

1. Build all 128 bit masks in lexicographic order.
2. Batch-evaluate the frozen model once per mask set.
3. Compute the Mobius transform with an in-place fast zeta/Mobius transform in float64.
4. Record every `I_x(S)`, subset bit mask, subset names, order, signed value, absolute value, and normalized value

```text
I_norm(S) = I_x(S) / max(|f(x)-f(b)|, epsilon_scale).
```

5. Verify reconstruction against direct `f(x)-f(b)` with absolute and relative error. Use a predeclared tolerance such as `64 * eps_float64 * sum_S |I_x(S)|`, with an absolute floor derived from output scale, instead of an unexplained decimal threshold.
6. Aggregate signed and absolute mass by order:

```text
signed_mass_k = sum_{|S|=k} I_x(S)
absolute_mass_k = sum_{|S|=k} |I_x(S)|.
```

Shapley-Taylor order-`k` summaries may be derived and labeled as a second estimand; they must not replace the full coefficient table.

#### 4.4 Stability analysis across seeds

At each point and baseline, report:

- pairwise Spearman correlation of signed coefficients and of absolute coefficients;
- sign agreement for coefficients above a predeclared practical magnitude floor;
- Jaccard overlap of top-10 subsets by absolute magnitude;
- median and maximum across-seed coefficient range;
- across-seed dispersion of `absolute_mass_k` for every order;
- the fraction of points at which a named subset remains in the top 10.

Summarize first by complete regime and then across regimes. Do not treat the 48 grid points as independent replicates. **Project inference:** a reasonable screening gate is median signed-rank correlation at least 0.80 and at least 80% sign agreement for practically nonzero dominant terms; these values are proposed engineering gates, not thresholds supplied by the cited literature.

#### 4.5 Negative and positive controls

- Additive synthetic oracle: all coefficients of order two or higher must be zero within transform tolerance.
- Pure product oracle: only the analytically expected high-order coefficient pattern may survive for the chosen baseline.
- Dummy-feature control: put-only `is_call` coefficients must be zero.
- Model randomization control following the logic of Adebayo et al.: progressively randomize trained layers and require the interaction table to change materially.
- Label-randomization control, only if training cost permits: a model trained on permuted labels must not reproduce the audited financial pattern.

#### 4.6 Financial invariance and shape tests

These are independent metamorphic checks. Passing them does not follow from a plausible interaction table.

- Terminal condition: at `tau_over_T=0`, compare `price_over_K` with the normalized payoff.
- Obstacle: require predicted American value to dominate payoff within the campaign tolerance.
- Spot monotonicity and convexity: evaluate ordered spot grids at fixed `(tau,r,q,sigma,T,is_call)` and compute first- and second-difference violations, excluding only predeclared numerical edge handling.
- Volatility and time monotonicity: compare matched valid regimes under the assumptions supported by Ekstrom [M22-S7].
- No-dividend call control: outside the put-only primary audit, compare the American-call pipeline with the analytic European call under `q=0`, as justified in the classical model [M22-S6].
- Scale homogeneity: reconstruct dimensional prices at matched `(S,K)` scalings; label this a pipeline identity check because normalization partially enforces it by construction.

Cross-reference interaction findings with these checks only as associations. For example, a stable `moneyness x sigma` coefficient plus successful monotonicity tests does not prove that the coefficient causes convexity.

### 5. Falsification conditions

M22 is falsified or downgraded to `invalid` if any of the following occurs:

- the 128 coalition outputs cannot reconstruct the direct output within the declared numerical tolerance;
- feature order, scaler, baseline, masker, or model hash is missing;
- any masked hybrid is out of domain and is clipped or substituted without being counted;
- higher-order coefficients survive the additive control above tolerance;
- dummy `is_call` interactions appear in a put-only audit;
- the transform does not change under model/data randomization controls;
- dominant interactions fail the predeclared seed-stability gate;
- purported financial consistency is inferred from coefficients without direct metamorphic tests;
- point-level results are pooled as independent observations despite shared regimes;
- the blind holdout is accessed during point, baseline, output, threshold, or narrative selection.

A negative result is valid evidence. The decision should be `diagnostic_only` if the transform is exact but seed-unstable, baseline-sensitive, or financially inconsistent.

### 6. Compute feasibility on Apple M5 / 16 GB

The exact transform is inexpensive. The primary panel requires

```text
48 points * 3 baselines * 2 outputs * 5 seeds * 128 masks = 184,320 forward evaluations.
```

These are tiny batched MLP evaluations. The coalition tensor is far below 1 MB per batch, and the full long-form coefficient table is only a few hundred thousand rows. CPU float64 should finish in minutes; peak memory should remain well below 1 GB beyond loaded models and dataset arrays. Retraining five models or label-randomized controls dominates cost and must reuse existing campaign artifacts when available. Heavy runs remain serialized.

### 7. Primary-source table

| ID | Primary source | DOI / arXiv / official URL | Direct support used here |
|---|---|---|---|
| M22-S1 | Owen, G. (1972), *Multilinear Extensions of Games*, Management Science 18(5-part-2), P64-P79 | [DOI 10.1287/mnsc.18.5.64](https://doi.org/10.1287/mnsc.18.5.64) | Multilinear extension of a cooperative set function. |
| M22-S2 | Sundararajan, M., Dhamdhere, K., and Agarwal, A. (2020), *The Shapley Taylor Interaction Index*, ICML, PMLR 119:9259-9268 | [PMLR](https://proceedings.mlr.press/v119/sundararajan20a.html) | Higher-order subset attribution and relation to the multilinear extension. |
| M22-S3 | Kang, J. S. et al. (2024), *Learning to Understand: Identifying Interactions via the Mobius Transform*, NeurIPS 37 | [DOI 10.52202/079017-1467](https://doi.org/10.52202/079017-1467) | Mobius coefficients as subset scores, higher-order interactions, and `2^n` full-transform size. |
| M22-S4 | Herren, A. and Hahn, P. R. (2022), *Statistical Aspects of SHAP: Functional ANOVA for Model Interpretation* | [arXiv:2208.09970](https://arxiv.org/abs/2208.09970) | Dependence of SHAP/fANOVA interpretation on distributional choices and all-subset terms. |
| M22-S5 | Adebayo, J. et al. (2018), *Sanity Checks for Saliency Maps*, NeurIPS 31 | [NeurIPS](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html) | Model- and data-randomization checks as necessary explanation diagnostics. |
| M22-S6 | Merton, R. C. (1973), *Theory of Rational Option Pricing*, Bell Journal of Economics and Management Science 4(1), 141-183 | [DOI 10.2307/3003143](https://doi.org/10.2307/3003143) | Classical rational option restrictions, including early-exercise implications. |
| M22-S7 | Ekstrom, E. (2004), *Properties of American Option Prices*, Stochastic Processes and their Applications 114(2), 265-278 | [DOI 10.1016/j.spa.2004.05.002](https://doi.org/10.1016/j.spa.2004.05.002) | Convexity in spot and monotonicity/continuity in volatility and time decay under stated assumptions. |

### 8. Claim safety

Allowed wording:

- "The exhaustive seven-feature coalition audit exactly reconstructs the masked set function at the audited points within float64 tolerance."
- "The reported interactions are stable/unstable across the five tested training seeds under the named baseline and masker."
- "Separate metamorphic tests found consistency/inconsistency with the tested financial restrictions."

Disallowed wording:

- "The network's true mechanism was recovered."
- "The interactions are causal, unique, or globally valid."
- "Exact `2^7` evaluation proves the surrogate is financially correct."
- "The model learned scale invariance" when scale normalization imposed it.
- Any solver-replacement or mathematical-proof claim.

**Explicit project inferences:** the 48-point panel, stability thresholds, baseline choices, normalization formula, and compute estimates are proposed protocol choices, not conclusions stated in the cited sources.

---

## 中文

### 1. 问题定义

M22 的目标应限定为：对一个已冻结的七输入期权价格代理模型，在明确指定的基线与遮蔽干预下，哪些输入子集能够重构模型在代表点上的输出；这些子集效应是否在独立训练种子之间稳定；它们是否与另外执行的金融不变性检验相容。

仓库中的七个输入为：`log_moneyness`、`tau_over_T`、`r`、`q`、`sigma`、`T` 和 `is_call`。对样本 `x`、基线 `b` 与特征子集 `S`，先定义有限集合函数 `v_x(S)=f(x_S,b_{N\S})-f(b)`，再通过 Mobius/Harsanyi 反演计算所有阶的 `I_x(S)`。完整的 128 次评估可以在浮点容差内精确重构这个有限集合函数 [M22-S1, M22-S2, M22-S3]。

这里的“精确”只属于给定的模型、样本、基线、遮蔽器和输出。主要输出应预先固定为 `price_over_K`；`premium_over_K` 只能作为单独的敏感性分析，因为扣除 payoff 会改变交互效应的分配。

### 2. 直接相关的既有工作

- Owen 的多线性扩展提供集合函数分解的代数基础 [M22-S1]。
- Shapley-Taylor 把贡献分配给不同阶的特征子集，并与集合函数的多线性扩展相关联 [M22-S2]。本项目应先保存完整 Mobius 系数，再派生 Shapley-Taylor 摘要。
- Kang 等人明确讨论了 Mobius 系数、高阶交互以及完整变换的 `2^n` 规模 [M22-S3]。当 `n=7` 时，穷举比近似算法更透明。
- Herren 和 Hahn 指出 SHAP/fANOVA 依赖特征分布与全子集项 [M22-S4]，因此基线和遮蔽分布是估计对象的一部分。
- Adebayo 等人的模型随机化和数据随机化检验说明，仅凭可视化合理性不足以证明解释有效 [M22-S5]。
- Merton 与 Ekstrom 给出了可用于独立变形测试的经典期权限制、凸性和单调性结果 [M22-S6, M22-S7]。

### 3. `2^7` 能证明与不能证明的内容

它能够证明：128 个联盟值已被完整计算；所有一阶至七阶 Mobius 系数均可恢复；这些系数能够重构 `f(x)-f(b)`；给定干预下的 dummy 特征可以被识别；每一阶的有符号和绝对交互质量可以被统计。

它不能证明：网络内部的因果机制；与基线无关的连续函数唯一分解；遮蔽混合样本位于真实数据流形；全局金融正确性；跨种子稳定性；Black-Scholes 障碍问题的符号证明；或大系数必然具有经济重要性。

在只审计美式看跌期权时，`is_call` 对样本与基线都为 0，因此它应当是 dummy 特征，名义上的 128 次评估会出现成对重复。这是正向控制，不代表期权类型不重要。另一个关键限制是，输入已经使用 `log(S/K)`，输出已经除以 `K`。**项目推断：** 比例齐次性在很大程度上由表示方式预置，不能把“没有执行价交互”表述成模型学会了齐次性。

### 4. 本项目的建议实验

1. 冻结模型、配置、提交、特征顺序、标准化器、遮蔽器和输出，并在 CPU float64 下评估。
2. 从验证集完整情景中预先选择 48 个点：执行区、延续区、自由边界附近和域内压力边缘各 12 个。保持情景标识，不把网格点当成独立样本。
3. 使用一个验证情景 medoid 作为主基线，并增加两个预先声明的有效基线做敏感性分析。禁止未经验证的全零基线和静默裁剪。
4. 对五个种子 `101,203,307,409,503` 使用完全相同的点、基线和 128 个掩码。
5. 保存每个子集的位掩码、名称、阶数、有符号值、绝对值和按 `|f(x)-f(b)|` 归一化的值；验证 Mobius 重构误差。
6. 报告跨种子的 Spearman 相关、主导项符号一致率、top-10 Jaccard、系数范围、各阶绝对质量及主导子集出现频率。
7. 添加加性函数、纯乘积函数、put-only `is_call`、模型随机化和条件允许时的标签随机化控制。
8. 独立执行终端条件、障碍条件、价格对现货的单调性与凸性、对波动率和期限的适用单调性、无分红看涨控制以及维度价格的比例齐次性测试。

**项目推断：** 中位有符号秩相关至少 0.80、实际非零主导项符号一致率至少 80%，可作为预先声明的工程筛选阈值；这些数值不是来源论文给出的定理阈值。

### 5. 证伪条件

出现以下任一情况时，M22 应判为 `invalid` 或降级为 `diagnostic_only`：完整联盟值不能在声明容差内重构直接输出；缺失模型或遮蔽元数据；非法混合输入被静默裁剪；加性控制出现显著高阶项；put-only 审计出现 `is_call` 交互；随机化模型后解释几乎不变；主导交互跨种子不稳定；仅凭交互系数宣称金融正确；把同一情景内网格点当成独立重复；或在选择点、基线、阈值与叙事时访问盲测 holdout。

### 6. Apple M5 / 16 GB 可行性

主面板总计 184,320 次小型 MLP 前向评估，可批处理完成。系数长表规模仅数十万行，CPU float64 预计数分钟完成，额外峰值内存远低于 1 GB。真正昂贵的是重新训练五个模型或标签随机化控制，因此应优先复用已提交的 campaign 模型产物，并串行执行训练。

### 7. 主要来源对应关系

主要来源表见英文第 7 节。M22-S1 至 M22-S4 支持集合函数、Mobius 和高阶交互的数学定义；M22-S5 支持随机化 sanity check；M22-S6 与 M22-S7 支持另行执行的金融限制测试。它们都不支持把遮蔽分解称为网络内部因果机制。

### 8. 声明安全

允许表述：“穷举七特征联盟审计在浮点容差内精确重构了指定遮蔽集合函数”；“所报告交互在五个测试种子之间稳定或不稳定”；“独立变形测试与指定金融限制相容或不相容”。

禁止表述：“恢复了网络真实机制”；“交互具有因果性、唯一性或全局有效性”；“`2^7` 证明代理模型金融正确”；“模型学会了由归一化预置的齐次性”；以及任何替代 PSOR 或连续问题数学证明的声明。
