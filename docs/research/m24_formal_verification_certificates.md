# M24 Formal Verification and Certificates

## English

### 1. Problem definition

M24 asks whether a stored American-option surface satisfies a hierarchy of independently recomputed discrete certificates and whether those checks detect known corruptions without falsely rejecting valid surfaces. The primary object is the repository's finite-dimensional Crank-Nicolson linear complementarity problem (LCP), not the continuous obstacle problem.

At time step `n`, the repository solves

```text
A x >= b,
x >= phi,
(x-phi)'(A x-b) = 0,
```

where `phi` is the interior payoff vector. Define the gap and dual residual

```text
g = x - phi,
w = A x - b.
```

An exact discrete LCP solution satisfies `g>=0`, `w>=0`, and `g_i*w_i=0` for every node. The certificate must recompute `A`, `b`, boundary contributions, `g`, and `w` from the committed configuration and previous time slice. Reusing the solver's cached residual would not independently audit the surface.

The word “formal” is permitted only for a precisely specified finite arithmetic statement with machine-checkable evidence. Ordinary float64 thresholds are diagnostics. High precision reduces roundoff but is not automatically rigorous. Interval arithmetic with outward rounding can support rigorous enclosures, but an indeterminate enclosure must be reported as `indeterminate`, not pass or fail.

### 2. Directly relevant prior work

1. Brennan and Schwartz introduced a finite-difference treatment of the American put and its early-exercise feature [M24-S1]. Jaillet, Lamberton, and Lapeyre derive American-option variational inequalities and justify the Brennan-Schwartz algorithm under their assumptions [M24-S2].
2. Cryer analyzes systematic/projected overrelaxation for a positive-definite quadratic programming/complementarity setting [M24-S3]. This supports PSOR's mathematical lineage but does not certify every matrix produced by this repository.
3. Cottle, Pang, and Stone give the standard LCP formulation and matrix-class conditions for existence and uniqueness, including P-matrix results [M24-S4]. Feasibility and complementarity alone identify an LCP solution; uniqueness requires an additional matrix condition.
4. Barles and Souganidis prove a convergence framework for monotone, stable, consistent schemes under a comparison principle [M24-S5]. A residual certificate for one grid is not a substitute for these continuum-convergence hypotheses.
5. Rump explains verified numerical computation with floating-point arithmetic and rigorous enclosures [M24-S6]. IEEE 754-2019 defines floating-point arithmetic [M24-S7], while the active IEEE 1788.1-2017 simplified standard specifies binary64 interval arithmetic and decorations as a subset of IEEE 1788-2015 [M24-S8]. These sources distinguish ordinary numerical agreement from verified bounds.
6. Ekstrom proves convexity and selected monotonicity properties of American option prices under stated assumptions [M24-S9]. These are useful shape checks, but they do not uniquely characterize the option price.
7. Higham's 2023 Acta Numerica survey gives a current authoritative account of floating-point behavior and error analysis [M24-S10]. Repeating a float64 calculation in higher precision tests sensitivity; only a sound enclosure or proof obligation yields a rigorous certificate.

### 3. Certificate hierarchy

Every check returns one of `certified_pass`, `diagnostic_pass`, `fail`, `indeterminate`, or `not_applicable`, plus a margin and provenance record.

#### Tier 0: artifact and finiteness certificate

- Verify config, code commit, solver matrix convention, surface shape, spot/time grids, payoff, option type, and artifact hashes.
- Reject NaN, infinity, duplicate/nonordered grid coordinates, inconsistent dimensions, missing time slices, or a surface whose stored config hash does not match the supplied config.
- Verify `tau[0]=0` and monotone increasing time-to-maturity orientation.

This is a deterministic data-integrity certificate, not a financial proof.

#### Tier 1: float64 discrete LCP diagnostics

For every interior node and time step, independently recompute:

```text
primal_violation = max_i max(-g_i, 0)
dual_violation = max_i max(-w_i, 0)
natural_residual = ||min(g,w)||_infinity
product_residual = max_i |g_i*w_i|
```

The natural residual is required because a single global dot product can conceal the location and scale of nodewise failures. Record the worst node, time, local scale, and all four values. A float64 pass is `diagnostic_pass` only.

#### Tier 2: terminal, boundary, obstacle, and shape checks

- Terminal: compare the `tau=0` row with payoff at every node.
- Boundary: independently evaluate `time_boundary_values` at every `tau` and compare both endpoints.
- Obstacle: check all nodes, not only the interior LCP vector.
- Put spot monotonicity: values must be nonincreasing in `S` under the tested model assumptions.
- Convexity in spot: use nonuniform-grid slope differences if the grid is not uniform; do not apply raw second differences to unequal spacing.
- Time monotonicity: test value against time to maturity only for the declared model class and orientation.
- Exercise-set topology: for a one-dimensional put, flag noncontiguous exercise nodes, but label this a topology diagnostic rather than a standalone proof.
- Boundary curve: verify finite/status conventions and expected direction only where the source assumptions and numerical extraction are applicable.

Shape checks are necessary controls and corruption detectors. A smooth, monotone, convex surface can still violate the LCP or be far from the solution.

#### Tier 3: high-precision recomputation

Reconstruct decimal configuration inputs and matrix coefficients in at least 80 decimal digits. Recompute `A`, `b`, `g`, and `w` using the stored candidate values interpreted as exact binary64 numbers. Compare signs and margins with float64.

High-precision agreement can diagnose cancellation and tolerance sensitivity. It remains `diagnostic_pass`, because high precision computes a more accurate approximation but does not by itself enclose the exact mathematical result [M24-S6, M24-S10].

#### Tier 4: interval or active-set certificate

Use outward-rounded intervals conforming to the operational requirements of IEEE 1788 or an audited validated-numerics library [M24-S8]. Two possible certificate strengths are:

1. Candidate residual enclosure: enclose all coefficient, matrix-vector, gap, and residual operations. Certify relaxed inequalities `g>=-epsilon`, `w>=-epsilon`, and `|min(g,w)|<=epsilon` only when the full intervals lie inside those bounds.
2. Exact active-set LCP certificate: infer a candidate exercise set `E` and continuation set `C`; impose `x_E=phi_E` and `w_C=0`; compute an interval enclosure of the resulting linear system; verify `w_E>=0` and `g_C>=0`. If the active-set system is uniquely enclosed and the LCP matrix has a certified uniqueness condition, this can certify the exact discrete LCP solution for that time step.

Near a free boundary, both `g_i` and `w_i` may be close to zero, so the active set can be numerically ambiguous. Wide intervals then produce `indeterminate`. They must never be narrowed by unsound rounding or converted to a pass because float64 looks plausible.

#### Matrix-condition certificate

Do not assume the Crank-Nicolson matrix is an M-matrix. In the repository's central-difference operator,

```text
lower = diffusion/dS^2 - drift/(2*dS),
upper = diffusion/dS^2 + drift/(2*dS).
```

The sign pattern can change with drift, grid spacing, and location. **Project inference:** verify a sufficient mesh/sign condition and strict diagonal dominance with outward rounding when available. Otherwise use a certified P-matrix test only at small dimensions or make no uniqueness claim. A residual-feasible candidate without a matrix certificate is a discrete solution candidate, not a unique-solution proof [M24-S4].

### 4. Proposed experiment for this project

#### 4.1 Valid-surface panel

Create a predeclared validation-only panel of at least 24 complete put surfaces covering:

- low, central, and high `sigma`;
- low and high `r` and `q` within the campaign domain;
- short and long `T`;
- base and stress-edge regimes;
- at least two grid resolutions, including the canonical grid.

Generate these surfaces with the audited solver at a tighter PSOR stopping tolerance than the certificate threshold. Keep label generation separate from verification. Record solver convergence, grid, truncation boundary, Rannacher setting, matrix hashes, and complete values.

Thresholds must be derived before injecting corruptions. Use a scale-aware bound combining outward-rounding error, matrix/vector norms, and the configured solver tolerance. The hard campaign obstacle gate remains `1e-10`, but other residual gates require their own numerical budget.

#### 4.2 Deliberate corruption suite

For each valid surface, create deterministic corruptions with a committed corruption seed. Preserve the original and record exact modified indices and amplitudes.

| Corruption family | Intended detector |
|---|---|
| NaN, infinity, truncated row, permuted time slice | Tier 0 integrity |
| Single node below payoff | obstacle and primal LCP |
| Positive bump at an exercise node | complementarity/product/natural residual |
| Continuation-node perturbation | dual and natural residual |
| Endpoint perturbation | boundary certificate |
| Terminal-row perturbation | terminal certificate |
| Adjacent-node swap or local upward spike | put monotonicity |
| Local concave dent preserving monotonicity | convexity |
| Smooth global bias preserving shape | LCP/terminal/boundary, illustrating shape insufficiency |
| Boundary-band perturbation near degeneracy | interval indeterminacy and sensitivity |
| Stored config or matrix hash mismatch | provenance/integrity |

Use signed amplitudes at `0.25x`, `1x`, `4x`, and `16x` the predeclared local tolerance scale. This yields detection curves rather than a single easy corruption level. A corruption may target several checks; the expected detector set must be declared before execution.

#### 4.3 Detection and false positives

For each certificate family report:

```text
detection_rate = detected targeted corruptions / targeted corruptions
false_positive_rate = valid surfaces classified fail / valid surfaces
indeterminate_rate = interval-indeterminate cases / attempted interval certificates
```

Also report localization accuracy for the injected node/time, detection by amplitude, and overlap among detectors. Treat `indeterminate` separately from false positive and false negative. Give exact binomial confidence intervals over complete surfaces/corruptions, while noting that multiple corruptions derived from one source surface are dependent.

The primary gate should require zero false positives on the valid panel and detection of all `4x` and `16x` targeted corruptions. **Project inference:** this is an engineering acceptance rule, not a theorem from the sources. Sub-tolerance `0.25x` corruptions are sensitivity probes and need not be detected.

#### 4.4 Independent theorem controls

- European analytic control: disable the obstacle and verify the linear scheme against Black-Scholes at grid-refinement levels.
- No-dividend American-call control: compare with the European call under the classical assumptions.
- Exact small LCP controls: construct tiny matrices with known active sets and rational solutions, including degenerate nodes.
- Corrupted-certificate controls: feed a valid `x` with wrong `A`, wrong `b`, and wrong `phi` separately to ensure provenance prevents vacuous passes.

These controls test implementation. They do not turn empirical detection rates into a general proof.

### 5. Falsification conditions

M24 is invalid or downgraded if:

- the checker reuses solver residuals instead of independently rebuilding the step equation;
- it checks only `(g'w)` and omits primal, dual, natural, and nodewise product residuals;
- boundary contributions or time orientation are wrong;
- nonuniform grids are tested with unweighted second differences;
- NaN comparisons silently pass;
- float64 or high-precision agreement is labeled rigorous interval proof;
- intervals are not outward rounded or library/version/rounding mode is missing;
- `indeterminate` is counted as pass;
- LCP uniqueness is claimed without a certified matrix condition;
- a discrete-grid certificate is generalized to the continuous obstacle problem;
- corruption types or amplitudes are chosen after viewing detector failures;
- valid surfaces used to tune thresholds are then reused without disclosure for false-positive estimation;
- the holdout is used to design certificates or thresholds.

The experiment is also falsified as a detector benchmark if it reports only detection and omits false positives, localization, amplitude sensitivity, or indeterminate cases.

### 6. Compute feasibility on Apple M5 / 16 GB

Float64 checks are cheap: a `121 x 121` surface with 120 time steps and 119 interior nodes requires small tridiagonal or dense matrix-vector operations. Twenty-four surfaces and all deterministic corruptions should fit comfortably in memory and generally run in minutes if matrices are handled sparsely.

High-precision and interval active-set solves are the risk. Avoid materializing large object-dtype tensors for all surfaces. Stream one surface and time slice at a time, preserve tridiagonal structure, and serialize high-precision work. A staged plan is:

1. Tier 0-2 on every valid and corrupted surface.
2. Tier 3 on every valid surface and all corruptions that disagree with float64 classification.
3. Tier 4 on tiny theorem controls, all near-threshold cases, and a predeclared stratified subset of at least six full surfaces.
4. Expand Tier 4 only if a timed calibration predicts completion within the one-hour M24 screen.

Memory should remain below 2 GB with streaming. If a compliant interval backend is unavailable or the screen budget is exceeded, report `resource_blocked` or `interval_unavailable`; do not substitute ordinary `decimal` arithmetic and call it an interval proof.

### 7. Primary-source table

| ID | Primary source | DOI / official URL | Direct support used here |
|---|---|---|---|
| M24-S1 | Brennan, M. J. and Schwartz, E. S. (1977), *The Valuation of American Put Options*, Journal of Finance 32(2):449-462 | [DOI 10.1111/j.1540-6261.1977.tb03284.x](https://doi.org/10.1111/j.1540-6261.1977.tb03284.x) | Finite-difference valuation of the American put and early exercise. |
| M24-S2 | Jaillet, P., Lamberton, D., and Lapeyre, B. (1990), *Variational Inequalities and the Pricing of American Options*, Acta Applicandae Mathematicae 21:263-289 | [DOI 10.1007/BF00047211](https://doi.org/10.1007/BF00047211), [HAL](https://hal.science/hal-01667008) | Variational-inequality formulation and justification of the Brennan-Schwartz algorithm. |
| M24-S3 | Cryer, C. W. (1971), *The Solution of a Quadratic Programming Problem Using Systematic Overrelaxation*, SIAM Journal on Control 9(3):385-392 | [DOI 10.1137/0309028](https://doi.org/10.1137/0309028) | Projected/systematic overrelaxation under stated matrix assumptions. |
| M24-S4 | Cottle, R. W., Pang, J.-S., and Stone, R. E. (2009), *The Linear Complementarity Problem*, SIAM Classics in Applied Mathematics | [DOI 10.1137/1.9780898719000](https://doi.org/10.1137/1.9780898719000) | LCP conditions, existence, multiplicity, P-matrix and uniqueness theory. |
| M24-S5 | Barles, G. and Souganidis, P. E. (1991), *Convergence of Approximation Schemes for Fully Nonlinear Second Order Equations*, Asymptotic Analysis 4(3):271-283 | [DOI 10.3233/ASY-1991-4305](https://doi.org/10.3233/ASY-1991-4305) | Continuum convergence requires monotonicity, stability, consistency, and comparison assumptions. |
| M24-S6 | Rump, S. M. (2010), *Verification Methods: Rigorous Results Using Floating-Point Arithmetic*, Acta Numerica 19:287-449 | [DOI 10.1017/S096249291000005X](https://doi.org/10.1017/S096249291000005X) | Validated numerics, rigorous enclosures, and distinction from approximate computation. |
| M24-S7 | IEEE (2019), *IEEE Standard for Floating-Point Arithmetic*, IEEE 754-2019 | [Official standard](https://standards.ieee.org/ieee/754/6210/) | Floating-point formats and rounding behavior. |
| M24-S8 | IEEE (2017), *IEEE Standard for Interval Arithmetic (Simplified)*, IEEE 1788.1-2017 | [Active official standard](https://standards.ieee.org/ieee/1788.1/6074/) | Binary64 interval operations, decorations, and compatibility with the full IEEE 1788 framework. |
| M24-S9 | Ekstrom, E. (2004), *Properties of American Option Prices*, Stochastic Processes and their Applications 114(2):265-278 | [DOI 10.1016/j.spa.2004.05.002](https://doi.org/10.1016/j.spa.2004.05.002) | Convexity and selected monotonicity/continuity properties under stated assumptions. |
| M24-S10 | Higham, N. J. (2023), *Floating-Point Arithmetic*, Acta Numerica 32:203-290 | [DOI 10.1017/S0962492922000101](https://doi.org/10.1017/S0962492922000101) | Current authoritative floating-point error analysis and limitations of finite precision. |

### 8. Claim safety

Allowed wording:

- "The candidate satisfies the recomputed discrete LCP inequalities and complementarity residual within the stated float64 tolerance."
- "An outward-rounded interval calculation certified the stated finite-dimensional inequalities for the named grid and configuration."
- "The detector found [rate] of predeclared corruptions with [rate] false positives on the valid panel."
- "The interval checker was indeterminate near the free boundary."

Disallowed wording:

- "The American option surface is formally verified" without naming the finite statement, arithmetic, grid, and assumptions.
- "High precision proves correctness."
- "Passing shape checks proves the LCP."
- "The discrete certificate proves the continuum PDE/variational inequality solution."
- "The certificate proves uniqueness" without a matrix-class certificate.
- Any claim that certificates replace PSOR, grid refinement, analytic controls, or independent solver validation.

**Explicit project inferences:** the certificate tiers, status vocabulary, 24-surface panel, corruption suite/amplitudes, acceptance gates, and staged compute plan are proposed protocol choices, not direct claims from the sources.

---

## 中文

### 1. 问题定义

M24 检验已存储的美式期权曲面是否满足一组独立重算的离散证书，并测量这些检查对已知损坏的检出率和对有效曲面的误报率。主要对象是仓库中的有限维 Crank-Nicolson LCP，而不是连续障碍问题。

对每个时间步，定义 `g=x-phi` 和 `w=A x-b`。精确离散 LCP 解应满足 `g>=0`、`w>=0` 和每个节点 `g_i*w_i=0`。检查器必须根据已提交配置、上一时间片和边界贡献独立重建 `A`、`b`、`g`、`w`；复用求解器缓存残差不构成独立审计。

“形式化”只能用于明确有限算术命题与可机检证据。float64 阈值只是诊断；高精度能降低舍入误差，但不自动产生严格证明；外向舍入区间可以给出严格包围，而无法判定的区间必须报告 `indeterminate`。

### 2. 直接相关的既有工作

- Brennan-Schwartz 与 Jaillet-Lamberton-Lapeyre 提供美式看跌有限差分、变分不等式和算法正当性的经典基础 [M24-S1, M24-S2]。
- Cryer 给出特定矩阵假设下的投影/系统超松弛分析 [M24-S3]，但不能直接证明仓库所有矩阵都满足条件。
- Cottle、Pang 和 Stone 给出标准 LCP 与 P-matrix 唯一性理论 [M24-S4]。
- Barles-Souganidis 说明连续极限收敛还需要单调、稳定、一致和比较原理 [M24-S5]，单网格残差不能替代这些条件。
- Rump、IEEE 754、当前有效的 IEEE 1788.1 简化标准与 Higham 区分普通浮点计算、高精度诊断和严格区间验证 [M24-S6, M24-S7, M24-S8, M24-S10]。
- Ekstrom 支持在明确假设下执行现货凸性及部分单调性检查 [M24-S9]。

### 3. 证书层级

所有检查返回 `certified_pass`、`diagnostic_pass`、`fail`、`indeterminate` 或 `not_applicable`，并保存 margin 和来源。

- Tier 0：核验配置、提交、矩阵约定、网格、形状、hash、时间方向及有限值；NaN、无穷、缺行和不一致维度必须失败。
- Tier 1：逐节点独立计算 primal violation、dual violation、`||min(g,w)||_infinity` natural residual 和最大 `|g_i*w_i|`。只检查全局内积不充分。
- Tier 2：检查终端 payoff、每个时间的两个边界、全网格障碍、看跌价格对 `S` 的单调性、现货凸性、适用的期限单调性、执行集连通性和边界状态。非均匀网格必须比较斜率，不能直接使用原始二阶差分。
- Tier 3：以至少 80 位十进制精度重建系数和残差，用于识别抵消和阈值敏感性；仍只能称 `diagnostic_pass`。
- Tier 4：用符合 IEEE 1788 思路的外向舍入区间包围候选残差，或按候选 active set 令执行区 `x=phi`、延续区 `w=0`，区间求解线性系统并核验另一侧不等式。自由边界附近区间过宽时应返回 `indeterminate`。

不能默认 CN 矩阵是 M-matrix。中央差分的上下系数符号会随 drift、网格和位置变化。**项目推断：** 应在可用时用外向舍入核验充分符号/网格条件和严格对角占优，否则只在小维度做 certified P-matrix 检查，或完全不声明唯一性。

### 4. 本项目的建议实验

预先选择至少 24 个完整验证集看跌曲面，覆盖不同 `sigma`、`r`、`q`、`T`、域内压力边缘和至少两种网格分辨率。用比证书阈值更严格的 PSOR 容差生成，并完整记录收敛、边界、Rannacher 设置和矩阵哈希。

对每个有效曲面生成可追踪损坏：NaN/无穷/缺行/时间片置换、低于 payoff 的单点、执行点正 bump、延续点扰动、端点扰动、终端扰动、相邻点交换、保持单调但破坏凸性的凹 dent、保持形状的全局 bias、自由边界退化扰动，以及配置或矩阵 hash 不匹配。每类使用局部容差的 `0.25x,1x,4x,16x` 幅度，提前声明目标 detector。

分别报告检出率、有效曲面误报率、区间无法判定率、损坏节点定位率、不同幅度的检出曲线和 detector 重叠。`indeterminate` 不能并入通过、误报或漏报。**项目推断：** 工程 gate 可要求有效面板零误报，并检出全部 `4x` 和 `16x` 目标损坏；`0.25x` 只作为灵敏度探针。

另外执行欧洲解析解、无分红美式看涨等价、小型已知 active-set LCP、错误 `A/b/phi` 等实现控制。这些控制检验实现，不能把经验检出率提升为普遍数学证明。

### 5. 证伪条件

以下情况使 M24 无效或降级：复用求解器残差；只检查 `g'w`；边界贡献或时间方向错误；非均匀网格使用原始二阶差分；NaN 静默通过；把 float64 或高精度称为区间证明；区间没有外向舍入；把 `indeterminate` 算作通过；没有矩阵条件却声明唯一性；把离散证书推广为连续 PDE 证明；看结果后才选择损坏或幅度；用同一有效曲面调阈值又不披露地估计误报；或使用 holdout 设计证书。

若只报告检出率而不报告误报、定位、幅度敏感性和无法判定案例，检测 benchmark 也不成立。

### 6. Apple M5 / 16 GB 可行性

对 `121 x 121` 曲面执行 Tier 0-2 很便宜，24 个有效曲面及其确定性损坏可以流式处理。高精度和区间 active-set 求解是主要风险。应逐曲面逐时间片处理并保留三对角结构，避免为全部曲面创建 object-dtype 大张量。

建议先对所有曲面执行 Tier 0-2；再对全部有效曲面及与 float64 分类冲突的损坏执行 Tier 3；Tier 4 先覆盖小型 theorem control、所有近阈值案例和至少 6 个预先分层选择的完整曲面。只有定时校准显示可在一小时内完成时才扩展。流式实现峰值内存应低于 2 GB。若没有合规区间后端，应报告 `interval_unavailable`，不能用普通 `decimal` 冒充区间证明。

### 7. 主要来源对应关系

主要来源表见英文第 7 节。M24-S1 至 M24-S5 支持美式期权离散/变分/LCP 和连续收敛边界；M24-S6 至 M24-S8、M24-S10 支持可靠浮点与区间验证；M24-S9 支持有限范围的形状检查。没有任何一个来源支持“通过若干数值检查即证明连续美式期权问题”。

### 8. 声明安全

允许表述：“候选面在指定 float64 容差内满足独立重算的离散 LCP 不等式和互补残差”；“外向舍入区间计算对指定网格与配置证明了明确的有限维不等式”；“检测器对预设损坏取得某检出率和误报率”；“区间检查在自由边界附近无法判定”。

禁止表述：不说明有限命题、算术、网格与假设却称“美式期权曲面被形式化验证”；“高精度证明正确”；“形状检查证明 LCP”；“离散证书证明连续变分不等式”；没有矩阵证书却称唯一；以及证书替代 PSOR、网格加密、解析控制或独立求解器验证的声明。
