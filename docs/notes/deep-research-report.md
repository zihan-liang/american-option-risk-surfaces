# 面向美式期权的自由边界感知风险曲面研究综述

## 执行摘要

- 经典文献非常明确地支持把美式期权定价写成**最优停止问题、变分不等式、障碍问题**，并在离散化后写成**线性互补问题**；因此，你当前以 VI/LCP 语言组织 CN/PSOR 基准与诊断，是站得住脚且与主流数值金融文献一致的。citeturn23view0turn26view0turn29view0
- 对于一维 Black–Scholes 美式 put，**Crank–Nicolson + PSOR** 仍然是完全合理的可复现实验基准：CN 是最常见的时间推进方案之一，PSOR 是求解离散 LCP 的经典方法；但文献同样强调，这并不是“新求解器”，而是一个可验证、可审计的基线。citeturn29view0turn24search1turn25search8
- 你的“**障碍违背**、**互补残差**”诊断不是附加装饰，而是离散 VI/LCP 是否被正确满足的核心检查项；从理论上讲，这些条件正是问题定义本身的一部分。citeturn26view0turn29view0turn23view0
- 文献强烈支持你对**非光滑初值**保持谨慎：CN 在 payoff kink 附近容易出现误差与振荡，Rannacher smoothing 或其它阻尼/稳定时间推进是标准补救；若还要报告 Greeks，尤其是 Gamma，这一点更重要。citeturn26view2turn26view3turn29view0
- 对**早行权边界**而言，经典结果支持“把边界当成单独对象来提取和研究”：美式 put 的边界可被积分方程、渐近展开或数值方法单独刻画；而带股息美式 call 也会出现真实自由边界。这个方向与“单独做 boundary diagnostic head”高度一致。citeturn23view2turn32view1turn31search3turn23view2
- 对**Greeks**，文献的总体态度是：可以从 PDE/FD 价格面得到 Delta 和 Gamma，但**Gamma 在自由边界与 payoff kink 附近尤其脆弱**；因此，你目前采用 mask、局部谨慎解释、以及单独验证 Greeks 的做法，比“直接把价格网络 autograd 当作可靠 Greek engine”更稳妥。citeturn29view0turn26view2turn28search13
- 对神经网络代理模型，文献支持“**加速**”与“**近似**”，但并不支持“ML 自动满足障碍、单调性、凸性和无套利结构”；恰恰相反，若没有结构化约束，往往需要额外机制来保证经济一致性。你对“低 RMSE 但仍可违规”的基线结论，与该方向总体认识一致。citeturn18search0turn18search6turn23view4turn23view3turn33search6
- 关于你下一步想做的**正 continuation-premium 参数化**，文献给出的是**间接而非直接**支持：早行权 premium/optimal stopping/障碍表示说明“在障碍之上加一个非负部分”是自然的工程参数化，但目前并没有成熟共识表明这已经是美式期权 ML 的标准答案。因此，这一主张应写成“受经典分解启发的合理设计”，而不是“已有文献证明的最佳架构”。citeturn23view5turn32view1turn20search7
- 关于**边界头与 Delta 头分离**，现有证据更支持“模块化更可解释、更可审计”，而不是“一网通吃”：深度最优停止文献主要学停止规则/继续值，微分机器学习文献则表明若希望可靠风险量，往往需要显式使用导数标签或微分正则化。citeturn18search7turn12search16turn37view1turn37view0
- 在实验设计上，文献越来越强调：**RMSE 不是充分指标**，尤其在非光滑或 OOD regime；同时，错误的数据划分会引入泄漏并夸大泛化能力。因此，你强调 regime-level split、失败样本上报、配置清单、诊断表和 claim audit，属于强文献支持的“研究工程”定位。citeturn36view2turn17view1turn35view3turn13search0turn13search1turn13search18

## American Option Formulation and Free Boundaries

美式期权的经典数学表述来自**最优停止**：价值函数等于在所有允许停止时刻上折现收益的上确界。转到 PDE 语言后，这个问题自然变成一个**障碍型自由边界问题**：在继续持有区满足 Black–Scholes 型 PDE，在行权区贴住 payoff，二者交界就是早行权边界。Jaillet、Lamberton 与 Lapeyre 的经典论文直接把美式期权与**变分不等式方法**联系起来，并讨论了相应数值方法；Peskir 则把美式 put 的最优停止边界刻画为一个由早行权 premium 表示导出的**非线性积分方程**的唯一解。citeturn23view0turn32view1

从工程实现角度，你把连续问题表述为 VI/obstacle problem，再把离散问题写成 LCP，是标准路径。Huang 与 Pang 的综述明确指出，许多美式期权定价模型都可以写成**涉及偏微分算子的 LCP/PDCP**；这正是你把“障碍满足”“互补残差”“自由边界提取”纳入统一诊断框架的理论来源。citeturn26view0

对**美式 put**，文献不但给出存在唯一性，而且给出边界形状信息。Chen 与 Chadam 证明了早行权边界问题的存在唯一性，并给出了**到期附近的高阶渐近展开**；后续工作还证明了零股息资产上美式 put 边界的**凸性**。因此，在你的项目里，把 boundary extraction 作为一个值得单独报告的对象，不仅合理，而且很符合该问题的经典研究传统。citeturn32view2turn32view0

对**美式 call**，文献必须分两种情形来写。无股息时，经典无套利结果表明提前行权通常不优，因此美式 call 与对应欧式 call 等价；这也是最标准的验证测试之一。带连续股息或离散股息时，这一结论失效，call 会出现真实自由边界。Evans、Kuske 与 Keller 在“资产带股息、临近到期”情形下同时给出了美式 put 与美式 call 的价格和边界近似；Cosma 等的实证/理论论文也再次强调了**股息建模对 call 边界至关重要**。citeturn22search12turn22search6turn31search3turn5search3

对你的论文写作而言，这一部分最安全的表述是：**“本项目采用文献中标准的 VI/LCP/obstacle/free-boundary 语言来组织美式期权数值基准与机器学习诊断。”** 不宜写成“本文提出了一种新的美式期权数学表述”，因为该表述已有非常成熟的先行文献。citeturn23view0turn26view0

## CN/PSOR and Finite-Difference Benchmarks

在 Black–Scholes 框架下，**Crank–Nicolson** 是最常用的有限差分时间推进之一；对美式期权，空间离散以后常得到 LCP，再用 **PSOR** 等投影迭代求解。Shi 等在介绍新迭代法时，专门把“PSOR 已被广泛用于美式期权 LCP”作为背景事实；Ikonen 与 Toivanen 也把 projected SOR 描述为经典且广泛使用的方法。对于你的项目，这足以支持“CN/PSOR 是适当 benchmark”，同时也提醒你不要把它包装成方法学创新。citeturn24search1turn25search8

Jaillet 等的经典论文还给出了一个很重要的历史连接：他们基于变分不等式方法，**完整论证了 Brennan–Schwartz 美式 put 估值算法**。这意味着，把 CN/PSOR 或同类离散 LCP 求解器作为一维美式 put 的验证基线，不只是工程上可行，而且有相当深的理论与历史根基。citeturn23view0

关于**边界条件与域截断**，Kangro 与 Nicolaides 明确指出：为了数值求解 Black–Scholes PDE，必须引入人工边界使计算域有界；他们还推导了不同远场边界条件导致的误差上界，并说明这些估计可用于**预先选择合适的截断边界位置**。这与“domain sensitivity study”高度一致：边界位置不是无关紧要的实现细节，而应该被系统汇报。citeturn27view0

关于**非光滑 payoff 初值**，Pooley、Vetzal 与 Forsyth 指出，payoff 本身或其导数的不连续，会使数值格式出现明显误差，尤其会污染**hedging parameters**；他们讨论了初值平均、网格平移、投影法等补救，并强调若不配合特殊时间推进，这些单独都不足以恢复预期精度。Giles 与 Carter 进一步从收敛分析角度说明，Rannacher 的启动程序——先用若干步 backward Euler，再切换到 CN——对一阶、二阶导数逼近的二阶收敛很关键。citeturn26view2turn26view3

这直接支持你项目中已经做的几项 benchmark 规范：  
其一，用欧式期权闭式解做 sanity check；其二，用“无股息美式 call = 欧式 call”做模型与实现一致性检查；其三，把 obstacle 与 complementarity residual 作为离散求解器是否真的在解目标问题的验收条件。前两项来自经典 Black–Scholes/Merton 结构，第三项来自 VI/LCP 问题本身。citeturn34view1turn22search12turn26view0

因此，CN/PSOR 在你的项目里最合理的定位不是“发明新求解器”，而是：**一个可验证、可复现、可诊断的 classical numerical benchmark，用于为下游 ML 标签提供可信基准。** 这个定位与现代 surrogate 文献的“solver emulator”思路也一致。citeturn18search6turn34view1

## Diagnostics: Obstacle, Complementarity, Shape, Grid, Domain, Runtime

如果你的连续问题是障碍问题/VI，而离散问题是 LCP，那么**障碍满足**与**互补残差**就是问题定义的数值化版本。Huang 与 Pang 强调，American option 的离散 PDCP/LCP 需要依赖对离散互补结构的理解来保证有限差分与 LCP 数值求解的收敛与稳定；这说明，把 obstacle violation 和 complementarity error 摆在诊断中心，并不是“额外加分项”，而是最基本的数值正确性检查。citeturn26view0

对**shape diagnostics**，经典无套利与期权曲面文献虽然更多聚焦欧式/市场曲面，但它们已经清楚说明：单调性、凸性、正值性这类结构不应被视为可有可无的“后处理美观条件”。Dugas、Yang、Cao 等工作之所以构造 shape-restricted 或 no-arbitrage NN，正是因为普通神经网络并不会自动保留这些结构。把这一思想迁移到你基于 PSOR 标签的美式期权 surrogate setting，是非常自然的。citeturn23view4turn33search6turn23view3

对**grid convergence** 与 **domain sensitivity**，数值分析文献给出的结论很直接：网格尺度、时间步长、远场截断都会系统影响误差，且这些误差可通过理论估计或经验收敛研究暴露出来。Kangro–Nicolaides 提供了人工边界误差控制的理论基础；Giles–Carter 和 Pooley–Vetzal–Forsyth 则说明，即使在固定 PDE/模型下，非光滑特征也会让“名义上的高阶格式”表现变差，尤其影响导数。citeturn27view0turn26view2turn26view3

对**runtime**，PDEBench 的一个很有价值的信息是：在科学机器学习里，单一 RMSE 并不足以评价方法，尤其在“非光滑”“湍流”这类复杂 regime 下，RMSE 可能看不见重要局部结构误差。虽然 PDEBench 不是期权定价论文，但它对“benchmark 不能只报一个均方根误差”的提醒，与你项目中强调 runtime、grid、domain、boundary、Greeks、shape 等多维诊断非常一致。citeturn36view2

从**可复现研究**角度看，你的 repository 设计方向也非常符合通行规范。Sandve 等强调保存中间结果、记录每一步如何产生结果；Wilson 等强调自动化工作流、版本化与可读代码；Bailey 等专门指出计算研究应像实验记录那样保留**负结果与正结果**，而不是只展示成功案例。对你的项目而言，这直接支持“failed or delicate regimes must be reported”。citeturn13search0turn13search1turn13search18

因此，这一部分最值得在最终论文中明确写出来的是：**本项目的贡献首先是“diagnostic-aware benchmark curation”，而不是“又做了一个价格回归器”。** 这一措辞既准确，也能与你希望的“研究工程”定位保持一致。citeturn17view2turn13search0turn13search1

## Boundary Extraction and Greeks

把**早行权边界**从数值价格面中显式提取出来，在文献里是成熟而重要的工作，而不是附属图形。AitSahlia 与 Lai 直接研究求解定义边界的积分方程，并指出标准美式期权的边界可以用少量时间节点上的**线性样条**很好近似，同时还能据此计算价格和 hedge parameters。对于你的项目，这正能支撑“boundary extraction 应该被单独报告”，而不是只把价格热图当最终结果。citeturn23view2

对**美式 put 边界行为**，Chen–Chadam 的存在唯一性与高阶渐近展开，Kuske–Keller 的临近到期分析，以及 Chen 等关于边界凸性的结果，共同说明：边界在到期附近有精细非线性结构，不能简单靠粗网格读数草率解释。也就是说，你若在报告中单独讨论“near-expiry extraction difficulty”，这是有充分文献依据的。citeturn32view2turn6search3turn32view0

对**带股息美式 call**，Evans、Kuske 与 Keller 的论文特别有用，因为它同时给出美式 put / call 在带股息情形临近到期的显式近似，并明确说明 call 在有股息时存在真实边界结构。这恰好对应你项目中的 dividend-paying American call 应用实验。citeturn31search3

关于 **Delta / Gamma**，从 PDE/FD 价格面直接做中心差分是成熟路线，而且几乎没有额外求解成本。in ’t Hout 最近的综述型数值论文明确说，通过求解相应 PDCP，可以以几乎零额外成本近似 Delta 与 Gamma；但同一篇论文也指出，CN 对 American-style options 的 Greeks 会出现**不理想的收敛行为**，并引用了 Forsyth–Vetzal 报告的“Gamma 在早行权边界附近的伪振荡”，且这种现象**即使用 backward Euler damping 也未必消失**。更关键的是，该文在做时间收敛评估时，专门把 ROI 选在**远离自由边界**的 continuation region 内。citeturn29view0

这给你的“**mask near payoff kinks and free boundaries**”提供了很强的文献支撑。严格说，文献更常用的术语是选择 ROI、使用 smoothing/damping、或强调 boundary-near interpretation requires caution；但其工程含义与你的 mask 策略一致：**不要把 kink/boundary 邻域的二阶差分当作全域都同样可靠的 Gamma truth。** citeturn26view2turn26view3turn29view0

对神经网络得到的 Greeks，文献总体并不鼓励“只因价格网络可微，就默认其 autograd Greeks 可靠”。Huge–Savine 的 Differential ML 和 Frandsen–Pedersen–Poulsen 的 differential regularization 都说明：如果你关心风险量，显式利用导数标签或微分正则化是有意义的，而且“价格拟合”与“风险拟合”并不是同一个任务。换句话说，你不应把价格 surrogate 自动宣传为 Greek engine。citeturn37view1turn37view0

## Neural Surrogates for Option Pricing

神经网络做期权定价并不是新方向。Hutchinson、Lo 与 Poggio 在 1994 年就提出用学习网络做衍生品定价与对冲，并明确指出这种方法**不是对无套利定价公式的替代**；Ruf 与 Wang 的综述进一步梳理了上百篇相关文献，说明该领域已形成成熟的文献脉络，包括输入变量、输出变量、基准模型、性能指标与数据划分方法。citeturn18search0turn18search4turn17view3

对于“**用 NN 近似定价器/求解器输出**”这一点，Liu、Oosterlee 与 Bohte 的论文非常贴合你的项目：他们把 ANN 训练在由成熟数值方法生成的数据集上，让网络作为原求解器的代理，以换取显著速度收益。这正是你现在以 diagnostic-passing PSOR 生成干净标签，再做 surrogate 学习的基本路线。citeturn18search6

对**美式期权**，现有神经网络文献分成几类。  
一类是直接价格回归，典型如 Gaspar 等对美式 put 市场价格的 NN 定价实验；这类工作常报告 RMSE/运行时收益，但较少系统审计障碍、互补、边界和形状。citeturn34view2  
一类是**最优停止/继续值学习**，例如 Becker、Cheridito 与 Jentzen 的 Deep Optimal Stopping，直接学习 stopping rule，而不是统一输出“价格+边界+Greeks”。这与“boundary or stopping as a separate task”更接近。citeturn18search7turn12search16  
还有一类是“加速已有模型”的 explainable/pricer surrogate，如 Anderson 与 Ulrych，明确写出其方法存在**速度—精度权衡**。这对你的报告非常有用，因为它能帮你把 ML 定位成 benchmark accelerator，而不是 benchmark replacement。citeturn34view1

从你的项目视角看，最重要的文献结论并不是“NN 很强”，而是：**代理模型的评价必须超越价格误差。** PDEBench 明确指出，标准 RMSE 在非光滑 regime 下并不是好代理；这与期权问题里的 payoff kink、自由边界、Gamma 奇异敏感性高度同构。你已有的 baseline 发现“低 RMSE 仍可能违反障碍与形状约束”，在方法论上恰好与这类科学机器学习 benchmark 认识一致。citeturn36view2

关于**数据划分**，Ruf–Wang 在期权 NN 综述中对真实金融时序数据给出很明确的警告：随机划分会破坏时间结构，造成信息泄漏，并低估泛化误差。更一般地，Kapoor–Narayanan 说明泄漏会在 ML-based science 中系统性地制造不可复现结果，而 SciML benchmark 文献则强调 OOD regime/generalization 应该被单独测试。对你基于数值生成标签的仓库而言，这意味着：**random point split 不足以说明 surrogate 对新参数区间、新 regime 或边界附近状态的泛化；regime-level split 更有解释力。** 这条结论带有一定跨领域推断色彩，但推断基础是强的。citeturn35view3turn17view1turn36view0

## Constraint-Aware and No-Arbitrage ML

约束感知期权机器学习的经典起点之一是 Dugas 等关于**把先验函数知识嵌入神经网络**的工作。他们直接针对期权价格函数提出一种带单调性与凸性结构的函数类，并证明其既保留这些性质，又具备通用逼近能力；实验上，他们在 call 定价任务上显示了比无约束回归更好的表现。citeturn23view4

随后，Yang、Zheng 与 Hospedales 提出 “**Rationality by Design**” 的 gated neural networks，主张通过网络结构直接编码经济上合理、满足无套利原则的 call surface；Cao、Liu 与 Zhai 则进一步从 no-arbitrage constraints 出发，构造 hGNN，并让部分权重/斜率选择直接服务于约束满足。两类文献都非常适合拿来支持你论文里的一个核心论点：**如果不加结构，普通 MLP 不会自动给出经济一致的输出；因此，constraint-aware architecture 是合理研究方向。** citeturn33search6turn23view3

更一般的机器学习文献也把方法分成两路：  
一类是**hard constraints by construction**，例如 monotonic architectures；  
另一类是**soft constraints by regularization/penalty**。Runje 与 Shankaranarayana 对这一差异有非常清楚的总结。对你的项目而言，这一分类尤其有用，因为你下一步可以很自然地把 surrogate 设计写成：**正值/障碍一致性倾向于做 hard parameterization，单调/凸性可作为 hard 或 soft 约束，具体取决于表达能力与训练难度。** citeturn11search9

但这里需要一个很重要的限缩：现有约束感知文献大多研究的是**欧式期权价格面**、隐含波动率面、或市场报价曲面，而不是带自由边界的**美式障碍问题**。也就是说，文献能够强力支持“约束需要被显式处理”，却只能间接支持“哪一种美式期权神经结构最好”。这一点必须在最终报告中说清楚。citeturn23view4turn23view3turn33search6

对你提出的**positive continuation-premium price surrogate**，最接近的经典金融依据不是现成的神经网络架构论文，而是**early exercise premium representation** 与障碍/最优停止表述。Kim、Carr–Jarrow–Myneni、Peskir 等文献都把美式价格与额外的早行权价值或边界方程联系起来，这使“在障碍/继续值之上叠一个非负成分”的工程想法很自然。可是，严格说，文献还没有把这个想法在你的一维 CN/PSOR-surrogate 场景中定型为标准架构。因此，这一主张应写成“**受经典 premium/obstacle 表示启发的合理参数化**”，而不是“已有文献证明的唯一正确做法”。citeturn23view5turn32view1turn20search7

## Modular Surrogates, Gaps, Tables, BibTeX, and Writing Guidance

**关于“Price、Boundary、Delta 分开做”**，我认为现有文献对你的立场是“支持模块化，反对过度统一”。Deep Optimal Stopping 直接学习 stopping rule；部分自由边界深度学习论文尝试显式处理 boundary；而 Differential ML 文献说明若风险量是最终用途的一部分，直接使用导数标签或微分正则会比“先拟合价格、再相信 autograd”更稳。综合这些证据，你完全可以把“positive price head / boundary head / bounded Delta head”写成**模块化 surrogate 设计**，并明确说明它们是不同任务、不同标签与不同诊断目标。citeturn18search7turn12search1turn37view1turn37view0

**文献缺口**主要在三处。第一，针对**一维美式期权自由边界**的“直接监督 boundary head”论文并不多；很多深度文献学的是 stopping policy，而不是你现在这种从精细 FD/PSOR 面上提取边界后的监督回归。第二，针对**American Greeks** 的神经监督文献明显少于价格文献，尤其缺少“带 kink/free-boundary mask 的 Greek 标签协议”。第三，针对**约束感知的美式 continuation premium 参数化**，目前多是可由经典理论启发的合理工程设计，而非已有共识标准。正因为如此，你的项目最合适的定位不是“提出普适神经美式求解器”，而是：**在可复现 classical benchmark 上，审计 baseline 失败，再探索结构化、模块化 surrogate 组件是否能更稳定地继承价格—边界—风险量三类性质。** citeturn36view2turn17view2turn17view1turn13search18

**核心参考文献表**如下。表中“Reference”列给出完整引文与 DOI/稳定链接；其余列分别回应你要求的“贡献、项目用途、支持/限定/反驳哪些主张”。  

| Reference | Topic | Method | Relevance to Project | Claim Supported | Caveat / Limitation |
|---|---|---|---|---|---|
| Jaillet, P., Lamberton, D., & Lapeyre, B. (1990). *Variational inequalities and the pricing of American options*. *Acta Applicandae Mathematicae*, 21, 263–289. DOI: 10.1007/BF00047211. citeturn23view0 | 美式期权数学表述 | VI / free-boundary / numerical discussion | 作为你全文 VI/obstacle 语言的经典理论起点 | 支持“项目基于标准 VI/LCP 表述” | 偏理论与经典数值，不涉及现代 ML。citeturn23view0 |
| Huang, J., & Pang, J.-S. (1998). *Option pricing and linear complementarity*. *Journal of Computational Finance*, 2(1), 31–60. DOI: 10.21314/JCF.1998.018. citeturn26view0 | LCP formulation | PDCP/LCP finite differences | 直接支持 obstacle/complementarity 诊断的必要性 | 支持“障碍与互补诊断是必要的” | 更偏离散互补理论，不给出你希望的具体 Python pipeline。citeturn26view0 |
| Kangro, R., & Nicolaides, R. (2000). *Far Field Boundary Conditions for Black–Scholes Equations*. *SIAM Journal on Numerical Analysis*, 38(4), 1357–1368. DOI: 10.1137/S0036142999355921. citeturn27view0 | 域截断与边界条件 | far-field error bounds | 为 domain sensitivity / truncation study 提供理论依据 | 支持“grid/domain studies are necessary” | 论文重点是欧式 PDE；对美式可作有力旁证而非专门结论。citeturn27view0 |
| Pooley, D. M., Vetzal, K. R., & Forsyth, P. A. (2003). *Convergence remedies for non-smooth payoffs in option pricing*. *Journal of Computational Finance*, 6(4), 25–40. DOI: 10.21314/JCF.2003.101. citeturn26view2 | kink / payoff non-smoothness | smoothing, grid shift, projection | 支持 payoff kink 附近特别是 Greeks 要谨慎处理 | 支持“Gamma near kinks/free boundary must be treated cautiously” | 讨论不只限于美式，更多是非光滑合同的一般数值问题。citeturn26view2 |
| Giles, M. B., & Carter, R. (2006). *Convergence analysis of Crank–Nicolson and Rannacher time-marching*. *Journal of Computational Finance*, 9(4). DOI: 10.21314/JCF.2006.152. citeturn26view3 | CN 与 Rannacher | convergence analysis | 直接支持你对 CN 初期阻尼的 benchmark 设计 | 支持“Rannacher smoothing matters for derivatives” | 主要针对 Black–Scholes 型离散；自由边界问题上还需结合其它文献。citeturn26view3 |
| Kim, I. J. (1990). *The Analytic Valuation of American Options*. *Review of Financial Studies*, 3(4), 547–572. Stable URL: RePEc handle in IDEAS. citeturn23view5 | 早行权 premium / boundary | analytic characterization | 为 continuation premium 设计提供金融理论背景 | **限定性支持**“positive premium parameterization is reasonable” | 更接近解析表示，不是 NN 架构论文。citeturn23view5 |
| Carr, P., Jarrow, R., & Myneni, R. (1992). *Alternative Characterizations of American Put Options*. *Mathematical Finance*, 2(2), 87–106. DOI: 10.1111/j.1467-9965.1992.tb00040.x. citeturn5search18 | 美式 put premium 表示 | integral / decomposition | 支持把 premium 或 boundary 作为独立对象研究 | 支持“boundary estimation can be a separate task” | 重点是 put；对带股息 call 需另引文献。citeturn5search18 |
| Peskir, G. (2005). *On the American option problem*. *Mathematical Finance*, 15(1), 169–181. Stable URL: Manchester repository. citeturn32view1 | put 边界唯一性 | local time / nonlinear integral equation | 支持边界是可被独立刻画的对象 | 支持“boundary head/diagnostic has literature basis” | 专注 put；不直接告诉你如何从 FD 网格中稳定提取 boundary。citeturn32view1 |
| Chen, X., & Chadam, J. (2007). *A Mathematical Analysis of the Optimal Exercise Boundary for American Put Options*. *SIAM Journal on Mathematical Analysis*. DOI: 10.1137/S0036141003437708. citeturn32view2 | put 边界存在唯一性与近到期渐近 | free-boundary analysis | 支持 near-expiry boundary behavior 的理论背景 | 支持“boundary behavior near expiry is delicate” | 偏理论；与一维 FD 提取步骤之间仍有工程间隙。citeturn32view2 |
| Evans, J. D., Kuske, R., & Keller, J. B. (2002). *American options on assets with dividends near expiry*. *Mathematical Finance*, 12(3), 219–237. DOI: 10.1111/1467-9965.02008. citeturn31search3 | 带股息 put/call 边界 | near-expiry asymptotics | 直接服务于 dividend-paying American call 应用部分 | 支持“股息会产生真实 call free boundary” | 临近到期渐近，不是全参数区间的通用数值方法。citeturn31search3 |
| in ’t Hout, K. J. (2024). *A note on the numerical approximation of Greeks for American-style options*. arXiv:2401.13361. Stable URL: arXiv. citeturn29view0 | American Greeks | PDCP + temporal discretization | 对 Delta/Gamma 的 mask 与谨慎解释非常有帮助 | 支持“Gamma near free boundary must be treated cautiously”与“Delta/Gamma need separate validation” | 新近预印本，未必像经典教材那样已充分沉淀。citeturn29view0 |
| Hutchinson, J. M., Lo, A. W., & Poggio, T. (1994). *A Nonparametric Approach to Pricing and Hedging Derivative Securities via Learning Networks*. *Journal of Finance*, 49(3), 851–889. DOI: 10.1111/j.1540-6261.1994.tb00081.x. citeturn18search0turn18search4 | 期权定价 NN 先驱 | direct-price learning | 可在文献综述中交代 NN option pricing 的历史起点 | 支持“NN surrogate pricing is established as a direction” | 论文自己就说其法不是无套利公式的替代；与你的 claim restrictions 一致。citeturn18search4 |
| Ruf, J., & Wang, W. (2020). *Neural networks for option pricing and hedging: a literature review*. *Journal of Computational Finance*, 24(1), 1–46. DOI: 10.21314/JCF.2020.390. citeturn9search1turn9search8turn35view3 | 文献综述 | survey | 最适合用来支持你关于数据划分、benchmark、任务分类的综述段落 | 支持“regime/chronological split 比随机器更可信” | 更多面向真实历史金融数据；你需把其泄漏思想转译到模拟参数格点数据。citeturn35view3 |
| Liu, S., Oosterlee, C. W., & Bohte, S. M. (2019). *Pricing Options and Computing Implied Volatilities using Neural Networks*. *Risks*, 7(1), 16. DOI: 10.3390/risks7010016. citeturn18search6 | solver emulator | supervised surrogate | 很适合支持你“PSOR 生成标签、NN 做代理”的工程范式 | 支持“ML can accelerate but not replace benchmark generation” | 主要是欧式/隐波任务，不处理美式自由边界约束。citeturn18search6 |
| Dugas, C., Bengio, Y., Bélisle, F., Nadeau, C., & Garcia, R. (2009). *Incorporating Functional Knowledge in Neural Networks*. *JMLR*, 10, 1239–1262. Stable URL: JMLR. citeturn23view4 | 形状约束 NN | monotone/convex by design | 支持你从结构层面编码单调/凸性 | 支持“direct-price MLP is not automatically shape-preserving” | 针对 call surfaces，而非美式障碍问题。citeturn23view4 |
| Yang, Y., Zheng, Y., & Hospedales, T. M. (2017). *Gated Neural Networks for Option Pricing: Rationality by Design*. AAAI. Stable URL: AAAI/arXiv. citeturn33search6 | 经济一致性架构 | no-arbitrage by design | 支持“architectures can enforce financial constraints by construction” | 支持“hard constraints are credible design choices” | 主要针对欧式 call，不直接覆盖美式 free boundary。citeturn33search6 |
| Cao, Y., Liu, X., & Zhai, J. (2021). *Option valuation under no-arbitrage constraints with neural networks*. *European Journal of Operational Research*, 293(1), 361–374. DOI: 10.1016/j.ejor.2020.12.003. citeturn23view3 | 无套利 NN | hGNN | 为“约束感知架构优于纯回归”提供较新实证支撑 | 支持“constraint-aware ML is meaningful” | 关注市场欧式期权与对冲，不是 PDE-supervised American labels。citeturn23view3 |
| Huge, B. N., & Savine, A. (2020). *Differential Machine Learning*. SSRN. DOI: 10.2139/ssrn.3591734. citeturn37view1 | Greeks 学习 | differential labels / AAD | 为单独 Delta head、导数监督提供关键思想来源 | 支持“Delta from price surrogate requires separate validation or direct supervision” | 不是专门讨论自由边界美式期权。citeturn37view1 |
| Frandsen, M. G., Pedersen, T. C., & Poulsen, R. (2022). *Delta force: option pricing with differential machine learning*. *Digital Finance*, 4(1), 1–15. DOI: 10.1007/s42521-021-00041-7. citeturn37view0 | 微分正则 | differential regularization | 支持“价格与风险量应联合考虑，而非只看价格误差” | 支持“Greeks need dedicated learning signal” | 任务设定偏 Monte Carlo / generic options，不是你的 1D American FD 仓库。citeturn37view0 |
| Becker, S., Cheridito, P., & Jentzen, A. (2019). *Deep Optimal Stopping*. *JMLR*, 20(74), 1–25. Stable URL: JMLR. citeturn18search7turn12search16 | 深度最优停止 | stopping-rule learning | 支持把“边界/停止”视为单独学习任务 | 支持“boundary estimation separate from price regression” | 更偏高维最优停止，不是你的一维 PDE-supervised setting。citeturn18search7 |
| Sandve, G. K., Nekrutenko, A., Taylor, J., & Hovig, E. (2013). *Ten Simple Rules for Reproducible Computational Research*. *PLoS Computational Biology*, 9(10), e1003285. DOI: 10.1371/journal.pcbi.1003285. citeturn13search0 | 可复现研究 | workflow / provenance | 直接支持配置清单、中间结果、实验矩阵 | 支持“reproducible computational finance engineering study” | 非金融专用，但原则普适。citeturn13search0 |
| Wilson, G., et al. (2014). *Best Practices for Scientific Computing*. *PLoS Biology*, 12(1), e1001745. DOI: 10.1371/journal.pbio.1001745. citeturn13search1 | 科学计算最佳实践 | code / automation / testing | 支持仓库工程、自动流程、配置化实验 | 支持“clean, auditable repository practices” | 不是美式期权专文，但非常适合方法节与附录实践规范。citeturn13search1 |

**Claim-to-Literature Map** 如下。  

| Claim | Key literature | Assessment |
|---|---|---|
| CN/PSOR is an appropriate benchmark. | Ikonen–Toivanen 对 PSOR 的“well known and widely applied”描述；Shi 等对 PSOR 在 American LCP 中广泛使用的陈述；Jaillet 等对经典数值方案的理论化讨论。 citeturn25search8turn24search1turn23view0 | **强支持**，但只能支持“benchmark”，不支持“novel solver”。 |
| Obstacle and complementarity diagnostics are necessary. | American pricing as VI/LCP/PDCP 的定义本身就包含这些条件。citeturn26view0turn29view0turn23view0 | **强支持**。 |
| Grid/domain/runtime studies are necessary. | Kangro–Nicolaides 的截断误差；Giles–Carter 与 Pooley–Vetzal–Forsyth 的收敛/非光滑性；PDEBench 对单一 RMSE 不足的提醒。 citeturn27view0turn26view2turn26view3turn36view2 | **强支持**。 |
| Gamma near the free boundary must be treated cautiously. | in ’t Hout 2024 明确回顾了 CN 下 Gamma 在边界附近的伪振荡；Pooley–Vetzal–Forsyth 强调非光滑性会污染 hedging parameters。 citeturn29view0turn26view2 | **强支持**。 |
| Direct-price neural surrogates are not automatically constraint-preserving. | Dugas 2009、Yang 2017、Cao 2021 都是因为普通网络不会自动满足金融形状/无套利结构才去显式编码约束。 citeturn23view4turn33search6turn23view3 | **强支持**。 |
| Positive continuation-premium parameterization is a reasonable way to enforce obstacle consistency. | Kim 1990、Carr–Jarrow–Myneni 1992、Peskir 2005 提供 premium/boundary 表示。 citeturn23view5turn5search18turn32view1 | **间接支持**；应写成“受经典表示启发的合理设计”，不要写成既有定论。 |
| Boundary estimation should be treated as a separate diagnostic task. | AitSahlia–Lai 为边界本身建模并用于 price/hedge；Peskir 直接研究边界积分方程；Deep Optimal Stopping 学 stopping rule。 citeturn23view2turn32view1turn18search7 | **中到强支持**。 |
| Delta from a price surrogate requires separate validation. | Differential ML / differential regularization 明确把导数标签当独立训练信号；American Greeks 文献表明 price 与 Greeks 收敛性质不同。 citeturn37view1turn37view0turn29view0 | **强支持**。 |
| Regime-level test splits are more meaningful than random point splits. | Ruf–Wang 关于随机划分与泄漏；Kapoor–Narayanan 关于 ML science 中系统性 leakage；SciML/OOD benchmark 文献关于 out-of-distribution generalization。 citeturn35view3turn17view1turn36view0 | **中等支持**；在你的模拟参数网格 setting 中，这是一条很合理的跨文献推断。 |

**推荐 BibTeX 条目**如下，优先覆盖经典教材/经典论文/高引用论文与你最相关的现代 ML 文献：

```bibtex
@article{JailletLambertonLapeyre1990,
  author  = {Jaillet, Patrick and Lamberton, Damien and Lapeyre, Bernard},
  title   = {Variational inequalities and the pricing of American options},
  journal = {Acta Applicandae Mathematicae},
  volume  = {21},
  pages   = {263--289},
  year    = {1990},
  doi     = {10.1007/BF00047211}
}

@article{HuangPang1998,
  author  = {Huang, Jacqueline and Pang, Jong-Shi},
  title   = {Option pricing and linear complementarity},
  journal = {Journal of Computational Finance},
  volume  = {2},
  number  = {1},
  pages   = {31--60},
  year    = {1998},
  doi     = {10.21314/JCF.1998.018}
}

@article{KangroNicolaides2000,
  author  = {Kangro, Raul and Nicolaides, Roy},
  title   = {Far Field Boundary Conditions for Black--Scholes Equations},
  journal = {SIAM Journal on Numerical Analysis},
  volume  = {38},
  number  = {4},
  pages   = {1357--1368},
  year    = {2000},
  doi     = {10.1137/S0036142999355921}
}

@article{PooleyVetzalForsyth2003,
  author  = {Pooley, David M. and Vetzal, Kenneth R. and Forsyth, Peter A.},
  title   = {Convergence remedies for non-smooth payoffs in option pricing},
  journal = {Journal of Computational Finance},
  volume  = {6},
  number  = {4},
  pages   = {25--40},
  year    = {2003},
  doi     = {10.21314/JCF.2003.101}
}

@article{GilesCarter2006,
  author  = {Giles, Michael B. and Carter, Rebecca},
  title   = {Convergence analysis of Crank--Nicolson and Rannacher time-marching},
  journal = {Journal of Computational Finance},
  volume  = {9},
  number  = {4},
  year    = {2006},
  doi     = {10.21314/JCF.2006.152}
}

@article{Kim1990,
  author  = {Kim, In Joon},
  title   = {The Analytic Valuation of American Options},
  journal = {Review of Financial Studies},
  volume  = {3},
  number  = {4},
  pages   = {547--572},
  year    = {1990},
  url     = {https://ideas.repec.org/a/oup/rfinst/v3y1990i4p547-72.html}
}

@article{CarrJarrowMyneni1992,
  author  = {Carr, Peter and Jarrow, Robert and Myneni, Ravi},
  title   = {Alternative Characterizations of American Put Options},
  journal = {Mathematical Finance},
  volume  = {2},
  number  = {2},
  pages   = {87--106},
  year    = {1992},
  doi     = {10.1111/j.1467-9965.1992.tb00040.x}
}

@article{Peskir2005,
  author  = {Peskir, Goran},
  title   = {On the American option problem},
  journal = {Mathematical Finance},
  volume  = {15},
  number  = {1},
  pages   = {169--181},
  year    = {2005},
  url     = {https://eprints.maths.manchester.ac.uk/id/eprint/408}
}

@article{ChenChadam2007,
  author  = {Chen, Xinfu and Chadam, John},
  title   = {A Mathematical Analysis of the Optimal Exercise Boundary for American Put Options},
  journal = {SIAM Journal on Mathematical Analysis},
  year    = {2007},
  doi     = {10.1137/S0036141003437708}
}

@article{EvansKuskeKeller2002,
  author  = {Evans, J. D. and Kuske, R. and Keller, J. B.},
  title   = {American options on assets with dividends near expiry},
  journal = {Mathematical Finance},
  volume  = {12},
  number  = {3},
  pages   = {219--237},
  year    = {2002},
  doi     = {10.1111/1467-9965.02008}
}

@article{HutchinsonLoPoggio1994,
  author  = {Hutchinson, James M. and Lo, Andrew W. and Poggio, Tomaso},
  title   = {A Nonparametric Approach to Pricing and Hedging Derivative Securities via Learning Networks},
  journal = {Journal of Finance},
  volume  = {49},
  number  = {3},
  pages   = {851--889},
  year    = {1994},
  doi     = {10.1111/j.1540-6261.1994.tb00081.x}
}

@article{RufWang2020,
  author  = {Ruf, Johannes and Wang, Weiguan},
  title   = {Neural networks for option pricing and hedging: a literature review},
  journal = {Journal of Computational Finance},
  volume  = {24},
  number  = {1},
  pages   = {1--46},
  year    = {2020},
  doi     = {10.21314/JCF.2020.390}
}

@article{LiuOosterleeBohte2019,
  author  = {Liu, Shuaiqiang and Oosterlee, Cornelis W. and Bohte, Sander M.},
  title   = {Pricing Options and Computing Implied Volatilities using Neural Networks},
  journal = {Risks},
  volume  = {7},
  number  = {1},
  pages   = {16},
  year    = {2019},
  doi     = {10.3390/risks7010016}
}

@article{DugasEtAl2009,
  author  = {Dugas, Charles and Bengio, Yoshua and B{\'e}lisle, Fran{\c c}ois and Nadeau, Claude and Garcia, Ren{\'e}},
  title   = {Incorporating Functional Knowledge in Neural Networks},
  journal = {Journal of Machine Learning Research},
  volume  = {10},
  pages   = {1239--1262},
  year    = {2009},
  url     = {https://jmlr.org/papers/v10/dugas09a.html}
}

@article{CaoLiuZhai2021,
  author  = {Cao, Yi and Liu, Xiaoquan and Zhai, Jia},
  title   = {Option valuation under no-arbitrage constraints with neural networks},
  journal = {European Journal of Operational Research},
  volume  = {293},
  number  = {1},
  pages   = {361--374},
  year    = {2021},
  doi     = {10.1016/j.ejor.2020.12.003}
}

@article{BeckerCheriditoJentzen2019,
  author  = {Becker, Sebastian and Cheridito, Patrick and Jentzen, Arnulf},
  title   = {Deep Optimal Stopping},
  journal = {Journal of Machine Learning Research},
  volume  = {20},
  number  = {74},
  pages   = {1--25},
  year    = {2019},
  url     = {https://jmlr.org/papers/v20/18-232.html}
}

@article{HugeSavine2020,
  author  = {Huge, Brian Norsk and Savine, Antoine},
  title   = {Differential Machine Learning},
  journal = {SSRN Electronic Journal},
  year    = {2020},
  doi     = {10.2139/ssrn.3591734}
}

@article{FrandsenPedersenPoulsen2022,
  author  = {Frandsen, Magnus Gr{\o}nnegaard and Pedersen, Tobias Cramer and Poulsen, Rolf},
  title   = {Delta force: option pricing with differential machine learning},
  journal = {Digital Finance},
  volume  = {4},
  number  = {1},
  pages   = {1--15},
  year    = {2022},
  doi     = {10.1007/s42521-021-00041-7}
}

@article{SandveEtAl2013,
  author  = {Sandve, Geir Kjetil and Nekrutenko, Anton and Taylor, James and Hovig, Eivind},
  title   = {Ten Simple Rules for Reproducible Computational Research},
  journal = {PLoS Computational Biology},
  volume  = {9},
  number  = {10},
  pages   = {e1003285},
  year    = {2013},
  doi     = {10.1371/journal.pcbi.1003285}
}

@article{WilsonEtAl2014,
  author  = {Wilson, Greg and Aruliah, Dhavide A. and Brown, C. Titus and Chue Hong, Neil P. and Davis, Matt and Guy, Richard T. and Haddock, Steven H. D. and Huff, Kathryn D. and Mitchell, Ian M. and Plumbley, Mark D. and Waugh, Ben and White, Ethan P. and Wilson, Paul},
  title   = {Best Practices for Scientific Computing},
  journal = {PLoS Biology},
  volume  = {12},
  number  = {1},
  pages   = {e1001745},
  year    = {2014},
  doi     = {10.1371/journal.pbio.1001745}
}
```

**Writing Guidance** 建议如下。  

可以安全使用的表述：

- “本文采用经典 VI/LCP 视角，将美式期权定价问题组织为可诊断的自由边界数值基准，并用 CN/PSOR 生成经诊断筛选的监督标签。” 这里在方法与数学背景部分引用 Jaillet、Huang–Pang、Kangro–Nicolaides。citeturn23view0turn26view0turn27view0
- “由于 payoff 非光滑与自由边界的共同作用，特别是对 Gamma，单纯价格误差不足以保证风险量可靠，因此本文将 Greeks 验证与价格验证分开报告。” 这里在 Greeks/diagnostics 部分引用 Pooley–Vetzal–Forsyth、Giles–Carter、in ’t Hout。citeturn26view2turn26view3turn29view0
- “本文把机器学习定位为经典基准的代理与诊断补充，而非替代。” 在引言或定位段可引用 Hutchinson–Lo–Poggio、Liu–Oosterlee–Bohte、Anderson–Ulrych。citeturn18search4turn18search6turn34view1
- “本文的约束感知 surrogate 设计受到早行权 premium 表示、障碍问题表述与无套利/形状约束神经网络文献的共同启发。” 在设计动机段可引用 Kim、Carr–Jarrow–Myneni、Dugas、Cao。citeturn23view5turn5search18turn23view4turn23view3
- “本文强调 regime-level 评测与失败区间报告，以避免随机点划分带来的过度乐观评估。” 在实验协议部分可引用 Ruf–Wang、Kapoor–Narayanan、Sandve、Wilson。citeturn35view3turn17view1turn13search0turn13search1

应避免的表述：

- 避免说“我们提出了一种新的 American option solver”。更准确的说法是“我们建立并审计了一个可复现 classical benchmark”。citeturn23view0turn24search1
- 避免说“单个神经网络同时解决价格、边界和 Greeks”。文献并未给出这样一个在你 setting 下已被确立的范式。citeturn18search7turn37view1
- 避免说“价格网络天然就是 Greek engine”。Differential ML 与 American Greeks 数值文献都不支持这种自动推论。citeturn37view1turn37view0turn29view0
- 避免说“ML 替代了 PSOR”。更稳妥的是“在 PSOR 标签可信的 regime 内学习近似器，并用经典诊断持续审计”。citeturn18search6turn34view1
- 避免把“positive continuation-premium parameterization”写成已有定论；建议写成“受文献启发的合理结构性假设，需经实验验证”。citeturn23view5turn32view1

最后，如果要把全文定位成你期望的版本，我建议摘要和引言都围绕以下一句展开：  
**“本项目是一项可复现的 research-engineering study：以经验证的 CN/PSOR 美式期权 benchmark 为核心，系统审计自由边界、障碍/互补条件与 Greeks 可靠性，再在此基础上评估并设计模块化、约束感知的神经 surrogate 组件。”** 这句话与现有文献最一致，也最不容易过度宣称。citeturn23view0turn26view0turn29view0turn23view3turn13search0turn13search1