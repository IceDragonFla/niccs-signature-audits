# Shuttle 密钥恢复研究：转移方向反转、γ 的精确恢复与提升方程

> 撰写日期：2026-10-19　作者：GLM-5.3 Max、JY Liu。本研究在算法发布方书面许可下进行（2026-10 起生效，
> 实现层攻击研究）。方向反转发现的完整报告（10 页 LaTeX）见
> [docs/shuttle_attack_report.pdf](docs/shuttle_attack_report.pdf)；战役全程
> 纪要（含至第十八次补充）与三份专报在 [docs/](docs/)。本报告汇总截至撰写日的
> 全部已确证结论与未决问题。

**摘要**：Shuttle 是免拒绝迭代拒绝采样（iterative rejection sampling）在 MLWE
上的 Fiat–Shamir 型签名实例化，NICCS 首轮提交物为其竞赛重参数化版本。我们
在规范书 Algorithm 22 与全部九个实现变体中发现一个转移方向反转：交替级数
转移概率 S_v 被赋给离心移动，而方案自身 §4.1 的安全论证要求它赋给向心移动。
单个语义位的反转使签名响应分布显著偏离目标高斯（秘密方向投影方差 +28.5%，
z = +35.4σ），公开数据 850 份签名即可 5σ 区分出厂实现与正确实现；实测
Rényi 散度超预算约 76 个比特，出厂实现的分布性安全论证前提被否定。在此
基础上，我们将签名文本中的互相关泄漏累积为分圆环元素 γ = s·s̄ 的精确值
（342M 签名 + 30 个训练密钥的仿射去偏，512/512 系数精确取整），由 γ 构造
攻击者理想，并完整执行了两条代数端局：子域塔（dim-128 uSVP 在 25 秒内解出
N(s₀)）与 Gentry–Szydlo 乘子链（全量逆向 + 自实现 + 真值验证后，判定
**结构性发散**——经典 GS 的多项式时间论证隐含的嵌入谱前提在本实例不成立）。
至此密钥提取被归约为单一、可精确量化的格问题：dim-256 扭曲格中 gap ≈ 1.01
的 uSVP（需 β ≈ 250+ 的 BKZ，超出当前技术）；或经提升方程
A² − yB² = N(s₀) 的整数小解绕开（其全部连续松弛与廉价捷径已被系统排除）。
最后，必须区分两个层面：全部恢复进展都以实现层的方向反转缺陷为前提——设计层
（两篇论文定义的正确构造）在我们已测的全部信道上保持完好。**实现层攻破与
设计层未破的差距，是本研究最重要的安全学结论**（§9.2）。

**Abstract**: Shuttle is a rejection-free iterative-rejection-sampling lattice
signature (MLWE, Fiat–Shamir). We find a transition-direction reversal in the
competition package's Algorithm 22 and all nine implementation variants: the
alternating-series transition probability S_v is assigned to the centrifugal
move, whereas the scheme's own security argument (§4.1) requires the
centripetal one. A single semantic bit inflates the secret-direction projection
variance by +28.5% (z = +35.4σ); 850 public signatures distinguish the shipped
implementation from a correct one at 5σ, and the measured Rényi divergence
exceeds the claimed budget by ~76 bits, invalidating the shipped distributional
security argument. We then accumulate the cross-correlation leakage into the
exact value of the cyclotomic element γ = s·s̄ (342M signatures, affine
de-biasing trained on 30 keys, 512/512 coefficients exact), construct the
attacker's ideal, and carry both algebraic endgames to conclusion: the subfield
tower (dim-128 uSVP yields N(s₀) in 25 seconds) and the Gentry–Szydlo
multiplier chain (fully reverse-engineered, implemented and ground-truth
validated, then judged **structurally divergent** on this instance—the
embedding-spectrum premise of the classical polynomial-time argument fails
here). Key extraction is thereby reduced to a single, precisely quantifiable
lattice problem: uSVP with gap ≈ 1.01 in the dim-256 twisted lattice (requiring
BKZ at β ≈ 250+, beyond current technology), or a bypass via the integer
small solutions of A² − yB² = N(s₀). Crucially, all recovery progress is
premised on the implementation-level transcription defect; the design level
(the correct construction of the two papers) remains intact on every channel
we have measured. The gap between an implementation break and a design break
is the central security lesson of this study.

## 1　对象、授权与结论概览

对象为 `shuttle` 提交包（SHUTTLE-128/256/512 × ref/avx2/avx512 九个变体；
SHUTTLE-128：n = 256，q = 15361，τ = 42，宽高斯速率 r = 825）。技术谱系：
Lyubashevsky 的 Fiat–Shamir with aborts → BLISS 双模高斯 → Gärtner 的迭代
拒绝采样（ePrint 2024/2052，CRYPTO 2025：把一次性大拒绝分解为 τ 步小转移，
条件于成功精确保持目标分布）→ Shuttle 论文（ePrint 2026/1991，ASIACRYPT
2026：去掉 abort，用 Rényi 散度界住每步漂移）。提交物是论文的大幅重参数化
重写。

按攻击目标的三个层级给出结论概要，逐项证据在后续各节：

1. **大幅缩减安全强度——成立**。出厂实现的分布性安全论证前提被实测否定
   （§2）；统计区分 850 份签名。
2. **从设计上攻破提交物——成立**。Algorithm 22 与其自身的 §4.1 证明矛盾，
   出厂代码没有任何有效的零知识论证覆盖。但这**不推翻**两篇论文：Gärtner
   理论与 Shuttle 论文的数学、伪代码均正确（我们独立重推验证，§2.3）。
3. **恢复等价密钥——尚未达成**，但已完全定界（§9.1）：全部代数量可由转录
   导出，密钥提取归约为单一格问题（dim-256 扭曲 uSVP，gap ≈ 1.01）或提升
   方程的整数小解。

## 2　发现：转移方向反转

### 2.1　缺陷的形态

Gärtner 框架中，转移概率由交替级数 S_v(y) = Σ_{k≥0}(−1)^k ρ_g(y+kv)/ρ_g(y)
给出。Shuttle 论文 Figure 1 约定 p_v(y) = Pr[T_v(y) = y − v]（向心），
Figure 2 的免对数实现 T3 中"区间命中 ⇒ 移动 y − v"。提交物的 Algorithm 22
（`irs.c::R_transition`，三后端逐行对应）区间测试本身实现正确且锚点正确
（端点公式与论文 T3 的 J_N 逐项相等），**唯一错在命中后的移动方向**：
出厂实现把 S_v 给了离心移动。即 P[向心] = 1 − S_v，应为 S_v。

### 2.2　为什么功能测试看不见它

验证方程依赖双峰恒等式 q·(±1) ≡ q (mod 2q)：IRS 每步转移 z ← z ± v 的
独立符号选择被该恒等式完全吸收——无论方向指派对错，签名永远合法。KAT
一致性、签名/验证正确性、确定性 DRNG 重放等全部出厂测试对本缺陷**原理上
不可见**（免疫性引理，证明见 docs/shuttle_attack_report.pdf §3）。缺陷
只能通过规范书内部一致性审计或输出分布的统计检验暴露；我们两者都做了。

### 2.3　实测证据与量化

固定密钥、84,000 次转移、2,000 次独立 (y, c, IRS) 试验：

| 统计量 | 出厂版 | 两行修补版 | 独立高斯对照 |
|---|---|---|---|
| Var(⟨z, v_j⟩)/σ²V（控制组 1.0061） | **1.2847** | 1.0063 | 0.9923 |
| 相对膨胀 z 值 | **+35.4σ** | +0.0 | — |

出厂签名 E‖z₁‖² 相对公开可计算的解析基线上移 +0.88%，200,000 份签名下
z = +76.6，折算 **850 份签名即 5σ**——免侧信道、免伪造、纯被动的公开
区分器。由均值偏移换算（KL ≥ δ²/2），实测 R₂(出厂‖理想) ≥ 1 + 2^−6.1/
签名，而规范书断言 IRS Rényi 因子 ≤ 1 + 2^−82：**超预算约 76 个比特**。
这不是裕量变小，是论证前提不成立。

修复是两行（三后端各一份，`flag64` 的两个常量对调），修补后膨胀归零
（z = +0.0）。我们同时建议提交方在 KAT 之外增加分布性回归测试
（Var(⟨z, v_j⟩)/σ²V 与 E‖z₁‖² 相对解析基线的偏移，5σ 阈值、10⁴ 量级
签名即可自动化）——免疫性引理的直接推论。

### 2.4　致错机理与文献边界

与两篇文献逐条对照后的定性：Gärtner 2024/2052 与 Shuttle 论文 2026/1991
的数学与伪代码均正确（论文不变量公式 ρ′_v(y) 我们独立重推逐项通过，含
中带边界情形）。提交物的方向指派恰是"Gärtner 的移动方向 × Shuttle 的
锚点约定"的交叉组合——两个正确约定的所有交叉组合中唯一错误的那格。
这解释了为何区间测试逐项正确而方向错误：转录发生在语义层而非数值层。
精度审计（阶段 I，docs/shuttle_高斯采样器精度深挖.md）已先行排除数值
近似：96-bit 反向 CDT、ApproxExp/ApproxLog、卷积+拒绝全部达标
（R₁₀₂₅ = 1+2^−95.40，χ²/df = 0.9985）——错误不在采样管线，这把攻击面
精确孤立到转移核的方向指派。

论文作者对 Gärtner 框架的批评中预见的是数值近似形态的脆弱（截断、定点、
近似比较）；本发现以未列举的第三种形态（方向语义）成真。对文献的影响：
不构成对 Gärtner 理论的反例，但表明 S_v 这类"方向语义重载"的记号是
转录错误的客观温床——Rényi 界的逐步核算对语义位零覆盖。

## 3　从区分到泄漏：γ 张量

方向反转使响应分布携带秘密的系统性信息。泄漏信道的完整地图（全部实证）：

| 信道 | 状态 | 证据 |
|---|---|---|
| 对角+跨块互相关 C_{kk′}(d) = c″·γ_{kk′}(d)，三块两两 6 对 | **泄漏** | c″ ≈ 84.0 六对共用，R² = 0.92–0.946（2×10⁵ 签名） |
| 均值信道 E[z\|c] | 关闭 | P[+rot] = ½S + ½(1−S) ≡ ½ 恒等式；实测 E[f] = −0.0025±0.0035 |
| 跨分量二阶边际 | 恰好抵消 | 实测 rms 0.23 = 噪声底；早期"线性求解恢复 s"为估计器假象，已撤回 |

oracle 诊断（给定真值模数与真值相对相位）下重建达 762/768（99.2%），
4M 签名 + 真相位下 768/768——**信息充分，瓶颈在求解**：频域泄漏 Gram 的
秩 1 主导分量给出 ĝ ĝ* 的方向，但每频率的离散相位（128 维）构成极深伪
整数盆地，十四族求解器（角同步 GN、含模数 GS、θ-L-BFGS、交叉目标 CD、
模拟退火 3×10⁷ 步、分块 Gauss-Seidel、fpylll 格 ILS、204 起点 6 进程
锦标赛等）全部止步于 270–460/768。真解是明确的全局最优（cost(s_true)
比全部伪解低 ≥3.3×）——搜索失败而非错误最优。

多密钥留一（LOO）偏差学习确认系统偏差张量的跨键共性分量约占 1/3；同密钥
独立 2M 流的差恰为理论蒙特卡洛噪声底（rms 2.024 对 2.022，逐位一致）——
"非仿射机制项"不存在，残差即统计底。这一判别决定了战略转向：**不与相位
搜索缠斗，改走环分解**——γ = s·s̄ 是 Z[ζ₅₁₂] 中的范数型对象，绕开相位
恢复。

## 4　γ 的精确恢复

数据侧终点由三件工作闭合：

1. **交叉通道代数**。(γ₀₁) + (γ₀₂) = (s₀)(s̄₁, s̄₂) = (s₀) 在 5/5 个真实
   密钥上精确成立（成员关系经精确整数重构验证，|det B₀| 与 |N(s₀)| 逐位
   相等）——理想生成只需两个交叉通道，噪声最大的对角通道完全绕开。
2. **跨键仿射去偏**。系统偏差经 17 抽头通道卷积仿射算子建模（LOO 交叉
   验证）：训练键数 6 → 15/4 处取整错误；18 → 1+1（定位边缘滑落 lag）；
   **30 → 0+0**。
3. **数据量**。342M 目标签名（12 × 14.25M × 2 批次，流式 14GB 随机源）。

结果：**γ₀₁、γ₀₂ 全部 512/512 系数精确取整**，与真值逐位相等；由取整 γ
构造的攻击者理想 (γ₀₁) + (γ₀₂) 中 s₀ ∈ J 精确成立。

## 5　代数端局之一：直接归约与工具链修复

给定 γ，密钥恢复等价于在分圆整数环中从理想恢复短向量。等价密钥类为
{sk, −sk}（双峰符号对称），e′ 可由公钥与 s 精确导出（harness_forge.c
端到端验证）。

**难度阶梯**（fpylll，mpfr-212）：n = 32/64 LLL 直接恢复（0.2/0.7 s）；
n = 128 BKZ-20 完整恢复 9 s（生成元 + 除法步精确整数验证）；n = 256 阻塞。
n = 256 的完整攻坚账本（全部实测，供复现者评估）：

| 尝试 | 结果 |
|---|---|
| LLL（各基变体，proved/fast） | ‖b₁‖ = 185.6，目标 14.3 |
| fplll BKZ β≥16 | babai "infinite loop" 中止（0.5.2/0.6.4 全部基变体；随机 256 维对照 0.7 s 干净通过——实例特异） |
| BKZ delta = 0.80 | 6.7 小时真实推进后中止 |
| G6K workout（MAX_SIEVING_DIM 重建至 280） | 24 h、筛维 ~143、29 GB、无命中 |
| 手版/检查点式 BKZ-30/40/50 | 数千轮头部不动 |

经验定性：512 条重简并轨道（±x^j·s₀）+ LLL-only 起点使实例远难于 uSVP
gap 估计（所需 β ≫ 143 筛维等效），接近 dim-256 近 SVP。

**工具链修复**：fplll 的 lll.cpp 在 babai 不收敛点直接抛错，我们改为硬
迭代上限的优雅降级（三机构建链：Windows 克隆 → 本地 WSL autoreconf →
远端编译）。补丁后 BKZ-40 在原卡死基族上首次完整运行（78 s）——但坏
轮廓是根因，标准 BKZ 管线仍不足以解 n = 256。补丁对社区的独立价值在于
解锁了该类退化基上的 BKZ 可用性。

## 6　代数端局之二：子域塔

结构：K = Q(ζ₅₁₂) ⊃ K′ = Q(ζ₂₅₆)，σ: x ↦ −x；
s₀ = A(y) + xB(y)（A、B 为偶/奇系数部，‖A‖, ‖B‖ ≈ 10），则

```
N(s₀) = A² − yB² ∈ R′ = Z[y]/(y¹²⁸ + 1)
```

（精确验证通过；|σ_j(s₀)|² 完全由 N(s₀) 决定，一致到 1e-15）。
(N(γ₀₁), N(γ₀₂)) = (N(s₀)) 是 dim-128 理想，行列式相等验证通过。
dim-128 实例的定量：det^{1/128} = 122.6，GH = 335.6，目标 ‖N(s₀)‖ =
194.4——**真 uSVP，gap 1.73**。

第一步已实战打通：攻击者数据（取整 γ 的相对范数）→ dim-128 理想 →
LLL + 渐进 BKZ-40，**25 秒精确弹出 N(s₀)**（规范代表 y⁶），无 babai
问题（`tower_step1.py`；校验 `tower_check2.py`）。

## 7　剩余缺口：提升方程的定界

塔的第二步是求小解 A, B（‖·‖ ≈ 10）使 A² − yB² = N(s₀)。信息论上 N
只丢失 s/σ(s) 的区分比特；问题良定义，但全部已知捷径被系统实验排除
（完整实验记录见 docs/shuttle.md 第十六、十七次补充）：

- **连续松弛全部失败且失败方式一致**：精确模长交替投影停 99–107/256；
  解析雅可比 Newton 真值驻留完美、90% 损坏暖启动即发散；PhaseLift SDP
  可行但松弛不紧（特征值简并 4.509/4.509/4.206，逃逸秩 1）；信赖域
  最小二乘 60 次多重启全部收敛到零残差伪解（重合 94–109/256）。二次
  系统在连续域不唯一——**整数性与小性是唯一的钉子**。
- **廉价代数捷径排除**：γ₀₁↔γ₀₂、γ₀₁↔N 的诚实 gcd 全部平凡（折叠
  mod x²⁵⁶+1 破坏整除性）；诚实乘积的整除性检验全部失败。
- **朴素搜索排除**：y-adic 前向递归中提升高半系数 q（|q| ≤ 40 实测）
  与 B 逐级耦合，真值前缀下每层 49 个存活对——DFS 为 49¹²⁸ 量级。

**端局已验证**：若得 Ñ(x²) 的 Z[x] 因子分解（Ñ 为诚实提升），35 秒
内分离 s·s(−x)。剩余唯一严肃路线：**Coppersmith/Cohn–Heninger 型格
构造**，把 |A|, |B| ≤ 3 级别的整性内嵌进移位多项式格（p^k 缩放）。
这是一个需要专门设计的构造性任务。

## 8　代数端局之三：Gentry–Szydlo 乘子链——彻底执行与结构性判定

与塔互补的第二条端局（不经过 N(s₀)）：给定 γ = v·v̄ 与理想 J = (s₀)，
GS 算法原则上直接恢复 v。我们对原始 GS 论文 §7.2–7.5 + 附录 F 的乘子链
路线与 Lenstra–Silverberg（ePrint 2014/430）的循环群表述做了全量逆向
（含 PARI 参考实现 ideaux.gp 的精度判定：其内层 petitlll 启发式为
n = 1024 调参，n = 16 即数值崩坏，故自实现），完成自实现并逐层真值验证：

- **扭曲 Gram 恒等式** B·M(1/γ)·Bᵀ = U·Uᵀ：256 维扭曲嵌入 LLL
  （mpfr@256，12 s），全部 256 行验证 b′_i = v·a_i 且 a_i 积分；
- **链恒等式**（自行推导）：γ_{j+1} = ρ_j·γ_j²/ρ_{j−1}²、
  v_{j+1}/v_j² = R_j/R_{j−1}²、B/F 理想互递推；层 0、层 1 精确通过
  （R_j = f^{2^j}·v_{j+1}，分母为 1、γ 自洽、乘积一致）。

**结构性判定：经典乘子链在本实例上发散。** 尺度递推
w_{j+1}(i) = w_j(i)²·|σ_i(c_j)| 与 |σ(f)| 的具体值无关（f^{2^j} 在定义中
精确消去）；链的可行性要求每层取到反向轮廓乘子——这等价于每层求解扭曲
最短向量本身（uSVP，gap ≈ 1.01）。经典 GS 的多项式时间论证隐含依赖
|σ(f)| 一致大于 1（NTRU 签名私钥天然满足；本实例嵌入谱
log|σ(s₀)| ∈ [−0.166, 3.459]、min|σ(s₀)| = 0.847，不满足）。实测确认：
对数尺寸逐层翻倍，与 w² 律精确吻合（v₁ 系数峰值 ≈ e^10，v₂ ≈ e^26.5），
层 2 的 B-步整数性断言失败。四条捷径（子格 v_j²(a₀)、随机混合降维
512→256、缩放 v′ = 2v——乘子逐元素不变、二段扭曲/单位再平衡）全部实测
排除。发散是数学性质而非实现 bug——层 0/1 恒等式精确通过、w² 律与实测
逐层吻合即为对照。Lenstra–Silverberg 定理 1.1 声称无条件多项式时间的
分次张量代数表示，我们未能落地为可计算的系数形式（张量平方运算的可实现
构造在文献中未给出）——开放残留。

基础设施配方（对复现者重要）：一切 LLL 限制在 dim = 256/512 行满秩
+ FPLLL.set_precision(256) + GSO.Mat(mpfr) 手动路径（512 维随机 27bit
29.9 s，真实扭曲 512 矩阵 1390 s）；fpylll 默认 wrapper 与 ld/dpe 在
大表项矩阵上系统性病态；flint 精确 LLL 与 HNF 均不可用（超时/表项溢出）。

## 9　终局归约与实现层—设计层的差距

### 9.1　密钥提取已归约为单一格问题

泄露侧至此完整：签名转录确定性地给出 γ = s₀s̄₀（512/512 精确）、理想
J = (s₀)、以及塔路线的 N(s₀)。由 GS 唯一性理论，(γ, J) 决定 s₀ 至 ±ζ^j
（1024 元歧义类，恰为等价密钥类 {±x^j·sk}）。于是密钥提取被归约为一个
可精确量化的格问题：**dim-256 扭曲格中的 uSVP**——λ₁ 为扭曲范数恰为 1 的
1024 重根轨道（±x^j·s₀），λ₁₀₂₅ 为分圆单位尾（≈ 1.01 倍），gap ≈ 1.01，
需要 β ≈ 250+ 的 BKZ，超出当前技术。对照 σ 侧（子域塔）gap = 1.73、
BKZ-40 即以 25 秒解决——两个二次扩张的不对称性就是整个端局。

对提交物而言，代数安全边际已为零：一切代数量均可由转录导出，剩余防线是
这一个格问题的硬度——或者提升方程若被攻克（§7），连这一防线也绕开。

### 9.2　实现层攻破与设计层未破的差距

必须明确区分两个层面，二者的结论相反，这是本研究最重要的安全学结论。

**实现层（提交物）：已攻破。** 全部恢复进展的前提是 Algorithm 22 的方向
反转转录缺陷。它使出厂实现的分布性安全论证失效（Rényi 超预算约 76 bit），
并使签名文本携带完整的互相关泄漏——γ、J、N(s₀) 全部由缺陷实现的转录
导出，代数安全边际归零（§9.1）。

**设计层（两篇论文定义的正确构造）：在我们已测的全部信道上保持完好。**
判定依据是我们自己的因果实验与数学核对：

- 两行补丁（方向对调）后，方差膨胀归零（1.2847 → 1.0063，z = +0.0，与
  控制组无差），公开区分器失效；
- 正确方向下不变量 ρ′_v(y) = ρ_g(y) 精确成立（我们独立重推逐项通过）：
  响应分布不依赖 witness，互相关泄漏张量失去来源——正确实现不发布 γ，
  攻击者就没有理想、没有塔、没有扭曲 uSVP 实例，本报告的整条代数链没有
  着力点；
- 均值信道与跨块二阶边际信道在设计层本就由恒等式关闭（§3）。

换言之：**一个语义位的转录错误，可以把一个设计完好的方案在实现层完全
击穿**——泄漏从微架构侧信道整体迁移到签名文本（攻击成本降为零），代数
安全边际从"完整"直接归零。防御上，这对应我们已通报的修复（两行补丁 +
分布性回归测试）；评估上，这意味着"设计安全 ≠ 提交物安全"——语义审计
必须与数值审计并列成为该技术路线落地的标准环节。对攻击者而言，本研究的
全部端局（塔、GS、提升方程）都以实现层缺陷为入口；设计层本身，在我们
动用过的全部手段下，仍然难以攻破。

## 10　恢复程度总表

| 层级 | 状态 | 量化 |
|---|---|---|
| 分布性安全论证 | **已否定** | Rényi 超预算 ~76 bit；850 份签名 5σ 区分 |
| 设计自洽性（提交物） | **已否定** | Algorithm 22 与 §4.1 矛盾；两行修复因果归零 |
| γ = s·s̄ | **精确恢复** | 512/512 系数，342M 签名 + 30 训练键 |
| N(s₀)（塔第一步） | **精确恢复** | dim-128 uSVP gap 1.73，BKZ-40 25 s |
| GS 乘子链 | **结构性发散** | w² 律逐层翻倍实测；min\|σ(s₀)\| = 0.847 破坏经典前提；层 2 整数性失败 |
| 密钥提取的归约 | **完成** | (γ, J) 决定 s₀ 至 ±ζ^j；dim-256 扭曲 uSVP gap ≈ 1.01，需 β ≈ 250+ |
| 等价密钥 {sk, ±sk} | **未达成** | 两条出路：提升方程整数小解（§7）或 gap-1.01 uSVP（§9.1），均超出当前算力 |
| 伪造 | 未尝试（依赖密钥） | 端局机械（Z[x] 因子分解 35 s、公钥预言机验证）已就绪 |
| 设计层（论文构造） | **未攻破** | 正确方向下全部已测信道关闭（§9.2） |

## 11　下一步工作

1. **Coppersmith/Cohn–Heninger 构造**（主线）：对 A² − yB² = N 设计移位
   多项式格。设计 + 实现预计一个专注工作日；端局已验证。
2. **Lenstra–Silverberg 张量代数的可计算实例化**：定理 1.1 声称无条件
   确定性多项式时间；把分次张量表示落地为可实现的系数运算是明确的开放
   问题（经典乘子链已判定发散，这是 GS 路线仅存的严肃出口）。
3. **更强归约算力**：gap ≈ 1.01 的 dim-256 扭曲 uSVP 需 β ≈ 250+，当前
   技术之外；若实现（GPU 筛分级），密钥提取与塔路线全链闭环。
4. **披露协调**：§2 的两行修复建议先行通报发布方（授权渠道已建立）。

## 12　复现指南

代码按战役阶段组织在 `code/`（脚本均为可独立运行的 Python/C，入口处有
用法注释）；文档在 `docs/`：

- 方向反转证据链：`harness_irs.c`（隔离 IRS 转储与方差实验）、
  `diff_kernel.c`（采样管线位级一致性）、`harness_sign.c` /
  `harness_forge.c`（签名/伪造闭环）；
- γ 恢复：`harness_corr*.c`（互相关谱累加，corr5/corr6 为已弃用版本，
  工程事故记录见 docs/shuttle.md 第十四次补充）、`recover.py`、
  `ein_analysis.py`（跨键仿射算子）；
- 理想与塔：`exact_ideal.py` / `cross_ideal.py`（理想恒等式验证）、
  `prog_bkz.py` / `ladder.py` / `tour.py`（归约阶梯）、`tower_step1.py`
  （dim-128 端到端）、`tower_check2.py`（校验）；
- 提升方程：`bm_lift.py`、`phaselift.py`（PhaseLift）、`lm_lift.py`、
  `dfs_lift.py`、`gcd_test.py`（排除实验全套）；
- GS 路线：`gs_chain.py`（链实现）、`gs_end.py` / `gs_end2.py` /
  `gs_endtest.py`（端局）、`gs_A.py`、`gs_validate*.py`（真值验证）、
  `gs_design.md`（设计文档）、`ideaux_*.gp` / `pari_input.gp` /
  `tlu.gp`（PARI 参考实现的判定实验）。

`supplementary/` 收录主线之外的完整求解器族（十四族相位求解器）与带日期
战役日志；`docs/shuttle.md` 十八次补充纪要与日志逐条对应。秘密材料与
大规模签名数据（.bin/.npy 工件，约 30 GB）不入库；数据由 harness 对官方
提交包实现再生成（官方工件已与审计副本逐字节核对一致）。fplll 补丁版
构建链与远端环境注意事项见第十四、十五次补充的工程记录。

## 13　边界与不主张

不推翻 Gärtner 2024/2052 与 Shuttle 2026/1991 的任何定理（两者的数学与
伪代码经我们独立重推与原文核对均正确）；不主张已实现等价密钥恢复（缺口
已定界为 gap-1.01 的 dim-256 扭曲 uSVP 或提升方程，均超出当前算力）；
GS 发散判定基于尺度递推恒等式与真值验证的链运行（层 0/1 精确通过、w² 律
与实测吻合）——是数学性质而非实现缺陷的推诿；区分器与泄漏不直接构成
UF-CMA 伪造。对提交物的全部否定性结论（§2、§10 前两行）独立成立，不依赖
文献判断。设计层未破的判定（§9.2）限定于我们已测的信道与已执行的攻击
面，不构成对设计本身的安全性证明。

## 参考文献

* L. Gärtner. *Compact Lattice Signatures via Iterative Rejection Sampling*.
  CRYPTO 2025（ePrint 2024/2052）.
* Shuttle 设计论文. ePrint 2026/1991（ASIACRYPT 2026 版）.
* L. Ducas, A. Durmus, T. Lepoint, V. Lyubashevsky. *Lattice Signatures and
  Bimodal Gaussians*. CRYPTO 2013.
* C. Gentry, M. Szydlo. *Cryptanalysis of the Revised NTRU Signature Scheme*.
  EUROCRYPT 2002（GS 算法）.
* A. Lenstra, M. Silverberg. *Reconstruction of a Number from Some Values of
  its Functions*. ePrint 2014/430（L-S 张量代数路线与唯一性理论）.
* H. Cohn, N. Heninger. *Approximate Common Divisors via Lattices*.
  ANTS 2012（Coppersmith 型构造的参考）.
* P.-M. Pellet-Mary, T. Espitau 等 PARI ideaux 参考实现（GS 实现参照与
  精度判定的对照物）.
* BLISS 泄漏家族：Flush-Gauss-Reload [GHLB+16]、Espitau 等 [EFGT17]、
  Pessl [Pes16]——本发现的信道从微架构侧信道迁移到签名文本，为同族新成员.
* Shuttle 提交包（NICCS 首轮）规范书与全部实现源码.
