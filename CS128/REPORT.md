# CS-128 学习型密钥恢复的量化壁垒：对称接受采样下的信息测定

> 撰写日期：2026-10-19　作者：GLM-5.3 Max、JY Liu。本目录不含独立代码：全部实验脚本位于仓库顶层
> [umodel/](../umodel/)（cs_* 前缀），运行方式见
> [code/README.md](code/README.md)。框架的完整方法学见
> [../umodel/ARCHITECTURE_v17v18.md](../umodel/ARCHITECTURE_v17v18.md)。

**摘要**：CS-128 是 NICCS 首轮征集的 BLISS 式格签名候选，其签名响应使用分块
共享的双模符号位。我们研究了仅凭公开签名的学习型密钥恢复，结论是一个
**量化的否定**：方案的潜变量结构真实存在——若观察者拥有完美的符号标签，
7.6 万条签名足以精确恢复密钥（oracle 实验）；但对称拒绝采样把标签信息压到
接近零——实测标签准确率 52.6%（抛硬币水平），半分相关 −0.0002 落在零假设
区间内。信息并非严格为零：精确接受模型给出每签名 +0.180±0.023 nats 的
对数似然比（8σ 显著），外加 0.19%/块的范数边界确定性揭示；由此，贝叶斯
最优分类器需要约 2×10⁶ 条签名，实际可达的标签器需要约 3×10⁸ 条。二阶矩
路线被独立关闭：实测估计量方差给出所需样本量 4.0×10¹⁰（中位）至
4.7×10¹²（p90）。CS-128 的安全性完全押在接受采样的对称性上——这不是一句
定性评语，而是被逐通道测定过的事实。

**Abstract**: CS-128 is a BLISS-style lattice signature candidate with blocked
shared bimodal sign bits. We study public-data-only learning attacks and obtain
a quantified negative result. The latent structure is real: with perfect sign
labels, 76,000 signatures suffice for exact key recovery (oracle experiment).
But symmetric rejection sampling hides the labels: measured label accuracy is
52.6% (coin level) and the split-half correlation −0.0002 lies within the null.
The information is not exactly zero—an exact acceptance-model LLR of
+0.180±0.023 nats per signature (8σ), plus a 0.19%-per-block norm-bound
revelation—so a Bayes-optimal classifier would need ~2×10⁶ signatures and any
practically constructible one ~3×10⁸. The second-moment route is separately
closed: measured estimator variances give required sample sizes of
4.0×10¹⁰ (median) to 4.7×10¹² (p90). The security of CS-128 rests entirely on
the symmetry of its acceptance sampling—and this is a measured fact, not an
opinion.

## 1　对象与问题设定

CS-128 的签名侧在分块共享的双模符号位上运行（BLISS 一族的标准形态——
正是 BiT-128 被攻破的同族结构），并以对称的拒绝采样保证发布分布的边缘
不可区分性。问题因此明确：**双模符号位的标签 b 在多大程度上可以从公开
签名中读出？** 如果标签可读，BiT 式的标记最小二乘即可恢复密钥；如果不可
读，需要回答差多少。

研究工具是 umodel 通用恢复框架（顶层目录）：SchemeSpec 声明式地登记环
代数、掩码模型、通道清单与公钥关系，引擎与解码阶梯只读规范。框架的适用
性闸门（check_applicability）对未推导的组合显式拒绝并给出理由——本报告
的否定结论全部经由该闸门产出，而不是跑出噪声后人工解释。

## 2　正结果：潜变量结构存在

oracle 臂给出结构性答案：若标签 b 由插桩侧车提供（审计数据，攻击者不可
用），坐标误差语义下 7.6 万条签名即达恢复门限，解码语义下约 4.8×10⁵ 条
——v14 的 oracle 闭合实验在 50 万条新鲜 dump 上以 oracle 块符号完成了
完整密钥恢复（分块 WLS → σ ≈ 0.195 → 窗口解码 → 全解验收，对真值逐项
通过）。这些数字的意义是**信息论参照系**：CS-128 的双模结构本身不提供
安全性，安全性只能来自标签的不可读性。这与 BiT-128 形成直接对照——同族
结构、同样的 oracle 通道，BiT 的标签准确率 96.19% 而此处 52.6%。

## 3　卡点：对称接受采样

实测（20 万条真实签名，参考实现插桩 dump）：

- 标签准确率 52.6%；
- assay 半分相关 −0.0002，零假设预期 +0.0005——公开统计量中无可检测的
  b 相关信号。

框架闸门据此拒绝标签路线。此处值得记录一个方法论细节：早期我们曾用一条
z₁ 回归统计量得出"信号饱和"的结论，后来在 BiT 上发现同类统计量的构造
伪象（见 BiT128/REPORT.md §8）；对 CS-128 我们改用多通道测定（§4–§5），
避免单一统计量的判决。

## 4　信息测定：不是零，是多少

精确接受模型（对拒绝采样条件化的逐块似然）给出严格量化：

- **LLR 信息**：+0.180±0.023 nats/签名（8σ）。对称接受并不完美——这是
  一个非零但极弱的信号。
- **范数边界揭示**：0.19%/块的确定性揭示，构成第二条信息源。
- **严格 (z₀, z₁) 通道信息**：+0.122±0.032 nats/签名；y₂ 层边缘化后，
  总信息被括在 [0.12, 0.18] nats/签名。

换算为所需签名数：

| 分类器 | 所需签名数 |
|---|---|
| 完美标签（oracle 参照） | 7.6×10⁴ |
| 贝叶斯最优（理论极限） | ~2×10⁶ |
| 实际可达标签器 | ~3×10⁸ |

"实际可达"与"贝叶斯最优"之间两个数量级以上的差距，正是对称接受采样的
安全余量所在：构造接近最优的分类器本身需要密钥信息——鸡生蛋结构。

## 5　二阶矩路线的独立关闭

不经标签、直接用二阶统计量估计秘密的路线被实验精确关闭：实测估计量方差
给出 required_N = 4.0×10¹⁰（中位）/ 4.7×10¹²（p90）；同一引擎的掩码
模型给出掩码受限口径 5.9×10⁸。三组数字同向，二阶矩路线在可预见的数据
量下无望。

## 6　实现层核查与一次更正

参考实现的打包格式（rANS + SM3 CTR-DRNG）被按位移植为纯 Python 解码器，
对捕获的签名整包验证 200/200——所有统计测定建立在正确的字节语义上。

我们曾报告参考实现的一处"对齐移位未定义行为"，**该发现已撤回**：那是
我们插桩 printf 的格式串错位，参考二进制中移位计数从未超过 2。真实的
实现偏离是定点舍入，已按位移植（对捕获的接受检验值 240/240 位精确），
且不改变信息界：实际实现 LLR +0.182±0.028，对意图模型 +0.180±0.023。
如实记录，因为它说明插桩本身的误差可以伪装成实现缺陷——审计者应当对
自己的仪器保持与对被审计对象同等的怀疑。

## 7　判定语义

本报告的结论形态是"拒绝 + 理由"，不是"跑完无结果"。umodel 的判定矩阵
把四类输入区分开：BiT-128（信号存在，恢复完成）、CS-128（信号存在但
极弱，本文）、COMPASS-128（信号被接受条件化消除，见其报告）、掩码受限
形态。区分"弱"与"零"不是修辞——前者随数据量增长终将可攻（3×10⁸ 条），
后者在任何数据量下都不可攻，二者的防御含义完全不同。

## 8　下一步工作

1. **逼近贝叶斯最优的标签器**。2×10⁶ 是理论底线；任何把标签准确率从
   52.6% 显著抬起的构造（例如联合多块的条件似然、范数揭示的显式利用）
   都会等比压缩所需数据量。这是 CS-128 上唯一有理论明确收益的进攻方向。
2. **新泄露通道**。学习路线的壁垒测定完毕后，剩余的攻击面在实现层
   （侧信道、故障）与格归约层（公钥本身）——前者超出公开数据约束，
   后者是方案安全声称的正面战场，均非本框架范围。
3. **接受对称性的形式化**。"对称接受采样把标签信息压到 ε"目前是测定值
   而非定理；给出 ε 的上界证明（或找到更紧的下界构造）是有独立价值的
   开放问题。

## 9　复现指南

见 [code/README.md](code/README.md)。需要强调数据边界：真实签名 dump
（插桩侧车含 b 真值）属本地审计数据，不入库；各脚本在数据缺失时自动
跳过对应演示并打印原因。公开可复现的部分包括：解码器正确性（对自生成
数据）、接受模型 LLR 的计算、所需样本量的预测管线与全部自测
（`../umodel/selftest.py` 29 项）。

## 参考文献

* L. Ducas, A. Durmus, T. Lepoint, V. Lyubashevsky. *Lattice Signatures and
  Bimodal Gaussians*. CRYPTO 2013（ePrint 2013/383）.
* V. Lyubashevsky. *Fiat–Shamir with Aborts*. ASIACRYPT 2009（拒绝采样
  对称性的构造源头）.
* CS 提交包：NICCS 首轮数字签名候选.
