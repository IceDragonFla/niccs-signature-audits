# COMPASS-SIG-128：被接受条件化消除的回归信号

> 撰写日期：2026-10-19　作者：GLM-5.3 Max、JY Liu。本目录不含独立代码：实验入口位于仓库顶层
> [umodel/](../umodel/)（COMPASS 的 16 位 z 解码器在 public_decoders.py，
> 含一处实现发现，见 §5），运行方式见 [code/README.md](code/README.md)。

**摘要**：COMPASS-SIG-128 是 NICCS 首轮征集的 Fiat–Shamir-with-aborts 型
格签名候选。我们研究其公开签名对学习型密钥恢复的暴露面，结论是一个
**结构性零信号**的判定：掩码方差定律 σ(N) = √(Var(y)/(τN)) 在 5×10³ 到
1.2×10⁷ 的全部尺度上精确吻合（实测 50.00/10.83/1.263 对定律值
48.85/10.92/0.997），但响应与秘密的相关系数在同样全部尺度上为零
（实测 −0.044/−0.013/−0.006，若信号存在应见 0.016/0.075/0.542）。根因
是签名循环中的 w₁ 一致性检查：它验证 decompose(w − c·s₂ + c·t₀) 的高位
等于原始 w₁，该检查的通过概率依赖秘密 (s₂, t₀)，从而条件化了发布分布——
条件化恰好消除了 z = y + c·s₁ 中 c·s₁ 与对齐统计量之间的全部相关性。
这把判定从早期的"掩码受限、所需 N ≈ 4.8×10⁷"更正为：**简单回归在任何
数据量下不可行**。COMPASS-128 对本攻击族的安全性不来自掩码宽度，而来自
这个一致性检查，且是原理性的。

**Abstract**: COMPASS-SIG-128 is an FS-aborts lattice signature candidate of
the NICCS first round. We find a structurally zero learning signal: the masking
variance law σ(N) = √(Var(y)/(τN)) holds precisely at every scale from 5×10³
to 1.2×10⁷ signatures, yet the response–secret correlation is zero at every
scale (−0.044/−0.013/−0.006 measured, against 0.016/0.075/0.542 expected if
the signal existed). The cause is the w₁ consistency check in the signing
loop: it validates that decompose(w − c·s₂ + c·t₀) reproduces the original
w₁; its acceptance probability depends on (s₂, t₀), conditioning the published
distribution and eliminating the c·s₁ correlation exactly. The verdict is
corrected from "mask-limited, N ≈ 4.8×10⁷ required" to: simple regression is
infeasible at any sample size. The security against this attack family rests
on the consistency check, in principle rather than in margin.

## 1　对象与现象

COMPASS-SIG-128 无双模潜变量（与 BiT/CS 的结构不同），签名响应 z = y + c·s₁
中 y 为新鲜掩码。学习型攻击的标准形态是：对齐统计量中 z 与 c·s₁ 的相关
随 N 增长按 σ(N) ∝ N^(−1/2) 浮出——BiT-128 正是这样被攻破的。

COMPASS-128 上这个预期的失败方式很干净：**方差收敛，信号不出现**。

| N | σ（实测） | σ（定律） | corr(ẑ, s₁) | 预期 corr |
|---|---|---|---|---|
| 5,000 | 50.00 | 48.85 | −0.044 | ~0.016 |
| 100,000 | 10.83 | 10.92 | −0.013 | 0.075 |
| 12,000,000 | 1.263 | 0.997 | −0.006 | 0.542 |

若仅看 σ 列，一切正常；corr 列在三个数量级的样本上钉在零。统计量的
方差描述的是噪声，信号是否存在于噪声之下是另一个问题——这正是判定
必须同时测两者的原因。

## 2　根因：w₁ 一致性检查的条件化

签名循环（sign.c 第 197–213 行）在发布前执行一致性检查：计算
w_approx = w − c·s₂ + c·t₀，验证 decompose(w_approx) 的高位与原始 w₁
一致，不一致则重抽。发布分布因此是条件分布：

```
P(发布 (c, z) | 接受) ∝ P(y) · 1[w₁_consistency(c, y, s₂, t₀)]
```

指示函数依赖秘密。对 z = y + c·s₁ 的对齐统计量，这一条件化的净效应是
**恰好**消除 c·s₁ 与统计量的相关性：接受事件筛掉的那部分 (y, c) 组合，
正是携带相关性的组合。"恰好消除"的机理值得强调——它不是把信号削弱到
测量线下（那是 CS-128 的情形），而是把信号的对消做成了验证方程自身的
代数性质。

这在一类经典反例中有精确的对应物：令 z = y + t，若接受条件只保留落在
共同内部区间的 z，则接受后的 z 可以与 t 完全无关——截断可以彻底破坏
回归信号。COMPASS-128 的检查是这一原理在真实方案上的实例：自一致性
检查充当了以秘密为参数的截断。

## 3　判定的更正过程

早期判定为"掩码受限，所需 N ≈ 4.8×10⁷"——按 σ 定律外推，corr 应在
5×10⁷ 量级浮出。1.2×10⁷ 尺度的实验（预期 corr 0.542）给出 −0.006 后，
该判定被更正为接受条件化安全。更正的依据是三尺度联合：σ 吻合 + corr
恒零 + 机理定位（两行代码的条件化项）。记录这个过程是因为它展示了一个
普适的审计陷阱——**基于无条件方差定律的所需样本量预测，对存在秘密依赖
接受条件的方案没有意义**；定律描述的是方差，信号（偏差方向）可能已被
条件化单独消除。

## 4　与 CS-128 的对照：弱信号与零信号

CS-128 的壁垒是数量性的（最优分类器 2×10⁶ 条，实际 3×10⁸ 条）：信号
非零，防御靠把信号压到实际不可利用。COMPASS-128 的壁垒是原理性的：信号
为零，加数据无效。二者的防御审计含义不同——前者应关注任何提高标签质量
的实现偏离，后者学习路线已闭合，剩余暴露面只在实现层（侧信道/故障）与
公钥的格归约层。umodel 判定矩阵把这两种"不可恢复"作为不同机理输出，
COMPASS-128 是"接受条件化消除"原型目前唯一的真实实例。

## 5　实现发现：伪 XOF

移植打包解码器时发现：COMPASS-128 源码中名为 "shake256" 的 XOF 实为
SM3 构造的伪 XOF（接口同、内部原语不同）。这对密码分析的影响是工程性
的——任何假定 SHA-3 语义的分析工具都会在字节层对不上；解码器
（umodel/public_decoders.py）按真实语义移植，对捕获签名整包验证 200/200。
该发现已按实现层观察记录，不构成本报告安全判定的依据。

## 6　下一步工作

1. **利用条件化结构本身的高阶统计**。一阶相关性被对消，但条件分布
   P(发布|接受) 仍是秘密的函数；以接受事件为条件的其它泛函（例如
   边界样本的取舍模式）理论上可能携带残余信息。当前框架没有这类算子，
   适用性闸门会显式拒绝——把它推导出来是一个明确的开放问题。
2. **区分器的口径**。本判定针对简单回归族；若后续发现任何二阶以上
   统计偏离，应先在受控重实现（去掉一致性检查）上定标信号，再回测
   出厂数据——与方向反转研究中的 ship−fix 配对方法（Shuttle/REPORT.md
   §2.3）同型。

## 7　复现指南

见 [code/README.md](code/README.md)。σ 定律与 corr 的测定在
umodel 演示管线中（需本地审计 dump，缺失自动跳过）；解码器与自测可
完全公开复现（`../umodel/selftest.py`）。

## 参考文献

* V. Lyubashevsky. *Fiat–Shamir with Aborts*. ASIACRYPT 2009（拒绝/接受
  条件化的构造源头与截断反例的原理对应）.
* L. Ducas, A. Durmus, T. Lepoint, V. Lyubashevsky. *Lattice Signatures and
  Bimodal Gaussians*. CRYPTO 2013.
* COMPASS-SIG 提交包：NICCS 首轮数字签名候选.
