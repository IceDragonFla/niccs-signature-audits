#!/usr/bin/env python3
# patch_tex.py — apply the review-response edits to the report .tex (run from the report dir)
import io, sys

p = "DARTS_1-2b_攻击报告.tex"
src = io.open(p, encoding="utf-8").read()
applied = []

def rep(old, new, tag):
    global src
    if old in src:
        src = src.replace(old, new, 1)
        applied.append(tag)
    else:
        applied.append("MISS:" + tag)

# (6) hash-match vs full-verification note after the false-match paragraph
rep(r"""$W=q/2^d=127.75$，$c\,u$ 在 $\tau=30$ 个系数处移位 $\pm 1$）：""",
r"""$W=q/2^d=127.75$，$c\,u$ 在 $\tau=30$ 个系数处移位 $\pm 1$）：

\noindent\textbf{术语说明（哈希匹配 $\neq$ 完整验证）}。本文的``分支匹配''指挑战哈希
$H(w_1^{\pm},\mu)=c$ 是否成立（\code{label\_extract} 仅重放该哈希判定）；完整验证还需
通过范数检查 $\|(z_1,z_2)\|_2<B''$。两者仅在极端范数情形不同（$4\times10^6$ 条诚实签名
的``无匹配''计数为 0）。本文所有标签与闭合判据均基于哈希匹配。""", "hash-note")

# (7) known-message model after the task list
rep(r"""侧还是右侧，进而恢复密钥。""",
r"""侧还是右侧，进而恢复密钥。

\noindent\textbf{攻击模型（已知消息）}。签名本身不包含消息，而分支重放与挑战再生都需要
消息：攻击者须知道每条签名对应的消息及其编码。实验中每条签名对应消息为 4 字节小端序
计数器 \code{LE32(start+j)}（$j$ 为签名序号）；消息序列与起始偏移是复现实验输入的一部分
（\S\ref{sec:repro}）。""", "known-msg")

# (3)-a weaken the SSD-section claim
rep(r"\item \textbf{误差已达信息论下限}。去缩放后每坐标 rms $=0.088$，而干净标签",
    r"\item \textbf{误差已达当前估计器的噪声地板}。去缩放前每坐标 rms $=0.086$（原始 $q$ 单位），而干净标签", "fisher-1")
rep(r"Fisher 地板 $\sigma_y/\sqrt{N_{\mathrm{clean}}\cdot\tau}=0.0867$——比值",
    r"的 Fisher 界 $\sigma_y/\sqrt{N_{\mathrm{clean}}\cdot\tau}=0.0867$——比值", "fisher-2")
rep(r"1.02，估计器已最优。SSD 的增益在其偏差主导的场景；在噪声主导的场景，",
    r"1.01——当前架构下估计器已无改进空间（注意：这是``标记 LS $+$ 当前完成器''的经验表现，不是信息论极限；更低数据量仍可能通过联合译码实现）。SSD 的增益在其偏差主导的场景；在噪声主导的场景，", "fisher-3")

# (3)-b weaken the wrap-section claim
rep(r"而后者已达 Fisher 最优。10$^5$ 量级的希望",
    r"而后者已达到当前架构的噪声地板（经验 Fisher 界的 1.02 倍——经验表现而非理论极限）。10$^5$ 量级的希望", "fisher-4")

# (13) BKZ number fix in the wrap section
rep(r"按 2016 估计需 BKZ $\beta\gtrsim150$（$2^{60+}$ 核操作），非作业规模。",
    r"按 2016 年格估计启发式，$5\times10^5$ 条签名（偏移 $\approx8$）即需 BKZ-$\beta\approx400$（core-SVP $\sim2^{130}$），与无签名的主攻击同阶；即便 $2\times10^6$ 条签名也需 $\beta\approx300$——签名数据对格规约成本的削减有限，非作业规模。", "bkz-fix")

# (11) conclusion rewrite
old_c = r"""\textbf{可以主张}：在 DARTS-128（1-2b 修订版）上，仅使用公钥与每密钥 $2.0\times10^{6}$ 条公开签名
（低于安全证明中对应的查询复杂度项 $\sim2^{64}$），通过双分支哈希判定获取确定性
双峰符号标签（覆盖率 $21\%$、零误差），经 $\zeta$ 域标记最小二乘与拒绝采样条件化收缩的
矩估计修正，以纯 CPU 在分钟级\textbf{精确恢复了全部秘密三元组 $(s_0,s_1,e)$}——恢复值与
真实密钥逐字节一致，并用未修改的参考验证器以恢复密钥完成 20/20 新消息伪造。攻击不需要
格规约、侧信道或插桩。"""
new_c = r"""\textbf{可以主张}：在 DARTS-128（1-2b 修订版）上，在``公开分支标签 $+$ 频域最小二乘
$+$ 三元完成器''的攻击架构下，两把独立测试密钥均表明约每密钥 $1.4$--$2.0\times10^{6}$
条已知消息签名（低于安全证明中对应的查询复杂度项 $\sim2^{64}$）即可实现可靠恢复：通过
双分支哈希判定获取确定性双峰符号标签（覆盖率 $21\%$、零误差），经 $\zeta$ 域标记最小
二乘、拒绝采样条件化收缩的矩估计修正与恒等式完成器，\textbf{精确恢复全部秘密三元组
$(s_0,s_1,e)$}——恢复值与真实密钥逐字节一致，并用未修改的参考验证器完成 20/20 新消息
伪造。攻击不需要格规约、侧信道或插桩。\emph{该数值是当前估计器与完成器的经验工作点，
不是信息论下限}；更低数据量仍可能通过联合三元模约束译码（\S6.6）或更好的似然估计实现。"""
rep(old_c, new_c, "conclusion")

# (10) 2^64 in the second 界定 item
rep(r"\item 攻击数据模型为已知消息 $+$ 大量签名。由于 $2.0\times10^{6}\ll2^{64}$，这不是边界外构造，",
    r"\item 攻击数据模型为已知消息 $+$ 大量签名。由于 $2.0\times10^{6}$ 低于安全证明中对应的查询复杂度项，这不是边界外构造，", "q64-2")

# (2) the scaling-table provenance + the unified numbers
old_t = r"""主实验的零误差数据点为每密钥 $2\times10^6$。逐系数舍入错误数随 $N$ \textbf{指数
衰减}（实测三个数据点，每倍增约 2.1 bit）：

\begin{center}\small
\begin{tabular}{@{}lccc@{}}
\toprule
签名总量 $N$ & $2\times10^4$ & $5\times10^5$ & $2\times10^6$ \\
\midrule
LS 舍入后错误系数（去缩放后） & 173 & 21 & 0--1 \\
每坐标 rms & 0.56 & 0.24 & \textbf{0.117} \\
\bottomrule
\end{tabular}
\end{center}"""
new_t = r"""主实验的零误差数据点为每密钥 $2\times10^6$。用\textbf{最终版估计器}（$\zeta$ 域标记
LS $+$ 逐路线矩估计 $\hat\rho$ 去缩放）在主密钥的同一签名流前缀上重测的统一数据：

\begin{center}\small
\begin{tabular}{@{}lcccccc@{}}
\toprule
签名总量 $N$（全部为已知消息） & $5\times10^4$ & $10^5$ & $2\times10^5$ & $5\times10^5$ & $10^6$ & $1.75\times10^6$ \\
\midrule
去缩放后每坐标 rms & 0.79 & 0.53 & 0.37 & 0.25 & 0.17 & \textbf{0.13} \\
舍入错误数（$s_0{+}s_1$ 合计） & 614 & 524 & 403 & 342 & 314 & \textbf{0} \\
干净标签数 $N_{\mathrm{clean}}$ & $1.0\times10^4$ & $2.1\times10^4$ & $4.2\times10^4$ & $1.05\times10^5$ & $2.1\times10^5$ & $3.7\times10^5$ \\
\bottomrule
\end{tabular}
\end{center}

\noindent 溯源说明：本表全部数据来自\textbf{同一次测量批次}（最终版 \code{attack.py}、
逐路线矩估计 $\hat\rho$ 去缩放、$s_0{+}s_1$ 合计错误数、\code{DARTS\_TRUTH} 评分）；
\emph{未}混入早期运行（早期运行的 $\hat\rho$ 拟合在 $N\le2\times10^5$ 时回退到常数，
且历史表中 $5\times10^5$ 行的 21 指的是 $s_0$ 单分量，34 为 $s_0{+}s_1$ 合计，二者
不可直接比较）。"""
rep(old_t, new_t, "table-prov")

# (2)-b the closure-curve table provenance
rep(r"""\noindent 优化后的闭合曲线（主密钥实测，干净测量——修复了前缀测量中
SIGPIPE 截断与陈旧输出文件的两个假象）：""",
r"""\noindent 优化后的闭合曲线（主密钥实测；错误数由评分脚本从保存的估计文件计算，
闭合结果以``未修改验证器 20/20 $+$ 与真值逐字节一致''为准；修复了前缀测量中
SIGPIPE 截断与陈旧输出文件的两个假象）：""", "closure-prov")

io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("applied:", applied)
