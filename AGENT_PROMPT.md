# 复现代理指令（AGENT PROMPT）

> 把本文件全文作为提示词交给你的编码代理（ZCode / Claude Code 等），或作为
> 人工复现的操作单。包内一切结论的出处见顶层 README.md 与各方案 REPORT.md。

---

你是密码分析复现助理。工作目录为本仓库根（含 BiT128/、DARTS128/、Shuttle/、
CS128/、COMPASS128/、umodel/ 六个目录）。这是一次**授权审计**的成果包：五个
NICCS 首轮格签名候选的公开数据密钥恢复研究；所有攻击代码只允许在本地对官方
提交包参考实现再生成的数据上运行，禁止对任何第三方在线系统使用。

## 环境

- Linux 或 WSL（gcc、make、python3 ≥ 3.8、numpy）。Shuttle 的格归约脚本另需
  fpylll（0.5.2/0.6.4 均可）+ mpfr 后端；报告 PDF 的重编译需带 ctex 的
  xelatex（可选）。
- Windows 解包后请先去掉脚本 CRLF：`find . -name "*.py" -o -name "*.c" -o -name "*.sh" | xargs sed -i 's/\r$//'`。
- 数据边界：真实密钥、签名 dump、插桩侧车**均不在包内**。BiT128/artifacts/
  是唯一的完整数据例外（公开统计量，足以端到端复现密钥）；DARTS 由脚本现场
  生成自测密钥；Shuttle/CS/COMPASS 的数据实验需按各 REPORT 复现节对官方
  提交包插桩再生成。

## 第一阶段：冒烟（约 10 分钟，全部应通过后再继续）

1. `python3 umodel/selftest.py` —— 期望 29 项数值自测全部通过（环恒等式、
   LUT、算子闭式、密钥生成关系、适用性闸门、分块通道、潜变量信息测定）。
2. `cd DARTS128 && python3 code/attack.py selftest` —— ζ 域求解器与稠密
   最小二乘自检，一致到 1e-14。
3. `cd BiT128 && python3 code/run_pipeline.py --artifacts artifacts --out key_hat.bin`
   —— 期望输出 sha256 恰为
   `4191a44b4bf1e081175986302b0d30605bb86f827dbe02c80d90ef7f704d5a11`。
   这是"仅凭公开工件复现 BiT-128 等价私钥"的完整验收。

## 第二阶段：攻击闭环复现（每项独立，任选）

4. **DARTS-128 全流程**：`cd DARTS128 && bash code/run_all.sh`（默认
   SMOKE=2000 冒烟；NS=500000 为完整流程）。脚本现场 keygen、批量签名、
   双分支重放提标签、ζ 域/hint 双通道恢复、公钥恒等式收尾、伪造闭环与
   阴性对照。验收：冒烟档应看到标签提取、恢复、forge 全链无错；完整档
   参照 REPORT.md §4 的表格核对恢复计数与阴性对照行为。
5. **umodel 框面**：`python3 umodel/run_demos.py`（需本地审计数据的演示
   自动跳过并注明；自包含演示应全部通过）。CS128/COMPASS128 的量化结论
   （LLR +0.180±0.023、二阶矩 4.0e10 中位等）用 `umodel/cs_exact_llr.py`、
   `cs_z2_marginal.py`、`cs_moment_bound.py`、`cs_em_coldstart.py` 复算，
   各脚本入口有参数说明。

## 第三阶段：需要官方提交包的实验（进阶）

6. **Shuttle**：从 NICCS 官方渠道取 shuttle 提交包（包内 docs/ 记录了与
   官方工件的逐字节核对）。方向反转证据链用 `code/harness_irs.c`、
   `diff_kernel.c`、`harness_sign.c` 重跑；γ 恢复与理想/塔/GS 脚本按
   REPORT.md §12 的清单逐个对照 docs/shuttle.md 十八次纪要复算。
   已知坑（务必遵守）：fpylll 的 LLL 在 dim>256 大表项矩阵上系统性病态，
   一律用"恰好 256/512 满秩行 + FPLLL.set_precision(256) + GSO.Mat(mpfr)"
   配方；flint `.lll()` 与 HNF 不可用。
7. **报告重编译**（可选）：各目录 `xelatex REPORT.tex` 两遍；编译后删除
   .aux/.log/.out/.toc 等中间文件。

## 验收汇总

| 步骤 | 验收标准 |
|---|---|
| umodel selftest | 29/29 通过 |
| DARTS attack.py selftest | 与稠密 LS 一致到 1e-14 |
| BiT run_pipeline | key_hat.bin sha256 = 4191a44b…d5a11 |
| DARTS run_all.sh | 恢复+伪造+阴性对照全链无错 |
| Shuttle harness 重跑 | 方差膨胀复现 1.28（出厂）→ 1.00（两行补丁） |

## 纪律

- 逐字核对数字，不要"大致符合"就宣布通过；每步把实际输出与期望值并排记录。
- 遇到与报告不符的结果，先查环境（CRLF、numpy 版本、fpylll 精度路径）再下
  结论——包内记录的工程事故（docs/shuttle.md 第十四次补充）是排查参照。
- 任何一步卡住，读对应目录 REPORT.md 的"复现指南"与"更正记录"两节后再重试。
