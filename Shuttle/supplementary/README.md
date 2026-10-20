# Shuttle 补充材料说明

本目录收录主线报告（../REPORT.md §11）之外的两类战役工件，全部为文本文件
（脚本与日志），不含任何密钥材料或大规模数据。

## solvers/ —— 相位搜索战役的完整求解器族

../REPORT.md §3 记录了"十四族求解器全部止步伪整数盆地"的结论；本目录
补齐 code/ 未包含的其余求解器与诊断脚本，使该结论可被逐族核对：

| 脚本 | 路线 |
|---|---|
| `ang_sync.py`, `ang_sync2.py` | 角同步（GN） |
| `ap_clean.py`, `ap_oracle_test.py` | 精确模长交替投影（AP） |
| `affine2.py`, `affine3.py`, `affine_test.py`, `anchor_test.py` | 跨键仿射模型族与锚点诊断 |
| `deep2.py`, `deep_search.py` | 深搜索 |
| `fp_ils.py`, `fp_ils2.py`, `gn_ils.py` | 格 ILS（GN 线性化、LLL+nearest-plane） |
| `strata.py`, `strata2.py`, `strata3.py` | 挑战调制分层解剖 |
| `bilinear_check.py`, `conj_check.py`, `cost_check.py`, `kern_diag.py` | 结构核校验 |
| `post4m.py`, `post5m.py`, `final_stack.py` | 数据冲刺后处理与合并 |
| `sa.c` | C 版模拟退火（3×10⁷ 步） |
| `flagval.py` | y-flag 耦合项数值检查 |

配合 code/ 中的 `theta_opt.py`、`two_comp.py`、`bm_lift.py` 等即构成完整清单。

## logs/ —— 带日期的战役日志

归约攻坚（`bkz_*`、`pbkz_*`、`lad*`、`tour*`、`fix.log`）、隔离 IRS 实验与
互相关谱累加（`einR*`、`corrA/B`）、数据冲刺批次（`camp0`–`camp7`、`mk1`–
`mk3`）的原始输出。它们是 ../docs/shuttle.md 各次补充纪要中数字的一手
出处；工程事故（corr5/corr6 弃用、camp1 作废）也在其中留有痕迹。
