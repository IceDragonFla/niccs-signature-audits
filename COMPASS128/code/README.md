# COMPASS-SIG-128 实验代码索引

COMPASS-128 的实验入口位于仓库顶层 `umodel/`。本目录只做索引。

```bash
python3 umodel/selftest.py        # 数值自测
python3 umodel/run_demos.py       # 演示管线；COMPASS 相关演示在缺本地
                                  # 审计 dump 时自动跳过并注明
```

## 入口与报告结论的对应

| 入口（umodel/） | 复现内容（COMPASS128/REPORT.md） | 数据需求 |
|---|---|---|
| `public_decoders.py`（COMPASS 16 位 z 分支） | §5 打包解码器与伪 XOF 发现：名为 shake256 的 XOF 实为 SM3 构造，按真实语义移植，整包验证 200/200 | 自生成数据可验证 |
| `run_demos.py` 的 COMPASS 演示 | §1 σ(N) 定律与 corr(ẑ, s₁) 的三尺度测定（5×10³/10⁵/1.2×10⁷） | 本地审计 dump |
| 框架适用性闸门（`spec.py` 的 check_applicability） | §4 判定矩阵：接受条件化消除原型的输出语义 | 自包含 |

## 数据边界

真实签名 dump 属本地审计数据，不入库。σ 定律与相关系数的测定需要
COMPASS-128 参考实现与固定密钥下的签名流（重放方式见 dump 生成脚本的
插桩说明，umodel 演示管线自动探测）；解码器与自测可完全公开复现。
