# Paper 长期成熟度运行手册

适用范围：Alpaca Paper。本文不授权自动实盘，也不把合成证据计入真实交易日。

## 每个预定交易日

1. 先用交易所日历版本登记预定交易日。
2. 交易日结束后写入一份不可变观察，引用每日闭环报告、研究质量报告和
   broker/local/PositionPlan 质量报告。
3. 即使当日缺数、失败、休市判断错误或无成交，也必须写观察；不得删除或
   跳过失败日。
4. 以下任一计数非零，当日即失败：
   - 未解释重复订单；
   - 持仓计划静默改写；
   - broker、本地与计划状态差异；
   - 未恢复故障。

`PaperMaturityStore` 对同一证据类型和日期实施不可变约束。REAL 与
SYNTHETIC 查询严格分离；相同日期的合成观察不能填补 REAL 缺口。

## 60 日门禁

报告从已经登记的预定交易日中取最近 60 日。通过条件：

- 恰有 60 个预定日；
- 每个预定日都有观察；
- 每日报告完整；
- 重复订单、计划改写、状态差异和未恢复故障均为零。

缺失观察会明确产生 `MISSING_OBSERVATION`，失败日不会被后续成功日覆盖。

## 故障演练

`PaperResilienceStore` 固定检查六种场景：

| 场景 | 预期 | 允许 submit |
| --- | --- | --- |
| 缺数 | BLOCKED | 0 |
| 超时 | BLOCKED | 0 |
| 重启 | RECOVERED | 0 个新提交 |
| 部分成交 | RECOVERED | 原始 1 个提交 |
| 休市 | NO_ACTION | 0 |
| Kill Switch | BLOCKED | 0 |

每个场景必须有审计引用和恢复引用；任何意外 submit 都使整个报告失败。

## 签收层级

- `ARCHITECTURE_READY`：允许使用严格隔离的 SYNTHETIC 门禁证明代码合同。
- `FINAL_REAL_READY`：成熟度和故障报告都必须是 REAL，且成熟度报告至少
  包含 60 个通过的真实预定交易日。

两个层级都保持 `live_authorized=false`。
