# Paper 闭环交付报告

日期：2026-07-27  
状态：当前 Alpaca Paper 闭环小目标已签收  
范围：Alpaca Paper；不授权自动实盘交易

## 已形成的闭环

```text
ResearchSnapshot
  → TradingAgents research run
  → CandidatePlan
  → FinalTradePlan + deterministic risk
  → OrderIntent
  → Alpaca Paper LMT + fill
  → PositionPlan
  → trade episode
  → frozen EpisodeReview
  → immutable strategy candidate
```

`trader.closed_loop_delivery.ClosedLoopDeliveryStore` 冻结上述阶段引用、Paper
场景、量化指标、恢复演练、按钮覆盖和已知限制。缺任一阶段引用、必需指标或
恢复证据时拒绝签收；策略候选不能直接修改 Runtime 参数。

## H 阶段证据

| 项目 | 结果 |
| --- | --- |
| H01 可观察性 | 31 个动作契约；SUCCESS/EMPTY/ERROR/BUSY；订单可追溯到来源、快照、研究、计划版本、风险、订单和成交 |
| H02 Discord | 默认不外发；授权、dry-run、去敏、失败和幂等审计 |
| H03 恢复 | DuckDB 哈希备份、只读校验、恢复到新目录；API/DB/Runtime 故障分类演练 |
| H04 Paper 闭环 | 隔离演练覆盖 BUY、SELL、拒绝、部分成交、UNKNOWN、重启零重复提交，并冻结完整阶段引用 |
| H05 NiceGUI | 31 个动作四状态自动合同已覆盖；交易记录页已展示最新订单完整解释；只读动作/订单解释 API 已真实启动验证；Web 默认仅绑定 `127.0.0.1`；用户浏览器截图确认 EMPTY 状态完整渲染 |
| H06 签收 | 报告、调用图、量化指标、完整回放、按钮报告、恢复演练与浏览器页面证据已签收 |

## 必需量化指标

每份闭环证据必须包含数据覆盖率、研究成功率、计划数、下单成功率、成交率、
滑点、最大回撤和已实现收益。盈利不是通过条件；所有数值必须有限且与证据
类型（`ISOLATED_PAPER` 或 `REAL_PAPER`）一起持久化。

## 已知限制

- 自动执行仅限 Alpaca Paper LMT；UI 不是第二下单入口。
- D04 的 20 个真实研究日和 E06 的 30 个真实持仓日仍需自然积累；合成证据
  不计入真实观察日。
- 新 Discord 报告默认禁止外发，必须显式授权。
- 浏览器截图验收时真实审计库没有订单，因此页面正确显示“暂无可解释订单 ·
  缺少 order”；有成交链的正向渲染由隔离 Paper 数据与自动化测试覆盖。
