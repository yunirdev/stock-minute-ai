# Paper 迁移签收

日期：2026-07-27  
当前状态：`ARCHITECTURE_READY`  
最终状态：等待 60 个 REAL 预定交易日，尚非 `FINAL_REAL_READY`

验证基线：392 tests passed；Ruff passed；compileall passed。

## 调用图

```text
ResearchSnapshot
  → TradingAgentsResearch
  → CandidatePlan → FinalTradePlan → RiskEvent
  → OrderIntent → BrokerOrder → Fill
  → PositionPlan → TradeEpisode → EpisodeReview
  → StrategyCandidate → PromotionEvidence
  → PaperMaturity → PaperMigrationSignoff
```

## 已完成架构

- 预定交易日与每日观察分表，缺失日和失败日不可跳过。
- REAL/SYNTHETIC 成熟度证据严格隔离。
- 60 日正门禁、缺失日/失败日负门禁和不可变冲突测试通过。
- 缺数、超时、重启、部分成交、休市、Kill Switch 六类故障矩阵完成。
- 最终签收必须同时引用成熟度、故障、闭环、全量验证、文档、调用图和限制。
- `FINAL_REAL` 使用任何 SYNTHETIC 证据都会 fail-closed。
- Runtime 已自动登记/冻结 REAL 日证据，并在美东 20:00 后生成经
  CHECKPOINT、哈希和只读打开验证的 trade/AI 数据库备份。
- 成交关闭后的 EpisodeReview/StrategyCandidate、31 个 UI 动作审计及
  Discord 授权网关均已连接生产调用方。

## 当前限制

- I01 的 60 个 REAL 预定交易日必须随 Runtime 自然积累。
- D04 和 E06 的自然观察窗口仍独立积累。
- 架构签收不授权 live，不承诺盈利，也不允许跳过失败日。
