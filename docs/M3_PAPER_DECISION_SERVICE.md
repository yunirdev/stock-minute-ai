# M3 PaperDecisionService 与 UI 解耦

## 决策链路

Runtime 提供 bars、broker positions、UniverseProvider 快照、策略统计和已验证 AI snapshot 给 `PaperDecisionService`。服务只输出可序列化 `StrategyDecision`，随后仍使用现有 `ATRPlanner`、`EqualWeightAllocator`、AI 安全门、`RiskEngine` 和 M2 durable order chain。

LLM 不设置 entry、数量、止损、止盈、订单类型或杠杆。策略只从可靠的样本外统计中按净收益、Sharpe、回撤、样本量和策略名确定性排序。

## 统一模型

`StrategyDecision` 包含 decision_id、strategy/version、params、side、有效期、regime、evidence、AI run_id、statistics_id、data/universe version、reason_codes 和 rejected_alternatives。decision_id 会写入 `strategy_decisions`、`decision_plan_links`、TradePlan metadata 及 M2 OrderIntent。

## UniverseProvider

只接受显式 CLI symbols、经过验证的 daily pool 或手工白名单；限制最大数量，检查 pool freshness，并生成 universe_version。UI 临时 JSON 不会直接进入订单链路。

## StrategyStatisticsRepository

M3 提供静态 JSON 导入和内存仓库。统计必须匹配 symbol/timeframe/regime，满足最小交易样本、数据区间、评估新鲜度和最大回撤要求；无可靠统计时不会让 AI 主观选择策略。

## AI advisory worker

`AdvisoryWorker` 使用单线程后台执行 AgentManager，避免阻塞 Runtime tick，并支持 timeout。Runtime 只使用 M1 验证后的 snapshot；Stub/过期/来源不完整的 advisory 会被拒绝。`allow_quant_without_ai` 默认关闭，纯量化模式必须显式开启。

## Shadow mode

`paper_decision_enabled=True` 且 `paper_decision_shadow_mode=True` 时生成并审计 StrategyDecision/TradePlan，经过 RiskEngine，但在最终 broker 边界强制不提交订单。关闭 shadow 后才复用 M1/M2 Paper 链；不会启用 alpaca_live。

## 测试

```text
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m ruff check trader tests
```

## 已知限制

默认未启用 M3 decision service，以保持既有 Runtime 行为兼容；启用后若没有可靠策略统计，服务会安全地产生空决策。静态统计仓库需要由外部回测流程提供可信 JSON，M3 不自动训练或评估模型。
