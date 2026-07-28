# 隔离 Paper 执行演练

本手册用于重复验证生产执行状态机，不连接 Alpaca、live 或任何外部网络。
演练使用内存 broker，数据库文件名必须包含 `smoke`，并且不会删除或覆盖
`trade.duckdb`、`ai_states.duckdb`、日志或用户配置。

## 运行

在项目根目录执行：

```powershell
.venv\Scripts\python.exe -m trader.paper_smoke --db .tmp\paper-smoke.duckdb
```

成功时 JSON 输出包含 `"ok": true`、`"network_used": false`，并报告：

- BUY 限价单的部分成交到完全成交；
- SELL/CLOSE 完全成交；
- 风控拒绝且不创建订单；
- 提交响应丢失后持久化为 `UNKNOWN`；
- Runtime 重启按 `client_order_id` 恢复订单且不重复提交；
- 恢复后的未知订单由模拟 broker 明确取消并进入 `CANCELED`。

同一个 smoke 数据库可以重复运行。每次使用新的 run/plan ID，历史证据保留。
若任一断言失败，命令以非零状态退出，不应继续 Paper 发布。

## 查询完整追溯

查询数据库中的全部 plan → risk → idempotency/order → fill 证据：

```powershell
.venv\Scripts\python.exe -m trader.audit_query --db .tmp\paper-smoke.duckdb
```

只查询某个计划：

```powershell
.venv\Scripts\python.exe -m trader.audit_query `
  --db .tmp\paper-smoke.duckdb `
  --plan-id <paper_smoke 输出的 plan_id>
```

每条记录必须至少具有：

- `plan.plan_id`、方向、价格、止损、数量和最终状态；
- `risk_events` 中的 `PLAN` 或 `PRE_SUBMIT` 确定性结论；
- 已提交订单的 `idempotency_key`、`client_order_id`、broker ID 和生命周期状态；
- 部分或完全成交对应的增量 `fills`。

风控拒绝的计划没有 `order`，但必须有 `BLOCKED` 风险证据。`UNKNOWN`
恢复场景最终为 `CANCELED`，且 `restart_resubmissions` 必须为 `0`。

## 故障判定

- `SMOKE_DB_NAME_MUST_CONTAIN_SMOKE`：换用专用 smoke 文件名，禁止指向生产库。
- `AUDIT_SCHEMA_MISSING`：数据库不是当前版本创建的演练/交易库；不要手工补表，
  先确认路径，再运行 smoke 初始化。
- `restart reconciliation unexpectedly blocked`：broker/local 事实不一致；保留数据库
  用 `trader.audit_query` 检查，禁止绕过对账。
- 任一 live/MKT 提交错误：这是预期硬边界；本演练不授权 live。

## 验收后的处理

演练数据库位于 `.tmp` 时属于可删除测试工件，但本命令自身不会删除它。生产运行仍只使用：

```powershell
.venv\Scripts\python.exe -m trader.main --auto-trade
```

该命令只有在 `broker_type=alpaca_paper`、Kill Switch 未触发、启动对账通过、
研究与确定性风控均通过时才允许提交 LMT。
