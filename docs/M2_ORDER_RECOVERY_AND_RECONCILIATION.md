# M2 Order Recovery and Reconciliation

## 状态机

`CREATED → RISK_APPROVED → PERSISTED → SENDING → OPEN/UNKNOWN → PARTIALLY_FILLED/FILLED`，终态包括 `CANCELED`、`REJECTED`、`EXPIRED`。网络异常进入 `UNKNOWN`，不会自动重试。

## 幂等方案

`idempotency_key` 由 plan_id、symbol、side、数量、限价、action 和 order leg 做 SHA-256，重复计划生成相同 key。`order_intents.idempotency_key` 有唯一约束；发送前先持久化，已有 broker_order_id 或 UNKNOWN/SENDING/OPEN 状态时不重复提交。

`client_order_id` 从 idempotency key 再次哈希派生，跨 Runtime 重启保持稳定。

## 成交与撤单

Broker 返回的累计 filled_qty 会在 Portfolio 中转换为增量；重复轮询不会重复计入仓位。部分成交保持 `PARTIALLY_FILLED`。Broker 查询失败不当作撤单或失败终态；只有明确的 CANCELED/REJECTED 才更新对应状态。

## 启动对账

Runtime 启动时读取 broker open orders、positions 和 recent fills，恢复本地 open orders，并写入 `reconciliation_reports`。存在无法解释的订单或 API 读取失败时设置交易阻塞，监控和审计仍继续；不会自动平仓或猜测成交。

## 数据库变更

新增向后兼容表 `order_intents` 和 `reconciliation_reports`；既有 `orders`、`fills`、`trade_plans` 不删除。`Portfolio.apply_fill` 通过已有 fills 计算增量。

## 测试

运行：

```text
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m ruff check trader tests
```

## 已知限制

现有 Broker 抽象对部分供应商只能返回空 open orders/recent fills；Alpaca 适配器已实现查询。对账无法解释的差异只阻塞新订单，不会自动修改 broker 状态。
