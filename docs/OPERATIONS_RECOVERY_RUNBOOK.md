# Paper 运行恢复手册

范围：Alpaca Paper、研究数据库与 Runtime；不授权自动实盘。

## 故障分级

- API：行情、TradingAgents、Alpaca Paper 或 Discord 连接失败。保留失败审计，
  不把空响应视为成功，不猜测订单状态。
- DATABASE：DuckDB 无法读取、锁冲突或完整性校验失败。停止写入和新单，
  保留原文件，使用备份副本验证。
- RUNTIME：心跳停止、进程异常或 Kill Switch 生效。恢复前先执行 broker/local
  对账，UNKNOWN 订单不得重提。

## 备份

使用 `RecoveryManager.create_backup` 将明确列出的 `.duckdb` 文件复制到新的
备份目录。每个副本必须通过只读 DuckDB 打开、记录大小和 SHA-256。不要删除
或覆盖 `trade.duckdb`、`ai_states.duckdb`、日志或 UI 偏好。

## 恢复演练

1. 选择临时恢复目录，禁止指向现有生产数据库路径。
2. 调用 `restore_to_new_directory`；目标文件已存在时必须停止。
3. 验证 SHA-256、只读打开及关键表查询。
4. 对 API、数据库和 Runtime 各执行一次预期失败演练并确认分类。
5. 真正恢复生产前停止 Runtime，备份当前文件，替换操作需单独人工授权。
6. 启动后先对账；存在未知订单、仓位或成交差异时保持阻塞。
