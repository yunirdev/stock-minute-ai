# ResearchSnapshot 质量报告

影子快照不改变 Runtime 或每日研究读取路径。每日研究会记录：

- `daily_research_snapshot_links`：每个 run/symbol 的快照写入状态；
- `daily_research_snapshot_comparisons`：实际 candidate 与冻结快照重放值的逐字段比较；
- `research_snapshot_run_bindings`：不可变的 run/symbol → snapshot 绑定。

生成最近 10 个研究交易日的报告：

```powershell
.venv\Scripts\python.exe -m trader.research_snapshot_quality `
  --db ai_states.duckdb `
  --required-days 10 `
  --min-coverage 0.95
```

加 `--save` 会把同一份 JSON 追加到
`research_snapshot_quality_reports`；相同 `report_id` 不重复写入。

`passed=true` 同时要求：

- 至少观察到指定数量的研究交易日；
- 快照可用率、可接受质量率、比较覆盖率和关键字段覆盖率达到门槛；
- 没有快照写入或内容哈希重放错误；
- 所有字段差异均已分类，`UNCLASSIFIED` 数量为零。

bars 或策略统计缺失会保留为 `MISSING/PARTIAL`，不会伪装成完整数据。
报告不影响下单，也不会删除任何历史快照。
