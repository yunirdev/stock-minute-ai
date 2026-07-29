# M1 Paper AI Safety

`auto_trade_paper` 只允许与 `broker_type=alpaca_paper` 一起启用；`alpaca_live` 会在配置构造时以 `AUTO_TRADE_REQUIRES_ALPACA_PAPER` 失败。

Runtime 读取 AI 快照后，必须同时满足：0–100 分、UTC timezone-aware `created_at`、默认 30 分钟内（`--ai-score-max-age-minutes` 可覆盖）、非 Stub、provider/model/run_id 和 provenance/source/generated_by 完整、达到 `min_ai_score`。任何失败都返回明确 `reason_code` 并拒绝计划，绝不调用 broker。

AI 数据库使用 `ADD COLUMN IF NOT EXISTS` 向后兼容迁移；旧表和旧数据不删除。旧记录的新字段保持 NULL 并 fail-closed，新 AgentManager 记录写入完整 provenance。

验证：`.venv\Scripts\python.exe -m pytest tests -q`。ruff 若环境安装了 ruff，可运行 `.venv\Scripts\python.exe -m ruff check trader tests`。

已知限制：迁移前的历史记录无法可靠推断真实来源，因此保留 NULL 并被安全门拒绝；迁移后的新记录可通过完整来源验证。
