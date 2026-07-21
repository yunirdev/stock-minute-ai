# AGENTS.md — stock-minute-ai 当前工程基线

> 必须先读：任何 Agent 开始代码工作前先读本文件；完成代码工作、测试或架构调整后，必须同步修改本文件。这里是唯一权威工程上下文，不再维护平行的规划/交接文档。

## 当前产品定义

- 当前版本是“AI 自动 Alpaca Paper 模拟交易”，不逐笔请求用户批准。
- 自动实盘不在范围内；`alpaca_live + auto_trade_paper` 必须拒绝启动。
- NiceGUI 是唯一用户界面，`Runtime` 是唯一生产交易循环。
- Agent/LLM 只生成分析与评分，不直接调用 broker。
- 未启用自动交易时记录 `DRY_RUN`；启用时，合格计划自动提交 Paper `LMT` 订单。

## 唯一运行链

```text
NiceGUI / AgentManager -> 多 Agent 分析 -> ai_states.duckdb
trader.main -> Runtime -> kill switch/watchdog/启动对账
  -> selection -> ATR TradePlan -> AI score gate -> allocation
  -> deterministic risk -> DRY_RUN 或 durable/idempotent Alpaca Paper LMT
  -> order polling -> portfolio/audit persistence
```

生产入口：

- UI：`python -m trader.monitor_nice`
- 引擎：`python -m trader.main [--auto-trade]`
- 交易循环：`trader/runtime.py::Runtime`
- Agent 调度：`trader/ai/manager.py::AgentManager`

已经删除且不得复活的旧路径：`scheduler.py`、内存 `PaperBroker`、独立 yfinance data-feed、旧 `OrchestratorAgent`、Streamlit UI 偏好、手工逐笔审批、无调用 Protocol 契约壳和 `PendingOrder`。

## 不可破坏的安全约束

1. 自动下单仅允许 `broker_type=alpaca_paper` 且 `auto_trade_paper=True`。
2. 自动执行只能创建 `LMT` 订单。
3. AI 分数必须存在、未过期并达到阈值。
4. 确定性风控、kill switch、幂等键、耐久订单记录和启动对账不得绕过。
5. 对账阻塞时不得提交新订单。
6. API Key/Secret 不进入日志、数据库、Git 或测试快照。
7. Agent 模块不得导入 broker 或执行订单。

## 活跃模块地图

- `trader/main.py`、`runtime.py`：CLI 与交易生命周期
- `trader/models.py`：共享数据模型与 AgentContext
- `selection.py`、`plan.py`、`allocator.py`、`risk_engine.py`：决策管道
- `broker/alpaca.py`：执行适配器
- `order_store.py`、`portfolio.py`、`audit.py`：耐久状态与审计
- `ai/manager.py`、`ai/agents/`：分析系统
- `monitor_nice.py`、`monitor_data.py`：用户界面及查询层
- `strategies/`、`strategy_core.py`、`factors/`、`backtest/`：研究与策略
- `watchdog.py`、`kill_switch.py`：运行安全
- `morning_brief.py`、`discord_report.py`、`notify.py`：报告和通知

## 文件与数据规则

保留并提交：源代码、测试、`README.md`、本文件、有效技术文档、配置模板。

可随时重建且不得提交：`__pycache__/`、`*.pyc`、pytest/ruff/mypy 缓存、`.nicegui/`、`.tmp/`、`*.egg-info/`、`conf/` 自动快照、外部下载目录 `github/` 与 `archive/`。

不得把 `trade.duckdb`、`ai_states.duckdb` 或日志当缓存删除；它们是用户运行记录。不得删除 `conf/ui_settings.json`，它是用户偏好。

## 每次工作的固定协议

开始前：

1. 读本文件和任务直接相关的代码。
2. 查看 `git status --short`，保留用户未提交的无关修改。
3. 从真实生产入口追调用链，不凭文件名判断活跃性。

结束前：

1. 删除本次产生的临时缓存。
2. 运行与风险相称的测试；涉及运行链至少跑完整 pytest、Ruff 和 compileall。
3. 更新“当前基线”和“最近变更”；替换陈旧条目，不无限追加流水账。
4. 检查 diff/status，确保秘密、数据库、日志或生成物没有进入提交。
5. 行为、入口或安全约束变化时同步更新 README。

## 验证命令

```powershell
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m ruff check trader tests
.venv\Scripts\python.exe -m compileall -q trader tests
```

## 当前基线

- 日期：2026-07-20
- 基线提交：本轮清理提交（用 `git rev-parse --short HEAD` 获取）
- 测试基线：137 passed；Ruff 和 compileall 通过
- 工作树：提交后应保持干净；运行测试产生的缓存不提交
- 明确技术债：Runtime 从 `ai_states.duckdb` 读取由 UI/AgentManager 生成的 AI 分数；完全无 UI 自动运行需要另行启动 Agent 生产侧。
- 当前目标：保持一条可理解、可审计、仅 Paper 自动执行的生产链，不保留“也许以后会用”的兼容实现。

## 最近变更

- 删除旧 Scheduler、PaperBroker、data-feed、orchestrator、Streamlit 偏好与无调用契约层。
- 将 `AgentContext` 合并进 `models.py`。
- 删除旧审批制残留、死测试、生成缓存和互相冲突的旧规划文档。
- 建立本文件作为后续 Agent 的唯一工程记忆。
