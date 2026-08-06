# stock-minute-ai

AI 辅助的美股分钟级 Alpaca Paper 交易系统：多智能体研究 + 经过验证的策略统计 + 确定性风控 + DuckDB 全链路审计 + NiceGUI 监控台。

> 自动交易目前只接 **Paper（模拟盘）**。系统从不请求单笔人工审批，也从不自动提交真实实盘订单——这条红线由代码强制，不是配置开关。

## 它怎么工作

```
【研究】盘前或盘后跑一次，当日只跑一次
        │
        ▼
daily_research → TradingAgents（多智能体：技术面/基本面/宏观/新闻/期权/ETF流向…）
   结果冻结写入 ai_states.duckdb，盘中不再重跑 LLM
        │
        ▼
【Runtime】交易时段内持续运行，标的由 --symbols 指定
        │
        ▼
runtime.py 读取当日冻结研究结果 + 实时分钟线
   → PaperDecision 决策门（策略信号 + holdout 统计 + 当日研究，三者齐备才放行）
   → ATR 计划 / 仓位分配 / 确定性风控（risk_engine）
   → CandidatePlan → FinalTradePlan → OrderIntent → Alpaca Paper LMT 单
        │
        ▼
DuckDB 审计 (trade.duckdb / ai_states.duckdb) + 每日收盘校验备份


【选股辅助】独立于 Runtime，在监控台里手动触发
        │
        ▼
data_cache（Alpaca/yfinance 本地 Parquet 缓存）
   → selection_pools 构建长线池 / 日内决策池 → conf/selection_pools.json
   → decision_trade_plans 生成候选计划供人工参考
```

**Runtime 是唯一能下单的组件**：AI agents 只产出分析，从不直接调用 broker。

选股池（`selection_pools`）是监控台侧的研究工具，**不在 Runtime 的自动执行链路上**——Runtime 的标的来自 `--symbols` 参数，选股结果需要你自己决定是否采纳。

## 功能

### 数据与选股
- **本地行情缓存** — Alpaca 优先、yfinance 兜底，Parquet 落盘 + 内存缓存，严格本地优先、不自动联网
- **确定性选股池** — 长线池 / 日内决策池，基于可靠 holdout 统计与流动性打分，重建失败保留上一次有效结果
- **观察列表自动瘦身** — 每次重建选股池后，自动清理已被淘汰标的的本地 bar 缓存；持仓中或有未终结挂单的标的强制豁免，永不误删

### 研究与决策（AI）
- **多智能体研究** — 技术面 / 基本面 / 宏观 / 新闻 / 期权 / ETF 资金流 / 机构持仓 / 牛熊辩论等独立 agent，通过 TradingAgents 编排
- **每日一次、结果冻结** — 研究批次在专属子进程环境运行（不污染生产 Runtime 依赖），结果写入 `ai_states.duckdb` 后当日不重跑
- **决策门禁** — 没有当日有效研究结果、没有可靠 holdout 统计、没有当前策略信号，一律不出计划

### 执行与风控
- **确定性风控** — ATR 止损止盈、仓位分配、熔断、Kill Switch，均不可被绕过
- **不可篡改的仓位版本链** — 成交通过事务写入单一 `PositionPlan` 版本链，部分成交/减仓/平仓都不能悄悄替换原始入场价、止损、目标或数量基线
- **幂等订单管道** — `CandidatePlan → FinalTradePlan → OrderIntent`，重启后自动对账、恢复缺失意图，`UNKNOWN` 订单从不猜测或重新提交

### 监控台
- **NiceGUI 决策台** — 6 大区块实时展示研究、持仓、计划、风控、订单链路
- **全部 24 个界面操作** 都走统一的可审计 action 网关，异步动作在真实结果出来前保持 BUSY 而不是假装完成

### 运维 / 审计
- **每日收盘校验备份** — 20:00 ET 后对 `trade.duckdb` / `ai_states.duckdb` 做一次幂等、带校验和的备份，源库从不被覆盖
- **DuckDB 全链路审计** — 计划、风控、订单、成交可追溯查询（`trader.audit_query`）
- **Discord 播报** — 统一授权/脱敏/幂等网关，外发默认需要显式开关或手动点击

## 环境准备

需要 **Python 3.13+** 和 **uv**。

```bash
setup.bat
# 编辑 .env，填入 Alpaca Paper Key（参考 .env.example）
```

## 运行

```bash
启动监控台.bat
```

或手动执行：

```bash
# 先在 UI 里下载足够的本地历史 K 线，再依次生成统计与研究结果
uv run python -m trader.strategy_statistics --symbols AAPL,MSFT --timeframe 5m
uv run python -m trader.daily_research --symbols AAPL,MSFT,NVDA --strategy-statistics-path conf/strategy_statistics.json

# 只记录 DRY_RUN，不下单
uv run python -m trader.main --symbols AAPL,MSFT

# 满足门禁的计划会以幂等 Alpaca Paper 限价单提交
uv run python -m trader.main --symbols AAPL,MSFT --auto-trade --min-ai-score 70
```

不带 `--auto-trade` 时，所有计划只记录为 `DRY_RUN`，不会真正下单。

## 配置

关键环境变量（完整列表见 [.env.example](.env.example)，`.env` 已 gitignore）：

| 变量 | 说明 |
|------|------|
| `BROKER_TYPE` | `alpaca_paper`（模拟盘）\| `alpaca_live`（真实资金，谨慎） |
| `ALPACA_API_KEY` / `ALPACA_API_SECRET` | Paper 账户密钥，从 [Alpaca Paper Dashboard](https://app.alpaca.markets/paper/dashboard/overview) 获取 |
| `MIN_AI_SCORE` | 自动交易的最低 AI 评分门槛 |
| `TRADINGAGENTS_PROJECT_DIR` / `TRADINGAGENTS_PYTHON` | TradingAgents 独立 Python 环境路径，未配置时研究批次明确标记 FAILED |
| `LLM_PROVIDER` | `ollama`（本地免费）\| `anthropic`（需 API Key）\| `stub`（离线测试） |
| `DISCORD_EXTERNAL_SEND_ENABLED` | 播报外发总闸，默认放行，设为 `false` 时全部拦截 |

## 入口

| 命令 | 用途 |
|------|------|
| `python -m trader.monitor_nice` | NiceGUI 监控台 |
| `python -m trader.main` | 生产 Runtime，消费当日冻结研究结果 |
| `python -m trader.daily_research` | 单次每日研究批次（可手动/自动化触发） |
| `python -m trader.strategy_statistics` | 基于本地缓存 K 线构建策略统计 |
| `python -m trader.paper_smoke` | 无网络依赖的执行/重启冒烟测试 |
| `python -m trader.audit_query` | 只读的计划/风控/订单/成交链路查询 |
| `python -m trader.research_snapshot_quality` | 10 日研究快照质量报告 |
| `python -m trader.data_hub_shadow` | Alpaca/Data Hub 双读质量巡检（只读） |
| `python -m trader.data_hub_replay` | 20 交易日本地/Alpaca 历史一致性回放 |
| `notebooks/research.py` | Marimo 研究笔记本 |

## 项目结构

```
stock-minute-ai/
├── trader/
│   ├── main.py                # Runtime 入口，唯一能下单的路径
│   ├── runtime.py              # 计划驱动主循环：决策→风控→执行→对账
│   ├── daily_research.py       # 每日一次的研究批次编排
│   ├── data_cache.py           # 本地 Parquet 行情缓存 + 观察列表清理
│   ├── selection_pools.py      # 长线/日内决策池构建与打分
│   ├── risk_engine.py          # 确定性风控：ATR / 熔断 / Kill Switch
│   ├── execution_pipeline.py   # CandidatePlan → FinalTradePlan → OrderIntent
│   ├── position_plans.py       # 不可篡改的仓位版本链
│   ├── portfolio.py            # 持仓/资金/成交记账（DuckDB）
│   ├── audit.py / audit_query.py  # 审计写入与只读查询
│   ├── production_operations.py   # 每日收盘校验备份
│   ├── monitor_nice.py         # NiceGUI 监控台（决策台 UI）
│   ├── ai/
│   │   ├── manager.py           # 多智能体调度（并行 ThreadPoolExecutor）
│   │   └── agents/              # 技术面/基本面/宏观/新闻/期权/ETF流向/牛熊辩论…
│   ├── broker/alpaca.py        # 唯一的 broker 适配（Alpaca Paper/Live）
│   ├── strategies/             # 策略注册与基类
│   ├── factors/                # 因子计算
│   ├── backtest/                # 因子/策略回测分析
│   └── teams/                   # 市场环境/维护类协同任务
├── tests/                       # pytest 用例
├── conf/                        # 选股池/候选池/行情快照等运行期 JSON 状态
├── docs/                        # 架构、运维手册、迁移记录
├── notebooks/research.py        # Marimo 研究笔记本
├── AGENTS.md                    # 工程基线（唯一权威）
└── setup.bat / 启动监控台.bat    # 一键环境搭建 / 启动
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 / 包管理 | Python 3.13+ / uv |
| 持久化 | DuckDB（交易审计 + AI 研究结果） |
| 行情数据 | Alpaca（主）+ yfinance（兜底），本地 Parquet 缓存 |
| Broker | Alpaca Paper / Live（`trader/broker/alpaca.py`） |
| 多智能体研究 | TradingAgents（LangChain/LangGraph，独立子进程环境） |
| 本地 LLM | Ollama（默认 qwen2.5:14b），可选 Anthropic |
| 监控台 | NiceGUI |
| 通知 | Discord（Bot Token 或 Webhook） |
| 测试 / 静态检查 | pytest / ruff |

## 安全边界

- Agent 从不直接调用 broker；只有 Runtime 能下单。
- 自动提交要求同时满足 `auto_trade_paper` 与 `alpaca_paper` broker 配置。
- 只提交 LMT（限价）单。
- 缺失、过期、不可信或不充分的 AI 证据一律拒绝出计划。
- 没有有效策略统计，不产生任何决策。
- Kill Switch、对账、幂等订单记录、风控规则均不可绕过。

## 文档

- 工程基线：[AGENTS.md](AGENTS.md)（唯一权威，改动前必读）
- 隔离执行演练：[docs/PAPER_SMOKE_RUNBOOK.md](docs/PAPER_SMOKE_RUNBOOK.md)
- 研究快照质量指标：[docs/RESEARCH_SNAPSHOT_QUALITY.md](docs/RESEARCH_SNAPSHOT_QUALITY.md)
- Data Hub 20 交易日观察流程：[docs/DATA_HUB_SHADOW_RUNBOOK.md](docs/DATA_HUB_SHADOW_RUNBOOK.md)
- 闭环 Paper 自动化与全部 NiceGUI 动作验收合同：[docs/CLOSED_LOOP_ACCEPTANCE.md](docs/CLOSED_LOOP_ACCEPTANCE.md)
- 运维恢复流程与阶段性交付报告：[docs/OPERATIONS_RECOVERY_RUNBOOK.md](docs/OPERATIONS_RECOVERY_RUNBOOK.md) / [docs/PAPER_CLOSED_LOOP_SIGNOFF.md](docs/PAPER_CLOSED_LOOP_SIGNOFF.md)
- 长周期成熟度证据与架构/最终签收的区别：[docs/PAPER_MATURITY_RUNBOOK.md](docs/PAPER_MATURITY_RUNBOOK.md) / [docs/PAPER_MIGRATION_SIGNOFF.md](docs/PAPER_MIGRATION_SIGNOFF.md)
- TradingAgents 本地环境搭建：[docs/TRADINGAGENTS_LOCAL_SETUP.md](docs/TRADINGAGENTS_LOCAL_SETUP.md)

NiceGUI Web 模式默认绑定 `127.0.0.1`，只有确需其他网卡时才显式设置 `QUANT_HOST`。
只读集成可用 `/api/ui-actions` 与 `/api/order-explanation/{plan_id}`。

## 验证

```bash
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m compileall -q trader tests
.venv\Scripts\python.exe -m ruff check trader tests
```

## 声明

本项目仅用于研究与学习，不构成投资建议。自动交易当前仅接 Paper 账户，任何真实资金决策请自行判断并承担后果。
