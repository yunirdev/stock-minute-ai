# stock-minute-ai：AI Agent 代码库接手指南

更新日期：2026-07-20

本文档描述当前代码事实，供新的 AI Agent 快速接手。若本文档与旧版 README.md、CLAUDE.md 冲突，以代码和测试为准。

## 1. 项目目标与当前阶段

目标是美股分钟级量化交易系统：

- 人工负责策略、因子和研究方法。
- 量化模块负责行情、信号、计划、仓位和风险。
- AI 辅助选股、市场状态判断和策略选择。
- 系统自动在 Alpaca Paper 模拟盘执行，不逐笔人工审批。
- 实盘不是当前目标，默认不得启用。

当前代码已经具备行情、策略库、回测、选股池、AI Agent、风控、Alpaca 下单、成交轮询、DuckDB 审计和 NiceGUI 监控台，但尚未形成真正的单进程 AI 自动交易闭环。

当前完成度概览：

| 能力 | 状态 |
|---|---|
| 本地行情缓存 | 已实现 |
| 技术策略和统一回测 | 已实现 |
| 因子 IC 和分位数研究 | 已实现 |
| 多层选股池 | 已实现 |
| 多 Agent 分析 | 已实现，主要由 UI 触发 |
| Paper 自动批准和下单 | 有可运行雏形 |
| 订单成交轮询和持仓写入 | 已实现 |
| Runtime 内部直接运行 AI | 未实现 |
| AI 自动选择策略 | 未实现 |
| 重启后的订单恢复与对账 | 不完整 |
| AI 决策效果反馈学习 | 未实现 |

## 2. 技术栈

- Python：要求 3.13 及以上；当前本机虚拟环境使用 Python 3.14。
- 包管理：uv，锁文件为 uv.lock；也支持标准 pip editable 安装。
- 数据分析：pandas、numpy、scipy、ta。
- 数据库：DuckDB。
- 行情与券商：alpaca-py、yfinance、requests。
- UI：NiceGUI、Plotly；原生窗口由 pywebview 提供。
- 配置：pydantic、pydantic-settings、python-dotenv。
- AI：本地 Ollama、可选 Anthropic、最终 Stub 降级。
- 新闻与 RSS：feedparser、urllib、Finnhub、SEC EDGAR、WallStreetCN 等。
- 研究：Marimo。
- 测试与质量：pytest、ruff。

pyproject.toml 中的直接依赖：

- duckdb
- marimo
- nicegui[native]
- numpy
- pandas
- plotly
- pydantic
- pydantic-settings
- pytest
- scipy
- python-dotenv
- requests
- ruff
- ta
- yfinance
- alpaca-py
- feedparser

当前所有依赖都放在 project.dependencies，没有拆分 production/dev extras。

## 3. 顶层目录

| 路径 | 用途 |
|---|---|
| trader/ | 主应用、交易、AI、研究和 UI 代码 |
| tests/ | 项目测试；当前完整运行结果为 101 passed |
| data/bars/ | 本地 Parquet K 线缓存，已被 gitignore |
| conf/ | 选股池、市场扫描、市场状态和 UI 偏好快照 |
| docs/ | 项目说明与本接手文档 |
| notebooks/research.py | Marimo 研究入口 |
| scripts/check.py | 列出本地缓存行情及时间范围 |
| archive/ | 大型旧数据库和日志备份，不是当前运行路径 |
| github/last30days-skill/ | vendored 外部项目，不属于交易系统核心 |
| logs/ | 运行日志，已被 gitignore |
| .venv/ | 本地虚拟环境，已被 gitignore |
| setup.bat | Windows 环境安装和修复 |
| 启动监控台.bat | Windows UI 启动器 |
| trade.duckdb | 交易审计数据库，本地文件，不提交 |
| ai_states.duckdb | AI Agent 状态和建议数据库，本地文件，不提交 |

app/ 当前不是主路径。archive/ 和 github/last30days-skill/ 不应被 setuptools 当成项目包。pyproject.toml 已限制只发现 trader*。

## 4. 总体架构

系统目前有四条相关但未完全统一的链路：

    研究链路
    Parquet bars -> factors/strategy_core -> factor_analysis/engine -> NiceGUI

    AI 分析链路
    NiceGUI -> AgentManager -> algorithmic agents + LLM agents
            -> ai_states.duckdb -> UI 报告

    主交易链路
    Runtime -> Alpaca bars -> ConsensusSelector -> ATRPlanner
            -> EqualWeightAllocator -> AI safety -> RiskEngine
            -> AlpacaBroker -> order polling -> Portfolio/AuditLog

    旧信号链路
    Scheduler -> strategy signal -> RiskEngine -> broker

runtime.py 是当前主交易运行时。scheduler.py 是历史兼容路径，不应作为新功能首选。

当前最大的架构断点：

    AgentManager 在 NiceGUI 中运行并写 ai_states.duckdb
                              |
                              v
    Runtime 只读取综合分，不直接运行 AgentManager

因此监控台当前不仅是展示层，还是 AI 分数生产者。目标架构应把 AI 决策服务移入 Runtime 或独立 worker，让 UI 只负责展示。

## 5. Runtime 主交易流程

入口：trader/main.py -> Runtime(config).run()

每个 tick 的主要顺序：

1. 检查 FileKillSwitch。
2. 运行 HeartbeatWatchdog。
3. 从 broker 获取账户权益和持仓；broker 是权威来源。
4. 设置日内起始权益并检查日内最大回撤。
5. 轮询进程内记录的已提交订单。
6. 按时间触发晨报。
7. 仅在正常交易时段生成新交易计划。
8. 从 Alpaca 拉取配置标的的 K 线并写入本地 Parquet 缓存。
9. 合并 WallStreetCN、SEC、Finnhub 和价格异动新闻。
10. 检查已有持仓的止损和止盈。
12. ConsensusSelector 对配置的 symbols 和 strategies 做策略投票选股。
13. ATRPlanner 生成 entry、stop_loss 和 take_profit。
14. EqualWeightAllocator 分配仓位。
15. RiskEngine.evaluate_plan 做计划风控。
15. auto_trade_paper 模式下，从 ai_states.duckdb 读取综合分并执行新鲜度/阈值校验。
16. AI 安全门与 RiskEngine 均通过后，计划标记 READY 并转换为 LMT OrderIntent；未开启自动交易时标记 DRY_RUN。
17. 后续 tick 轮询成交并写 fills、equity snapshot 和审计记录。
18. 盘后生成 review。

关键现实：

- Runtime 的股票集合来自 CLI --symbols，不是全市场动态 universe。
- Runtime 的策略集合来自 CLI --strategies，不会由 AI 自动选择。
- Runtime 不运行 AgentManager，只读 ai_states.duckdb。
- auto_trade_paper=True 只允许与 broker_type=alpaca_paper 组合。
- _execute_plan 只接受通过 AI 安全门与确定性风控的自动模拟盘计划。
- 关闭 auto_trade_paper 时计划只记录 DRY_RUN，不生成待处理状态。

## 6. 核心领域模型和接口

trader/models.py 定义：

- Side
- OrderStatus
- Bar
- Signal
- RiskVerdict
- OrderIntent
- Fill
- PendingOrder
- Position
- Candidate
- TradePlan
- Advisory
- NewsEvent
- ReviewReport
- Alert
- Notification

trader/contracts.py 定义的主要 Protocol：

- Selector
- Planner
- Allocator
- PortfolioManager
- PositionMonitor
- PlanRiskChecker
- Notifier
- Agent
- NewsSource
- Reviewer
- Watchdog
- KillSwitch
- UniverseProvider
- MarketCalendar

新实现优先遵守这些数据模型和 Protocol，避免在 UI、Runtime 或 Agent 中创建重复模型。

## 7. 交易与风险模块

| 文件 | 主要职责 |
|---|---|
| trader/runtime.py | 主计划驱动运行时 |
| trader/main.py | CLI 配置和 Runtime 入口 |
| trader/scheduler.py | 历史信号驱动路径 |
| trader/selection.py | 对指定策略做当前 bar 多空投票 |
| trader/plan.py | ATRPlanner，生成入场、止损、止盈 |
| trader/allocator.py | EqualWeightAllocator |
| trader/risk_engine.py | 计划风控、日内回撤、连续失败熔断 |
| trader/position_monitor.py | 持仓止损和止盈触发 |
| trader/broker/alpaca.py | Alpaca 下单、撤单、状态、成交、持仓、权益 |
| trader/broker/paper.py | 内存 PaperBroker，主要用于测试或本地模拟 |
| trader/portfolio.py | 成交应用和权益快照 |
| trader/audit.py | 信号、订单、风险、计划和心跳审计 |
| trader/watchdog.py | heartbeat 告警和文件 kill switch |
| trader/market_calendar.py | 美股交易时段判断 |

默认 RiskConfig：

- max_position_pct = 0.20
- max_trade_risk_pct = 0.005
- daily_drawdown_limit_pct = 0.03
- max_consecutive_failures = 3
- allow_short = False

订单执行当前硬编码为：

- order_type = LMT
- tif = DAY
- 不允许自动改成市价单

## 8. 策略、回测和因子

### 8.1 统一策略入口

trader/strategy_core.py 的 compute_signals 是现有 DataFrame 策略统一入口。输入 OHLCV，输出：

- strat_signal：1 买入、-1 卖出、0 持有
- strat_exec_px：建议执行价格

当前内置 24 个策略：

1. 全仓买入并持有
2. 5/20 均线金叉死叉
3. 10/30 均线双线波段
4. 20/60 均线长期趋势
5. MACD 零轴
6. MACD 信号线
7. RSI 震荡
8. KDJ 极值反转
9. 布林带突破
10. 布林带均值回归
11. CCI 顺势
12. 唐奇安通道
13. 上周高低点
14. 三阳买两阴卖
15. 5% 小网格
16. 10% 大网格
17. BBI 上穿下穿
18. BBI 回踩不破
19. BBI 回踩加斜率
20. BBI 跌破反抽
21. ADX 趋势过滤
22. Stochastic 超买超卖
23. Williams %R
24. MFI 量价共振

trader/strategies/registry.py 用 StrategyRegistry 包装这些 DataFrame 策略。不要再创建第三套策略执行逻辑。

### 8.2 回测

trader/engine.py 的 simulate 是统一回测和模拟成交入口：

- 默认使用下一根 open 成交，降低前视偏差。
- 支持 close 对照模式。
- 支持 long/short 开关。
- 支持 fee_bps。
- 支持 slippage_bps；买入向上滑、卖出向下滑。
- 支持 max_position_pct、leverage。
- 支持按自然日重置的日内回撤熔断。
- 输出 SimResult、权益曲线和 Trade 列表。

NiceGUI 研究页默认使用 5 bps 单边滑点。

### 8.3 因子研究

trader/factors/ 定义 Factor 接口和技术因子。

trader/backtest/factor_analysis.py 提供单标的简化 Alphalens 风格分析：

- Spearman IC
- 滚动 IC
- IC mean/std/ICIR
- 因子分位数前瞻收益
- 各分位数组合累计收益

当前没有完整的跨标的 point-in-time 因子研究框架、walk-forward 策略选择器或生存者偏差处理。

## 9. 选股系统

项目同时存在两套选股概念：

### 9.1 Runtime 选股

trader/selection.py 的 ConsensusSelector：

- 对每个 symbol 运行配置的全部策略。
- 只看最后一个 bar 的非零信号。
- score = 看多策略数 / 策略总数 * 100。
- 输出 Candidate。

这不是 AI 选股，也没有根据历史策略表现加权。

### 9.2 研究/UI 选股池

trader/market_scan.py：

- 构建宽市场 universe。
- 可扫描最多 10000 个 symbol。
- 使用最新日线、价格、成交额、动量、质量和风险条件筛选。
- 结果写 conf/market_scan_report.json。

trader/selection_pools.py：

- long_term：长期关注池。
- daily_decision：3 到 7 个决策池候选。
- 支持 standard 和 aggressive 风格。
- 可选读取 ai_states.duckdb 分数。
- 保存 selection_pools.json 和 decision_pool_report.json。

trader/decision_trade_plans.py：

- 将决策池变成研究型计划。
- 输出触发条件、止损、止盈、建议权重、风险和阻塞原因。
- 保存 decision_trade_plans.json。

trader/hot_universe.py、index_universe.py、symbol_master.py 分别负责热点池、指数成分和全市场 symbol master。

目前研究/UI 选股池没有直接接入 Runtime 的 universe 和计划生成主链路。

## 10. AI 系统

### 10.1 LLM 客户端

trader/ai/client.py：

- OllamaClient：本地 HTTP API。
- AnthropicClient：可选云端。
- StubLLMClient：最终降级，通常产生中性 50 分。
- make_client 根据配置和可用性选择客户端。
- Ollama 会查询本地模型并选择配置模型或可用模型。

AI 分数全为 50 通常表示 Stub 降级，不代表真实 AI 判断。

### 10.2 AgentManager

trader/ai/manager.py 是当前主要调度器。

轨道 A：无 LLM 算法 Agent，可用 4 个 worker 并行：

- quant
- etf_flow
- options
- elite_holdings

轨道 B：LLM Agent，为避免本地 GPU 竞争只使用 1 个 worker：

- macro
- fundamental
- technical
- news
- web_research

Phase 2：

- bull_bear，依赖前面结果后再运行。

单 Agent 超时为 900 秒。

综合分静态权重：

| Agent | 权重 |
|---|---:|
| macro | 25% |
| fundamental | 20% |
| quant | 15% |
| options | 12% |
| etf_flow | 10% |
| elite_holdings | 10% |
| technical | 5% |
| news | 2% |
| web_research | 1% |

bull_bear 单独展示，不参与加权。

重要限制：

- 权重是人工静态设置，没有用历史收益训练。
- AgentManager 主要由 monitor_nice.py 触发。
- Runtime 通过 get_composite_scores_from_db 读取数据库，不直接运行 AgentManager。
- 读取综合分当前没有严格的新鲜度、数据版本或 model provenance 门槛。
- AI 当前只改变 plan.confidence，不选择 strategy 或 strategy params。

trader/ai/agents/orchestrator.py 是另一套汇总骨架，不是 Runtime 当前主路径；不要把它误认为实际自动交易协调器。

## 11. 数据来源和缓存

### 11.1 实时/交易数据

- Alpaca：Runtime 实时 K 线、账户、持仓、订单和成交。
- yfinance：研究缓存、基本面和部分行情补充。

### 11.2 新闻和事件

- WallStreetCN
- SEC EDGAR 8-K
- Finnhub
- Yahoo RSS/chart
- CNBC RSS
- 官方 Fed/BLS/BEA/EIA 事件
- 价格异动事件
- 可选 Web Research/Agent Reach

### 11.3 Parquet 缓存

trader/data_cache.py：

- 路径：data/bars/{SYMBOL}_{TIMEFRAME}.parquet
- 内存缓存由线程锁保护。
- 本地文件新鲜时优先使用，不联网。
- 支持 1m、5m、15m、30m、1h、1d。
- 支持后台 warm-up 和按需增量更新。
- UI 研究通常读取本地缓存。

## 12. DuckDB 与 JSON 持久化

### 12.1 trade.duckdb

AuditLog 表：

- signals
- orders
- risk_events
- heartbeat
- trade_plans

Portfolio 表：

- fills
- equity_snapshots

### 12.2 ai_states.duckdb

AgentManager 表：

- agent_states：每个 Agent 的状态、分数、运行时间和摘要。
- ai_advisories：每条结构化 Advisory。

### 12.3 conf JSON

- symbol_master.json
- index_universe.json
- hot_universe.json
- market_scan_report.json
- selection_pools.json
- decision_pool_report.json
- daily_candidates.json
- decision_trade_plans.json
- market_regime.json
- ui_settings.json

这些 JSON 是研究快照和 UI 状态，不是 broker 权威持仓来源。

## 13. NiceGUI 监控台

入口：trader/monitor_nice.py。

导航页面：

- 总览
- 交易记录
- 决策台
- 选股池
- 研究
- 风控
- 维护
- 系统

主要功能：

- 展示账户权益、持仓、订单、成交、心跳和风险状态。
- 运行 AgentManager 并显示 Agent 进度、综合评分和详细报告。
- 构建长期池和决策池。
- 运行因子分析和策略回测。
- 展示决策交易计划。
- 管理 UI 偏好。

默认使用原生 WebView；设置 QUANT_WEB=1 时用浏览器模式。

monitor_data.py 是 UI 的 DuckDB/Alpaca 数据读取层。业务逻辑应尽量留在 trader 模块，不继续堆进 20 万字节以上的 monitor_nice.py。

## 14. 配置

配置模板：.env.example。真实 .env 已被 gitignore，禁止打印或提交其内容。

关键变量：

- BROKER_TYPE：alpaca_paper 或 alpaca_live。
- DATA_FEED_TYPE：当前为 alpaca。
- ALPACA_API_KEY
- ALPACA_API_SECRET
- ALPACA_DATA_FEED
- SYMBOLS：部分下载/预热脚本使用，不是 Runtime 唯一权威来源。
- DB_PATH
- TRADE_DB_PATH
- LLM_PROVIDER：ollama、anthropic 或 stub。
- OLLAMA_BASE_URL
- OLLAMA_MODEL
- OLLAMA_THINK
- ANTHROPIC_API_KEY
- DISCORD_BOT_TOKEN
- DISCORD_CHANNEL_ID
- DISCORD_WEBHOOK_URL
- SEC_USER_AGENT
- FINNHUB_API_KEY

config.py 的 Settings 只读取声明的环境变量。TradingConfig 的 symbols、strategies 等主要由 CLI 或代码显式构造。

## 15. 运行方式

推荐先安装：

    setup.bat

启动监控台：

    .venv\Scripts\python.exe trader\monitor_nice.py

浏览器模式：

    $env:QUANT_WEB = '1'
    .venv\Scripts\python.exe trader\monitor_nice.py

Runtime DRY-RUN：

    .venv\Scripts\python.exe -m trader.main --symbols AAPL,MSFT --tf 5m

当前 Paper 自动交易入口：

    .venv\Scripts\python.exe -m trader.main --symbols AAPL,MSFT --tf 5m --broker-type alpaca_paper --auto-trade --min-ai-score 65

运行测试：

    .venv\Scripts\python.exe -m pytest tests -q

环境自检：

    .venv\Scripts\python.exe scripts\check.py

可编辑安装：

    .venv\Scripts\python.exe -m pip install -e .

当前验证状态：

- editable install 成功。
- Python 语法检查成功。
- 101 tests passed。
- .pytest_cache 在本机有权限警告，但不影响测试逻辑。

## 16. 测试覆盖

测试模块：

- account risk
- architecture scaffold
- brief review
- daily candidates
- data feed
- decision trade plans
- unified engine golden results
- intraday levels
- M0 contracts
- manual push
- market scan
- morning brief
- official events
- runtime safety
- selection pools

关键安全和行为测试包括：

- stale bar 拒绝。
- 策略回测黄金结果。
- 下一根 open 与 close 成交模式。
- 滑点对买卖两腿均为不利方向。
- 风控和计划校验。
- Paper/Alpaca 成交同步。

新增金钱或执行路径必须留下最小可运行测试。

## 17. 当前 P0/P1 缺口

### P0：开始无人值守 Paper 前必须修复

1. 强制 auto_trade_paper 只能与 broker_type=alpaca_paper 组合；其他组合启动即失败。
2. AI 分数必须校验 created_at、新鲜度、模型来源和非 Stub 状态。
3. Runtime 启动时从 Alpaca 对账 open orders、positions 和最近 fills，不能只依赖内存 _open_orders。
4. 每个交易意图需要稳定 idempotency key，重启和重试不能重复下单。
5. 明确订单超时、部分成交、撤单失败和 API 断线行为。

### P1：实现目标架构

1. 将 AI 决策调用放进 Runtime 或独立 worker，解除对 UI 的运行依赖。
2. 将 selection_pools/decision_trade_plans 接到 Runtime universe。
3. 定义统一 StrategyDecision：
   symbol、strategy、params、side、confidence、target_weight、有效期和证据。
4. AI 选策略应基于相似市场状态下的样本外表现，不是 LLM 主观分数。
5. 将每次候选、未选择原因、策略选择、计划、订单和结果写成可关联审计链。
6. 建立固定策略、策略投票、AI 选择和买入持有的对照实验。

### P2：研究质量

1. walk-forward 和严格样本外测试。
2. point-in-time universe 与生存者偏差控制。
3. 公司行动、交易日历、点差、滑点、手续费和不可成交模拟。
4. 策略按 market regime 的条件表现数据库。
5. Paper 长期运行后的漂移、回撤和策略失效监控。

## 18. 新 Agent 接手 Checklist

1. 先读本文件、pyproject.toml、trader/config.py、trader/runtime.py。
2. 不读取或输出真实 .env 密钥。
3. 不删除或覆盖用户的 DuckDB、Parquet、conf JSON 和日志。
4. 不把 archive/ 或 github/last30days-skill/ 当作主应用代码。
5. 修改前确认 Runtime 主路径与 UI 研究路径的差异。
6. 保持 LMT、kill switch、风险上限和 Paper 隔离。
7. 先运行目标测试，再运行全部 tests。
8. 不以 AI 分数高作为盈利证据；必须使用样本外净收益对照。
9. 当前产品目标是无人审批的 Alpaca Paper 自动交易，不是实盘。

## 19. 建议下一项实现

最小且正确的下一步不是增加更多 Agent，而是建立 Runtime 内部的 PaperDecisionService：

    fresh bars + current positions + candidate pool
                         |
                         v
    quant features + validated strategy statistics
                         |
                         v
    StrategyDecision
                         |
                         v
    existing ATRPlanner + Allocator + RiskEngine
                         |
                         v
    Alpaca Paper LMT + reconciliation + audit

LLM 只补充新闻和事件证据。价格、仓位、止损和下单必须继续由确定性的量化和风险代码负责。
