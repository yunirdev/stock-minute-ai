# stock-minute-ai — Claude Code 会话指南

每次会话开始时请先核对以下内容，确保建议和修改与系统架构一致。

---

## 一、项目定位

美股分钟级 AI 自动模拟盘交易系统（仅 Alpaca Paper；无逐笔人工审批）。

- 数据来源：Alpaca Markets（实盘/虚拟盘）
- 持久化：DuckDB（`trade.duckdb` 交易审计，`ai_states.duckdb` AI 状态）
- 前端：NiceGUI（`trader/monitor_nice.py`）
- 测试：`uv run python -m pytest tests/`（44 个测试，必须全绿）

---

## 二、双管道架构（核心）

| 管道 | 文件 | 驱动方式 | 用途 |
|------|------|----------|------|
| **信号管道** | `trader/scheduler.py` | TA 信号 → risk → LMT 订单 | 策略回路 |
| **计划管道** | `trader/runtime.py` | selection → TradePlan → AI safety → risk → LMT | AI 辅助决策 |

两者**并列运行**，互不干扰。`runtime.py` 是主力，`scheduler.py` 保留兼容。

---

## 三、模块全图

```
trader/
├── config.py           TradingConfig (pydantic)，含 auto_trade_paper 安全开关
├── models.py           所有数据模型（Bar/TradePlan/Candidate/Advisory…）
├── contracts.py        15 个 Protocol 接口定义
│
├── runtime.py          计划驱动 Pipeline（M1，主运行时）
├── scheduler.py        信号驱动循环（历史兼容）
│
├── selection.py        ConsensusSelector — TA 多空票数 → Candidate 0-100 分
├── plan.py             ATRPlanner — entry/stop/tp 计划
├── allocator.py        EqualWeightAllocator — 仓位分配
├── risk_engine.py      RiskEngine + evaluate_plan() — 双重风控
│
├── ai/
│   ├── client.py       make_client()：Ollama → Anthropic → Stub 自动降级
│   ├── manager.py      AgentManager — 并行调度 + DuckDB 持久化
│   └── agents/
│       ├── base.py         AgentBase 基类 + StubAgent
│       ├── technical.py    TA 信号 + LLM 综合打分（串行第 1）
│       ├── news.py         WSCN/yfinance 新闻情绪（串行第 2）
│       ├── web_research.py 热点话题研究，max 3 标的（串行第 3）
│       ├── bull_bear.py    多空三轮辩论，依赖前两个结果（串行第 4）
│       └── orchestrator.py 协调器骨架（未启用）
│
├── broker/alpaca.py    AlpacaBroker — 下单/持仓/权益
├── portfolio.py        Portfolio — 快照 + apply_fill
├── audit.py            AuditLog — DuckDB 写入
├── watchdog.py         FileKillSwitch + HeartbeatWatchdog
├── market_calendar.py  SimpleMarketCalendar — 美东时段判断
│
├── monitor_data.py     UI 数据层（live_alpaca_equity + DuckDB 查询）
└── monitor_nice.py     NiceGUI 前端（4 个实况页 + 决策台）
```

---

## 四、安全红线（不可违反）

1. **AI agent 不直连 broker**：只产出 `Advisory` 或 `TradePlan`，统一由 Runtime 执行
2. **仅自动模拟盘执行**：`auto_trade_paper=True`、Alpaca Paper 且 `!kill_switch.engaged()` 才执行
3. **只挂 LMT**：`order_type="LMT"` 硬编码，禁止 market order
4. **AI 安全门前置**：评分缺失、过期或不足时计划直接 `REJECTED`
5. **密钥不入库**：日志不打印 API Key / Secret；`.env` 已从 git 移除

---

## 五、LLM 配置（决策台评分依赖）

`make_client()` 自动降级顺序：
1. **Ollama**（本地，`http://localhost:11434`）
   - 自动发现已安装模型，读 `.env` 的 `OLLAMA_MODEL` 优先
   - Agent 串行执行（`_MAX_WORKERS=1`），避免本地 GPU 争抢
   - 思考链默认关闭，可在 `.env` 设 `OLLAMA_THINK=true` 开启
2. **Anthropic**（Ollama 不可达时，若 `.env` 有 `ANTHROPIC_API_KEY`）
3. **StubLLMClient**（最终兜底，所有评分返回 50，会打 ⚠️ 警告）

### 本机配置（RTX 5070 Ti · 16GB VRAM）

| 模型 | VRAM | 速度 | 推荐 |
|------|------|------|------|
| `qwen2.5:14b` | 8.9GB | 30-60s/次 | **当前使用，首选** |
| `phi4` | 9.1GB | 30-60s/次 | 备选 |
| `gemma4` | 9.6GB | 20-40s/次 | 已安装，可切换 |
| `qwen3.6` | 24GB（溢出 CPU）| 5-15min/次 | 不推荐日常使用 |

切换模型：`.env` 改 `OLLAMA_MODEL=<模型名>`，重启 UI 即生效。

> 决策台显示全 50 分 = Ollama 未运行 或 LLM 超时 → StubLLMClient 在工作。
> 确认 Ollama 在跑：`curl http://localhost:11434/api/tags`

---

## 六、已知局限 / 待完成

| 项目 | 状态 | 说明 |
|------|------|------|
| Fundamental Agent | 计划中 | yfinance P/E EPS，stub 占位 |
| Sentiment Agent | 计划中 | Reddit/Twitter，stub 占位 |
| position_monitor M2 | 待做 | 多层风控（动态止损） |
| ANTHROPIC_API_KEY | 用户配置 | 决策台要出非 50 分必须配置 |

---

## 七、会话启动 Checklist

每次开始前核对：
- [ ] `uv run python -m pytest tests/` — 44 tests 全绿
- [ ] `git status` — 当前在 `main` 分支，无意外修改
- [ ] `.env` 含 `ALPACA_API_KEY` / `ALPACA_API_SECRET` / `BROKER_TYPE=alpaca_paper`
- [ ] 决策台要真实评分 → 确认 Ollama 在跑（`curl http://localhost:11434/api/tags`）且 `OLLAMA_MODEL=qwen2.5:14b`
- [ ] 自动下单仅允许 `BROKER_TYPE=alpaca_paper`；不开启时必须保持 `DRY_RUN`

---

## 八、常用命令

```bash
# 启动监控 UI（nicegui 包没有 __main__.py，不能用 -m nicegui 跑，得直接跑脚本）
uv run python trader/monitor_nice.py
# WebView 原生窗口有问题时，改浏览器模式：
set QUANT_WEB=1 && uv run python trader/monitor_nice.py

# 运行测试
uv run python -m pytest tests/ -v

# 启动计划驱动 runtime（默认 DRY-RUN，不下单）
uv run python -m trader.main

# 查看 DuckDB 数据
uv run python -c "import duckdb; c=duckdb.connect('trade.duckdb'); print(c.execute('SHOW TABLES').df())"
```
