# stock-minute-ai — Claude Code 会话指南

每次会话开始时请先核对以下内容，确保建议和修改与系统架构一致。

---

## 一、项目定位

美股分钟级 AI 辅助交易系统（**AI 辅助，人工审批，不自动下单**）。

- 数据来源：Alpaca Markets（实盘/虚拟盘）
- 持久化：DuckDB（`trade.duckdb` 交易审计，`ai_states.duckdb` AI 状态）
- 前端：NiceGUI（`trader/monitor_nice.py`）
- 测试：`uv run python -m pytest tests/`（44 个测试，必须全绿）

---

## 二、双管道架构（核心）

| 管道 | 文件 | 驱动方式 | 用途 |
|------|------|----------|------|
| **信号管道** | `trader/scheduler.py` | TA 信号 → risk → LMT 订单 | 策略回路 |
| **计划管道** | `trader/runtime.py` | selection → TradePlan → AI → approve → LMT | AI 辅助决策 |

两者**并列运行**，互不干扰。`runtime.py` 是主力，`scheduler.py` 保留兼容。

---

## 三、模块全图

```
trader/
├── config.py           TradingConfig (pydantic)，含 execution_enabled=False
├── models.py           所有数据模型（Bar/TradePlan/Candidate/Advisory…）
├── contracts.py        15 个 Protocol 接口定义
│
├── runtime.py          计划驱动 Pipeline（M1，主运行时）
├── scheduler.py        信号驱动循环（历史兼容）
│
├── selection.py        ConsensusSelector — TA 多空票数 → Candidate 0-100 分
├── plan.py             ATRPlanner — entry/stop/tp 计划
├── allocator.py        EqualWeightAllocator — 仓位分配
├── approval.py         AutoApprover（默认 auto_approve=False → PENDING）
├── risk_engine.py      RiskEngine + evaluate_plan() — 双重风控
│
├── ai/
│   ├── client.py       make_client()：Ollama → Anthropic → Stub 自动降级
│   ├── manager.py      AgentManager — 并行调度 + DuckDB 持久化
│   └── agents/
│       ├── technical.py    TA + LLM 打分
│       ├── news.py         RSS/yfinance 情绪
│       ├── bull_bear.py    LLM 三轮辩论
│       └── web_research.py 热点研究
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

1. **AI 不下单**：agent/LLM 只产出 `Advisory` 或 `TradePlan(DRAFT)`，绝不调用 broker
2. **执行需双开关**：`config.execution_enabled=True` **且** `!kill_switch.engaged()` 才执行
3. **只挂 LMT**：`order_type="LMT"` 硬编码，禁止 market order
4. **人在回路默认开**：`AutoApprover(auto_approve=False)` → 所有计划默认 `PENDING`
5. **密钥不入库**：日志不打印 API Key / Secret；`.env` 已从 git 移除

---

## 五、LLM 配置（决策台评分依赖）

`make_client()` 自动降级顺序：
1. **Ollama**（本地，`http://localhost:11434`）
   - 自动发现已安装模型（不要求必须是 llama3.2）
   - 优先选有 `tools` 能力的模型（如 qwen3.6、gemma4）
   - 只需 `ollama serve` 在后台运行即可，无需额外配置
2. **Anthropic**（Ollama 不可达时，若 `.env` 有 `ANTHROPIC_API_KEY`）
3. **StubLLMClient**（最终兜底，所有评分返回 50，会打 ⚠️ 警告）

> 决策台显示全 50 分 = Ollama 未运行 + 无 Anthropic Key → StubLLMClient 在工作。
> 本机已有 `qwen3.6`（36B）和 `gemma4`（8B），`ollama serve` 即可。

---

## 六、已知局限 / 待完成

| 项目 | 状态 | 说明 |
|------|------|------|
| Fundamental Agent | 计划中 | yfinance P/E EPS，stub 占位 |
| Sentiment Agent | 计划中 | Reddit/Twitter，stub 占位 |
| 交互式审批 | 待做 | Discord / UI 按钮一键 APPROVE |
| position_monitor M2 | 待做 | 多层风控（动态止损） |
| ANTHROPIC_API_KEY | 用户配置 | 决策台要出非 50 分必须配置 |

---

## 七、会话启动 Checklist

每次开始前核对：
- [ ] `uv run python -m pytest tests/` — 44 tests 全绿
- [ ] `git status` — 当前在 `main` 分支，无意外修改
- [ ] `.env` 含 `ALPACA_API_KEY` / `ALPACA_API_SECRET` / `BROKER_TYPE=alpaca_paper`
- [ ] 决策台要真实评分 → 确认 `.env` 有 `ANTHROPIC_API_KEY` 或本地 Ollama 运行中
- [ ] `execution_enabled=False`（默认）→ 修改为 `True` 前需人工确认

---

## 八、常用命令

```bash
# 启动监控 UI
uv run python -m nicegui trader/monitor_nice.py

# 运行测试
uv run python -m pytest tests/ -v

# 启动计划驱动 runtime（默认 DRY-RUN，不下单）
uv run python -m trader.main

# 查看 DuckDB 数据
uv run python -c "import duckdb; c=duckdb.connect('trade.duckdb'); print(c.execute('SHOW TABLES').df())"
```
