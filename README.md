# stock-minute-ai

美股分钟级 AI 自动模拟交易系统：Alpaca 行情与 Paper Trading、多 Agent 分析、确定性风控、DuckDB 审计、NiceGUI 控制台。

> AI 自动交易只允许 Alpaca Paper 模拟盘，不逐笔请求批准。系统明确禁止自动实盘交易。

## 快速开始

要求 Python 3.13+ 和 [uv](https://github.com/astral-sh/uv)。

```powershell
setup.bat
# 在 .env 中填写 Alpaca Paper API Key
启动监控台.bat
```

手动运行：

```powershell
uv run python trader/monitor_nice.py
uv run python -m trader.main
uv run python -m trader.main --symbols AAPL,MSFT --auto-trade --min-ai-score 70
```

未传 `--auto-trade` 时，交易计划只记录为 `DRY_RUN`；开启后，符合全部条件的计划自动提交 Alpaca Paper 限价单。

## 用户侧流程

1. 启动 NiceGUI 决策台，配置并启动 AI Agent。
2. Agent 分析技术面、基本面、宏观、新闻、期权、ETF 流向等信息，将评分写入 `ai_states.duckdb`。
3. Runtime 依次执行选股、ATR 计划、AI 评分门、仓位分配和确定性风控。
4. 未开启自动交易时只保存 `DRY_RUN`；开启时提交耐久化、幂等的 Alpaca Paper `LMT` 订单。
5. Runtime 轮询成交、更新组合与审计记录；kill switch 或启动对账异常会阻止新订单。

## 安全红线

- Agent/LLM 不直接调用 broker，只产出分析结果。
- 自动下单要求 `auto_trade_paper=True` 且 `broker_type=alpaca_paper`。
- 自动实盘配置会被拒绝；自动执行只提交 `LMT` 限价单。
- AI 评分缺失、过期或不足，以及任一风控失败，计划都会被拒绝。
- kill switch、幂等检查和启动对账不可绕过。
- API Key/Secret 不写日志、不进 Git。

## 主要入口

- `trader.monitor_nice`：NiceGUI 用户界面
- `trader.main`：唯一生产交易引擎入口
- `trader.runtime.Runtime`：唯一生产交易循环
- `trader.ai.manager.AgentManager`：Agent 调度与 AI 状态持久化
- `notebooks/research.py`：Marimo 研究工具

代码结构、当前基线、验证命令和 Agent 工作协议以 [AGENTS.md](AGENTS.md) 为唯一权威来源。

## 数据与迁移

以下文件不进 Git：`.env`、`trade.duckdb`、`ai_states.duckdb`、`conf/ui_settings.json`、行情缓存、市场快照和运行日志。数据库是用户历史，不应作为缓存删除。

## 验证

```powershell
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m compileall -q trader tests
.venv\Scripts\python.exe -m ruff check trader tests
```

## 已知局限

- 部分 Agent 使用 yfinance 免费数据，可能滞后或缺字段。
- Ollama 不可用时会降级为 Stub 评分；持续显示 50 分通常代表本地模型未运行或超时。
- 自动交易仅为模拟盘，不代表策略在实盘中可盈利。

本项目仅供个人学习与研究，不构成投资建议。
