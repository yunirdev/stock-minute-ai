# stock-minute-ai

**美股分钟级 AI 辅助交易系统** —— Alpaca 实时数据 + 多 Agent AI 分析 + DuckDB 全量审计 + NiceGUI 决策前端。

> ⚠️ **AI 自动交易仅支持 Alpaca Paper 模拟盘。** Agent/LLM 不直接调用券商；交易计划必须通过 AI 评分门、确定性风控、kill switch、幂等与启动对账后，才由 Runtime 提交 `LMT` 限价单。未开启「AI 自动交易」时只记录 `DRY_RUN`。详见下方「安全模型」。

---

## 这是什么

一个本地运行、零外部服务依赖（除 Alpaca/yfinance 这类公开数据 API）的交易辅助系统：

- 从 [Alpaca Markets](https://alpaca.markets/) 拉实时/历史分钟级行情（虚拟盘或实盘账户）
- 10+ 个独立 AI/算法 Agent（技术面、基本面、宏观、期权、ETF 资金流、机构持仓、多空辩论……）并行分析候选标的，产出评分与理由
- 本地 Ollama 跑开源 LLM 做综合打分（免费、不出本机；没装 Ollama 也能跑，自动降级）
- 开启「AI 自动交易」后，达标计划由 Runtime 自动提交至 Alpaca Paper；关闭时记录为 `DRY_RUN`
- 一切都落 [DuckDB](https://duckdb.org/) 单文件数据库：零基础设施，没有数据库服务要维护
- [NiceGUI](https://nicegui.io/) 做的本地决策台前端：实况监控 + AI 决策面板 + 历史回测

---

## 安全模型（不可绕过的红线）

| 红线 | 说明 |
|------|------|
| **AI agent 不直连 broker** | Agent/LLM 只产出 `Advisory` 或 `TradePlan`；只有 Runtime 可以经过统一安全链提交订单 |
| **仅自动模拟盘执行** | `auto_trade_paper=True`、`broker_type=alpaca_paper` 且 kill switch 未触发时才允许提交订单 |
| **只挂限价单** | `order_type="LMT"` 硬编码在下单路径里，没有市价单选项 |
| **评分与风控前置** | AI 评分过期/缺失/不足或任一确定性风控不通过，计划直接 `REJECTED` |
| **密钥不落库** | 日志不打印 API Key/Secret；`.env` 不进 git |

虚拟盘（`BROKER_TYPE=alpaca_paper`）下勾选「AI 自动交易」后，引擎会在 AI 评分与全部风控达标时自动提交限价单。该模式不能与 `alpaca_live` 组合，配置阶段会直接拒绝启动。

---

## 环境要求

- Python ≥ 3.13
- [uv](https://github.com/astral-sh/uv) 包管理器（没装的话装它的脚本会自动拉）
- 一个 [Alpaca Markets](https://alpaca.markets/) 账号（免费注册，先用 Paper Trading 的 key 即可，不需要实盘资金）
- **可选**：本地装 [Ollama](https://ollama.com/) 跑开源 LLM 做 AI 综合评分（不装也能跑，AI 评分会显示固定的 50 分兜底值，不影响其他功能）
- **可选**：Discord Bot/Webhook（推送晨报、复盘和交易通知）、Finnhub API key（额外新闻源）、Anthropic API key（Ollama 不可用时的云端兜底）

---

## 快速开始（Windows）

```bash
git clone <repo-url>
cd stock-minute-ai
setup.bat
```

`setup.bat` 会自动：检查/安装 Python 依赖（`uv sync`）、从 `.env.example` 生成 `.env`（如果还没有）、检测本机是否装了 Ollama 并列出已装模型。

**它不会自动做的事**，需要你手动补：

1. **打开 `.env`，填入真实的 Alpaca key**（必填）：
   ```dotenv
   ALPACA_API_KEY=你的key
   ALPACA_API_SECRET=你的secret
   ```
   Paper Trading 的 key 在 https://app.alpaca.markets/paper/dashboard/overview 获取。**Paper 和 Live 的 key 是分开的，别混用**，混用会直接 401。

2. **如果想要真实 AI 评分**（不装这步也能跑，只是评分全是兜底的 50 分）：
   - 装 [Ollama](https://ollama.com/download)
   - 拉一个模型：`ollama pull qwen2.5:14b`（推荐，约 9GB，金融推理质量和速度的平衡点）
   - **模型选择跟你的显卡显存有关**，不是固定的——显存富裕可以换更大的模型，显存紧张（比如 8GB 卡）建议换 `qwen2.5:7b` 一类更小的模型。`.env` 里的 `OLLAMA_MODEL` 改成你实际拉的模型名即可，系统会自动发现已安装模型，配置的模型不存在时自动换一个可用的，不会崩。

3. **跑一遍测试，确认环境没问题**：
   ```bash
   uv run python -m pytest tests/ -v
   ```
   应该全绿（写这份文档时是 42 个测试）。

4. 双击 `启动监控台.bat`，或手动：
   ```bash
   uv run python trader/monitor_nice.py
   ```
   默认开桌面 WebView 窗口；如果 WebView 在你机器上有问题，设 `QUANT_WEB=1` 环境变量后再跑，会改用浏览器打开 `http://127.0.0.1:8080`。

---

## 常用命令

```bash
# 启动决策台 UI（桌面窗口模式）
uv run python trader/monitor_nice.py

# 启动决策台 UI（浏览器模式，WebView 有问题时用）
# cmd:        set QUANT_WEB=1 && uv run python trader/monitor_nice.py
# PowerShell: $env:QUANT_WEB=1; uv run python trader/monitor_nice.py

# 启动计划驱动 runtime 引擎（默认 DRY-RUN，不真实下单）
uv run python -m trader.main
uv run python -m trader.main --symbols AAPL,MSFT --auto-trade --min-ai-score 70

# 手动触发一次晨报（正常由引擎在美东 9AM 自动发）
uv run python -m trader.morning_brief

# 跑全部测试
uv run python -m pytest tests/ -v

# 看 DuckDB 里有什么
uv run python -c "import duckdb; c=duckdb.connect('trade.duckdb'); print(c.execute('SHOW TABLES').df())"
```

---

## 架构总览

```
Alpaca Markets (行情 + 虚拟盘/实盘下单)
        │
        ├─ trader/runtime.py ──► selection → plan(ATR) → AI safety → risk → LMT 下单
        │                         （计划驱动管道，当前主力）
        │
        └─ trader/scheduler.py ──► TA 信号 → risk → LMT 下单
                                    （信号驱动管道，历史兼容，与上面并行不冲突）

trader/ai/manager.py 并行调度 10+ 个 Agent（技术面/基本面/宏观/期权/ETF资金流/
机构持仓/新闻情绪/多空辩论...），结果写入 ai_states.duckdb，决策台实时展示。

trader/monitor_nice.py（NiceGUI）── 本地前端：实况监控 / AI 决策台 / 历史回测
trader/morning_brief.py / discord_report.py ── 晨报、复盘、AI 分析推送 Discord

全部交易/审计数据 → trade.duckdb；AI 状态/历史 → ai_states.duckdb
```

更细的模块清单和会话开发须知见 [CLAUDE.md](CLAUDE.md)。

---

## 项目结构

```
stock-minute-ai/
├── trader/
│   ├── main.py              # CLI 入口：启动 runtime 引擎
│   ├── runtime.py           # 计划驱动管道（主力）
│   ├── scheduler.py         # 信号驱动管道（兼容保留）
│   ├── config.py            # pydantic 配置（.env 解析）
│   ├── models.py            # 全部数据模型
│   ├── contracts.py         # Protocol 接口定义
│   ├── selection.py / plan.py / allocator.py / risk_engine.py
│   │                        # 选股 → 计划 → 仓位分配 → AI 安全门 → 风控
│   ├── ai/
│   │   ├── manager.py       # Agent 并行调度 + DuckDB 持久化
│   │   ├── client.py        # Ollama → Anthropic → Stub 自动降级
│   │   └── agents/          # 10+ 个独立 Agent（technical/fundamental/macro/
│   │                        #   options/etf_flow/elite_holdings/quant/news/
│   │                        #   web_research/bull_bear...）
│   ├── broker/               # Alpaca 下单封装（paper + live）
│   ├── strategies/ + strategy_core.py + factors/
│   │                        # 策略库 + 技术因子（供回测/选股复用）
│   ├── backtest/             # 因子分析回测引擎
│   ├── teams/                # 团队协作框架（市场环境感知、维护任务）
│   ├── monitor_nice.py       # NiceGUI 决策台前端（主入口）
│   ├── monitor_data.py       # 前端数据层（DuckDB 查询 + Alpaca 实时权益）
│   ├── morning_brief.py      # 晨报生成（宏观/新闻/板块/持仓）
│   ├── discord_report.py / notify.py
│   │                        # Discord 推送
│   ├── news.py / calendar_events.py
│   │                        # 新闻源（华尔街见闻/SEC EDGAR/Finnhub/价格异动）+ 财报日历
│   └── watchdog.py           # Kill switch + 心跳监控
├── tests/                    # pytest 测试
├── conf/                     # 运行期生成：UI 偏好、市场状态缓存（不进 git）
├── notebooks/research.py     # Marimo 响应式研究 notebook（`marimo edit notebooks/research.py`）
├── setup.bat                 # 一键安装脚本
├── 启动监控台.bat              # 一键启动决策台
├── .env.example               # 配置模板，复制为 .env 后填真实值
└── CLAUDE.md                  # 架构细节 + 会话开发须知
```

---

## 换新电脑迁移须知

代码本身不依赖任何本机路径，`git clone` 到哪都能跑。但以下几类东西**不会**跟着 git 走（且本来就不该进 git）：

| 不会自动迁移的东西 | 是什么 | 要不要带 |
|---|---|---|
| `.env` | API key / 配置 | **必须重新填**，不能复制旧的就完事——如果旧机器和新机器要同时跑，注意 Paper/Live key 别串 |
| `trade.duckdb` | 全部交易审计记录 | 想保留历史就手动拷过去；不拷就是全新空白账本 |
| `ai_states.duckdb` | AI Agent 历史评分/决策记录 | 同上，纯历史数据，不拷不影响功能 |
| `conf/ui_settings.json` | 决策台 UI 偏好（上次打开的 tab、自选标的等） | 不拷的话新机器上是默认值，不影响功能 |
| `conf/market_regime.json` | 市场环境状态缓存 | 会自动重新生成，不用管 |
| Ollama 本身 + 模型 | 本地 LLM 推理 | 必须在新机器重新装 Ollama + `ollama pull`，且**模型选择要按新机器显卡重新评估**，不是直接照搬 |

Alpaca/yfinance 数据抓取本身跟机器无关，只要网络通、API key 配对，新机器上抓数据的行为和准确性跟原机器完全一致。

---

## 已知局限

- Fundamental / 宏观等部分 Agent 依赖 yfinance 免费数据，存在 24-48h 滞后，且小盘股部分字段经常缺失
- 无实时财报会议纪要、无管理层语调分析这类需要付费数据源的深度信息
- 回测支持单边滑点与手续费；决策台默认计入 5 bps 单边滑点，可按券商和标的调整
- 决策台显示 AI 评分全部是 50 分 → 说明 Ollama 没在跑或超时，StubLLMClient 在兜底，不代表系统出错

---

## 免责声明

本项目仅用于个人学习与研究，**不构成任何投资建议**。AI 自动交易会直接操作 Alpaca Paper 模拟盘，不逐笔请求确认；所有模拟盈亏与配置风险由使用者自行承担。当前自动交易明确禁止 `alpaca_live`。
