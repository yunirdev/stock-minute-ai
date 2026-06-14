# PLAN.md — 美股 AI 交易平台 · 模块化骨架 & 多 Agent 协同开发计划

> 本文件是**权威开发计划**与**多 agent 协同契约**。任何 agent（或人）开始写代码前先读这里。
> 维护规则：接口契约一旦在 M0 冻结，**不得单方面修改**；要改先在本文件的「契约变更记录」追加提议并标注影响面。

---

## 0. 怎么用这份文档

- **人类**：把本文 §6 的「工单」逐个派给独立 agent；§5 是每个模块的精确契约。
- **AI agent**：你会被分配一个 §6 工单。只在你工单指定的文件里写代码；依赖别的模块时**只 import §4/§5 的契约**（数据模型 + Protocol），**绝不 import 别人的实现细节**，也不要改别人的文件。完成后必须带单元测试，并对照工单的「验收」自查。
- **北极星**（§1）发生冲突时以本文件为准；本文件与代码冲突时，说明代码漂移了，需要修代码或在此记录变更。

---

## 1. 项目愿景（北极星）

一个**本地优先、决策优先、AI 辅助**的个人美股交易平台。核心价值不是"更多策略/指标"，而是：

> **把多个信号源（TA 策略 + 未来 AI 模型）收敛成少量「可审核的交易计划」**，常驻运行、定时+异动推送到 Discord，人审核（或按开关自动）后执行；最终走向自动量化 + 自动实盘。

关键特征：
1. **决策优先**：第一屏是"今天买什么/卖什么/为什么"，不是引擎状态。
2. **收敛而非叠加**：多策略 → 共识 → 少量计划。解决"单指标太多、审核不过来"。
3. **纪律化下单**：核心产物是 `TradePlan`（入手价/止损/止盈/持有），**不下 market order**。
4. **人在回路 → 自动**：先 paper + 人工审核，信任建立后逐步放开自动实盘。
5. **AI 是旁路**：AI 产出建议/计划/复盘，**永不直接下单**，必须过确定性的风控层。

---

## 2. 总体架构

### 主链（每一轮按顺序流动）
```
数据 → 选股 → 信号 → 交易计划 → 组合分配 → 风控(多层) → 执行 → 盯盘
data   select strat   plan       allocator    risk        exec   monitor
```

### 旁路与输出
```
触发源:  新闻/异动 (news/events)  ──┐
                                    ├─> 喂入主链 + AI
AI 旁路: 多 agent (ai/agents) ──────┘    产出 Advisory / TradePlan(DRAFT)

输出:    Discord 推送 (notify) · UI 审阅 (monitor) · 账本审计 (portfolio/audit)
复盘:    review (盘后归因) · watchdog (健康看门狗) · 人在回路确认 (approval)
驱动:    runtime (定时 + 异动触发 + 交易日历时段)
```

### 设计铁律
- **模块 = 一个接口契约 + 一个实现**。新增能力 = 写一个实现同一接口的新适配器，**主链一行都不改**。
- 模块之间**只通过 §4 数据模型 + §5 Protocol 通信**，不依赖彼此内部。
- 一切可 stub：接口先齐，实现可占位，保证后续填空不冲突。

---

## 3. 现有资产（复用，不要重写）

| 现有模块 | 提供的契约 | 复用方式 |
|---|---|---|
| `trader/engine.py` | `simulate()` → `SimResult` | 回测/研究唯一引擎，新模块不要再造回测 |
| `trader/models.py` | `Bar/Signal/OrderIntent/Fill/Position/RiskVerdict/PendingOrder/Side/OrderStatus` | 全平台数据模型基座，新模型加在这里 |
| `trader/strategy_core.py` | `compute_signals(df, strategy, **params)` · `STRATEGY_OPTIONS` · `DEFAULT_STRATEGY_PARAMS` | 信号层，selection 共识打分复用它 |
| `trader/strategies/` | `Strategy/IndicatorStrategy/ScriptStrategy` · `registry` | 信号源注册表 |
| `trader/risk_engine.py` | `RiskEngine.evaluate/check_equity/...` | 风控第 1 层（pre-trade），多层风控在其上扩展 |
| `trader/broker/` | `BrokerAdapter` · `PaperBroker` | 执行适配器接口；实盘 = 新增适配器 |
| `trader/order_manager.py` | 订单提交/挂单处理 | 订单状态机在此强化 |
| `trader/portfolio.py` `audit.py` | 账本 + 审计 DuckDB | 账本/审计沿用 |
| `trader/data_cache.py` `data_feed*.py` | `get_bars` · `list_cached_files` | 数据层；多源在此扩展 |
| `trader/monitor_nice.py` `monitor_data.py` | NiceGUI 前端 + 纯数据层 | UI；决策台/选股池接真实数据 |
| `trader/config.py` | `TradingConfig`（pydantic-settings） | 统一配置；新模块配置加字段 |
| `trader/scheduler.py` | 轮询调度 | **改造**成 `runtime`（定时+异动+日历） |

> ⚠️ `trader/ui_prefs.py` 是旧 Streamlit 残留，**不要用**（NiceGUI 用 `conf/ui_settings.json` 持久化）。

---

## 4. 核心数据模型契约（M0 冻结，加在 `trader/models.py`）

现有模型见 §3。**新增以下模型**（agent 实现 M0 时按此写）：

```python
@dataclass
class Candidate:
    """选股输出：一个候选标的及其可解释打分。"""
    symbol: str
    score: float                       # 0-100 综合/共识分
    rank: int
    reasons: Dict[str, Any]            # {"votes": {strategy: +1/-1}, "factors": {...}} 可追溯
    as_of: datetime = field(default_factory=utc_now)

@dataclass
class TradePlan:
    """核心产物：纪律化交易计划（不下 market；entry/stop/tp 都是预设价位）。"""
    plan_id: str
    symbol: str
    side: Side
    action: str                        # OPEN | ADD | REDUCE | CLOSE | HOLD
    entry_price: float                 # 入手价（挂 LMT）
    stop_loss: float                   # 止损价
    take_profit: float                 # 止盈价
    target_weight: float = 0.0         # 目标组合权重（allocator 填）
    qty: float = 0.0                   # 数量（allocator/risk 填）
    confidence: float = 1.0
    rationale: str = ""                # 为什么：哪些信号/agent/新闻
    source: str = "consensus"          # consensus | ai | manual
    status: str = "DRAFT"              # DRAFT | APPROVED | REJECTED | LIVE | CLOSED
    created_at: datetime = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Advisory:
    """AI 旁路产出的建议工件（advisory artifact，带模型元数据，永不直接执行）。"""
    advisory_id: str
    kind: str                          # selection | plan | review | news | risk_review
    agent: str                         # 产出它的 agent 角色名
    payload: Dict[str, Any]            # 结构化内容
    confidence: float = 0.0
    model: str = ""                    # 模型 id / 版本
    created_at: datetime = field(default_factory=utc_now)

@dataclass
class NewsEvent:
    """新闻/异动/日历/社区 触发事件。"""
    event_id: str
    kind: str                          # news | price_move | calendar | community
    symbol: Optional[str]
    title: str
    summary: str = ""
    url: Optional[str] = None
    severity: float = 0.0              # 异动强度 0-1
    ts: datetime = field(default_factory=utc_now)
    source: str = ""

@dataclass
class ReviewReport:
    """盘后复盘归因。"""
    report_id: str
    period: str                        # daily | weekly
    market_summary: str
    portfolio_pnl: float
    attribution: Dict[str, Any]
    trades: list = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)

@dataclass
class Alert:
    """看门狗/系统告警。"""
    level: str                         # info | warn | critical
    source: str
    message: str
    ts: datetime = field(default_factory=utc_now)

@dataclass
class Notification:
    """推送统一载体（notify 用）。"""
    title: str
    body: str
    kind: str = "info"                 # selection | plan | review | news | alert | info
    fields: Dict[str, Any] = field(default_factory=dict)
    plan_id: Optional[str] = None      # 若是计划，带上以支持 Discord/UI 一键审批
```

**数据流串联**（agent 必须遵守的产物链）：
```
NewsEvent ─┐
           ├─> Candidate ─> Signal ─> TradePlan(DRAFT)
Bar ───────┘                              │
                                          ├─ allocator 填 target_weight/qty
                                          ├─ RiskEngine 审 → status=APPROVED/REJECTED
                                          ├─ approval(人/开关) → status=APPROVED
                                          └─ order_manager → OrderIntent(LMT) → Broker → Fill
Advisory  ── 旁路，喂 selection/plan/review/notify，不进执行
```

---

## 5. 模块接口契约（M1+）

> 约定：所有 Protocol 放 `trader/contracts.py`（M0 创建）。各模块实现该 Protocol。
> 每个模块**必带** `tests/test_<module>.py` 和一个可注入的 stub 实现。

### 5.1 `selection` — 选股（🟡 基础版）
- 文件：`trader/selection.py`
- 职责：对 universe 跑「策略共识」打分，输出候选排名。
```python
class Selector(Protocol):
    def select(self, universe: list[str], timeframe: str, as_of: datetime) -> list[Candidate]: ...
```
- 基础版：对每个标的跑 `strategy_core.compute_signals` 的多个策略，统计当前 bar 的看多/看空票数 → `score = 100 * 多票/总票`，填 `reasons["votes"]`。
- 依赖（只读）：`data_cache.get_bars`、`strategy_core`、`models.Candidate`。
- 验收：给定本地有数据的标的列表，返回按 score 降序的 `Candidate`，reasons 可追溯到具体策略票。

### 5.2 `plan` — 交易计划（🟡 基础版）
- 文件：`trader/plan.py`
```python
class Planner(Protocol):
    def make_plan(self, cand: Candidate, latest_bar: Bar, params: dict) -> TradePlan: ...
```
- 基础版：`entry = 现价或回踩价`；`stop_loss = entry - k*ATR`；`take_profit = entry + r*k*ATR`（r 默认 2）。`action` 由是否已持仓决定。
- 依赖：`models`、`data_cache`（取 ATR 所需 bars）。
- 验收：输入 Candidate+bar，输出 entry/stop/tp 三价齐全且 stop<entry<tp（多头），rationale 非空。

### 5.3 `allocator` — 仓位分配（🔴 留接口 + 基础版）
- 文件：`trader/allocator.py`
```python
class Allocator(Protocol):
    def allocate(self, plans: list[TradePlan], equity: float,
                 positions: dict[str, Position]) -> list[TradePlan]: ...
```
- 基础版：等权 + 单标的上限（`config.risk.max_position_pct`）+ 现金约束；填 `target_weight` 与 `qty`。
- 验收：N 个计划分配后总权重 ≤ 1，单个 ≤ 上限；现金不足时按 rank 截断。

### 5.4 `portfolio_manager` — 组合协调（🔴 留接口，stub 可空实现）
- 文件：`trader/portfolio_manager.py`
```python
class PortfolioManager(Protocol):
    def reconcile(self, plans: list[TradePlan], positions: dict[str, Position]) -> list[TradePlan]: ...
```
- 基础版：相关性/集中度检查可先 pass-through（仅留接口与测试桩）。
- 验收：接口存在、被 runtime 调用、stub 返回原计划不报错。

### 5.5 `position_monitor` — 盯盘守护（🟡 基础版）
- 文件：`trader/position_monitor.py`
```python
class PositionMonitor(Protocol):
    def check(self, positions: dict[str, Position], live_plans: dict[str, TradePlan],
              latest: dict[str, Bar]) -> list[TradePlan]: ...   # 返回触发的平仓/调整计划
```
- 基础版：现价跌破 stop_loss → 生成 CLOSE 计划；触及 take_profit → CLOSE/REDUCE。
- 验收：构造跌破止损的 bar，返回对应 CLOSE TradePlan。

### 5.6 `risk` — 多层风控（🔴 在现有 `risk_engine` 上扩展）
- 文件：`trader/risk_engine.py`（扩展，不重写）
- 现有：pre-trade `evaluate` + 日内回撤 `check_equity` + 连败熔断。
- 新增接口（追加方法，向后兼容）：
```python
def evaluate_plan(self, plan: TradePlan, equity: float,
                  positions: dict[str, Position]) -> RiskVerdict: ...   # 计划级审查
def check_portfolio(self, positions: dict[str, Position], equity: float) -> Optional[Alert]: ...  # 组合级
```
- 基础版：`evaluate_plan` 复用 `evaluate` 的仓位/敞口逻辑 + 校验 stop/tp 合理。
- 验收：不合理计划（无止损、超敞口）被拒并给 reason。

### 5.7 `execution` — 执行（🟡 强化现有 order_manager + broker）
- 文件：`trader/order_manager.py`（强化）
- 职责：APPROVED 的 TradePlan → `OrderIntent(order_type="LMT")` → Broker；管理订单状态机（未成交/超时/改撤），复用现有 `PendingOrder`。
- 红线：**只下 LMT**，不下 MKT；`live` 模式需 `config` 开关开启。
- 验收：APPROVED 计划生成 LMT OrderIntent；超时挂单按 `PendingOrder.max_bars_alive` 撤销。

### 5.8 `notify` — 推送（🟡 基础版）
- 文件：`trader/notify.py`
```python
class Notifier(Protocol):
    def send(self, note: Notification) -> bool: ...
```
- 基础版：`ConsoleNotifier`（打印）+ `DiscordNotifier`（webhook，URL 从 config/env 读，缺失则降级到 console）。
- 红线：webhook URL 只从本地 config/env 读，**不硬编码**；发送前不泄露密钥。
- 验收：传 Notification，console 实现必成功；discord 实现无 URL 时优雅降级。

### 5.9 `ai/agents` — 多 agent 分析（⚪ 接口 + 1 个实现 + 其余 stub）
- 文件：`trader/ai/agents/` (`base.py` + 各角色)
```python
class Agent(Protocol):
    role: str          # scout | sentiment | planner | risk_reviewer | reviewer | orchestrator
    def run(self, ctx: "AgentContext") -> list[Advisory]: ...
```
- 角色：`scout`(选股侦察) `sentiment`(新闻情绪) `planner`(制订计划) `risk_reviewer`(计划二审) `reviewer`(复盘) `orchestrator`(汇总各 agent → 最终建议)。
- 基础版：定义 `Agent`/`AgentContext` + 实现 `orchestrator`（调度其他 agent，先用规则版聚合）；其余角色给返回空 Advisory 的 stub。
- 红线：**agent 只产出 `Advisory`/`TradePlan(status=DRAFT)`，绝不调用 broker / order_manager**。LLM 调用走可注入的 client，便于离线测试。
- 验收：orchestrator 跑通、产出 Advisory；stub agents 不报错；无任何下单调用。

### 5.10 `news` — 新闻/异动（⚪ 接口 + stub）
- 文件：`trader/news.py`
```python
class NewsSource(Protocol):
    def poll(self, since: datetime) -> list[NewsEvent]: ...
```
- 基础版：`PriceMoveSource`（用本地 bars 算涨跌幅异动，可真做）+ `NewsSourceStub`（占位）。
- 验收：PriceMove 源能从 bars 产出 price_move 类 NewsEvent。

### 5.11 `review` — 复盘归因（🟡 基础版）
- 文件：`trader/review.py`
```python
class Reviewer(Protocol):
    def review(self, period: str, as_of: datetime) -> ReviewReport: ...
```
- 基础版：从 `monitor_data`/账本读当日 equity/fills，算 PnL + 简单归因，组装 ReviewReport（再交 notify 推送）。
- 验收：有数据时产出非空 ReviewReport；无数据时优雅返回空报告。

### 5.12 `watchdog` + `kill_switch` — 健康看门狗 & 急停（🔴）
- 文件：`trader/watchdog.py`
```python
class Watchdog(Protocol):
    def check(self) -> list[Alert]: ...        # 数据新鲜度/心跳/broker 连接/异常波动
class KillSwitch(Protocol):
    def engaged(self) -> bool: ...
    def engage(self, reason: str) -> None: ...  # 一键全停，写状态文件
```
- 基础版：watchdog 检查 heartbeat 新鲜度 + 数据缺口 → Alert；kill_switch 用状态文件（`conf/kill_switch.json`），runtime 每轮先查。
- 验收：心跳过期产出 critical Alert；engage 后 runtime 跳过执行。

### 5.13 `universe` + `calendar`（🟡 基础版）
- `trader/universe.py`：`get_universe(name) -> list[str]`（自选池/板块/全市场，从 config/文件读）。
- `trader/market_calendar.py`：`session_now() -> "pre"|"open"|"post"|"closed"`（美股时段，先用简单时区规则）。
- 验收：universe 返回标的列表；calendar 在不同时间返回正确时段。

### 5.14 `runtime` — 常驻编排（🔴 改造 scheduler）
- 文件：`trader/runtime.py`（由 `scheduler.py` 演进）
- 职责：常驻循环；按 calendar 时段 + 定时 + news 异动**触发**一轮 pipeline：
```
每轮: kill_switch?→ watchdog → news.poll → (盘前)selection → plan → allocator
      → portfolio_manager → risk → approval(人/开关) → execution → portfolio
      → (盘中)position_monitor → (盘后)review → notify
```
- 红线：execution 仅在 `config.execution.enabled` 且 `not kill_switch.engaged()` 时进行；默认 paper。
- 验收：能在本地跑通一轮（paper），各阶段有日志 + 关键产物落审计；kill_switch/halt 时跳过下单。

### 5.15 `approval` — 人在回路（🟡 基础版）
- 文件：`trader/approval.py`
```python
class Approver(Protocol):
    def decide(self, plan: TradePlan) -> str: ...   # APPROVED | REJECTED | PENDING
```
- 基础版：`AutoApprover`（按 config 开关：关=全部 PENDING 等人审；开=按规则自动 APPROVED）+ 记录到审计。Discord/UI 一键审批后续接入。
- 验收：开关关闭时计划停在 PENDING（不下单）；开启时按规则放行。

---

## 6. 多 Agent 协同方案

### 6.1 协同铁律
1. **契约冻结**：M0 完成后，`models.py` 新模型 + `contracts.py` 的 Protocol 不得单方面改。需变更 → 在 §10「契约变更记录」追加，标注受影响模块。
2. **单一所有权**：每个 agent 只写自己工单的文件，**不改他人文件**。需要他人能力 → 通过其 Protocol 注入（构造函数传依赖）。
3. **只依赖契约**：`import` 只允许引用 `models` / `contracts` / 你工单声明的依赖接口。禁止 import 他人实现模块。
4. **自带测试**：每个模块交付 `tests/test_<module>.py`，覆盖基础版主路径 + 空数据/异常路径。
5. **stub 优先**：依赖未完成的模块时，用其 stub（M0 一并提供 stub fixtures）。
6. **集成归属**：`runtime` 由集成 agent 负责串联；其他 agent 不在 runtime 里塞逻辑。

### 6.2 依赖批次（DAG，可并行的在同一批）
```
批次 0（串行，阻塞全部）：M0 契约层
  - 扩展 models.py（§4 新模型）
  - 新建 contracts.py（§5 所有 Protocol）
  - 新建各模块 stub + tests/conftest.py fixtures

批次 1（全并行，只依赖契约）：
  selection · plan · allocator · notify · news · review
  · universe · market_calendar · watchdog · ai/agents · approval

批次 2（依赖批次1接口）：
  portfolio_manager · position_monitor · execution(order_manager 强化) · risk(扩展)

批次 3（集成）：
  runtime 编排串联 · UI 决策台/选股池接 selection/plan · kill_switch 接入

批次 4（未来）：
  实盘 broker 适配器 · Discord/UI 交互式审批 · 多源数据(基本面/情绪/期权)
```

### 6.3 Agent 工单模板
```
## 工单：<module>
目标：一句话
交付物：trader/<module>.py（或目录） + tests/test_<module>.py
实现接口：<contracts.Xxx Protocol>
可读依赖：<models / 某 Protocol>（仅接口）
禁止触碰：其他模块实现、runtime
基础版范围：<本轮只做什么>
验收标准：<可自查的 1-3 条>
红线：<若涉及下单/密钥/AI，写明安全约束>
```

---

## 7. 里程碑

| 里程碑 | 内容 | 完成标志 |
|---|---|---|
| **M0** | 契约层：models 新模型 + contracts.py + stubs + fixtures | `pytest` 通过；所有 Protocol 可 import |
| **M1** | 主链 paper 跑通：selection→plan→allocator→risk→execution→portfolio | runtime 在本地跑完一轮，审计有记录（paper） |
| **M2** | 组合与盯盘：allocator 真分配 + position_monitor 止损止盈 + 多层 risk | 构造行情能触发止损平仓 |
| **M3** | 旁路：notify(Discord) + news(异动) + review(复盘) + ai/orchestrator | 一轮结束推送选股/计划/复盘到 console/Discord |
| **M4** | 人在回路：approval + watchdog + kill_switch + UI 接真实决策数据 | 计划默认 PENDING 待审；急停可跳过下单；UI 决策台显示真实 Candidate/TradePlan |
| **M5** | 自动实盘（未来）：实盘 broker 适配器 + 交互式审批 + 自动开关分级放开 | 实盘适配器通过 paper↔live 切换测试；默认仍 paper |

---

## 8. 安全护栏（红线，写进每个相关模块）

1. **AI 不下单**：agent/LLM 只产出 `Advisory` 或 `TradePlan(status=DRAFT)`；任何下单路径都不得从 ai/ 直接发起。
2. **执行需开关 + 非急停**：`execution` 仅在 `config.execution.enabled and not kill_switch.engaged()` 时进行；默认 paper broker。
3. **只挂 LMT**：不下 market order（用户要求 + 更稳）。
4. **多层风控前置**：pre-trade + 计划级 + 组合级 + 日内回撤 + 连败熔断，任一拒绝即不执行。
5. **人在回路默认开**：`approval` 默认让计划停在 PENDING；自动放行需显式配置并分级。
6. **paper/live 隔离**：环境通过 config 明确切换，UI 显著标识当前环境，防误操作。
7. **密钥不入库**：webhook/broker key 只从 `.env`/config 读；`.env` 已 gitignore；日志/推送不打印密钥。
8. **审计可追溯**：每个 TradePlan 落审计时记录 rationale（哪些信号/agent/新闻）。

---

## 9. 目录结构（目标）

```
trader/
  models.py            # 数据模型（M0 扩展）
  contracts.py         # 所有 Protocol（M0 新建）
  config.py            # 配置（按需加字段）
  data_cache.py  data_feed*.py        # 数据层
  universe.py  market_calendar.py     # 标的池 / 交易日历
  strategy_core.py  strategies/       # 信号层（已有）
  selection.py         # 选股
  plan.py              # 交易计划
  allocator.py  portfolio_manager.py  # 组合层
  risk_engine.py       # 多层风控（扩展）
  order_manager.py  broker/           # 执行层
  position_monitor.py  # 盯盘
  approval.py          # 人在回路
  notify.py            # 推送
  news.py              # 新闻/异动
  review.py            # 复盘
  watchdog.py          # 看门狗 / 急停
  ai/agents/           # 多 agent（base + 角色）
  runtime.py           # 常驻编排（由 scheduler 演进）
  engine.py            # 回测引擎（已有）
  portfolio.py  audit.py              # 账本 / 审计
  monitor_nice.py  monitor_data.py    # UI
tests/                 # 每模块一份
docs/                  # 设计文档（含归档的旧架构文档）
notebooks/research.py  # Marimo 研究端
PLAN.md                # 本文件
```

---

## 10. 编码 & 测试约定

- Python ≥ 3.13；全量类型注解；数据模型用 `@dataclass`；接口用 `typing.Protocol`。
- 依赖注入：模块通过构造函数接收依赖接口，便于替换/测试（不在模块内 new 具体实现）。
- 错误处理：对外函数失败返回空/降级，不让单点异常拖垮 runtime（runtime 每阶段 try/except + Alert）。
- 注释与文案用中文；代码标识符英文。
- Lint：`ruff`；测试：`pytest`，每模块基础版主路径 + 边界（空数据/异常）。
- 时间统一 UTC（`models.utc_now`）。

### 契约变更记录（变更接口必须在此追加）
- *(暂无。M0 冻结后在此登记每次契约变更：日期 / 改了什么 / 影响哪些模块。)*

---

## 11. 验收总清单（逐项打勾即"骨架完成"）

- [ ] M0：`models` 新模型 + `contracts.py` + 各模块 stub + `pytest` 绿
- [ ] selection 用共识打分输出可追溯 Candidate
- [ ] plan 产出 entry/stop/tp 齐全的 TradePlan
- [ ] allocator 等权+上限分配，总权重 ≤ 1
- [ ] risk 计划级 + 组合级审查可拒绝不合理计划
- [ ] execution 只下 LMT，受开关+急停控制
- [ ] position_monitor 能触发止损/止盈平仓计划
- [ ] notify 推送到 console，Discord 无 URL 时降级
- [ ] news PriceMove 源从 bars 产异动事件
- [ ] review 产出复盘报告
- [ ] ai/orchestrator 跑通、只产 Advisory、零下单调用
- [ ] watchdog + kill_switch：异常告警 + 急停阻断执行
- [ ] approval 默认 PENDING（不自动下单）
- [ ] runtime 串联跑通一轮（paper），审计有记录
- [ ] UI 决策台/选股池接真实 selection/plan
- [ ] 安全护栏 §8 全部生效
```
