# 迁移任务总表

状态：Active  
总路线：[PROJECT_MIGRATION_PLAN.md](PROJECT_MIGRATION_PLAN.md)  
目标架构：[TARGET_ARCHITECTURE.md](TARGET_ARCHITECTURE.md)  
当前闭环验收：[CLOSED_LOOP_ACCEPTANCE.md](CLOSED_LOOP_ACCEPTANCE.md)

## 以后只需要发送这一句

```text
按 docs/MIGRATION_TASK_BOARD.md 继续下一项。读取 AGENTS.md，选择表中第一个依赖已完成的 TODO/VERIFY 任务；一次只完成一个任务，运行该项相关测试、全量 pytest、Ruff 和 compileall，通过后更新任务表状态与验证证据。不要改任务明确排除的范围，不要覆盖现有用户改动。
```

如需持续执行，可发送：

```text
按 docs/MIGRATION_TASK_BOARD.md 从下一项开始持续执行；严格逐项验收，某项失败就停在该项修复，不跳过，不扩大范围。
```

## 状态与执行规则

| 状态 | 含义 |
| --- | --- |
| `TODO` | 尚未开始 |
| `VERIFY` | 代码可能已有，但必须重新验证，不能直接视为完成 |
| `IN_PROGRESS` | 正在处理；同一时间只能有一个 |
| `BLOCKED` | 外部依赖阻塞，必须填写阻塞记录 |
| `DONE` | 所有验收通过且已填写证据 |

严格按 ID 顺序执行。只有依赖全部 `DONE` 才能开始下一项。每项只修改“范围”内模块、必要测试、迁移文档和 `AGENTS.md`。所有代码任务统一运行：相关测试、`.venv\Scripts\python.exe -m pytest tests -q`、Ruff、compileall。

## A. 当前生产路径正确性

| ID | 状态 | 依赖 | 范围 | 目标 | 明确不做 | 专项验收 |
| --- | --- | --- | --- | --- | --- | --- |
| A01 | DONE | — | `paper_decision.py` → `plan.py` → Runtime + tests | BUY/SELL 方向显式贯穿，SELL 绝不变成 BUY | UI、数据源、TA、DB 迁移 | BUY/SELL 回归；错误方向用例通过 |
| A02 | DONE | A01 | `runtime.py`、`allocator.py`、`risk_engine.py`、`order_store.py` | 现有仓位+开放买单+新单不超过累计仓位上限 | UI、每日研究、遗留 Agent | 已有仓位、重复 tick、开放单、部分成交、重启均覆盖；超限无 submit |
| A03 | DONE | A02 | `risk_engine.py`、计划风险字段、Runtime | 强制实际执行路径的单笔最大风险 | 不改配置默认值、UI | 超风险拒绝、边界通过、非法价格 fail-closed |
| A04 | DONE | A03 | `order_store.py`、Alpaca adapter、portfolio、Runtime | 证明幂等、部分成交增量、UNKNOWN 不重复提交 | 不猜测 broker 状态、不自动平仓 | 重复计划、网络未知、重复轮询、重启回归 |
| A05 | DONE | A04 | Runtime 启动对账、portfolio 恢复、order store | broker 仓位/订单/成交可恢复；无法解释时阻塞新单 | 不删数据库、不接管未知订单 | 空本地库+broker 仓位、开放单、recent fills、API 失败测试 |
| A06 | DONE | A05 | Kill Switch、watchdog、Runtime submit 边界 | Paper-only、LMT-only、Kill Switch 覆盖全部入口 | 不加人工审批、不授权 live | 静态调用检查与负向端到端测试 |
| A07 | DONE | A06 | Paper smoke harness、运行手册、审计查询 | 可重复演练 BUY/SELL/拒绝/部分成交/UNKNOWN/重启 | 不连接 live、不改策略 | 隔离 Paper 演练；订单完整追溯到 plan/risk/idempotency/audit |

## B. 统一研究快照

| ID | 状态 | 依赖 | 范围 | 目标 | 明确不做 | 专项验收 |
| --- | --- | --- | --- | --- | --- | --- |
| B01 | DONE | A07 | `models.py`、snapshot store/migration、tests | 定义 `ResearchSnapshot`、source manifest、quality、时间与版本 | 不改 Runtime 输入、不加外部 API | 向后兼容 schema；序列化、迁移、非法时间/质量测试 |
| B02 | DONE | B01 | daily research、data cache、strategy statistics | 对当前数据调用影子写快照 | 不切读取路径、不改变研究结论 | 实际输入与快照逐字段对照；失败来源可见 |
| B03 | DONE | B02 | 快照冻结、去重、重放、保留策略 | 同一研究 run 使用不可变快照 | 不删除旧记录 | 不可变、重复写、跨日、未来时间、重放测试 |
| B04 | DONE | B03 | 快照质量报告 | 连续 10 个交易日可审计 | 不影响下单、不改 UI 主流程 | 关键字段覆盖率达标，所有差异已分类 |

## C. Data Hub 与多维数据

| ID | 状态 | 依赖 | 范围 | 目标 | 明确不做 | 专项验收 |
| --- | --- | --- | --- | --- | --- | --- |
| C01 | DONE | B04 | Data Hub、source registry、质量策略 | 统一适配器、超时、TTL、缓存和降级契约 | 不切生产读取、不让 Agent 下单 | registry、质量、超时、缓存、降级单测 |
| C02 | DONE | C01 | Alpaca market/account/order、本地 cache、Yahoo fallback | 统一行情和 broker 事实，Alpaca 为主源 | Yahoo 不作无标记执行价格 | 双读差异测试；所有降级显式 |
| C03 | DONE | C02 | SEC EDGAR、财务/公告/内幕 adapter | 统一公司与监管事实 | 不用 LLM 补造事实 | source/as_of/修订/缺失/限流测试 |
| C04 | DONE | C03 | Finnhub、Nasdaq、WallStreetCN、Yahoo/RSS | 统一新闻、事件与日历 | 新闻不直接下单 | 去重、时间边界、来源冲突、降级测试 |
| C05 | DONE | C04 | FRED、StockTwits、Reddit、Polymarket | 补齐宏观、情绪、预期 | 社交情绪不作 broker 事实 | 新鲜度、覆盖、失败、低质量标记测试 |
| C06 | DONE | C05 | 双读比较与质量日报 | 连续 20 日无未分类关键差异 | 不切执行输入 | 关键差异为零或有批准规则；配额、延迟、失败率达标 |

## D. TradingAgents 接入

| ID | 状态 | 依赖 | 范围 | 目标 | 明确不做 | 专项验收 |
| --- | --- | --- | --- | --- | --- | --- |
| D01 | DONE | C06 | TA subprocess、worker payload、research store | 输入/输出关联 snapshot/run/model/data version | 不在 tick 内运行、不导入 broker | payload、超时、崩溃、非法/陈旧输出测试 |
| D02 | DONE | D01 | TA 外部数据登记 | TA 补充来源进入 snapshot manifest | 不保留不透明第二管线 | 每次调用都有 source/as_of/quality/failure；可重放 |
| D03 | DONE | D02 | daily freeze、AI trust gate | Runtime 只读当天冻结可信证据 | 不增加 enabled/shadow 双路径 | stale/untrusted/wrong-date/重复 run fail-closed |
| D04 | BLOCKED | D03 | 每日批次报告 | 连续 20 日可追溯研究 | 不扩大 universe | 成功率、耗时、覆盖、模型失败和降级有报告 |
| D04-ACCEL | DONE | D03 | 批次质量历史/合成门禁、真实 API 客户端验证 | 解除后续代码迁移等待，不替代 D04 自然观察 | 不把历史/合成证据写成真实成功日、不切执行输入 | 20 日合成正/负门禁、真实失败库报告、两个模型真实 API/TA 客户端调用、失败恢复测试 |

## E. 持仓计划生命周期

| ID | 状态 | 依赖 | 范围 | 目标 | 明确不做 | 专项验收 |
| --- | --- | --- | --- | --- | --- | --- |
| E01 | DONE | D04-ACCEL | models、position plan store/migration | 定义成交基线和 `PositionPlan` 版本链 | 不改下单、不删旧记录 | 兼容迁移、版本并发、非法转换、恢复测试 |
| E02 | DONE | E01 | fill handling → PositionPlan | 首次/部分成交维护同一持仓计划 | 每日研究不覆盖计划 | 首次、分批、重复 fill、减仓、清仓测试 |
| E03 | DONE | E02 | `InvalidationEvent` store/validators | 定义价格、broker、公司行动、交易限制、策略失效事件 | LLM 文本不单独触发 | 来源、时间、去重、无效事件测试 |
| E04 | DONE | E03 | evaluator → `PositionAdjustment` → OrderIntent | 自动减仓、退出、止损收紧 | 多头止损不放宽、不人工审批 | 事件→版本→订单端到端；重复事件不重复下单 |
| E05 | DONE | E04 | Runtime reconciliation + PositionPlan recovery | 重启恢复计划、事件游标和未完成调整 | 不猜测未知状态 | 开放调整单、部分成交、重启、未知订单测试 |
| E06 | BLOCKED | E05 | Paper 持仓报告 | 连续 30 日仓位可双向解释 | 不扩新策略 | 零静默改写、零重复调整、broker/本地/计划一致 |
| E06-ACCEL | DONE | E05 | 30 日隔离合成门禁 + 真实 Paper 首日对账 | 解除后续代码迁移等待，不替代 E06 自然观察 | 不把合成日写入 REAL、不伪造 broker 仓位 | 合成 30 日正/负门禁；REAL/SYNTHETIC 隔离；真实 broker/local/plan 首日一致 |

## F. 执行状态机收口

| ID | 状态 | 依赖 | 范围 | 目标 | 明确不做 | 专项验收 |
| --- | --- | --- | --- | --- | --- | --- |
| F01 | DONE | E06-ACCEL | CandidatePlan、FinalTradePlan、OrderIntent | 唯一状态机和稳定 ID/版本引用 | 不改 UI、不删兼容层 | 非法转换 fail-closed；有效期/证据/方向/风险测试 |
| F02 | DONE | F01 | Runtime production callers | 新仓和调整只走唯一管线 | 不加第二 submit 入口 | 调用图和静态测试证明无旁路 |
| F03 | DONE | F02 | 已验证无调用的旧适配层/状态/开关 | 删除重复生产路径 | 不删用户 DB/日志/偏好/在用研究工具 | `rg` 无生产调用；测试与文档同步 |
| F04 | DONE | F03 | fills、PositionPlan、market replay、performance attribution store | 把计划、实际成交、滑点、收益和风险事件归入同一交易 episode | 不让 LLM 改写成交事实 | 部分成交、减仓、清仓、跨日、重启归因测试 |
| F05 | DONE | F04 | review artifact、error taxonomy、explainability | 生成事实/决策/执行/结果分层的冻结复盘记录 | 不把亏损自动解释成策略失效 | 成功、拒绝、无成交、数据失败、broker 失败均可回放 |
| F06 | DONE | F05 | strategy candidate/version store、experiment boundary | 复盘只能产生不可变策略候选版本 | 不直接修改生产参数、不在线训练后立即下单 | 数据/代码/参数版本、训练/holdout 边界、重复生成和恢复测试 |
| F07 | DONE | F06 | holdout、历史回放、Paper champion/challenger、promotion audit | 用量化证据晋升或拒绝候选并保留回滚版本 | 不以单次盈亏晋升、不承诺盈利 | 样本外指标、成本/滑点、回撤、最小样本、拒绝原因、回滚测试 |

## G. 全市场筛选

| ID | 状态 | 依赖 | 范围 | 目标 | 明确不做 | 专项验收 |
| --- | --- | --- | --- | --- | --- | --- |
| G01 | DONE | F07 | universe registry/version、asset metadata | 建立股票/ETF/基金范围版本 | 不把全部标的送给 LLM | 资产类型、退市、不可交易、重复、版本测试 |
| G02 | DONE | G01 | deterministic screening、holdout、流动性/质量 | 形成可解释重点池 | 不改变持仓计划 | 入池/出池可审计；失败不清空有效池 |
| G03 | DONE | G02 | research budget、batch、retry | 控制研究容量、配额和完成时间 | 不降低质量换数量 | 配额、超时、优先级、断点续跑测试 |
| G04 | DONE | G03 | universe/focus-pool 日报 | 连续 20 日达到覆盖、成本、时效预算 | 不接新订单逻辑 | 报告完整；每日批次在窗口内完成 |

## H. 可观察性与简报

| ID | 状态 | 依赖 | 范围 | 目标 | 明确不做 | 专项验收 |
| --- | --- | --- | --- | --- | --- | --- |
| H01 | DONE | G04 | monitor_data、NiceGUI read models、button action contracts | 展示来源、快照、研究、计划、风险、订单和版本；建立全部按钮动作契约 | UI 不下单、不改 broker 权限 | 任一订单完整追溯；31 个按钮动作覆盖成功/空数据/错误/重复点击；UI 关闭不影响 Runtime |
| H02 | DONE | H01 | Discord builder、send audit、配置 | 每日/异常简报并记录发送 | 未授权不外发、不发送 secrets | dry-run、格式、去敏、失败、去重测试；授权后才实发 |
| H03 | DONE | H02 | 运行手册、故障分级、备份恢复 | 可恢复数据库并定位阻塞 | 不删用户记录 | 临时副本恢复及 API/DB/Runtime 故障演练 |
| H04 | DONE | H03 | 完整 Paper 生产路径、审计与回放 | 连续完成数据→分析→计划→执行→风险→复盘→策略候选闭环 | 不授权 live、不用 UI 逐笔审批、不自动晋升策略 | BUY/SELL/拒绝/部分成交/UNKNOWN/重启及一次完整成交后复盘均有稳定 ID 调用链 |
| H05 | DONE | H04 | NiceGUI 全页面、按钮契约、read models | 所有按钮产生正确完整结果并能解释关联记录 | UI 不成为第二 Runtime | 31 个动作逐项通过成功、空数据、错误、忙碌/幂等测试；浏览器 Paper 演练通过 |
| H06 | DONE | H05 | 闭环量化报告、调用图、限制与恢复证据 | 宣布当前 Paper 闭环小目标完成 | 不承诺盈利、不视为实盘授权 | 全量验证、量化指标、完整回放、按钮报告、恢复演练全部签收 |

## I. 长期 Paper 成熟度验收

| ID | 状态 | 依赖 | 范围 | 目标 | 明确不做 | 专项验收 |
| --- | --- | --- | --- | --- | --- | --- |
| I01 | BLOCKED | H06 | 完整系统 | 连续 60 个预定交易日稳定 | 不授权 live、不跳过失败日 | 每日报告齐全；零未解释重复单、计划改写、状态差异 |
| I01-ACCEL | DONE | H06 | 预定日/观察 store、60 日隔离门禁 | 完成长周期证据架构并解除故障演练代码等待 | 不把 SYNTHETIC 计入 REAL、不伪造时间 | 60 日正门禁；缺失/失败/改写负门禁；REAL/SYNTHETIC 隔离 |
| I02 | DONE | I01-ACCEL | Data Hub、TA、Runtime、broker、Kill Switch | 演练缺数、超时、重启、部分成交、休市、Kill Switch | 不在 live 演练 | 每种故障有预期、审计和恢复；无意外 submit |
| I03-ARCH | DONE | I02 | 全部生产路径、签收 store 与文档 | 完成不可变双层签收架构 | 不用隔离证据宣布最终 REAL 完成 | ARCHITECTURE_READY；FINAL_REAL 对合成/不足60日证据 fail-closed |
| I03 | BLOCKED | I01 + I02 + I03-ARCH | 全部生产路径与文档 | 宣布 Paper 目标迁移完成 | 不视为 live 授权、不承诺盈利 | 全量验证、运行报告、迁移记录、限制、调用图全部签收 |

## 生产激活补强（不替代 REAL 时间门槛）

| ID | 状态 | 生产连接 | 验收 |
| --- | --- | --- | --- |
| P01 | DONE | Runtime 每轮登记预定交易日，并在美东 20:00 后冻结 REAL 成熟度观察 | 缺失日显式失败；REAL/SYNTHETIC 不混用 |
| P02 | DONE | 成交关闭后自动生成冻结 EpisodeReview 与保守 StrategyCandidate | 亏损不自动判为策略失效；不自动改生产参数 |
| P03 | DONE | 31 个 NiceGUI 动作全部进入统一 BUSY/SUCCESS/EMPTY/ERROR 审计；Discord 全部通过授权、去敏、幂等网关 | 未授权不外发；重复动作和重复简报有持久证据 |
| P04 | DONE | Runtime 美东 20:00 后每日一次 CHECKPOINT、哈希和只读校验的 trade/AI DuckDB 备份 | 不覆盖源库；重复 tick 幂等；失败只记录脱敏错误类型 |

## 完成证据（只能追加，不能删除历史）

| ID | 完成日期 | 代码/迁移摘要 | 相关测试 | 全量验证 | 已知限制 |
| --- | --- | --- | --- | --- | --- |
| A01 | 2026-07-26 | ATRPlanner 改为必须接收显式 `Side`；Runtime 将 `StrategyDecision.side` 直接传入计划，缺少决策时 fail-closed | 方向相关 21 passed | 全量 170 passed；Ruff、compileall passed | pytest 配置存在既有 `cache_dir` 未知选项警告，不影响结果 |
| A02 | 2026-07-26 | OrderIntentStore 汇总非终态 BUY 剩余敞口；Allocator 仅分配剩余额度；RiskEngine 与提交前边界累计校验 | 累计仓位相关 16 passed | 全量 176 passed；Ruff、compileall passed | 当前多头持仓按本次计划价格估值；A03 将单独处理止损风险比例 |
| A03 | 2026-07-26 | TradePlan 按 `abs(entry-stop) × qty` 强制单笔止损风险；非有限/缺失数值 fail-closed；Runtime 提交前重检新增敞口 | A03 回归 9 passed；A02/A03 联合 24 passed | 全量 185 passed；Ruff、compileall passed | 减仓和平仓不受新增敞口风险门槛阻断；未修改配置默认值 |
| A04 | 2026-07-26 | UNKNOWN/SENDING/OPEN/部分成交跨重启防重提；Portfolio 保留 broker 累计 Fill 并只入账增量；订单库正确保存累计/剩余数量 | A02–A04 联合 26 passed | 全量 190 passed；Ruff、compileall passed | 查询失败保持非终态；不自动猜测 broker 结果 |
| A05 | 2026-07-26 | Portfolio 从持久成交增量恢复；启动时先吸收可解释成交，再按 broker/client order id 恢复开放单并核对仓位数量 | A02–A05 联合 33 passed | 全量 197 passed；Ruff、compileall passed | 未知订单、成交、仓位/数量差异及 broker API 失败均审计并阻塞新单 |
| A06 | 2026-07-26 | Runtime 提交前重检 Alpaca Paper 与 Kill Switch；Alpaca adapter 独立拒绝 live/MKT；未知 broker fail-closed | A06 安全边界 7 passed；相关回归 23 passed | 全量 204 passed；Ruff、compileall passed | 静态 AST 检查确认 Runtime 是唯一生产 `place_order` 调用者 |
| A07 | 2026-07-26 | 新增无网络 Paper smoke、只读 plan/risk/idempotency/order/fill 查询及运行手册；风险结论按 plan 持久化 | A07 专项及相关回归 22 passed；实际 CLI 演练通过 | 全量 207 passed；Ruff、compileall passed | smoke 数据库名必须含 `smoke`；UNKNOWN 仅由模拟 broker 明确取消，不猜测结果 |
| B01 | 2026-07-26 | 定义冻结 `ResearchSnapshot`/source manifest/质量与版本契约；新增兼容旧表的 DuckDB store 与序列化 | B01 专项 10 passed | 全量 217 passed；Ruff、compileall passed | 仅建立 schema，未改变 Runtime 或每日研究读取/结论 |
| B02 | 2026-07-26 | 每日筛选旁路捕获实际 bars/策略统计/候选字段并写 ResearchSnapshot；run-symbol link 显式记录 WRITTEN/FAILED | B02 专项 4 passed；每日研究相关 17 passed | 全量 221 passed；Ruff、compileall passed | 读取路径和研究结论未切换；缺 bars/统计以 MISSING/PARTIAL 记录 |
| B03 | 2026-07-26 | 快照内容哈希冻结、同内容去重、run-symbol 不可变绑定、跨日重放及 KEEP_ALL 保留策略 | B01–B03/每日研究相关 25 passed | 全量 225 passed；Ruff、compileall passed | 不提供自动删除；持久内容被篡改时重放 fail-closed |
| B04 | 2026-07-26 | 冻结后逐字段比较 candidate；新增最近 N 交易日覆盖/质量/比较/哈希报告与可选持久审计 | B04/影子/每日研究相关 14 passed；10 日与负向窗口通过 | 全量 228 passed；Ruff、compileall passed | 生产库仍需自然积累 10 个交易日；报告不切读取、不影响下单 |
| C01 | 2026-07-26 | 新增生产中立 Data Hub、source registry、适配器结果/质量契约、线程超时、TTL/stale cache 与显式 fallback 链 | C01 专项 5 passed | 全量 233 passed；Ruff、compileall passed | 尚未注册或切换真实数据源；超时线程只能取消未开始任务，适配器仍须自身网络超时 |
| C02 | 2026-07-26 | 统一 Alpaca 行情与 broker facts envelope；本地 cache/Yahoo 仅显式不可执行降级；新增市场双读差异分类 | C01/C02 专项 9 passed | 全量 237 passed；Ruff、compileall passed | 尚未切 Runtime 生产读取；Yahoo/local `execution_eligible=false` |
| C03 | 2026-07-26 | 新增 SEC EDGAR CompanyFacts/submissions 客户端与公司事实 adapter；统一财务版本链、公司披露及 Form 3/4/5 内幕申报，缺失与速率限制显式化 | C01–C03 专项 14 passed | 全量 242 passed；Ruff、compileall passed | 尚未切生产读取；内幕数据当前记录 SEC 申报元数据，不解析 Form 4 XML 交易明细 |
| C04 | 2026-07-26 | 新增 Finnhub/Nasdaq/WallStreetCN/Yahoo/RSS 新闻事件聚合 adapter；统一时间窗、跨源去重、来源优先级、冲突留痕与研究专用边界 | C01–C04 专项 20 passed | 全量 248 passed；Ruff、compileall passed | 尚未切 Runtime/晨报读取；跨源语义去重当前基于规范化标题、标的、日期和日历事件身份 |
| C05 | 2026-07-26 | 新增 FRED 宏观事实与 StockTwits/Reddit/Polymarket 研究信号 adapter；统一新鲜度、必需序列覆盖、样本覆盖、低质量和失败状态 | C01–C05 专项 25 passed | 全量 253 passed；Ruff、compileall passed | 尚未切 MacroAgent/晨报读取；Reddit 无结构化标签时仅保留低质量参与度信号，不推断情绪 |
| C06 | 2026-07-26 | 新增跨领域双读比较、关键/研究差异分类、限值与到期批准规则、DuckDB 幂等观察存储及 20 日质量门禁日报 | C01–C06 专项 30 passed | 全量 258 passed；Ruff、compileall passed | 未切执行输入；合成 20 日验收已通过，真实来源仍需自然积累 20 个观察交易日 |
| C06-ACCEL | 2026-07-26 | 新增独立历史正确性回放；AAPL/MSFT 最近 20 个交易日共 40 个本地/Alpaca OHLCV 比较在 1 bps 下零差异 | 加速/影子/质量专项 11 passed | 全量 264 passed；Ruff、compileall passed | `accelerated_shadow_delivery_ready=true` 仅解除后续代码迁移等待；不替代实时窗口，`execution_input_switched=false` |
| D01 | 2026-07-26 | TA subprocess 新增不可变 invocation contract；研究项持久化 run/snapshot/内容哈希/data/model/config/invocation 关联；worker 回传错配、陈旧、非法、超时和崩溃均 fail-closed | D01 与相关研究专项 24 passed | 全量 273 passed；Ruff、compileall passed | 旧 20 条研究 item 向后兼容保留，关联字段为空；只有 D01 后的新研究 run 具备完整 invocation 关联 |
| D02 | 2026-07-26 | worker 登记 market/fundamentals/news/sentiment 配置来源、vendor chain、as-of、抓取时间、质量与失败；持久化独立不可变 TA 输出快照并关联 research item | D02 contract 专项 8 passed；相关研究 23 passed | 全量 274 passed；Ruff、compileall passed | 旧研究 item 无 TA 输出快照；新调用缺 manifest、字段非法或输出不可重放会 fail-closed |
| D03 | 2026-07-26 | AI trust gate 只接受唯一成功 run、当天未过期、输入/输出双快照可重放、data/model/invocation 一致且全部 TA 来源为 OK 的证据 | D03 与 Runtime/决策相关 22 passed | 全量 276 passed；Ruff、compileall passed | 旧研究记录和重复成功 run 会返回空 AI 证据；需运行一次 D01–D02 新合同研究批次 |
| D04 | 2026-07-26 | 新增批次质量报告与持久化；修复筛选/快照批次时钟、超时中断恢复、非空错误码与 worker 脱敏诊断；真实库遗留 RUNNING/PENDING 已恢复 | D04 专项、研究/TA 合同相关 33 passed；无模型筛选→快照集成通过 | 全量 284 passed；Ruff、compileall passed | Ollama 与 TA 客户端已真实连通，但按用户要求未完成资源密集型完整图；真实库仍无 D01–D03 成功批次且仅 1/20 日，未伪造通过 |
| D04-ACCEL | 2026-07-27 | 按闭环验收“自然观察不阻塞代码迁移”规则，使用分离的20日合成质量门禁、真实失败库报告和真实 Ollama/TA 客户端验证解除 E01 代码等待 | 20 日合成正/负门禁、研究/TA 合同与恢复相关 36 passed | 全量 284 passed；Ruff、compileall passed | D04 仍为 BLOCKED；加速证据不计入真实观察日、不代表完整 TA 图成功、不授权执行输入切换 |
| E01 | 2026-07-27 | 新增冻结 PositionPlan 成交基线、head/current pointer 与 append-only 版本表；乐观 expected-version 冲突、基线不可变、状态转换和重启恢复均 fail-closed；真实 trade.duckdb 已兼容增表 | E01 专项 6 passed | 全量 290 passed；Ruff、compileall passed | 尚未连接 fill/Runtime 或改变下单；E02 才维护首次、部分、减仓与清仓成交 |
| E02 | 2026-07-27 | Runtime 已将确认的累计成交投影为同一 PositionPlan 版本链；首次、分批、重复、减仓和清仓均幂等处理，计划版本与 fill 游标同一事务提交；Paper smoke 同步覆盖该生产路径，真实 trade.duckdb 已兼容新增 fill 事件表 | E02/订单恢复/Paper smoke 专项 17 passed | 全量 293 passed；Ruff、compileall passed | 启动 reconciliation 尚不补投影历史成交，留给 E05；E03 才引入失效事件 |
| E03 | 2026-07-27 | 新增确定性 InvalidationEvent 合同、类型专属权威来源、版本/时序/新鲜度/事实校验、稳定 ID、同源去重与冲突拒绝；价格、broker、公司行动、交易限制和策略失效均需结构化证据，纯 LLM 文本 fail-closed；真实 trade.duckdb 已兼容增表 | E03/PositionPlan 专项 17 passed | 全量 301 passed；Ruff、compileall passed | 事件尚不改变计划或创建订单；该执行连接由 E04 完成 |
| E04 | 2026-07-27 | 新增确定性 PositionAdjustment evaluator/store；验证事件与计划版本在同一事务生成 EXIT_PENDING/REDUCING/ACTIVE 新版本，自动退出/减仓经 Runtime 既有安全门生成幂等 Paper LMT OrderIntent，止损收紧不生成订单且多头/空头均禁止放宽；重复事件不重复版本或提交；真实 trade.duckdb 已兼容增表 | E04/E03/PositionPlan/订单恢复专项 26 passed | 全量 305 passed；Ruff、compileall passed | 重启时补齐 PLANNED 调整、开放调整单和成交游标由 E05 完成 |
| E05 | 2026-07-27 | Runtime 启动对账先恢复未连接订单的 PLANNED 调整，再按 broker 累计成交补 PositionPlan 独立游标；开放调整单、部分成交、重复启动和最终清仓均恢复同一链，成交后 adjustment 完成；首次计划漏写可从审计 TradePlan 重建；UNKNOWN 不猜测、不重提且缺 broker 事实继续阻塞 | E05/启动对账专项 15 passed；E02–E05/Paper smoke/订单恢复组合 23 passed | 全量 309 passed；Ruff、compileall passed | 兼容旧成交：没有审计 TradePlan 的迁移前 BUY 不伪造 PositionPlan；E06 才累计 Paper 仓位日报 |
| E06 | 2026-07-27 | 新增 REAL/SYNTHETIC 强隔离的持仓质量观察与 30 日报告，检查 broker/local/plan 数量、未关联版本改写和重复调整；Runtime 每次成功获取 broker 事实即留痕。新增显式 broker 权威组合基线，保留全部旧 fills 并只重放基线后成交；Alpaca 持仓/现金 API 不再把失败伪装为空。真实库已备份并迁移，7 笔旧 `PAPER-*` QQQ fills 保留，权威基线后真实对账通过 | E06 专项/对账/执行边界 20 passed | 全量 315 passed；Ruff、compileall passed | REAL 仅 1/30 日，当前日零差异、零静默改写、零重复调整；自然窗口未完成 |
| E06-ACCEL | 2026-07-27 | 用独立 SYNTHETIC 观察完成 30 日正门禁，并验证错仓、静默改写、重复调整负门禁；REAL/SYNTHETIC 查询严格分区。真实 Alpaca Paper API 成功形成首日 REAL 证据，`1/30` 且当日通过 | E06 合成正/负与隔离专项 4 passed | 全量 315 passed；Ruff、compileall passed | 只解除 F01 代码等待；不计入 E06 REAL 天数，不代表 30 日生产稳定性 |
| F01 | 2026-07-27 | 新增 CandidatePlan→FinalTradePlan→OrderIntent 持久状态机；稳定 ID 绑定 decision/strategy/data/evidence/risk config 与版本，有效期、方向、证据、风险拒绝和非法转换均 fail-closed；OrderIntent/持久表兼容增加候选、最终计划、风险和证据引用；真实库已备份并迁移 | F01/订单幂等/执行边界专项 17 passed | 全量 319 passed；Ruff、compileall passed | Runtime 生产调用仍走 TradePlan 兼容入口；F02 才切为唯一管线 |
| F02 | 2026-07-27 | Runtime 新仓、持仓监控退出、Invalidation 调整和重启补单全部先进入 CandidatePlan→FinalTradePlan→OrderIntent；prepared intent 保留 final/risk/evidence 引用后由唯一提交函数持久化和触达 broker；崩溃后可从管线 intent 引用续跑 | F02 调用图/状态机/调整专项 21 passed | 全量 320 passed；Ruff、compileall passed | `_execute_plan` 名称仍作为内部提交实现保留，F03 将移除该旧适配命名和无调用残留 |
| F03 | 2026-07-27 | 删除 `_execute_plan` 旧适配入口，内部提交改为必须接收状态机产生的 OrderIntent；Paper smoke 和全部提交边界测试改走 `_execute_via_pipeline`；移除 TradePlan 的 SHADOW/LIVE/CLOSED 遗留状态说明。稳定 fallback、旧 SENDING/UNKNOWN 幂等兼容和风险拒绝终态/审计同步收口 | F03/订单幂等/执行边界/Paper smoke 专项 32 passed；`rg _execute_plan` 为零 | 全量 320 passed；Ruff、compileall passed | READ_ONLY_SHADOW 仅保留在 Data Hub 观察工具，不是执行状态或第二生产路径 |
| F04 | 2026-07-27 | 新增以 PositionPlan 为 episode ID 的不可变归因快照；Runtime 正常/重启成交投影后同步计划版本、实际累计 fill、部分/减仓/清仓、跨日、已实现收益、不利滑点、失效事件和调整计数；重复同步按内容哈希幂等；真实库已备份并兼容增表 | F04 episode 专项 1 passed；E02–E05/Paper smoke 组合 12 passed | 全量 321 passed；Ruff、compileall passed | 无 OrderIntent 限价的迁移前 fill 将实际价视为计划价，滑点为零但明确保留原成交事实 |
| F05 | 2026-07-27 | 新增冻结 episode review 工件，严格分离 facts/decision/execution/result 四层并使用稳定 outcome/error taxonomy；SUCCESS、RISK_REJECTED、NO_FILL、DATA_FAILURE、BROKER_FAILURE 均内容哈希幂等和可重放；亏损结果不会自动标记策略失效；真实库已备份并兼容增表 | F05 专项 6 passed | 全量 327 passed；Ruff、compileall passed | 尚未连接自动复盘调度；F06 将消费冻结复盘生成策略候选 |
| F06 | 2026-07-27 | 新增 append-only 策略候选版本库和严格实验边界；候选必须引用冻结 EpisodeReview，并记录生产基线、数据、代码、参数内容哈希及非重叠 training/holdout 区间；同内容幂等，参数变化只追加父子版本，重启可恢复且没有生产晋升/配置修改入口 | F06/F05 专项 13 passed | 全量 334 passed；Ruff、compileall passed | 尚未评估或晋升候选；F07 将独立执行 holdout、历史回放和 Paper champion/challenger，并保留拒绝与回滚审计 |
| F07 | 2026-07-27 | 新增冻结 champion/challenger 比较与 append-only 发布事件；同一引擎/行情/成本模型计算 holdout 和非重叠历史回放的净收益、Sharpe、最大回撤、交易数、费用与滑点，并结合最小 Paper 会话/交易样本决定 PROMOTED/REJECTED；单次盈亏不能晋升，过期冠军基线拒绝，晋升幂等且保存可恢复回滚版本；发布库不写 Runtime 配置 | F07/F06/F05/engine 相关 25 passed | 全量 339 passed；Ruff、compileall passed | 当前只形成可审计的策略发布版本，不自动替换 Runtime 参数；G01 开始建立全市场 universe 版本 |
| G01 | 2026-07-27 | 新增 content-addressed、append-only universe registry；统一股票/ETF/基金类型、交易所、ACTIVE/INACTIVE/DELISTED、tradable、来源和 as-of 元数据；大小写重复幂等、冲突重复拒绝，旧版本永久可恢复，退市/不可交易资产保留审计但不进入 eligible 范围 | G01 专项 6 passed | 全量 356 passed；Ruff、compileall passed | 资产来源采集仍由外部适配器提供；registry 不调用 LLM 或 broker |
| G02 | 2026-07-27 | 新增确定性重点池版本；对 universe 每个成员冻结 reliable holdout、流动性、数据质量、综合分数、排名和拒绝原因；只从 active/tradable 资产入池，来源不完整或异常空池时保留上一有效版本并记录失败 | G01–G02 专项 10 passed | 全量 356 passed；Ruff、compileall passed | 不改变 PositionPlan、Runtime 或订单；同分按 symbol 稳定排序 |
| G03 | 2026-07-27 | 新增从冻结重点池规划的 durable research budget queue；强制符号数/估算成本/估算时长、稳定优先级和批大小，失败按上限重试、运行中任务超时恢复，实际成本/运行窗口超限延后剩余任务，重启不重领已完成工作 | G01–G03 专项 14 passed | 全量 356 passed；Ruff、compileall passed | 队列提供研究执行边界，不导入 broker；外部/每日研究执行器必须按 claim/finish 合同消费 |
| G04 | 2026-07-27 | 新增 immutable universe/focus-pool/budget 每日质量观察与 20-session gate；逐日核对全 universe 筛选覆盖、重点池规模、研究完成/失败/延后、实际成本和完成窗口；同证据类型同日不可改写，REAL/SYNTHETIC 严格分区；20 日合成正门禁及失败日负门禁通过 | G 阶段专项 17 passed | 全量 356 passed；Ruff、compileall passed | SYNTHETIC 20 日仅完成代码加速验收，不能计入 REAL 生产观察；真实窗口需自然积累，不接订单逻辑 |
| H01 | 2026-07-27 | 新增 31 个 NiceGUI 动作契约、持久 BUSY/终态审计和订单全链路解释 read model | H01 专项 5 passed | 全量 373 passed；Ruff、compileall passed | UI 仍不提交订单；重要动作具有独立审计引用 |
| H02 | 2026-07-27 | 新增默认关闭的 Discord delivery gate、dry-run/SENT/FAILED/BLOCKED 审计、去敏与幂等键 | H02 相关 37 passed | 全量 373 passed；Ruff、compileall passed | 未显式授权时不触达外部 sender；webhook/token 不进入消息或诊断 |
| H03 | 2026-07-27 | 新增 DuckDB 哈希备份、只读验证、恢复到新目录和 API/DB/Runtime 故障演练；补运行手册 | H03 专项 4 passed | 全量 373 passed；Ruff、compileall passed | 不覆盖原库、不删除用户记录 |
| H04 | 2026-07-27 | 新增冻结闭环交付证据，强制 snapshot→research→plan→risk→intent→fill→position→episode→review→strategy candidate 引用、Paper 场景、指标与恢复证据 | H04 及闭环相关 19 passed | 全量 373 passed；Ruff、compileall passed | ISOLATED_PAPER/REAL_PAPER 明确分型；不自动晋升策略 |
| H05 | 2026-07-27 | NiceGUI 交易记录页展示最新订单来源→研究→计划→风险→intent→fill 解释；31 动作四状态合同、只读 API、仅本机服务和隔离 Paper 演练完成；用户浏览器截图确认完整解释 EMPTY 状态正确渲染 | H 阶段专项 25 passed；浏览器页面证据通过 | 全量 374 passed；Ruff、compileall passed | 截图时真实审计库没有订单，明确显示 `缺少 order`；成交调用链由隔离 Paper smoke 正/负场景验证 |
| H06 | 2026-07-27 | 冻结闭环量化证据、调用图、限制、恢复演练、按钮报告和浏览器页面证据完成签收 | H 阶段关键闭环回归 13 passed | 全量 374 passed；Ruff、compileall passed | 完成的是当前 Alpaca Paper 闭环小目标；D04/E06 自然观察继续，不授权 live、不承诺盈利 |
| I01-ACCEL | 2026-07-27 | 新增不可变预定交易日、每日成熟度观察和60日门禁；失败日不可跳过，同日冲突拒绝，REAL/SYNTHETIC 强隔离 | I01 架构专项 3 passed | 全量 384 passed；Ruff、compileall passed | 60日 SYNTHETIC 只验证架构，不计入 REAL |
| I02 | 2026-07-27 | 新增六类固定故障矩阵；每项绑定预期结果、submit 数、审计和恢复引用，任一意外 submit 使报告失败 | I02 专项 3 passed | 全量 384 passed；Ruff、compileall passed | 当前是隔离故障证据；真实运行故障继续按日留痕 |
| I03-ARCH | 2026-07-27 | 新增不可变 ARCHITECTURE/FINAL_REAL 双层签收；完整连接成熟度、故障、闭环、验证、文档、调用图和限制 | I03 专项 4 passed；I阶段组合 14 passed | 全量 384 passed；Ruff、compileall passed | FINAL_REAL 强制60日 REAL 成熟度与 REAL 故障证据；live 始终未授权 |
| P01–P04 | 2026-07-27 | 补齐 REAL 成熟度、复盘候选、Discord、31动作审计和每日验证备份的生产调用方；所有入口保持 Paper-only、fail-closed | 新增生产连接专项 23 passed | 全量 392 passed；Ruff、compileall passed | D04/E06/I01/I03 仍须自然 REAL 证据；真实故障不自动注入 |

## 阻塞记录

| ID | 日期 | 阻塞原因 | 已完成的安全检查 | 解除条件 |
| --- | --- | --- | --- | --- |
| D04 | 2026-07-26 | API/客户端已恢复，但尚无完整 TradingAgents 图成功结果，且自然 20 日窗口尚未形成 | 两个配置模型通过真实 Ollama 与 TA 客户端；修复快照时钟、超时恢复、错误留痕；真实库不再有 RUNNING/PENDING 或空失败码 | 在资源允许时生成至少一个 D01–D03 新合同成功批次；自然窗口继续积累至 20 日 |
| E06 | 2026-07-27 | 30 个真实 Paper 交易日无法在单次迁移会话中产生；当前只有 1 个通过日 | 报告实现、30 日合成正/负门禁、REAL/SYNTHETIC 隔离、真实 Alpaca Paper 首日三方一致均已验证 | Runtime 持续积累至 30 个 REAL 交易日，且 mismatch/silent rewrite/duplicate adjustment 始终为零 |
| I01 | 2026-07-27 | 60 个真实预定交易日无法在单次架构迁移会话中产生；隔离60日门禁不能替代自然时间 | 预定日/观察不可变合同、60日合成正负门禁和 REAL/SYNTHETIC 隔离均通过 | Runtime 按交易所日历登记并累计60个 REAL 日，所有日报完整且四类异常计数为零 |
| I03 | 2026-07-27 | 最终 Paper 迁移签收依赖 I01 的60日 REAL 报告 | ARCHITECTURE_READY 及 FINAL_REAL 负向门禁已通过；完整文档和调用图已生成 | I01 READY 且六类 REAL 故障报告通过后生成 FINAL_REAL_READY |
