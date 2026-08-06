"""覆盖三处执行/风控升级：

1. marketable-limit 缓冲（execution_pipeline.marketable_limit_price）——入场/
   出场分别用不同缓冲幅度，把静态限价单换成有成交概率保障的限价单。
2. 追踪止损（position_monitor.TrailingStopEvaluator + runtime._apply_trailing_stops）
   ——复用 M0 就建好但从未被调用过的 TIGHTEN_STOP 机制。
3. AI 信念驱动的止盈风格（runtime._select_stop_style / _far_take_profit）——
   高信念仓位一次性锁定"不设固定止盈上限"，不在 tick 级别临场问 AI。
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from trader.audit import AuditLog
from trader.config import RiskConfig, TradingConfig
from trader.execution_pipeline import marketable_limit_price
from trader.invalidation_events import InvalidationEventStore
from trader.models import Fill, Position, Side, TradePlan
from trader.order_lifecycle import OrderIntentStore
from trader.position_adjustments import PositionAdjustmentStore
from trader.position_monitor import TrailingStopEvaluator
from trader.position_plans import PositionPlanFillProjector, PositionPlanStore
from trader.portfolio import Portfolio
from trader.risk_engine import RiskEngine
from trader.runtime import Runtime, _far_take_profit, _select_stop_style

NOW = datetime(2026, 7, 27, 16, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# 1. marketable-limit 缓冲
# ---------------------------------------------------------------------------

def test_entry_buy_gets_small_upward_buffer():
    price = marketable_limit_price(100.0, Side.BUY, action="OPEN")
    assert price == 100.15  # +0.15%


def test_entry_sell_short_gets_small_downward_buffer():
    price = marketable_limit_price(100.0, Side.SELL, action="OPEN")
    assert price == 99.85  # -0.15%


def test_exit_close_gets_larger_buffer_than_entry():
    # 平多（SELL）比入场（BUY）方向相反，且缓冲幅度更大——出场没成交比
    # 入场没成交代价更高（风控失效 vs 错过机会），所以缓冲不对称。
    exit_price = marketable_limit_price(100.0, Side.SELL, action="CLOSE")
    entry_price = marketable_limit_price(100.0, Side.BUY, action="OPEN")
    assert exit_price == 99.5  # -0.5%
    assert abs(100.0 - exit_price) > abs(entry_price - 100.0)


def test_reduce_action_treated_as_urgent_like_close():
    assert marketable_limit_price(100.0, Side.SELL, action="REDUCE") == 99.5


# ---------------------------------------------------------------------------
# 2. TrailingStopEvaluator（纯函数，不接触任何 store）
# ---------------------------------------------------------------------------

def _rising_bars(n=20, start=100.0, step=1.0, half_range=0.5):
    bars = []
    close = start
    for _ in range(n):
        bars.append(
            SimpleNamespace(
                close=close, high=close + half_range, low=close - half_range,
            )
        )
        close += step
    return bars


def test_trailing_stop_tightens_when_price_has_run_up():
    plan = TradePlan(
        plan_id="p1", symbol="AAPL", side=Side.BUY, action="OPEN",
        entry_price=100.0, stop_loss=110.0, take_profit=200.0, qty=10,
    )
    bars = _rising_bars()
    evaluator = TrailingStopEvaluator()
    candidates = evaluator.evaluate(
        {"AAPL": Position(symbol="AAPL", qty=10, avg_entry_px=100.0)},
        {"AAPL": plan},
        {"AAPL": bars},
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.symbol == "AAPL"
    assert candidate.new_stop_loss > plan.stop_loss  # 只收紧
    assert candidate.new_stop_loss < plan.take_profit


def test_trailing_stop_never_loosens():
    # 止损已经比"追踪算出来的候选值"更紧——不应该返回任何候选（防止放松）。
    plan = TradePlan(
        plan_id="p2", symbol="AAPL", side=Side.BUY, action="OPEN",
        entry_price=100.0, stop_loss=118.0, take_profit=200.0, qty=10,
    )
    bars = _rising_bars()
    evaluator = TrailingStopEvaluator()
    candidates = evaluator.evaluate(
        {"AAPL": Position(symbol="AAPL", qty=10, avg_entry_px=100.0)},
        {"AAPL": plan},
        {"AAPL": bars},
    )
    assert candidates == []


def test_trailing_stop_skips_flat_or_missing_positions():
    plan = TradePlan(
        plan_id="p3", symbol="AAPL", side=Side.BUY, action="OPEN",
        entry_price=100.0, stop_loss=90.0, take_profit=200.0, qty=10,
    )
    evaluator = TrailingStopEvaluator()
    assert evaluator.evaluate(
        {"AAPL": Position(symbol="AAPL", qty=0.0, avg_entry_px=100.0)},
        {"AAPL": plan},
        {"AAPL": _rising_bars()},
    ) == []
    assert evaluator.evaluate(
        {"AAPL": Position(symbol="AAPL", qty=10.0, avg_entry_px=100.0)},
        {},  # no live plan
        {"AAPL": _rising_bars()},
    ) == []


# ---------------------------------------------------------------------------
# 3. AI 信念驱动的止盈风格
# ---------------------------------------------------------------------------

def test_high_conviction_selects_trailing_only():
    # 参数是方向无关的 conviction（0-1，AIScoreValidationResult.confidence），
    # 不是双极 ai_score（0-100）——这样 BUY/SELL 用同一把尺子，不会出现"SELL
    # 永远够不到高信念门槛"的问题。
    assert _select_stop_style(0.90, "HIGH") == "TRAILING_ONLY"


def test_high_conviction_but_not_high_complexity_stays_capped():
    assert _select_stop_style(0.95, "MEDIUM") == "CAPPED"


def test_high_complexity_but_low_conviction_stays_capped():
    assert _select_stop_style(0.70, "HIGH") == "CAPPED"


def test_missing_conviction_stays_capped():
    assert _select_stop_style(None, "HIGH") == "CAPPED"


def test_far_take_profit_scales_with_risk_distance_buy():
    plan = TradePlan(
        plan_id="p4", symbol="AAPL", side=Side.BUY, action="OPEN",
        entry_price=100.0, stop_loss=95.0, take_profit=110.0, qty=10,
    )
    assert _far_take_profit(plan) == 140.0  # 100 + 8*(100-95)


def test_far_take_profit_scales_with_risk_distance_sell():
    plan = TradePlan(
        plan_id="p5", symbol="AAPL", side=Side.SELL, action="OPEN",
        entry_price=100.0, stop_loss=105.0, take_profit=90.0, qty=10,
    )
    assert _far_take_profit(plan) == 60.0  # 100 - 8*(105-100)


# ---------------------------------------------------------------------------
# 4. Runtime._apply_trailing_stops 端到端——两处写入（PositionPlan 新版本 +
#    内存 _live_plans）都要生效，否则追踪止损只是一条审计记录。
# ---------------------------------------------------------------------------

class _AuditCapture:
    def log_reconciliation(self, *_a, **_k):
        return None

    def log_plan_risk_event(self, *_a, **_k):
        return None


def _config(path) -> TradingConfig:
    return TradingConfig(
        db_path=str(path),
        broker_type="alpaca_paper",
        auto_trade_paper=True,
        risk=RiskConfig(max_position_pct=0.20, max_trade_risk_pct=0.05),
    )


def _runtime(config):
    runtime = Runtime.__new__(Runtime)
    runtime._cfg = config
    runtime._risk = RiskEngine(config)
    runtime._order_store = OrderIntentStore(config.db_path)
    runtime._position_plan_store = PositionPlanStore(config.db_path)
    runtime._position_plan_projector = PositionPlanFillProjector(
        runtime._position_plan_store
    )
    runtime._invalidation_event_store = InvalidationEventStore(config.db_path)
    runtime._position_adjustment_store = PositionAdjustmentStore(config.db_path)
    runtime._portfolio = Portfolio(config)
    runtime._trailing_stop = TrailingStopEvaluator()
    runtime._kill = SimpleNamespace(engaged=lambda: False)
    runtime._reconciliation_blocked = False
    runtime._open_orders = {}
    runtime._live_plans = {}
    runtime._monitor_plans = {}
    runtime._signal_store = SimpleNamespace(
        mark_exit=lambda *_, **__: None, apply_fill=lambda *_, **__: None,
    )
    runtime._bug_reporter = SimpleNamespace(
        capture_exception=lambda *_, **__: None
    )
    runtime._audit = _AuditCapture()
    return runtime


def test_apply_trailing_stops_updates_both_position_plan_and_live_plan(tmp_path):
    config = _config(tmp_path / "trade.duckdb")
    runtime = _runtime(config)

    # 开仓：先建一张 PositionPlan（走真实的 PositionPlanFillProjector，用的是
    # position_plans.py 里那个刚打开的默认 invalidation_rules——顺带验证
    # STRATEGY_INVALIDATED 确实已经加进默认允许列表，否则这一步会在
    # process_invalidation_event 内部报 INVALIDATION_RULE_NOT_CONFIGURED。
    # PositionPlan 首个版本要求 stop_loss < entry < take_profit（合法建仓）；
    # 追踪止损把它往上收紧、越过原始 entry 价，靠的是第 2 版起这条约束放宽成
    # 只要求 stop_loss < take_profit（models.py 里已经这么设计），不是本次
    # 改动新引入的行为。
    entry_plan = TradePlan(
        plan_id="entry-plan-aapl", symbol="AAPL", side=Side.BUY, action="OPEN",
        entry_price=100.0, stop_loss=95.0, take_profit=200.0, qty=10,
        created_at=NOW - timedelta(minutes=5),
    )
    fill = Fill(
        order_id="entry-order", intent_id="entry-intent", symbol="AAPL",
        side=Side.BUY, filled_qty=10, avg_price=100.0,
        fill_time=NOW - timedelta(minutes=5),
    )
    runtime._portfolio.apply_fill(fill)
    runtime._position_plan_projector.apply(
        fill=fill, applied_delta=None, trade_plan=entry_plan,
    )
    runtime._live_plans["AAPL"] = entry_plan

    positions = {"AAPL": Position(symbol="AAPL", qty=10, avg_entry_px=100.0)}
    raw_bars_map = {"AAPL": _rising_bars()}  # 价格从 100 一路涨到 119

    runtime._apply_trailing_stops(positions, raw_bars_map, NOW)

    # 内存里的止损被收紧了……
    assert runtime._live_plans["AAPL"].stop_loss > 95.0
    # ……而且落地到了持久化的 PositionPlan 新版本，不只是内存状态。
    current = runtime._position_plan_store.current_for_symbol("AAPL")
    assert current is not None
    assert current.version == 2
    assert current.stop_loss == runtime._live_plans["AAPL"].stop_loss


def test_apply_trailing_stops_is_noop_without_favorable_move(tmp_path):
    config = _config(tmp_path / "trade.duckdb")
    runtime = _runtime(config)
    entry_plan = TradePlan(
        plan_id="entry-plan-msft", symbol="MSFT", side=Side.BUY, action="OPEN",
        entry_price=100.0, stop_loss=95.0, take_profit=200.0, qty=10,
        created_at=NOW - timedelta(minutes=5),
    )
    fill = Fill(
        order_id="entry-order-2", intent_id="entry-intent-2", symbol="MSFT",
        side=Side.BUY, filled_qty=10, avg_price=100.0,
        fill_time=NOW - timedelta(minutes=5),
    )
    runtime._portfolio.apply_fill(fill)
    runtime._position_plan_projector.apply(
        fill=fill, applied_delta=None, trade_plan=entry_plan,
    )
    runtime._live_plans["MSFT"] = entry_plan

    positions = {"MSFT": Position(symbol="MSFT", qty=10, avg_entry_px=100.0)}
    # 价格从 100 一路跌到 81——追踪止损候选值会比现有止损更松，一次都不该收紧。
    runtime._apply_trailing_stops(
        positions, {"MSFT": _rising_bars(start=100.0, step=-1.0)}, NOW
    )

    assert runtime._live_plans["MSFT"].stop_loss == 95.0
    current = runtime._position_plan_store.current_for_symbol("MSFT")
    assert current.version == 1
