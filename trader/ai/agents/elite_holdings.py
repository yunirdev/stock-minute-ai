"""
trader/ai/agents/elite_holdings.py
大咖持仓 Agent — 追踪顶级机构与国会议员的持仓动向。

数据来源（均免费，无需 API Key）：
  ① ARK Invest（Cathie Wood）          — arkfunds.io，每日更新持仓 + 交易明细
     · ARKK / ARKQ / ARKF / ARKG / ARKW 五支 ETF
  ② SEC EDGAR 13F — Berkshire (Buffett) CIK 1067983，季度，45天滞后
  ③ SEC EDGAR 13F — Scion (Burry)      CIK 1649339，季度，45天滞后（仅 5-15 仓位）
  ④ yfinance institutional_holders      — 前五大机构持仓比例
  ⑤ yfinance insider_transactions       — 公司内部人员近期买卖（CFO/CEO/董事）
  ⑥ Twitter 搜索                        — Pelosi / 国会议员交易新闻

⚠ 局限（请知悉）：
  - Berkshire / Scion 13F 季度数据滞后 45 天（法律要求），非实时
  - ARK 持仓为昨日数据（最新），交易数据 1 个月内
  - 国会议员 STOCK Act 同样最多 45 天滞后
  - Trump 非国会议员，不受 STOCK Act 约束，无个股交易实时披露

评分权重：
  ARK 持仓 + 近期交易  35%
  Berkshire 持仓变化   25%
  Burry 持仓           20%
  内部人净买卖         10%
  国会 Twitter 信号    10%
"""

from __future__ import annotations

import logging
import re
import urllib.request
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from trader.contracts import AgentContext
from trader.models import Advisory
from .base import AgentBase

logger = logging.getLogger(__name__)

# ── 大咖 CIK & 配置 ────────────────────────────────────────────────────────────
_BERKSHIRE_CIK = "0001067983"  # Berkshire Hathaway (Buffett)
_SCION_CIK = "0001649339"  # Scion Asset Management (Burry)
_SCION_DATA_ID = "1649339"  # EDGAR data path
_BERKSHIRE_DATA_ID = "1067983"

_ARK_FUNDS = ["ARKK", "ARKQ", "ARKF", "ARKG", "ARKW"]
_ARK_FUND_WEIGHT = {
    "ARKK": 1.5,
    "ARKQ": 1.0,
    "ARKF": 1.0,
    "ARKG": 1.0,
    "ARKW": 1.0,
}  # ARKK 旗舰权重更高

_SEC_HEADERS = {
    "User-Agent": "stock-minute-ai/1.0 contact@example.com",
    "Accept-Encoding": "identity",
}
_ARK_HEADERS = {"User-Agent": "Mozilla/5.0"}

_INSIDER_WINDOW_DAYS = 90
_CONGRESS_QUERIES = [
    "{symbol} congress stock trade STOCK Act",
    "{symbol} Pelosi bought",
]


class EliteHoldingsAgent(AgentBase):
    """
    大咖持仓 Agent。
    ARK（每日）+ Berkshire 13F + Scion 13F + 内部人交易 + 国会 Twitter。
    """

    role = "elite_holdings"

    def __init__(self, client=None) -> None:
        self._twitter = None
        try:
            from trader.ai.web_research import get_web_research_client

            self._twitter = get_web_research_client()
        except Exception:
            pass

    def run(self, ctx: AgentContext) -> List[Advisory]:
        # 一次性批量拉取（避免每个 symbol 重复请求）
        ark_data = _fetch_all_ark()  # {ticker: {funds, recent_buy, recent_sell}}
        berk_cur, berk_prev = _fetch_two_quarters(_BERKSHIRE_CIK, _BERKSHIRE_DATA_ID)
        scion_cur, scion_prev = _fetch_two_quarters(_SCION_CIK, _SCION_DATA_ID)

        advisories: List[Advisory] = []
        for cand in ctx.candidates:
            try:
                adv = self._analyze(
                    cand.symbol, ark_data, berk_cur, berk_prev, scion_cur, scion_prev
                )
                if adv:
                    advisories.append(adv)
            except Exception as exc:
                logger.warning("EliteHoldingsAgent 跳过 %s: %s", cand.symbol, exc)
        return advisories

    # ── 核心分析 ─────────────────────────────────────────────────────────────

    def _analyze(
        self,
        symbol: str,
        ark_data: Dict,
        berk_cur: Dict,
        berk_prev: Dict,
        scion_cur: Dict,
        scion_prev: Dict,
    ) -> Optional[Advisory]:

        score = 50.0
        signals: List[str] = []
        details: Dict[str, Any] = {}

        # ① ARK Invest（35%）
        ark_sig, ark_detail = _ark_signal(symbol, ark_data)
        score += ark_sig * 0.35
        details["ark"] = ark_detail
        if ark_detail.get("held_by"):
            signals.append(f"ARK持有({','.join(ark_detail['held_by'])})")
        if ark_detail.get("recent_buy"):
            signals.append("ARK近期买入")
        if ark_detail.get("recent_sell"):
            signals.append("ARK近期卖出")

        # ② Berkshire / Buffett（25%）
        berk_sig, berk_detail = _holder_signal(
            symbol, berk_cur, berk_prev, label="Berkshire"
        )
        score += berk_sig * 0.25
        details["berkshire"] = berk_detail
        if berk_detail.get("held"):
            action = berk_detail.get("action", "")
            signals.append(f"Berkshire {action or '持仓'}")

        # ③ Burry / Scion（20%）— 集中仓位，信号强度高
        scion_sig, scion_detail = _holder_signal(
            symbol, scion_cur, scion_prev, label="Scion", concentrated=True
        )
        score += scion_sig * 0.20
        details["scion"] = scion_detail
        if scion_detail.get("held"):
            action = scion_detail.get("action", "")
            signals.append(f"Burry(Scion) {action or '持仓'}")

        # ④ 内部人交易（10%）
        insider_sig, insider_detail = _yf_insider(symbol)
        score += insider_sig * 0.10
        details["insider"] = insider_detail
        if insider_sig > 3:
            signals.append("内部人净买入")
        elif insider_sig < -3:
            signals.append("内部人净卖出")

        # ⑤ 机构持仓比例（背景参考）
        inst_sig, inst_detail = _yf_institutional(symbol)
        score += inst_sig * 0.05
        details["institutional"] = inst_detail

        # ⑥ 国会 Twitter（10%）
        congress_sig, congress_news = self._congress_twitter(symbol)
        score += congress_sig * 0.10
        details["congress_news"] = congress_news[:3]
        if congress_news:
            signals.append(
                f"国会交易新闻(+{congress_sig:.0f}pt)"
                if congress_sig >= 0
                else f"国会交易新闻({congress_sig:.0f}pt)"
            )

        elite_score = self._clamp_score(int(score))
        confidence = 0.3 + min(0.55, len(signals) * 0.08)

        if elite_score >= 65:
            stance = "accumulating"
        elif elite_score <= 40:
            stance = "distributing"
        else:
            stance = "neutral"

        logger.info(
            "EliteHoldingsAgent %s: score=%d stance=%s signals=%s",
            symbol,
            elite_score,
            stance,
            signals,
        )

        return self._advisory(
            kind="elite_holdings",
            payload={
                "symbol": symbol,
                "elite_score": elite_score,
                "stance": stance,
                "signals": signals,
                "ark": ark_detail,
                "berkshire": berk_detail,
                "scion": scion_detail,
                "institutional": inst_detail,
                "insider": insider_detail,
                "congress_news": congress_news[:5],
                "data_note": "ARK每日更新；13F季报滞后45天；国会数据来自Twitter",
            },
            confidence=confidence,
            model="algorithmic",
        )

    def _congress_twitter(self, symbol: str) -> Tuple[float, List[str]]:
        if not self._twitter or not self._twitter.has_twitter():
            return 0.0, []
        news: List[str] = []
        for tpl in _CONGRESS_QUERIES:
            q = tpl.format(symbol=symbol)
            try:
                news.extend(self._twitter.search_twitter(q, n=3))
            except Exception:
                pass
        if not news:
            return 0.0, []
        buy_cnt = sum(
            1 for t in news if re.search(r"\b(buy|bought|purchase|acquired)\b", t, re.I)
        )
        sell_cnt = sum(
            1 for t in news if re.search(r"\b(sell|sold|disposed)\b", t, re.I)
        )
        return (buy_cnt - sell_cnt) * 4.0, news


# ── ARK Invest ─────────────────────────────────────────────────────────────────


def _fetch_all_ark() -> Dict[str, Any]:
    """
    从 arkfunds.io 批量获取所有 ARK ETF 的持仓和近期交易。
    返回 {ticker: {"held_by": [...], "weight": float, "recent_buy": bool, "recent_sell": bool}}
    """
    result: Dict[str, Any] = {}

    for fund in _ARK_FUNDS:
        # 持仓
        try:
            url = f"https://arkfunds.io/api/v2/etf/holdings?symbol={fund}"
            req = urllib.request.Request(url, headers=_ARK_HEADERS)
            with urllib.request.urlopen(req, timeout=12) as r:
                d = json.loads(r.read())
            for h in d.get("holdings", []):
                ticker = h.get("ticker", "").upper()
                if not ticker:
                    continue
                if ticker not in result:
                    result[ticker] = {
                        "held_by": [],
                        "weight": 0.0,
                        "recent_buy": False,
                        "recent_sell": False,
                    }
                result[ticker]["held_by"].append(fund)
                result[ticker]["weight"] = max(
                    result[ticker]["weight"],
                    float(h.get("weight_pct") or h.get("market_value_weight") or 0),
                )
        except Exception as exc:
            logger.debug("ARK %s holdings 失败: %s", fund, exc)

        # 近期交易
        try:
            url2 = f"https://arkfunds.io/api/v2/etf/trades?symbol={fund}&period=1m"
            req2 = urllib.request.Request(url2, headers=_ARK_HEADERS)
            with urllib.request.urlopen(req2, timeout=12) as r:
                d2 = json.loads(r.read())
            for t in d2.get("trades", []):
                ticker = t.get("ticker", "").upper()
                if not ticker:
                    continue
                result.setdefault(
                    ticker,
                    {
                        "held_by": [],
                        "weight": 0.0,
                        "recent_buy": False,
                        "recent_sell": False,
                    },
                )
                direction = t.get("direction", "").lower()
                if direction == "buy":
                    result[ticker]["recent_buy"] = True
                elif direction == "sell":
                    result[ticker]["recent_sell"] = True
        except Exception as exc:
            logger.debug("ARK %s trades 失败: %s", fund, exc)

    logger.info("ARK data: %d tickers tracked", len(result))
    return result


def _ark_signal(symbol: str, ark_data: Dict) -> Tuple[float, Dict]:
    """计算 ARK 信号分 [-50, 50]。"""
    detail: Dict[str, Any] = {
        "held_by": [],
        "weight": 0.0,
        "recent_buy": False,
        "recent_sell": False,
    }
    entry = ark_data.get(symbol.upper())
    if not entry:
        return 0.0, detail

    detail.update(
        entry
    )  # 含 weight，原样透传给上层展示，但下面打分目前不按 weight 区分
    held_by = entry.get("held_by", [])
    bought = entry.get("recent_buy", False)
    sold = entry.get("recent_sell", False)

    # 持仓基础分：持有越多 ARK 基金 = 越高
    hold_score = sum(_ARK_FUND_WEIGHT.get(f, 1.0) for f in held_by) * 6
    hold_score = min(30, hold_score)  # 上限 30

    # 交易信号
    trade_score = 0.0
    if bought and not sold:
        trade_score = 15.0  # 纯买入
    elif bought and sold:
        trade_score = 3.0  # 同月有买有卖，温和
    elif sold and not bought:
        trade_score = -12.0  # 纯卖出

    # ARKK 旗舰持有额外加权
    flagship_bonus = 8.0 if "ARKK" in held_by else 0.0

    total = hold_score + trade_score + flagship_bonus
    return total, detail


# ── SEC EDGAR 13F 通用解析 ────────────────────────────────────────────────────


def _fetch_two_quarters(cik: str, data_id: str) -> Tuple[Dict, Dict]:
    """返回最近两季 13F 持仓 {规范化公司名: 持股数}，失败返回 ({}, {})。"""
    try:
        accs = _get_13f_accessions(cik, n=2)
        cur = _parse_13f(accs[0], data_id) if len(accs) >= 1 else {}
        prev = _parse_13f(accs[1], data_id) if len(accs) >= 2 else {}
        return cur, prev
    except Exception as exc:
        logger.warning("13F 获取失败 CIK=%s: %s", cik, exc)
        return {}, {}


def _get_13f_accessions(cik: str, n: int = 2) -> List[str]:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    req = urllib.request.Request(url, headers=_SEC_HEADERS)
    with urllib.request.urlopen(req, timeout=12) as r:
        d = json.loads(r.read())
    filings = d["filings"]["recent"]
    return [
        filings["accessionNumber"][i]
        for i, f in enumerate(filings["form"])
        if f == "13F-HR"
    ][:n]


def _parse_13f(accession: str, data_id: str) -> Dict[str, float]:
    """解析一份 13F infotable XML → {规范化公司名: 持股数}。"""
    acc_nodash = accession.replace("-", "")
    dir_url = f"https://www.sec.gov/Archives/edgar/data/{data_id}/{acc_nodash}/"
    req = urllib.request.Request(dir_url, headers=_SEC_HEADERS)
    with urllib.request.urlopen(req, timeout=12) as r:
        html = r.read().decode("utf-8", errors="replace")

    xml_files = re.findall(r'href="(/Archives/[^"]+\.xml)"', html, re.IGNORECASE)
    info_path = next(
        (x for x in xml_files if "primary" not in x.lower()),
        xml_files[-1] if xml_files else None,
    )
    if not info_path:
        return {}

    req2 = urllib.request.Request(
        f"https://www.sec.gov{info_path}", headers=_SEC_HEADERS
    )
    with urllib.request.urlopen(req2, timeout=15) as r:
        xml = r.read().decode("utf-8", errors="replace")

    names = re.findall(r"<nameOfIssuer>([^<]+)</nameOfIssuer>", xml)
    shares = re.findall(r"<sshPrnamt>([^<]+)</sshPrnamt>", xml)

    if len(names) != len(shares):
        # SEC XML 偶尔有缺失/多余字段（修订版、脚注行等），按较短的对齐，
        # 不直接抛异常中断整个 13F 解析 —— 但要留痕，否则数据缺口无声无息。
        logger.warning(
            "_parse_13f(%s): nameOfIssuer=%d 与 sshPrnamt=%d 数量不一致，按较短对齐",
            accession,
            len(names),
            len(shares),
        )

    holdings: Dict[str, float] = {}
    for n, s in zip(names, shares):  # noqa: B905 — 故意宽松对齐+上面记警告，不用 strict 中断整个解析
        key = _norm(n)
        holdings[key] = holdings.get(key, 0) + int(s.replace(",", ""))
    return holdings


def _holder_signal(
    symbol: str,
    current: Dict,
    prev: Dict,
    label: str = "",
    concentrated: bool = False,
) -> Tuple[float, Dict]:
    """在持仓字典中查找 symbol，计算信号分。"""
    detail: Dict[str, Any] = {"held": False, "change_pct": None, "label": label}
    if not current:
        return 0.0, detail

    company_name = _get_company_name(symbol)
    match_key = _find_key(company_name or symbol, current)
    if not match_key:
        return 0.0, detail

    cur_shares = current[match_key]
    pre_shares = prev.get(match_key, 0)
    detail.update(
        {"held": True, "company_matched": match_key, "current_shares": cur_shares}
    )

    # 集中型投资人（Burry）：持有本身信号更强
    base_hold = 15.0 if concentrated else 5.0

    if pre_shares == 0:
        detail["action"] = "new_position"
        return base_hold + 10, detail

    change_pct = (cur_shares / pre_shares - 1) * 100
    detail["change_pct"] = round(change_pct, 1)

    if change_pct >= 10:
        detail["action"] = "increased"
        return base_hold + 8, detail
    elif change_pct <= -10:
        detail["action"] = "decreased"
        return -8.0, detail
    else:
        detail["action"] = "unchanged"
        return base_hold, detail


def _find_key(company_name: str, holdings: Dict) -> Optional[str]:
    if not company_name:
        return None
    words = _norm(company_name).split()
    key_words = [
        w
        for w in words[:3]
        if len(w) > 2 and w not in {"INC", "CORP", "LTD", "CO", "THE", "GROUP"}
    ]
    if not key_words:
        return None
    for key in holdings:
        if all(w in key for w in key_words[:2]):
            return key
    return None


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"&amp;", "&", name).strip().upper())


def _get_company_name(symbol: str) -> Optional[str]:
    try:
        import yfinance as yf

        info = yf.Ticker(symbol).info
        return info.get("longName") or info.get("shortName")
    except Exception:
        return None


# ── yfinance 机构 & 内部人 ─────────────────────────────────────────────────────


def _yf_institutional(symbol: str) -> Tuple[float, Dict]:
    try:
        import yfinance as yf

        df = yf.Ticker(symbol).institutional_holders
        if df is None or df.empty:
            return 0.0, {}
        top = df.head(5)
        holders = [
            {
                "holder": row.get("Holder", ""),
                "pct_out": round(float(row.get("% Out", 0)) * 100, 2),
            }
            for _, row in top.iterrows()
        ]
        total_pct = sum(h["pct_out"] for h in holders)
        return min(8.0, total_pct * 0.15), {
            "top_holders": holders,
            "total_top5_pct": round(total_pct, 1),
        }
    except Exception:
        return 0.0, {}


def _yf_insider(symbol: str) -> Tuple[float, Dict]:
    try:
        import yfinance as yf

        df = yf.Ticker(symbol).insider_transactions
        if df is None or df.empty:
            return 0.0, {}
        cutoff = datetime.now() - timedelta(days=_INSIDER_WINDOW_DAYS)
        for col in ("Start Date", "Date"):
            if col in df.columns:
                df = df[df[col] >= cutoff]
                break
        if df.empty:
            return 0.0, {"recent_transactions": 0}

        txn_col = "Transaction" if "Transaction" in df.columns else "Typ"
        buy_mask = df.get(txn_col, "").str.contains(
            "Buy|Purchase", na=False, case=False
        )
        sell_mask = df.get(txn_col, "").str.contains("Sell|Sale", na=False, case=False)
        share_col = "Shares" if "Shares" in df.columns else "Value"
        buy_v = int(df[buy_mask][share_col].fillna(0).sum()) if buy_mask.any() else 0
        sell_v = int(df[sell_mask][share_col].fillna(0).sum()) if sell_mask.any() else 0
        net = buy_v - sell_v
        denom = max(buy_v + sell_v, 1)
        return min(10.0, max(-10.0, net / denom * 15)), {
            "recent_transactions": len(df),
            "buy_val": buy_v,
            "sell_val": sell_v,
            "net_direction": "buy" if net > 0 else "sell" if net < 0 else "neutral",
        }
    except Exception:
        return 0.0, {}
