from __future__ import annotations

from typing import List


def build_account_risk_lines(
    positions: List[dict],
    equity: float,
    cash_pct: float,
    movers: List[dict],
) -> List[str]:
    if not positions or equity <= 0:
        return []

    lines = ["\n**今日账户风险优先级**"]
    sorted_pos = sorted(
        positions,
        key=lambda p: abs(float(p.get("market_value", 0))),
        reverse=True,
    )
    top = sorted_pos[0]
    top_sym = str(top.get("symbol", "")).upper()
    top_weight = float(top.get("market_value", 0)) / equity * 100
    top_line = f"最大暴露：{top_sym} {top_weight:.1f}%"
    if top_weight >= 35:
        top_line += "（集中度高，优先管理）"
    elif top_weight >= 20:
        top_line += "（中等集中，留意单股波动）"
    lines.append("• " + top_line)
    if top_weight >= 35:
        lines.append(f"• 账户动作：先评估 {top_sym} 风险，不新增同方向高 beta 仓位。")

    mover_map = {str(m.get("symbol", "")).upper(): m for m in movers}
    impacted = []
    for pos in sorted_pos:
        sym = str(pos.get("symbol", "")).upper()
        mover = mover_map.get(sym)
        if mover and abs(float(mover.get("pct", 0))) >= 0.5:
            impacted.append(f"{sym} {float(mover.get('pct', 0)):+.2f}%")
    if impacted:
        lines.append("• 盘前异动持仓：" + "、".join(impacted[:4]) + "，开盘先处理风险。")
    else:
        lines.append("• 暂无盘前显著异动持仓，按指数方向管理。")

    if cash_pct < 20:
        lines.append(f"• 现金缓冲 {cash_pct:.1f}% 偏低，避免继续加速满仓。")
        lines.append("• 仓位约束：新计划优先减半或等待确认后再提交。")
    elif cash_pct > 50:
        lines.append(f"• 现金缓冲 {cash_pct:.1f}% 充足，可等待确认后再动。")
    else:
        lines.append(f"• 现金缓冲 {cash_pct:.1f}% 中性，适合分批而非一次性加仓。")

    return lines
