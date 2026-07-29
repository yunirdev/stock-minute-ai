"""Source-level guard: `pending_buy_notional` must be initialized before the
try/except that calls pending_buy_notional_by_symbol() inside _tick(). That
call is the only statement in the block that can raise before the name is
otherwise assigned; if it raises, the except sets `plans = []` and the
following `for plan in plans:` never runs the line that reads
`pending_buy_notional` -- but only because plans is empty. Any change to
that control flow (e.g. a non-empty fallback for `plans`) would turn this
into a real UnboundLocalError. A plain default assignment ahead of the try
removes the fragility outright; this test pins it down so it can't
regress silently.
"""
import ast
from pathlib import Path

RUNTIME_SRC = Path(__file__).resolve().parents[1] / "trader" / "runtime.py"


def test_pending_buy_notional_is_bound_before_the_risky_try_block():
    source = RUNTIME_SRC.read_text(encoding="utf-8")
    tree = ast.parse(source)
    tick_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_tick"
    )

    try_block = next(
        node
        for node in ast.walk(tick_fn)
        if isinstance(node, ast.Try)
        and any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "pending_buy_notional_by_symbol"
            for n in ast.walk(node)
        )
    )

    assignments_before_try = [
        target.id
        for node in tick_fn.body
        if getattr(node, "lineno", 0) < try_block.lineno
        for stmt in ast.walk(node)
        if isinstance(stmt, (ast.Assign, ast.AnnAssign))
        for target in ([stmt.target] if isinstance(stmt, ast.AnnAssign) else stmt.targets)
        if isinstance(target, ast.Name)
    ]

    assert "pending_buy_notional" in assignments_before_try, (
        "pending_buy_notional must be assigned a default before the try block "
        "that calls pending_buy_notional_by_symbol(), otherwise a future change "
        "to the except branch's `plans` fallback could make it referenced "
        "while unbound."
    )
