"""Discord 尺寸约束层：裁剪沿语义边界、超长分页不丢内容、字段合法。"""
from __future__ import annotations

from trader.discord_limits import (
    DESCRIPTION_MAX,
    EMBED_TOTAL_MAX,
    FIELD_NAME_MAX,
    FIELD_VALUE_MAX,
    FIELDS_MAX,
    TITLE_MAX,
    allocate_budget,
    embed_text_length,
    fit_fields,
    fit_notification,
    split_text,
    truncate_text,
)
from trader.models import Notification


# ── truncate_text ────────────────────────────────────────────────────────────


def test_short_text_is_untouched():
    assert truncate_text("短文本", 100) == "短文本"


def test_truncate_prefers_paragraph_boundary():
    """段落边界落在预算的合理位置时，宁可少显示一点也要保住完整的一节。"""
    first = "第一段" + "内容" * 30          # 约 63 字，占 100 预算的大半
    text = first + "\n\n第二段应当整段留到下一页" + "x" * 200
    out = truncate_text(text, 100)
    assert out.startswith("第一段")
    assert "第二段应当整段留到下一页" not in out


def test_truncate_ignores_a_boundary_that_wastes_most_of_the_budget():
    """但边界太靠前就不用它——为保一个 5 字的段落而浪费 90% 预算不划算。"""
    text = "很短\n\n" + "后面是一大段连续内容" * 30
    out = truncate_text(text, 200)
    # 应该继续往后取，而不是只输出"很短"
    assert len(out) > 100


def test_truncate_falls_back_to_line_boundary():
    text = "\n".join(f"第 {i} 行的内容占位" for i in range(40))
    out = truncate_text(text, 120)
    body = out.split("\n…")[0]
    # 每一行都应该是完整的，不能出现半行
    for line in body.split("\n"):
        assert line == "" or line.endswith("行的内容占位")


def test_truncate_states_how_much_was_dropped():
    """省略必须说清楚少了多少——只给一个 '…' 读者无法判断要不要看全文。"""
    text = "开头\n" + "\n".join("填充行" for _ in range(50))
    out = truncate_text(text, 60)
    assert "未显示" in out
    assert "行" in out


def test_truncate_never_exceeds_limit():
    for limit in (20, 50, 100, 500):
        out = truncate_text("中文内容" * 500, limit)
        assert len(out) <= limit, f"limit={limit} 实际={len(out)}"


# ── split_text ───────────────────────────────────────────────────────────────


def test_split_keeps_all_content_within_page_budget():
    text = "\n".join(f"第 {i} 行" for i in range(60))
    pages = split_text(text, 100, max_pages=5)
    assert len(pages) > 1
    for page in pages:
        assert len(page) <= 100


def test_split_marks_overflow_when_pages_run_out():
    text = "\n".join(f"第 {i} 行内容" for i in range(500))
    pages = split_text(text, 100, max_pages=2)
    assert len(pages) == 2
    # 页数触顶时必须如实标注，而不是静默丢弃
    assert "未显示" in pages[-1]


def test_split_single_page_when_it_fits():
    assert split_text("正好放得下", 100) == ["正好放得下"]


# ── fit_fields ───────────────────────────────────────────────────────────────


def test_fields_respect_name_and_value_limits():
    fields = {"名" * 400: "值" * 3000}
    out = fit_fields(fields)
    for name, value in out.items():
        assert len(name) <= FIELD_NAME_MAX
        assert len(value) <= FIELD_VALUE_MAX


def test_too_many_fields_collapse_into_a_counted_note():
    fields = {f"字段{i}": f"值{i}" for i in range(40)}
    out = fit_fields(fields)
    assert len(out) <= FIELDS_MAX
    assert any("未显示" in v for v in out.values())


# ── fit_notification ─────────────────────────────────────────────────────────


def test_normal_notification_passes_through_unchanged():
    note = Notification(title="标题", body="正文", kind="plan", fields={"a": "1"})
    out = fit_notification(note)
    assert len(out) == 1
    assert out[0].body == "正文"
    assert out[0].title == "标题"


def test_oversized_body_is_paginated_not_dropped():
    note = Notification(title="复盘", body="\n".join(f"第 {i} 条持仓" for i in range(2000)))
    out = fit_notification(note)
    assert len(out) > 1
    for page in out:
        assert len(page.body) <= DESCRIPTION_MAX
        assert embed_text_length(page) <= EMBED_TOTAL_MAX
    # 分页标题要能看出总页数
    assert "1/" in out[0].title


def test_fields_only_ride_on_the_first_page():
    """字段是整条消息的摘要，每页重复既浪费额度又让人以为是不同数据。"""
    note = Notification(
        title="研究",
        body="\n".join(f"第 {i} 行" for i in range(3000)),
        fields={"状态": "COMPLETED"},
    )
    out = fit_notification(note)
    assert len(out) > 1
    assert out[0].fields
    assert all(not page.fields for page in out[1:])


def test_total_budget_respected_when_fields_are_huge():
    """fields 占掉大半额度时，正文预算要相应收缩，不能让加总破 6000。"""
    note = Notification(
        title="标题",
        body="正文" * 3000,
        fields={f"字段{i}": "值" * 900 for i in range(6)},
    )
    out = fit_notification(note)
    assert embed_text_length(out[0]) <= EMBED_TOTAL_MAX


def test_title_is_clamped():
    note = Notification(title="标" * 500, body="正文")
    out = fit_notification(note)
    assert len(out[0].title) <= TITLE_MAX


# ── allocate_budget ──────────────────────────────────────────────────────────


def test_short_items_are_not_truncated_at_all():
    """内容少的时候一个字都不该裁——这正是固定 _trunc(x, 180) 做不到的。"""
    lengths = [50, 60, 40]
    quotas = allocate_budget(lengths, 1000, min_each=80)
    for want, got in zip(lengths, quotas):
        assert got >= want


def test_long_items_share_the_squeeze_evenly():
    """都超长时压缩要均摊，不能第一条写满、最后一条只剩标题。"""
    quotas = allocate_budget([2000, 2000, 2000], 600, min_each=80)
    assert sum(quotas) <= 600
    assert max(quotas) - min(quotas) <= 2


def test_short_items_never_hoard_more_than_they_need():
    """只要 20 字的条目不该占住 80 的下限额度——那 60 得回流给写得长的。"""
    quotas = allocate_budget([1000, 1000, 20], 420, min_each=80)
    assert quotas[2] <= 20
    # 短条目让出来的额度全部归两个长条目，一点不浪费
    assert quotas[0] + quotas[1] >= 400 - 20


def test_allocate_handles_empty_and_zero():
    assert allocate_budget([], 100) == []
    assert allocate_budget([10, 20], 0) == [0, 0]
