"""Discord 消息尺寸约束 —— 报告长度的单一事实来源。

在这个模块出现之前，长度控制散落在各个 builder 里，而且全是硬编码的魔数：
morning_brief 对四条正文各写一次 ``body[:4000]``、discord_delivery 脱敏时写
``text[:4000]``、daily_discord 写 ``_trunc(thesis, 180)``。这带来三个问题：

1. **数字是错的。** Discord 的 description 上限是 4096 而不是 4000，真正会
   把消息打回来的是另外两条没人管的规则——单个 field.value 上限 1024，以及
   一条消息里所有 embed 文本加总不得超过 6000。daily_research 的消息往
   fields 里塞五个标的的 thesis，超限后 Discord 直接返回 400，而调用链上没
   有重试，这条消息就凭空消失了。
2. **切在哪里全靠运气。** ``[:4000]`` 会从半个句子、半个 Markdown 粗体标记
   中间砍断，读者看到的是一段烂尾文字，而且完全不知道后面还有没有内容。
3. **超长就等于丢失。** 截断是静默的，没人知道今天的复盘被砍掉了三条持仓。

这里的做法是：把官方限制集中定义，裁剪时优先沿语义边界（段落 > 行 > 句子）
下刀，并且**永远如实标注省略了多少**；正文超出单条预算时分页成多条消息，而
不是把尾巴扔掉。builder 只管把内容写完整，不需要知道任何数字。

限制取自官方文档 https://docs.discord.com/developers/resources/message
（2026-08 核对）。计数单位是 Unicode code point，中文字符记 1，因此 Python
的 len() 与 Discord 的口径一致，不需要额外换算。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List

from .models import Notification

# ── 官方硬限制 ───────────────────────────────────────────────────────────────
TITLE_MAX = 256
DESCRIPTION_MAX = 4096
FIELD_NAME_MAX = 256
FIELD_VALUE_MAX = 1024
FIELDS_MAX = 25
FOOTER_MAX = 2048
AUTHOR_NAME_MAX = 256
#: 一条消息里所有 embed 的 title + description + field.name + field.value
#: + footer.text + author.name 加总的上限。这条最容易被忽略，因为单看每个
#: 字段都合法。
EMBED_TOTAL_MAX = 6000
CONTENT_MAX = 2000
EMBEDS_PER_MESSAGE = 10

#: 单条通知最多分几页。分页是为了不丢内容，但无限分页会把频道刷屏，所以留一
#: 个上限；超出的部分在最后一页如实标注。
DEFAULT_MAX_PAGES = 3

#: 给分页页码、省略提示这类附加文字预留的余量。裁剪时先扣掉它，避免加完标注
#: 反而超限。
_MARKUP_RESERVE = 80


def _omission_note(chars: int, lines: int) -> str:
    """省略提示 —— 说清楚少了多少，而不是甩一个孤零零的省略号。

    读者据此判断"要不要去决策台看全文"；一个 "…" 给不了这个判断依据。
    """
    if lines > 1:
        return f"…（另有 {lines} 行 / {chars} 字未显示）"
    return f"…（另有 {chars} 字未显示）"


def truncate_text(text: str, limit: int, *, note: bool = True) -> str:
    """把 text 压到 limit 以内，优先沿语义边界下刀。

    优先级：段落（空行）> 行 > 句末标点 > 硬切。前三种都能让读者看到一个
    完整的意群，硬切是兜底。
    """
    text = str(text or "")
    if len(text) <= limit:
        return text
    if limit <= 0:
        return ""

    tail = _omission_note(0, 0) if note else ""
    budget = max(1, limit - len(tail) - 4)

    head = text[:budget]

    # 段落边界优先：保住完整的一节，比多塞两行半截话有用
    cut = head.rfind("\n\n")
    if cut < budget * 0.5:
        cut = head.rfind("\n")
    if cut < budget * 0.5:
        # 中英文句末标点都找一遍
        cut = max(head.rfind(mark) for mark in ("。", "！", "？", ". ", "; ", "；"))
        if cut > 0:
            cut += 1
    if cut < budget * 0.5:
        cut = budget

    kept = text[:cut].rstrip()
    dropped = text[cut:]
    if not note:
        return kept
    return kept + "\n" + _omission_note(len(dropped), dropped.count("\n") + 1)


def split_text(
    text: str,
    budget: int,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> List[str]:
    """按预算把正文切成若干页，尽量在段落/行边界断开。

    返回至少一页。页数触顶时，最后一页尾部如实标注还剩多少没显示——超长的
    代价是"你知道自己没看全"，而不是"你以为看全了"。
    """
    text = str(text or "")
    if budget <= 0:
        return [""]
    if len(text) <= budget:
        return [text]

    pages: List[str] = []
    remaining = text
    while remaining and len(pages) < max_pages:
        if len(remaining) <= budget:
            pages.append(remaining)
            remaining = ""
            break
        # 本页不加省略标注（后面还有页），所以 note=False
        head = truncate_text(remaining, budget, note=False)
        if not head:
            head = remaining[:budget]
        pages.append(head)
        remaining = remaining[len(head):].lstrip("\n")

    if remaining:
        # 页数已经用完还有剩余：在最后一页标注清楚
        note = _omission_note(len(remaining), remaining.count("\n") + 1)
        last = pages[-1]
        if len(last) + len(note) + 1 > budget:
            last = truncate_text(last, budget - len(note) - 1, note=False)
        pages[-1] = last + "\n" + note

    return pages or [""]


def allocate_budget(
    lengths: List[int],
    total: int,
    *,
    min_each: int = 60,
) -> List[int]:
    """把 total 字符额度分给 N 个条目，短的不动、长的按比例压缩。

    这是"不写死每条摘要多少字"的关键。固定 ``_trunc(thesis, 180)`` 的毛病
    在于它跟实际内容无关：五个标的里有三个只写了 60 字时，剩下两个明明还有
    空间却照样被砍到 180；反过来五个都写了 800 字时，180 又太挤。

    这里的分法是：先满足所有能完整放下的条目（短的一个字不动），把它们没用
    完的额度回收，再分给超长的条目——所以内容少的时候完全不裁，内容多的时候
    才逐步降级，且降级是均摊的，不会出现"第一条写满、最后一条只剩标题"。

    min_each 是每条的下限，保证再挤也能看出个大概；条目太多导致连下限都凑不
    齐时，返回的额度会低于 min_each，由调用方决定是否少显示几条。
    """
    count = len(lengths)
    if count == 0:
        return []
    if total <= 0:
        return [0] * count

    floor = min(min_each, max(1, total // count))
    # 下限也不该超过条目自己的需要：只写了 20 字的 thesis 占住 80 的额度，
    # 那 60 就白白浪费了，本该回流给写得长的那几条。
    quota = [min(floor, length) for length in lengths]
    spare = total - sum(quota)
    if spare <= 0:
        # 连下限都不够分，等分了事
        share = max(1, total // count)
        return [share] * count

    # 多轮回收：每轮把"要得比配额少"的条目多出来的额度退回池子，
    # 再平分给仍然吃紧的条目，直到没人能再多拿。
    while spare > 0:
        hungry = [i for i in range(count) if lengths[i] > quota[i]]
        if not hungry:
            break
        share = spare // len(hungry)
        if share <= 0:
            # 余量不够整除了，按顺序把零头补给最靠前的饥饿条目
            for i in hungry[:spare]:
                quota[i] += 1
            break
        moved = 0
        for i in hungry:
            take = min(share, lengths[i] - quota[i])
            quota[i] += take
            moved += take
        if moved == 0:
            break
        spare -= moved
    return quota


def fit_fields(fields: Dict[str, Any]) -> Dict[str, str]:
    """把 embed fields 压进 Discord 的形状：名 256 / 值 1024 / 最多 25 条。

    超出 25 条时不是直接丢弃，而是把溢出条目的数量并进最后一条，让读者知道
    还有内容。
    """
    if not fields:
        return {}

    out: Dict[str, str] = {}
    items = list(fields.items())
    # 留一格给溢出提示
    keep = items[: FIELDS_MAX - 1] if len(items) > FIELDS_MAX else items

    for key, value in keep:
        name = truncate_text(str(key), FIELD_NAME_MAX, note=False)
        out[name] = truncate_text(str(value), FIELD_VALUE_MAX)

    if len(items) > FIELDS_MAX:
        dropped = len(items) - len(keep)
        out["…"] = f"另有 {dropped} 项未显示（Discord 单条最多 {FIELDS_MAX} 个字段）"
    return out


def embed_text_length(note: Notification) -> int:
    """按 Discord 的口径统计一条通知会占用多少 embed 字符额度。"""
    total = len(note.title) + len(note.body)
    for key, value in (note.fields or {}).items():
        total += len(str(key)) + len(str(value))
    return total


def fit_notification(
    note: Notification,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> List[Notification]:
    """把一条通知整形成若干条合法通知。

    预算分配的顺序反映重要性：title 和 fields 是结构化摘要，先保住；剩下的
    额度全给正文，正文放不下就分页。builder 因此不需要预估自己会写多长——
    内容短的时候一个字都不会被动，内容长的时候按语义降级而不是被腰斩。
    """
    title = truncate_text(str(note.title or ""), TITLE_MAX, note=False)
    fields = fit_fields(note.fields or {})

    fields_len = sum(len(k) + len(v) for k, v in fields.items())
    # 正文预算 = min(单字段上限, 总额度 - 标题 - 字段 - 标注余量)
    budget = min(
        DESCRIPTION_MAX,
        EMBED_TOTAL_MAX - len(title) - fields_len - _MARKUP_RESERVE,
    )
    if budget <= 0:
        # 极端情况：fields 自己就快把 6000 吃满。正文至少留一点，
        # 并把 fields 砍到能容下正文为止。
        fields = dict(list(fields.items())[:5])
        fields_len = sum(len(k) + len(v) for k, v in fields.items())
        budget = max(
            200,
            EMBED_TOTAL_MAX - len(title) - fields_len - _MARKUP_RESERVE,
        )

    pages = split_text(str(note.body or ""), budget, max_pages=max_pages)

    if len(pages) == 1:
        return [replace(note, title=title, body=pages[0], fields=fields)]

    # 分页时 fields 只挂在第一页：它是整条消息的摘要，每页重复既浪费额度
    # 又让读者以为是不同的数据。
    out: List[Notification] = []
    total = len(pages)
    for index, page in enumerate(pages, start=1):
        page_title = truncate_text(
            f"{title}（{index}/{total}）", TITLE_MAX, note=False
        )
        out.append(
            replace(
                note,
                title=page_title,
                body=page,
                fields=fields if index == 1 else {},
            )
        )
    return out
