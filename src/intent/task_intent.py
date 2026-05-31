"""任务意图判断与搜索关键词构造。"""

from __future__ import annotations

ACTION_ITEM_KEYWORDS = ["行动项", "负责人", "截止", "跟进", "任务", "待办", "分工"]
RISK_KEYWORDS = ["风险", "问题", "阻塞", "延期", "依赖", "不稳定", "待确认"]
ACTION_SEARCH_KEYWORDS = ["负责", "完成", "截止", "跟进", "同步", "推进", "排期", "待办"]
RISK_SEARCH_KEYWORDS = ["风险", "问题", "阻塞", "延期", "依赖", "不稳定", "待确认"]


def should_extract_action_items(task: str) -> bool:
    """判断当前任务是否需要输出可追溯行动项。"""
    return any(keyword in task for keyword in ACTION_ITEM_KEYWORDS)


def should_use_keyword_search(task: str) -> bool:
    """判断当前任务是否需要先进行关键词证据检索。"""
    return any(keyword in task for keyword in [*ACTION_ITEM_KEYWORDS, *RISK_KEYWORDS])


def build_search_keywords(task: str) -> list[str]:
    """根据任务意图生成关键词搜索列表。"""
    keywords: list[str] = []
    if any(keyword in task for keyword in ACTION_ITEM_KEYWORDS):
        keywords.extend(ACTION_SEARCH_KEYWORDS)
    if any(keyword in task for keyword in RISK_KEYWORDS):
        keywords.extend(RISK_SEARCH_KEYWORDS)
    return _dedupe(keywords)


def _dedupe(values: list[str]) -> list[str]:
    """按原顺序去重。"""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
