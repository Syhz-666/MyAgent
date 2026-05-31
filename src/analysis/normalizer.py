"""结构化分析结果归一化。"""

from __future__ import annotations

from typing import Any


def normalize_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    """将分析结果归一化为通用 Schema。"""
    if not isinstance(analysis, dict):
        return {
            "summary": "待确认",
            "sections": [{"title": "分析结果", "items": [str(analysis)]}],
            "follow_up_questions": [],
            "action_items": [],
        }

    if "summary" in analysis and "sections" in analysis:
        return {
            "summary": str(analysis.get("summary", "待确认")),
            "sections": normalize_sections(analysis.get("sections", [])),
            "follow_up_questions": _normalize_string_list(analysis.get("follow_up_questions", []) or []),
            "action_items": normalize_action_items(analysis.get("action_items", []) or []),
        }

    sections: list[dict[str, Any]] = []

    key_conclusions = analysis.get("key_conclusions", []) or []
    if key_conclusions:
        sections.append({"title": "关键结论", "items": [str(item) for item in key_conclusions]})

    risks = analysis.get("risks", []) or []
    if risks:
        sections.append({"title": "风险与问题", "items": [str(item) for item in risks]})

    action_items = normalize_action_items(analysis.get("action_items", []) or [])
    if action_items and not sections:
        sections.append({"title": "行动项概览", "items": [str(item.get("task", "待确认")) for item in action_items]})

    return {
        "summary": str(analysis.get("meeting_summary", analysis.get("summary", "待确认"))),
        "sections": sections or [{"title": "分析结果", "items": ["待确认"]}],
        "follow_up_questions": _normalize_string_list(analysis.get("follow_up_questions", []) or []),
        "action_items": action_items,
    }


def normalize_sections(sections: Any) -> list[dict[str, Any]]:
    """归一化动态章节。"""
    if not isinstance(sections, list):
        return [{"title": "分析结果", "items": [str(sections)]}]

    normalized_sections: list[dict[str, Any]] = []
    for section in sections:
        if isinstance(section, dict):
            items = section.get("items", [])
            if isinstance(items, str):
                items = [items]
            normalized_sections.append(
                {
                    "title": str(section.get("title", "未命名章节")),
                    "items": items if isinstance(items, list) else [str(items)],
                }
            )
        else:
            normalized_sections.append({"title": "分析结果", "items": [str(section)]})

    return normalized_sections or [{"title": "分析结果", "items": ["待确认"]}]


def normalize_action_items(action_items: Any) -> list[dict[str, str]]:
    """归一化可追溯行动项。"""
    if not isinstance(action_items, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in action_items:
        if isinstance(item, dict):
            task = str(item.get("task", "")).strip()
            if not task:
                continue
            normalized.append(
                {
                    "task": task,
                    "owner": str(item.get("owner", "待确认") or "待确认"),
                    "deadline": str(item.get("deadline", "待确认") or "待确认"),
                    "priority": str(item.get("priority", "待确认") or "待确认"),
                    "evidence": str(item.get("evidence", "待确认") or "待确认")[:80],
                    "confidence": str(item.get("confidence", "待确认") or "待确认"),
                }
            )
    return normalized


def _normalize_string_list(values: Any) -> list[str]:
    """归一化字符串列表。"""
    if isinstance(values, list):
        return [str(value) for value in values]
    if values:
        return [str(values)]
    return []
