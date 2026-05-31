"""Markdown 报告生成。"""

from __future__ import annotations

from typing import Any

try:
    from ..analysis.normalizer import normalize_analysis
except ImportError:  # pragma: no cover - 支持直接导入
    from analysis.normalizer import normalize_analysis


def build_markdown_report(analysis: dict[str, Any], steps: list[Any], task: str = "") -> str:
    """根据 task、LLM 分析结果和执行轨迹生成 Markdown 报告。"""
    normalized = normalize_analysis(analysis)
    lines: list[str] = []
    title = task or "通用文本分析"

    lines.append(f"# 分析报告:{title}")
    lines.append("")

    lines.append("## 概要")
    lines.append("")
    lines.append(str(normalized.get("summary", "待确认")))
    lines.append("")

    sections = normalized.get("sections", [])
    if sections:
        for section in sections:
            section_title = _sanitize_heading(str(section.get("title", "未命名章节")))
            items = section.get("items", [])
            lines.append(f"## {section_title}")
            lines.append("")
            if isinstance(items, list) and items:
                for item in items:
                    lines.append(f"- {item}")
            elif items:
                lines.append(str(items))
            else:
                lines.append("- 待确认")
            lines.append("")
    else:
        lines.append("## 分析结果")
        lines.append("")
        lines.append("- 待确认")
        lines.append("")

    action_items = normalized.get("action_items", []) or []
    if action_items:
        lines.append("## 可追溯行动项")
        lines.append("")
        lines.append("| 行动项 | 负责人 | 截止时间 | 优先级 | 可信度 | 依据片段 |")
        lines.append("|--------|--------|----------|--------|--------|----------|")
        for item in action_items:
            lines.append(
                "| {task} | {owner} | {deadline} | {priority} | {confidence} | {evidence} |".format(
                    task=_escape_markdown_table(str(item.get("task", "待确认"))),
                    owner=_escape_markdown_table(str(item.get("owner", "待确认"))),
                    deadline=_escape_markdown_table(str(item.get("deadline", "待确认"))),
                    priority=_escape_markdown_table(str(item.get("priority", "待确认"))),
                    confidence=_escape_markdown_table(str(item.get("confidence", "待确认"))),
                    evidence=_escape_markdown_table(str(item.get("evidence", "待确认"))),
                )
            )
        lines.append("")

    follow_up_questions = normalized.get("follow_up_questions", []) or []
    if follow_up_questions:
        lines.append("## 待确认问题")
        lines.append("")
        for question in follow_up_questions:
            lines.append(f"- {question}")
        lines.append("")

    lines.append("## 执行过程")
    lines.append("")
    lines.append("| 步骤 | 思考 | 动作 | 观察结果 | 状态 |")
    lines.append("|------|------|------|----------|------|")
    for step in steps:
        thought = _escape_markdown_table(str(step.thought))
        action = _escape_markdown_table(str(step.action))
        observation = _escape_markdown_table(str(step.observation))
        status = _escape_markdown_table(str(step.status))
        lines.append(f"| {step.step} | {thought} | {action} | {observation} | {status} |")

    lines.append("")
    return "\n".join(lines)


def _escape_markdown_table(value: str) -> str:
    """转义 Markdown 表格中的特殊字符。"""
    return value.replace("|", "\\|").replace("\n", " ")


def _sanitize_heading(value: str) -> str:
    """清理 Markdown 标题，避免空标题或换行。"""
    cleaned = value.replace("\n", " ").strip().lstrip("#").strip()
    return cleaned or "未命名章节"
