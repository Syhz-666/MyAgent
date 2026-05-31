"""文本分析 Prompt 构造。"""

from __future__ import annotations

from typing import Any


def build_text_analysis_messages(
    task: str,
    text: str,
    search_results: dict[str, Any] | None = None,
    extract_action_items: bool = False,
) -> list[dict[str, str]]:
    """构造通用文本分析 messages。"""
    system_prompt = (
        "你是一个通用文本分析 Agent.用户会指定分析目标, 请根据目标从文本中提取信息."
        "下面“文本”字段一定是用户提供的原始文本, 不得声称文本缺失或未提供."
        "如果用户任务与文本内容明显不匹配（如任务询问'国外发展'但文本只涉及中国情况），在 summary 中如实指出差异，sections 仅提取文本中勉强相关的部分并标注'文本未覆盖'，不要强行将无关内容当答案."
        "summary、sections、follow_up_questions 是通用输出骨架."
        "action_items 是可选业务增强字段, 只有在明确要求提取行动项、负责人、截止时间、跟进事项、任务分工或待办事项时才输出."
        "如果原文没有明确某类信息, 对应内容必须标记为“待确认”或“未在文本中找到对应信息”, 不能编造."
        "只输出合法 JSON, 不要输出 Markdown, 不要添加额外解释."
    )
    action_item_instruction = _build_action_item_instruction(extract_action_items)
    search_context = _format_search_results(search_results)
    user_prompt = f"""
任务:{task}

请根据任务分析下面的文本, 并输出 JSON.

基础 JSON 格式必须包含:
{{
  "summary": "文本概要, 2-4 句话",
  "sections": [
    {{
      "title": "章节标题, 由任务和文本内容决定",
      "items": ["要点1", "要点2"]
    }}
  ],
  "follow_up_questions": ["待确认问题1"]
}}

{action_item_instruction}

要求:
1. summary 必须存在;
2. sections 只包含与用户任务直接相关的章节, sections 的划分服务于用户任务, 而不是还原原文的段落结构;
3. 如果任务是具体问题（如”xxx是什么””xxx的原因是什么”）, 用 1-2 个 sections 聚焦回答即可;
4. 如果任务是全面总结（如”按主题归纳””提取所有要点”）, 可以展开更多章节;
5. sections[].title 应该贴合用户任务;
6. sections[].items 应该是字符串列表;
7. 信息缺失时写”待确认”或”未在文本中找到对应信息”;
8. 不要编造负责人、截止时间、优先级、结论或风险;
9. 搜索结果只是证据候选, 不代表最终结论, 必须结合全文判断.

搜索结果候选:
{search_context}

文本:
{text}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_action_item_instruction(extract_action_items: bool) -> str:
    """构建是否输出 action_items 的 prompt 片段。"""
    if not extract_action_items:
        return (
            "action_items 规则:\n"
            "- 当前任务不是行动项类任务, 不要输出 action_items 字段;\n"
            "- 不要为了填充字段而编造任务、负责人或截止时间."
        )

    return """
action_items 规则:
- 当前任务涉及行动项、负责人、截止时间、跟进事项、任务分工或待办事项, 请额外输出 action_items 字段;
- action_items 必须是数组, 每项格式为:
  {
    "task": "行动项描述",
    "owner": "负责人或待确认",
    "deadline": "截止时间或待确认",
    "priority": "high / medium / low / 待确认",
    "evidence": "原文依据片段, 最长 80 字",
    "confidence": "high / medium / low"
  }
- evidence 必须是原文中能支撑该行动项的连续片段, 不能写“根据全文总结”;
- 原文没有明确负责人或截止时间时写“待确认”;
- 找不到明确依据时不要强行生成行动项.
""".strip()


def _format_search_results(search_results: dict[str, Any] | None) -> str:
    """将 keyword_search 结果压缩为 prompt 可读文本。"""
    if not search_results:
        return "无"

    matches = search_results.get("matches", []) if isinstance(search_results, dict) else []
    if not matches:
        return "无命中"

    lines: list[str] = []
    for match in matches[:10]:
        if not isinstance(match, dict):
            continue
        keyword = match.get("keyword", "")
        line_no = match.get("line_no", "")
        line = match.get("line", "")
        before = " / ".join(str(item) for item in match.get("context_before", []) or [])
        after = " / ".join(str(item) for item in match.get("context_after", []) or [])
        context = ""
        if before:
            context += f" 上文:{before}"
        if after:
            context += f" 下文:{after}"
        lines.append(f"- 关键词:{keyword}; 行号:{line_no}; 命中:{line};{context}")

    return "\n".join(lines) or "无命中"
