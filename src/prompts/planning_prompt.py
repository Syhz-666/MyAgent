"""计划生成 Prompt 构造。"""

from __future__ import annotations

import json
from typing import Any


def build_planning_messages(task: str, tool_descriptions: list[dict[str, Any]]) -> list[dict[str, str]]:
    """构造文本处理计划生成 messages。"""
    system_prompt = (
        "你是一个文本处理 Agent 的 Planner."
        "你只能使用用户提供的工具名称, 不能编造工具."
        "请只输出合法 JSON, 不要输出 Markdown, 不要添加额外解释."
    )
    user_prompt = f"""
用户任务:{task}

可用工具:
{json.dumps(tool_descriptions, ensure_ascii=False, indent=2)}

请生成执行计划, JSON 格式必须为:
{{
  "steps": [
    {{
      "step_id": 1,
      "goal": "步骤目标",
      "tool_name": "工具名",
      "tool_input": {{}},
      "expected_output": "预期输出"
    }}
  ]
}}

约束:
1. 只能使用可用工具列表中的工具;
2. 必须先读取文件, 再分析文本, 再生成报告, 最后写入文件;
3. keyword_search 是可选工具, 如使用, 必须位于 file_reader 之后、text_extractor 之前;
4. file_reader 的 tool_input 可以留空或包含 path;
5. text_extractor 的 tool_input 必须包含 task;
6. build_report 的 tool_input 必须包含 task;
7. file_writer 必须最后执行.
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
