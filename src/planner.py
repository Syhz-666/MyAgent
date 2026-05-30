"""任务规划模块。

第三阶段 3B 引入 LLMPlanner：
- SimplePlanner 继续作为稳定降级方案；
- LLMPlanner 根据用户 task 和可用工具描述生成结构化计划；
- validate_plan 用于拦截非法工具名、异常步骤顺序和缺失关键工具的计划。
"""

from __future__ import annotations

import json
from typing import Any

try:
    from .schemas import PlanStep
except ImportError:  # pragma: no cover - 支持直接导入
    from schemas import PlanStep


class SimplePlanner:
    """规则版任务规划器，作为第三阶段的稳定降级方案。"""

    def create_plan(self, task: str, input_path: str, output_path: str) -> list[PlanStep]:
        """根据任务与输入输出路径生成通用文本处理计划。"""
        return [
            PlanStep(
                step_id=1,
                goal="读取输入文本文件",
                tool_name="file_reader",
                tool_input={"path": input_path},
                expected_output="输入文件的文本内容",
            ),
            PlanStep(
                step_id=2,
                goal="根据用户任务分析文本并提取结构化信息",
                tool_name="text_extractor",
                tool_input={"text": "", "task": task},
                expected_output="包含 summary、sections、follow_up_questions 的结构化分析结果",
            ),
            PlanStep(
                step_id=3,
                goal="根据任务和分析结果构建 Markdown 报告",
                tool_name="build_report",
                tool_input={"analysis": "", "steps": [], "task": task},
                expected_output="Markdown 格式的任务分析报告",
            ),
            PlanStep(
                step_id=4,
                goal="写入 Markdown 报告到本地文件",
                tool_name="file_writer",
                tool_input={"path": output_path},
                expected_output="报告文件已保存",
            ),
        ]


class LLMPlanner:
    """LLM 驱动的动态规划器。

    输入：用户 task + tool_descriptions。
    输出：结构化 PlanStep 列表。
    """

    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client

    def create_plan(
        self,
        task: str,
        input_path: str,
        output_path: str,
        tool_descriptions: list[dict[str, Any]],
    ) -> list[PlanStep]:
        """根据任务和工具描述生成计划。"""
        raw_plan = self._request_plan(task, tool_descriptions)
        plan = self._parse_plan(raw_plan)
        return self._normalize_plan(plan, task, input_path, output_path)

    def _request_plan(self, task: str, tool_descriptions: list[dict[str, Any]]) -> Any:
        """调用 LLMClient 获取原始计划。"""
        if hasattr(self.llm_client, "plan_text_processing"):
            return self.llm_client.plan_text_processing(task, tool_descriptions)
        if hasattr(self.llm_client, "create_plan"):
            return self.llm_client.create_plan(task, tool_descriptions)
        raise RuntimeError("LLM 客户端缺少 plan_text_processing 方法，无法动态规划")

    def _parse_plan(self, raw_plan: Any) -> list[PlanStep]:
        """将 LLM 返回内容解析为 PlanStep 列表。"""
        if isinstance(raw_plan, str):
            raw_plan = _parse_json_content(raw_plan)

        if isinstance(raw_plan, dict):
            raw_steps = raw_plan.get("steps", [])
        elif isinstance(raw_plan, list):
            raw_steps = raw_plan
        else:
            raise ValueError("LLMPlanner 返回内容不是合法的计划结构")

        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("LLMPlanner 未返回有效 steps")

        plan: list[PlanStep] = []
        for index, item in enumerate(raw_steps, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"第 {index} 个计划步骤不是对象")

            tool_name = str(item.get("tool_name", "")).strip()
            if not tool_name:
                raise ValueError(f"第 {index} 个计划步骤缺少 tool_name")

            tool_input = item.get("tool_input", {})
            if not isinstance(tool_input, dict):
                tool_input = {}

            plan.append(
                PlanStep(
                    step_id=int(item.get("step_id") or index),
                    goal=str(item.get("goal") or f"执行 {tool_name}"),
                    tool_name=tool_name,
                    tool_input=tool_input,
                    expected_output=str(item.get("expected_output") or ""),
                )
            )

        return plan

    def _normalize_plan(
        self,
        plan: list[PlanStep],
        task: str,
        input_path: str,
        output_path: str,
    ) -> list[PlanStep]:
        """补全关键参数，并将 step_id 规范为连续编号。"""
        normalized: list[PlanStep] = []
        for index, step in enumerate(plan, start=1):
            tool_input = dict(step.tool_input)

            if step.tool_name == "file_reader":
                tool_input["path"] = tool_input.get("path") or input_path
            elif step.tool_name == "text_extractor":
                tool_input["text"] = tool_input.get("text") or ""
                tool_input["task"] = tool_input.get("task") or task
            elif step.tool_name == "build_report":
                tool_input["analysis"] = tool_input.get("analysis") or ""
                tool_input["steps"] = tool_input.get("steps") or []
                tool_input["task"] = tool_input.get("task") or task
            elif step.tool_name == "file_writer":
                tool_input["path"] = tool_input.get("path") or output_path

            normalized.append(
                PlanStep(
                    step_id=index,
                    goal=step.goal,
                    tool_name=step.tool_name,
                    tool_input=tool_input,
                    expected_output=step.expected_output,
                )
            )

        return normalized


def validate_plan(plan: list[PlanStep], available_tools: set[str]) -> bool:
    """校验计划是否合法。

    当前项目仍是文本处理 Agent，因此要求：
    - 计划非空；
    - 所有 tool_name 都在工具注册表中；
    - step_id 连续；
    - 必须包含 file_reader、text_extractor、build_report、file_writer；
    - 关键工具顺序必须是读取 → 分析 → 报告 → 写入。
    """
    if not plan:
        return False

    expected_step_ids = list(range(1, len(plan) + 1))
    if [step.step_id for step in plan] != expected_step_ids:
        return False

    tool_names = [step.tool_name for step in plan]
    if any(tool_name not in available_tools for tool_name in tool_names):
        return False

    required_tools = ["file_reader", "text_extractor", "build_report", "file_writer"]
    if any(tool_name not in tool_names for tool_name in required_tools):
        return False

    return (
        _first_index(tool_names, "file_reader")
        < _first_index(tool_names, "text_extractor")
        < _first_index(tool_names, "build_report")
        < _first_index(tool_names, "file_writer")
    )


def _first_index(values: list[str], target: str) -> int:
    """返回 target 第一次出现的位置。"""
    try:
        return values.index(target)
    except ValueError:
        return 10**9


def _parse_json_content(content: str) -> Any:
    """解析被 Markdown 代码块包裹或纯 JSON 的 LLM 输出。"""
    text = content.strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").removesuffix("```").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLMPlanner 返回内容不是合法 JSON：{e}")
