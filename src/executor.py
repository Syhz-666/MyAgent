"""计划执行模块。

Executor 负责根据 PlanStep 调用对应工具，并返回统一的 Observation。
第三阶段 3A 中，工具注册切换为通用 TextExtractor，同时保留旧工具名兼容。
"""

from __future__ import annotations

from typing import Any, Callable

try:
    from .memory import AgentMemory
    from .schemas import Observation, PlanStep
    from .tools import BaseTool, BuildReportTool, FileReader, FileWriter, LLMAnalyzeMeetingTool, TextExtractor
except ImportError:  # pragma: no cover - 支持直接导入
    from memory import AgentMemory
    from schemas import Observation, PlanStep
    from tools import BaseTool, BuildReportTool, FileReader, FileWriter, LLMAnalyzeMeetingTool, TextExtractor


class AgentExecutor:
    """执行 PlanStep 的统一执行器。"""

    def __init__(self, llm_client: Any, report_builder: Callable[..., str]) -> None:
        text_extractor = TextExtractor(llm_client)
        self.tools: dict[str, BaseTool] = {
            "file_reader": FileReader(),
            "file_writer": FileWriter(),
            "text_extractor": text_extractor,
            "llm_analyze_meeting": LLMAnalyzeMeetingTool(llm_client),
            "build_report": BuildReportTool(report_builder),
        }

    def execute(self, step: PlanStep, memory: AgentMemory) -> Observation:
        """执行一个计划步骤，并返回 Observation。"""
        tool = self.tools.get(step.tool_name)
        if tool is None:
            return Observation(
                step_id=step.step_id,
                tool_name=step.tool_name,
                success=False,
                error=f"未知工具：{step.tool_name}",
            )

        try:
            tool_input = self._build_tool_input(step, memory)
            result = tool.run(**tool_input)
            return Observation(
                step_id=step.step_id,
                tool_name=step.tool_name,
                success=result.success,
                output=result.output,
                error=result.error,
            )
        except Exception as e:
            return Observation(
                step_id=step.step_id,
                tool_name=step.tool_name,
                success=False,
                error=str(e),
            )

    def _build_tool_input(self, step: PlanStep, memory: AgentMemory) -> dict[str, Any]:
        """根据工具名筛选并补全参数。

        不直接把整个 memory.data 传给工具，避免出现工具不接受的多余参数。
        PlanStep 中已填的值优先，为空时从 Memory 补充。
        """
        if step.tool_name == "file_reader":
            return {"path": step.tool_input.get("path") or memory.get("input_path")}

        if step.tool_name == "text_extractor":
            return {
                "text": step.tool_input.get("text") or memory.get("text") or memory.get("meeting_text", ""),
                "task": step.tool_input.get("task") or memory.get("task", ""),
            }

        if step.tool_name == "llm_analyze_meeting":
            return {
                "meeting_text": step.tool_input.get("meeting_text") or memory.get("meeting_text") or memory.get("text", ""),
                "task": step.tool_input.get("task") or memory.get("task", ""),
            }

        if step.tool_name == "build_report":
            return {
                "analysis": step.tool_input.get("analysis") or memory.get("analysis", {}),
                "steps": step.tool_input.get("steps") or memory.get("steps", []),
                "task": step.tool_input.get("task") or memory.get("task", ""),
            }

        if step.tool_name == "file_writer":
            return {
                "path": step.tool_input.get("path") or memory.get("output_path"),
                "content": step.tool_input.get("content") or memory.get("content") or memory.get("report", ""),
            }

        return dict(step.tool_input)
