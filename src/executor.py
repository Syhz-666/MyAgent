"""计划执行模块。

Executor 负责根据 PlanStep 调用对应工具，并返回统一的 Observation。
第三阶段 3B 中，Executor 暴露 tool_descriptions，供 LLMPlanner 了解可用工具。
"""

from __future__ import annotations

from typing import Any, Callable

try:
    from .memory import AgentMemory
    from .schemas import Observation, PlanStep
    from .tools import (
        BaseTool,
        BuildReportTool,
        FileReader,
        FileWriter,
        KeywordSearch,
        LLMAnalyzeMeetingTool,
        TableCleaner,
        TableProfiler,
        TableReader,
        TableReportBuilder,
        TableWriter,
        TextExtractor,
    )
except ImportError:  # pragma: no cover - 支持直接导入
    from memory import AgentMemory
    from schemas import Observation, PlanStep
    from tools import (
        BaseTool,
        BuildReportTool,
        FileReader,
        FileWriter,
        KeywordSearch,
        LLMAnalyzeMeetingTool,
        TableCleaner,
        TableProfiler,
        TableReader,
        TableReportBuilder,
        TableWriter,
        TextExtractor,
    )


class AgentExecutor:
    """执行 PlanStep 的统一执行器。"""

    def __init__(self, llm_client: Any, report_builder: Callable[..., str]) -> None:
        text_extractor = TextExtractor(llm_client)
        self.tools: dict[str, BaseTool] = {
            "file_reader": FileReader(),
            "file_writer": FileWriter(),
            "text_extractor": text_extractor,
            "keyword_search": KeywordSearch(),
            "llm_analyze_meeting": LLMAnalyzeMeetingTool(llm_client),
            "build_report": BuildReportTool(report_builder),
            "table_reader": TableReader(),
            "table_profiler": TableProfiler(),
            "table_cleaner": TableCleaner(),
            "table_writer": TableWriter(),
            "table_report_builder": TableReportBuilder(),
        }
        self.tool_descriptions: list[dict[str, Any]] = [
            {
                "name": "file_reader",
                "description": "读取本地文本文件，输出原文文本内容。",
                "params": {"path": "输入文件路径"},
            },
            {
                "name": "text_extractor",
                "description": "根据用户任务从文本中提取结构化信息，输出 summary、sections、follow_up_questions，可选输出 action_items。",
                "params": {
                    "text": "原文文本，由 file_reader 输出补充",
                    "task": "用户任务描述",
                    "search_results": "keyword_search 输出的证据候选，可选",
                    "extract_action_items": "是否要求输出可追溯行动项",
                },
            },
            {
                "name": "keyword_search",
                "description": "按关键词搜索文本，返回命中行、行号和上下文，作为后续分析的证据候选。",
                "params": {
                    "text": "原文文本，由 file_reader 输出补充",
                    "keywords": "关键词列表",
                    "context_lines": "上下文行数",
                    "max_results": "最大命中数量",
                },
            },
            {
                "name": "build_report",
                "description": "根据结构化分析结果、执行步骤和用户任务生成 Markdown 报告。",
                "params": {"analysis": "分析结果 dict", "steps": "执行步骤列表", "task": "用户任务描述"},
            },
            {
                "name": "file_writer",
                "description": "将 Markdown 报告写入本地文件。",
                "params": {"path": "输出文件路径", "content": "报告内容，由 build_report 输出补充"},
            },
            {
                "name": "table_reader",
                "description": "读取 CSV 文件，转换为统一表格结构，并自动推断列类型。",
                "params": {"path": "输入 CSV 文件路径"},
            },
            {
                "name": "table_profiler",
                "description": "分析表格数据质量问题，输出结构化诊断结果。",
                "params": {"table_data": "table_reader 输出的表格结构"},
            },
            {
                "name": "table_cleaner",
                "description": "根据诊断结果执行安全清洗，不删除可能有业务意义的数据。",
                "params": {"table_data": "原始表格结构", "profile": "质量诊断结果"},
            },
            {
                "name": "table_writer",
                "description": "将清洗后的表格数据写入本地 CSV 文件。",
                "params": {"table_data": "table_cleaner 输出", "path": "输出 CSV 文件路径"},
            },
            {
                "name": "table_report_builder",
                "description": "根据表格诊断结果和清洗操作生成 Markdown 清洗报告。",
                "params": {"table_data": "原始表格", "profile": "诊断结果", "cleaner_output": "清洗结果"},
            },
        ]

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
        """根据工具名筛选并补全参数。"""
        if step.tool_name == "file_reader":
            # 输入路径必须以 Agent 运行上下文为准，避免 LLMPlanner 编造路径覆盖真实文件。
            return {"path": memory.get("input_path") or step.tool_input.get("path")}

        if step.tool_name == "table_reader":
            return {"path": memory.get("input_path") or step.tool_input.get("path")}

        if step.tool_name == "table_profiler":
            return {"table_data": memory.get("table_data") or step.tool_input.get("table_data")}

        if step.tool_name == "table_cleaner":
            return {
                "table_data": memory.get("table_data") or step.tool_input.get("table_data"),
                "profile": memory.get("table_profile") or step.tool_input.get("profile"),
            }

        if step.tool_name == "table_writer":
            return {
                "table_data": memory.get("cleaned_data") or step.tool_input.get("table_data"),
                "path": memory.get("table_output_path") or step.tool_input.get("path") or memory.get("output_path"),
            }

        if step.tool_name == "table_report_builder":
            return {
                "table_data": memory.get("table_data") or step.tool_input.get("table_data"),
                "profile": memory.get("table_profile") or step.tool_input.get("profile"),
                "cleaner_output": memory.get("cleaner_output") or step.tool_input.get("cleaner_output"),
                "steps": memory.get("steps", []) or step.tool_input.get("steps") or [],
                "cleaned_path": memory.get("cleaned_table_path") or step.tool_input.get("cleaned_path") or "",
            }

        if step.tool_name == "keyword_search":
            return {
                "text": memory.get("text") or memory.get("meeting_text", "") or step.tool_input.get("text", ""),
                "keywords": step.tool_input.get("keywords", []),
                "context_lines": step.tool_input.get("context_lines", 1),
                "max_results": step.tool_input.get("max_results", 10),
            }

        if step.tool_name == "text_extractor":
            # LLMPlanner 可能会把占位符字符串写入 text（如“原文文本，由 file_reader 输出补充”）。
            # 分析时必须优先使用 file_reader 已写入 Memory 的真实原文，避免把占位符传给模型。
            return {
                "text": memory.get("text") or memory.get("meeting_text", "") or step.tool_input.get("text", ""),
                "task": memory.get("task", "") or step.tool_input.get("task", ""),
                "search_results": memory.get("keyword_search_results") or step.tool_input.get("search_results"),
                "extract_action_items": memory.get("extract_action_items", step.tool_input.get("extract_action_items", False)),
            }

        if step.tool_name == "llm_analyze_meeting":
            return {
                "meeting_text": step.tool_input.get("meeting_text") or memory.get("meeting_text") or memory.get("text", ""),
                "task": step.tool_input.get("task") or memory.get("task", ""),
            }

        if step.tool_name == "build_report":
            # LLMPlanner 可能会把占位符字符串写入 analysis / steps。
            # 报告构建必须优先使用 Memory 中真实的上游结果，避免参数污染。
            return {
                "analysis": memory.get("analysis", {}) or step.tool_input.get("analysis") or {},
                "steps": memory.get("steps", []) or step.tool_input.get("steps") or [],
                "task": memory.get("task", "") or step.tool_input.get("task") or "",
            }

        if step.tool_name == "file_writer":
            # 输出路径必须以 Agent 运行上下文为准，避免 LLMPlanner 编造路径覆盖真实目标。
            return {
                "path": memory.get("output_path") or step.tool_input.get("path"),
                "content": memory.get("content") or memory.get("report", "") or step.tool_input.get("content") or "",
            }

        return dict(step.tool_input)
