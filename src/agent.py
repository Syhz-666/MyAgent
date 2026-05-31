"""任务驱动式通用文本处理 Agent。

第六阶段目标：
1. agent.py 只保留 Agent 主流程编排；
2. LLM、Prompt、意图判断、结果归一化和报告生成已拆分到独立模块；
3. 保持 CLI、Web UI、KeywordSearch、可选 action_items 和 Mock 降级行为不变。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

try:
    from .executor import AgentExecutor
    from .intent.task_intent import (
        build_search_keywords,
        should_extract_action_items,
        should_use_keyword_search,
    )
    from .llm import LLMClient, MockLLMClient, OpenAILLMClient
    from .memory import AgentMemory
    from .planner import LLMPlanner, SimplePlanner, validate_plan
    from .reports.markdown_report import build_markdown_report
    from .schemas import Observation, PlanStep
except ImportError:  # pragma: no cover - 支持直接运行 python src/agent.py
    from executor import AgentExecutor
    from intent.task_intent import build_search_keywords, should_extract_action_items, should_use_keyword_search
    from llm import LLMClient, MockLLMClient, OpenAILLMClient
    from memory import AgentMemory
    from planner import LLMPlanner, SimplePlanner, validate_plan
    from reports.markdown_report import build_markdown_report
    from schemas import Observation, PlanStep


@dataclass
class AgentStep:
    """记录 Agent 每一步的执行轨迹。"""

    step: int
    thought: str
    action: str
    observation: str
    status: str = "success"


@dataclass
class AgentResult:
    """Agent 执行结果。"""

    success: bool
    output_path: str = ""
    report: str = ""
    error: str = ""
    steps: list[AgentStep] = field(default_factory=list)


class TextProcessingAgent:
    """任务驱动式通用文本处理 Agent。"""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            try:
                self.llm_client = OpenAILLMClient()
            except RuntimeError:
                self.llm_client = MockLLMClient()
        self.simple_planner = SimplePlanner()
        self.llm_planner = LLMPlanner(self.llm_client)
        self.executor = AgentExecutor(self.llm_client, build_markdown_report)
        self.steps: list[AgentStep] = []
        self.last_planner_mode = "unknown"
        self.last_planner_error: str = ""

    def run(
        self,
        input_path: str,
        output_path: str,
        task: str = "请整理这份文本, 提取核心信息并生成结构化报告.",
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> AgentResult:
        """执行任务驱动式文本处理 Agent。"""
        self.steps = []
        memory = AgentMemory()
        memory.set("task", task)
        memory.set("input_path", input_path)
        memory.set("output_path", output_path)
        memory.set("steps", self.steps)
        memory.set("extract_action_items", should_extract_action_items(task))

        plan = self._create_plan(task, input_path, output_path)

        for plan_step in plan:
            observation = self.executor.execute(plan_step, memory)
            memory.add_observation(observation)
            memory.update_from(plan_step, observation)
            self._record_observation(plan_step, observation)
            memory.set("steps", self.steps)
            if on_step:
                on_step(self.steps[-1])

            if not observation.success:
                return AgentResult(
                    success=False,
                    error=observation.error,
                    report=memory.get("report", ""),
                    steps=self.steps,
                )

        final_report = self._build_final_report(memory)

        final_write_step = PlanStep(
            step_id=len(plan) + 1,
            goal="回写包含完整执行摘要的最终报告",
            tool_name="file_writer",
            tool_input={"path": output_path},
            expected_output="最终报告已保存",
        )
        final_write_observation = self.executor.execute(final_write_step, memory)
        memory.add_observation(final_write_observation)
        memory.update_from(final_write_step, final_write_observation)
        self._record_observation(final_write_step, final_write_observation)
        if on_step:
            on_step(self.steps[-1])

        if not final_write_observation.success:
            return AgentResult(
                success=False,
                error=f"最终报告回写失败:{final_write_observation.error}",
                report=final_report,
                steps=self.steps,
            )

        return AgentResult(
            success=True,
            output_path=memory.get("output_path", output_path),
            report=final_report,
            steps=self.steps,
        )

    def _create_plan(self, task: str, input_path: str, output_path: str) -> list[PlanStep]:
        """优先使用 LLMPlanner，失败或校验不通过时回退 SimplePlanner。"""
        available_tools = set(self.executor.tools.keys())
        try:
            plan = self.llm_planner.create_plan(
                task=task,
                input_path=input_path,
                output_path=output_path,
                tool_descriptions=self.executor.tool_descriptions,
            )
            plan = self._enhance_plan(task, plan)
            if validate_plan(plan, available_tools):
                self.last_planner_mode = "llm_planner"
                return plan
        except Exception as e:
            self.last_planner_error = str(e)

        self.last_planner_mode = "simple_planner_fallback"
        return self._enhance_plan(task, self.simple_planner.create_plan(task, input_path, output_path))

    def _enhance_plan(self, task: str, plan: list[PlanStep]) -> list[PlanStep]:
        """根据任务意图增强计划，必要时插入 keyword_search 并标记是否提取行动项。"""
        enhanced: list[PlanStep] = []
        should_use_search = should_use_keyword_search(task)
        has_keyword_search = any(step.tool_name == "keyword_search" for step in plan)
        search_inserted = False
        extract_action_items = should_extract_action_items(task)

        for step in plan:
            if step.tool_name == "text_extractor":
                tool_input = dict(step.tool_input)
                tool_input["task"] = tool_input.get("task") or task
                tool_input["extract_action_items"] = extract_action_items

                if should_use_search and not has_keyword_search and not search_inserted:
                    enhanced.append(
                        PlanStep(
                            step_id=0,
                            goal="根据用户任务搜索原文中的候选证据片段",
                            tool_name="keyword_search",
                            tool_input={
                                "keywords": build_search_keywords(task),
                                "context_lines": 1,
                                "max_results": 10,
                            },
                            expected_output="关键词命中行及上下文",
                        )
                    )
                    search_inserted = True

                enhanced.append(
                    PlanStep(
                        step_id=0,
                        goal=step.goal,
                        tool_name=step.tool_name,
                        tool_input=tool_input,
                        expected_output=step.expected_output,
                    )
                )
            else:
                enhanced.append(
                    PlanStep(
                        step_id=0,
                        goal=step.goal,
                        tool_name=step.tool_name,
                        tool_input=dict(step.tool_input),
                        expected_output=step.expected_output,
                    )
                )

        return _renumber_plan(enhanced)

    def _record_observation(self, plan_step: PlanStep, observation: Observation) -> None:
        """将执行层 Observation 转换为展示层 AgentStep。"""
        self.steps.append(
            AgentStep(
                step=plan_step.step_id,
                thought=plan_step.goal,
                action=plan_step.tool_name,
                observation=self._format_observation(plan_step, observation),
                status="success" if observation.success else "failed",
            )
        )

    def _format_observation(self, plan_step: PlanStep, observation: Observation) -> str:
        """将不同工具的输出压缩成适合报告展示的观察文本。"""
        if not observation.success:
            return observation.error

        if plan_step.tool_name == "file_reader":
            return f"成功读取输入文件, 共 {len(str(observation.output))} 个字符."

        if plan_step.tool_name == "keyword_search" and isinstance(observation.output, dict):
            match_count = len(observation.output.get("matches", []))
            keyword_count = len(observation.output.get("keywords", []))
            return f"关键词搜索完成, 使用 {keyword_count} 个关键词, 命中 {match_count} 条候选证据."

        if plan_step.tool_name in {"text_extractor", "llm_analyze_meeting"} and isinstance(observation.output, dict):
            section_count = len(observation.output.get("sections", []))
            question_count = len(observation.output.get("follow_up_questions", []))
            action_count = len(observation.output.get("action_items", []) or [])
            if action_count:
                return f"文本分析完成, 生成 {section_count} 个动态章节, {action_count} 条可追溯行动项, {question_count} 个待确认问题."
            return f"文本分析完成, 生成 {section_count} 个动态章节, {question_count} 个待确认问题."

        if plan_step.tool_name == "build_report":
            return f"Markdown 报告生成完成, 共 {len(str(observation.output))} 个字符."

        if plan_step.tool_name == "file_writer":
            return f"报告写入成功:{observation.output}"

        return _truncate(str(observation.output), 120)

    def _build_final_report(self, memory: AgentMemory) -> str:
        """在所有计划步骤完成后，重建包含完整执行摘要的最终报告。"""
        analysis = memory.get("analysis", {})
        task = memory.get("task", "")
        final_report = build_markdown_report(analysis, self.steps, task)
        memory.set("report", final_report)
        memory.set("content", final_report)
        return final_report


# 向后兼容第二阶段入口和旧 import。
MeetingReportAgent = TextProcessingAgent


def _renumber_plan(plan: list[PlanStep]) -> list[PlanStep]:
    """重排计划步骤编号。"""
    return [
        PlanStep(
            step_id=index,
            goal=step.goal,
            tool_name=step.tool_name,
            tool_input=dict(step.tool_input),
            expected_output=step.expected_output,
        )
        for index, step in enumerate(plan, start=1)
    ]


def _truncate(value: str, max_length: int) -> str:
    """截断过长文本，保持表格可读性。"""
    if len(value) <= max_length:
        return value
    return value[: max_length - 1] + "…"


if __name__ == "__main__":
    try:
        from .console_trace import print_step
    except ImportError:  # pragma: no cover - 支持直接运行 python src/agent.py
        from console_trace import print_step

    project_root = Path(__file__).resolve().parents[1]
    input_file = project_root / "demo" / "input" / "meeting_notes.txt"
    output_file = project_root / "demo" / "output" / "meeting_report.md"

    agent = TextProcessingAgent()
    result = agent.run(
        str(input_file),
        str(output_file),
        on_step=print_step,
    )

    if result.success:
        print(f"报告生成成功:{result.output_path}")
    else:
        print(f"报告生成失败:{result.error}")
