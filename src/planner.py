"""任务规划模块。

第三阶段 3A 仍保留规则版 SimplePlanner，不引入 LLMPlanner。
区别是固定计划已经切换为通用文本处理流程，并将 task 传递给分析和报告工具。
"""

from __future__ import annotations

try:
    from .schemas import PlanStep
except ImportError:  # pragma: no cover - 支持直接导入
    from schemas import PlanStep


class SimplePlanner:
    """规则版任务规划器，作为第三阶段 3A 的稳定执行计划。"""

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
