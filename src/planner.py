"""任务规划模块。

第二阶段先实现规则版 SimplePlanner，用固定计划体现 Agent Loop 结构。
后续可以将 SimplePlanner 替换为 LLM Planner，实现动态规划。
"""

from __future__ import annotations

try:
    from .schemas import PlanStep
except ImportError:  # pragma: no cover - 支持直接导入
    from schemas import PlanStep


class SimplePlanner:
    """规则版任务规划器。"""

    def create_plan(self, task: str, input_path: str, output_path: str) -> list[PlanStep]:
        """根据任务与输入输出路径生成结构化计划。"""
        return [
            PlanStep(
                step_id=1,
                goal="读取会议记录文件",
                tool_name="file_reader",
                tool_input={"path": input_path},
                expected_output="会议记录文本内容",
            ),
            PlanStep(
                step_id=2,
                goal="分析会议内容并提取结构化信息",
                tool_name="llm_analyze_meeting",
                tool_input={"meeting_text": ""},
                expected_output="会议概要、关键结论、行动项、风险和待确认问题",
            ),
            PlanStep(
                step_id=3,
                goal="将分析结果构建为 Markdown 报告",
                tool_name="build_report",
                tool_input={"analysis": "", "steps": []},
                expected_output="Markdown 格式的会议整理报告",
            ),
            PlanStep(
                step_id=4,
                goal="写入 Markdown 报告到本地文件",
                tool_name="file_writer",
                tool_input={"path": output_path},
                expected_output="报告文件已保存",
            ),
        ]
