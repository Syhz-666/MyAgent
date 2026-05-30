"""Agent 短期记忆模块。

Memory 负责保存一次 Agent 执行过程中的上下文、中间结果和工具观察。
第三阶段 3A 中，Memory 同时兼容第二阶段的 meeting_text 与通用 text。
"""

from __future__ import annotations

from typing import Any

try:
    from .schemas import Observation, PlanStep
except ImportError:  # pragma: no cover - 支持直接导入
    from schemas import Observation, PlanStep


class AgentMemory:
    """Agent 执行过程中的上下文存储。"""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.observations: list[Observation] = []

    def set(self, key: str, value: Any) -> None:
        """写入一个上下文值。"""
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """读取一个上下文值。"""
        return self.data.get(key, default)

    def add_observation(self, observation: Observation) -> None:
        """记录一次工具执行观察。"""
        self.observations.append(observation)

    def update_from(self, step: PlanStep, observation: Observation) -> None:
        """根据工具执行结果更新上下文。

        约定：
        - file_reader 的输出保存为 text，同时兼容 meeting_text；
        - text_extractor / llm_analyze_meeting 的输出保存为 analysis；
        - build_report 的输出保存为 report，同时也作为 file_writer 的 content；
        - file_writer 的输出保存为 written_path。
        """
        if not observation.success:
            return

        if step.tool_name == "file_reader":
            self.set("text", observation.output)
            self.set("meeting_text", observation.output)
        elif step.tool_name in {"text_extractor", "llm_analyze_meeting"}:
            self.set("analysis", observation.output)
        elif step.tool_name == "build_report":
            self.set("report", observation.output)
            self.set("content", observation.output)
        elif step.tool_name == "file_writer":
            self.set("written_path", observation.output)

    def to_dict(self) -> dict[str, Any]:
        """返回当前上下文快照，便于调试。"""
        return dict(self.data)
