"""第二阶段 Agent Loop 数据结构。

本文件只定义执行层数据结构，供 Planner、Executor、Memory 之间流转使用。
展示层的数据结构（AgentStep、AgentResult）仍保留在 agent.py 中。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlanStep:
    """Agent 计划中的单个步骤。"""

    step_id: int
    goal: str
    tool_name: str
    tool_input: dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""


@dataclass
class Observation:
    """工具执行后的观察结果。"""

    step_id: int
    tool_name: str
    success: bool
    output: Any = None
    error: str = ""


@dataclass
class ExecutionContext:
    """一次 Agent 执行的上下文。"""

    task: str
    input_path: str
    output_path: str
