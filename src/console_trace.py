"""终端展示工具。

用于在命令行中实时展示 Agent 每一步的工作状态。
"""

from __future__ import annotations

from typing import Any


def print_step(step: Any) -> None:
    """实时打印单步执行信息，作为 AgentStep 的回调使用。"""
    icon = "OK" if step.status == "success" else "FAIL"
    print(f"  [{icon}] Step {step.step}: {step.action}")
    print(f"       目标: {step.thought}")
    print(f"       结果: {step.observation}")
    print()


