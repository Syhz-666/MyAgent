"""工具基类。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """工具执行结果。"""

    success: bool
    output: Any = None
    error: str = ""


class BaseTool(ABC):
    """所有工具的抽象基类。

    每个工具必须定义 name、description，并实现 run 方法。
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """执行工具逻辑，返回 ToolResult。"""
        ...

    def validate(self, **kwargs) -> bool:
        """可选：参数校验钩子，子类可按需覆盖。"""
        return True
