"""文件读取工具。"""

from pathlib import Path

from .base_tool import BaseTool, ToolResult


class FileReader(BaseTool):
    """读取本地文本资料的工具。"""

    name = "file_reader"
    description = "读取本地 txt、md、json、csv 等文本文件内容。"

    def run(self, path: str, encoding: str = "utf-8") -> ToolResult:
        file_path = Path(path)

        if not file_path.exists():
            return ToolResult(success=False, error=f"文件不存在：{path}")

        if not file_path.is_file():
            return ToolResult(success=False, error=f"路径不是文件：{path}")

        try:
            content = file_path.read_text(encoding=encoding)
            return ToolResult(success=True, output=content)
        except UnicodeDecodeError as e:
            return ToolResult(success=False, error=f"编码错误：{e}")
