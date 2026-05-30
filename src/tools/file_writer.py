"""文件写入工具。"""

from pathlib import Path

from .base_tool import BaseTool, ToolResult


class FileWriter(BaseTool):
    """写入本地文本结果的工具。"""

    name = "file_writer"
    description = "将文本内容写入本地文件，常用于生成 Markdown 报告或 JSON 结果。"

    def run(self, path: str, content: str, encoding: str = "utf-8") -> ToolResult:
        file_path = Path(path)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding=encoding)
            return ToolResult(success=True, output=str(file_path))
        except OSError as e:
            return ToolResult(success=False, error=f"写入失败：{e}")
