"""报告生成模块。"""

from .markdown_report import build_markdown_report
from .table_report import build_table_cleaning_report

__all__ = ["build_markdown_report", "build_table_cleaning_report"]
