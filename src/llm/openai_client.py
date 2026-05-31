"""OpenAI SDK 兼容 LLM 客户端。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - 允许在未安装 openai 时使用 MockLLMClient
    OpenAI = None

try:
    from ..analysis.normalizer import normalize_analysis
    from ..prompts.planning_prompt import build_planning_messages
    from ..prompts.text_analysis_prompt import build_text_analysis_messages
except ImportError:  # pragma: no cover - 支持直接导入
    from analysis.normalizer import normalize_analysis
    from prompts.planning_prompt import build_planning_messages
    from prompts.text_analysis_prompt import build_text_analysis_messages


class OpenAILLMClient:
    """基于 OpenAI SDK 兼容接口的 LLM 客户端, 默认对接 DeepSeek.

    可通过环境变量配置:
    - OPENAI_API_KEY:模型 API Key
    - OPENAI_BASE_URL:接口地址, 默认 DeepSeek
    - OPENAI_MODEL:模型名称, 默认 deepseek-chat
    """

    def __init__(self, model: str | None = None) -> None:
        if OpenAI is None:
            raise RuntimeError("未安装 openai 依赖, 请先执行:pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = self._read_api_key_from_file()

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.getenv("OPENAI_MODEL", "deepseek-chat")

        if not api_key:
            raise RuntimeError(
                "缺少 API Key, 请设置 OPENAI_API_KEY 环境变量, "
                "或在项目根目录创建 MyAgentAPIKey.txt 文件"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _read_api_key_from_file() -> str:
        project_root = Path(__file__).resolve().parents[2]
        key_file = project_root / "MyAgentAPIKey.txt"
        if key_file.exists():
            return key_file.read_text(encoding="utf-8").strip()
        return ""

    def plan_text_processing(self, task: str, tool_descriptions: list[dict[str, Any]]) -> dict[str, Any]:
        """根据 task 和工具列表动态生成执行计划。"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=build_planning_messages(task, tool_descriptions),
            temperature=0.1,
        )
        content = response.choices[0].message.content or "{}"
        return _parse_json_content(content)

    def analyze_text(
        self,
        text: str,
        task: str,
        search_results: dict[str, Any] | None = None,
        extract_action_items: bool = False,
    ) -> dict[str, Any]:
        """根据用户任务动态分析文本, 返回通用骨架和可选行动项。"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=build_text_analysis_messages(
                task=task,
                text=text,
                search_results=search_results,
                extract_action_items=extract_action_items,
            ),
            temperature=0.2,
        )
        content = response.choices[0].message.content or "{}"
        return normalize_analysis(_parse_json_content(content))

    def analyze_meeting(self, meeting_text: str) -> dict[str, Any]:
        """兼容第二阶段旧接口。"""
        return self.analyze_text(
            meeting_text,
            "请整理这份会议记录, 提取会议概要, 关键结论, 行动项和风险点.",
            extract_action_items=True,
        )


def _parse_json_content(content: str) -> dict[str, Any]:
    """解析模型返回的 JSON, 兼容被代码块包裹的情况。"""
    text = content.strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").removesuffix("```").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM 返回内容不是合法 JSON:{e}")
