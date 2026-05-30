"""固定流程 Agent。

当前版本用于跑通题目一的最小闭环：
1. 读取输入会议记录；
2. 调用 LLM 对会议内容进行结构化分析；
3. 生成 Markdown 报告；
4. 调用文件写入工具保存报告。

说明：
- 这里的 Agent Loop 先采用固定流程，便于快速完成可运行 Demo。
- 后续可以继续拆分出 planner.py、executor.py、memory.py，实现更动态的计划与工具选择。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - 允许在未安装 openai 时使用 MockLLMClient
    OpenAI = None

try:
    from .tools import FileReader, FileWriter, ToolResult
except ImportError:  # pragma: no cover - 支持直接运行 python src/agent.py
    from tools import FileReader, FileWriter, ToolResult


@dataclass
class AgentStep:
    """记录 Agent 每一步的执行轨迹。"""

    step: int
    thought: str
    action: str
    observation: str
    status: str = "success"


@dataclass
class AgentResult:
    """Agent 执行结果。"""

    success: bool
    output_path: str = ""
    report: str = ""
    error: str = ""
    steps: list[AgentStep] = field(default_factory=list)


class LLMClient(Protocol):
    """LLM 客户端协议。"""

    def analyze_meeting(self, meeting_text: str) -> dict[str, Any]:
        """分析会议记录并返回结构化结果。"""
        ...


class OpenAILLMClient:
    """基于 OpenAI SDK 兼容接口的 LLM 客户端，默认对接 DeepSeek。

    可通过环境变量配置：
    - OPENAI_API_KEY：模型 API Key
    - OPENAI_BASE_URL：接口地址，默认 DeepSeek
    - OPENAI_MODEL：模型名称，默认 deepseek-chat
    """

    def __init__(self, model: str | None = None) -> None:
        if OpenAI is None:
            raise RuntimeError("未安装 openai 依赖，请先执行：pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = self._read_api_key_from_file()

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.getenv("OPENAI_MODEL", "deepseek-chat")

        if not api_key:
            raise RuntimeError(
                "缺少 API Key，请设置 OPENAI_API_KEY 环境变量，"
                "或在项目根目录创建 myagentapikey.txt 文件"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _read_api_key_from_file() -> str:
        project_root = Path(__file__).resolve().parents[1]
        key_file = project_root / "MyAgentAPIKey.txt"
        if key_file.exists():
            return key_file.read_text(encoding="utf-8").strip()
        return ""

    def analyze_meeting(self, meeting_text: str) -> dict[str, Any]:
        system_prompt = (
            "你是一个办公资料整理 Agent，擅长从会议记录中提取结构化信息。"
            "请只输出合法 JSON，不要输出 Markdown，不要添加额外解释。"
            "如果原文没有明确负责人或截止时间，必须填写“待确认”，不能编造。"
        )
        user_prompt = f"""
请分析下面的会议记录，并输出 JSON。

JSON 字段要求：
{{
  "meeting_summary": "会议概要，2-4 句话",
  "key_conclusions": ["关键结论1", "关键结论2"],
  "action_items": [
    {{
      "task": "行动项",
      "owner": "负责人，没有则写待确认",
      "deadline": "截止时间，没有则写待确认",
      "priority": "高/中/低/待确认",
      "evidence": "原文依据片段，尽量不超过80字"
    }}
  ],
  "risks": ["风险或待确认问题"],
  "follow_up_questions": ["会后待确认事项"]
}}

会议记录：
{meeting_text}
""".strip()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content or "{}"
        return _parse_json_content(content)


class MockLLMClient:
    """本地模拟 LLM 客户端。

    用于没有 API Key 时跑通 Demo。它不会真正理解文本，只根据当前示例会议记录生成稳定的结构化结果。
    后续接入真实模型时，可替换为 OpenAILLMClient。
    """

    def analyze_meeting(self, meeting_text: str) -> dict[str, Any]:
        return {
            "meeting_summary": (
                "本次会议围绕“华风灵境 Agent：办公资料整理与行动项提取助手”的 MVP 方案展开。"
                "团队确认第一版聚焦会议记录整理，采用本地命令行 Demo，输出 Markdown 报告。"
                "项目重点是体现 Agent Loop、工具调用、结构化输出和可追溯行动项提取。"
            ),
            "key_conclusions": [
                "第一版项目聚焦会议记录整理与行动项提取，不扩展到复杂办公全场景。",
                "Demo 必须体现 Agent Loop 和工具调用过程。",
                "工具层先实现 FileReader 和 FileWriter，后续再增加 TextExtractor、KeywordSearch 等工具。",
                "产品亮点确定为可追溯的行动项提取。",
                "最终输出采用 Markdown 报告，路径暂定为 demo/output/meeting_report.md。",
                "若信息缺失，Agent 应标记为“待确认”，不得编造负责人、时间或结论。",
            ],
            "action_items": [
                {
                    "task": "完成 tools 层基础代码，包括 base_tool.py、file_reader.py 和 file_writer.py，并保证工具返回 ToolResult。",
                    "owner": "王磊",
                    "deadline": "今天 18:00 前",
                    "priority": "高",
                    "evidence": "王磊负责在今天 18:00 前完成 tools 层基础代码，包括 base_tool.py、file_reader.py 和 file_writer.py",
                },
                {
                    "task": "补充一个简单的工具调用示例，方便 Agent 接入。",
                    "owner": "王磊",
                    "deadline": "明天中午前",
                    "priority": "中",
                    "evidence": "王磊还需要在明天中午前补一个简单的工具调用示例，方便 Agent 接入。",
                },
                {
                    "task": "完成 agent.py 的最小固定流程版本，跑通读取会议记录、生成报告、写入结果的闭环。",
                    "owner": "赵敏",
                    "deadline": "明天 20:00 前",
                    "priority": "高",
                    "evidence": "赵敏负责在明天 20:00 前完成 agent.py 的最小固定流程版本",
                },
                {
                    "task": "设计 Markdown 报告模板。",
                    "owner": "陈雪",
                    "deadline": "明天下午 16:00 前",
                    "priority": "中",
                    "evidence": "陈雪负责在明天下午 16:00 前设计 Markdown 报告模板",
                },
                {
                    "task": "准备 2 份不同风格的测试会议记录，并检查输出报告是否存在无依据编造内容。",
                    "owner": "周航",
                    "deadline": "6 月 1 日前",
                    "priority": "中",
                    "evidence": "周航负责在 6 月 1 日前准备 2 份不同风格的测试会议记录",
                },
                {
                    "task": "整理 Demo 演示说明。",
                    "owner": "林佳",
                    "deadline": "6 月 1 日前",
                    "priority": "低",
                    "evidence": "林佳负责在 6 月 1 日前整理 Demo 演示说明",
                },
                {
                    "task": "完成 README 初稿。",
                    "owner": "李明",
                    "deadline": "6 月 2 日前",
                    "priority": "中",
                    "evidence": "李明负责在 6 月 2 日前完成 README 初稿",
                },
            ],
            "risks": [
                "会议记录中的负责人和截止时间可能不明确，Agent 不应强行补全。",
                "后续接入模型后，可能受到上下文长度和输出格式稳定性的影响。",
                "文件路径在不同操作系统下可能存在兼容问题，应尽量使用 pathlib.Path。",
                "依据片段过长会影响表格可读性，需要控制长度。",
            ],
            "follow_up_questions": [
                "是否需要在第一版中加入 text_extractor.py，还是先把抽取逻辑放在 agent.py 中？",
                "README 是否使用中文还是中英文双语？",
                "Demo 是否需要录屏展示？",
            ],
        }


class MeetingReportAgent:
    """会议记录整理 Agent。"""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.reader = FileReader()
        self.writer = FileWriter()
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            try:
                self.llm_client = OpenAILLMClient()
            except RuntimeError:
                self.llm_client = MockLLMClient()
        self.steps: list[AgentStep] = []

    def run(self, input_path: str, output_path: str) -> AgentResult:
        """执行固定流程 Agent。

        Args:
            input_path: 输入会议记录文件路径。
            output_path: 输出 Markdown 报告路径。

        Returns:
            AgentResult: 执行结果。
        """
        self.steps = []

        read_result = self._read_input(input_path)
        if not read_result.success:
            return AgentResult(success=False, error=read_result.error, steps=self.steps)

        meeting_text = read_result.output
        analysis = self._analyze_with_llm(meeting_text)
        if not analysis:
            return AgentResult(
                success=False,
                error="LLM 分析失败，未生成有效结构化结果",
                steps=self.steps,
            )

        report = build_markdown_report(analysis, self.steps)
        write_result = self._write_report(output_path, report)
        if not write_result.success:
            return AgentResult(success=False, error=write_result.error, steps=self.steps)

        final_report = build_markdown_report(analysis, self.steps)
        final_write_result = self.writer.run(path=output_path, content=final_report)
        if not final_write_result.success:
            return AgentResult(
                success=False,
                error=f"最终报告回写失败：{final_write_result.error}",
                report=final_report,
                steps=self.steps,
            )

        return AgentResult(
            success=True,
            output_path=write_result.output,
            report=final_report,
            steps=self.steps,
        )

    def _read_input(self, input_path: str) -> ToolResult:
        self._record_step(
            thought="需要先读取用户提供的会议记录文件。",
            action="file_reader",
            observation=f"准备读取文件：{input_path}",
        )
        result = self.reader.run(path=input_path)
        if result.success:
            self.steps[-1].observation = f"成功读取输入文件，共 {len(result.output)} 个字符。"
        else:
            self.steps[-1].status = "failed"
            self.steps[-1].observation = result.error
        return result

    def _analyze_with_llm(self, meeting_text: str) -> dict[str, Any]:
        self._record_step(
            thought="需要调用 LLM 对会议记录进行结构化分析，提取结论、行动项和风险。",
            action="llm_analyze_meeting",
            observation="准备调用 LLM 进行会议内容分析。",
        )
        try:
            analysis = self.llm_client.analyze_meeting(meeting_text)
        except (ValueError, RuntimeError) as e:
            self.steps[-1].status = "failed"
            self.steps[-1].observation = str(e)
            return {}
        action_count = len(analysis.get("action_items", []))
        conclusion_count = len(analysis.get("key_conclusions", []))
        self.steps[-1].observation = (
            f"LLM 分析完成，提取到 {conclusion_count} 条关键结论、{action_count} 个行动项。"
        )
        return analysis

    def _write_report(self, output_path: str, report: str) -> ToolResult:
        self._record_step(
            thought="需要调用文件写入工具，将最终 Markdown 报告保存到本地。",
            action="file_writer",
            observation=f"准备写入报告：{output_path}",
        )
        result = self.writer.run(path=output_path, content=report)
        if result.success:
            self.steps[-1].observation = f"报告写入成功：{result.output}"
        else:
            self.steps[-1].status = "failed"
            self.steps[-1].observation = result.error
        return result


    def _record_step(self, thought: str, action: str, observation: str, status: str = "success") -> None:
        self.steps.append(
            AgentStep(
                step=len(self.steps) + 1,
                thought=thought,
                action=action,
                observation=observation,
                status=status,
            )
        )


def build_markdown_report(analysis: dict[str, Any], steps: list[AgentStep]) -> str:
    """根据 LLM 分析结果和执行轨迹生成 Markdown 报告。"""
    lines: list[str] = []
    lines.append("# 会议行动项整理报告")
    lines.append("")

    lines.append("## 1. 会议概要")
    lines.append("")
    lines.append(str(analysis.get("meeting_summary", "待确认")))
    lines.append("")

    lines.append("## 2. 关键结论")
    lines.append("")
    for item in analysis.get("key_conclusions", []) or ["待确认"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 3. 行动项")
    lines.append("")
    lines.append("| 行动项 | 负责人 | 截止时间 | 优先级 | 依据片段 |")
    lines.append("|--------|--------|----------|--------|----------|")
    action_items = analysis.get("action_items", [])
    if action_items:
        for item in action_items:
            task = _escape_markdown_table(str(item.get("task", "待确认")))
            owner = _escape_markdown_table(str(item.get("owner", "待确认")))
            deadline = _escape_markdown_table(str(item.get("deadline", "待确认")))
            priority = _escape_markdown_table(str(item.get("priority", "待确认")))
            evidence = _escape_markdown_table(_truncate(str(item.get("evidence", "待确认")), 80))
            lines.append(f"| {task} | {owner} | {deadline} | {priority} | {evidence} |")
    else:
        lines.append("| 待确认 | 待确认 | 待确认 | 待确认 | 待确认 |")
    lines.append("")

    lines.append("## 4. 风险与待确认问题")
    lines.append("")
    risks = analysis.get("risks", []) or []
    follow_up_questions = analysis.get("follow_up_questions", []) or []
    for item in risks + follow_up_questions:
        lines.append(f"- {item}")
    if not risks and not follow_up_questions:
        lines.append("- 暂无")
    lines.append("")

    lines.append("## 5. Agent 执行过程摘要")
    lines.append("")
    lines.append("| 步骤 | 思考 | 动作 | 观察结果 | 状态 |")
    lines.append("|------|------|------|----------|------|")
    for step in steps:
        thought = _escape_markdown_table(step.thought)
        action = _escape_markdown_table(step.action)
        observation = _escape_markdown_table(step.observation)
        status = _escape_markdown_table(step.status)
        lines.append(f"| {step.step} | {thought} | {action} | {observation} | {status} |")

    lines.append("")
    return "\n".join(lines)


def _parse_json_content(content: str) -> dict[str, Any]:
    """解析模型返回的 JSON，兼容被代码块包裹的情况。"""
    text = content.strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").removesuffix("```").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM 返回内容不是合法 JSON：{e}")


def _escape_markdown_table(value: str) -> str:
    """转义 Markdown 表格中的特殊字符。"""
    return value.replace("|", "\\|").replace("\n", " ")


def _truncate(value: str, max_length: int) -> str:
    """截断过长文本，保持表格可读性。"""
    if len(value) <= max_length:
        return value
    return value[: max_length - 1] + "…"


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    input_file = project_root / "demo" / "input" / "meeting_notes.txt"
    output_file = project_root / "demo" / "output" / "meeting_report.md"

    agent = MeetingReportAgent()
    result = agent.run(str(input_file), str(output_file))

    if result.success:
        print(f"报告生成成功：{result.output_path}")
    else:
        print(f"报告生成失败：{result.error}")
