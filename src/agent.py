"""任务驱动式通用文本处理 Agent。

第三阶段 3A 目标：
1. 保留 SimplePlanner 的固定执行计划，保证 Demo 稳定；
2. 将会议记录专用分析升级为 task 驱动的通用文本分析；
3. 使用统一 Schema：summary、sections、follow_up_questions；
4. 根据 task 和 sections 动态生成 Markdown 报告；
5. 保留 MeetingReportAgent 兼容别名，避免旧入口失效。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - 允许在未安装 openai 时使用 MockLLMClient
    OpenAI = None

try:
    from .executor import AgentExecutor
    from .memory import AgentMemory
    from .planner import SimplePlanner
    from .schemas import Observation, PlanStep
except ImportError:  # pragma: no cover - 支持直接运行 python src/agent.py
    from executor import AgentExecutor
    from memory import AgentMemory
    from planner import SimplePlanner
    from schemas import Observation, PlanStep


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

    def analyze_text(self, text: str, task: str) -> dict[str, Any]:
        """根据用户任务分析文本并返回统一结构化结果。"""
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
                "或在项目根目录创建 MyAgentAPIKey.txt 文件"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _read_api_key_from_file() -> str:
        project_root = Path(__file__).resolve().parents[1]
        key_file = project_root / "MyAgentAPIKey.txt"
        if key_file.exists():
            return key_file.read_text(encoding="utf-8").strip()
        return ""

    def analyze_text(self, text: str, task: str) -> dict[str, Any]:
        """根据用户任务动态分析文本，返回统一 JSON Schema。"""
        system_prompt = (
            "你是一个文本分析 Agent。用户会指定分析目标，请根据目标从文本中提取信息。"
            "如果原文没有明确某类信息，对应内容必须标记为“待确认”，不能编造。"
            "只输出合法 JSON，不要输出 Markdown，不要添加额外解释。"
        )
        user_prompt = f"""
任务：{task}

请根据任务分析下面的文本，并输出 JSON。

JSON 格式必须固定为：
{{
  "summary": "文本概要，2-4 句话",
  "sections": [
    {{
      "title": "章节标题，由任务和文本内容决定",
      "items": ["要点1", "要点2"]
    }}
  ],
  "follow_up_questions": ["待确认问题1"]
}}

要求：
1. summary 必须存在；
2. sections 至少包含 1 个章节；
3. sections[].title 应该贴合用户任务；
4. sections[].items 应该是字符串列表；
5. 信息缺失时写“待确认”，不要编造负责人、时间、结论或风险。

文本：
{text}
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
        return _normalize_analysis(_parse_json_content(content))

    def analyze_meeting(self, meeting_text: str) -> dict[str, Any]:
        """兼容第二阶段旧接口。"""
        return self.analyze_text(
            meeting_text,
            "请整理这份会议记录，提取会议概要、关键结论、行动项和风险点。",
        )


class MockLLMClient:
    """本地模拟 LLM 客户端。

    用于没有 API Key 时跑通 Demo。第三阶段 3A 中，Mock 会根据 task 关键词返回不同结构，
    从而证明 --task 参数确实会影响输出。
    """

    def analyze_text(self, text: str, task: str) -> dict[str, Any]:
        task_text = task.lower()

        if any(keyword in task for keyword in ["风险", "待确认", "问题"]):
            return {
                "summary": "本次文本中存在若干需要关注的风险与待确认事项，主要集中在依赖条件、时间安排、验收标准和后续资源协调上。Agent 已根据用户任务聚焦风险识别，而不是生成完整行动项报告。",
                "sections": [
                    {
                        "title": "风险点",
                        "items": [
                            "部分任务依赖外部接口或模型输出，存在稳定性和格式不可控风险。",
                            "如果负责人、截止时间或验收口径不明确，后续执行可能出现责任边界不清。",
                            "Demo 现场可能受到 API Key、网络环境或模型响应格式影响，需要保留 Mock 降级方案。",
                        ],
                    },
                    {
                        "title": "待确认事项",
                        "items": [
                            "是否需要为不同任务单独设计更严格的输出字段校验规则。",
                            "是否需要在第三阶段 3B 中引入更强的计划校验和回退机制。",
                            "是否需要保留第二阶段会议报告格式作为兼容模式。",
                        ],
                    },
                ],
                "follow_up_questions": [
                    "LLMPlanner 失败时是否直接回退 SimplePlanner，还是先要求模型重新生成计划？",
                    "报告章节是否需要限制最大数量，避免输出过长？",
                ],
            }

        if any(keyword in task for keyword in ["行动项", "负责人", "截止", "任务"]):
            return {
                "summary": "本次文本主要围绕项目推进、阶段任务和后续交付展开。Agent 已按用户要求重点提取行动项、负责人、截止时间和执行依据。",
                "sections": [
                    {
                        "title": "行动项",
                        "items": [
                            "完成 3A 改造：负责人待确认；截止时间待确认；内容包括 TextExtractor、动态报告和 task 透传。",
                            "验证同一份输入在不同 task 下生成不同报告：负责人待确认；截止时间待确认。",
                            "保留 MeetingReportAgent 兼容别名，避免旧 Demo 和旧入口失效。",
                        ],
                    },
                    {
                        "title": "负责人和截止时间",
                        "items": [
                            "如原文未明确负责人，标记为待确认，不进行编造。",
                            "如原文未明确截止时间，标记为待确认，不进行编造。",
                        ],
                    },
                ],
                "follow_up_questions": [
                    "是否需要把行动项输出为 Markdown 表格？",
                    "是否需要为行动项增加优先级字段？",
                ],
            }

        if any(keyword in task for keyword in ["观点", "主题", "分类", "总结", "核心"]):
            return {
                "summary": "文本的核心观点可以归纳为 Agent 架构演进、任务驱动分析、稳定 Demo 和后续动态规划四类主题。Agent 已按用户要求进行主题化总结。",
                "sections": [
                    {
                        "title": "Agent 架构演进",
                        "items": [
                            "项目从固定流程升级到计划驱动 Agent Loop，再进一步走向任务驱动文本处理。",
                            "Planner、Executor、Memory、Tool 的分层让项目更接近真实 Agent 产品原型。",
                        ],
                    },
                    {
                        "title": "任务驱动能力",
                        "items": [
                            "第三阶段 3A 的重点是让 --task 参数真正影响分析目标和报告结构。",
                            "统一 Schema 让通用分析结果既稳定又具备扩展性。",
                        ],
                    },
                    {
                        "title": "稳定性设计",
                        "items": [
                            "MockLLMClient 和 SimplePlanner 能保证无 API Key 时仍可运行。",
                            "兼容别名可以降低重构对旧入口的影响。",
                        ],
                    },
                ],
                "follow_up_questions": ["是否需要在 3B 阶段让不同任务产生不同计划步骤？"],
            }

        if "interview" in task_text or "summary" in task_text:
            return {
                "summary": "The text is processed as a general document analysis task. The agent focuses on task-driven extraction and produces adaptive sections based on the user's goal.",
                "sections": [
                    {
                        "title": "Key Points",
                        "items": [
                            "The agent now accepts a custom task and uses it to guide text analysis.",
                            "The report structure is generated from the unified sections schema.",
                            "Fallback behavior remains available for stable demos.",
                        ],
                    }
                ],
                "follow_up_questions": ["Should the output language always follow the user's task language?"],
            }

        return {
            "summary": "Agent 已根据用户任务对文本进行通用分析，并生成统一结构化结果。当前 Mock 模式不会真正理解全文，但会按照任务类型返回可用于 Demo 的稳定结果。",
            "sections": [
                {
                    "title": "关键信息",
                    "items": [
                        "当前流程已从会议记录专用分析升级为通用文本分析。",
                        "分析结果使用 summary、sections、follow_up_questions 统一结构。",
                        "报告标题和章节会根据用户 task 与 sections 动态生成。",
                    ],
                }
            ],
            "follow_up_questions": ["是否需要接入真实模型以获得更贴合原文的分析结果？"],
        }

    def analyze_meeting(self, meeting_text: str) -> dict[str, Any]:
        """兼容第二阶段旧接口。"""
        return self.analyze_text(
            meeting_text,
            "请整理这份会议记录，提取会议概要、关键结论、行动项和风险点。",
        )


class TextProcessingAgent:
    """任务驱动式通用文本处理 Agent。"""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            try:
                self.llm_client = OpenAILLMClient()
            except RuntimeError:
                self.llm_client = MockLLMClient()
        self.planner = SimplePlanner()
        self.executor = AgentExecutor(self.llm_client, build_markdown_report)
        self.steps: list[AgentStep] = []

    def run(
        self,
        input_path: str,
        output_path: str,
        task: str = "请整理这份文本，提取核心信息并生成结构化报告。",
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> AgentResult:
        """执行任务驱动式文本处理 Agent。"""
        self.steps = []
        memory = AgentMemory()
        memory.set("task", task)
        memory.set("input_path", input_path)
        memory.set("output_path", output_path)
        memory.set("steps", self.steps)

        plan = self.planner.create_plan(task, input_path, output_path)

        for plan_step in plan:
            observation = self.executor.execute(plan_step, memory)
            memory.add_observation(observation)
            memory.update_from(plan_step, observation)
            self._record_observation(plan_step, observation)
            memory.set("steps", self.steps)
            if on_step:
                on_step(self.steps[-1])

            if not observation.success:
                return AgentResult(
                    success=False,
                    error=observation.error,
                    report=memory.get("report", ""),
                    steps=self.steps,
                )

        final_report = self._build_final_report(memory)

        final_write_step = PlanStep(
            step_id=len(plan) + 1,
            goal="回写包含完整执行摘要的最终报告",
            tool_name="file_writer",
            tool_input={"path": output_path},
            expected_output="最终报告已保存",
        )
        final_write_observation = self.executor.execute(final_write_step, memory)
        memory.add_observation(final_write_observation)
        memory.update_from(final_write_step, final_write_observation)
        self._record_observation(final_write_step, final_write_observation)
        if on_step:
            on_step(self.steps[-1])

        if not final_write_observation.success:
            return AgentResult(
                success=False,
                error=f"最终报告回写失败：{final_write_observation.error}",
                report=final_report,
                steps=self.steps,
            )

        return AgentResult(
            success=True,
            output_path=memory.get("output_path", output_path),
            report=final_report,
            steps=self.steps,
        )

    def _record_observation(self, plan_step: PlanStep, observation: Observation) -> None:
        """将执行层 Observation 转换为展示层 AgentStep。"""
        self.steps.append(
            AgentStep(
                step=plan_step.step_id,
                thought=plan_step.goal,
                action=plan_step.tool_name,
                observation=self._format_observation(plan_step, observation),
                status="success" if observation.success else "failed",
            )
        )

    def _format_observation(self, plan_step: PlanStep, observation: Observation) -> str:
        """将不同工具的输出压缩成适合报告展示的观察文本。"""
        if not observation.success:
            return observation.error

        if plan_step.tool_name == "file_reader":
            return f"成功读取输入文件，共 {len(str(observation.output))} 个字符。"

        if plan_step.tool_name in {"text_extractor", "llm_analyze_meeting"} and isinstance(observation.output, dict):
            section_count = len(observation.output.get("sections", []))
            question_count = len(observation.output.get("follow_up_questions", []))
            return f"文本分析完成，生成 {section_count} 个动态章节、{question_count} 个待确认问题。"

        if plan_step.tool_name == "build_report":
            return f"Markdown 报告生成完成，共 {len(str(observation.output))} 个字符。"

        if plan_step.tool_name == "file_writer":
            return f"报告写入成功：{observation.output}"

        return _truncate(str(observation.output), 120)

    def _build_final_report(self, memory: AgentMemory) -> str:
        """在所有计划步骤完成后，重建包含完整执行摘要的最终报告。"""
        analysis = memory.get("analysis", {})
        task = memory.get("task", "")
        final_report = build_markdown_report(analysis, self.steps, task)
        memory.set("report", final_report)
        memory.set("content", final_report)
        return final_report


# 向后兼容第二阶段入口和旧 import。
MeetingReportAgent = TextProcessingAgent


def build_markdown_report(analysis: dict[str, Any], steps: list[AgentStep], task: str = "") -> str:
    """根据 task、LLM 分析结果和执行轨迹生成 Markdown 报告。"""
    normalized = _normalize_analysis(analysis)
    lines: list[str] = []
    title = task or "通用文本分析"

    lines.append(f"# 分析报告：{title}")
    lines.append("")

    lines.append("## 概要")
    lines.append("")
    lines.append(str(normalized.get("summary", "待确认")))
    lines.append("")

    sections = normalized.get("sections", [])
    if sections:
        for section in sections:
            section_title = _sanitize_heading(str(section.get("title", "未命名章节")))
            items = section.get("items", [])
            lines.append(f"## {section_title}")
            lines.append("")
            if isinstance(items, list) and items:
                for item in items:
                    lines.append(f"- {item}")
            elif items:
                lines.append(str(items))
            else:
                lines.append("- 待确认")
            lines.append("")
    else:
        lines.append("## 分析结果")
        lines.append("")
        lines.append("- 待确认")
        lines.append("")

    follow_up_questions = normalized.get("follow_up_questions", []) or []
    if follow_up_questions:
        lines.append("## 待确认问题")
        lines.append("")
        for question in follow_up_questions:
            lines.append(f"- {question}")
        lines.append("")

    lines.append("## 执行过程")
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


def _normalize_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    """将分析结果归一化为第三阶段 3A 的统一 Schema。

    兼容旧版会议结构，避免历史代码或旧 Mock 输出导致报告构建失败。
    """
    if not isinstance(analysis, dict):
        return {
            "summary": "待确认",
            "sections": [{"title": "分析结果", "items": [str(analysis)]}],
            "follow_up_questions": [],
        }

    if "summary" in analysis and "sections" in analysis:
        sections = analysis.get("sections") or []
        normalized_sections: list[dict[str, Any]] = []
        for section in sections:
            if isinstance(section, dict):
                items = section.get("items", [])
                if isinstance(items, str):
                    items = [items]
                normalized_sections.append(
                    {
                        "title": str(section.get("title", "未命名章节")),
                        "items": items if isinstance(items, list) else [str(items)],
                    }
                )
            else:
                normalized_sections.append({"title": "分析结果", "items": [str(section)]})

        return {
            "summary": str(analysis.get("summary", "待确认")),
            "sections": normalized_sections or [{"title": "分析结果", "items": ["待确认"]}],
            "follow_up_questions": analysis.get("follow_up_questions", []) or [],
        }

    # 兼容第二阶段会议分析结构。
    sections: list[dict[str, Any]] = []

    key_conclusions = analysis.get("key_conclusions", []) or []
    if key_conclusions:
        sections.append({"title": "关键结论", "items": [str(item) for item in key_conclusions]})

    action_items = analysis.get("action_items", []) or []
    if action_items:
        formatted_actions = []
        for item in action_items:
            if isinstance(item, dict):
                formatted_actions.append(
                    "；".join(
                        [
                            f"任务：{item.get('task', '待确认')}",
                            f"负责人：{item.get('owner', '待确认')}",
                            f"截止时间：{item.get('deadline', '待确认')}",
                            f"优先级：{item.get('priority', '待确认')}",
                        ]
                    )
                )
            else:
                formatted_actions.append(str(item))
        sections.append({"title": "行动项", "items": formatted_actions})

    risks = analysis.get("risks", []) or []
    if risks:
        sections.append({"title": "风险与问题", "items": [str(item) for item in risks]})

    return {
        "summary": str(analysis.get("meeting_summary", analysis.get("summary", "待确认"))),
        "sections": sections or [{"title": "分析结果", "items": ["待确认"]}],
        "follow_up_questions": analysis.get("follow_up_questions", []) or [],
    }


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


def _sanitize_heading(value: str) -> str:
    """清理 Markdown 标题，避免空标题或换行。"""
    cleaned = value.replace("\n", " ").strip().lstrip("#").strip()
    return cleaned or "未命名章节"


def _truncate(value: str, max_length: int) -> str:
    """截断过长文本，保持表格可读性。"""
    if len(value) <= max_length:
        return value
    return value[: max_length - 1] + "…"


if __name__ == "__main__":
    from console_trace import print_step

    project_root = Path(__file__).resolve().parents[1]
    input_file = project_root / "demo" / "input" / "meeting_notes.txt"
    output_file = project_root / "demo" / "output" / "meeting_report.md"

    agent = TextProcessingAgent()
    result = agent.run(
        str(input_file),
        str(output_file),
        on_step=print_step,
    )

    if result.success:
        print(f"报告生成成功：{result.output_path}")
    else:
        print(f"报告生成失败：{result.error}")
