"""任务驱动式通用文本处理 Agent.

第五阶段目标:
1. 保持真实 LLM API 为默认主路径, Mock 仅作为降级;
2. 引入 keyword_search 证据检索工具;
3. 根据 task 意图决定是否检索证据、是否输出可追溯行动项;
4. 保持通用文本处理定位, 非行动项任务不强制渲染行动项表格.
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
    from .planner import LLMPlanner, SimplePlanner, validate_plan
    from .schemas import Observation, PlanStep
except ImportError:  # pragma: no cover - 支持直接运行 python src/agent.py
    from executor import AgentExecutor
    from memory import AgentMemory
    from planner import LLMPlanner, SimplePlanner, validate_plan
    from schemas import Observation, PlanStep


ACTION_ITEM_KEYWORDS = ["行动项", "负责人", "截止", "跟进", "任务", "待办", "分工"]
RISK_KEYWORDS = ["风险", "问题", "阻塞", "延期", "依赖", "不稳定", "待确认"]
ACTION_SEARCH_KEYWORDS = ["负责", "完成", "截止", "跟进", "同步", "推进", "排期", "待办"]
RISK_SEARCH_KEYWORDS = ["风险", "问题", "阻塞", "延期", "依赖", "不稳定", "待确认"]


@dataclass
class AgentStep:
    """记录 Agent 每一步的执行轨迹."""

    step: int
    thought: str
    action: str
    observation: str
    status: str = "success"


@dataclass
class AgentResult:
    """Agent 执行结果."""

    success: bool
    output_path: str = ""
    report: str = ""
    error: str = ""
    steps: list[AgentStep] = field(default_factory=list)


class LLMClient(Protocol):
    """LLM 客户端协议."""

    def analyze_text(
        self,
        text: str,
        task: str,
        search_results: dict[str, Any] | None = None,
        extract_action_items: bool = False,
    ) -> dict[str, Any]:
        """根据用户任务分析文本并返回统一结构化结果."""
        ...

    def plan_text_processing(self, task: str, tool_descriptions: list[dict[str, Any]]) -> dict[str, Any]:
        """根据用户任务和工具描述生成结构化计划."""
        ...


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
        project_root = Path(__file__).resolve().parents[1]
        key_file = project_root / "MyAgentAPIKey.txt"
        if key_file.exists():
            return key_file.read_text(encoding="utf-8").strip()
        return ""

    def plan_text_processing(self, task: str, tool_descriptions: list[dict[str, Any]]) -> dict[str, Any]:
        """根据 task 和工具列表动态生成执行计划."""
        system_prompt = (
            "你是一个文本处理 Agent 的 Planner."
            "你只能使用用户提供的工具名称, 不能编造工具."
            "请只输出合法 JSON, 不要输出 Markdown, 不要添加额外解释."
        )
        user_prompt = f"""
用户任务:{task}

可用工具:
{json.dumps(tool_descriptions, ensure_ascii=False, indent=2)}

请生成执行计划, JSON 格式必须为:
{{
  "steps": [
    {{
      "step_id": 1,
      "goal": "步骤目标",
      "tool_name": "工具名",
      "tool_input": {{}},
      "expected_output": "预期输出"
    }}
  ]
}}

约束:
1. 只能使用可用工具列表中的工具;
2. 必须先读取文件, 再分析文本, 再生成报告, 最后写入文件;
3. keyword_search 是可选工具, 如使用, 必须位于 file_reader 之后、text_extractor 之前;
4. file_reader 的 tool_input 可以留空或包含 path;
5. text_extractor 的 tool_input 必须包含 task;
6. build_report 的 tool_input 必须包含 task;
7. file_writer 必须最后执行.
""".strip()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
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
        """根据用户任务动态分析文本, 返回通用骨架和可选行动项."""
        system_prompt = (
            "你是一个通用文本分析 Agent.用户会指定分析目标, 请根据目标从文本中提取信息."
            "下面“文本”字段一定是用户提供的原始文本, 不得声称文本缺失或未提供."
            "如果用户任务与文本内容明显不匹配（如任务询问'国外发展'但文本只涉及中国情况），在 summary 中如实指出差异，sections 仅提取文本中勉强相关的部分并标注'文本未覆盖'，不要强行将无关内容当答案."
            "summary、sections、follow_up_questions 是通用输出骨架."
            "action_items 是可选业务增强字段, 只有在明确要求提取行动项、负责人、截止时间、跟进事项、任务分工或待办事项时才输出."
            "如果原文没有明确某类信息, 对应内容必须标记为“待确认”或“未在文本中找到对应信息”, 不能编造."
            "只输出合法 JSON, 不要输出 Markdown, 不要添加额外解释."
        )
        action_item_instruction = _build_action_item_instruction(extract_action_items)
        search_context = _format_search_results(search_results)
        user_prompt = f"""
任务:{task}

请根据任务分析下面的文本, 并输出 JSON.

基础 JSON 格式必须包含:
{{
  "summary": "文本概要, 2-4 句话",
  "sections": [
    {{
      "title": "章节标题, 由任务和文本内容决定",
      "items": ["要点1", "要点2"]
    }}
  ],
  "follow_up_questions": ["待确认问题1"]
}}

{action_item_instruction}

要求:
1. summary 必须存在;
2. sections 只包含与用户任务直接相关的章节, sections 的划分服务于用户任务, 而不是还原原文的段落结构;
3. 如果任务是具体问题（如”xxx是什么””xxx的原因是什么”）, 用 1-2 个 sections 聚焦回答即可;
4. 如果任务是全面总结（如”按主题归纳””提取所有要点”）, 可以展开更多章节;
5. sections[].title 应该贴合用户任务;
6. sections[].items 应该是字符串列表;
7. 信息缺失时写”待确认”或”未在文本中找到对应信息”;
8. 不要编造负责人、截止时间、优先级、结论或风险;
9. 搜索结果只是证据候选, 不代表最终结论, 必须结合全文判断.

搜索结果候选:
{search_context}

文本:
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
        """兼容第二阶段旧接口."""
        return self.analyze_text(
            meeting_text,
            "请整理这份会议记录, 提取会议概要, 关键结论, 行动项和风险点.",
            extract_action_items=True,
        )


class MockLLMClient:
    """本地模拟 LLM 客户端.

    用于没有 API Key 时跑通 Demo.Mock 仅作为降级占位, 不作为主要产品能力来源.
    """

    def plan_text_processing(self, task: str, tool_descriptions: list[dict[str, Any]]) -> dict[str, Any]:
        """根据 task 生成可验证差异的动态计划."""
        if any(keyword in task for keyword in RISK_KEYWORDS):
            analysis_goal = "聚焦风险点和待确认事项, 分析文本内容"
            report_goal = "生成风险与待确认事项专项报告"
        elif _should_extract_action_items(task):
            analysis_goal = "聚焦行动项, 负责人, 截止时间和依据片段, 分析文本内容"
            report_goal = "生成可追溯行动项报告"
        elif any(keyword in task for keyword in ["观点", "主题", "分类", "总结", "核心"]):
            analysis_goal = "按主题分类总结核心观点"
            report_goal = "生成主题分类总结报告"
        else:
            analysis_goal = "根据用户任务进行通用文本分析"
            report_goal = "生成通用文本分析报告"

        return {
            "steps": [
                {
                    "step_id": 1,
                    "goal": "读取输入文本文件, 为后续分析提供原文",
                    "tool_name": "file_reader",
                    "tool_input": {},
                    "expected_output": "输入文件文本内容",
                },
                {
                    "step_id": 2,
                    "goal": analysis_goal,
                    "tool_name": "text_extractor",
                    "tool_input": {"task": task},
                    "expected_output": "符合统一 Schema 的结构化分析结果",
                },
                {
                    "step_id": 3,
                    "goal": report_goal,
                    "tool_name": "build_report",
                    "tool_input": {"task": task},
                    "expected_output": "Markdown 报告内容",
                },
                {
                    "step_id": 4,
                    "goal": "将最终 Markdown 报告写入指定输出文件",
                    "tool_name": "file_writer",
                    "tool_input": {},
                    "expected_output": "报告文件路径",
                },
            ]
        }

    def analyze_text(
        self,
        text: str,
        task: str,
        search_results: dict[str, Any] | None = None,
        extract_action_items: bool = False,
    ) -> dict[str, Any]:
        task_text = task.lower()

        if any(keyword in task for keyword in RISK_KEYWORDS):
            return _normalize_analysis(
                {
                    "summary": "本次文本中存在若干需要关注的风险与待确认事项.Agent 已根据用户任务聚焦风险识别, 不会强行生成行动项.",
                    "sections": [
                        {
                            "title": "风险点",
                            "items": _mock_risk_items(text, search_results),
                        },
                        {
                            "title": "待确认事项",
                            "items": [
                                "如原文未明确负责人、截止时间或验收口径, 需要后续确认.",
                                "如风险仅来自上下文推断, 应在最终决策前人工复核.",
                            ],
                        },
                    ],
                    "follow_up_questions": [
                        "是否需要进一步区分高、中、低优先级风险？",
                        "是否需要将风险转化为后续跟进行动项？",
                    ],
                }
            )

        if extract_action_items:
            return _normalize_analysis(
                {
                    "summary": "本次文本包含可跟进事项.Agent 已按用户要求提取行动项, 并为每条行动项保留原文依据片段.",
                    "sections": [
                        {
                            "title": "行动项概览",
                            "items": ["已识别文本中的待办、负责人、截止时间或跟进线索."],
                        }
                    ],
                    "action_items": _mock_action_items(text, search_results),
                    "follow_up_questions": [
                        "是否需要进一步指定每个行动项的优先级？",
                        "是否需要将待确认负责人或截止时间补全？",
                    ],
                }
            )

        if any(keyword in task for keyword in ["观点", "主题", "分类", "总结", "核心"]):
            return _normalize_analysis(
                {
                    "summary": "文本的核心观点可以按主题进行归纳.Agent 已根据用户任务进行通用总结, 不强行生成行动项.",
                    "sections": [
                        {
                            "title": "核心观点",
                            "items": _mock_general_items(text),
                        },
                        {
                            "title": "结构化总结",
                            "items": [
                                "文本信息已按主题组织, 便于进一步复盘或汇报.",
                                "如需更细分类, 可以在任务中指定分类维度.",
                            ],
                        },
                    ],
                    "follow_up_questions": ["是否需要按业务、技术、风险等维度进一步拆分？"],
                }
            )

        if "interview" in task_text or "summary" in task_text:
            return _normalize_analysis(
                {
                    "summary": "The text is processed as a general document analysis task. The agent uses task-driven extraction and adaptive report generation.",
                    "sections": [
                        {
                            "title": "Key Points",
                            "items": _mock_general_items(text),
                        }
                    ],
                    "follow_up_questions": ["Should optional tools be introduced in the next stage?"],
                }
            )

        return _normalize_analysis(
            {
                "summary": "Agent 已根据用户任务对文本进行通用分析, 并生成统一结构化结果.Mock 模式仅用于降级占位.",
                "sections": [
                    {
                        "title": "关键信息",
                        "items": _mock_general_items(text),
                    }
                ],
                "follow_up_questions": ["是否需要指定更具体的分析维度？"],
            }
        )

    def analyze_meeting(self, meeting_text: str) -> dict[str, Any]:
        """兼容第二阶段旧接口."""
        return self.analyze_text(
            meeting_text,
            "请整理这份会议记录, 提取会议概要, 关键结论, 行动项和风险点.",
            extract_action_items=True,
        )


class TextProcessingAgent:
    """任务驱动式通用文本处理 Agent."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            try:
                self.llm_client = OpenAILLMClient()
            except RuntimeError:
                self.llm_client = MockLLMClient()
        self.simple_planner = SimplePlanner()
        self.llm_planner = LLMPlanner(self.llm_client)
        self.executor = AgentExecutor(self.llm_client, build_markdown_report)
        self.steps: list[AgentStep] = []
        self.last_planner_mode = "unknown"
        self.last_planner_error: str = ""

    def run(
        self,
        input_path: str,
        output_path: str,
        task: str = "请整理这份文本, 提取核心信息并生成结构化报告.",
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> AgentResult:
        """执行任务驱动式文本处理 Agent."""
        self.steps = []
        memory = AgentMemory()
        memory.set("task", task)
        memory.set("input_path", input_path)
        memory.set("output_path", output_path)
        memory.set("steps", self.steps)
        memory.set("extract_action_items", _should_extract_action_items(task))

        plan = self._create_plan(task, input_path, output_path)

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
                error=f"最终报告回写失败:{final_write_observation.error}",
                report=final_report,
                steps=self.steps,
            )

        return AgentResult(
            success=True,
            output_path=memory.get("output_path", output_path),
            report=final_report,
            steps=self.steps,
        )

    def _create_plan(self, task: str, input_path: str, output_path: str) -> list[PlanStep]:
        """优先使用 LLMPlanner, 失败或校验不通过时回退 SimplePlanner."""
        available_tools = set(self.executor.tools.keys())
        try:
            plan = self.llm_planner.create_plan(
                task=task,
                input_path=input_path,
                output_path=output_path,
                tool_descriptions=self.executor.tool_descriptions,
            )
            plan = self._enhance_plan(task, plan)
            if validate_plan(plan, available_tools):
                self.last_planner_mode = "llm_planner"
                return plan
        except Exception as e:
            self.last_planner_error = str(e)

        self.last_planner_mode = "simple_planner_fallback"
        return self._enhance_plan(task, self.simple_planner.create_plan(task, input_path, output_path))

    def _enhance_plan(self, task: str, plan: list[PlanStep]) -> list[PlanStep]:
        """根据任务意图增强计划，必要时插入 keyword_search 并标记是否提取行动项."""
        enhanced: list[PlanStep] = []
        should_use_search = _should_use_keyword_search(task)
        has_keyword_search = any(step.tool_name == "keyword_search" for step in plan)
        search_inserted = False
        extract_action_items = _should_extract_action_items(task)

        for step in plan:
            if step.tool_name == "text_extractor":
                tool_input = dict(step.tool_input)
                tool_input["task"] = tool_input.get("task") or task
                tool_input["extract_action_items"] = extract_action_items

                if should_use_search and not has_keyword_search and not search_inserted:
                    enhanced.append(
                        PlanStep(
                            step_id=0,
                            goal="根据用户任务搜索原文中的候选证据片段",
                            tool_name="keyword_search",
                            tool_input={
                                "keywords": _build_search_keywords(task),
                                "context_lines": 1,
                                "max_results": 10,
                            },
                            expected_output="关键词命中行及上下文",
                        )
                    )
                    search_inserted = True

                enhanced.append(
                    PlanStep(
                        step_id=0,
                        goal=step.goal,
                        tool_name=step.tool_name,
                        tool_input=tool_input,
                        expected_output=step.expected_output,
                    )
                )
            else:
                enhanced.append(
                    PlanStep(
                        step_id=0,
                        goal=step.goal,
                        tool_name=step.tool_name,
                        tool_input=dict(step.tool_input),
                        expected_output=step.expected_output,
                    )
                )

        return _renumber_plan(enhanced)

    def _record_observation(self, plan_step: PlanStep, observation: Observation) -> None:
        """将执行层 Observation 转换为展示层 AgentStep."""
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
        """将不同工具的输出压缩成适合报告展示的观察文本."""
        if not observation.success:
            return observation.error

        if plan_step.tool_name == "file_reader":
            return f"成功读取输入文件, 共 {len(str(observation.output))} 个字符."

        if plan_step.tool_name == "keyword_search" and isinstance(observation.output, dict):
            match_count = len(observation.output.get("matches", []))
            keyword_count = len(observation.output.get("keywords", []))
            return f"关键词搜索完成, 使用 {keyword_count} 个关键词, 命中 {match_count} 条候选证据."

        if plan_step.tool_name in {"text_extractor", "llm_analyze_meeting"} and isinstance(observation.output, dict):
            section_count = len(observation.output.get("sections", []))
            question_count = len(observation.output.get("follow_up_questions", []))
            action_count = len(observation.output.get("action_items", []) or [])
            if action_count:
                return f"文本分析完成, 生成 {section_count} 个动态章节, {action_count} 条可追溯行动项, {question_count} 个待确认问题."
            return f"文本分析完成, 生成 {section_count} 个动态章节, {question_count} 个待确认问题."

        if plan_step.tool_name == "build_report":
            return f"Markdown 报告生成完成, 共 {len(str(observation.output))} 个字符."

        if plan_step.tool_name == "file_writer":
            return f"报告写入成功:{observation.output}"

        return _truncate(str(observation.output), 120)

    def _build_final_report(self, memory: AgentMemory) -> str:
        """在所有计划步骤完成后, 重建包含完整执行摘要的最终报告."""
        analysis = memory.get("analysis", {})
        task = memory.get("task", "")
        final_report = build_markdown_report(analysis, self.steps, task)
        memory.set("report", final_report)
        memory.set("content", final_report)
        return final_report


# 向后兼容第二阶段入口和旧 import.
MeetingReportAgent = TextProcessingAgent


def build_markdown_report(analysis: dict[str, Any], steps: list[AgentStep], task: str = "") -> str:
    """根据 task, LLM 分析结果和执行轨迹生成 Markdown 报告."""
    normalized = _normalize_analysis(analysis)
    lines: list[str] = []
    title = task or "通用文本分析"

    lines.append(f"# 分析报告:{title}")
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

    action_items = normalized.get("action_items", []) or []
    if action_items:
        lines.append("## 可追溯行动项")
        lines.append("")
        lines.append("| 行动项 | 负责人 | 截止时间 | 优先级 | 可信度 | 依据片段 |")
        lines.append("|--------|--------|----------|--------|--------|----------|")
        for item in action_items:
            lines.append(
                "| {task} | {owner} | {deadline} | {priority} | {confidence} | {evidence} |".format(
                    task=_escape_markdown_table(str(item.get("task", "待确认"))),
                    owner=_escape_markdown_table(str(item.get("owner", "待确认"))),
                    deadline=_escape_markdown_table(str(item.get("deadline", "待确认"))),
                    priority=_escape_markdown_table(str(item.get("priority", "待确认"))),
                    confidence=_escape_markdown_table(str(item.get("confidence", "待确认"))),
                    evidence=_escape_markdown_table(str(item.get("evidence", "待确认"))),
                )
            )
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
    """将分析结果归一化为第五阶段通用 Schema."""
    if not isinstance(analysis, dict):
        return {
            "summary": "待确认",
            "sections": [{"title": "分析结果", "items": [str(analysis)]}],
            "follow_up_questions": [],
            "action_items": [],
        }

    if "summary" in analysis and "sections" in analysis:
        return {
            "summary": str(analysis.get("summary", "待确认")),
            "sections": _normalize_sections(analysis.get("sections", [])),
            "follow_up_questions": _normalize_string_list(analysis.get("follow_up_questions", []) or []),
            "action_items": _normalize_action_items(analysis.get("action_items", []) or []),
        }

    sections: list[dict[str, Any]] = []

    key_conclusions = analysis.get("key_conclusions", []) or []
    if key_conclusions:
        sections.append({"title": "关键结论", "items": [str(item) for item in key_conclusions]})

    risks = analysis.get("risks", []) or []
    if risks:
        sections.append({"title": "风险与问题", "items": [str(item) for item in risks]})

    action_items = _normalize_action_items(analysis.get("action_items", []) or [])
    if action_items and not sections:
        sections.append({"title": "行动项概览", "items": [str(item.get("task", "待确认")) for item in action_items]})

    return {
        "summary": str(analysis.get("meeting_summary", analysis.get("summary", "待确认"))),
        "sections": sections or [{"title": "分析结果", "items": ["待确认"]}],
        "follow_up_questions": _normalize_string_list(analysis.get("follow_up_questions", []) or []),
        "action_items": action_items,
    }


def _normalize_sections(sections: Any) -> list[dict[str, Any]]:
    """归一化动态章节."""
    if not isinstance(sections, list):
        return [{"title": "分析结果", "items": [str(sections)]}]

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

    return normalized_sections or [{"title": "分析结果", "items": ["待确认"]}]


def _normalize_action_items(action_items: Any) -> list[dict[str, str]]:
    """归一化可追溯行动项."""
    if not isinstance(action_items, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in action_items:
        if isinstance(item, dict):
            task = str(item.get("task", "")).strip()
            if not task:
                continue
            normalized.append(
                {
                    "task": task,
                    "owner": str(item.get("owner", "待确认") or "待确认"),
                    "deadline": str(item.get("deadline", "待确认") or "待确认"),
                    "priority": str(item.get("priority", "待确认") or "待确认"),
                    "evidence": str(item.get("evidence", "待确认") or "待确认")[:80],
                    "confidence": str(item.get("confidence", "待确认") or "待确认"),
                }
            )
    return normalized


def _normalize_string_list(values: Any) -> list[str]:
    """归一化字符串列表."""
    if isinstance(values, list):
        return [str(value) for value in values]
    if values:
        return [str(values)]
    return []


def _should_extract_action_items(task: str) -> bool:
    """判断当前任务是否需要输出可追溯行动项。"""
    return any(keyword in task for keyword in ACTION_ITEM_KEYWORDS)


def _should_use_keyword_search(task: str) -> bool:
    """判断当前任务是否需要先进行关键词证据检索。"""
    return any(keyword in task for keyword in [*ACTION_ITEM_KEYWORDS, *RISK_KEYWORDS])


def _build_search_keywords(task: str) -> list[str]:
    """根据任务意图生成关键词搜索列表。"""
    keywords: list[str] = []
    if any(keyword in task for keyword in ACTION_ITEM_KEYWORDS):
        keywords.extend(ACTION_SEARCH_KEYWORDS)
    if any(keyword in task for keyword in RISK_KEYWORDS):
        keywords.extend(RISK_SEARCH_KEYWORDS)
    return _dedupe(keywords)


def _renumber_plan(plan: list[PlanStep]) -> list[PlanStep]:
    """重排计划步骤编号。"""
    return [
        PlanStep(
            step_id=index,
            goal=step.goal,
            tool_name=step.tool_name,
            tool_input=dict(step.tool_input),
            expected_output=step.expected_output,
        )
        for index, step in enumerate(plan, start=1)
    ]


def _dedupe(values: list[str]) -> list[str]:
    """按原顺序去重。"""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _build_action_item_instruction(extract_action_items: bool) -> str:
    """构建是否输出 action_items 的 prompt 片段。"""
    if not extract_action_items:
        return (
            "action_items 规则:\n"
            "- 当前任务不是行动项类任务, 不要输出 action_items 字段;\n"
            "- 不要为了填充字段而编造任务、负责人或截止时间."
        )

    return """
action_items 规则:
- 当前任务涉及行动项、负责人、截止时间、跟进事项、任务分工或待办事项, 请额外输出 action_items 字段;
- action_items 必须是数组, 每项格式为:
  {
    "task": "行动项描述",
    "owner": "负责人或待确认",
    "deadline": "截止时间或待确认",
    "priority": "high / medium / low / 待确认",
    "evidence": "原文依据片段, 最长 80 字",
    "confidence": "high / medium / low"
  }
- evidence 必须是原文中能支撑该行动项的连续片段, 不能写“根据全文总结”;
- 原文没有明确负责人或截止时间时写“待确认”;
- 找不到明确依据时不要强行生成行动项.
""".strip()


def _format_search_results(search_results: dict[str, Any] | None) -> str:
    """将 keyword_search 结果压缩为 prompt 可读文本。"""
    if not search_results:
        return "无"

    matches = search_results.get("matches", []) if isinstance(search_results, dict) else []
    if not matches:
        return "无命中"

    lines: list[str] = []
    for match in matches[:10]:
        if not isinstance(match, dict):
            continue
        keyword = match.get("keyword", "")
        line_no = match.get("line_no", "")
        line = match.get("line", "")
        before = " / ".join(str(item) for item in match.get("context_before", []) or [])
        after = " / ".join(str(item) for item in match.get("context_after", []) or [])
        context = ""
        if before:
            context += f" 上文:{before}"
        if after:
            context += f" 下文:{after}"
        lines.append(f"- 关键词:{keyword}; 行号:{line_no}; 命中:{line};{context}")

    return "\n".join(lines) or "无命中"


def _mock_action_items(text: str, search_results: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """根据输入文本生成结构正确的 Mock 行动项。"""
    evidence_candidates = _extract_evidence_candidates(text, search_results)
    if not evidence_candidates:
        evidence_candidates = ["原文未找到明确行动项，需人工确认"]

    action_items: list[dict[str, str]] = []
    for evidence in evidence_candidates[:3]:
        action_items.append(
            {
                "task": _summarize_evidence_as_task(evidence),
                "owner": _guess_owner(evidence),
                "deadline": _guess_deadline(evidence),
                "priority": "medium",
                "evidence": evidence[:80],
                "confidence": "medium" if "待确认" in evidence else "high",
            }
        )
    return action_items


def _mock_risk_items(text: str, search_results: dict[str, Any] | None = None) -> list[str]:
    """根据输入文本生成 Mock 风险项。"""
    candidates = _extract_evidence_candidates(text, search_results)
    if candidates:
        return [f"候选风险或问题:{candidate}" for candidate in candidates[:3]]
    return ["未在文本中找到明确风险关键词, 建议结合业务背景人工复核."]


def _mock_general_items(text: str) -> list[str]:
    """根据输入文本生成 Mock 通用要点。"""
    sentences = _split_text_units(text)
    return [sentence[:120] for sentence in sentences[:3]] or ["文本内容较少, 需要更多上下文才能生成详细分析."]


def _extract_evidence_candidates(text: str, search_results: dict[str, Any] | None = None) -> list[str]:
    """优先从搜索结果中提取依据候选，否则从文本中按简单规则截取。"""
    candidates: list[str] = []
    if isinstance(search_results, dict):
        for match in search_results.get("matches", []) or []:
            if isinstance(match, dict) and match.get("line"):
                candidates.append(str(match["line"]).strip())

    if candidates:
        return _dedupe([candidate for candidate in candidates if candidate])

    trigger_words = [*ACTION_SEARCH_KEYWORDS, *RISK_SEARCH_KEYWORDS]
    for unit in _split_text_units(text):
        if any(word in unit for word in trigger_words):
            candidates.append(unit)

    return _dedupe(candidates)


def _split_text_units(text: str) -> list[str]:
    """按行和常见中文标点切分文本片段。"""
    units: list[str] = []
    for line in text.splitlines():
        current = line.strip()
        if not current:
            continue
        for sep in ["。", "；", ";"]:
            current = current.replace(sep, "\n")
        units.extend(part.strip() for part in current.split("\n") if part.strip())
    if not units and text.strip():
        units.append(text.strip())
    return units


def _summarize_evidence_as_task(evidence: str) -> str:
    """用简单规则将依据片段压缩为 Mock 行动项描述。"""
    cleaned = evidence.strip()
    if len(cleaned) <= 36:
        return cleaned
    return cleaned[:35] + "…"


def _guess_owner(evidence: str) -> str:
    """从依据片段中粗略识别负责人，否则待确认。"""
    for marker in ["负责", "由"]:
        if marker in evidence:
            before = evidence.split(marker, 1)[0].strip(" ，,。:：")
            if before:
                return before[-8:]
    return "待确认"


def _guess_deadline(evidence: str) -> str:
    """从依据片段中粗略识别截止时间，否则待确认。"""
    for marker in ["今天", "明天", "本周", "下周", "周一", "周二", "周三", "周四", "周五", "月底"]:
        if marker in evidence:
            return marker
    return "待确认"


def _parse_json_content(content: str) -> dict[str, Any]:
    """解析模型返回的 JSON, 兼容被代码块包裹的情况."""
    text = content.strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").removesuffix("```").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM 返回内容不是合法 JSON:{e}")


def _escape_markdown_table(value: str) -> str:
    """转义 Markdown 表格中的特殊字符."""
    return value.replace("|", "\\|").replace("\n", " ")


def _sanitize_heading(value: str) -> str:
    """清理 Markdown 标题, 避免空标题或换行."""
    cleaned = value.replace("\n", " ").strip().lstrip("#").strip()
    return cleaned or "未命名章节"


def _truncate(value: str, max_length: int) -> str:
    """截断过长文本, 保持表格可读性."""
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
        print(f"报告生成成功:{result.output_path}")
    else:
        print(f"报告生成失败:{result.error}")
