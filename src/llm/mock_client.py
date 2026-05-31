"""Mock LLM 客户端。"""

from __future__ import annotations

from typing import Any

try:
    from ..analysis.normalizer import normalize_analysis
    from ..intent.task_intent import (
        ACTION_SEARCH_KEYWORDS,
        RISK_KEYWORDS,
        RISK_SEARCH_KEYWORDS,
        should_extract_action_items,
    )
except ImportError:  # pragma: no cover - 支持直接导入
    from analysis.normalizer import normalize_analysis
    from intent.task_intent import ACTION_SEARCH_KEYWORDS, RISK_KEYWORDS, RISK_SEARCH_KEYWORDS, should_extract_action_items


class MockLLMClient:
    """本地模拟 LLM 客户端。

    用于没有 API Key 时跑通 Demo.Mock 仅作为降级占位, 不作为主要产品能力来源。
    """

    def plan_text_processing(self, task: str, tool_descriptions: list[dict[str, Any]]) -> dict[str, Any]:
        """根据 task 生成可验证差异的动态计划。"""
        if any(keyword in task for keyword in RISK_KEYWORDS):
            analysis_goal = "聚焦风险点和待确认事项, 分析文本内容"
            report_goal = "生成风险与待确认事项专项报告"
        elif should_extract_action_items(task):
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
            return normalize_analysis(
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
            return normalize_analysis(
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
            return normalize_analysis(
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
            return normalize_analysis(
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

        return normalize_analysis(
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
        """兼容第二阶段旧接口。"""
        return self.analyze_text(
            meeting_text,
            "请整理这份会议记录, 提取会议概要, 关键结论, 行动项和风险点.",
            extract_action_items=True,
        )


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
