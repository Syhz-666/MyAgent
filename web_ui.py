"""MyAgent 第四阶段 Web UI 入口。

使用 Gradio 构建功能选择页和通用文本处理助手聊天界面。
"""

from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import gradio as gr

from src.agent import AgentStep, TextProcessingAgent


PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = PROJECT_ROOT / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
PASTED_DIR = RUNTIME_DIR / "pasted"
OUTPUT_DIR = RUNTIME_DIR / "outputs"

WELCOME_MESSAGE = """
欢迎使用通用文本处理助手。

你可以上传一份文本文件，或直接粘贴文本内容，然后输入分析需求。

当前版本每次消息都会基于当前上传文件或粘贴文本重新分析；如需修改结果，请调整任务描述后重新发送。
""".strip()

def ensure_runtime_dirs() -> None:
    """确保 Web UI 运行时目录存在。"""
    for directory in [UPLOAD_DIR, PASTED_DIR, OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def initial_history() -> list[dict[str, str]]:
    """返回聊天窗口初始消息。"""
    return [{"role": "assistant", "content": WELCOME_MESSAGE}]


def on_send(
    task: str,
    history: list[dict[str, str]] | None,
    uploaded_file: Any,
    pasted_text: str,
) -> Iterator[tuple[list[dict[str, str]], str, Any]]:
    """处理用户发送的任务，调用 Agent，并更新聊天历史和下载文件。"""
    ensure_runtime_dirs()
    history = list(history or initial_history())
    task = (task or "").strip()

    if not task:
        history.append(
            {
                "role": "assistant",
                "content": "请先输入分析需求，例如：提取行动项、识别风险点、按主题总结观点。",
            }
        )
        yield history, task, gr.update(value=None, visible=False)
        return

    try:
        input_path, source_notice = prepare_input_file(uploaded_file, pasted_text)
    except ValueError as exc:
        history.append({"role": "assistant", "content": str(exc)})
        yield history, task, gr.update(value=None, visible=False)
        return

    history.append(
        {
            "role": "user",
            "content": f"{source_notice}\n\n**任务**：{task}",
        }
    )
    history.append({"role": "assistant", "content": "正在分析，请稍候..."})
    yield history, "", gr.update(value=None, visible=False)

    agent = TextProcessingAgent()
    output_path = make_output_path(task)
    captured_steps: list[AgentStep] = []

    def capture_step(step: AgentStep) -> None:
        captured_steps.append(step)

    result = agent.run(
        input_path=str(input_path),
        output_path=str(output_path),
        task=task,
        on_step=capture_step,
    )

    if history and history[-1]["content"] == "正在分析，请稍候...":
        history.pop()

    if result.success:
        history.append(
            {
                "role": "assistant",
                "content": "分析完成。\n\n" + strip_execution_section(result.report),
            }
        )
        history.append(
            {
                "role": "assistant",
                "content": build_execution_trace(agent.last_planner_mode, agent.last_planner_error, result.steps),
            }
        )
        yield history, "", gr.update(value=result.output_path, visible=True)
        return

    failure_content = f"分析失败：{result.error or '未知错误'}"
    if result.report:
        failure_content += "\n\n" + strip_execution_section(result.report)
    history.append({"role": "assistant", "content": failure_content})

    steps = result.steps or captured_steps
    if steps:
        history.append(
            {
                "role": "assistant",
                "content": build_execution_trace(agent.last_planner_mode, agent.last_planner_error, steps),
            }
        )

    yield history, task, gr.update(value=None, visible=False)


def prepare_input_file(uploaded_file: Any, pasted_text: str) -> tuple[Path, str]:
    """根据上传文件或粘贴文本生成 Agent 可读取的输入文件。"""
    ensure_runtime_dirs()

    upload_path = resolve_uploaded_file_path(uploaded_file)
    if upload_path is not None:
        if not upload_path.exists() or not upload_path.is_file():
            raise ValueError("上传文件不存在，请重新选择文件。")

        safe_name = safe_filename(upload_path.name)
        target = UPLOAD_DIR / f"{timestamp()}_{uuid.uuid4().hex[:8]}_{safe_name}"
        shutil.copy2(upload_path, target)
        return target, f"**输入来源**：已接收上传文件 `{upload_path.name}`。"

    text = (pasted_text or "").strip()
    if text:
        target = PASTED_DIR / f"{timestamp()}_{uuid.uuid4().hex[:8]}_pasted.txt"
        target.write_text(text, encoding="utf-8")
        return target, f"**输入来源**：已接收粘贴文本，共 {len(text)} 个字符。"

    raise ValueError("请先上传一个文本文件，或在“粘贴文本”区域输入内容。")


def resolve_uploaded_file_path(file_value: Any) -> Path | None:
    """兼容不同 Gradio 版本的文件对象结构，解析上传文件路径。"""
    if not file_value:
        return None

    if isinstance(file_value, (str, Path)):
        return Path(file_value)

    if isinstance(file_value, dict):
        for key in ["path", "name"]:
            value = file_value.get(key)
            if value:
                return Path(value)
        return None

    for attr in ["path", "name"]:
        value = getattr(file_value, attr, None)
        if value:
            return Path(value)

    return None


def make_output_path(task: str) -> Path:
    """根据任务生成报告输出路径。"""
    ensure_runtime_dirs()
    task_stem = safe_filename(task[:32]) or "report"
    return OUTPUT_DIR / f"{timestamp()}_{uuid.uuid4().hex[:8]}_{task_stem}.md"


def safe_filename(value: str) -> str:
    """清理文件名中的不安全字符。"""
    cleaned = "".join(char if char.isalnum() or char in "-_. " else "_" for char in value)
    return cleaned.strip(" ._") or "input.txt"


def timestamp() -> str:
    """生成用于文件名的时间戳。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def strip_execution_section(report: str) -> str:
    """聊天区的报告正文中移除内嵌执行过程，执行过程会作为独立消息展示。"""
    marker = "\n## 执行过程\n"
    if marker in report:
        return report.split(marker, 1)[0].rstrip() + "\n"
    return report


def build_execution_trace(planner_mode: str, planner_error: str, steps: list[AgentStep]) -> str:
    """构建单独展示的 Agent Loop 执行过程。"""
    lines = [
        "### Agent 执行过程",
        "",
        f"- Planner 模式：`{planner_mode or 'unknown'}`",
    ]

    if planner_error:
        lines.append(f"- Planner 回退原因：{planner_error}")

    lines.extend(
        [
            "",
            "| 步骤 | 思考 | 工具调用 | 观察结果 | 状态 |",
            "|------|------|----------|----------|------|",
        ]
    )

    for step in steps:
        lines.append(
            "| {step} | {thought} | {action} | {observation} | {status} |".format(
                step=step.step,
                thought=escape_markdown_table(step.thought),
                action=escape_markdown_table(step.action),
                observation=escape_markdown_table(step.observation),
                status=escape_markdown_table(step.status),
            )
        )

    return "\n".join(lines)


def escape_markdown_table(value: str) -> str:
    """转义 Markdown 表格内容。"""
    return str(value).replace("|", "\\|").replace("\n", " ")


def show_chat_page() -> tuple[Any, Any]:
    """从功能选择页切换到聊天页。"""
    return gr.update(visible=False), gr.update(visible=True)


def show_home_page() -> tuple[Any, Any]:
    """从聊天页返回功能选择页。"""
    return gr.update(visible=True), gr.update(visible=False)


def reset_chat() -> tuple[list[dict[str, str]], Any]:
    """清空聊天历史并隐藏下载组件。"""
    return initial_history(), gr.update(value=None, visible=False)


def create_chatbot() -> gr.Chatbot:
    """创建聊天组件，并兼容不同 Gradio 版本的参数差异。"""
    try:
        return gr.Chatbot(value=initial_history(), type="messages", height=460)
    except TypeError:
        return gr.Chatbot(value=initial_history(), height=460)


def build_ui() -> gr.Blocks:
    """组装完整 Web UI。"""
    ensure_runtime_dirs()

    with gr.Blocks(title="MyAgent") as demo:
        with gr.Column(visible=True) as home_page:
            gr.Markdown(
                """
# MyAgent — AI 办公助手

选择一个功能开始使用。
""".strip()
            )
            with gr.Group():
                gr.Markdown(
                    """
### 通用文本处理助手

支持上传文本文件或粘贴文本内容，并根据你的任务完成提取行动项、总结观点、识别风险等操作。
""".strip()
                )
                enter_button = gr.Button("进入", variant="primary")

            with gr.Group():
                gr.Markdown("### 更多功能开发中\n\n后续可以继续扩展简历筛选、文档问答、日报生成等 Agent 能力。")

        with gr.Column(visible=False) as chat_page:
            with gr.Row():
                back_button = gr.Button("返回首页")
                clear_button = gr.Button("清空对话")

            gr.Markdown("## 通用文本处理助手")
            chatbot = create_chatbot()

            upload_file = gr.File(
                label="上传文件",
                file_types=[".txt", ".md", ".json", ".csv"],
                type="filepath",
            )
            pasted_text = gr.Textbox(
                label="粘贴文本",
                lines=6,
                placeholder="如果不上传文件，可以在这里粘贴需要分析的文本。",
            )
            task_input = gr.Textbox(
                label="输入需求",
                placeholder="例如：提取行动项和负责人；识别风险点；按主题总结核心观点。",
                lines=2,
            )
            send_button = gr.Button("发送", variant="primary")
            report_download = gr.File(label="下载 Markdown 报告", visible=False, interactive=False)

            send_inputs = [task_input, chatbot, upload_file, pasted_text]
            send_outputs = [chatbot, task_input, report_download]
            send_button.click(on_send, inputs=send_inputs, outputs=send_outputs)
            task_input.submit(on_send, inputs=send_inputs, outputs=send_outputs)

            clear_button.click(reset_chat, outputs=[chatbot, report_download])

        enter_button.click(show_chat_page, outputs=[home_page, chat_page])
        back_button.click(show_home_page, outputs=[home_page, chat_page])

    return demo


if __name__ == "__main__":
    build_ui().queue().launch()
