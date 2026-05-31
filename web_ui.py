"""MyAgent Web UI 入口。

使用 Gradio 构建功能选择页、通用文本处理助手和表格处理助手界面。
"""

from __future__ import annotations

import csv
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

TABLE_DEFAULT_TASK = "清洗这份表格数据"
TABLE_INITIAL_STATUS = """
### 处理摘要

请上传 CSV 文件，输入清洗需求后点击“开始处理”。
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
    """处理文本任务，调用 Agent，并更新聊天历史和下载文件。"""
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


def on_table_send(task: str, uploaded_file: Any) -> Iterator[tuple[str, Any, Any, Any]]:
    """处理表格清洗任务，输出摘要、预览和两个下载文件。"""
    ensure_runtime_dirs()
    task = (task or "").strip() or TABLE_DEFAULT_TASK

    try:
        input_path, source_notice = prepare_table_input_file(uploaded_file)
    except ValueError as exc:
        yield table_error_message(str(exc)), gr.update(value="", visible=False), gr.update(value=None, visible=False), gr.update(value=None, visible=False)
        return

    yield (
        f"### 处理摘要\n\n{source_notice}\n\n正在清洗表格，请稍候...",
        gr.update(value="", visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
    )

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

    steps = result.steps or captured_steps
    if not result.success:
        status = table_error_message(result.error or "未知错误")
        if steps:
            status += "\n\n" + build_execution_trace(agent.last_planner_mode, agent.last_planner_error, steps)
        yield status, gr.update(value="", visible=False), gr.update(value=None, visible=False), gr.update(value=None, visible=False)
        return

    report_path = Path(result.output_path)
    cleaned_path = make_cleaned_table_output_path(report_path)
    status = build_table_result_summary(
        report=result.report,
        planner_mode=agent.last_planner_mode,
        planner_error=agent.last_planner_error,
        steps=result.steps,
        cleaned_path=cleaned_path,
        report_path=report_path,
    )
    preview = build_csv_preview(cleaned_path)

    yield (
        status,
        gr.update(value=preview, visible=True),
        gr.update(value=str(report_path), visible=True),
        gr.update(value=str(cleaned_path), visible=cleaned_path.exists()),
    )


def build_table_result_summary(
    report: str,
    planner_mode: str,
    planner_error: str,
    steps: list[AgentStep],
    cleaned_path: Path,
    report_path: Path,
) -> str:
    """从清洗报告中提取页面摘要。"""
    metrics = extract_table_report_metrics(report)
    lines = [
        "### 处理摘要",
        "",
        "- 状态：处理完成",
        f"- 原始行数：{metrics.get('原始行数', '待确认')}",
        f"- 原始列数：{metrics.get('原始列数', '待确认')}",
        f"- 清洗后行数：{metrics.get('清洗后行数', '待确认')}",
        f"- 问题类型数：{metrics.get('问题类型数', '0')}",
        f"- 清洗操作数：{metrics.get('清洗操作数', '0')}",
        f"- 需人工确认事项：{metrics.get('需人工确认事项', '0')}",
        "",
        "### 输出文件",
        "",
        f"- 清洗后表格：`{cleaned_path}`",
        f"- 清洗报告：`{report_path}`",
        "",
        build_execution_trace(planner_mode, planner_error, steps),
    ]
    return "\n".join(lines)


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


def prepare_table_input_file(uploaded_file: Any) -> tuple[Path, str]:
    """复制上传的 CSV 文件，生成表格工具链可读取的输入文件。"""
    ensure_runtime_dirs()

    upload_path = resolve_uploaded_file_path(uploaded_file)
    if upload_path is None:
        raise ValueError("请先上传一个 CSV 文件。")
    if not upload_path.exists() or not upload_path.is_file():
        raise ValueError("上传文件不存在，请重新选择文件。")
    if upload_path.suffix.lower() != ".csv":
        raise ValueError("当前表格处理页面仅支持 CSV 文件。")

    safe_name = safe_filename(upload_path.name)
    target = UPLOAD_DIR / f"{timestamp()}_{uuid.uuid4().hex[:8]}_{safe_name}"
    shutil.copy2(upload_path, target)
    return target, f"**输入来源**：已接收 CSV 文件 `{upload_path.name}`。"


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


def make_cleaned_table_output_path(report_path: str | Path) -> Path:
    """根据表格清洗报告路径推导 cleaned CSV 路径。"""
    path = Path(report_path)
    return path.with_name(f"{path.stem}_cleaned.csv")


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


def extract_table_report_metrics(report: str) -> dict[str, str]:
    """提取表格清洗报告中的关键指标。"""
    metrics: dict[str, str] = {}
    for line in report.splitlines():
        stripped = line.strip()
        for label in ["原始行数", "原始列数", "清洗后行数"]:
            prefix = f"- {label}："
            if stripped.startswith(prefix):
                metrics[label] = stripped.removeprefix(prefix).strip()

    metrics["问题类型数"] = str(_count_issue_rows(_extract_markdown_section(report, "识别到的问题")))
    metrics["清洗操作数"] = str(_count_meaningful_bullets(_extract_markdown_section(report, "已执行的清洗操作")))
    metrics["需人工确认事项"] = str(_count_meaningful_bullets(_extract_markdown_section(report, "需要人工确认")))
    return metrics


def _extract_markdown_section(report: str, title: str) -> list[str]:
    """按二级标题抽取 Markdown 章节内容。"""
    lines = report.splitlines()
    marker = f"## {title}"
    start_index: int | None = None
    for index, line in enumerate(lines):
        if line.strip() == marker:
            start_index = index + 1
            break

    if start_index is None:
        return []

    section: list[str] = []
    for line in lines[start_index:]:
        if line.startswith("## "):
            break
        section.append(line)
    return section


def _count_issue_rows(section_lines: list[str]) -> int:
    """统计问题表格中的数据行数量。"""
    count = 0
    for line in section_lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if stripped.startswith("| 问题类型") or stripped.startswith("|----------"):
            continue
        count += 1
    return count


def _count_meaningful_bullets(section_lines: list[str]) -> int:
    """统计有效 bullet 数，忽略“暂无/未执行”等空状态提示。"""
    bullets = [line.strip() for line in section_lines if line.strip().startswith("- ")]
    if not bullets:
        return 0
    empty_markers = ["暂无", "未执行", "未识别"]
    return sum(1 for bullet in bullets if not any(marker in bullet for marker in empty_markers))


def build_csv_preview(path: Path, max_rows: int = 10) -> str:
    """构建清洗后 CSV 的 Markdown 预览。"""
    lines = ["### 清洗后表格预览（前 10 行）", ""]
    if not path.exists() or not path.is_file():
        lines.append("清洗后的 CSV 文件不存在，无法预览。")
        return "\n".join(lines)

    rows = read_csv_rows(path)
    if not rows:
        lines.append("清洗后的 CSV 文件为空。")
        return "\n".join(lines)

    headers = rows[0]
    data_rows = rows[1 : max_rows + 1]
    lines.extend(markdown_table(headers, data_rows))
    return "\n".join(lines)


def read_csv_rows(path: Path) -> list[list[str]]:
    """兼容常见编码读取 CSV。"""
    last_error: Exception | None = None
    for encoding in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                return [row for row in csv.reader(file)]
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return []


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """将二维表格转换为 Markdown 表格。"""
    if not headers:
        headers = ["列1"]

    column_count = len(headers)
    table_lines = [
        "| " + " | ".join(_format_preview_cell(header) for header in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        padded = list(row) + [""] * max(0, column_count - len(row))
        table_lines.append("| " + " | ".join(_format_preview_cell(cell) for cell in padded[:column_count]) + " |")
    return table_lines


def _format_preview_cell(value: Any, max_length: int = 40) -> str:
    """格式化预览单元格。"""
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    if len(text) > max_length:
        text = text[: max_length - 1] + "…"
    return escape_markdown_table(text)


def table_error_message(message: str) -> str:
    """构建表格页面错误提示。"""
    return f"### 处理摘要\n\n处理失败：{message}"


def escape_markdown_table(value: str) -> str:
    """转义 Markdown 表格内容。"""
    return str(value).replace("|", "\\|").replace("\n", " ")


def show_chat_page() -> tuple[Any, Any, Any]:
    """从功能选择页切换到文本聊天页。"""
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)


def show_table_page() -> tuple[Any, Any, Any]:
    """从功能选择页切换到表格处理页。"""
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


def show_home_page() -> tuple[Any, Any, Any]:
    """返回功能选择页。"""
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


def reset_chat() -> tuple[list[dict[str, str]], Any]:
    """清空聊天历史并隐藏下载组件。"""
    return initial_history(), gr.update(value=None, visible=False)


def reset_table_page() -> tuple[Any, str, str, Any, Any, Any]:
    """重置表格处理页。"""
    return (
        gr.update(value=None),
        TABLE_DEFAULT_TASK,
        TABLE_INITIAL_STATUS,
        gr.update(value="", visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
    )


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
                enter_text_button = gr.Button("进入文本处理", variant="primary")

            with gr.Group():
                gr.Markdown(
                    """
### 表格处理助手

上传 CSV 文件，自动诊断问题、清洗数据、生成处理报告。
""".strip()
                )
                enter_table_button = gr.Button("进入表格处理", variant="primary")

            gr.Markdown("### 更多功能正在开发\n\n后续可以继续扩展简历筛选、文档问答、日报生成等办公能力。")

        with gr.Column(visible=False) as chat_page:
            with gr.Row():
                back_button = gr.Button("返回首页")
                clear_button = gr.Button("清空对话")

            gr.Markdown("## 通用文本处理助手")
            chatbot = create_chatbot()

            upload_file = gr.File(
                label="上传文本文件",
                file_types=[".txt", ".md", ".json"],
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

        with gr.Column(visible=False) as table_page:
            with gr.Row():
                table_back_button = gr.Button("返回首页")
                table_clear_button = gr.Button("清空结果")

            gr.Markdown(
                """
## 表格处理助手

上传 CSV 文件，输入清洗需求后，MyAgent 会自动完成质量诊断、安全清洗、报告生成和文件输出。
""".strip()
            )
            table_upload_file = gr.File(
                label="上传 CSV 文件",
                file_types=[".csv"],
                type="filepath",
            )
            table_task_input = gr.Textbox(
                label="输入清洗需求",
                value=TABLE_DEFAULT_TASK,
                placeholder="例如：清洗空行、统一日期格式、去重。",
                lines=2,
            )
            table_send_button = gr.Button("开始处理", variant="primary")
            table_status = gr.Markdown(TABLE_INITIAL_STATUS)
            table_preview = gr.Markdown(visible=False)

            with gr.Row():
                table_report_download = gr.File(label="下载清洗报告 MD", visible=False, interactive=False)
                table_cleaned_download = gr.File(label="下载清洗后 CSV", visible=False, interactive=False)

            table_send_inputs = [table_task_input, table_upload_file]
            table_send_outputs = [table_status, table_preview, table_report_download, table_cleaned_download]
            table_send_button.click(on_table_send, inputs=table_send_inputs, outputs=table_send_outputs)
            table_task_input.submit(on_table_send, inputs=table_send_inputs, outputs=table_send_outputs)

            table_clear_button.click(
                reset_table_page,
                outputs=[
                    table_upload_file,
                    table_task_input,
                    table_status,
                    table_preview,
                    table_report_download,
                    table_cleaned_download,
                ],
            )

        enter_text_button.click(show_chat_page, outputs=[home_page, chat_page, table_page])
        enter_table_button.click(show_table_page, outputs=[home_page, chat_page, table_page])
        back_button.click(show_home_page, outputs=[home_page, chat_page, table_page])
        table_back_button.click(show_home_page, outputs=[home_page, chat_page, table_page])

    return demo


def launch_web_ui() -> None:
    """启动 Web UI，并自动打开浏览器。"""
    app = build_ui().queue()
    try:
        app.launch(inbrowser=True)
    except TypeError:
        # 兼容不支持 inbrowser 参数的旧版 Gradio。
        app.launch()


if __name__ == "__main__":
    launch_web_ui()
