"""Demo 一键运行脚本。

用法：
    python demo/run_demo.py
    python demo/run_demo.py --input demo/input/meeting_notes.txt --output demo/output/meeting_report.md
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.agent import MeetingReportAgent
from src.console_trace import print_step


def main():
    parser = argparse.ArgumentParser(description="华风灵境 Agent Demo")
    parser.add_argument(
        "--input",
        default=str(project_root / "demo" / "input" / "meeting_notes.txt"),
        help="输入会议记录文件路径",
    )
    parser.add_argument(
        "--output",
        default=str(project_root / "demo" / "output" / "meeting_report_3B_action.md"),
        help="输出 Markdown 报告路径",
    )
    parser.add_argument(
        "--task",
        default="总结内容",
        help="用户任务描述",
    )
    args = parser.parse_args()

    agent = MeetingReportAgent()
    print(f"用户目标：{args.task}\n")
    result = agent.run(args.input, args.output, task=args.task, on_step=print_step)

    print(f"\nPlanner mode: {agent.last_planner_mode}")
    if agent.last_planner_error:
        print(f"Planner error: {agent.last_planner_error}")

    if result.success:
        print(f"\n报告生成成功：{result.output_path}")
    else:
        print(f"\n报告生成失败：{result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
