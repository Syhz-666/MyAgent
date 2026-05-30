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


def main():
    parser = argparse.ArgumentParser(description="华风灵境 Agent Demo")
    parser.add_argument(
        "--input",
        default=str(project_root / "demo" / "input" / "meeting_notes.txt"),
        help="输入会议记录文件路径",
    )
    parser.add_argument(
        "--output",
        default=str(project_root / "demo" / "output" / "meeting_report.md"),
        help="输出 Markdown 报告路径",
    )
    args = parser.parse_args()

    agent = MeetingReportAgent()
    result = agent.run(args.input, args.output)

    if result.success:
        print(f"报告生成成功：{result.output_path}")
        for step in result.steps:
            status_icon = "✓" if step.status == "success" else "✗"
            print(f"  {status_icon} Step {step.step}: {step.action} — {step.observation}")
    else:
        print(f"报告生成失败：{result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
