# MyAgent：面向办公功能的统一 Agent 原型

MyAgent 是一个本地可运行的办公 Agent 产品原型，面向常见办公场景提供文本处理和表格处理能力，根据用户任务和输入文件类型自动选择合适的工具链完成任务。

当前版本重点验证：用户给出自然语言任务和办公文件后，Agent 能够规划执行步骤、调用工具、记录过程、生成可下载的结果文件。

---

## 1. 这个 Agent 是什么

MyAgent 是一个统一办公 Agent，核心能力包括：

- 文本分析：对会议纪要、项目记录、需求文档等文本进行总结、行动项提取、风险识别和观点归纳。
- 表格清洗：对 CSV 表格进行质量诊断、空行/重复行处理、缺失值标记、异常值提示，并输出清洗后的 CSV 和处理报告。
- 执行过程可追踪：每次任务都会记录 Agent Loop 中的计划、工具调用和观察结果。
- Web UI 与 CLI 双入口：既可以通过浏览器界面使用，也可以通过命令行运行。

产品定位可以概括为：

```text
一个统一的办公 Agent，多条办公能力工具链。
```

---

## 2. 解决什么办公场景问题

MyAgent 面向日常办公中“文件多、整理慢、结果难追溯”的问题，当前覆盖两个典型场景。

### 2.1 文本处理场景

适用于：

- 会议记录整理
- 项目复盘材料总结
- 需求文档要点提取
- 行动项、负责人、截止时间提取
- 风险点、阻塞项、待确认问题识别

示例任务：

```text
提取行动项和负责人
识别这份会议记录中的风险点
按主题总结核心观点
生成一份结构化分析报告
```

输出结果包括：

- 概要
- 动态章节
- 可追溯行动项，包含负责人、截止时间、依据片段、可信度
- 待确认问题
- Agent 执行过程

### 2.2 表格处理场景

适用于：

- 销售表、客户表、任务台账等 CSV 数据清洗
- 去除空行、重复行
- 识别缺失值
- 推断列类型
- 保留但提示可能异常的数据，例如负数金额
- 生成清洗报告和清洗后 CSV

示例任务：

```text
清洗这份表格数据
帮我检查这份 CSV 的数据质量
去重并标记缺失值
```

输出结果包括：

- 清洗后 CSV 文件
- Markdown 清洗报告
- 问题清单
- 清洗操作记录
- 需要人工确认的异常值
- Agent 执行过程

---

## 3. Agent Loop 如何实现

MyAgent 的核心是一个任务驱动的 Agent Loop：

```text
用户任务 / 输入文件
↓
识别任务意图和输入类型
↓
生成或选择执行计划 Plan
↓
Executor 调用工具
↓
Memory 保存中间结果和观察
↓
生成报告 / 输出文件
↓
返回执行轨迹和结果
```

### 3.1 文本任务链路

文本类任务会优先尝试使用 LLMPlanner 生成计划，失败时回退到 SimplePlanner。对于需要证据增强的任务，会自动插入关键词检索工具。

典型链路：

```text
file_reader
→ keyword_search，可选
→ text_extractor
→ build_report
→ file_writer
```

说明：

- `file_reader` 读取输入文本。
- `keyword_search` 在行动项、风险类任务中检索候选证据片段。
- `text_extractor` 调用 LLM 对文本进行结构化分析。
- `build_report` 生成 Markdown 报告。
- `file_writer` 写入本地结果文件。

### 3.2 表格任务链路

表格处理目前是确定性规则链路，不依赖 LLM 规划，保证 CSV 清洗行为稳定可控。

典型链路：

```text
table_reader
→ table_profiler
→ table_cleaner
→ table_writer
→ table_report_builder
→ file_writer
```

说明：

- `table_reader` 读取 CSV，推断列类型。
- `table_profiler` 诊断空行、重复行、缺失值、格式问题等。
- `table_cleaner` 执行安全清洗，不删除可能有业务意义的异常行。
- `table_writer` 输出清洗后的 CSV。
- `table_report_builder` 生成清洗报告。
- `file_writer` 写入最终 Markdown 报告。

### 3.3 Memory 与 Observation

每个工具执行后都会返回统一的 Observation。AgentMemory 会保存：

- 用户任务
- 输入路径
- 输出路径
- 文本内容或表格数据
- 分析结果
- 清洗结果
- 报告内容
- 每一步执行轨迹

最终报告中的“执行过程”就是由这些步骤记录生成的。

---

## 4. 支持哪些常用工具

当前工具分为通用工具、文本工具和表格工具。

### 4.1 通用工具

| 工具 | 作用 |
|------|------|
| `file_reader` | 读取本地文本文件 |
| `file_writer` | 写入 Markdown 报告或结果文件 |
| `build_report` | 根据结构化分析结果生成 Markdown 报告 |

### 4.2 文本处理工具

| 工具 | 作用 |
|------|------|
| `text_extractor` | 根据用户任务进行文本结构化分析 |
| `keyword_search` | 根据关键词搜索原文证据片段 |
| `llm_analyze_meeting` | 兼容早期会议纪要分析入口 |

### 4.3 表格处理工具

| 工具 | 作用 |
|------|------|
| `table_reader` | 读取 CSV，标准化表头并推断列类型 |
| `table_profiler` | 诊断空行、空列、重复行、缺失值、类型异常、文本不规范等问题 |
| `table_cleaner` | 删除空行/重复行，标记缺失值，保留异常值并记录 warning |
| `table_writer` | 输出清洗后的 CSV 文件 |
| `table_report_builder` | 生成表格清洗 Markdown 报告 |

---

## 5. 产品亮点

### 5.1 一个 Agent，多种办公能力

MyAgent 不是“文本 Agent + 表格 Agent”的简单拼接，而是同一个 Agent 根据任务和文件类型选择不同工具链。这样更符合真实办公助手的使用方式：用户只需要提出任务，系统负责判断如何执行。

### 5.2 任务驱动，而不是固定按钮流程

用户可以输入自然语言任务，例如：

```text
提取行动项和负责人
识别风险点
清洗这份表格数据
```

Agent 会根据任务动态选择或增强执行计划。

### 5.3 模块化架构，具备持续扩展能力

项目具备继续扩展办公能力的潜力：新增能力时不需要重写 Agent 主循环，只需要按现有结构增加新的工具、报告模板和意图路由。例如新增表格处理能力时，就是在同一个 Agent 内接入了一条新的表格工具链，而不是重新开发一个独立 Agent。

这种结构证明 MyAgent 可以继续扩展到文档问答、日报生成、简历筛选、合同审阅等更多办公场景。

### 5.4 真实 LLM 默认，Mock 仅作降级

项目默认使用兼容 OpenAI SDK 的真实 LLM 接口。MockLLMClient 只在缺少 API Key、依赖不可用或网络不可用时作为占位降级，保证 Demo 可以跑通，但产品主路径面向真实 LLM 行为设计。

### 5.5 证据增强与可追溯行动项

行动项类任务会自动使用 `keyword_search` 检索原文证据，最终报告中的行动项会尽量包含：

- 行动项描述
- 负责人
- 截止时间
- 优先级
- 原文依据片段
- 可信度

### 5.6 表格清洗强调“安全”

表格清洗不会随意删除可能有业务含义的数据。例如负数金额不会被删除，而是保留并记录为需要人工确认的 warning。

### 5.7 Web UI 可演示

项目提供 Gradio Web UI：

- 首页选择办公能力
- 文本处理页面支持上传文件或粘贴文本
- 表格处理页面支持上传 CSV、查看清洗摘要、预览清洗后表格、下载结果文件

---

## 6. 项目结构

```text
MyAgent/
├── web_ui.py                    # Gradio Web UI 入口
├── run_web_ui.bat               # Windows 双击启动脚本
├── requirements.txt             # Python 依赖
├── demo/
│   ├── input/                   # 示例输入文件
│   └── output/                  # 示例输出文件
└── src/
    ├── main.py                  # CLI 入口
    ├── agent.py                 # Agent 主流程编排
    ├── executor.py              # 工具执行器
    ├── memory.py                # Agent 短期记忆
    ├── planner.py               # LLMPlanner / SimplePlanner / plan 校验
    ├── schemas.py               # PlanStep / Observation 等结构
    ├── intent/                  # 任务意图判断
    ├── prompts/                 # LLM prompt 构造
    ├── llm/                     # LLM 客户端与 Mock 降级
    ├── analysis/                # LLM 输出归一化
    ├── reports/                 # Markdown 报告模板
    └── tools/                   # Agent 可调用工具
```

---

## 7. 如何安装和配置

### 7.1 环境要求

- Python 3.9 或以上
- Windows / macOS / Linux 均可运行
- 如使用 Web UI，需要安装 Gradio

### 7.2 安装依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

如果 Windows 上 `python` 命令不可用，可以使用：

```bash
py -m pip install -r requirements.txt
```

### 7.3 配置 LLM API Key

方式一：设置环境变量：

```bash
set OPENAI_API_KEY=你的APIKey
```

方式二：在项目根目录创建文件：

```text
MyAgentAPIKey.txt
```

文件内容填写 API Key。

可选环境变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_API_KEY` | LLM API Key | 无 |
| `OPENAI_BASE_URL` | OpenAI SDK 兼容接口地址 | `https://api.deepseek.com` |
| `OPENAI_MODEL` | 模型名称 | `deepseek-chat` |

如果没有配置 API Key，系统会自动降级到 MockLLMClient，用于本地演示闭环。

---

## 8. 如何启动和使用

### 8.1 启动 Web UI

在项目根目录执行：

```bash
py web_ui.py
```

或：

```bash
python web_ui.py
```

Windows 用户也可以直接双击：

```text
run_web_ui.bat
```

启动后浏览器会自动打开 MyAgent 页面。

### 8.2 使用文本处理助手

1. 首页选择“通用文本处理助手”。
2. 上传 `.txt` / `.md` / `.json` 文件，或直接粘贴文本。
3. 输入任务，例如：

```text
提取行动项和负责人
```

4. 点击“发送”。
5. 查看分析结果和 Agent 执行过程。
6. 下载 Markdown 报告。

### 8.3 使用表格处理助手

1. 首页选择“表格处理助手”。
2. 上传 `.csv` 文件。
3. 输入任务，例如：

```text
清洗这份表格数据
```

4. 点击“开始处理”。
5. 查看处理摘要和清洗后表格预览。
6. 下载：
   - 清洗报告 `.md`
   - 清洗后表格 `.csv`

---

## 9. 命令行使用示例

### 9.1 默认示例

```bash
py -m src.main
```

默认会处理：

```text
demo/input/meeting_notes.txt
```

并输出：

```text
demo/output/meeting_report.md
```

### 9.2 文本任务

```bash
py -m src.main --input "demo/input/meeting_notes.txt" --output "demo/output/meeting_report.md" --task "提取行动项和负责人"
```

### 9.3 表格任务

```bash
py -m src.main --input "demo/input/messy_sales.csv" --output "demo/output/messy_sales_report.md" --task "清洗这份表格数据"
```

表格任务会额外生成清洗后的 CSV：

```text
demo/output/messy_sales_report_cleaned.csv
```

---

## 10. 当前支持范围与限制

当前已支持：

- 文本文件：`.txt`、`.md`、`.json`
- 表格文件：`.csv`
- Markdown 报告输出
- 清洗后 CSV 输出
- Web UI 使用
- CLI 使用
- 无 API Key 时 Mock 降级演示

当前限制：

- 表格处理 P0 阶段仅支持 CSV，暂不支持 `.xlsx` 和多 Sheet。
- 表格清洗以规则为主，暂未使用 LLM 理解复杂清洗策略。
- 日期、金额、布尔值当前主要用于类型推断和异常识别，进一步标准化可以作为后续增强。
- Web UI 当前以本地单用户演示为主，未做用户权限、任务队列和多人协作。

---

## 11. 后续可扩展方向

可以继续扩展为更完整的办公 Agent：

- 支持 Excel `.xlsx` 多 Sheet 读取和写入
- 增加日期、金额、布尔值标准化
- 输出异常记录 CSV
- 使用 LLM 生成表格清洗策略 JSON
- 支持文档问答
- 支持日报、周报自动生成
- 支持简历筛选、合同审阅等更多办公工具链
- 增加任务历史、文件管理和多轮上下文

---

## 12. 一句话总结

MyAgent 是一个面向办公场景的统一 Agent 原型。它通过 Agent Loop 将任务理解、计划生成、工具调用、短期记忆、报告生成和文件输出串联起来，让用户可以用自然语言完成文本分析和表格清洗等常见办公任务。
