# 表格清洗报告：messy_sales.csv

## 输入文件

- 文件名：messy_sales.csv
- 原始行数：5
- 原始列数：5
- 清洗后行数：3

## 识别到的问题

| 问题类型 | 字段 | 数量 | 严重程度 | 示例行 |
|----------|------|------|----------|--------|
| empty_rows | - | 1 | low | 4 |
| duplicate_rows | - | 1 | medium | 3 |
| missing_values | 客户 | 1 | medium | 4 |
| missing_values | 金额 | 2 | medium | 4, 5 |
| missing_values | 日期 | 1 | medium | 4 |
| missing_values | 是否处理 | 1 | medium | 4 |
| missing_values | 备注 | 2 | medium | 4, 5 |

## 已执行的清洗操作

- drop_empty_rows：1 项
- drop_duplicates：1 项
- mark_missing：2 个单元格

## 需要人工确认

- `金额`：数值为负数，已保留未删除，示例行：2

## 输出文件

- 清洗后表格：demo\output\messy_sales_report_cleaned.csv
- 清洗报告：本文件

## 执行过程

| 步骤 | 思考 | 动作 | 观察结果 | 状态 |
|------|------|------|----------|------|
| 1 | 读取 CSV 表格并推断列类型 | table_reader | 成功读取表格, 共 5 行、5 列. | success |
| 2 | 诊断表格中的空行、重复、缺失值和格式问题 | table_profiler | 表格质量诊断完成, 发现 7 类问题. | success |
| 3 | 执行安全清洗并保留需要人工确认的异常值 | table_cleaner | 表格清洗完成, 执行 3 类操作, 记录 1 类需人工确认事项. | success |
| 4 | 写入清洗后的 CSV 表格 | table_writer | 清洗后表格写入成功:demo\output\messy_sales_report_cleaned.csv | success |
| 5 | 生成表格清洗 Markdown 报告 | table_report_builder | 表格清洗报告生成完成, 共 1071 个字符. | success |
