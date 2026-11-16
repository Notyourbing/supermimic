# 20261114：历史训练批次

本目录保存原始八组训练的配置、日志与权重。当前报告使用其中的 LAFAN1 和未筛选 BONES 模型；日常访问统一通过[实验入口](../../experiments/README.md)，固定超参数及历史标识映射集中在[配置说明](../../../docs/EXPERIMENT_CONFIG.md)。

八组历史训练均已完成，原批次并行墙钟 19.80 h、总计 137.38 GPU 小时。当前报告中的两个模型合计 30.03 GPU 小时。两种统计范围不同，详见[结果报告](../../../RESULT.md)。原始模型目录和记录保留用于追溯，不作为下一阶段的实验命名。

## 当前使用的模型

| 实验 | 入口 | 下一阶段用途 |
|---|---|---|
| LAFAN1 | [lafan1/](../../experiments/lafan1/) | 复用模型核验论文指标 |
| 未筛选 BONES | [bones_unfiltered/](../../experiments/bones_unfiltered/) | 复用模型作为训练数据筛选的对照 |

筛选后模型将在新的实验目录中从头训练。执行顺序与单卡预算见 [PLAN](../../../PLAN_12_16_0.md)。旧模型和完整恢复文件不作为新训练的初始化。

## 归档文件索引

| 文件 | 用途 |
|---|---|
| [training_jobs.json](training_jobs.json) | 原始命令、GPU 分工及数据路径 |
| [training_results.json](training_results.json) | 原始八组训练的审计统计 |
| [checkpoint_manifest.json](checkpoint_manifest.json) | 最终权重 SHA256 及恢复文件核验范围 |
| [comparison_analysis.json](comparison_analysis.json) | 历史配置对照，仅留档 |
| [validated_probe_results.json](validated_probe_results.json) | 原批次启动前的独占短测记录 |
| [benchmark.json](benchmark.json)、[BENCHMARK.md](BENCHMARK.md) | 当时的性能预测，不是最终实测结果 |
| [paper_comparison_sources.json](paper_comparison_sources.json) | 论文版本、页码与证据来源 |
| [当前训练统计](../../experiments/training_results.json) | 当前报告使用的两个模型 |
| [报告图表](../../../docs/figures/20261114_8xa100/) | PNG、矢量 PDF 和数据索引 |

原始日志与 TensorBoard 事件位于每个历史模型目录；`hardware.jsonl` 保存硬件采样。资源统计只覆盖模型自身运行时间，不包含其结束后的其他作业。大文件保留本机，Git 主要保存配置与统计。

## 历史监控与恢复

```bash
python3 scripts/monitoring/campaign_dashboard.py reproduction/campaigns/20261114_8xa100
tmux attach -t supermimic
```

上述面板显示历史批次原标识，用于核对原始作业。TensorBoard 地址为 `http://127.0.0.1:6006`。新实验入口使用符号链接，原服务路径不变。

完整恢复要求原任务配置与数据路径匹配，且会重置物理 episode。当前批次已完成，不要重复启动历史队列。本目录四个 Python 文件为[历史调度脚本](../../../scripts/README.md)的兼容链接；原启动过程见[历史批次说明](../../../docs/archive/TRAINING_CAMPAIGN_20261114.md)。
