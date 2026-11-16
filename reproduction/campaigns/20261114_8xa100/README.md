# 20261114：八卡正式训练批次

**状态：八组训练均已完成。** 单批并行墙钟 19.80 h，总计 137.38 GPU 小时。完整结果与图表见 [RESULT.md](../../../RESULT.md)，下一阶段见 [动作测评计划](../../../PLAN_12_16_0.md)。

## 模型与路径

| GPU | 模型目录 | 数据 | 表示 | 规模 | 最终快照 |
|---:|---|---|---|---|---|
| 0 | `lafan1_gencoord_x1` | LAFAN1 | GenCoord | x1 | `00320000` |
| 1 | `lafan1_maxcoord_x1` | LAFAN1 | MaxCoord | x1 | `00320000` |
| 2 | `boneseed_gencoord_x1` | BONES 原始 | GenCoord | x1 | `00320000` |
| 3 | `boneseed_maxcoord_x1` | BONES 原始 | MaxCoord | x1 | `00320000` |
| 4 | `lafan1_gencoord_x2` | LAFAN1 | GenCoord | x2 | `00160000` |
| 5 | `lafan1_maxcoord_x2` | LAFAN1 | MaxCoord | x2 | `00160000` |
| 6 | `boneseed_gencoord_x2` | BONES 原始 | GenCoord | x2 | `00160000` |
| 7 | `boneseed_maxcoord_x2` | BONES 原始 | MaxCoord | x2 | `00160000` |

每个模型是独立单卡训练。x1 为 4,096 环境、WM batch 2,048、policy batch 4,096；x2 同时翻倍。两种规模均采集 1,310,720,000 条状态转移。

最终权重在 `<模型目录>/latest/`，实际目标为 `checkpoints/<最终快照>/`，包含 `world_model.pt`、`actor.pt`、`critic.pt`。`resume.pt` 为完整训练恢复文件。大文件只保留本机，Git 中主要保存配置和统计。

## 文件索引

| 文件 | 用途 |
|---|---|
| [training_jobs.json](training_jobs.json) | 原始命令、GPU 分工与数据路径，保留历史记录 |
| [training_results.json](training_results.json) | 已核验训练统计、末段指标与资源占用 |
| [checkpoint_manifest.json](checkpoint_manifest.json) | 最终权重清单、SHA256 及恢复文件核验范围 |
| [comparison_analysis.json](comparison_analysis.json) | 规模与表示对照的派生统计 |
| [validated_probe_results.json](validated_probe_results.json) | 已确认独占的八组短测通过记录 |
| [benchmark.json](benchmark.json)、[BENCHMARK.md](BENCHMARK.md) | 正式启动前的性能预测，区别于最终实际耗时 |
| [paper_comparison_sources.json](paper_comparison_sources.json) | 论文版本、页码与证据来源 |
| [报告图表](../../../docs/figures/20261114_8xa100/) | PNG、矢量 PDF 和绘图数据索引 |

原始日志与 TensorBoard 事件位于每个模型目录；`hardware.jsonl` 保存硬件采样。统计资源使用时按模型自身运行窗口截取，不能把结束后的其他作业算入本项目。

## 查看与恢复

```bash
# 在仓库根目录执行
python3 scripts/monitoring/campaign_dashboard.py reproduction/campaigns/20261114_8xa100
tmux attach -t supermimic
```

TensorBoard 地址为 `http://127.0.0.1:6006`。原服务与 tmux 仍使用本批次路径，目录整理没有移动模型、日志或恢复文件。

继续训练时使用 `training_jobs.json` 中对应模型的原始命令，明确追加 `--resume <模型目录>/resume.pt`。数据路径与配置需要匹配；恢复会重置物理 episode，并非逐比特仿真恢复。当前批次已完成，不要重复运行八卡启动队列。

本目录原有的四个 Python 文件是兼容链接，代码规范位置为 [scripts/training/](../../../scripts/README.md)。这些调度脚本专用于该历史批次。原启动过程、资源竞争记录与时间预测保存在 [历史批次说明](../../../docs/archive/TRAINING_CAMPAIGN_20261114.md)。
