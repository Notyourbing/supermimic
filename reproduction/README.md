# 实验产物与环境入口

环境安装、数据准备及服务器迁移请阅读 [复现指南](../docs/guides/reproduction.md)，脚本位置见 [脚本索引](../scripts/README.md)，完整目录说明见 [项目结构](../docs/PROJECT_STRUCTURE.md)，固定超参数见 [配置说明](../docs/EXPERIMENT_CONFIG.md)。

| 路径 | 内容 |
|---|---|
| `run_env.sh`、`bootstrap.sh`、`prepare_data.sh` | 环境运行、安装与数据准备的固定入口 |
| `setup/` | 版本、安装约束及审计记录 |
| `data/` | 数据清单及新服务器的数据准备默认位置 |
| `smoke_tests/` | 短测、性能探测和历史验证证据 |
| `experiments/` | 按数据集命名的当前模型入口、训练统计与后续重训目录 |
| `campaigns/` | 正式训练批次、模型、恢复文件与日志 |
| `evaluations/` | 完整动作测评输出 |

已有 LAFAN1 与 BONES 模型的训练结果见 [结果报告](../RESULT.md) 与 [历史训练批次](campaigns/20261114_8xa100/README.md)。下一步先筛选 BONES 并单卡从头重训，再复用旧模型完成对照与 LAFAN1 核验；任务顺序和预算见 [实验计划](../PLAN_12_16_0.md)。

`scripts/`、`results/`、`reports/` 等旧路径为兼容入口，规范位置见项目结构说明。数据本体、大模型、日志和视频不会通过 Git 分发。
