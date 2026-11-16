# 实验产物与环境入口

环境安装、数据准备及服务器迁移请阅读 [复现指南](../docs/guides/reproduction.md)，脚本位置见 [脚本索引](../scripts/README.md)，完整目录说明见 [项目结构](../docs/PROJECT_STRUCTURE.md)。

| 路径 | 内容 |
|---|---|
| `run_env.sh`、`bootstrap.sh`、`prepare_data.sh` | 环境运行、安装与数据准备的固定入口 |
| `setup/` | 版本、安装约束及审计记录 |
| `data/` | 数据清单及新服务器的数据准备默认位置 |
| `smoke_tests/` | 短测、性能探测和历史验证证据 |
| `campaigns/` | 正式训练批次、模型、恢复文件与日志 |
| `evaluations/` | 完整动作测评输出 |

八组正式训练已完成：[结果报告](../RESULT.md)、[训练批次](campaigns/20261114_8xa100/README.md)。下一步为 [完整动作测评](../PLAN_12_16_0.md)。

`scripts/`、`results/`、`reports/` 等旧路径为兼容入口，规范位置见项目结构说明。数据本体、大模型、日志和视频不会通过 Git 分发。
