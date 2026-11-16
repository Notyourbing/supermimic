# 当前实验入口

实验按数据集和训练数据是否筛选命名。研究问题、优先级和预算见 [PLAN](../../PLAN_12_16_0.md)，固定超参数与历史路径映射见 [配置说明](../../docs/EXPERIMENT_CONFIG.md)。

| 入口 | 状态 | 用途 |
|---|---|---|
| [lafan1/](lafan1/) | 已完成训练 | 补齐 LAFAN1 论文指标 |
| [bones_unfiltered/](bones_unfiltered/) | 已完成训练 | 未筛选训练对照 |
| `bones_filtered/` | 待筛选与训练 | 仅改变训练数据的主实验 |
| [training_results.json](training_results.json) | 已核验训练统计 | 当前报告和图表的数据来源，不含正式动作测评结果 |

已有模型通过相对符号链接访问历史产物；日志、权重、恢复文件和旧运行路径保持可追溯。后续新训练直接使用 `bones_filtered/`，不在名称中编码固定超参数。每个目录中的配置、日志、TensorBoard 和检查点布局一致。

正式测评结果保存到相邻的 `evaluations/`。模型权重和大型数据不随 Git 分发，克隆仓库后需从原服务器迁移；复制时保留符号链接。
