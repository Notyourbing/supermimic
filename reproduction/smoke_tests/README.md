# 短测与性能探测记录

本目录保存有限步 smoke test、加载/回放检查和性能探测，不能直接解释为训练完成或论文测评结果。正式模型见 [campaigns/](../campaigns/20261114_8xa100/README.md)，正式测评结果使用相邻的 `evaluations/`。

| 名称或文件 | 含义 |
|---|---|
| `smoke_*` | 初始两种状态表示的短训与回放验证 |
| `benchmark_*` | 准备阶段的性能与 batch 探测，部分在共享 GPU 上运行 |
| `exclusive_20261114_*` | 第一轮计划独占的探测，实际发生资源竞争；名称不构成独占证据 |
| `allocated_20261114_queue_1794624587_1_*` | 已确认独占且全部通过的八组探测 |
| `allocated_20261114_queue_1794624829_2_*` | 取消的重复探测，只保留本机 |
| `final_checks.json`、`upstream_pytest.xml`、`hardware/` | 环境与源码验证记录 |
| `performance_summary.*` | 准备阶段性能汇总 |

新短测写入新的任务名，不覆盖历史记录。旧 `reproduction/results/` 是本目录的符号链接，历史脚本和日志中的路径仍可解析。
