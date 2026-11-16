# 复现脚本索引

以下命令从仓库根目录执行，统一使用 `bash reproduction/run_env.sh` 进入固定环境。`src/supermimic/scripts/` 是上游原生 CLI，本目录是复现辅助工具。固定超参数与工程设置见 [配置说明](../docs/EXPERIMENT_CONFIG.md)，当前模型通过 [实验入口](../reproduction/experiments/README.md)访问。

| 目录 | 主要脚本 | 用途 |
|---|---|---|
| `data/` | `check_bones_access.py`、`download_*.py`、`convert_*.py`、`finalize_bones.py` | 数据下载、格式转换与验证；通常使用 `prepare_data.sh` 串联 |
| `setup/` | `fetch_wheels.py` | 原服务器安装传输诊断辅助工具；普通安装用 `bootstrap.sh` |
| `validation/` | `check_paths.py`、`preflight.py`、`eval_smoke.py`、`nccl_smoke.py` | 路径可移植性、有限步学习、模型加载/回放与 GPU 通信验证 |
| `training/` | `train_campaign.py` | 可恢复的单模型正式训练 |
| `training/` | `probe_all.py`、`queue_campaign.py`、`launch_training.py`、`summarize_benchmark.py` | **20261114 批次专用**的八卡调度与短测汇总；不是通用新实验启动入口 |
| `monitoring/` | `campaign_dashboard.py`、`capture_process_util.py`、`gpu_availability.py` | 进度、显存、GPU 使用率及资源检查 |
| `reporting/` | `write_training_result.py`、`plot_training_results.py` | 生成当前训练报告与图表 |
| `reporting/` | `summarize_preparation.py`、`write_preparation_report.py` | 重建历史准备阶段的统计和报告 |
| `evaluation/` | 当前仅有说明文档 | 下一阶段完整动作测评工具的位置，实施范围见根目录计划 |

## 常用命令

查看训练封装参数：

```bash
bash reproduction/run_env.sh python scripts/training/train_campaign.py --help
```

查看进度（只读，不启动训练）：

```bash
python3 scripts/monitoring/campaign_dashboard.py reproduction/campaigns/20261114_8xa100
```

基于已核验统计重新生成报告正文：

```bash
bash reproduction/run_env.sh python scripts/reporting/write_training_result.py --report-only
```

同时从原始日志重新绘图：

```bash
bash reproduction/run_env.sh python scripts/reporting/write_training_result.py --refresh-report
```

不加上述参数会执行完整权重与遥测核查，需要本机的大文件。单独绘图可以运行 `scripts/reporting/plot_training_results.py`。

旧 `reproduction/scripts/` 中的同名链接继续支持此前的命令。所有数据下载、短训和正式训练脚本均应按其实际用途运行；部分历史脚本没有 `--help` 模式，不将它们作为目录检查命令执行。
