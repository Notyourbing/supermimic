# 项目结构与路径约定

当前项目为官方 SuperMimic 的 G1 复现仓库。根目录保留用户指定的 `RESULT.md` 与 `PLAN_12_16_0.md` 作为结果和计划入口，其余内容按用途归档。

```text
supermimic/
├── README.md                       # 项目与常用入口
├── RESULT.md                       # 当前训练结果报告
├── PLAN_12_16_0.md                  # BONES 筛选、单卡重训与测评计划
├── src/supermimic/                 # 官方算法、模型、环境和原生 CLI
├── tests/                          # 算法与组件测试
├── scripts/                        # 本 fork 的复现工具
│   ├── data/                       # 数据权限检查、下载、转换、打包
│   ├── setup/                      # 环境安装的辅助工具
│   ├── validation/                 # 有限步 smoke test、加载与通信检查
│   ├── training/                   # 正式训练、八卡批次调度和短测汇总
│   ├── monitoring/                 # 进度面板与 GPU 观测
│   ├── reporting/                  # 统计、绘图和 Markdown 报告生成
│   └── evaluation/                 # 完整动作测评工具的预留位置
├── docs/
│   ├── EXPERIMENT_CONFIG.md        # 固定超参数、工程设置与历史标识映射
│   ├── guides/                     # 当前复现与迁移指南
│   ├── archive/                    # 已完成阶段的历史文档
│   ├── figures/20261114_8xa100/     # 当前报告图表：PNG / PDF / 数据索引
│   └── templates/                  # 报告生成模板
├── reproduction/
│   ├── run_env.sh                  # 在固定 Conda 环境中执行命令
│   ├── paths.sh                    # 运行环境、依赖和数据路径的共同入口
│   ├── local_paths.example.sh       # 可选本机设置模板
│   ├── bootstrap.sh                # 环境安装
│   ├── prepare_data.sh             # 数据流水线入口
│   ├── setup/                      # 依赖约束、版本和安装审计
│   ├── data/                       # 数据准备默认目录；Git 中主要为清单
│   ├── smoke_tests/                # 短测、性能探测及其历史证据
│   ├── experiments/               # 按数据集命名的当前模型入口与统计
│   ├── campaigns/                  # 正式训练，每个批次一个目录
│   └── evaluations/                # 正式动作测评的结果目录
└── pyproject.toml / uv.lock         # 上游包配置；本机恢复使用固定依赖流程
```

## 查找一组正式训练产物

当前模型从 `reproduction/experiments/lafan1/` 和 `reproduction/experiments/bones_unfiltered/` 访问，筛选后训练使用 `reproduction/experiments/bones_filtered/`。固定参数不再编码进日常实验名称；历史映射集中在 [EXPERIMENT_CONFIG.md](EXPERIMENT_CONFIG.md)。

`reproduction/campaigns/20261114_8xa100/` 的八个模型目录保持原路径。每个目录包含配置、训练日志、标量、TensorBoard 事件和模型快照：

```text
<模型名>/
├── config.json / env_config.json / dataset_manifest.json
├── train.log / metrics.jsonl / status.json
├── tensorboard/
├── checkpoints/<采集循环编号>/
│   ├── world_model.pt
│   ├── actor.pt
│   └── critic.pt
├── latest -> checkpoints/<最终编号>/
└── resume.pt
```

正式训练目录约 110 GB，包含完整恢复文件与历史快照。模型权重和数值训练参数保持原样；可执行配置、清单与命令中的路径已统一规范化为仓库相对路径。TensorBoard 和 tmux 继续访问原训练路径。

## 旧路径如何兼容

| 旧入口 | 当前规范位置 | 兼容方式 |
|---|---|---|
| `reproduction/scripts/*.py` | `scripts/<用途>/*.py` | 保留相对符号链接，历史命令继续可用 |
| 批次内的四个 Python 调度脚本 | `scripts/training/` | 批次内保留相对符号链接 |
| `reproduction/results/` | `reproduction/smoke_tests/` | 相对目录链接，不复制短测数据 |
| `reproduction/reports/` | `docs/templates/` | 相对目录链接 |
| 批次内的 `results_figures/` | `docs/figures/20261114_8xa100/` | 相对目录链接 |
| `reproduction/README.md` 及旧准备文档入口 | `docs/guides/`、`docs/archive/` | 保留简短导航页 |

新命令与新文档使用规范位置；旧入口用于兼容。符号链接目标均在仓库内，克隆到其他 Linux 路径后仍成立。复制时使用保留符号链接的工具，例如 `rsync -a`，避免展开兼容链接后复制出两套产物。

## 当前服务器的外部目录

| 本机位置（相对于工作区根目录） | 内容 |
|---|---|
| `official_ws/supermimic/` | 本仓库 |
| `official_ws/mjlab_research/` | 固定版本的配套依赖，与本仓库相邻 |
| `official_reproduction/data/` | 本批训练实际使用的完整 LAFAN1 / BONES 数据 |
| `SuperMimic.pdf` | 原论文 |
| `Dex-Supermimic/`、`Dex-Supermimic-reproduction/`、旧 `reproduction/` | 早期灵巧手实验及文档，属于另一项复现 |

仓库位置由入口脚本自动推导，不依赖原服务器的用户名或目录。数据与依赖通过 `SUPERMIMIC_DATA_DIR`、`SUPERMIMIC_MJLAB_DIR` 设置；本机覆盖保存在不进入 Git 的 `reproduction/local_paths.sh`。新服务器通过 [复现指南](guides/reproduction.md) 重新建立 editable 安装并设置自己的存储位置；仓库内验证清单不包含数据本体。

## 后续文件放置规则

新工具放入 `scripts/` 下对应用途目录；完整动作评测工具放入 `scripts/evaluation/`，输出放入 `reproduction/evaluations/<计划或批次>/<协议版本>/`。实验配置与小型统计可以进入 Git，权重、恢复文件、数据本体、逐帧轨迹、视频和运行日志留在本机。新训练采用数据集与实验目的命名，不在目录名中拼接固定超参数。新的报告图表放入 `docs/figures/<批次>/`，不混入算法源码。
