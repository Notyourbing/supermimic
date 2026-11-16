# 官方 SuperMimic：复现与服务器迁移指南

当前模型与后续实验使用统一配置，详见[固定配置说明](../EXPERIMENT_CONFIG.md)。结果见 [RESULT.md](../../RESULT.md)，下一阶段按优先级完成 [BONES 筛选、单卡重训与完整动作测评](../../PLAN_12_16_0.md)。筛选后训练从头初始化；旧 BONES 与 LAFAN1 模型用于推理测评。本指南说明环境与数据如何恢复，历史准备证据见 [准备报告](../archive/PREPARATION_REPORT.md)。

## 1. 安装固定环境

本仓库对应 Python 3.13、Torch 2.14.1/CUDA 13.0、MuJoCo/Warp 3.16 和固定版本的配套 mjlab。依赖约束见 [resolved_constraints.txt](../../reproduction/setup/resolved_constraints.txt)。服务器需要兼容的 NVIDIA 驱动；安装脚本不修改驱动。

```bash
git clone https://github.com/Notyourbing/supermimic.git
cd supermimic
export SUPERMIMIC_ENV_PREFIX="$HOME/miniconda3/envs/supermimic-official"
export SUPERMIMIC_MJLAB_DIR="../mjlab_research"
bash reproduction/bootstrap.sh
bash reproduction/run_env.sh python -c 'import torch,mujoco; print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),mujoco.__version__)'
```

先在本机安装 Miniconda，再运行上述流程。`bootstrap.sh` 安装环境与两个 editable 包，不启动训练。配套 mjlab 固定到 `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd`；`mjviser` 固定为 0.0.16。不要直接用默认 `uv sync` 覆盖环境，上游旧锁文件与当前配套依赖不一致。

运行任何复现脚本均使用 `bash reproduction/run_env.sh ...`。该入口自动进入当前仓库根目录，保留线程设置、EGL 和离线 W&B 配置，并优先导入当前检出目录的源码。离线安装还需要自行复制 wheel 与依赖源码，见 [setup/](../../reproduction/setup/README.md)。

## 2. 恢复数据

```bash
bash reproduction/prepare_data.sh lafan
bash reproduction/run_env.sh hf auth login
bash reproduction/prepare_data.sh bones
```

Hugging Face 登录在本机终端进行；BONES 需要该账户已有数据访问权限。默认输出位于 `reproduction/data/lafan1/` 和 `reproduction/data/boneseed/`。完整准备建议预留至少 80 GB 磁盘。

LAFAN1 使用 G1 重定向的 40 条动作；BONES 使用原始 142,220 条动作。数据流水线统一原始单位与旋转约定，通过官方加载器重采样到 50 Hz，并保存来源、摘要和转换校验。BONES 原始集合不同于论文筛选后的 129,514 条集合。

### 路径约定与外部数据盘

所有保存到配置和清单的相对路径均以**当前仓库根目录**为基准。数据统一记录为 `reproduction/data/...`；运行时可用 `SUPERMIMIC_DATA_DIR` 将这个逻辑前缀映射到另一个数据盘。下载、转换、打包、训练和回放共享同一设置，无需逐个修改历史配置。

```bash
# 可选：数据实际保存在与仓库相邻的 motion_data 目录。
export SUPERMIMIC_DATA_DIR="../motion_data"
bash reproduction/run_env.sh python scripts/validation/check_paths.py
# 数据复制或准备完成后，再检查当前模型的数据能否找到。
bash reproduction/run_env.sh python scripts/validation/check_paths.py --check-data
```

也可复制 `reproduction/local_paths.example.sh` 为 `reproduction/local_paths.sh`，填写本机的数据、Conda 环境与 mjlab 位置。该文件不进入 Git；三个变量中的相对路径均以仓库为基准，环境变量可覆盖示例中的默认值。Conda 默认从当前用户的 `$HOME/miniconda3/envs/supermimic-official` 查找，mjlab 默认位于 `../mjlab_research`。

当前服务器通过这份本地设置访问已有数据，数据本体未搬动。新服务器仍需复制或重新下载数据、模型及日志，Git 克隆只包含代码与小型配置。迁移时应重新填写本机设置，保留仓库内部的相对符号链接。显式传入绝对路径仍受支持，但不会写死到源码中；可移植的配置优先使用上述逻辑数据前缀。

## 3. 在新服务器验证

仓库中的历史通过记录不代表新机器已经验证。先运行源码测试，再在可用 GPU 上进行有限步短测：

```bash
CUDA_VISIBLE_DEVICES='' bash reproduction/run_env.sh pytest -q tests
CUDA_VISIBLE_DEVICES=0 bash reproduction/run_env.sh python scripts/validation/preflight.py \
  --name server2_smoke --iterations 48 \
  --motion-file reproduction/data/lafan1/npz/run1_subject5.npz
CUDA_VISIBLE_DEVICES=0 bash reproduction/run_env.sh python scripts/validation/eval_smoke.py \
  --run-dir reproduction/smoke_tests/server2_smoke --num-envs 64 --steps 200 --render
```

短测输出位于 `reproduction/smoke_tests/<名称>/`。为新服务器选择新的名称，避免覆盖历史结果。需要通信验证且八卡可用时，再运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash reproduction/run_env.sh \
  torchrun --standalone --nproc_per_node=8 scripts/validation/nccl_smoke.py
```

`eval_smoke.py` 仅验证加载与有限步回放，不给出论文动作成功率。当前正式训练封装以单模型单卡方式运行，通信测试通过不等于实现了单模型 DDP。

## 4. 查看现有训练与恢复模型

训练批次位于 [20261114_8xa100](../../reproduction/campaigns/20261114_8xa100/README.md)。当前模型通过 [LAFAN1](../../reproduction/experiments/lafan1/)和[未筛选 BONES](../../reproduction/experiments/bones_unfiltered/)入口访问，两模型已经按预算完成，合计 30.03 GPU 小时；这些是原服务器的实际成本，不能直接当作其他硬件的预算。

```bash
python3 scripts/monitoring/campaign_dashboard.py reproduction/campaigns/20261114_8xa100
tmux attach -t supermimic
```

TensorBoard 为 `http://127.0.0.1:6006`；需重新启动服务时使用：

```bash
bash reproduction/run_env.sh tensorboard \
  --logdir reproduction/campaigns/20261114_8xa100 --host 127.0.0.1 --port 6006
```

每个模型的 `latest/` 指向最终权重，`resume.pt` 保存训练恢复状态。继续训练使用批次中 `training_jobs.json` 对应命令并明确指定 `--resume`；恢复必须匹配原任务配置和同一数据集；路径会先规范化，允许迁移服务器目录，但仍拒绝把未筛选数据恢复到筛选数据。旧二进制恢复文件无需重新保存。恢复会重置物理 episode，并非逐比特恢复仿真。不要再次启动已完成批次的八卡队列。

## 5. 修改和重建报告

正文模板位于 [result_template.md](../templates/result_template.md)，图表位于 [figures/20261114_8xa100/](../figures/20261114_8xa100/)。

```bash
bash reproduction/run_env.sh python scripts/reporting/write_training_result.py --report-only
bash reproduction/run_env.sh python scripts/reporting/write_training_result.py --refresh-report
```

第一条只更新 Markdown 和派生比较，第二条同时读取本机原始标量重新绘图。不加参数会重新核查大权重与遥测文件。更多脚本见 [脚本索引](../../scripts/README.md)，目录迁移与旧路径兼容方式见 [项目结构](../PROJECT_STRUCTURE.md)。
