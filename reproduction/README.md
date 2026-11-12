# 官方 SuperMimic 复现交接

这里保存 2026-11-12 完成的环境、数据准备、测试与性能探测。**正式训练没有启动**。核心训练源码保持上游实现，新增内容在 `reproduction/`。

## 已完成

- 固定 SuperMimic 基线 `abc124c1bd5976341034668af3d12c087f2f1f55` 与配套 mjlab `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd`。
- Conda Python 3.13.18、Torch 2.14.1/CUDA 13.0、MuJoCo/Warp 3.16、Warp 1.18，依赖检查通过。固定 mjviser 0.0.16，避开 0.0.17 引入 MuJoCo 4 包覆盖同名模块的冲突。
- 官方 217 项测试通过；8× A100-80GB 的 CUDA/BF16/NCCL 检查通过。
- GenCoord、MaxCoord 的原生短训、真实参数更新、CUDA Graph 捕获与重放、检查点保存加载、EGL 回放通过；共 8 个有严格循环上限的学习探测、7 个回放检查。
- LAFAN1 40 条、441,140 帧；BONES 原始 142,220 条、51,915,358 帧，全部转换、打包和校验。BONES 全量 GPU 加载与短时学习更新通过。

[准备报告](PREPARATION_REPORT.md) · [机器验证记录](results/final_checks.json) · [性能表](results/performance_summary.csv) · [下一步](NEXT_STEPS.md)

## 在另一台服务器恢复

需要 Linux x86_64、可运行 CUDA 13 的 NVIDIA 驱动、Miniconda，以及有 BONES-SEED 权限的 Hugging Face 账号。原服务器驱动为 610.43.02；脚本不修改驱动。仅恢复公开 LAFAN1 不需要 Hugging Face 登录。完整数据准备建议预留至少 80 GB 磁盘和充足内存（全库打包在原服务器使用约数十 GB RAM）。

```bash
git clone https://github.com/Notyourbing/supermimic.git
cd supermimic
export REPO="$PWD"
# 可选：指定环境和配套依赖目录；否则使用下面这两个默认位置。
export SUPERMIMIC_ENV_PREFIX="$HOME/miniconda3/envs/supermimic-official"
export SUPERMIMIC_MJLAB_DIR="$(dirname "$REPO")/mjlab_research"
bash reproduction/bootstrap.sh
bash reproduction/run_env.sh python -c 'import torch,mujoco; print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),mujoco.__version__)'
```

`bootstrap.sh` 克隆固定的配套 mjlab，安装精确依赖与两个 editable 包；不会启动训练。不要直接对旧 `uv.lock` 运行默认 `uv sync`，它与当前配套源码依赖不一致。Miniconda 未安装时先按 [官方说明](https://www.anaconda.com/docs/getting-started/miniconda/install) 安装。

### 恢复数据

```bash
bash reproduction/prepare_data.sh lafan
bash reproduction/run_env.sh hf auth login  # 仅在本机终端输入 token
bash reproduction/prepare_data.sh bones
```

LAFAN1 按上游 Git blob ID 验证原始 CSV。BONES 固定数据提交 `2f59b2077b9da34dd4e43618e705c7cb962c9a66`，归档 SHA-256 见 `setup/bones_access.json`。此文件的历史 `access_verified` 不代表新服务器已经登录；准备脚本会重新验证当前凭据。token 和签名下载 URL 不写入仓库。

BONES 原始格式是厘米、角度、外旋 xyz 欧拉角，120 Hz；适配后交给官方加载器重采样至 50 Hz。流水线直接读取 tar，避免展开大量中间 CSV。下载可断点续传；原服务器数据 CDN 直连最快，若新机需要代理，设置 `BONES_USE_PROXY=1`。CPU 转换可用 `BONES_WORKERS=8` 降低并发。

本仓库包含下载来源、摘要、转换与打包验证记录，**不分发数据本体、权重、Conda 环境、wheel 或认证信息**。若能从旧服务器复制数据，保留相同 `reproduction/data/` 相对布局并重新核对摘要即可。

### 在新服务器重新验证

`results/` 和本目录报告是原服务器的历史证据，不是新服务器已通过测试的证明。请使用新名字保存自己的运行，避免覆盖历史记录。

```bash
CUDA_VISIBLE_DEVICES='' bash reproduction/run_env.sh pytest -q tests
# 以下通信测试仅在 8 卡可用时运行；卡数可调整。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash reproduction/run_env.sh \
  torchrun --standalone --nproc_per_node=8 reproduction/scripts/nccl_smoke.py
CUDA_VISIBLE_DEVICES=0 bash reproduction/run_env.sh python reproduction/scripts/preflight.py \
  --name server2_gencoord_smoke --representation gencoord --iterations 48
CUDA_VISIBLE_DEVICES=1 bash reproduction/run_env.sh python reproduction/scripts/preflight.py \
  --name server2_maxcoord_smoke --representation maxcoord --iterations 48
CUDA_VISIBLE_DEVICES=0 bash reproduction/run_env.sh python reproduction/scripts/eval_smoke.py \
  --run-dir "$REPO/reproduction/results/server2_gencoord_smoke" --num-envs 64 --steps 200 --render
```

短训每项仅允许 16–160 个收集循环，有限损失、模型更新与两段 CUDA Graph 都通过才生成成功摘要。原服务器首次冷编译约 11–14 分钟，缓存后的完整短测约 1–2 分钟；新硬件需重新测量。原服务器缓存位于 `/tmp/torchinductor_zhuobing` 和用户 Warp cache，没有上传编译产物。

正式训练的可审阅命令在 [TRAINING_COMMANDS.md](TRAINING_COMMANDS.md)，本轮没有执行。

## 性能与协议边界

原服务器 8 卡都有其他用户任务，已全部保留。全量原始 BONES 默认配置约 0.576 s/收集循环、33.6 GiB 本进程峰值；按当时共享负载，32 万循环约 51 h，代码默认 100 万约 160 h。8 卡全量测评先预留 45–90 min/检查点，属于排程估计。独占利用率与速度必须在新服务器复测。

当前 runner 没有单模型 DDP。8 卡可跑独立实验或分片评测；NCCL 测试通过不等于实现了同步训练。优先保留官方 WM batch 2048 / policy batch 4096、窗口 8、想象长度 32；翻倍 batch 会把每收集循环学习频率从 0.4 降为 0.2。

**BONES 原始全集不是论文的 129,514 条过滤集。** 当前固定提交未提供精确筛选清单。当前代码的 Muon + fused Adam、固定学习率、默认 100 万收集循环，也与论文 Adam/余弦调度/32 万的表格有差别。正式论文指标对照前先明确协议。

## 文件用途

- `setup/resolved_constraints.txt`：安装约束；`pip-freeze.txt` 和 `conda-explicit.txt` 是原机器审计快照，不作为跨路径安装命令。
- `scripts/preflight.py`、`eval_smoke.py`、`capture_process_util.py`：短训、回放和进程级利用率观测。
- `download_lafan.py`、`convert_lafan.py`、`download_bones_ranges.py`、`convert_bones_stream.py`、`finalize_bones.py`：数据流水线。
- `summarize_preparation.py`、`write_preparation_report.py`：重建原服务器固定命名探测的报告；新机先看自己的 `summary.json`，不要混算两台机器数据。
- `fetch_wheels.py`：原机器的传输诊断辅助工具，需要 `logs/install.log` 的 uv verbose 解析记录；常规恢复使用 `bootstrap.sh`。离线恢复还需自行拷贝 `setup/wheels/`，不能仅靠 Git 完成。
