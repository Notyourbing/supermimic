> 本文记录 2026-11-12 原服务器的实测结果。迁移步骤见 [README](README.md)，本机路径只表示历史证据；新服务器必须重新验证。

# 官方 SuperMimic：环境、数据与训练前验证报告

本次准备对象是 [HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic) 的 Unitree G1 全身动作跟踪。旧灵巧手复现保留在原目录。本次没有启动正式训练，没有设置正式训练的自动启动队列；所有学习更新仅发生在有严格循环上限的准备探测中。

## 1. 安装与可复现版本

| 项目 | 本机配置 |
|---|---|
| Conda 环境 | `/home/zhuobing/miniconda3/envs/supermimic-official` |
| Python | 3.13.18 |
| PyTorch / CUDA | 2.14.1+cu130 / 13.0 |
| MuJoCo / MuJoCo Warp | 3.16.0 / 3.16.0 |
| Warp / mjlab / rsl-rl-lib | 1.18.0 / 1.6.0 / 5.5.2 |
| SuperMimic 提交 | `abc124c1bd5976341034668af3d12c087f2f1f55` |
| 配套 mjlab_research 提交 | `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd` |
| 硬件 | 8× A100-SXM4-80GB，NVLink；128 逻辑 CPU |

源码为当前仓库，配套 `mjlab_research` 在当前仓库的相邻目录。两个工作树保持干净；所有本机辅助程序位于本目录。安装清单和源码记录见 [setup/source_manifest.json](setup/source_manifest.json)、[setup/pip-freeze.txt](setup/pip-freeze.txt)、[setup/conda-explicit.txt](setup/conda-explicit.txt)。`uv pip check` 已通过。

SuperMimic 自带旧锁文件与当前配套 mjlab 的依赖不一致，因此按固定的当前源码解析后另外保存了精确版本。实际遇到的 `mjviser 0.0.17` / `mujoco-core 4.0.1` 命名空间冲突已修复，固定 `mjviser==0.0.16` 仍满足官方要求。已验证的 wheel 保存在 `setup/wheels`，可用 [setup/restore_environment.sh](setup/restore_environment.sh) 离线恢复。

无需重复安装，运行现有环境：

```bash
bash /home/zhuobing/projects/supermimic/official_reproduction/run_env.sh python --version
```

日志使用离线 W&B，本地检查点、ONNX 和评测脚本不需要 W&B 登录。上游云端 artifact 清理因没有 API key 会输出警告，本地保存与重新加载已独立验证。

## 2. 数据准备

| 数据 | 状态 | 数量 / 时长 | 训练加载路径 |
|---|---|---|---|
| LAFAN1 G1 | 原始 Git blob 校验，官方转换与打包完成 | 40 条，441,140 帧，约 2.451 h | `data/lafan1/packed` |
| BONES-SEED G1 原始全集 | 官方归档 SHA-256 校验，转换及打包校验完成 | 142,220 条，51,915,358 帧，287.629 h | `data/boneseed/packed_unfiltered` |

BONES-SEED 归档 23,499,973,647 字节，固定数据提交 `2f59b2077b9da34dd4e43618e705c7cb962c9a66`。归档摘要为 `52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a`。142,220 条转换错误为 0；打包后逐字段检查形状、float32 类型、有限值、四元数范数和运动边界，并记录了各 `.npy` 文件哈希。验证文件见 [data/boneseed/packed_unfiltered/verified.json](data/boneseed/packed_unfiltered/verified.json)。

LAFAN1 用官方转换器从 30 Hz 重采样到 50 Hz。BONES-SEED 的原始 CSV 与 LAFAN1 格式不同，先按 [NVIDIA 的数据说明](https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/getting_started/seed_g1_csv_preparation.rst) 将厘米换为米、角度换为弧度、外旋 xyz 欧拉角换为四元数，再使用官方 `MotionLoader` 从 120 Hz 重采样到 50 Hz 并计算速度。29 个关节顺序由官方加载器核对。没有把带表头的原始 BONES CSV 当成 LAFAN1 CSV 直接读取。

全库通过官方加载器的 GPU 回放检查。打包格式仅保存 root pose/twist 和关节位置/速度，身体状态由官方 FK 计算；参考库常驻 GPU，避免训练时 CPU DataLoader 搬运。

**原始全集不是论文的过滤基准。** 论文使用 129,514 条、约 264.56 h 的过滤集合；本机准备的 `packed_unfiltered` 明确保留原始 142,220 条。当前发布仓库没有精确的过滤清单或对应处理入口，因此不能声称已经取得论文那份完全相同的数据集合，也不能把两者的失败数直接比较。

## 3. Smoke test 证据

- 官方测试 217 项全部通过：[results/upstream_pytest.xml](results/upstream_pytest.xml)。
- 8 张卡的 CUDA、BF16 矩阵计算和 NCCL all-reduce 全部通过：`results/hardware/gpu0.json` 至 `gpu7.json`。
- `SuperMimic-GenCoord-G1`、`SuperMimic-MaxCoord-G1` 各运行 48 个原生收集循环，检查世界模型与策略参数真实变化、损失有限，以及两段 CUDA Graph 捕获和多次重放。
- 两种表示均保存并重新加载世界模型和 actor；64 环境、200 步的本地回放及 EGL 无界面渲染均通过。
- 默认训练规模及 batch 翻倍方案分别做了限定循环的完整学习更新；另在同一张 GPU 上复测，检查缓存复用及性能。
- 全量原始 BONES 的 GPU 加载 / 回放检查通过；其短时训练结果见表和原始 `summary.json`。

所有对比的训练种子固定为 42，回放检查固定为 2001。短测策略尚未收敛，回放中的跌倒计数不是正式复现的成功率。图片 `results/smoke_*/headless_smoke.png` 只是渲染链路检查。

仿真短测中确实出现了上游默认物理求解器迭代达到上限的提示。没有因此改动仿真协议；所有完成的检查均保持有限状态和损失。训练效果与物理精度仍须在正式协议下评估。

## 4. 性能与显存实测

**这些是共享 GPU 的测量。** 开始准备时 8 张卡上均有另一位用户的训练进程，每卡占约 30.7 GiB，已完整保留。下表剔除了首次编译 / 图捕获的预热区间与最后清理区间。显存为 NVML 峰值扣除已有任务的稳定基线，约值包含 Warp；它与 Torch allocator 峰值分别记录。

| 配置 | WM / policy batch | 每收集循环的学习更新频率 | 秒 / 收集循环 | 本进程显存峰值 GiB | 32 万收集循环外推 h |
|---|---:|---:|---:|---:|---:|
| GenCoord，官方默认 batch | 2048 / 4096 | 0.4 | 0.615 | 20.2 | 54.6 |
| MaxCoord，官方默认 batch | 2048 / 4096 | 0.4 | 0.596 | 22.3 | 53.0 |
| GenCoord，batch 翻倍 | 4096 / 8192 | 0.2 | 0.495 | 33.5 | 44.0 |
| GenCoord，默认 batch，缓存复测 | 2048 / 4096 | 0.4 | 0.601 | 19.9 | 53.4 |
| GenCoord，batch 翻倍，同 GPU 复测 | 4096 / 8192 | 0.2 | 0.463 | 33.9 | 41.1 |
| GenCoord，全量 BONES，默认 batch | 2048 / 4096 | 0.4 | 0.576 | 33.6 | 51.2 |

完整数据与计算方式见 [results/performance_summary.json](results/performance_summary.json)、[CSV](results/performance_summary.csv)。每次 WM 更新仍含 10 个 minibatch，窗口为 8；策略想象长度仍为 32，critic 为 5×4 次更新。使用官方 TF32、`torch.compile` 和 CUDA Graph，没有引入训练 BF16 或修改损失。

同一张卡两次短测中，默认 batch 与翻倍 batch 的秒 / 循环约为 0.601 和 0.463，观察到约 1.30× 的吞吐差异。共享负载会波动，而且翻倍 batch 改变了学习更新频率，因此该结果只用于筛选性能方案，不是等效果加速结论。

更新频率由当前原生代码决定：

```text
num_envs / (wm.num_batches × wm.batch_size × wm.horizon) × replay_ratio
```

4096 环境、WM batch 2048 时为 0.4；仅将 WM batch 翻倍变成 0.2。在同样 32 万个收集循环下，后者的优化器更新次数约减半。正式复现优先保留默认 batch；若采用更大 batch，应另做质量对照并同时记录收集循环和学习更新数。

全量原始 BONES 的 71 维 float32 参考库占 **13.73 GiB**。与 LAFAN1 相比额外增加约 13.6 GiB：默认方案预计本进程约 34–36 GiB；翻倍 batch 约 47 GiB。现有其他任务仍占约 30.7 GiB 时，后一方案距离 80 GiB 容量边界较近，正式大数据训练不宜据小数据显存直接套用。

独占本项目的 GPU-Util 尚未测得，不能把其他任务产生的全卡利用率算成本项目结果。 全量短测中，预热后有效的本进程 SM 利用率采样均值为 46.2%，最高 93%（52 个有效样本；缺失样本不补零）。这是共享资源下的进程级采样，不等价于独占 GPU-Util。

## 5. 训练与评测的时间预算

全量原始 BONES 默认配置：10,000 个收集循环约 **1.6 小时**；320,000 个收集循环约 **51.2 小时**；代码默认 1,000,000 个收集循环约 **159.9 小时**。

这些数字按本机当前共享负载线性外推，未承诺达到某一回报所需的收敛时间。LAFAN1 默认两种表示的 32 万收集循环约 53–55 h；翻倍 batch 的 32 万循环约 41–44 h，但更新次数不同。软件缓存命中后，一次 40 循环完整探测约 95 s；首次冷编译的对应 smoke test 总共约 11–14 min。正式开始前应保留缓存，给图捕获与初始化单列启动预算。

评测短测在 1024 环境上约 5,272 transitions/s。按总帧数除以吞吐并理想分摊到 8 卡，全量原始 BONES 的纯计算下限约 20.5 min / 检查点。最长动作 9,007 帧、动作长短不齐、进程启动和尾部工作会抬高墙钟时间，建议先预留 **45–90 min / 检查点 / 8 卡**；单卡保守预留 **3–6 h**。LAFAN1 的 40 条动作并行度较小，先预留 **10–60 min / 检查点**。这些是排程估计，尚未用收敛策略做完整库的正式评测。

## 6. 8 卡使用方案与复现边界

1. 每张卡固定一个独立训练进程，CPU 线程先设为 4；共享官方内核缓存，参考库从打包 NPY 顺序加载到 GPU。训练主循环没有 CPU DataLoader，增大 DataLoader workers 对此路径没有直接作用。
2. 若覆盖数据规模实验，可将 2 种状态表示 × 4 种数据规模分配到 8 张卡，统一种子和其余协议；具体数据子集应在论文过滤协议确定后生成清单。检查点评测按动作分片，按长度安排任务，减少尾部等待。
3. 默认 batch 保留为复现基线；WM 4096 / policy 8192 是已通过短测的速度候选。对全量数据需结合显存余量和效果再选。
4. 当前官方自定义 runner 没有 DDP / 梯度同步入口。NCCL 检查通过仅证明通信环境就绪，不能把 `torchrun` 启动的 8 份独立模型称为一个同步模型，也不能把单任务时间直接除以 8。单模型同步多卡还需要单独实现与验证 normalizer、采样分布和梯度同步。

当前公开实现与论文也有协议差异：隐藏线性层使用 Muon，其他参数用 fused Adam；学习率固定，默认预算 1,000,000 个收集循环。论文表格给出 Adam、余弦学习率和 320,000 次循环。当前默认观测是表示状态 / 历史 1，本次显式启用传感器 `proprio` / 历史 10。以上均写入配置，没有把当前公开代码与论文数字视作完全相同的实验。

正式训练仍未启动。后续可直接复用本环境、已验证的两条任务入口、数据格式和性能候选；全量论文指标需要先补齐严格的数据筛选协议，并在资源可用时重新标定独占性能。
