# 官方 SuperMimic：8×A100 本机复现（2026-11-14）

**当前状态：正式训练尚未启动，自动队列等待 8 卡释放。** 初始显存快照接近空闲，但随后用户 `yunzhe` 的八个进程各分配约 31 GB 显存，导致本次首轮测试存在竞争。首轮 5 组集成通过、2 组大配置 OOM、1 组取消，名称中的 `exclusive` 是初始计划命名，不能视为独占证据。此前控制台初报的 90%–95% 是全卡 NVML 使用率，不能归因于本项目。`benchmark.json` 已将 `exclusive_confirmed` 标为 false。

`queue_campaign.py` 每 30 秒检查资源，连续三次空闲后，用新运行名重跑全部短测，检测中途竞争；八组全通过并完成恢复验证后，才运行 `launch_training.py`。不会终止其他用户任务。独占短测失败会停队列并保留日志，等待调查，不会带病进入正式训练。可在 tmux 的 `queue` 窗口观察队列，Ctrl-C 取消排队。

本批运行针对当前官方开源实现，固定训练种子 42、proprio 观测和 10 帧历史。源代码基线是 `b1da9b1`（上游算法来源 `abc124c`），mjlab 固定 `dd51a9f`。原生 learner、优化器、物理参数、窗口和 reward 保持当前实现。训练封装新增本地观测与恢复功能，没有把独立任务称为多卡同步训练。

## 计划的正式实验分配

| GPU | 数据 | 表示 | 环境数 | WM batch | Policy batch | 采样循环预算 |
|---|---|---|---:|---:|---:|---:|
| 0 | LAFAN1 | GenCoord | 4096 | 2048 | 4096 | 320000 |
| 1 | LAFAN1 | MaxCoord | 4096 | 2048 | 4096 | 320000 |
| 2 | BONES unfiltered | GenCoord | 4096 | 2048 | 4096 | 320000 |
| 3 | BONES unfiltered | MaxCoord | 4096 | 2048 | 4096 | 320000 |
| 4 | LAFAN1 | GenCoord | 8192 | 4096 | 8192 | 160000 |
| 5 | LAFAN1 | MaxCoord | 8192 | 4096 | 8192 | 160000 |
| 6 | BONES unfiltered | GenCoord | 8192 | 4096 | 8192 | 160000 |
| 7 | BONES unfiltered | MaxCoord | 8192 | 4096 | 8192 | 160000 |

这是四组默认基线与四组采样规模对照，每组独占一张卡。每个正式任务预算均为 1,310,720,000 条真实环境 transition。WM minibatch 数 10、horizon 8、policy horizon 32、replay ratio 16，两个规模的 learner 更新频率均为每采样循环 0.4 次。放大组的总优化器更新次数约减半、每次样本数翻倍，因此它是训练规模消融，效果等价性必须靠最终评测判定。短测只筛选吞吐、显存、数值与 CUDA Graph 正常性，不能证明收敛质量。

这张表在资源释放、短测通过之后执行。恢复验证已经通过：保存后从 40 循环恢复到 80，更新数 13→29、两模型继续更新、两套图重捕获成功、TensorBoard 记录到 80。完整恢复文件约 4.1 GB（默认 LAFAN 配置）。

GPU 使用率是 NVML 的时间占用比例，不等于 Tensor Core 算力利用率。用 transition/s、更新数、显存和最终效果共同判断效率。预热、编译和 checkpoint IO 单列，不计入稳定吞吐估算。

## 数据与协议

实际数据位于 `/home/zhuobing/projects/supermimic/official_reproduction/data`。LAFAN1 共 40 条、441140 帧、2.45 小时；BONES 当前完整公开授权归档共 142220 条、51915358 帧、287.63 小时，已校验转换后的 packed 数据以及归档 SHA256。BONES 的 `packed_unfiltered` **不等于**论文筛选后的 129514 条动作，当前不能声称复现论文数据协议。当前源码的 Muon/fused Adam、恒定学习率也与论文 Adam/余弦配置存在差异。本批首先复现官方发布实现，320000 是显式采样循环预算，不宣称与论文迭代数完全等价。

## 查看运行

```bash
cd /home/zhuobing/projects/supermimic/official_ws/supermimic
tmux attach -t supermimic
```

默认 `progress` 窗口显示每卡任务、准确的采样循环数、learner 更新数、进度、移动吞吐与 ETA；`gpu` 看显卡；`train0` 至 `train7` 为正式训练日志；`tensorboard` 为服务窗口。`Ctrl-b w` 选择窗口，`Ctrl-b d` 脱离并保留任务。

TensorBoard 本机地址 `http://127.0.0.1:6006`，需要远程查看时在自己电脑建立 SSH 转发：

```bash
ssh -L 6006:127.0.0.1:6006 USER@SERVER
```

然后浏览器打开相同地址。事件包括原生 loss/reward/episode 指标及 GPU 使用率、显存和功耗。硬件采样每 2 秒保存，训练标量每 50 个采样循环保存。

## 持久化与恢复

每个任务独立目录保存 `config.json`（所有超参数、数据路径、提交号）、`train.log`、`metrics.jsonl`、`status.json`、`tensorboard/`。每 5000 循环保存带步数的 WM/actor/critic，`latest` 指向最近完整权重。每 10000 循环原子替换 `resume.pt`，包含模型、normalizer、全部优化器、当前 replay buffer、采样估计、计数和 RNG。整个训练数据集保留，replay 保存当前环形窗口；不会保存每一条历史 transition 的无限轨迹。

恢复时使用 `training_jobs.json` 中原命令，追加 `--resume <任务目录>/resume.pt`。配置和数据路径必须匹配，可调大 `--iterations`。恢复会重置物理环境 episode，并在 replay 边界添加 truncated，避免跨重启拼接轨迹；这属于统计连续训练，不是物理状态逐 bit 恢复。SIGTERM/SIGINT 请求会在当前更新完成后保存恢复文件并退出；SIGKILL/机器宕机只能使用上次完成的恢复文件。不要通过重新运行八卡启动脚本重复创建任务。

所有大权重、replay、TensorBoard 事件、日志都保留本机，不推入 Git。源代码和小型实验清单可以迁移；权重和数据需要另外复制。

## 时间估计与结果

`benchmark.json` 与 `BENCHMARK.md` 保存本次独占卡短测统计。初步工时按排除前 5 次 learner 更新与末尾清理的稳定循环时间计算，增加 20% 余量；正式阶段 `status.json` 持续滚动更新 ETA。预计 GPU 小时为每个任务卡上时长之和，并行墙钟时间取最长任务，二者不能混用。编译新 shape 可能额外需要数分钟。

首轮由于竞争不具备完整八组独占工时预测。当前仅作资源预算：单任务约 20–55 小时，八组量级约 160–400 GPU 小时；不作为已测准的完工承诺。待自动队列独占重测，`BENCHMARK.md` 将更新为新的八组统计和总 GPU 小时。

短测原始证据在 `reproduction/results/exclusive_20261114_*`。旧准备报告中的 50 多小时预测是在其他用户任务共用 GPU 下测得，不适用于这次独占运行。最终效果尚未测得；所有跑完后的模型仍需固定评测协议与 reference 对照视频验证。
