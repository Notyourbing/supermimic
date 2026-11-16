"""Audit completed local runs and regenerate the evidence-backed RESULT.md."""

import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = REPO / "reproduction/campaigns/20261114_8xa100"


def utc(timestamp):
    return datetime.fromtimestamp(timestamp, UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def fmt(value):
    return f"{value:.2f}" if value is not None else "未采得"


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from tensors(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from tensors(v)


def main():
    jobs = json.loads((CAMPAIGN / "training_jobs.json").read_text())
    result = []
    histories = {}
    checkpoints = []
    for job in jobs:
        name = job["name"]
        run = CAMPAIGN / name
        cfg = json.loads((run / "config.json").read_text())
        status = json.loads((run / "status.json").read_text())
        assert status["status"] == "completed"
        assert status["iteration"] == job["iterations"]
        assert status["wm_graph"] and status["policy_graph"]
        assert not (run / "error.json").exists()
        rows = [
            json.loads(line)
            for line in (run / "metrics.jsonl").read_text().splitlines()
        ]
        assert len(rows) == job["iterations"] // 50
        assert [r["iteration"] for r in rows] == list(
            range(50, job["iterations"] + 1, 50)
        )
        assert all(
            math.isfinite(v)
            for row in rows
            for v in row.values()
            if isinstance(v, (int, float))
        )
        assert rows[-1]["iteration"] == status["iteration"]
        tail = [r for r in rows if r["iteration"] > 0.9 * job["iterations"]]
        means = {
            k: statistics.mean(r[k] for r in tail)
            for k in tail[-1]
            if k not in ("iteration", "time", "updates")
        }
        last = rows[-1]
        latest = run / "latest"
        meta = json.loads((latest / "metadata.json").read_text())
        assert meta["completed_iterations"] == job["iterations"]
        checkpoint_files = {}
        for model in ("world_model", "actor", "critic"):
            path = latest / f"{model}.pt"
            state = torch.load(path, map_location="cpu", weights_only=True)
            ts = list(tensors(state))
            assert ts and all(torch.isfinite(t).all().item() for t in ts)
            checkpoint_files[model] = {
                "path": str(path.resolve().relative_to(REPO)),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
                "tensor_count": len(ts),
                "finite": True,
            }
        # Metadata-only mmap avoids reading the multi-GB replay tensors into RAM.
        resume_path = run / "resume.pt"
        resume = torch.load(
            resume_path, map_location="cpu", mmap=True, weights_only=False
        )
        assert (
            resume["step"] == job["iterations"]
            and resume["updates"] == status["updates"]
        )
        assert resume["replay_size"] == 1000
        assert set(resume["optimizers"]) == {"wm", "actor", "critic"}
        assert all(len(o) > 0 for o in resume["optimizers"].values())
        checkpoint_files["resume"] = {
            "path": str(resume_path.relative_to(REPO)),
            "bytes": resume_path.stat().st_size,
            "step": resume["step"],
            "updates": resume["updates"],
            "replay_size": resume["replay_size"],
            "optimizer_groups": list(resume["optimizers"]),
            "scope": "metadata checked using mmap; replay contents not rescanned",
        }
        del resume
        snapshots = sorted((run / "checkpoints").iterdir())
        assert len(snapshots) == job["iterations"] // 5000
        checkpoints.append(
            {"name": name, "snapshot_count": len(snapshots), "files": checkpoint_files}
        )
        row = {
            "name": name,
            "gpu": job["gpu"],
            "seed": cfg["seed"],
            "dataset": "LAFAN1" if name.startswith("lafan1") else "BONES unfiltered",
            "representation": cfg["runner_cfg"]["representation"],
            "scale": 1 if name.endswith("x1") else 2,
            "num_envs": job["num_envs"],
            "iterations": status["iteration"],
            "transitions": status["transitions"],
            "updates": status["updates"],
            "start": cfg["start_time"],
            "finish": status["time"],
            "hours": (status["time"] - cfg["start_time"]) / 3600,
            "training_commit": cfg["git_commit"],
            "tail_count": len(tail),
            "tail_means": means,
            "last_logged": last,
            "torch_reserved_final_gib": status["torch_reserved_gib"],
            "snapshot_count": len(snapshots),
            "metric_rows": len(rows),
        }
        result.append(row)
        histories[name] = rows

    # Restrict telemetry to each model's own training interval. Do not include
    # idle time or unrelated jobs after this model has completed.
    accum = {
        r["gpu"]: {
            "util": [],
            "sm": [],
            "memory": [],
            "foreign_samples": 0,
            "steady_samples": 0,
            "active_samples": 0,
        }
        for r in result
    }
    by_gpu = {r["gpu"]: r for r in result}
    with (CAMPAIGN / "hardware.jsonl").open() as fh:
        for line in fh:
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue  # An active monitor may be appending its last row.
            t = sample["time"]
            for gpu in sample["gpus"]:
                r = by_gpu[gpu["index"]]
                if not r["start"] <= t <= r["finish"]:
                    continue
                a = accum[r["gpu"]]
                if not gpu.get("campaign_process_count", 0):
                    continue
                a["active_samples"] += 1
                a["memory"].append(gpu["memory_used_gib"])
                if t < histories[r["name"]][1]["time"]:
                    continue
                a["steady_samples"] += 1
                if gpu.get("foreign_process_count", 0):
                    a["foreign_samples"] += 1
                else:
                    a["util"].append(gpu["utilization"])
                if gpu.get("campaign_sm_utilization") is not None:
                    a["sm"].append(gpu["campaign_sm_utilization"])
    for r in result:
        a = accum[r["gpu"]]
        r["hardware"] = {
            "steady_samples": a["steady_samples"],
            "active_samples": a["active_samples"],
            "foreign_samples": a["foreign_samples"],
            "unshared_device_util_mean": statistics.mean(a["util"])
            if a["util"]
            else None,
            "process_sm_mean": statistics.mean(a["sm"]) if a["sm"] else None,
            "nvml_sampled_peak_gib": max(a["memory"]) if a["memory"] else None,
        }
    report = {
        "generated_utc": utc(datetime.now(UTC).timestamp()),
        "all_eight_completed": True,
        "total_gpu_hours": sum(r["hours"] for r in result),
        "wall_hours": (
            max(r["finish"] for r in result) - min(r["start"] for r in result)
        )
        / 3600,
        "tail_fraction": 0.1,
        "formal_evaluation_available": False,
        "runs": result,
    }
    (CAMPAIGN / "training_results.json").write_text(json.dumps(report, indent=2) + "\n")
    (CAMPAIGN / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoints, indent=2) + "\n"
    )
    figures = CAMPAIGN / "results_figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.7), constrained_layout=True)
    colors = {"gencoord": "#0072B2", "maxcoord": "#D55E00"}
    for r in result:
        rows = histories[r["name"]]
        x = np.array([v["iteration"] * r["num_envs"] for v in rows]) / 1e9
        # 100 equal-transition bins; all configurations share the same x-axis
        # and smoothing width. No uncertainty band is inferred from one seed.
        bins = np.minimum((x / 1.31072 * 100).astype(int), 99)
        for col, (key, factor) in enumerate(
            [("Train/mean_reward", 1), ("Metrics/motion/error_body_pos", 100)]
        ):
            xs, ys = [], []
            for b in sorted(set(bins)):
                mask = bins == b
                xs.append(x[mask].mean())
                ys.append(np.array([v[key] for v in rows])[mask].mean() * factor)
            ax = axes[0 if r["dataset"] == "LAFAN1" else 1, col]
            ax.plot(
                xs,
                ys,
                color=colors[r["representation"]],
                linestyle="-" if r["scale"] == 1 else "--",
                linewidth=1.35,
                label=f"{'GenCoord' if r['representation'] == 'gencoord' else 'MaxCoord'} x{r['scale']}",
            )
    for i, label in enumerate(["LAFAN1", "BONES (unfiltered)"]):
        for col, metric in enumerate(
            ["Training logger return", "Aligned body position error (cm)"]
        ):
            ax = axes[i, col]
            ax.set_title(label)
            ax.set_xlabel("Collected transitions (billions)")
            ax.set_ylabel(metric)
            ax.grid(alpha=0.22, linewidth=0.6)
            ax.set_xlim(0, 1.31072)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(
        figures / "training_curves.pdf",
        metadata={"Title": "SuperMimic training diagnostics"},
    )
    fig.savefig(figures / "training_curves.png", dpi=200)
    plt.close(fig)
    write_markdown(report)
    print(
        json.dumps(
            {
                "all_eight_completed": True,
                "total_gpu_hours": report["total_gpu_hours"],
                "wall_hours": report["wall_hours"],
                "result": str(REPO / "RESULT.md"),
            },
            indent=2,
        )
    )


def write_markdown(report):
    runs = report["runs"]
    prefix = "reproduction/campaigns/20261114_8xa100"
    lines = [
        "# 官方 SuperMimic 训练结果",
        "",
        f"统计日期：{report['generated_utc']}。对应批次：`20261114_8xa100`。",
        "",
        "**八组训练均按预定预算完成，最终 WM、actor、critic 和恢复文件完整；日志中所有数值标量有限。** "
        f"八张 A100-80GB 合计占卡时长 **{report['total_gpu_hours']:.2f} GPU 小时**，并行墙钟 **{report['wall_hours']:.2f} 小时**。"
        "这是当前官方开源实现的训练结果；本批没有独立全库评测结果，不能将训练完成表述为复现了论文成功率。",
        "",
        "## 1. 数据和实验协议",
        "",
        "- 官方算法来源提交 `abc124c1`，本地训练封装提交 `ffd271d`，配套 mjlab `dd51a9f`。八组训练均固定种子 **42**。",
        "- LAFAN1：40 条动作、441,140 帧，约 2.45 小时。BONES：142,220 条动作、51,915,358 帧，约 287.63 小时，使用 `packed_unfiltered`。BONES 不等于论文筛选后的 129,514 条动作集合。",
        "- 机器人为 Unitree G1；输入为 proprio 与 10 帧历史；WM 为三层 2048 隐藏单元 MLP，策略使用多编码器与 32 维潜变量。物理步长 0.005 s、decimation 4，控制频率 50 Hz。",
        "- 保留当前源码的 Muon/fused Adam、固定学习率 2e-4；WM 窗口 8、每次 10 个 minibatch，policy 想象长度 32，replay ratio 16。当前协议与论文 Adam/余弦学习率不完全相同。",
        "- x1：4096 环境、WM batch 2048、policy batch 4096、320,000 采样循环。x2：8192 环境、WM batch 4096、policy batch 8192、160,000 循环。每组均采集 **1,310,720,000 条真实环境 transition**。",
        "- 两种规模每个采样循环的 learner 更新频率均约 0.4；实际更新次数分别为 **127,997 / 63,997**。x2 每次样本数增加而总更新数减少，因此属于规模对照，不能预先假定训练效果等价。",
        "- 八卡各训练一个独立模型；没有将八个进程作为单模型 DDP。训练因**达到预算**结束，没有使用验证集收敛判据自动终止。",
        "",
        "## 2. 完成情况和实际耗时",
        "",
        "时间按各任务 `config.json:start_time` 到 `status.json:time` 计算，包含初始化、编译与最后保存，不含排队和先前准备。GPU 小时为各任务占卡时长之和，不是 GPU kernel 活跃时长。",
        "",
        "| GPU | 任务 | 完成循环 | Learner 更新 | 耗时 / GPU-h | 完成时间（UTC） | 权重快照数 |",
        "|---:|---|---:|---:|---:|---|---:|",
    ]
    for r in runs:
        lines.append(
            f"| {r['gpu']} | `{r['name']}` | {r['iterations']:,} | {r['updates']:,} | {r['hours']:.2f} | {utc(r['finish'])} | {r['snapshot_count']} |"
        )
    lines += [
        "",
        "所有任务状态均为 `completed`，不存在 `error.json`；最终 `latest/metadata.json` 与训练预算一致，两套 CUDA Graph 标志均为 true。结束时间依据持久化状态，不使用控制台 ETA。",
        "",
        "## 3. 训练末段指标",
        "",
        "主表统计最后 **10% 采样预算**内各日志行的算术均值：x1 为最后 640 行，x2 为最后 320 行，每 50 循环一行。这是对训练过程日志的平滑汇总，不是动作平均、帧加权的独立测评，也不是置信区间。最终单行指标可在 [机器可读汇总]("
        + prefix
        + "/training_results.json) 的 `last_logged` 中核查。",
        "",
        "| 任务 | 训练回报 ↑ | Logger 长度 / step ↑ | 对齐身体位置误差 / cm ↓ | 身体姿态误差 / rad ↓ | 全局锚点位置误差 / cm ↓ | WM Loss/state ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        m = r["tail_means"]
        lines.append(
            f"| `{r['name']}` | {m['Train/mean_reward']:.3f} | {m['Train/mean_episode_length']:.2f} | {100 * m['Metrics/motion/error_body_pos']:.3f} | {m['Metrics/motion/error_body_rot']:.4f} | {100 * m['Metrics/motion/error_anchor_pos']:.3f} | {m['Loss/state']:.6f} |"
        )
    lines += [
        "",
        "**指标解释：**",
        "",
        "- 回报来自真实仿真训练日志 `Train/mean_reward`，不是想象轨迹的 `Policy/reward`。它受噪声策略、return-based RSI、自适应采样分布和日志累积规则影响。",
        "- 原生 logger 对最近 100 次 flush 的回报/长度求均值，再对最后 10% 日志求均值。其累积器在 `terminated` 或累计长度达到 500 时清空，`motion_end` 等 truncated 不单独清空。因此尤其在 BONES 短片段上，长度不能解释为每条完整动作的存活时长，回报也不能直接转成成功率。",
        "- 身体位置误差是参考姿态经锚点水平位置与 yaw 对齐后，各跟踪刚体位置距离的均值；本表将 m 换为 cm。全局锚点误差未作该对齐，两者不能混淆。姿态误差为四元数角距离的刚体均值，单位 rad。",
        "- `Loss/state` 是 replay 中多步预测误差的加权训练损失。GenCoord 与 MaxCoord 的预测通道和尺度不同，不能仅据二者 loss 的大小判断哪种表示更好。",
        "",
        "### 相同采样量下的规模比较",
        "",
        "| 数据 / 表示 | x2 吞吐倍数 | 耗时减少 | 回报变化 | 身体位置误差变化 | 身体姿态误差变化 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for a in runs[:4]:
        b = next(
            r
            for r in runs
            if r["dataset"] == a["dataset"]
            and r["representation"] == a["representation"]
            and r["scale"] == 2
        )
        ma, mb = a["tail_means"], b["tail_means"]
        lines.append(
            f"| {a['dataset']} / {a['representation']} | {a['hours'] / b['hours']:.3f}× | {(1 - b['hours'] / a['hours']) * 100:.1f}% | {mb['Train/mean_reward'] - ma['Train/mean_reward']:+.3f} | {100 * (mb['Metrics/motion/error_body_pos'] - ma['Metrics/motion/error_body_pos']):+.3f} cm | {mb['Metrics/motion/error_body_rot'] - ma['Metrics/motion/error_body_rot']:+.4f} rad |"
        )
    lines += [
        "",
        "x2 在四组上均提高相同采样量下的吞吐，缩短约 19%–21% 时间。LAFAN1 的回报和对齐身体位置误差接近：回报约 43.84–44.51、位置误差约 3.02–3.13 cm。BONES 对齐身体位置误差约 13.41–14.13 cm，明显高于 LAFAN1。BONES MaxCoord 的 x2 回报更高、位置误差更小，但姿态误差从约 0.1568 增至 0.2072 rad，因此不能概括为所有指标都改善。不同配置采样分布会随训练改变，这些差异仍需固定协议测评确认。",
        "",
        "## 4. 训练曲线",
        "",
        f"![Training curves]({prefix}/results_figures/training_curves.png)",
        "",
        f"[矢量 PDF]({prefix}/results_figures/training_curves.pdf)。横轴统一为真实采样 transition 数，每条曲线按相同采样量分为 100 个区间并画区间均值；实线为 x1，虚线为 x2，蓝色为 GenCoord，橙色为 MaxCoord。上排为 LAFAN1，下排为 BONES；左列回报越高越好，右列对齐身体位置误差越低越好。没有用单次运行制造置信区间。曲线用于观察训练趋势，不能替代独立测评或证明收敛。",
        "",
        "## 5. 全程 GPU 使用情况",
        "",
        "以下为每个模型自身运行窗口内的实测，不包含模型结束后其他用户的作业。均值从第 100 个循环日志之后统计，剔除初始编译；整卡均值仅采用无外部计算进程的采样，进程 SM 均值直接来自 NVML 进程采样。显存为运行窗口内约每 2 秒采样所得峰值，不保证捕获更短的瞬时峰值。",
        "",
        "| GPU | 整卡 Util 均值 | 本进程 SM 均值 | 显存采样峰值 / GiB | 稳定期样本数 | 存在外部计算进程的样本数 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        h = r["hardware"]
        lines.append(
            f"| {r['gpu']} | {fmt(h['unshared_device_util_mean'])}% | {fmt(h['process_sm_mean'])}% | {fmt(h['nvml_sampled_peak_gib'])} | {h['steady_samples']:,} | {h['foreign_samples']:,} |"
        )
    lines += [
        "",
        "GPU-Util/SM 利用率表示采样期间的活跃程度，不等于理论 FLOPS 利用率。初始共享负载的短测未混入上述结果。",
        "",
        "## 6. 模型、日志和可复核证据",
        "",
        f"本机任务根目录：`{CAMPAIGN}`。每个任务名对应一个子目录。",
        "",
        "- 默认组最终模型：`<任务>/checkpoints/00320000/`；放大组：`<任务>/checkpoints/00160000/`。`latest` 已指向对应最终目录，含 `world_model.pt`、`actor.pt`、`critic.pt`。",
        "- 每 5,000 循环保留一次权重：默认组每组 64 份、放大组每组 32 份。`resume.pt` 含最终模型、normalizer、优化器、replay、采样状态及 RNG；恢复会重置物理 episode，不是 bitwise 仿真恢复。",
        "- 本次核查对 24 个最终权重文件逐个完成 CPU 加载、张量有限性检查并计算 SHA256；对八个恢复文件以 mmap 核对最终步数、更新数、replay 长度及优化器分组。未重新扫描多 GB replay 的全部内容。",
        f"- [权重清单与 SHA256]({prefix}/checkpoint_manifest.json) · [训练统计与最终单行值]({prefix}/training_results.json) · [运行配置与启动命令]({prefix}/training_jobs.json)。",
        "- 原始 `train.log`、`metrics.jsonl`、`status.json`、`tensorboard/`、`hardware.jsonl`、数据集与权重均保留本机；Git 只保存报告、曲线、代码、配置和小型统计文件，不分发受限数据或大权重。",
        "",
        "重新生成报告及图表（需要本机完整日志/权重）：",
        "",
        "```bash",
        "bash reproduction/run_env.sh python reproduction/scripts/write_training_result.py",
        "```",
        "",
        "TensorBoard 本机地址为 `http://127.0.0.1:6006`；运行目录为 `reproduction/campaigns/20261114_8xa100`。",
        "",
        "## 7. 尚不能据此下结论的部分",
        "",
        "目前没有本批最终模型的固定起点全动作评测、独立验证集成功率或 reference/策略同步视频。训练集自适应采样结果不能与论文的完整动作成功率直接对表，也不能据此宣布所有模型已严格收敛。下一步应对八个最终模型使用相同固定协议，从动作首帧评测、按动作统计失败率与误差，并检查 BONES 数据筛选协议；再基于测评选择最终模型和制作对照演示。",
    ]
    (REPO / "RESULT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
