"""Audit completed local runs and regenerate the evidence-backed RESULT.md."""

import argparse
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path

import torch
from plot_training_results import make_figures

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
    write_comparison_analysis(report)
    (CAMPAIGN / "training_results.json").write_text(json.dumps(report, indent=2) + "\n")
    (CAMPAIGN / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoints, indent=2) + "\n"
    )
    make_figures(report, histories)
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


def write_comparison_analysis(report):
    """Derived observations; no metric is relabeled as a paper benchmark score."""
    runs = {r["name"]: r for r in report["runs"]}
    pairs = []
    for representation in ("gencoord", "maxcoord"):
        for scale in (1, 2):
            a = runs[f"lafan1_{representation}_x{scale}"]
            b = runs[f"boneseed_{representation}_x{scale}"]
            ma, mb = a["tail_means"], b["tail_means"]
            pairs.append(
                {
                    "representation": representation,
                    "scale": scale,
                    "bones_to_lafan_wall_ratio": b["hours"] / a["hours"],
                    "extra_sampled_vram_gib": b["hardware"]["nvml_sampled_peak_gib"]
                    - a["hardware"]["nvml_sampled_peak_gib"],
                    "wm_loss_change_percent": (mb["Loss/state"] / ma["Loss/state"] - 1)
                    * 100,
                    "body_position_error_ratio": mb["Metrics/motion/error_body_pos"]
                    / ma["Metrics/motion/error_body_pos"],
                }
            )
    a, b = runs["boneseed_maxcoord_x1"], runs["boneseed_maxcoord_x2"]
    tradeoff = {
        key: (b["tail_means"][key] / a["tail_means"][key] - 1) * 100
        for key in (
            "Train/mean_reward",
            "Metrics/motion/error_body_pos",
            "Metrics/motion/error_body_rot",
        )
    }
    representations = []
    for dataset in ("lafan1", "boneseed"):
        for scale in (1, 2):
            gen = runs[f"{dataset}_gencoord_x{scale}"]
            maxcoord = runs[f"{dataset}_maxcoord_x{scale}"]
            representations.append(
                {
                    "dataset": dataset,
                    "scale": scale,
                    "maxcoord_vs_gencoord_body_position_percent": (
                        maxcoord["tail_means"]["Metrics/motion/error_body_pos"]
                        / gen["tail_means"]["Metrics/motion/error_body_pos"]
                        - 1
                    )
                    * 100,
                    "maxcoord_vs_gencoord_body_orientation_percent": (
                        maxcoord["tail_means"]["Metrics/motion/error_body_rot"]
                        / gen["tail_means"]["Metrics/motion/error_body_rot"]
                        - 1
                    )
                    * 100,
                }
            )
    analysis = {
        "source": "training_results.json; last 10% logged training metrics",
        "paper_version": "arXiv:2610.28480v2",
        "formal_benchmark_comparable": False,
        "cross_dataset_pairs": pairs,
        "nominal_exposure_ratio_not_epochs": {
            "definition": "collected transitions / reference library frame count; not measured per-motion visits",
            "lafan1": 1310720000 / 441140,
            "bones_unfiltered": 1310720000 / 51915358,
            "reference_frame_count_ratio": 51915358 / 441140,
        },
        "bones_maxcoord_x2_vs_x1_percent_changes": tradeoff,
        "representation_comparisons": representations,
        "interpretation": "descriptive observations under this protocol; not causal ablations or unseen-motion generalization evidence",
    }
    (CAMPAIGN / "comparison_analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n"
    )


def write_markdown(report):
    """Render the readable report template; numerical tables use audited values."""
    runs = sorted(report["runs"], key=lambda r: r["gpu"])
    purposes = {
        0: "建立小型动作库的广义坐标基准",
        1: "与 GPU 0 对照，检验最大坐标表示",
        2: "检验广义坐标在大型动作库上的表现",
        3: "与 GPU 2 对照，检验大型动作库上的表示差异",
        4: "与 GPU 0 对照，检验扩大规模后的速度与精度",
        5: "与 GPU 1 对照，检验扩大规模后的速度与精度",
        6: "与 GPU 2 对照，检验大型动作库的规模收益",
        7: "与 GPU 3 对照，检验大型动作库的规模收益",
    }
    gpu_table = [
        "| GPU | 数据集 | 世界模型表示 | 训练规模 | 实验目的 | 耗时 / h |",
        "|---:|---|---|---|---|---:|",
    ]
    metrics_table = [
        "| GPU | 数据集 | 配置 | 训练回报 ↑ | 身体位置误差 / cm ↓ | 身体姿态误差 / rad ↓ |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for r in runs:
        dataset = "LAFAN1" if r["dataset"] == "LAFAN1" else "BONES（未筛选）"
        rep = "GenCoord" if r["representation"] == "gencoord" else "MaxCoord"
        gpu_table.append(
            f"| {r['gpu']} | {dataset} | {rep} | x{r['scale']} | {purposes[r['gpu']]} | {r['hours']:.2f} |"
        )
        m = r["tail_means"]
        metrics_table.append(
            f"| {r['gpu']} | {dataset} | {rep} x{r['scale']} | {m['Train/mean_reward']:.3f} | "
            f"{m['Metrics/motion/error_body_pos'] * 100:.3f} | {m['Metrics/motion/error_body_rot']:.4f} |"
        )
    values = {
        "WALL_HOURS": f"{report['wall_hours']:.2f}",
        "GPU_HOURS": f"{report['total_gpu_hours']:.2f}",
        "AUDIT_TIME": report["generated_utc"],
        "GPU_TABLE": "\n".join(gpu_table),
        "METRICS_TABLE": "\n".join(metrics_table),
    }
    template = (REPO / "reproduction/reports/result_template.md").read_text()
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    assert "{{" not in template, "Unresolved report template placeholder"
    (REPO / "RESULT.md").write_text(template)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate Markdown and derived comparisons from the audited summary only.",
    )
    mode.add_argument(
        "--refresh-report",
        action="store_true",
        help="Regenerate Markdown and figures from audited results and logs; skip checkpoint and telemetry audits.",
    )
    args = parser.parse_args()
    if args.report_only or args.refresh_report:
        summary = json.loads((CAMPAIGN / "training_results.json").read_text())
        write_comparison_analysis(summary)
        if args.refresh_report:
            make_figures(summary)
        write_markdown(summary)
        print(REPO / "RESULT.md")
    else:
        main()
