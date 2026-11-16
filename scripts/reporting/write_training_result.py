"""Audit completed local runs and regenerate the evidence-backed RESULT.md."""

import argparse
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path

import torch
from plot_training_results import make_figures, selected_runs

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
    write_selected_analysis(report)
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


def write_selected_analysis(report):
    """Publish only the selected runs; retain historical audit files unchanged."""
    runs = selected_runs(report)
    a, b = runs
    configurations = []
    for r in runs:
        cfg = json.loads((CAMPAIGN / r["name"] / "config.json").read_text())
        runner = cfg["runner_cfg"]
        configuration = {
            "representation": runner["representation"],
            "num_envs": r["num_envs"],
            "wm_batch_size": runner["wm"]["batch_size"],
            "wm_batches": runner["wm"]["num_batches"],
            "wm_window": runner["wm"]["horizon"],
            "policy_batch_size": runner["policy"]["batch_size"],
            "policy_horizon": runner["policy"]["horizon"],
            "history_length": runner["history_length"],
            "learning_rate": runner["policy"]["lr"],
        }
        configurations.append(configuration)
    assert configurations[0] == configurations[1]
    assert configurations[0]["num_envs"] == 8192
    assert configurations[0]["wm_batch_size"] == 4096
    assert configurations[0]["policy_batch_size"] == 8192
    fields = (
        "name",
        "gpu",
        "dataset",
        "representation",
        "scale",
        "num_envs",
        "iterations",
        "transitions",
        "updates",
        "start",
        "finish",
        "hours",
        "training_commit",
        "tail_count",
        "tail_means",
        "hardware",
    )
    result = {
        "scope": "GenCoord x2; LAFAN1 and unfiltered BONES",
        "source": "training_results.json and per-run config.json",
        "audit_utc": report["generated_utc"],
        "formal_evaluation_available": False,
        "configuration": configurations[0],
        "total_gpu_hours": sum(r["hours"] for r in runs),
        "wall_hours": (max(r["finish"] for r in runs) - min(r["start"] for r in runs))
        / 3600,
        "runs": [{k: r[k] for k in fields} for r in runs],
        "bones_to_lafan": {
            "reference_frame_ratio": 51915358 / 441140,
            "wall_ratio": b["hours"] / a["hours"],
            "extra_peak_gib": b["hardware"]["nvml_sampled_peak_gib"]
            - a["hardware"]["nvml_sampled_peak_gib"],
            "world_model_loss_ratio": b["tail_means"]["Loss/state"]
            / a["tail_means"]["Loss/state"],
            "body_position_error_ratio": b["tail_means"][
                "Metrics/motion/error_body_pos"
            ]
            / a["tail_means"]["Metrics/motion/error_body_pos"],
        },
    }
    (CAMPAIGN / "gencoord_x2_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


def write_markdown(report):
    """Render the current two-model report from its shared selection rule."""
    runs = selected_runs(report)
    gpu_table = [
        "| 原训练 GPU | 数据集 | 任务目录 | 耗时 / h |",
        "|---:|---|---|---:|",
    ]
    metrics_table = [
        "| 数据集 | 训练回报 ↑ | 身体位置误差 / cm ↓ | 身体姿态误差 / rad ↓ |",
        "|---|---:|---:|---:|",
    ]
    for r in runs:
        dataset = "LAFAN1" if r["dataset"] == "LAFAN1" else "BONES（未筛选）"
        gpu_table.append(
            f"| {r['gpu']} | {dataset} | `{r['name']}` | {r['hours']:.2f} |"
        )
        m = r["tail_means"]
        metrics_table.append(
            f"| {dataset} | {m['Train/mean_reward']:.3f} | "
            f"{m['Metrics/motion/error_body_pos'] * 100:.3f} | {m['Metrics/motion/error_body_rot']:.4f} |"
        )
    values = {
        "GPU_HOURS": f"{sum(r['hours'] for r in runs):.2f}",
        "AUDIT_TIME": report["generated_utc"],
        "GPU_TABLE": "\n".join(gpu_table),
        "METRICS_TABLE": "\n".join(metrics_table),
    }
    template = (REPO / "docs/templates/result_template.md").read_text()
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
        write_selected_analysis(summary)
        if args.refresh_report:
            make_figures(summary)
        write_markdown(summary)
        print(REPO / "RESULT.md")
    else:
        main()
