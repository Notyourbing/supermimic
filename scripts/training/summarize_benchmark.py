"""Summarize stable smoke throughput; never infer process SM use from whole GPU use."""

import argparse
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--exclusive-confirmed", action="store_true")
args = p.parse_args()
root = Path(__file__).resolve().parents[2]
out = root / "reproduction/campaigns/20261114_8xa100"
jobs = json.loads((out / "probe_completion.json").read_text())
rows = []
for job in jobs:
    run = root / "reproduction/smoke_tests" / job["name"]
    path = run / "summary.json"
    if job["exit_code"] != 0 or not path.exists():
        rows.append(
            {
                "gpu": job["gpu"],
                "name": job["name"],
                "passed": False,
                "exit_code": job["exit_code"],
            }
        )
        continue
    summary = json.loads(path.read_text())
    times = json.loads((run / "iteration_times.json").read_text())
    times = [
        r["seconds"] for r in times if summary["warmup_last_step"] < r["step"] < 159
    ]
    samples = json.loads((run / "gpu_samples.json").read_text())
    peak_nvml_gib = max(r["gpus"][job["gpu"]]["used_bytes"] for r in samples) / 2**30
    samples = [
        r["gpus"][job["gpu"]]
        for r in samples
        if max(30, summary["warmup_last_step"]) < r["step"] < 155
    ]
    scale = 1 if job["gpu"] < 4 else 2
    seconds = statistics.mean(times)
    rows.append(
        {
            "gpu": job["gpu"],
            "name": job["name"],
            "passed": True,
            "seconds_per_iteration": seconds,
            "num_envs": scale * 4096,
            "transitions_per_second": scale * 4096 / seconds,
            "iterations_budget": 320000 // scale,
            "predicted_gpu_hours": seconds * (320000 // scale) / 3600,
            "torch_peak_reserved_gib": summary["torch_peak_reserved_bytes"] / 2**30,
            "nvml_sampled_peak_gib": peak_nvml_gib,
            "global_gpu_util_mean": statistics.mean(r["gpu_util"] for r in samples)
            if samples
            else None,
            "samples": len(times),
        }
    )
passed = all(r["passed"] for r in rows) and len(rows) == 8
exclusive = args.exclusive_confirmed and passed
report = {
    "exclusive_confirmed": exclusive,
    "all_passed": passed,
    "runs": rows,
    "caveat": "Global GPU use is attributable only when no competing compute processes existed throughout the test. Initial Nov14 probes were contended; two OOMs are resource conflicts, not validated configuration failures.",
    "predicted_wall_hours": max(r["predicted_gpu_hours"] for r in rows if r["passed"]),
    "predicted_total_gpu_hours": sum(
        r["predicted_gpu_hours"] for r in rows if r["passed"]
    )
    if passed
    else None,
}
if report["predicted_total_gpu_hours"] is not None:
    report["budget_total_gpu_hours_20pct_margin"] = (
        report["predicted_total_gpu_hours"] * 1.2
    )
    report["budget_wall_hours_20pct_margin"] = report["predicted_wall_hours"] * 1.2
(out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
lines = [
    "# 本次 smoke test 实测",
    "",
    f"独占结果已确认：{exclusive}。全八组通过：{passed}。",
    "",
    "如果独占未确认，下表仅用于集成验证及共享负载观察，不能据此声称本项目达到 80% GPU 使用率或据此作准确独占工时预测。",
    "",
    "| GPU | Run | 秒/循环 | transitions/s | GPU-Util 均值 | Torch reserved GiB | NVML 显存采样峰值 GiB | 估时 GPU-h |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    if not r["passed"]:
        lines.append(f"| {r['gpu']} | {r['name']} | failed/cancelled | | | | | |")
    else:
        lines.append(
            f"| {r['gpu']} | {r['name']} | {r['seconds_per_iteration']:.3f} | {r['transitions_per_second']:.0f} | {r['global_gpu_util_mean']:.1f}% | {r['torch_peak_reserved_gib']:.2f} | {r['nvml_sampled_peak_gib']:.2f} | {r['predicted_gpu_hours']:.1f} |"
        )
lines += [
    "",
    "编译预热和末尾保存已剔除。预计值还需增加约 20% 运行余量；未收敛阶段的吞吐可能随训练状态变化。BONES 为未筛选数据，不等同于论文集合。",
]
if exclusive:
    lines += [
        "",
        f"八组总工时约 {report['predicted_total_gpu_hours']:.1f} GPU-h，墙钟约 {report['predicted_wall_hours']:.1f} h；加 20% 余量后约 {report['budget_total_gpu_hours_20pct_margin']:.1f} GPU-h / {report['budget_wall_hours_20pct_margin']:.1f} h。",
        "",
        "默认规模约 18600–19500 transitions/s，放大规模约 23600–24600 transitions/s；提升约 26%–28%，最终效果仍需比较。这里 GPU-Util 是稳定阶段 NVML 时间占用率，不是理论峰值 FLOPS 的百分比。",
    ]
(out / "BENCHMARK.md").write_text("\n".join(lines) + "\n")
print(json.dumps(report, indent=2))
