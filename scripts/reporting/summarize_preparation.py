"""Build transparent timing / memory estimates from the bounded probes."""

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "reproduction"
BASELINE = json.loads((ROOT / "setup/gpu_baseline_after_tests.json").read_text())
BASE_MEMORY = {
    row["index"]: row["existing_other_jobs_memory_bytes"] for row in BASELINE["gpus"]
}
NAMES = [
    "benchmark_default_gencoord",
    "benchmark_default_maxcoord",
    "benchmark_largebatch_gencoord",
    "benchmark_cached_gencoord",
    "benchmark_samegpu_largebatch",
    "benchmark_bones_gencoord",
]
rows = []
for name in NAMES:
    folder = ROOT / "smoke_tests" / name
    if not (folder / "summary.json").exists():
        continue
    summary = json.loads((folder / "summary.json").read_text())
    cfg = summary["config"]
    intervals = json.loads((folder / "iteration_times.json").read_text())
    cutoff = max(20, summary["warmup_last_step"])
    # Last boundary includes W&B/environment cleanup. Do not count it as training.
    stable = [
        row
        for row in intervals
        if cutoff < row["step"] < cfg["max_iterations"] - 1
        and row["step"] % cfg["save_interval"]
    ]
    mean = statistics.mean(row["seconds"] for row in stable)
    batches = cfg["wm"]["num_batches"]
    batch = cfg["wm"]["batch_size"]
    n_envs = json.loads((folder / "config.json").read_text())["args"]["num_envs"]
    frequency = n_envs / (batches * batch * cfg["wm"]["horizon"]) * cfg["replay_ratio"]
    gpu = int(summary["cuda_visible_devices"])
    samples = json.loads((folder / "gpu_samples.json").read_text())
    own_peak = max(s["gpus"][gpu]["used_bytes"] - BASE_MEMORY[gpu] for s in samples)
    row = {
        "probe": name,
        "representation": cfg["representation"],
        "num_envs": n_envs,
        "wm_batch": batch,
        "policy_batch": cfg["policy"]["batch_size"],
        "learner_updates_per_collection": frequency,
        "observed_learner_updates": summary["learner_updates"],
        "steady_samples": len(stable),
        "seconds_per_collection_iteration": mean,
        "collection_transitions_per_second": n_envs / mean,
        "torch_peak_allocated_GiB": summary["torch_peak_allocated_bytes"] / 2**30,
        "torch_peak_reserved_GiB": summary["torch_peak_reserved_bytes"] / 2**30,
        "estimated_process_peak_GiB_from_nvml_delta": own_peak / 2**30,
        "init_seconds": summary["init_seconds"],
        "probe_loop_seconds_including_compile": summary[
            "loop_seconds_including_compile"
        ],
        "estimate_10000_collection_iterations_hours": mean * 10_000 / 3600,
        "estimate_320000_collection_iterations_hours": mean * 320_000 / 3600,
        "estimate_default_1000000_collection_iterations_hours": mean * 1_000_000 / 3600,
    }
    rows.append(row)
eval_rows = []
for name in [
    "benchmark_default_gencoord",
    "benchmark_default_maxcoord",
    "benchmark_largebatch_gencoord",
]:
    path = ROOT / "smoke_tests" / name / "eval_1024.json"
    if path.exists():
        value = json.loads(path.read_text())
        eval_rows.append(
            {
                "probe": name,
                "num_envs": 1024,
                "transitions_per_second": value["transitions_per_second"],
                "status": value["status"],
            }
        )
result = {
    "hardware_shared": True,
    "formal_training_started": False,
    "training": rows,
    "evaluation": eval_rows,
    "interpretation": [
        "All timings were measured while unrelated jobs occupied each GPU; these are shared-resource scheduling estimates, not exclusive A100 benchmarks.",
        "Iterations mean the current runner's collection loops, not guaranteed learning updates. The default learner-update frequency is 0.4.",
        "Changing both batches by 2x halves update frequency at fixed collection settings. Equal collection loops do not mean equal optimizer update counts.",
        "NVML process peaks are estimated by subtracting the stable pre-existing GPU memory baseline; Torch peaks exclude Warp allocations.",
        "A raw-full-BONES dataset consumes extra GPU memory beyond LAFAN; add the packed verification manifest's resident_float32_bytes and keep a memory margin.",
        "Dataset frames / throughput / number of GPUs is a padding-free evaluation lower bound, not a complete benchmark ETA. Clip length, scheduling, startup and tail work must be added.",
    ],
}
(ROOT / "smoke_tests/performance_summary.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
with (ROOT / "smoke_tests/performance_summary.csv").open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(result, indent=2), flush=True)
