"""Generate the current GenCoord x2 report figures from audited local evidence."""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = REPO / "reproduction/campaigns/20261114_8xa100"
RUN_NAMES = ("lafan1_gencoord_x2", "boneseed_gencoord_x2")
COLORS = ("#0072B2", "#D55E00")
LABELS = ("LAFAN1", "BONES\n(unfiltered)")


def selected_runs(report):
    """Select only the two named models, independently of historical audit scope."""
    runs = {r["name"]: r for r in report["runs"]}
    selected = [runs[name] for name in RUN_NAMES]
    assert all(r["representation"] == "gencoord" and r["scale"] == 2 for r in selected)
    return selected


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def grid(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)


def bars(ax, values, fmt, ylim):
    ax.bar([0, 1], values, color=COLORS, width=0.5)
    ax.set_xticks([0, 1], LABELS)
    ax.set_ylim(0, ylim)
    for i, value in enumerate(values):
        ax.annotate(
            format(value, fmt),
            (i, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    grid(ax)


def make_figures(report, histories=None):
    directory = REPO / "docs/figures/20261114_8xa100"
    directory.mkdir(parents=True, exist_ok=True)
    runs = selected_runs(report)
    if histories is None:
        histories = {
            r["name"]: [
                json.loads(line)
                for line in (CAMPAIGN / r["name"] / "metrics.jsonl")
                .read_text()
                .splitlines()
            ]
            for r in runs
        }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Dataset composition uses the converted reference manifest, not rollout logs.
    manifest = json.loads(
        (REPO / "reproduction/data/lafan1/manifest_converted.json").read_text()
    )
    categories = defaultdict(lambda: {"clips": 0, "minutes": 0.0})
    for clip in manifest["clips"]:
        category = re.sub(r"\d+$", "", clip["name"].split("_subject")[0])
        categories[category]["clips"] += 1
        categories[category]["minutes"] += clip["seconds"] / 60
    names = {
        "dance": "Dance",
        "fallAndGetUp": "Fall / get up",
        "fight": "Fight",
        "fightAndSports": "Fight / sports",
        "jumps": "Jump",
        "run": "Run",
        "sprint": "Sprint",
        "walk": "Walk",
    }
    ordered = sorted(categories, key=lambda k: categories[k]["minutes"], reverse=True)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.1),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.1, 1]},
    )
    ax = axes[0]
    vals = [categories[k]["minutes"] for k in ordered]
    ax.barh(range(len(vals)), vals, color="#0072B2", height=0.65)
    ax.set_yticks(range(len(vals)), [names[k] for k in ordered])
    ax.invert_yaxis()
    ax.set_xlim(0, 66)
    for i, k in enumerate(ordered):
        ax.text(
            vals[i] + 0.7,
            i,
            f"{vals[i]:.1f} ({categories[k]['clips']} clips)",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("Reference duration (minutes)")
    ax.set_title("(a) LAFAN1: 40 clips, 2.45 h")
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.22)
    ax = axes[1]
    counts = [142220, 129667, 129514]
    ax.barh(
        range(3),
        np.array(counts) / 1000,
        height=0.55,
        color=["#0072B2", "#999999", "#009E73"],
    )
    ax.set_yticks(
        range(3),
        [
            "Raw release\n(this study)",
            "After name filter\n(paper)",
            "After feasibility filter\n(paper benchmark)",
        ],
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 172)
    for i, count in enumerate(counts):
        ax.text(count / 1000 + 1.5, i, f"{count:,}", va="center", fontsize=9)
    ax.set_xlabel("Number of motion clips (thousands)")
    ax.set_title("(b) BONES: raw and paper-filtered sets")
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.22)
    save(fig, directory, "dataset_overview")

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6), constrained_layout=True)
    for i, run in enumerate(runs):
        rows = histories[run["name"]]
        transitions = np.array([v["iteration"] * run["num_envs"] for v in rows])
        bins = np.minimum((transitions / run["transitions"] * 100).astype(int), 99)
        for col, (key, factor, label) in enumerate(
            [
                ("Train/mean_reward", 1, "Training return"),
                (
                    "Metrics/motion/error_body_pos",
                    100,
                    "Aligned body position error (cm)",
                ),
            ]
        ):
            values = np.array([v[key] for v in rows]) * factor
            unique = sorted(set(bins))
            x = [transitions[bins == b].mean() / 1e9 for b in unique]
            y = [values[bins == b].mean() for b in unique]
            ax = axes[i, col]
            ax.plot(x, y, color=COLORS[i], linewidth=1.5)
            ax.set_title(f"({chr(97 + 2 * i + col)}) {LABELS[i].replace(chr(10), ' ')}")
            ax.set_xlabel("Collected transitions (billions)")
            ax.set_ylabel(label)
            ax.set_xlim(0, run["transitions"] / 1e9)
            grid(ax)
    save(fig, directory, "training_curves")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
    bars(axes[0], [r["hours"] for r in runs], ".2f", 18)
    axes[0].set_title("(a) Training time")
    axes[0].set_ylabel("Wall-clock hours")
    bars(
        axes[1], [r["hardware"]["unshared_device_util_mean"] for r in runs], ".2f", 110
    )
    axes[1].set_title("(b) Mean GPU utilization")
    axes[1].set_ylabel("Utilization (%)")
    axes[1].axhline(80, color="#555555", linestyle=":", linewidth=1)
    bars(axes[2], [r["hardware"]["nvml_sampled_peak_gib"] for r in runs], ".2f", 65)
    axes[2].set_title("(c) Peak GPU memory")
    axes[2].set_ylabel("Sampled peak (GiB)")
    save(fig, directory, "resource_usage")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), constrained_layout=True)
    loss = [r["tail_means"]["Loss/state"] for r in runs]
    error = [r["tail_means"]["Metrics/motion/error_body_pos"] * 100 for r in runs]
    bars(axes[0], loss, ".6f", max(loss) * 1.25)
    axes[0].set_title("(a) World-model training loss")
    axes[0].set_ylabel("Weighted state-prediction loss")
    bars(axes[1], error, ".3f", max(error) * 1.25)
    axes[1].set_title("(b) Body position error")
    axes[1].set_ylabel("Aligned error (cm)")
    save(fig, directory, "model_control_gap")

    (directory / "figure_data.json").write_text(
        json.dumps(
            {
                "scope": "GenCoord x2 only",
                "run_names": list(RUN_NAMES),
                "sources": [
                    "../../../reproduction/campaigns/20261114_8xa100/gencoord_x2_results.json",
                    *[
                        f"../../../reproduction/campaigns/20261114_8xa100/{name}/metrics.jsonl"
                        for name in RUN_NAMES
                    ],
                    "../../../reproduction/data/lafan1/manifest_converted.json",
                ],
                "dataset_categories": categories,
                "bones_paper_filter_counts": [142220, 129667, 129514],
                "paper_source": "arXiv:2610.28480v2, Table A.7",
                "curve_bins": 100,
                "curve_x": "collected real-simulation transitions / 1e9",
                "end_metrics": "means over the last 10% of training-log rows",
                "world_model_losses": loss,
                "body_position_errors_cm": error,
                "gpu_hours": [r["hours"] for r in runs],
                "gpu_util_means": [
                    r["hardware"]["unshared_device_util_mean"] for r in runs
                ],
                "sampled_peak_gib": [
                    r["hardware"]["nvml_sampled_peak_gib"] for r in runs
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    make_figures(json.loads((CAMPAIGN / "training_results.json").read_text()))
    print(REPO / "docs/figures/20261114_8xa100")
