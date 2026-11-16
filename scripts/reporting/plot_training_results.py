"""Make publication-style figures from audited results and local training logs."""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = REPO / "reproduction/campaigns/20261114_8xa100"
COLORS = {"gencoord": "#0072B2", "maxcoord": "#D55E00"}
GROUPS = [
    ("lafan1", "gencoord"),
    ("lafan1", "maxcoord"),
    ("boneseed", "gencoord"),
    ("boneseed", "maxcoord"),
]
GROUP_LABELS = [
    "LAFAN1\nGenCoord",
    "LAFAN1\nMaxCoord",
    "BONES\nGenCoord",
    "BONES\nMaxCoord",
]


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def grid(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)


def scale_legend(ax, **kwargs):
    ax.legend(
        handles=[
            Patch(facecolor="white", edgecolor="#555555", label="x1: standard"),
            Patch(
                facecolor="white",
                edgecolor="#555555",
                hatch="///",
                label="x2: enlarged",
            ),
        ],
        frameon=False,
        **kwargs,
    )


def paired_bars(ax, runs, value, fmt=".2f", annotate_gpu=False):
    for j, (dataset, representation) in enumerate(GROUPS):
        for scale in (1, 2):
            run = runs[f"{dataset}_{representation}_x{scale}"]
            y = value(run)
            x = j + (-0.19 if scale == 1 else 0.19)
            ax.bar(
                x,
                y,
                width=0.34,
                color=COLORS[representation],
                edgecolor="white",
                hatch="" if scale == 1 else "///",
                linewidth=0.6,
            )
            label = format(y, fmt)
            if annotate_gpu:
                label += f"\nG{run['gpu']}"
            ax.annotate(
                label,
                (x, y),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(range(4), GROUP_LABELS)
    grid(ax)


def make_figures(report, histories=None):
    directory = REPO / "docs/figures/20261114_8xa100"
    directory.mkdir(parents=True, exist_ok=True)
    runs = {r["name"]: r for r in report["runs"]}
    if histories is None:
        histories = {
            name: [
                json.loads(line)
                for line in (CAMPAIGN / name / "metrics.jsonl").read_text().splitlines()
            ]
            for name in runs
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

    # Every curve uses 100 bins with identical widths in collected transitions.
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), constrained_layout=True)
    for run in runs.values():
        rows = histories[run["name"]]
        x = np.array([v["iteration"] * run["num_envs"] for v in rows]) / 1e9
        bins = np.minimum((x / 1.31072 * 100).astype(int), 99)
        for col, (key, factor) in enumerate(
            [("Train/mean_reward", 1), ("Metrics/motion/error_body_pos", 100)]
        ):
            values = np.array([v[key] for v in rows]) * factor
            unique = sorted(set(bins))
            xs = [x[bins == b].mean() for b in unique]
            ys = [values[bins == b].mean() for b in unique]
            ax = axes[0 if run["dataset"] == "LAFAN1" else 1, col]
            rep = "GenCoord" if run["representation"] == "gencoord" else "MaxCoord"
            ax.plot(
                xs,
                ys,
                color=COLORS[run["representation"]],
                linestyle="-" if run["scale"] == 1 else "--",
                linewidth=1.5,
                label=f"{rep} x{run['scale']}",
            )
    for i, dataset in enumerate(["LAFAN1", "BONES (unfiltered)"]):
        for col, metric in enumerate(
            ["Training return (higher is better)", "Aligned body position error (cm)"]
        ):
            ax = axes[i, col]
            ax.set_title(f"({chr(97 + 2 * i + col)}) {dataset}")
            ax.set_xlabel("Collected transitions (billions)")
            ax.set_ylabel(metric)
            ax.set_xlim(0, 1.31072)
            grid(ax)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    save(fig, directory, "training_curves")

    # Separate dataset axes retain detail without implying equal error magnitudes.
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.3), constrained_layout=True)
    for i, dataset in enumerate(["lafan1", "boneseed"]):
        for j, (key, factor, label, fmt) in enumerate(
            [
                (
                    "Metrics/motion/error_body_pos",
                    100,
                    "Aligned body position error (cm)",
                    ".2f",
                ),
                (
                    "Metrics/motion/error_body_rot",
                    1,
                    "Body orientation error (rad)",
                    ".3f",
                ),
            ]
        ):
            ax = axes[i, j]
            heights = []
            for p, rep in enumerate(["gencoord", "maxcoord"]):
                for scale in (1, 2):
                    run = runs[f"{dataset}_{rep}_x{scale}"]
                    val = run["tail_means"][key] * factor
                    heights.append(val)
                    x = p + (-0.18 if scale == 1 else 0.18)
                    ax.bar(
                        x,
                        val,
                        width=0.32,
                        color=COLORS[rep],
                        edgecolor="white",
                        hatch="" if scale == 1 else "///",
                        linewidth=0.6,
                    )
                    ax.annotate(
                        format(val, fmt),
                        (x, val),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        fontsize=9,
                    )
            ax.set_ylim(0, max(heights) * 1.25)
            ax.set_xticks([0, 1], ["GenCoord", "MaxCoord"])
            ax.set_ylabel(label)
            ax.set_title(
                f"({chr(97 + 2 * i + j)}) {'LAFAN1' if i == 0 else 'BONES (unfiltered)'}"
            )
            grid(ax)
    scale_legend(axes[0, 0], loc="upper right", fontsize=8, ncol=2)
    save(fig, directory, "tracking_accuracy")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    paired_bars(axes[0], runs, lambda r: r["hours"])
    axes[0].set_ylim(0, 25)
    axes[0].set_ylabel("Wall-clock time (hours)")
    axes[0].set_title("(a) Same budget: 1.311 billion transitions")
    scale_legend(axes[0], loc="upper left", fontsize=8, ncol=2)
    paired_bars(
        axes[1], runs, lambda r: r["hardware"]["unshared_device_util_mean"], fmt=".1f"
    )
    axes[1].set_ylim(0, 110)
    axes[1].set_ylabel("Mean GPU utilization (%)")
    axes[1].set_title("(b) Steady training phase")
    axes[1].axhline(80, color="#555555", linestyle=":", linewidth=1)
    axes[1].text(
        3.45,
        78,
        "80% target",
        va="top",
        ha="right",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2},
    )
    save(fig, directory, "training_efficiency")

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    paired_bars(
        ax, runs, lambda r: r["hardware"]["nvml_sampled_peak_gib"], annotate_gpu=True
    )
    ax.set_ylim(0, 73)
    ax.set_ylabel("Sampled peak GPU memory (GiB)")
    ax.set_title("GPU memory during each model's training interval")
    scale_legend(ax, loc="upper left", fontsize=9, ncol=2)
    save(fig, directory, "gpu_memory")

    # Ratios compare datasets within the SAME representation and scale.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True)
    ratios = []
    for p, rep in enumerate(["gencoord", "maxcoord"]):
        for scale in (1, 2):
            a = runs[f"lafan1_{rep}_x{scale}"]["tail_means"]
            b = runs[f"boneseed_{rep}_x{scale}"]["tail_means"]
            ratio = {"representation": rep, "scale": scale}
            for col, key in enumerate(["Loss/state", "Metrics/motion/error_body_pos"]):
                y = b[key] / a[key]
                ratio[key] = y
                x = p + (-0.18 if scale == 1 else 0.18)
                axes[col].bar(
                    x,
                    y,
                    width=0.32,
                    color=COLORS[rep],
                    edgecolor="white",
                    hatch="" if scale == 1 else "///",
                    linewidth=0.6,
                )
                axes[col].annotate(
                    f"{y:.2f}",
                    (x, y),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )
            ratios.append(ratio)
    for col, title in enumerate(
        ["(a) World-model training loss", "(b) Aligned body position error"]
    ):
        ax = axes[col]
        ax.set_xticks([0, 1], ["GenCoord", "MaxCoord"])
        ax.set_title(title)
        ax.set_ylabel("BONES / LAFAN1 ratio")
        ax.axhline(1, color="#555555", linestyle=":", linewidth=1)
        ax.set_ylim(0, 1.8 if col == 0 else 5.8)
        grid(ax)
    scale_legend(axes[0], loc="upper left", fontsize=8, ncol=2)
    save(fig, directory, "model_control_gap")

    (directory / "figure_data.json").write_text(
        json.dumps(
            {
                "sources": [
                    "../../../reproduction/campaigns/20261114_8xa100/training_results.json",
                    "../../../reproduction/campaigns/20261114_8xa100/*/metrics.jsonl",
                    "../../../reproduction/data/lafan1/manifest_converted.json",
                ],
                "dataset_categories": categories,
                "bones_paper_filter_counts": [142220, 129667, 129514],
                "paper_source": "arXiv:2610.28480v2, Table A.7",
                "curve_bins": 100,
                "curve_x": "collected real-simulation transitions / 1e9",
                "end_metrics": "means over the last 10% of training-log rows",
                "cross_dataset_ratios": ratios,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    make_figures(json.loads((CAMPAIGN / "training_results.json").read_text()))
    print(REPO / "docs/figures/20261114_8xa100")
