"""Launch the validated eight independent official-source training runs once."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = root / "reproduction/campaigns/20261114_8xa100"
sys.path.insert(0, str(root / "scripts/monitoring"))
from gpu_availability import snapshot  # noqa: E402

if not snapshot()["idle"]:
    raise SystemExit("GPU resources busy: no formal jobs were launched.")
if not json.loads((out / "benchmark.json").read_text())["exclusive_confirmed"]:
    raise SystemExit("A successful unshared benchmark is required.")
if (out / "training_jobs.json").exists():
    raise SystemExit(
        "Already launched. Use explicit checkpoint continuation for a restart."
    )
complete = json.loads((out / "probe_completion.json").read_text())
if len(complete) != 8 or any(j["exit_code"] != 0 for j in complete):
    raise SystemExit("All eight probes must pass before launch.")
validation = json.loads((out / "recovery_validation.json").read_text())
if validation["status"] != "passed":
    raise SystemExit("Checkpoint save/reload validation required.")
jobs = []
for gpu in range(8):
    data = "lafan1" if gpu % 4 < 2 else "boneseed"
    rep = "gencoord" if gpu % 2 == 0 else "maxcoord"
    scale = 1 if gpu < 4 else 2
    probe = root / "reproduction/smoke_tests" / complete[gpu]["name"]
    summary = json.loads((probe / "summary.json").read_text())
    if summary["status"] != "passed" or not all(summary["parameters_changed"].values()):
        raise SystemExit(f"Failed probe: {probe}")
    name = f"{data}_{rep}_x{scale}"
    packed = f"reproduction/data/{data}/" + (
        "packed" if data == "lafan1" else "packed_unfiltered"
    )
    iterations = 320000 // scale
    cmd = [
        "taskset",
        "-c",
        f"{gpu * 8}-{gpu * 8 + 7}",
        "bash",
        str(root / "reproduction/run_env.sh"),
        "python",
        "-u",
        str(root / "scripts/training/train_campaign.py"),
        "--out",
        str(out / name),
        "--motion-file",
        packed,
        "--iterations",
        str(iterations),
        "--num-envs",
        str(4096 * scale),
        "--wm-batch",
        str(2048 * scale),
        "--policy-batch",
        str(4096 * scale),
        "--representation",
        rep,
        "--save-interval",
        "5000",
        "--resume-interval",
        "10000",
        "--log-interval",
        "50",
    ]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONUNBUFFERED="1",
        TORCHINDUCTOR_COMPILE_THREADS="4",
    )
    (out / name).mkdir(exist_ok=True)
    with (out / name / "train.log").open("w") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=root,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    jobs.append(
        {
            "gpu": gpu,
            "name": name,
            "pid": proc.pid,
            "iterations": iterations,
            "num_envs": 4096 * scale,
            "command": [
                str(Path(arg).relative_to(root))
                if Path(arg).is_absolute() and Path(arg).is_relative_to(root)
                else arg
                for arg in cmd
            ],
            "motion_file": packed,
        }
    )
    subprocess.run(
        [
            "tmux",
            "new-window",
            "-t",
            "supermimic",
            "-n",
            f"train{gpu}",
            shlex.join(["tail", "-F", str(out / name / "train.log")]),
        ],
        check=True,
    )
(out / "training_jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
subprocess.run(["tmux", "select-window", "-t", "supermimic:progress"], check=True)
print(json.dumps(jobs, indent=2))
