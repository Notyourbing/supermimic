import argparse
import json
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--tag", default="initial")
args = p.parse_args()
jobs = []
for gpu in range(8):
    data = "lafan1" if gpu % 4 < 2 else "boneseed"
    rep = "gencoord" if gpu % 2 == 0 else "maxcoord"
    scale = 1 if gpu < 4 else 2
    name = f"allocated_20261114_{args.tag}_{data}_{rep}_x{scale}"
    packed = (
        "/home/zhuobing/projects/supermimic/official_reproduction/data/"
        + data
        + ("/packed" if data == "lafan1" else "/packed_unfiltered")
    )
    cmd = [
        "taskset",
        "-c",
        f"{gpu * 8}-{gpu * 8 + 7}",
        "bash",
        str(root / "reproduction/run_env.sh"),
        "python",
        "-u",
        str(root / "reproduction/scripts/preflight.py"),
        "--name",
        name,
        "--iterations",
        "160",
        "--num-envs",
        str(4096 * scale),
        "--wm-batch",
        str(2048 * scale),
        "--wm-batches",
        "10",
        "--policy-batch",
        str(4096 * scale),
        "--representation",
        rep,
        "--motion-file",
        packed,
    ]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONUNBUFFERED="1",
        TORCHINDUCTOR_COMPILE_THREADS="4",
    )
    fh = (out / f"probe_{args.tag}_gpu{gpu}.log").open("w")
    p = subprocess.Popen(
        cmd,
        cwd=root,
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    jobs.append({"gpu": gpu, "name": name, "pid": p.pid, "process": p, "log": fh})
(out / "probe_jobs.json").write_text(
    json.dumps(
        [{k: v for k, v in j.items() if k not in ("process", "log")} for j in jobs],
        indent=2,
    )
)
for j in jobs:
    j["exit_code"] = j["process"].wait()
    j["log"].close()
    print(j["name"], j["exit_code"], flush=True)
(out / "probe_completion.json").write_text(
    json.dumps(
        [{k: v for k, v in j.items() if k not in ("process", "log")} for j in jobs],
        indent=2,
    )
)
