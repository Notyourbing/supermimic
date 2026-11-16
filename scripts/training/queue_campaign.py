"""Wait for eight idle GPUs, validate exclusive smoke tests, then launch once."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = root / "reproduction/campaigns/20261114_8xa100"
sys.path.insert(0, str(root / "scripts/monitoring"))
from gpu_availability import snapshot  # noqa: E402


def status(state, **extra):
    value = {"state": state, "time": time.time(), **extra}
    tmp = out / "queue_status.tmp"
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(out / "queue_status.json")
    print(json.dumps(value), flush=True)


if (out / "training_jobs.json").exists():
    raise SystemExit("Training already launched; use explicit per-run resume instead.")
attempt = 0
while True:
    stable = 0
    while stable < 3:
        resources = snapshot()
        stable = stable + 1 if resources["idle"] else 0
        status("waiting_for_eight_idle_gpus", stable_checks=stable, resources=resources)
        time.sleep(30)
    attempt += 1
    tag = f"queue_{int(time.time())}_{attempt}"
    status("running_exclusive_smoke", tag=tag)
    with (out / f"{tag}.log").open("w") as fh:
        process = subprocess.Popen(
            [sys.executable, str(root / "scripts/training/probe_all.py"), "--tag", tag],
            cwd=root,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
        contended = False
        known_probe_pids = set()
        while process.poll() is None:
            resources = snapshot()
            foreign = []
            for app in resources["processes"]:
                if app["pid"] in known_probe_pids:
                    continue
                lookup = subprocess.run(
                    ["ps", "-p", str(app["pid"]), "-o", "stat=,args="],
                    text=True,
                    capture_output=True,
                )
                command = lookup.stdout.strip()
                # NVML may briefly retain an exiting process. Do not treat its
                # transient '[python] <defunct>' command as a competing job.
                if command.startswith("Z"):
                    continue
                if str(root / "scripts/validation/preflight.py") in command:
                    known_probe_pids.add(app["pid"])
                    continue
                if (
                    command
                    and str(root / "scripts/validation/preflight.py") not in command
                ):
                    foreign.append(app)
            if foreign:
                contended = True
                status("smoke_resource_conflict", processes=foreign)
                # Only cancel process groups created by this attempt.
                jobs_path = out / "probe_jobs.json"
                if jobs_path.exists():
                    for job in json.loads(jobs_path.read_text()):
                        if tag in job["name"]:
                            try:
                                os.killpg(job["pid"], signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                process.wait(timeout=60)
                break
            time.sleep(5)
    if contended:
        continue
    jobs = json.loads((out / "probe_completion.json").read_text())
    if any(job["exit_code"] != 0 for job in jobs):
        status("smoke_failed_requires_investigation", tag=tag)
        raise SystemExit(1)
    subprocess.run(
        [sys.executable, str(root / "scripts/training/summarize_benchmark.py"), "--exclusive-confirmed"],
        cwd=root,
        check=True,
    )
    if not snapshot()["idle"]:
        continue
    status("launching_formal_training", tag=tag)
    subprocess.run(
        [sys.executable, str(root / "scripts/training/launch_training.py")], cwd=root, check=True
    )
    status("formal_training_launched", manifest=str(out / "training_jobs.json"))
    break
