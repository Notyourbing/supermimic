"""Observe one preparation process, separating its NVML samples from other jobs."""

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import pynvml

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--pid", type=int, required=True)
p.add_argument("--gpu", type=int, required=True)
p.add_argument("--run-dir", type=Path, required=True)
p.add_argument("--seconds", type=int, default=600)
args = p.parse_args()
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
last_timestamp = 0
started = time.monotonic()
path = args.run_dir / "process_util.jsonl"
with path.open("w") as out:
    while time.monotonic() - started < args.seconds:
        try:
            os.kill(args.pid, 0)
        except ProcessLookupError:
            break
        samples = []
        try:
            samples = pynvml.nvmlDeviceGetProcessUtilization(handle, last_timestamp)
        except pynvml.NVMLError:
            pass
        if samples:
            last_timestamp = max(sample.timeStamp for sample in samples)
        own = [sample for sample in samples if sample.pid == args.pid]
        memory = None
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if proc.pid == args.pid:
                memory = proc.usedGpuMemory
        step = None
        phases = args.run_dir / "phases.jsonl"
        if phases.exists():
            lines = phases.read_text().splitlines()
            if lines:
                try:
                    step = json.loads(lines[-1])["step"]
                except (ValueError, KeyError):
                    pass
        row = {
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "pid": args.pid,
            "physical_gpu": args.gpu,
            "elapsed_seconds": time.monotonic() - started,
            "last_completed_update_step": step,
            "own_sm_util": own[-1].smUtil if own else None,
            "global_gpu_util": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            "own_used_memory_bytes": memory,
        }
        out.write(json.dumps(row) + "\n")
        out.flush()
        time.sleep(1)
