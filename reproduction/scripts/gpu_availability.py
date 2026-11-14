"""Read-only resource gate; never signals processes belonging to other jobs."""

import csv
import io
import subprocess


def snapshot():
    def query(kind, fields):
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            text=True,
        )
        return list(csv.reader(io.StringIO(output), skipinitialspace=True))

    gpus = query("gpu", "index,memory.used,memory.free,utilization.gpu")
    apps = query("compute-apps", "pid,used_gpu_memory")
    return {
        "gpus": [
            {
                "gpu": int(i),
                "used_mib": int(used),
                "free_mib": int(free),
                "utilization": int(util),
            }
            for i, used, free, util in gpus
        ],
        "processes": [{"pid": int(pid), "used_mib": used} for pid, used in apps],
        "idle": len(gpus) == 8 and not apps and all(int(g[1]) < 2048 for g in gpus),
    }
