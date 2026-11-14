"""Read-only terminal dashboard and persistent hardware/TensorBoard telemetry."""

import argparse
import json
import os
import time
from pathlib import Path


def dashboard(root):
    print("\033[2J\033[H", end="")
    print(
        "SuperMimic official | seed 42 | "
        + time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    )
    print(
        "GPU  RUN                                  ITERATION       UPDATES  PROGRESS   ETA(h)  SEC/ITER  STATE"
    )
    manifest = root / "training_jobs.json"
    if not manifest.exists():
        queue = root / "queue_status.json"
        if queue.exists():
            state = json.loads(queue.read_text())
            print("\nQueue:", state["state"])
            resources = state.get("resources", {})
            for gpu in resources.get("gpus", []):
                print(
                    f"GPU {gpu['gpu']}  used={gpu['used_mib'] / 1024:.1f} GiB  "
                    f"free={gpu['free_mib'] / 1024:.1f} GiB  device util={gpu['utilization']}%"
                )
            print(
                "Competing compute PIDs:",
                ", ".join(str(p["pid"]) for p in resources.get("processes", [])),
            )
            print(
                "Formal training starts only after 8 idle GPUs and successful exclusive probes."
            )
        else:
            print("Smoke tests / recovery validation in progress.")
        print("\nTensorBoard: http://127.0.0.1:6006 | Ctrl-b w: select tmux window")
        return
    for job in json.loads(manifest.read_text()):
        path = root / job["name"] / "status.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        it, total = state.get("iteration", 0), job["iterations"]
        eta = state.get("eta_hours")
        seconds = state.get("seconds_per_iteration")
        stale = time.time() - state.get("time", time.time()) > 600
        try:
            os.kill(job["pid"], 0)
            alive = True
        except ProcessLookupError:
            alive = False
        label = state.get("status", "initializing")
        if not alive and label not in ("completed", "interrupted", "failed"):
            label = "EXITED: CHECK LOG"
        elif stale:
            label += " (stale)"
        print(
            f"{job['gpu']:>3}  {job['name']:<36} {it:>7}/{total:<7} {state.get('updates', 0):>7} {100 * it / total:>8.2f}% {eta if eta is not None else 0:>8.2f} {seconds if seconds is not None else 0:>9.3f}  {label}"
        )
    print(
        "\nTensorBoard: http://127.0.0.1:6006 | tmux windows: progress / gpu / train0..7 / tensorboard"
    )
    print("ETA excludes compile warmup. GPU telemetry and raw metrics remain on disk.")


def monitor(root):
    import pynvml
    from torch.utils.tensorboard import SummaryWriter

    pynvml.nvmlInit()
    writer = SummaryWriter(str(root / "hardware/tensorboard"), flush_secs=20)
    handles = [
        pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())
    ]
    step = 0
    command_cache = {}
    with (root / "hardware.jsonl").open("a", buffering=1) as fh:
        while True:
            row = {"time": time.time(), "gpus": []}
            for index, handle in enumerate(handles):
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                values = {
                    "index": index,
                    "utilization": util.gpu,
                    "memory_used_gib": mem.used / 2**30,
                    "power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                }
                processes = []
                for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    if process.pid not in command_cache:
                        try:
                            command_cache[process.pid] = (
                                Path(f"/proc/{process.pid}/cmdline")
                                .read_bytes()
                                .decode(errors="replace")
                            )
                        except (FileNotFoundError, PermissionError):
                            command_cache[process.pid] = ""
                    command = command_cache[process.pid]
                    belongs = (
                        str(root.resolve()) in command
                        and "train_campaign.py" in command
                    )
                    processes.append(
                        {
                            "pid": process.pid,
                            "campaign": belongs,
                            "used_gib": process.usedGpuMemory / 2**30,
                        }
                    )
                values["processes"] = processes
                values["foreign_process_count"] = sum(
                    not p["campaign"] for p in processes
                )
                values["campaign_process_count"] = sum(p["campaign"] for p in processes)
                # Process SM sampling is not supported by every NVML/driver.
                # Missing measurements remain null, never replaced by device use.
                own_pids = {p["pid"] for p in processes if p["campaign"]}
                values["campaign_sm_utilization"] = None
                if own_pids:
                    try:
                        samples = pynvml.nvmlDeviceGetProcessUtilization(
                            handle, int((time.time() - 2) * 1_000_000)
                        )
                        own = [s.smUtil for s in samples if s.pid in own_pids]
                        if own:
                            values["campaign_sm_utilization"] = sum(own) / len(own)
                    except pynvml.NVMLError:
                        pass
                row["gpus"].append(values)
                for key in (
                    "utilization",
                    "memory_used_gib",
                    "power_w",
                    "foreign_process_count",
                    "campaign_process_count",
                    "campaign_sm_utilization",
                ):
                    if values[key] is None:
                        continue
                    writer.add_scalar(f"GPU{index}/{key}", values[key], step)
            fh.write(json.dumps(row) + "\n")
            step += 1
            time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.monitor:
        monitor(args.root)
    else:
        while True:
            try:
                dashboard(args.root)
            except (ValueError, FileNotFoundError) as error:
                print(error)
            if args.once:
                break
            time.sleep(5)
