"""Bounded official-runner smoke tests / timing probes; never a full campaign."""

import argparse
import dataclasses
import hashlib
import json
import os
import statistics
import threading
import time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")

import mjlab.tasks  # noqa: E402,F401
import supermimic.tasks  # noqa: E402,F401
import torch  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab.utils.random import seed_rng  # noqa: E402
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg  # noqa: E402
from supermimic.scripts.train_utils import create_env, setup_device  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--representation", choices=["gencoord", "maxcoord"], default="gencoord"
    )
    p.add_argument("--name", required=True)
    p.add_argument("--iterations", type=int, default=48)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--wm-batch", type=int, default=128)
    p.add_argument("--wm-batches", type=int, default=2)
    p.add_argument("--policy-batch", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=1000)
    p.add_argument("--history", type=int, default=10)
    p.add_argument(
        "--motion-file", default=str(ROOT / "data/lafan1/npz/run1_subject5.npz")
    )
    args = p.parse_args()
    if not 16 <= args.iterations <= 160:
        p.error("Only bounded 16–160 iteration preparation probes are allowed.")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    seed_rng(42)
    device = setup_device("cuda:0")
    task_id = (
        "SuperMimic-"
        + ("GenCoord" if args.representation == "gencoord" else "MaxCoord")
        + "-G1"
    )
    cfg = SuperMimicRunnerCfg()
    cfg.representation = args.representation
    cfg.actor_obs = "proprio"
    cfg.history_length = args.history
    cfg.max_iterations = args.iterations
    cfg.buffer_size = args.buffer_size
    cfg.run_name = "PREPARATION_ONLY_" + args.name
    cfg.log_interval = 4
    # Exercise the native save/export at step 0; save updated final bundles below.
    # Avoid periodic export overhead contaminating the short timing window.
    cfg.save_interval = args.iterations + 1
    cfg.wm.batch_size = args.wm_batch
    cfg.wm.num_batches = args.wm_batches
    cfg.policy.batch_size = args.policy_batch
    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(
        out / "config.json",
        {"task_id": task_id, "runner_cfg": dataclasses.asdict(cfg), "args": vars(args)},
    )
    started = time.perf_counter()
    env_cfg = load_env_cfg(task_id, play=False)
    env_cfg.seed = 42
    env_cfg.commands["motion"].motion_file = args.motion_file
    env_cfg.commands["motion"].sampling_mode = "external"
    env = create_env(env_cfg, args.num_envs, device)
    runner = SuperMimicRunner(env, cfg, task_id, registry_name=None, device=device)
    runner._create_log_dir = lambda *parts: out
    init_seconds = time.perf_counter() - started
    wm = runner.world_model_learner
    policy = runner.policy_learner
    before = {
        "wm": next(runner._dynamics.parameters()).detach().clone(),
        "actor": next(policy.actor.parameters()).detach().clone(),
    }
    phase_rows = []
    state = {"step": -1, "updates": 0}
    intervals = []
    last_boundary = None

    original_step = runner.step

    def timed_step():
        nonlocal last_boundary
        now = time.perf_counter()
        if last_boundary is not None:
            intervals.append({"step": state["step"], "seconds": now - last_boundary})
        last_boundary = now
        state["step"] += 1
        return original_step()

    runner.step = timed_step

    for name, learner in [("wm", wm), ("policy", policy)]:
        original_update = learner.update

        def timed_update(*a, _name=name, _fn=original_update, **kw):
            t = time.perf_counter()
            logs = _fn(*a, **kw)
            torch.cuda.synchronize()
            secs = time.perf_counter() - t
            values = {k: float(v) for k, v in logs.items()}
            if not all(torch.isfinite(torch.tensor(v)) for v in values.values()):
                raise FloatingPointError(f"Nonfinite {_name} diagnostics: {values}")
            if _name == "policy":
                state["updates"] += 1
            row = {
                "phase": _name,
                "step": state["step"],
                "seconds": secs,
                "stats": values,
            }
            phase_rows.append(row)
            with (out / "phases.jsonl").open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"[PROBE] {_name} step={state['step']} time={secs:.4f}s", flush=True)
            return logs

        learner.update = timed_update

    samples = []
    stop = threading.Event()

    def monitor():
        try:
            import pynvml

            pynvml.nvmlInit()
            handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(pynvml.nvmlDeviceGetCount())
            ]
            while not stop.wait(1):
                row = {
                    "elapsed": time.perf_counter() - started,
                    "step": state["step"],
                    "gpus": [],
                }
                for i, handle in enumerate(handles):
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    row["gpus"].append(
                        {"index": i, "gpu_util": util.gpu, "used_bytes": mem.used}
                    )
                samples.append(row)
        except Exception as e:
            print(f"[MONITOR] {type(e).__name__}: {e}", flush=True)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    train_start = time.perf_counter()
    try:
        runner.run()  # The unmodified authors' collection / update / checkpoint loop.
        torch.cuda.synchronize()
        loop_seconds = time.perf_counter() - train_start
        intervals.append(
            {"step": state["step"], "seconds": time.perf_counter() - last_boundary}
        )
        final = out / "final"
        final.mkdir(exist_ok=True)
        torch.save(runner._wm.bundle(), final / "world_model.pt")
        torch.save(policy.actor.state_dict(), final / "actor.pt")
        if hasattr(policy, "critic"):
            torch.save(policy.critic.state_dict(), final / "critic.pt")
        actor_state = torch.load(
            final / "actor.pt", map_location=device, weights_only=True
        )
        policy.actor.load_state_dict(actor_state)
        runner._wm.load_bundle(
            torch.load(final / "world_model.pt", map_location=device, weights_only=True)
        )
        changed = {
            "wm": bool((before["wm"] != next(runner._dynamics.parameters())).any()),
            "actor": bool((before["actor"] != next(policy.actor.parameters())).any()),
        }
        if not all(changed.values()):
            raise AssertionError(f"Parameters did not update: {changed}")
        if wm._captured_update._graph is None or policy._captured_update._graph is None:
            raise AssertionError(
                "The smoke budget did not exercise both CUDA graph captures."
            )
        # After 4 full learner updates, compilation/capture has completed.
        pol_rows = [r for r in phase_rows if r["phase"] == "policy"]
        cutoff = pol_rows[4]["step"] if len(pol_rows) > 4 else args.iterations
        steady = [r["seconds"] for r in intervals if r["step"] > cutoff]
        summary = {
            "status": "passed",
            "preparation_only": True,
            "task": task_id,
            "seed": 42,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "init_seconds": init_seconds,
            "loop_seconds_including_compile": loop_seconds,
            "learner_updates": state["updates"],
            "warmup_last_step": cutoff,
            "steady_steps": len(steady),
            "steady_seconds_per_iteration": statistics.mean(steady) if steady else None,
            "steady_median_seconds": statistics.median(steady) if steady else None,
            "parameters_changed": changed,
            "wm_graph": True,
            "policy_graph": True,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "parameters": {
                "wm": sum(p.numel() for p in runner._dynamics.parameters()),
                "actor": sum(p.numel() for p in policy.actor.parameters()),
            },
            "checkpoint_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in final.glob("*.pt")
            },
            "config": dataclasses.asdict(cfg),
            "performance_caveat": "Whole-device NVML samples do not identify process utilization. Verify competing processes throughout the probe using the campaign resource record before attributing utilization or using timings as exclusive measurements.",
        }
        atomic_json(out / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        stop.set()
        thread.join(timeout=3)
        atomic_json(out / "gpu_samples.json", samples)
        atomic_json(out / "iteration_times.json", intervals)


if __name__ == "__main__":
    main()
