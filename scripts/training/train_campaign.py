"""Local campaign harness: native learners, TensorBoard, recoverable checkpoints.

Resume preserves models/optimizers/replay/sampling estimates, but resets simulator
episodes. It is a statistical continuation, not bitwise simulator restoration.
"""

import argparse
import dataclasses
import fcntl
import json
import os
import random
import signal
import statistics
import subprocess
import time
from collections import deque
from pathlib import Path

import mjlab.tasks  # noqa: F401
import numpy as np
import supermimic.tasks  # noqa: F401
import torch
import wandb
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.random import seed_rng
from supermimic.runners.supermimic_runner import (
    Ratio,
    SuperMimicRunner,
    SuperMimicRunnerCfg,
)
from supermimic.scripts.train_utils import create_env, setup_device
from supermimic.utils.reproduction_paths import (
    REPO_ROOT,
    portable_tree,
    repo_path,
    resolve_motion_path,
    same_motion_source,
)
from torch.utils.tensorboard import SummaryWriter


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(portable_tree(value), indent=2, default=str) + "\n")
    tmp.replace(path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def atomic_save(path, value):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


class CampaignRunner(SuperMimicRunner):
    def optimizer_groups(self):
        return {
            "wm": self.world_model_learner.optimizers,
            "actor": self.policy_learner.actor_optimizers,
            "critic": self.policy_learner.critic_optimizers,
        }

    def checkpoint(self, step, ratio, full=False):
        started = time.monotonic()
        # Serialize disk-heavy saves across eight processes. Old files remain valid
        # until atomic replacement completes, including in case of SIGKILL.
        with (self.out.parent / ".checkpoint.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            dest = self.out / "checkpoints" / f"{step:08d}"
            dest.mkdir(parents=True, exist_ok=True)
            atomic_save(dest / "world_model.pt", cpu_tree(self._wm.bundle()))
            atomic_save(
                dest / "actor.pt", cpu_tree(self.policy_learner.actor.state_dict())
            )
            atomic_save(
                dest / "critic.pt", cpu_tree(self.policy_learner.critic.state_dict())
            )
            atomic_json(
                dest / "metadata.json",
                {"completed_iterations": step, "updates": self.updates},
            )
            latest = self.out / "latest.tmp"
            latest.unlink(missing_ok=True)
            latest.symlink_to(dest.relative_to(self.out), target_is_directory=True)
            latest.replace(self.out / "latest")
            if full:
                buf = self.replay_buffer
                sampler = self._sampler
                state = {
                    "format": 1,
                    "step": step,
                    "updates": self.updates,
                    "config": self.saved_config,
                    "world_model": cpu_tree(self._wm.bundle()),
                    "actor": cpu_tree(self.policy_learner.actor.state_dict()),
                    "critic": cpu_tree(self.policy_learner.critic.state_dict()),
                    "optimizers": {
                        k: [cpu_tree(o.state_dict()) for o in opts]
                        for k, opts in self.optimizer_groups().items()
                    },
                    "replay": {k: cpu_tree(v) for k, v in buf._storage.items()},
                    "replay_step": buf._step,
                    "replay_size": buf.num_transitions,
                    "ratio_prev": ratio._prev,
                    "sampler": {
                        k: cpu_tree(getattr(sampler, k))
                        for k in ("value", "count", "effort")
                    }
                    if sampler is not None
                    else None,
                    "rng_python": random.getstate(),
                    "rng_numpy": np.random.get_state(),
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state(),
                    "resume_semantics": "optimizer/replay continuation with simulator episode reset",
                }
                atomic_save(self.out / "resume.pt", state)
                del state
        self.writer.add_scalar(
            "Perf/checkpoint_seconds", time.monotonic() - started, step
        )
        self.writer.flush()

    def restore(self, path, ratio):
        state = torch.load(path, map_location="cpu", weights_only=False)
        old = state["config"]
        for key in ("task_id", "runner_cfg"):
            if key == "task_id" and old[key] != self.saved_config[key]:
                raise ValueError("Resume task mismatch")
        # Validate training-relevant shapes/settings; only budget/name/log cadence
        # may change on continuation.
        old_cfg, new_cfg = (
            portable_tree(old["runner_cfg"]),
            portable_tree(self.saved_config["runner_cfg"]),
        )
        for key in ("max_iterations", "run_name", "log_interval", "save_interval"):
            old_cfg.pop(key, None)
            new_cfg.pop(key, None)
        if (
            old_cfg != new_cfg
            or old["args"]["num_envs"] != self.saved_config["args"]["num_envs"]
        ):
            raise ValueError("Resume configuration mismatch")
        if not same_motion_source(
            old["args"]["motion_file"], self.saved_config["args"]["motion_file"]
        ):
            raise ValueError("Resume dataset mismatch")
        self._wm.load_bundle(state["world_model"])
        self.policy_learner.actor.load_state_dict(state["actor"])
        self.policy_learner.critic.load_state_dict(state["critic"])
        for key, opts in self.optimizer_groups().items():
            for opt, saved in zip(opts, state["optimizers"][key], strict=True):
                opt.load_state_dict(saved)
        buf = self.replay_buffer
        for key, tensor in state["replay"].items():
            buf._storage[key].copy_(tensor)
        buf._step, buf.num_transitions = state["replay_step"], state["replay_size"]
        # Reset simulator episodes cannot be joined to pre-resume replay windows.
        buf._storage["truncated"][:, (buf._step - 1) % buf.buffer_size] = True
        ratio._prev = state["ratio_prev"]
        if self._sampler is not None and state["sampler"] is not None:
            for key, tensor in state["sampler"].items():
                getattr(self._sampler, key).copy_(tensor)
            self.motion_cmd.set_sampling_distribution(self._sampler._distribution())
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"])
        torch.cuda.set_rng_state(state["rng_cuda"])
        self.updates = state["updates"]
        return state["step"]

    def run_campaign(self, out, config, resume=None):
        self.out, self.saved_config = out, config
        self.writer = SummaryWriter(str(out / "tensorboard"), flush_secs=20)
        self._create_log_dir = lambda: out
        self._init_wandb(out)
        self.env.reset()
        self._seed_sim_history()
        ratio = Ratio(
            self.env.num_envs
            / self.cfg.wm.num_batches
            / self.cfg.wm.batch_size
            / self.cfg.wm.horizon
            * self.cfg.replay_ratio
        )
        self.updates = 0
        step = self.restore(resume, ratio) if resume else 0
        self.completed = step
        stop = False

        def request_stop(*_):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        history = deque(maxlen=200)
        total_start = time.monotonic()
        last_log = time.monotonic()
        wm_time = pi_time = 0.0
        transitions = 0
        losses = {}
        status = "running"
        original_log = wandb.log

        def log_metrics(row, **kwargs):
            scalars = {k: float(v) for k, v in row.items()}
            if not all(np.isfinite(v) for v in scalars.values()):
                raise FloatingPointError(
                    f"Nonfinite metrics at {self.completed}: {scalars}"
                )
            for key, value in scalars.items():
                self.writer.add_scalar(key, value, self.completed)
            with (out / "metrics.jsonl").open("a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "iteration": self.completed,
                            "updates": self.updates,
                            "time": time.time(),
                            **scalars,
                        }
                    )
                    + "\n"
                )
            original_log(row, **kwargs)

        wandb.log = log_metrics
        try:
            while step < self.cfg.max_iterations and not stop:
                tic = time.monotonic()
                transitions += self.step()
                for _ in range(ratio(step)):
                    if self.replay_buffer.num_transitions < self.cfg.wm.horizon:
                        continue
                    data = (
                        self.replay_buffer.sample(
                            self.cfg.wm.horizon, self.cfg.wm.batch_size
                        )
                        for _ in range(self.cfg.wm.num_batches)
                    )
                    t = time.monotonic()
                    losses.update(self.world_model_learner.update(data))
                    wm_time += time.monotonic() - t
                    t = time.monotonic()
                    losses.update(self.policy_learner.update())
                    pi_time += time.monotonic() - t
                    self.updates += 1
                step += 1
                self.completed = step
                # Exclude compilation/CUDA graph warmup from the moving ETA.
                if self.updates >= 6:
                    history.append(time.monotonic() - tic)
                if step % self.cfg.log_interval == 0 or step == self.cfg.max_iterations:
                    self.logger.log(
                        it=step // self.cfg.log_interval,
                        total_it=self.cfg.max_iterations // self.cfg.log_interval,
                        collect_time=max(
                            0, time.monotonic() - last_log - wm_time - pi_time
                        ),
                        wm_learn_time=wm_time,
                        policy_learn_time=pi_time,
                        loss_dict=losses,
                        num_collected_steps=transitions,
                    )
                    mean = statistics.mean(history) if history else None
                    atomic_json(
                        out / "status.json",
                        {
                            "status": "running",
                            "pid": os.getpid(),
                            "time": time.time(),
                            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "iteration": step,
                            "max_iterations": self.cfg.max_iterations,
                            "transitions": step * self.env.num_envs,
                            "updates": self.updates,
                            "seconds_per_iteration": mean,
                            "eta_hours": mean * (self.cfg.max_iterations - step) / 3600
                            if mean
                            else None,
                            "elapsed_hours": (time.monotonic() - total_start) / 3600,
                            "torch_allocated_gib": torch.cuda.memory_allocated()
                            / 2**30,
                            "torch_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                            "wm_graph": self.world_model_learner._captured_update._graph
                            is not None,
                            "policy_graph": self.policy_learner._captured_update._graph
                            is not None,
                        },
                    )
                    self.writer.flush()
                    last_log = time.monotonic()
                    wm_time = pi_time = 0.0
                    transitions = 0
                if step % self.cfg.save_interval == 0:
                    self.checkpoint(
                        step, ratio, full=step % config["args"]["resume_interval"] == 0
                    )
                    last_log = time.monotonic()
            self.checkpoint(step, ratio, full=True)
            status = "interrupted" if stop else "completed"
        except BaseException as exc:
            atomic_json(
                out / "error.json",
                {"iteration": step, "error": repr(exc), "time": time.time()},
            )
            status = "failed"
            raise
        finally:
            wandb.log = original_log
            self.writer.close()
            status_path = out / "status.json"
            value = json.loads(status_path.read_text()) if status_path.exists() else {}
            atomic_json(
                status_path,
                value | {"status": status, "iteration": step, "time": time.time()},
            )
            self._cleanup()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--motion-file", required=True)
    p.add_argument(
        "--representation", choices=["gencoord", "maxcoord"], default="gencoord"
    )
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--wm-batch", type=int, default=2048)
    p.add_argument("--policy-batch", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=320000)
    p.add_argument("--save-interval", type=int, default=5000)
    p.add_argument("--resume-interval", type=int, default=10000)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    args.out = repo_path(args.out)
    args.motion_file = str(resolve_motion_path(args.motion_file))
    if args.resume is not None:
        args.resume = repo_path(args.resume)
    if args.resume_interval % args.save_interval:
        p.error("resume-interval must be a multiple of save-interval")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "config.json").exists() and args.resume is None:
        p.error("Existing run: use --resume or a new --out directory")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    seed_rng(42)
    cfg = SuperMimicRunnerCfg()
    cfg.representation = args.representation
    cfg.actor_obs, cfg.history_length = "proprio", 10
    cfg.max_iterations = args.iterations
    cfg.run_name = args.out.name
    cfg.save_interval, cfg.log_interval = args.save_interval, args.log_interval
    cfg.wm.batch_size, cfg.policy.batch_size = args.wm_batch, args.policy_batch
    task_id = (
        "SuperMimic-"
        + ("GenCoord" if args.representation == "gencoord" else "MaxCoord")
        + "-G1"
    )
    config = {
        "task_id": task_id,
        "seed": 42,
        "runner_cfg": dataclasses.asdict(cfg),
        "args": vars(args),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT
        ).strip(),
        "torch": torch.__version__,
        "start_time": time.time(),
        "protocol": "current official source; BONES unfiltered where applicable; no paper-equivalence claim",
    }
    config = portable_tree(config)
    atomic_json(
        args.out / ("config_resume.json" if args.resume else "config.json"), config
    )
    device = setup_device("cuda:0")
    atomic_json(
        args.out / "status.json",
        {
            "status": "initializing",
            "pid": os.getpid(),
            "time": time.time(),
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "iteration": 0,
            "max_iterations": args.iterations,
        },
    )
    env_cfg = load_env_cfg(task_id, play=False)
    env_cfg.seed = 42
    env_cfg.sim.nan_guard.output_dir = str(repo_path("reproduction/runtime/nan_dumps"))
    env_cfg.commands["motion"].motion_file = args.motion_file
    env_cfg.commands["motion"].sampling_mode = "external"
    env_cfg.scene.num_envs = args.num_envs
    atomic_json(args.out / "env_config.json", dataclasses.asdict(env_cfg))
    motion_path = Path(args.motion_file)
    for manifest in (
        motion_path / "verified.json",
        motion_path.parent / "manifest_converted.json",
    ):
        if manifest.is_file():
            atomic_json(
                args.out / "dataset_manifest.json", json.loads(manifest.read_text())
            )
            break
    env = create_env(env_cfg, args.num_envs, device)
    runner = CampaignRunner(env, cfg, task_id, registry_name=None, device=device)
    runner.run_campaign(args.out, config, args.resume)


if __name__ == "__main__":
    main()
