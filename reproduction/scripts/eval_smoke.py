"""Reload local official checkpoints and exercise headless policy evaluation.

The output is an integration / throughput check of a briefly updated policy,
not a motion-tracking accuracy result or a paper benchmark.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import mjlab.tasks  # noqa: E402,F401
import supermimic.tasks  # noqa: E402,F401
import torch  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab.utils.random import seed_rng  # noqa: E402
from mjlab.viewer.offscreen_renderer import OffscreenRenderer  # noqa: E402
from mjlab.viewer.viewer_config import ViewerConfig  # noqa: E402
from PIL import Image  # noqa: E402
from supermimic.representations import extract_actor_goal  # noqa: E402
from supermimic.runners.run_config import load_runner_cfg  # noqa: E402
from supermimic.runners.supermimic_runner import build_actor, build_world_model  # noqa: E402
from supermimic.scripts.train_utils import create_env, setup_device  # noqa: E402
from supermimic.tasks.task_spec import build_actor_obs_td  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--render", action="store_true")
    p.add_argument("--motion-file")
    p.add_argument("--output-name")
    args = p.parse_args()
    if not 40 <= args.steps <= 1000:
        p.error("Evaluation preparation checks must be 40–1000 steps.")
    torch.set_num_threads(4)
    seed_rng(2001)
    device = setup_device("cuda:0")
    config = json.loads((args.run_dir / "config.json").read_text())
    cfg = load_runner_cfg(config)
    env_cfg = load_env_cfg(config["task_id"], play=False)
    motion_file = args.motion_file or config["args"]["motion_file"]
    env_cfg.commands["motion"].motion_file = motion_file
    env_cfg.commands["motion"].sampling_mode = "start"
    env = create_env(env_cfg, args.num_envs, device)
    wm = build_world_model(cfg, env, device)
    wm_path = args.run_dir / "final/world_model.pt"
    actor_path = args.run_dir / "final/actor.pt"
    wm.load_bundle(torch.load(wm_path, map_location=device, weights_only=True))
    actor = build_actor(cfg, wm, device)
    actor.load_state_dict(
        torch.load(actor_path, map_location=device, weights_only=True)
    )
    actor.eval()
    wm.dynamics.eval()
    history = None
    failure_count = 0
    max_abs_action = 0.0
    renderer = None
    if args.render:
        viewer_cfg = ViewerConfig(
            width=640,
            height=480,
            distance=2.4,
            entity_name="robot",
            origin_type=ViewerConfig.OriginType.ASSET_ROOT,
        )
        renderer = OffscreenRenderer(
            env.unwrapped.sim.mj_model,
            viewer_cfg,
            env.unwrapped.scene,
            sim_model=env.unwrapped.sim.model,
            expanded_fields=env.unwrapped.sim.expanded_fields,
        )
        renderer.initialize()
    obs, _ = env.reset()
    measured_start = None
    with torch.inference_mode():
        for step in range(args.steps):
            if step == 20:
                torch.cuda.synchronize()
                measured_start = time.perf_counter()
            state = obs["full_state"]
            rep = wm.representation.represent(state)
            ctx = wm.sim_context.read(env.unwrapped)
            sensor = wm.representation.compute_actor_sensor(
                rep, ctx, sim_actor_obs=obs["actor"]
            )
            if history is None:
                history = sensor.unsqueeze(1).repeat(1, cfg.history_length, 1)
            else:
                history = torch.cat([history[:, 1:], sensor.unsqueeze(1)], dim=1)
            goal = extract_actor_goal(obs["goal"])
            actions = actor(build_actor_obs_td({"motion": goal}, history))
            if not torch.isfinite(actions).all() or not torch.isfinite(state).all():
                raise FloatingPointError("Nonfinite policy action or simulation state")
            max_abs_action = max(max_abs_action, float(actions.abs().max()))
            obs, reward, dones, extras = env.step(actions)
            failure_count += int(env.unwrapped.termination_manager.terminated.sum())
            if dones.any():
                # Reset padding repeats one fresh sensor realization.
                new_state = obs["full_state"]
                new_rep = wm.representation.represent(new_state)
                new_ctx = wm.sim_context.read(env.unwrapped)
                reset_sensor = wm.representation.compute_actor_sensor(
                    new_rep, new_ctx, sim_actor_obs=obs["actor"]
                )
                history = torch.where(
                    dones.bool().view(-1, 1, 1),
                    reset_sensor.unsqueeze(1).expand(-1, cfg.history_length, -1),
                    history,
                )
            if renderer and step == 1:
                renderer.update(env.unwrapped.sim.data)
                frame = renderer.render()
                if frame.std() < 1:
                    raise AssertionError("EGL renderer produced a blank frame")
                Image.fromarray(frame).save(args.run_dir / "headless_smoke.png")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - measured_start
    result = {
        "status": "passed",
        "preparation_only": True,
        "task": config["task_id"],
        "motion_file": motion_file,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "num_envs": args.num_envs,
        "steps": args.steps,
        "warmup_steps": 20,
        "measured_seconds": elapsed,
        "transitions_per_second": (args.steps - 20) * args.num_envs / elapsed,
        "failures_of_untrained_smoke_policy": failure_count,
        "max_abs_action": max_abs_action,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "finite_actions_and_state": True,
        "headless_render": bool(renderer),
        "checkpoint_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [wm_path, actor_path]
        },
        "caveat": "Short integration probe only, with existing unrelated GPU jobs; not benchmark accuracy or exclusive GPU throughput.",
    }
    (args.run_dir / (args.output_name or f"eval_{args.num_envs}.json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)
    if renderer:
        renderer.close()
    env.close()


if __name__ == "__main__":
    main()
