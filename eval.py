"""Evaluate a trained policy: success / crash statistics, optional video or live viewer.

Usage:
    python eval.py --run logs/goal_nav/<timestamp>                      # latest checkpoint, hardest level
    python eval.py --run logs/goal_nav/<timestamp> --level 2 --record   # also save an mp4 of env 0
    python eval.py --run logs/goal_nav/<timestamp> --obstacles 8 8 --dist 10 12   # a specific setting
    python eval.py --run logs/goal_nav/<timestamp> --viewer --num_envs 1
"""

import argparse
import copy
import glob
import json
import os
import re

import torch
from rsl_rl.runners import OnPolicyRunner

from backends.base import make_backend
from config.drone_params import DRONES
from task.goal_nav_task import GoalNavTask


def latest_checkpoint(run_dir: str) -> str:
    ckpts = glob.glob(os.path.join(run_dir, "model_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No model_*.pt in {run_dir}")
    return max(ckpts, key=lambda p: int(re.findall(r"model_(\d+)\.pt", p)[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="training log directory")
    parser.add_argument("--ckpt", default=None, help="checkpoint file; defaults to the latest in --run")
    parser.add_argument("--level", type=int, default=-1, help="curriculum level to evaluate (-1 = hardest)")
    parser.add_argument("--obstacles", type=int, nargs=2, metavar=("MIN", "MAX"), help="override the obstacle count")
    parser.add_argument("--dist", type=float, nargs=2, metavar=("MIN", "MAX"), help="override the goal distance (m)")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--record", action="store_true", help="save an mp4 of env 0 into the run directory")
    parser.add_argument("--record_seconds", type=float, default=20.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(os.path.join(args.run, "cfgs.json")) as f:
        saved = json.load(f)
    env_cfg, train_cfg = saved["env_cfg"], saved["train_cfg"]
    levels = env_cfg["curriculum"]["levels"]
    level_idx = args.level % len(levels)
    level = dict(levels[level_idx])
    if args.obstacles:
        level["num_obstacles"] = args.obstacles
    if args.dist:
        level["goal_dist"] = args.dist
    levels[level_idx] = level
    env_cfg["curriculum"]["promote_success_rate"] = 2.0  # never change level during evaluation

    backend = make_backend(
        saved["args"]["backend"],
        DRONES[env_cfg["drone"]],
        env_cfg["sim_dt"],
        device=args.device,
        show_viewer=args.viewer,
        render_camera=args.record,
    )
    env = GoalNavTask(env_cfg, backend, args.num_envs)
    env.curriculum.level_idx = level_idx
    env.reset()
    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir=None, device=str(env.device))
    ckpt = args.ckpt or latest_checkpoint(args.run)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=str(env.device))
    (o_min, o_max), (d_min, d_max) = level["num_obstacles"], level["goal_dist"]
    setting = f"{o_min}-{o_max} obstacles, goal {d_min}-{d_max} m"
    print(f"Loaded {ckpt}; evaluating level {level_idx}: {setting}")

    video_path = None
    if args.record:
        video_path = os.path.join(args.run, f"eval_level{level_idx}_obs{o_min}-{o_max}_dist{d_min:g}-{d_max:g}.mp4")
        backend.start_recording(video_path, fps=int(round(1.0 / env.dt)))
    record_steps = int(args.record_seconds / env.dt)

    metrics: dict[str, list[torch.Tensor]] = {}
    finished, step = 0, 0
    obs = env.get_observations()
    with torch.inference_mode():
        while finished < args.episodes or (args.record and step < record_steps):
            obs, _, dones, extras = env.step(policy(obs))
            finished += int(dones.sum())
            step += 1
            for key, value in extras["log"].items():
                if key.startswith("Metrics/"):
                    metrics.setdefault(key, []).append(value.flatten())
            if args.record or args.viewer:
                backend.follow_camera(0, backend.get_drone_state().pos[0], env.corridor_dir[0])
            if args.record and step == record_steps:
                backend.stop_recording()

    print(f"\n=== {finished} episodes on level {level_idx} ({setting}) ===")
    for key, values in metrics.items():
        print(f"{key[len('Metrics/'):]:>18}: {torch.cat(values).mean().item():.3f}")
    if video_path:
        print(f"Video saved to {video_path}")


if __name__ == "__main__":
    main()
