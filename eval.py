"""Evaluate a trained policy: success / crash statistics, optional video, top-down plot or live viewer.

Usage:
    python eval.py --run logs/rooms/<timestamp>                       # latest checkpoint, hardest level
    python eval.py --run logs/rooms/<timestamp> --level 2 --record    # also save an mp4 of env 0
    python eval.py --run logs/rooms/<timestamp> --rooms 6 6 --pillars 5 5 --partitions 2 2   # a specific setting
    python eval.py --run logs/rooms/<timestamp> --plot 12             # top-down trajectories of 12 episodes
    python eval.py --run logs/rooms/<timestamp> --viewer --num_envs 1
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

LEVEL_KEYS = ("rooms", "partitions", "free_walls", "pillars", "door_width")


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
    parser.add_argument("--rooms", type=int, nargs=2, metavar=("MIN", "MAX"), help="override the room count")
    parser.add_argument("--partitions", type=int, nargs=2, metavar=("MIN", "MAX"), help="override partitions per room")
    parser.add_argument("--free_walls", type=int, nargs=2, metavar=("MIN", "MAX"), help="override free walls per room")
    parser.add_argument("--pillars", type=int, nargs=2, metavar=("MIN", "MAX"), help="override pillars per room")
    parser.add_argument("--door_width", type=float, nargs=2, metavar=("MIN", "MAX"), help="override door width (m)")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--record", action="store_true", help="save an mp4 of env 0 into the run directory")
    parser.add_argument("--record_seconds", type=float, default=30.0)
    parser.add_argument("--plot", type=int, default=0, metavar="N", help="save top-down trajectories of N episodes")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(os.path.join(args.run, "cfgs.json")) as f:
        saved = json.load(f)
    env_cfg, train_cfg = saved["env_cfg"], saved["train_cfg"]
    levels = env_cfg["curriculum"]["levels"]
    level_idx = args.level % len(levels)
    level = dict(levels[level_idx])
    for key in LEVEL_KEYS:
        if getattr(args, key):
            level[key] = getattr(args, key)
    levels[level_idx] = level
    env_cfg["curriculum"]["initial_level"] = level_idx
    env_cfg["curriculum"]["promote_success_rate"] = 2.0  # never change level during evaluation
    env_cfg["rooms"]["pool_size"] = min(env_cfg["rooms"]["pool_size"], max(args.episodes, args.num_envs))
    args.num_envs = max(args.num_envs, args.plot)

    backend = make_backend(
        saved["args"]["backend"],
        DRONES[env_cfg["drone"]],
        env_cfg["sim_dt"],
        device=args.device,
        show_viewer=args.viewer,
        render_camera=args.record,
    )
    env = GoalNavTask(env_cfg, backend, args.num_envs)
    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir=None, device=str(env.device))
    ckpt = args.ckpt or latest_checkpoint(args.run)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=str(env.device))
    setting = ", ".join(f"{key} {level[key][0]:g}-{level[key][1]:g}" for key in LEVEL_KEYS)
    print(f"Loaded {ckpt}; evaluating level {level_idx}: {setting}")

    video_path = None
    if args.record:
        video_path = os.path.join(args.run, f"eval_level{level_idx}_rooms{level['rooms'][0]}-{level['rooms'][1]}.mp4")
        backend.start_recording(video_path, fps=int(round(1.0 / env.dt)))
    record_steps = int(args.record_seconds / env.dt)

    # First episode of envs 0..plot-1, for the top-down plot (copied: the env overwrites a layout when it resets).
    plot_layouts = env.layout[torch.arange(args.plot, device=env.device)]
    plot_traj = [[] for _ in range(args.plot)]
    plot_outcome = [None] * args.plot

    metrics: dict[str, list[torch.Tensor]] = {}
    finished, step = 0, 0
    obs = env.get_observations()
    with torch.inference_mode():
        while finished < args.episodes or (args.record and step < record_steps) or None in plot_outcome:
            pos = backend.get_drone_state().pos
            for i in range(args.plot):
                if plot_outcome[i] is None:
                    plot_traj[i].append(pos[i, :2].tolist())
            obs, _, dones, extras = env.step(policy(obs))
            finished += int(dones.sum())
            step += 1
            log = extras["log"]
            for key, value in log.items():
                if key.startswith("Metrics/"):
                    metrics.setdefault(key, []).append(value.flatten())
            if args.plot and dones[: args.plot].any():
                ended = dones.nonzero().flatten()
                for row, i in enumerate(ended.tolist()):
                    if i < args.plot and plot_outcome[i] is None:
                        plot_outcome[i] = next(
                            name for name, key in (("success", "success_rate"), ("obstacle", "crash_obstacle"),
                                                   ("wall", "crash_wall"), ("out", "out_of_bounds"), ("timeout", "timeout"))
                            if log[f"Metrics/{key}"][row] > 0
                        )  # fmt: skip
            if args.record or args.viewer:
                backend.follow_camera(0, backend.get_drone_state().pos[0])
            if args.record and step == record_steps:
                backend.stop_recording()

    print(f"\n=== {finished} episodes on level {level_idx} ({setting}) ===")
    for key, values in metrics.items():
        print(f"{key[len('Metrics/'):]:>18}: {torch.cat(values).mean().item():.3f}")
    if video_path:
        print(f"Video saved to {video_path}")
    if args.plot:
        from plot_layouts import save_grid

        path = os.path.join(args.run, f"eval_level{level_idx}_topdown.png")
        items = [
            {"lay": plot_layouts[i], "radii": env.pillar_radii, "traj": torch.tensor(plot_traj[i]),
             "outcome": plot_outcome[i], "label": f"env {i}: {plot_outcome[i]}"}
            for i in range(args.plot)
        ]  # fmt: skip
        save_grid(items, path, f"level {level_idx}: {setting}")
        print(f"Top-down plot saved to {path}")


if __name__ == "__main__":
    main()
