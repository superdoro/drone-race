"""Train the goal-navigation policy with rsl-rl PPO.

Usage:
    python train.py --num_envs 4096 --max_iterations 3000 --exp_name rooms
    python train.py --resume logs/rooms/<timestamp>/model_3000.pt --level 3   # continue from a curriculum level
    CUDA_VISIBLE_DEVICES=1 python train.py ...   # pick the GPU on a multi-GPU machine
"""

import argparse
import copy
import json
import os
from datetime import datetime

import torch
from rsl_rl.runners import OnPolicyRunner

from backends.base import make_backend
from config.drone_params import DRONES
from config.task_cfg import get_cfgs
from task.goal_nav_task import GoalNavTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="genesis")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=1500)
    parser.add_argument("--exp_name", default="rooms")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", default=None, help="checkpoint (.pt) to continue from")
    parser.add_argument("--level", type=int, default=0, help="curriculum level to start at")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    env_cfg, train_cfg = get_cfgs()
    env_cfg["curriculum"]["initial_level"] = args.level
    log_dir = os.path.join("logs", args.exp_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "cfgs.json"), "w") as f:
        json.dump({"env_cfg": env_cfg, "train_cfg": train_cfg, "args": vars(args)}, f, indent=2)

    backend = make_backend(args.backend, DRONES[env_cfg["drone"]], env_cfg["sim_dt"], device=args.device, seed=args.seed)
    env = GoalNavTask(env_cfg, backend, args.num_envs)
    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir, device=str(env.device))
    if args.resume:
        runner.load(args.resume)
    print(f"Logging to {log_dir}")
    runner.learn(num_learning_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
