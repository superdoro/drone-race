"""Sanity checks for the task + backend: shapes / NaNs, hover, collision detection and frame conventions.

Usage: python check_env.py [--backend genesis] [--num_envs 64]
"""

import argparse
import copy

import torch

from backends.base import make_backend
from config.drone_params import DRONES
from config.task_cfg import get_cfgs
from task.goal_nav_task import GoalNavTask


def make_task(cfg, args):
    backend = make_backend(args.backend, DRONES[cfg["drone"]], cfg["sim_dt"], device=args.device)
    return GoalNavTask(cfg, backend, args.num_envs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="genesis")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_envs", type=int, default=64)
    args = parser.parse_args()

    base_cfg, _ = get_cfgs()
    cfg = copy.deepcopy(base_cfg)
    cfg["init"]["max_tilt_deg"] = 0.0
    cfg["curriculum"]["levels"] = [{"num_obstacles": (4, 4), "goal_dist": (6.0, 8.0)}]
    task = make_task(cfg, args)
    n = task.num_envs

    # 1) Observation / reward shapes and finiteness under random actions.
    obs = task.get_observations()["policy"]
    print(f"[1] obs dim = {obs.shape[1]}, num_actions = {task.num_actions}")
    for _ in range(100):
        obs, rew, dones, extras = task.step(2.0 * torch.rand(n, 4, device=task.device) - 1.0)
        assert obs["policy"].shape == (n, obs["policy"].shape[1]) and rew.shape == (n,) and dones.shape == (n,)
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rew).all()
    print("[1] random actions: shapes OK, no NaN/Inf")

    # 2) Zero action = hover thrust + zero body rates: the drone should stay put for 2 s.
    task.reset()
    start = task.backend.get_drone_state().pos.clone()
    for _ in range(int(2.0 / task.dt)):
        task.step(torch.zeros(n, 4, device=task.device))
    drift = (task.backend.get_drone_state().pos - start).norm(dim=-1)
    print(f"[2] hover drift after 2 s: max {drift.max():.4f} m")
    assert drift.max() < 0.05

    # 3) Body-rate tracking: a +roll-rate command must produce +x body angular velocity, same for pitch / yaw.
    for axis, name in ((1, "roll"), (2, "pitch"), (3, "yaw")):
        task.reset()
        a = torch.zeros(n, 4, device=task.device)
        a[:, axis] = 0.25
        for _ in range(10):
            task.step(a)
        w = task.backend.get_drone_state().ang_vel.mean(dim=0)
        target = 0.25 * task.controller.max_body_rates[axis - 1]
        print(f"[3] {name}-rate cmd {target:.2f} rad/s -> body ang vel {w.tolist()}")
        assert abs(w[axis - 1] - target) < 0.2 * target

    # 4) An obstacle placed on top of the drone must end the episode with a crash.
    task.reset()
    pos = task.backend.get_drone_state().pos
    task.obs_xy[:, 0] = pos[:, :2]
    task.obs_active[:, 0] = True
    _, rew, dones, extras = task.step(torch.zeros(n, 4, device=task.device))
    print(f"[4] obstacle overlap -> done {dones.float().mean():.2f}, crash rate {extras['log']['Metrics/crash_obstacle'].mean():.2f}")
    assert dones.all() and (extras["log"]["Metrics/crash_obstacle"] == 1).all()

    print("All checks passed.")


if __name__ == "__main__":
    main()
