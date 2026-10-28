"""Reward terms for 'reach the goal as fast as possible without hitting anything'."""

import torch


def compute_rewards(
    prev_dist: torch.Tensor,
    dist: torch.Tensor,
    ang_vel: torch.Tensor,
    raw_actions: torch.Tensor,
    actions: torch.Tensor,
    last_actions: torch.Tensor,
    min_clearance: torch.Tensor,
    crashed: torch.Tensor,
    reached: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rcfg = cfg["rewards"]
    margin = rcfg["proximity_margin"]
    terms = {
        # meters gained toward the goal this step (same idea as the gate-progress reward of Swift, Nature 2023)
        "progress": rcfg["progress"] * (prev_dist - dist),
        # constant cost per step: together with the discount it makes faster arrival strictly better
        "time": torch.full_like(dist, -rcfg["time"]),
        "ang_vel": -rcfg["ang_vel"] * ang_vel.square().sum(dim=-1),
        # yaw is free in a heading-invariant task, so without this the policy spins continuously
        "yaw_rate": -rcfg["yaw_rate"] * ang_vel[:, 2].square(),
        "action_rate": -rcfg["action_rate"] * (actions - last_actions).square().sum(dim=-1),
        # actions are clipped to [-1, 1] by the env; penalize the overshoot so the Gaussian mean / std stay in range
        "action_bound": -rcfg["action_bound"] * (raw_actions.abs() - 1.0).clamp_min(0.0).square().sum(dim=-1),
        # shaped warning before an actual collision (clearance = gap between the drone body and the closest surface)
        "proximity": -rcfg["proximity"] * ((margin - min_clearance) / margin).clamp(0.0, 1.0),
        "crash": -rcfg["crash"] * crashed.float(),
        "goal": rcfg["goal"] * reached.float(),
    }
    return sum(terms.values()), terms
