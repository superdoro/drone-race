"""Reward terms for 'get through the rooms to the goal without hitting anything, at a moderate speed'."""

import torch


def speed_cap(obstacle_clearance: torch.Tensor, rcfg: dict) -> torch.Tensor:
    """Allowed speed (N,): `max_speed` in the open, falling linearly to `slow_speed` as the clearance to the nearest wall
    or pillar shrinks below `slow_clearance` (0 disables the slow-down)."""
    if rcfg["slow_clearance"] <= 0.0:
        return torch.full_like(obstacle_clearance, rcfg["max_speed"])
    f = (obstacle_clearance / rcfg["slow_clearance"]).clamp(0.0, 1.0)
    return rcfg["slow_speed"] + (rcfg["max_speed"] - rcfg["slow_speed"]) * f


def compute_rewards(
    prev_remaining: torch.Tensor,
    remaining: torch.Tensor,
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    raw_actions: torch.Tensor,
    actions: torch.Tensor,
    last_actions: torch.Tensor,
    min_clearance: torch.Tensor,
    obstacle_clearance: torch.Tensor,
    obstacle_danger: torch.Tensor,
    wall_danger: torch.Tensor,
    crashed: torch.Tensor,
    reached: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rcfg = cfg["rewards"]
    margin = rcfg["proximity_margin"]
    terms = {
        # meters gained along the route (through doors and partition openings), so it never rewards pushing into a
        # wall; a potential difference, so flying back and forth through a door earns nothing
        "progress": rcfg["progress"] * (prev_remaining - remaining),
        # small constant cost per step so that hovering in place is not an option
        "time": torch.full_like(remaining, -rcfg["time"]),
        # soft speed limit, lower in clutter: being faster earns nothing, slowing down leaves time to steer
        "speed_limit": -rcfg["speed_limit"] * (lin_vel.norm(dim=-1) - speed_cap(obstacle_clearance, rcfg)).clamp_min(0.0).square(),
        "ang_vel": -rcfg["ang_vel"] * ang_vel.square().sum(dim=-1),
        # yaw is free in a heading-invariant task, so without this the policy spins continuously
        "yaw_rate": -rcfg["yaw_rate"] * ang_vel[:, 2].square(),
        "action_rate": -rcfg["action_rate"] * (actions - last_actions).square().sum(dim=-1),
        # actions are clipped to [-1, 1] by the env; penalize the overshoot so the Gaussian mean / std stay in range
        "action_bound": -rcfg["action_bound"] * (raw_actions.abs() - 1.0).clamp_min(0.0).square().sum(dim=-1),
        # shaped warning before an actual collision (clearance = gap between the drone body and the closest surface)
        "proximity": -rcfg["proximity"] * ((margin - min_clearance) / margin).clamp(0.0, 1.0),
        # closing speed toward a nearby pillar / free-standing wall (task/geometry.py: approach_danger); the progress
        # term pulls straight through these, so without it the only warning is the crash itself
        "approach": -rcfg["approach"] * obstacle_danger,
        # same for room walls and partitions, weaker: the route already leads through their openings, this only asks
        # to cross an opening square to it and at its height instead of clipping its edges
        "approach_wall": -rcfg["approach_wall"] * wall_danger,
        "crash": -rcfg["crash"] * crashed.float(),
        "goal": rcfg["goal"] * reached.float(),
    }
    return sum(terms.values()), terms
