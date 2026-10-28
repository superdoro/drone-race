"""Random start / goal / obstacle layouts and the success-driven curriculum. Pure torch."""

import math

import torch

from utils.math_utils import quat_from_euler


def _uniform(n: int, low: float, high: float, device) -> torch.Tensor:
    return low + (high - low) * torch.rand(n, device=device)


def sample_start_goal(n: int, cfg: dict, level: dict, device):
    """Start near the origin with a small random tilt and any heading; goal at a random horizontal bearing."""
    z_low, z_high = cfg["arena"]["spawn_z"]
    start = torch.zeros(n, 3, device=device)
    start[:, 2] = _uniform(n, z_low, z_high, device)

    dist = _uniform(n, *level["goal_dist"], device)
    bearing = _uniform(n, -math.pi, math.pi, device)
    goal = torch.stack(
        [dist * torch.cos(bearing), dist * torch.sin(bearing), _uniform(n, z_low, z_high, device)], dim=-1
    )

    max_tilt = math.radians(cfg["init"]["max_tilt_deg"])
    quat = quat_from_euler(
        _uniform(n, -max_tilt, max_tilt, device),
        _uniform(n, -max_tilt, max_tilt, device),
        _uniform(n, -math.pi, math.pi, device),
    )
    return start, goal, quat


def sample_num_obstacles(n: int, level: dict, device) -> torch.Tensor:
    """Per-env obstacle count, uniform in the level's (min, max) so easier layouts stay in the training mix."""
    low, high = level["num_obstacles"]
    return torch.randint(low, high + 1, (n,), device=device)


def sample_obstacles(start: torch.Tensor, goal: torch.Tensor, radii: torch.Tensor, num_active: torch.Tensor, cfg: dict):
    """Scatter `num_active` (n,) cylinders around the start->goal segment so the straight line is likely blocked.

    Slots beyond `num_active`, or too close to the start or the goal, are deactivated.
    Returns xy (n, K, 2) and active (n, K).
    """
    ocfg = cfg["obstacles"]
    n, num_slots = start.shape[0], radii.shape[0]
    device = start.device

    seg = goal[:, :2] - start[:, :2]
    normal = torch.stack([-seg[:, 1], seg[:, 0]], dim=-1) / seg.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    t = ocfg["along_range"][0] + (ocfg["along_range"][1] - ocfg["along_range"][0]) * torch.rand(
        n, num_slots, device=device
    )
    lateral = ocfg["lateral_spread"] * (2.0 * torch.rand(n, num_slots, device=device) - 1.0)
    xy = start[:, None, :2] + t[..., None] * seg[:, None] + lateral[..., None] * normal[:, None]

    clearance = radii[None] + ocfg["endpoint_clearance"]
    far_from_start = (xy - start[:, None, :2]).norm(dim=-1) > clearance
    far_from_goal = (xy - goal[:, None, :2]).norm(dim=-1) > clearance
    in_budget = torch.arange(num_slots, device=device)[None] < num_active[:, None]
    return xy, in_budget & far_from_start & far_from_goal


class Curriculum:
    """Global difficulty level, promoted when the success rate over a window of finished episodes is high enough."""

    def __init__(self, cfg: dict):
        self.levels = cfg["levels"]
        self.promote_rate = cfg["promote_success_rate"]
        self.window = cfg["window"]
        self.level_idx = 0
        self._successes = 0
        self._episodes = 0

    @property
    def level(self) -> dict:
        return self.levels[self.level_idx]

    @property
    def max_obstacles(self) -> int:
        return max(level["num_obstacles"][1] for level in self.levels)

    def update(self, success: torch.Tensor) -> bool:
        """Feed the success flags of the episodes that just ended; return True when the level changed."""
        self._successes += int(success.sum())
        self._episodes += success.numel()
        if self._episodes < self.window:
            return False
        promote = self._successes / self._episodes >= self.promote_rate and self.level_idx < len(self.levels) - 1
        self._successes = self._episodes = 0
        if promote:
            self.level_idx += 1
        return promote
