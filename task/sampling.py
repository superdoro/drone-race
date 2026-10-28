"""Random start / goal / obstacle layouts and the success-driven curriculum. Pure torch."""

import math

import torch

from task.corridor import left_normal
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
    """Fill the corridor with up to `num_active` (n,) cylinders, one per longitudinal row.

    The free length between start and goal is split into equal rows of at least `min_spacing` (the count is capped
    to fit), so pillars never pile up into an impassable wall. Laterally a pillar can sit anywhere across the
    corridor, up to touching a side wall, so hugging a wall is not a safe way around them.
    Returns xy (n, K, 2) and active (n, K).
    """
    ocfg, half_width = cfg["obstacles"], cfg["corridor"]["half_width"]
    n, num_slots = start.shape[0], radii.shape[0]
    device = start.device

    seg = goal[:, :2] - start[:, :2]
    length = seg.norm(dim=-1)
    direction = seg / length.clamp_min(1e-6)[:, None]
    margin = ocfg["endpoint_clearance"] + (radii.max() if num_slots > 0 else 0.0)
    span = (length - 2.0 * margin).clamp_min(0.0)
    num_active = torch.minimum(num_active, torch.floor(span / ocfg["min_spacing"]).long())
    active = torch.arange(num_slots, device=device)[None] < num_active[:, None]

    # Random row order per env, so slot radii do not always appear at the same place along the corridor.
    keys = torch.where(active, torch.rand(n, num_slots, device=device), 2.0)
    row = keys.argsort(dim=1).argsort(dim=1).float()
    spacing = span / num_active.clamp_min(1)
    x = margin + (row + 0.2 + 0.6 * torch.rand(n, num_slots, device=device)) * spacing[:, None]
    y = (half_width - radii)[None] * (2.0 * torch.rand(n, num_slots, device=device) - 1.0)
    xy = start[:, None, :2] + x[..., None] * direction[:, None] + y[..., None] * left_normal(direction)[:, None]
    return xy, active


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

    @property
    def max_goal_dist(self) -> float:
        return max(level["goal_dist"][1] for level in self.levels)

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
